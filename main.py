from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Optional, List, Dict, Any
import base64, io, os, cv2, tempfile
import numpy as np
from tensorflow.keras.models import load_model

# === Initialization ===
app = FastAPI(title="SmartAlign Detection API")

# In-memory data stores
users: Dict[str, Dict[str, Any]] = {}
sessions: Dict[str, List[Dict[str, Any]]] = {}
reports: Dict[str, Dict[str, Any]] = {}

# Load actual models (adjust paths as needed)
models = {
    "exercise": load_model("yoga_pose_model_best.h5"),       # Exercise posture model
    "workplace": load_model("posture_model_best.h5"),       # Sitting/workplace posture
    "driving": load_model("saved_models/weights_best_vanilla.keras"),  # Driver safety
    "handkey": load_model("models/best.h5")                # Hand keypoint model
}

# Default parameters
DEFAULT_THRESHOLD_SECONDS = 120
DEFAULT_SENSITIVITY = 1.0

# === Schemas ===
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    contact: Optional[str]

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    userId: str
    token: str

class DetectFrameResponse(BaseModel):
    sessionId: str
    type: str
    score: float
    status: str
    alert: bool
    keypoints: Optional[List[List[float]]] = None

class SettingsRequest(BaseModel):
    thresholdSeconds: int
    alertTypes: List[str]
    sensitivity: Optional[float]

# === Auth Dependency ===
async def get_current_user(token: str = Query(...)):
    user = users.get(token)
    if not user:
        raise HTTPException(401, "Invalid token")
    return user

# === Auth Endpoints ===
@app.post("/auth/signup", response_model=AuthResponse)
def signup(req: SignupRequest):
    user_id = str(uuid4())
    users[user_id] = {
        "id": user_id,
        "name": req.name,
        "email": req.email,
        "password": req.password,
        "contact": req.contact,
        "settings": {"thresholdSeconds": DEFAULT_THRESHOLD_SECONDS,
                      "alertTypes": ["popup"],
                      "sensitivity": DEFAULT_SENSITIVITY}
    }
    return {"userId": user_id, "token": user_id}

@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    for u in users.values():
        if u["email"] == req.email and u["password"] == req.password:
            return {"userId": u["id"], "token": u["id"]}
    raise HTTPException(401, "Invalid credentials")

# === Helper Functions ===
def _process_frame(img_bytes: bytes, detection_type: str):
    # Convert bytes to OpenCV image
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if detection_type == "handkey":
        # Preprocess for keypoints model
        img_resized = cv2.resize(img, (224,224)) / 255.0
        inp = np.expand_dims(img_resized, axis=0)
        preds = models["handkey"].predict(inp)[0]
        coords = preds.reshape(-1,2).tolist()
        return None, coords
    else:
        # Single score for posture/safety models
        img_resized = cv2.resize(img, (64,64)) if detection_type=="driving" else cv2.resize(img, (224,224))
        inp = np.expand_dims(img_resized/255.0, axis=0)
        score = float(models[detection_type].predict(inp).flatten()[0])
        return score, None

# === Detection Endpoint ===
@app.post("/detect-frame", response_model=DetectFrameResponse)
def detect_frame(
    user=Depends(get_current_user),
    detection_type: str = Query(..., regex="^(exercise|workplace|driving|handkey)$"),
    file: UploadFile = File(...)
):
    # Read and process frame
    img_bytes = file.file.read()
    score, keypoints = _process_frame(img_bytes, detection_type)
    status = "good" if (keypoints or score>=0.5) else "bad"
    alert = (status == "bad")

    # Log session
    sess_list = sessions.setdefault(user["id"], [])
    # reuse last sessionId if same type
    if sess_list and sess_list[-1]["type"] == detection_type:
        sess = sess_list[-1]
    else:
        sess = {"sessionId": str(uuid4()), "type": detection_type, "frames": []}
        sess_list.append(sess)
    sess["frames"].append({"score": score, "status": status, "alert": alert})

    return {
        "sessionId": sess["sessionId"],
        "type": detection_type,
        "score": score or 0.0,
        "status": status,
        "alert": alert,
        "keypoints": keypoints
    }

# === Video Endpoint ===
@app.post("/detect-video")
def detect_video(
    user=Depends(get_current_user),
    detection_type: str = Query(..., regex="^(exercise|workplace|driving|handkey)$"),
    video: UploadFile = File(...),
    sample_rate: int = Query(5, description="Process every Nth frame")
):
    # Save temp video
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1])
    tmp.write(video.file.read()); tmp.flush(); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    results = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if idx % sample_rate == 0:
            # encode frame back to bytes
            _, buf = cv2.imencode('.jpg', frame)
            score, keypoints = _process_frame(buf.tobytes(), detection_type)
            status = "good" if (keypoints or score>=0.5) else "bad"
            results.append({"frame": idx, "score": score, "status": status})
        idx += 1
    cap.release(); os.unlink(tmp.name)
    return {"summary": results}

# === Session & Report ===
@app.post("/sessions/{session_id}/end")
def end_session(session_id: str, user=Depends(get_current_user)):
    # find session
    usr_sess = sessions.get(user["id"], [])
    sess = next((s for s in usr_sess if s["sessionId"]==session_id), None)
    if not sess:
        raise HTTPException(404, "Session not found")
    frames = sess["frames"]
    bad_count = sum(1 for f in frames if f["status"]=="bad")
    total = len(frames)
    pct = (bad_count/total*100) if total else 0
    feedback = f"You had poor {sess['type']} posture {pct:.1f}% of the time. Here are tips to improve..."
    rep_id = str(uuid4())
    reports[rep_id] = {"reportId": rep_id, "userId": user["id"], "sessionId": session_id,
                       "type": sess["type"], "pctPoor": pct, "feedback": feedback}
    return {"feedback": feedback, "reportId": rep_id}

@app.get("/users/{user_id}/reports")
def list_reports(user_id: str, user=Depends(get_current_user)):
    return [r for r in reports.values() if r["userId"]==user_id]

@app.get("/users/{user_id}/reports/{report_id}/download")
def download_report(user_id: str, report_id: str, user=Depends(get_current_user)):
    rep = reports.get(report_id)
    if not rep or rep["userId"]!=user_id:
        raise HTTPException(404, "Report not found")
    content = f"Report {report_id}\nType: {rep['type']}\nPoor: {rep['pctPoor']:.1f}%\nFeedback: {rep['feedback']}"
    return JSONResponse({"report": content})

# === Analytics & Settings ===
@app.get("/users/{user_id}/analytics")
def analytics(user_id: str, period: Optional[str]=Query("7d"), user=Depends(get_current_user)):
    sess_list = sessions.get(user_id, [])
    total = len(sess_list)
    bad = sum(f["status"]=="bad" for s in sess_list for f in s["frames"])
    good = sum(f["status"]=="good" for s in sess_list for f in s["frames"])
    return {"sessions": total, "badFrames": bad, "goodFrames": good}

@app.put("/users/{user_id}/settings")
def update_settings(user_id: str, settings: SettingsRequest, user=Depends(get_current_user)):
    if user_id!=user["id"]:
        raise HTTPException(403, "Forbidden")
    user["settings"].update(settings.dict())
    return user["settings"]

# === Run ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
from uuid import uuid4

# Ensure MediaPipe is installed
try:
    import mediapipe as mp
except ModuleNotFoundError:
    st.error("Please install mediapipe: `pip install mediapipe`")
    st.stop()

# Initialize MediaPipe detectors
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize state
def init_state():
    st.session_state.setdefault('sessions', [])  # list of {sessionId,type,frames}
    st.session_state.setdefault('reports', {})   # reportId->report
    st.session_state.setdefault('settings', {'thresholdSeconds':60, 'angleThresholds':{'exercise':20,'sitting':10,'driving':15}})
    st.session_state.setdefault('bad_start', None)
init_state()

# Sidebar
page = st.sidebar.selectbox("Page", ["Detection","End Session","Reports","Analytics","Settings"])
dtype_map = {'Exercise':'exercise','Sitting':'sitting','Driving':'driving','HandKey':'hand'}

# Compute torso angle relative to vertical
def compute_torso_angle(landmarks, h, w):
    # landmarks indices: 11 left shoulder,12 right shoulder,23 left hip,24 right hip
    pts = landmarks.landmark
    # average shoulders and hips
    sx = (pts[11].x+pts[12].x)/2 * w
    sy = (pts[11].y+pts[12].y)/2 * h
    hx = (pts[23].x+pts[24].x)/2 * w
    hy = (pts[23].y+pts[24].y)/2 * h
    dx,dy = sx-hx, hy-sy
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    return angle

# Process a single frame
def process_frame(frame, dtype):
    h, w = frame.shape[:2]
    status = 'Unknown'
    # Hand keypoints
    if dtype=='hand':
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(img_rgb)
        if res.multi_hand_landmarks:
            status='Good Hand'
            for handLms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        else:
            status='No Hand'
    else:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(img_rgb)
        if res.pose_landmarks:
            angle = compute_torso_angle(res.pose_landmarks, h, w)
            thresh = st.session_state.settings['angleThresholds'][dtype]
            if angle <= thresh:
                status='Good Posture'
            else:
                status='Bad Posture'
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        else:
            status='No Person'
    return frame, status

# Annotate, alert, and log
def annotate_and_show(frame, dtype, display, status_area):
    now = time.time()
    disp,status = process_frame(frame, dtype)
    color = (0,255,0) if 'Good' in status else (0,0,255)
    cv2.putText(disp, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    display.image(disp[:,:,::-1], channels='RGB')
    # continuous bad alert
    threshold = st.session_state.settings['thresholdSeconds']
    if 'Bad' in status or 'No' in status and dtype!='hand':
        if st.session_state.bad_start is None:
            st.session_state.bad_start = now
        elif now - st.session_state.bad_start >= threshold:
            st.session_state.bad_start = now
            status_area.warning(f"⚠️ {status} persisted for {threshold}s")
    else:
        st.session_state.bad_start = None
    # log
    sessions = st.session_state.sessions
    if sessions and sessions[-1]['type']==dtype:
        sess = sessions[-1]
    else:
        sess={'sessionId':str(uuid4()),'type':dtype,'frames':[]}
        sessions.append(sess)
    sess['frames'].append({'timestamp':now,'status':status})

# Detection
if page=='Detection':
    st.title('Detection')
    dtype_name = st.selectbox('Mode', list(dtype_map.keys()))
    dtype = dtype_map[dtype_name]
    mode = st.radio('Input', ['Upload Image','Upload Video','Live Camera'], key='inmod')
    display = st.empty(); status_area = st.empty()
    if mode=='Upload Image':
        f=st.file_uploader('Image', type=['png','jpg'], key='imgf')
        if f:
            arr=np.frombuffer(f.read(),np.uint8)
            img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            annotate_and_show(img, dtype, display, status_area)
    elif mode=='Upload Video':
        f=st.file_uploader('Video', type=['mp4','avi'], key='vidf')
        if f:
            tmp=tempfile.NamedTemporaryFile(delete=False)
            tmp.write(f.read()); tmp.close()
            cap=cv2.VideoCapture(tmp.name); fd=st.empty()
            while cap.isOpened():
                ret,frm=cap.read();
                if not ret: break
                annotate_and_show(frm, dtype, fd, status_area)
            cap.release(); os.unlink(tmp.name)
    else:
        img=st.camera_input('Camera', key='cam')
        if img:
            arr=np.frombuffer(img.read(),np.uint8)
            frm=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            annotate_and_show(frm, dtype, display, status_area)

# End Session
elif page=='End Session':
    st.title('End Session & Feedback')
    sessions=st.session_state.sessions
    if sessions:
        sel=st.selectbox('Session',[f"{s['sessionId']} ({s['type']})" for s in sessions], key='ss')
        if st.button('End Session', key='end'):
            sid=sel.split()[0]; s=next(x for x in sessions if x['sessionId']==sid)
            bad=sum(1 for f in s['frames'] if 'Bad' in f['status'] or 'No' in f['status'])
            tot=len(s['frames']); pct=bad/tot*100 if tot else 0
            fb=f"Issues detected in {pct:.1f}% frames. Please improve posture."
            rid=str(uuid4())
            st.session_state.reports[rid]={'reportId':rid,'sessionId':sid,'type':s['type'],'pct':pct,'feedback':fb}
            st.success(fb)
    else: st.info('No sessions')

# Reports
elif page=='Reports':
    st.title('Reports')
    for r in st.session_state.reports.values():
        st.subheader(r['reportId']); st.write(f"{r['sessionId']} {r['type']} {r['pct']:.1f}%"); st.write(r['feedback']); st.download_button('Download',r['feedback'],file_name=f"r_{r['reportId']}.txt")

# Analytics
elif page=='Analytics':
    st.title('Analytics'); frames=[f for s in st.session_state.sessions for f in s['frames']]; good=sum(1 for f in frames if 'Good' in f['status']); bad=len(frames)-good; st.bar_chart({'Good':good,'Bad':bad})

# Settings
elif page=='Settings':
    st.title('Settings'); thr=st.slider('Missing-keypoint threshold (sec)',10,300,st.session_state.settings['thresholdSeconds'], key='th'); st.session_state.settings['thresholdSeconds']=thr; st.write(f"Threshold: {thr}s")

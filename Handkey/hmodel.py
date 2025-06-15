import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image, ImageFile
import concurrent.futures
import time
import shutil
import glob
import json
import cv2
import seaborn as sns

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
BACKUP_DIR = os.path.join(BASE_DIR, 'backup_imgs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

for d in (BACKUP_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_TRAIN = 500   # reduced number of training images
MAX_VAL = 50      # reduced number of validation images
EPOCHS = 30       # fewer epochs for speed

# Utility: image integrity check
def check_image(path):
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        img.load()
        img.resize(IMG_SIZE)
        return None
    except Exception as e:
        return (path, str(e))

# Find corrupted images
def find_corrupted_images(directory):
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                images.append(os.path.join(root, f))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check_image, images))
    return [r for r in results if r]

# Move corrupted images to backup
def move_corrupted(corrupted):
    for path, _ in corrupted:
        rel = os.path.relpath(os.path.dirname(path), TRAIN_DIR)
        dest = os.path.join(BACKUP_DIR, rel)
        os.makedirs(dest, exist_ok=True)
        shutil.move(path, os.path.join(dest, os.path.basename(path)))

# Load keypoints from label files
def load_keypoints(directory):
    kp = {}
    lbl_dir = os.path.join(directory, 'labels')
    img_dir = os.path.join(directory, 'images')
    files = glob.glob(os.path.join(lbl_dir, '*.json')) + glob.glob(os.path.join(lbl_dir, '*.txt'))
    for lf in files:
        base = os.path.splitext(os.path.basename(lf))[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            p = os.path.join(img_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            continue
        points = []
        try:
            if lf.endswith('.json'):
                data = json.load(open(lf))
                for key in ('keypoints', 'hand_landmarks', 'landmarks'):
                    if key in data:
                        points = data[key]
                        break
                if not points:
                    for v in data.values():
                        if isinstance(v, list) and all(isinstance(pt, (list, tuple)) and len(pt)>=2 for pt in v):
                            points = v
                            break
            else:
                for line in open(lf):
                    vals = line.strip().split()
                    if len(vals) >= 2:
                        x, y = map(float, vals[:2])
                        points.append([x, y])
        except Exception:
            continue
        if not points:
            continue
        arr = np.array(points, dtype=np.float32)
        if arr.max() > 1.0:
            w, h = Image.open(img_path).size
            arr[:, 0] /= w
            arr[:, 1] /= h
        kp[img_path] = arr.flatten()
    return kp

# Data generator subclassing Keras Sequence
class KeypointDataGenerator(Sequence):
    def __init__(self, directory, batch_size, img_size, shuffle=True, max_images=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.kp_dict = load_keypoints(directory)
        self.paths = list(self.kp_dict.keys())
        if max_images and len(self.paths) > max_images:
            self.paths = list(np.random.choice(self.paths, max_images, replace=False))
        self.indexes = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs, kps = [], []
        for i in inds:
            p = self.paths[i]
            im = cv2.imread(p)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.img_size) / 255.0
            imgs.append(im)
            kps.append(self.kp_dict[p])
        return np.array(imgs, dtype=np.float32), np.array(kps, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Build CNN model for keypoint regression
def build_model(input_shape, num_kp):
    inp = Input(shape=input_shape)
    x = inp
    for filters in (64, 128, 256, 512):
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(num_kp * 2, activation='linear')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model

# Visualization of keypoints on images
def visualize_keypoints(img, kps, save_path=None):
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    h, w = img.shape[:2]
    for i in range(0, len(kps), 2):
        cv2.circle(img, (int(kps[i] * w), int(kps[i+1] * h)), 3, (0, 255, 0), -1)
        cv2.putText(img, str(i//2), (int(kps[i] * w) + 5, int(kps[i+1] * h) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    plt.imshow(img)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()

# ===== Main Execution =====
if __name__ == '__main__':
    print("Step 1: Checking for corrupted images...")
    corrupted = find_corrupted_images(TRAIN_DIR) + find_corrupted_images(VAL_DIR)
    if corrupted:
        print(f"Moving {len(corrupted)} corrupted images to backup...")
        move_corrupted(corrupted)
    else:
        print("No corrupted images found.")

    print("Step 2: Setting up data generators...")
    train_gen = KeypointDataGenerator(TRAIN_DIR, BATCH_SIZE, IMG_SIZE, True, MAX_TRAIN)
    val_gen   = KeypointDataGenerator(VAL_DIR, BATCH_SIZE, IMG_SIZE, False, MAX_VAL)
    _, sample_kps = train_gen[0]
    num_kp = sample_kps.shape[1] // 2
    print(f"Using {len(train_gen.paths)} train samples and {len(val_gen.paths)} val samples")


    print("Step 3: Building model...")
    model = build_model(IMG_SIZE + (3,), num_kp)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'best.h5'), monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    print("Step 4: Training...")
    start = time.time()
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1)
    print(f"Training finished in {(time.time()-start)/60:.2f} minutes")

    final_model_path = os.path.join(MODELS_DIR, 'final.h5')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    print("Step 5: Plotting training history...")
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='train_mae')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.legend(); plt.title('MAE')
    plt.savefig(os.path.join(RESULTS_DIR, 'history.png'))

    print("Step 6: Evaluation and visualization...")
    val_loss, val_mae = model.evaluate(val_gen)
    print(f"Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

    imgs, true_kps = val_gen[0]
    pred_kps = model.predict(imgs)
    for i in range(min(5, len(imgs))):
        visualize_keypoints(imgs[i], true_kps[i], save_path=os.path.join(RESULTS_DIR, f'{i}_gt.png'))
        visualize_keypoints(imgs[i], pred_kps[i], save_path=os.path.join(RESULTS_DIR, f'{i}_pred.png'))

    all_kps = np.vstack([val_gen[i][1] for i in range(len(val_gen))])
    corr = np.corrcoef(all_kps.T)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, center=0)
    plt.savefig(os.path.join(RESULTS_DIR, 'corr.png'))

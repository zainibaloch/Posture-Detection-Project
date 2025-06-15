import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from PIL import Image, ImageFile
import concurrent.futures
import time
import shutil
import pandas as pd
import json
import re
import glob

# Force PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set paths based on your actual directory structure
base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is located
train_dir = os.path.join(base_dir, 'imgs', 'train')
valid_dir = os.path.join(base_dir, 'imgs', 'valid')
test_dir = os.path.join(base_dir, 'imgs', 'test')
backup_dir = os.path.join(base_dir, 'imgs', 'backup_imgs')

# Create models and results directory if they don't exist
models_dir = os.path.join(base_dir, 'models')
results_dir = os.path.join(base_dir, 'results')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Image parameters
img_width, img_height = 224, 224
batch_size = 32

# Create backup folder if it doesn't exist
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

# Step 1: Find and remove corrupted images
def check_image(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify image integrity
        img.close()
        
        # Also try to load and resize the image
        img = Image.open(image_path)
        img.load()
        img = img.resize((img_width, img_height))
        return None  # No error means image is good
    except Exception as e:
        return image_path, str(e)  # Return path and error for corrupted images

def find_corrupted_images(directory):
    print(f"Checking images in {directory}...")
    
    # Get all image files
    all_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} images. Checking for corruption...")
    
    # Check images in parallel for faster processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check_image, all_images))
    
    # Filter out None results (good images)
    corrupted_images = [r for r in results if r is not None]
    return corrupted_images

def move_corrupted_images(corrupted_images_list):
    if not corrupted_images_list:
        print("No corrupted images to move.")
        return
    
    print(f"Found {len(corrupted_images_list)} corrupted images.")
    
    # Create backup structure
    for img_path, _ in corrupted_images_list:
        # Get relative path structure
        rel_path = os.path.dirname(img_path)
        backup_path = os.path.join(backup_dir, os.path.relpath(rel_path, start=os.path.dirname(train_dir)))
        
        # Create directory if it doesn't exist
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        
        # Move the file
        backup_file_path = os.path.join(backup_path, os.path.basename(img_path))
        try:
            shutil.move(img_path, backup_file_path)
            print(f"Moved corrupted image: {img_path} -> {backup_file_path}")
        except Exception as e:
            print(f"Failed to move {img_path}: {str(e)}")
    
    print("All corrupted images have been moved to the backup directory.")

# Function to create class directories and organize images by class
def organize_images_by_class(directory, class_pattern=r'(bad|good)_posture'):
    """
    Organizes images into class directories based on file name patterns.
    Returns a dictionary mapping original file paths to new file paths.
    """
    print(f"Organizing images in {directory} by class...")
    
    # Get all image files
    image_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) and not file.startswith('._'):
            image_files.append(file)
    
    # Create class directories if they don't exist
    class_dirs = {}
    file_mapping = {}
    
    for file in image_files:
        # Extract class name from file
        match = re.search(class_pattern, file)
        if match:
            class_name = match.group(1) + "_posture"
            if class_name not in class_dirs:
                class_dir = os.path.join(directory, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                class_dirs[class_name] = class_dir
            
            # Create mapping of original to new path
            src_path = os.path.join(directory, file)
            dst_path = os.path.join(class_dirs[class_name], file)
            file_mapping[src_path] = dst_path
    
    return file_mapping, list(class_dirs.keys())

# Step 1: Check for and handle corrupted images
print("Step 1: Finding and removing corrupted images...")
corrupted_train = find_corrupted_images(train_dir)
corrupted_valid = find_corrupted_images(valid_dir)
corrupted_test = find_corrupted_images(test_dir)

# Print information about corrupted images
if corrupted_train:
    print("\nCorrupted images in training set:")
    for img_path, error in corrupted_train:
        print(f"- {img_path}: {error}")

if corrupted_valid:
    print("\nCorrupted images in validation set:")
    for img_path, error in corrupted_valid:
        print(f"- {img_path}: {error}")

if corrupted_test:
    print("\nCorrupted images in test set:")
    for img_path, error in corrupted_test:
        print(f"- {img_path}: {error}")

# If any corrupted images were found, move them to backup folder
corrupted_all = corrupted_train + corrupted_valid + corrupted_test
if corrupted_all:
    print("\nMoving corrupted images to backup folder...")
    move_corrupted_images(corrupted_all)
else:
    print("\nNo corrupted images found. Proceeding with training.")

# Step 2: Organize images by class (if needed)
print("\nStep 2: Organizing images by class...")

# Check if organization is needed (look for existing class directories)
train_subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
valid_subdirs = [d for d in os.listdir(valid_dir) if os.path.isdir(os.path.join(valid_dir, d))]
test_subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

# Create temporary directories for organization if needed
temp_train_dir = os.path.join(base_dir, 'temp_train')
temp_valid_dir = os.path.join(base_dir, 'temp_valid')
temp_test_dir = os.path.join(base_dir, 'temp_test')

# If there are no class subdirectories, organize the images
organize_needed = len([d for d in train_subdirs if d in ['good_posture', 'bad_posture']]) < 2

if organize_needed:
    print("Creating class subdirectories for the model...")
    
    # Create temp directories
    for temp_dir in [temp_train_dir, temp_valid_dir, temp_test_dir]:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    
    # Create class directories and copy images
    for src_dir, dst_dir in [(train_dir, temp_train_dir), 
                             (valid_dir, temp_valid_dir), 
                             (test_dir, temp_test_dir)]:
        # Create 'good_posture' and 'bad_posture' directories
        good_dir = os.path.join(dst_dir, 'good_posture')
        bad_dir = os.path.join(dst_dir, 'bad_posture')
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)
        
        # Copy images to appropriate directories
        for img in os.listdir(src_dir):
            if img.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) and not img.startswith('._'):
                src_path = os.path.join(src_dir, img)
                if 'bad_posture' in img:
                    dst_path = os.path.join(bad_dir, img)
                elif 'good_posture' in img:
                    dst_path = os.path.join(good_dir, img)
                else:
                    continue  # Skip files that don't match patterns
                    
                # Copy the file
                shutil.copy2(src_path, dst_path)
                print(f"Copied {img} to {os.path.dirname(dst_path)}")
    
    # Update directory paths to use temp directories
    train_dir = temp_train_dir
    valid_dir = temp_valid_dir
    test_dir = temp_test_dir

# Step 3: Training the model
print("\nStep 3: Training the model...")
time.sleep(2)  # Pause briefly to let the user see the information

# List class directories
class_dirs = sorted([d for d in os.listdir(train_dir) 
                    if os.path.isdir(os.path.join(train_dir, d))])
print(f"Found classes: {class_dirs}")

# Count images in each class
print("\nTraining set:")
total_train = 0
for class_dir in class_dirs:
    class_path = os.path.join(train_dir, class_dir)
    count = len([f for f in os.listdir(class_path) 
                if os.path.isfile(os.path.join(class_path, f)) and 
                f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
    total_train += count
    print(f"  {class_dir}: {count} images")
print(f"Total training images: {total_train}")

print("\nValidation set:")
total_valid = 0
for class_dir in class_dirs:
    class_path = os.path.join(valid_dir, class_dir)
    if os.path.exists(class_path):
        count = len([f for f in os.listdir(class_path) 
                    if os.path.isfile(os.path.join(class_path, f)) and 
                    f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
        total_valid += count
        print(f"  {class_dir}: {count} images")
print(f"Total validation images: {total_valid}")

print("\nTest set:")
total_test = 0
for class_dir in class_dirs:
    class_path = os.path.join(test_dir, class_dir)
    if os.path.exists(class_path):
        count = len([f for f in os.listdir(class_path) 
                    if os.path.isfile(os.path.join(class_path, f)) and 
                    f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
        total_test += count
        print(f"  {class_dir}: {count} images")
print(f"Total test images: {total_test}")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test data
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Display class mapping
print("\nClass indices mapping:")
for class_name, index in train_generator.class_indices.items():
    print(f"  {class_name}: {index}")

# Build CNN model
def build_model(num_classes):
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_width, img_height, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC()]
    )
    
    return model

# Create and compile the model
num_classes = len(class_dirs)
model = build_model(num_classes)

# Model summary
model.summary()

# Set up callbacks for early stopping, model checkpointing and learning rate reduction
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(models_dir, 'posture_model_best.h5'), 
                   monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
]

# Train the model
epochs = 50
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size + 1,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size + 1,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
final_model_path = os.path.join(models_dir, 'posture_model_final.h5')
model.save(final_model_path)
print(f"Model saved as '{final_model_path}'")
print(f"Best model saved as '{os.path.join(models_dir, 'posture_model_best.h5')}'")

# Plot training history
def plot_training_history(history):
    # Create figure with multiple subplots
    metrics = ['accuracy', 'loss', 'precision', 'recall', 'auc']
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if i < len(axes):
            if metric in history.history:
                axes[i].plot(history.history[metric], label=f'Training {metric.capitalize()}')
                axes[i].plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
                axes[i].set_title(f'Model {metric.capitalize()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'), dpi=300)
    plt.show()

plot_training_history(history)

# Load the best model for evaluation
best_model = load_model(os.path.join(models_dir, 'posture_model_best.h5'))

# Evaluate the model on test data
print("\nEvaluating model on test data...")
test_results = best_model.evaluate(test_generator)
metrics_names = best_model.metrics_names
print("\nTest Metrics:")
for name, value in zip(metrics_names, test_results):
    print(f"  {name}: {value:.4f}")

# Get predictions
test_generator.reset()
y_pred_prob = best_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred_prob, axis=1)

# Get true labels
y_true = test_generator.classes

# Calculate per-class metrics
print("\nClassification Report:")
class_labels = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print(report)

# Save the report to a file
with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
plt.show()

# Calculate ROC curve and AUC for each class
def plot_roc_curves(y_true, y_pred_prob, class_labels):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Convert y_true to one-hot encoding
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))
    
    for i, color, cls in zip(range(len(class_labels)), colors[:len(class_labels)], class_labels):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{cls} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=300)
    plt.show()

# Calculate precision-recall curve for each class
def plot_precision_recall_curves(y_true, y_pred_prob, class_labels):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Convert y_true to one-hot encoding
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))
    
    for i, color, cls in zip(range(len(class_labels)), colors[:len(class_labels)], class_labels):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        avg_precision = np.mean(precision)
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{cls} (AP = {avg_precision:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'precision_recall_curves.png'), dpi=300)
    plt.show()

# Plot ROC and Precision-Recall curves
plot_roc_curves(y_true, y_pred_prob, class_labels)
plot_precision_recall_curves(y_true, y_pred_prob, class_labels)

# Save all metrics to a CSV file
metrics_data = {
    'Metric': metrics_names,
    'Value': test_results
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(os.path.join(results_dir, 'model_metrics.csv'), index=False)

# Example function to predict a new image
def predict_posture(image_path, model, class_indices):
    from tensorflow.keras.preprocessing import image
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction[0])
    
    # Get class name
    class_names = list(class_indices.keys())
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    return predicted_class, confidence

print("\nTraining and evaluation complete!")
print(f"- Final model saved as: {final_model_path}")
print(f"- Best model saved as: {os.path.join(models_dir, 'posture_model_best.h5')}")
print(f"- Training history plots saved in: {results_dir}")
print(f"- Confusion matrix saved as: {os.path.join(results_dir, 'confusion_matrix.png')}")
print(f"- ROC curves saved as: {os.path.join(results_dir, 'roc_curves.png')}")
print(f"- Precision-Recall curves saved as: {os.path.join(results_dir, 'precision_recall_curves.png')}")
print(f"- Classification report saved as: {os.path.join(results_dir, 'classification_report.txt')}")
print(f"- All metrics saved as: {os.path.join(results_dir, 'model_metrics.csv')}")

# Clean up temporary directories if they were created
if organize_needed:
    for temp_dir in [temp_train_dir, temp_valid_dir, temp_test_dir]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    print("\nTemporary directories cleaned up.")

if corrupted_all:
    print(f"\nNote: {len(corrupted_all)} corrupted images were moved to the '{backup_dir}' directory")
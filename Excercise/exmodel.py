import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image, ImageFile
import concurrent.futures
import time
import shutil

# Force PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set paths
train_dir = 'train'
test_dir = 'test'
backup_dir = 'backup_imgs'

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
        
        # Also try to load and resize the image - this catches more potential issues
        img = Image.open(image_path)
        img.load()
        img = img.resize((224, 224))
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
        backup_path = os.path.join(backup_dir, os.path.relpath(rel_path, start='.'))
        
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

print("Step 1: Finding and removing corrupted images...")
# Find corrupted images
corrupted_train = find_corrupted_images(train_dir)
corrupted_test = find_corrupted_images(test_dir)

# Print information about corrupted images
if corrupted_train:
    print("\nCorrupted images in training set:")
    for img_path, error in corrupted_train:
        print(f"- {img_path}: {error}")

if corrupted_test:
    print("\nCorrupted images in test set:")
    for img_path, error in corrupted_test:
        print(f"- {img_path}: {error}")

# If any corrupted images were found, move them to backup folder
corrupted_all = corrupted_train + corrupted_test
if corrupted_all:
    print("\nMoving corrupted images to backup folder...")
    move_corrupted_images(corrupted_all)
else:
    print("\nNo corrupted images found. Proceeding with training.")

print("\nStep 2: Training the model...")
time.sleep(2)  # Pause briefly to let the user see the information

# Dataset exploration after cleanup
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"Yoga poses classes: {class_names}")

# Count images in each class
print("\nTraining set:")
train_counts = {}
total_train = 0
for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        train_counts[class_name] = count
        total_train += count
        print(f"  {class_name}: {count} images")
print(f"Total training images: {total_train}")

print("\nTest set:")
test_counts = {}
total_test = 0
for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        test_counts[class_name] = count
        total_test += count
        print(f"  {class_name}: {count} images")
print(f"Total test images: {total_test}")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
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
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fourth convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and compile the model
num_classes = len(class_names)
model = build_model(num_classes)

# Model summary
model.summary()

# Set up callbacks for early stopping and model checkpointing
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('yoga_pose_model_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train the model
epochs = 50
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=max(1, test_generator.samples // batch_size),
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('yoga_pose_model_final.h5')
print("Model saved as 'yoga_pose_model_final.h5'")
print("Best model saved as 'yoga_pose_model_best.h5'")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_history(history)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Get predictions
test_generator.reset()
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = test_generator.classes

# Calculate per-class accuracy
print("\nClassification Report:")
report = classification_report(y_true, y_pred_classes, 
                            target_names=list(test_generator.class_indices.keys()))
print(report)

# Save the report to a file
with open('classification_report.txt', 'w') as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_generator.class_indices.keys()),
            yticklabels=list(test_generator.class_indices.keys()))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Example function to predict a new image
def predict_pose(image_path, model, class_indices):
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
print(f"- Model saved as: yoga_pose_model_final.h5")
print(f"- Best model saved as: yoga_pose_model_best.h5")
print(f"- Training history plot saved as: training_history.png")
print(f"- Confusion matrix saved as: confusion_matrix.png")
print(f"- Classification report saved as: classification_report.txt")

if corrupted_all:
    print(f"\nNote: {len(corrupted_all)} corrupted images were moved to the '{backup_dir}' directory")
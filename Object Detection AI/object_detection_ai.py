import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained model (misalnya MobileNet SSD atau YOLO yang telah dilatih)
model = load_model('object_detection_model.h5')

# Load label classes
labels = ['person', 'car', 'bicycle', 'dog', 'cat']  # Contoh label

def detect_objects(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224)) / 255.0  # Normalisasi
    image_expanded = np.expand_dims(image_resized, axis=0)
    
    predictions = model.predict(image_expanded)[0]
    detected_class = labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    return f"Detected: {detected_class} with confidence {confidence:.2f}"

# Contoh penggunaan
if __name__ == "__main__":
    image_path = "test_image.jpg"  # Ganti dengan path gambar yang ingin diuji
    result = detect_objects(image_path)
    print(result)

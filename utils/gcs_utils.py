import tensorflow as tf
import numpy as np
import cv2

def read_image_from_gcs(image_path):
    try:
        with tf.io.gfile.GFile(image_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

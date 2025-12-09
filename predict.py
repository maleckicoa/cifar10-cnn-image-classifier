
import os
import io
import numpy as np

import tensorflow as tf
from PIL import Image

from tensorflow.keras.models import load_model

model_path = os.path.join(os.path.dirname(__file__), "mobilenetv2_cifar10_best.keras")
model = load_model(model_path, compile=False, safe_mode=False)

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def predict_image(image_bytes):

    try:

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
        img = tf.image.resize(img, (96, 96))
        img = img / 255.0
        img = tf.expand_dims(img, 0)  # shape: (1, 96, 96, 3)


        pred = model.predict(img, verbose=2)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = float(np.max(pred))
        
        return {
            "prediction": class_names[pred_class],
            "confidence": confidence,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

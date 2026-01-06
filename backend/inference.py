import numpy as np
from tensorflow.keras.models import load_model

MODEL_VERSION = "v1"
model = load_model("models/mnist_cnn.h5")

def predict_digit(img_array: np.ndarray):
    img = img_array / 255.0
    img = img.reshape(1, 28, 28, 1)
    probs = model.predict(img)[0]
    return int(np.argmax(probs)), float(np.max(probs))

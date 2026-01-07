import os
import numpy as np
import tensorflow as tf

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = os.getenv("MODEL_PATH", "/models/mnist_latest.h5")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

_model = None


# =====================================================
# MODEL LOADING
# =====================================================
def load_model():
    """
    Charge le modèle entraîné par le pipeline Prefect / MLflow.
    """
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"❌ Modèle introuvable à {MODEL_PATH}. "
                "Le pipeline d'entraînement doit être exécuté."
            )

        print(f"✅ Chargement du modèle depuis {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)

    return _model


# =====================================================
# INFERENCE
# =====================================================
def predict_digit(img_array: np.ndarray):
    """
    Prédit un chiffre MNIST à partir d'une image 28x28.

    Args:
        img_array (np.ndarray): image 28x28, valeurs 0-255

    Returns:
        tuple: (classe prédite, confiance)
    """
    model = load_model()

    # Normalisation
    img = img_array.astype("float32") / 255.0

    # Inversion si fond blanc (Streamlit)
    if np.mean(img) > 0.5:
        img = 1.0 - img

    # Reshape
    img = img.reshape(1, 28, 28, 1)

    # Prédiction
    probs = model.predict(img, verbose=0)[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return predicted_class, confidence

import os
import streamlit as st
import numpy as np
import requests
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ===============================
# CONFIGURATION
# ===============================
BACKEND_URL = os.getenv("BACKEND_URL", "http://mnist-backend:8000")

st.set_page_config(page_title="MNIST Feedback", layout="centered")
st.title("✏️ MNIST – Test & Feedback")

# ===============================
# CANVAS
# ===============================
canvas = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ===============================
# UTILS
# ===============================
def canvas_to_png(image_data) -> bytes:
    img = Image.fromarray(image_data.astype("uint8")).convert("L")
    img = img.resize((28, 28))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Prédire"):
    if canvas.image_data is None:
        st.warning("Veuillez dessiner un chiffre.")
    else:
        image_bytes = canvas_to_png(canvas.image_data)

        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                files={"file": ("digit.png", image_bytes, "image/png")},
                timeout=5
            )
            response.raise_for_status()

            data = response.json()
            st.session_state["prediction"] = data
            st.session_state["image_bytes"] = image_bytes

            st.success(
                f"Prédiction : **{data['predicted_label']}** "
                f"(confiance = {data['confidence']:.2f})"
            )

        except requests.RequestException as e:
            st.error(f"Erreur API : {e}")

# ===============================
# FEEDBACK
# ===============================
if "prediction" in st.session_state:
    st.divider()
    st.subheader("❌ Mauvaise prédiction ?")

    true_label = st.selectbox(
        "Sélectionnez le bon chiffre",
        list(range(10)),
        index=st.session_state["prediction"]["predicted_label"]
    )

    if st.button("📤 Envoyer correction"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/correct",
                data={
                    "prediction_id": st.session_state["prediction"]["prediction_id"],
                    "true_label": true_label,
                },
                files={
                    "file": ("digit.png", st.session_state["image_bytes"], "image/png")
                },
                timeout=5
            )
            response.raise_for_status()

            st.success("✅ Correction enregistrée")
            st.session_state.clear()

        except requests.RequestException as e:
            st.error(f"Erreur lors de l'envoi de la correction : {e}")

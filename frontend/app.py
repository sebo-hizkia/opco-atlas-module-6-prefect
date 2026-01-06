import streamlit as st
import numpy as np
import requests
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("✏️ MNIST – Test & Feedback")

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

if st.button("🔍 Prédire"):
    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L")
        img = img.resize((28, 28))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = requests.post(
            "http://backend:8000/predict",
            files={"file": ("digit.png", buf, "image/png")}
        )

        if response.ok:
            data = response.json()
            st.session_state["prediction"] = data
            st.success(f"Prédiction : {data['predicted_label']} ({data['confidence']:.2f})")

if "prediction" in st.session_state:
    st.subheader("❌ Mauvaise prédiction ?")
    true_label = st.selectbox("Correction", list(range(10)))
    if st.button("Envoyer correction"):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        requests.post(
            "http://backend:8000/correct",
            data={"prediction_id": st.session_state["prediction"]["prediction_id"],
                  "true_label": true_label},
            files={"file": ("digit.png", buf, "image/png")}
        )
        st.success("Correction enregistrée")

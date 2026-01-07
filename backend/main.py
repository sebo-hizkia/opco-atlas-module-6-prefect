from fastapi import FastAPI, UploadFile, File, Depends, Form
from sqlalchemy.orm import Session
import uuid
import numpy as np
from PIL import Image
import io

from database import SessionLocal, engine
from models import Base, PredictionLog, Feedback
from schemas import PredictionResponse
from inference import predict_digit, MODEL_VERSION

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MNIST Feedback API")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    img_array = np.array(img)

    pred, conf = predict_digit(img_array)

    prediction = PredictionLog(
        id=uuid.uuid4(),
        predicted_label=pred,
        confidence=conf,
        model_version=MODEL_VERSION
    )
    db.add(prediction)
    db.commit()

    return PredictionResponse(
        prediction_id=prediction.id,
        predicted_label=pred,
        confidence=conf,
        model_version=MODEL_VERSION
    )


@app.post("/correct")
async def correct(
    prediction_id: str = Form(...),  # ← Utiliser Form() au lieu de Pydantic model
    true_label: int = Form(...),     # ← Utiliser Form() pour les champs simples
    file: UploadFile = File(...),    # ← File() pour le fichier
    db: Session = Depends(get_db)
):
    image_bytes = await file.read()

    feedback = Feedback(
        prediction_id=prediction_id,
        true_label=true_label,
        predicted_label=None,
        model_version=MODEL_VERSION,
        image=image_bytes
    )
    db.add(feedback)
    db.commit()

    return {"status": "correction enregistrée"}

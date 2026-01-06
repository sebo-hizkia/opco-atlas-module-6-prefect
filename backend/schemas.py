from pydantic import BaseModel
from uuid import UUID

class PredictionResponse(BaseModel):
    prediction_id: UUID
    predicted_label: int
    confidence: float
    model_version: str


class CorrectionRequest(BaseModel):
    prediction_id: UUID
    true_label: int

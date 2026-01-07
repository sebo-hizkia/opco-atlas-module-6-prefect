from sqlalchemy import Column, Integer, Float, Text, TIMESTAMP, LargeBinary
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(TIMESTAMP, server_default=func.now())
    predicted_label = Column(Integer)
    confidence = Column(Float)
    model_version = Column(Text)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True))
    created_at = Column(TIMESTAMP, server_default=func.now())
    true_label = Column(Integer)
    predicted_label = Column(Integer)
    model_version = Column(Text)
    image = Column(LargeBinary)

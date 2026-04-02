from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
from datetime import datetime
from .database import Base

# SQLAlchemy model for logging predictions to the SQLite database
class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    decision = Column(String(10), nullable=False)          # APPROVED / DECLINED
    probability = Column(Float, nullable=False)
    fico_score = Column(Float, nullable=True)
    annual_income = Column(Float, nullable=True)
    debt_to_income_ratio = Column(Float, nullable=True)
    client_ip = Column(String(45), nullable=True)
    features_json = Column(Text, nullable=False)           # full feature dict as JSON
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

# Model to store users
class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    is_active     = Column(Boolean, default=True, nullable=False)

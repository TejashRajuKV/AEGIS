"""
AEGIS Model Record
==================
SQLAlchemy model for tracking registered models.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON

from app.models.database import Base


class ModelRecord(Base):
    """Stores metadata for registered ML models."""
    __tablename__ = "model_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)
    version = Column(String(20), default="1.0.0")
    dataset_name = Column(String(100), nullable=True)

    # Performance metrics
    accuracy = Column(Float, nullable=True)
    fairness_metrics = Column(JSON, nullable=True)

    # Model artifacts
    checkpoint_path = Column(String(500), nullable=True)
    is_active = Column(Integer, default=1)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "dataset_name": self.dataset_name,
            "accuracy": self.accuracy,
            "fairness_metrics": self.fairness_metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

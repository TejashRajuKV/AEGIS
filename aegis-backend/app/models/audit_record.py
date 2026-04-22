"""
AEGIS Audit Record Model
=========================
SQLAlchemy model for storing fairness audit results.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON, Boolean

from app.models.database import Base


class AuditRecord(Base):
    """Stores results of fairness audits."""
    __tablename__ = "audit_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)

    # Overall metrics
    accuracy = Column(Float, nullable=False)
    overall_fair = Column(Boolean, default=False)

    # Fairness metric details stored as JSON
    demographic_parity_gap = Column(Float, nullable=True)
    equalized_odds_fpr_gap = Column(Float, nullable=True)
    equalized_odds_fnr_gap = Column(Float, nullable=True)
    calibration_diff = Column(Float, nullable=True)

    # Detailed results
    metrics_json = Column(JSON, nullable=True)
    sensitive_features = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)

    # Pipeline tracking
    pipeline_id = Column(String(100), nullable=True, index=True)
    duration_secs = Column(Float, nullable=True)

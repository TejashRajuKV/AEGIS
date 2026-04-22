"""AEGIS XGBoost Model Wrapper."""
import numpy as np
from typing import List, Optional
from .base_wrapper import BaseModelWrapper
import logging

logger = logging.getLogger(__name__)


class XGBoostWrapper(BaseModelWrapper):
    """Wrapper for XGBoost classification models."""

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self._model = model
        self._feature_names = feature_names or []
        self._classes = list(getattr(model, "classes_", [0, 1]))
        logger.info("XGBoostWrapper initialized")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions."""
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions."""
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        else:
            preds = self._model.predict(X)
            probs = np.zeros((len(preds), 2))
            probs[np.arange(len(preds)), preds] = 1.0
            return probs

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if self._feature_names:
            return self._feature_names
        if hasattr(self._model, "feature_names_in_"):
            return list(self._model.feature_names_in_)
        return []

    def get_classes(self) -> List[int]:
        """Get class labels."""
        return [int(c) for c in self._classes]

    def get_model_type(self) -> str:
        return "xgboost"

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from XGBoost model."""
        if hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_.copy()
        return None

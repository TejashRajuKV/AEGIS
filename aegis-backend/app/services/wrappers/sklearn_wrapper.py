"""AEGIS Scikit-learn Model Wrapper."""
import numpy as np
from typing import List, Optional
from .base_wrapper import BaseModelWrapper
import logging

logger = logging.getLogger(__name__)


class SklearnWrapper(BaseModelWrapper):
    """Wrapper for scikit-learn classification models.

    Supports LogisticRegression, RandomForestClassifier, SVC,
    GradientBoostingClassifier, and any sklearn classifier with
    predict/predict_proba interface.
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """Initialize wrapper with a fitted scikit-learn model.

        Args:
            model: A fitted scikit-learn classifier.
            feature_names: List of feature names.
        """
        self._model = model
        self._feature_names = feature_names or []
        self._classes = list(getattr(model, "classes_", [0, 1]))
        logger.info(f"SklearnWrapper initialized for {type(model).__name__}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions using the underlying sklearn model."""
        X = np.asarray(X, dtype=np.float64)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions.

        For binary classification, returns (n_samples, 2) matrix.
        For models without predict_proba (like some SVMs), falls back
        to decision_function.
        """
        X = np.asarray(X, dtype=np.float64)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        elif hasattr(self._model, "decision_function"):
            scores = self._model.decision_function(X)
            if scores.ndim == 1:
                probs = np.zeros((len(scores), 2))
                probs[:, 1] = 1.0 / (1.0 + np.exp(-scores))
                probs[:, 0] = 1.0 - probs[:, 1]
                return probs
            return scores
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
        """Return 'sklearn'."""
        return "sklearn"

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from the model if available."""
        if hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_.copy()
        if hasattr(self._model, "coef_"):
            return np.abs(self._model.coef_[0])
        return None

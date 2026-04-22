"""AEGIS Base Model Wrapper - Abstract interface for all model wrappers."""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Any


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers.

    Provides a unified interface to wrap different ML frameworks
    (scikit-learn, XGBoost, PyTorch, TensorFlow) behind a common API.
    """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probability predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names the model expects."""
        pass

    @abstractmethod
    def get_classes(self) -> List[int]:
        """Get the list of class labels."""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Get the model framework type (sklearn, xgboost, pytorch, tensorflow)."""
        pass

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores if available.

        Returns:
            Array of feature importance scores or None if not supported.
        """
        return None

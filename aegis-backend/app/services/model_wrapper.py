"""
Model Wrapper
=============
Universal model wrapper that dispatches to the correct framework-specific
wrapper (sklearn, xgboost, pytorch, tensorflow) based on model_type.

Provides a uniform predict/predict_proba/get_info interface regardless
of the underlying ML framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger("aegis.services.model_wrapper")


class ModelWrapper:
    """
    Universal model wrapper that dispatches to framework-specific wrappers.

    Supports sklearn, xgboost, pytorch, and tensorflow models with a
    uniform predict/predict_proba/get_info interface.

    Parameters
    ----------
    model :
        The ML model to wrap.
    model_type :
        Framework type: 'sklearn', 'xgboost', 'pytorch', 'tensorflow', or 'auto'.
        If 'auto', the framework is detected from the model's class.
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "sklearn",
    ) -> None:
        self.model = model

        # Auto-detect model type if requested
        if model_type == "auto":
            model_type = self._detect_model_type(model)

        self.model_type = model_type
        self._wrapped = self._create_wrapper(model, model_type)

        logger.info(
            "ModelWrapper: type=%s, model_class=%s",
            model_type, type(model).__name__,
        )

    def predict(self, X: Any) -> np.ndarray:
        """
        Generate class predictions.

        Parameters
        ----------
        X :
            Input features (numpy array, DataFrame, list, etc.).

        Returns
        -------
        np.ndarray of shape (n_samples,) with predicted class labels.
        """
        X_arr = np.asarray(X, dtype=np.float64)

        if self.model_type == "sklearn":
            return self._sklearn_predict(X)

        elif self.model_type == "xgboost":
            return self._xgboost_predict(X)

        elif self.model_type == "pytorch":
            return self._pytorch_predict(X_arr)

        elif self.model_type == "tensorflow":
            return self._tensorflow_predict(X_arr)

        else:
            return self._generic_predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Generate class probabilities.

        Parameters
        ----------
        X :
            Input features (numpy array, DataFrame, list, etc.).

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes) with class probabilities.
        """
        X_arr = np.asarray(X, dtype=np.float64)

        if self.model_type == "sklearn":
            return self._sklearn_predict_proba(X)

        elif self.model_type == "xgboost":
            return self._xgboost_predict_proba(X)

        elif self.model_type == "pytorch":
            return self._pytorch_predict_proba(X_arr)

        elif self.model_type == "tensorflow":
            return self._tensorflow_predict_proba(X_arr)

        else:
            return self._generic_predict_proba(X)

    def get_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns
        -------
        Dict with model type, class name, and framework-specific info.
        """
        info: Dict[str, Any] = {
            "model_type": self.model_type,
            "model_class": type(self.model).__name__,
            "model_module": type(self.model).__module__,
        }

        if self.model_type == "sklearn":
            info.update(self._sklearn_info())
        elif self.model_type == "xgboost":
            info.update(self._xgboost_info())
        elif self.model_type == "pytorch":
            info.update(self._pytorch_info())
        elif self.model_type == "tensorflow":
            info.update(self._tensorflow_info())

        return info

    # ------------------------------------------------------------------
    # Sklearn methods
    # ------------------------------------------------------------------
    def _sklearn_predict(self, X: Any) -> np.ndarray:
        """Predict using sklearn model."""
        preds = self.model.predict(X)
        return np.asarray(preds)

    def _sklearn_predict_proba(self, X: Any) -> np.ndarray:
        """Get probabilities using sklearn model."""
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            return np.asarray(proba)
        # Fallback: convert predictions to one-hot
        preds = self.model.predict(X)
        classes = np.unique(preds)
        n_classes = len(classes)
        one_hot = np.zeros((len(preds), n_classes))
        for i, cls in enumerate(classes):
            one_hot[preds == cls, i] = 1.0
        return one_hot

    def _sklearn_info(self) -> Dict[str, Any]:
        """Get sklearn model info."""
        info: Dict[str, Any] = {
            "framework": "sklearn",
        }
        if hasattr(self.model, "get_params"):
            info["params"] = self.model.get_params(deep=False)
        if hasattr(self.model, "n_features_in_"):
            info["n_features_in"] = int(self.model.n_features_in_)
        if hasattr(self.model, "classes_"):
            info["classes"] = list(self.model.classes_)
            info["n_classes"] = len(self.model.classes_)
        if hasattr(self.model, "feature_importances_"):
            info["has_feature_importances"] = True
        return info

    # ------------------------------------------------------------------
    # XGBoost methods
    # ------------------------------------------------------------------
    def _xgboost_predict(self, X: Any) -> np.ndarray:
        """Predict using XGBoost model."""
        import xgboost as xgb
        if isinstance(self.model, xgb.XGBClassifier):
            preds = self.model.predict(X)
        else:
            preds = self.model.predict(X)
        return np.asarray(preds).flatten()

    def _xgboost_predict_proba(self, X: Any) -> np.ndarray:
        """Get probabilities using XGBoost model."""
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            return np.asarray(proba)
        # XGBModel (non-classifier) may only have predict
        preds = self.model.predict(X)
        n_classes = int(np.max(preds)) + 1 if len(preds) > 0 else 2
        one_hot = np.zeros((len(preds), n_classes))
        for i, p in enumerate(preds):
            one_hot[i, int(p)] = 1.0
        return one_hot

    def _xgboost_info(self) -> Dict[str, Any]:
        """Get XGBoost model info."""
        import xgboost as xgb

        info: Dict[str, Any] = {
            "framework": "xgboost",
            "xgb_class": type(self.model).__name__,
        }
        if isinstance(self.model, xgb.XGBClassifier):
            info["n_estimators"] = getattr(self.model, "n_estimators", None)
            info["max_depth"] = getattr(self.model, "max_depth", None)
            info["learning_rate"] = getattr(self.model, "learning_rate", None)
            if hasattr(self.model, "n_classes_"):
                info["n_classes"] = int(self.model.n_classes_)
        return info

    # ------------------------------------------------------------------
    # PyTorch methods
    # ------------------------------------------------------------------
    def _pytorch_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using PyTorch model."""
        try:
            import torch

            proba = self._pytorch_predict_proba(X)
            return np.argmax(proba, axis=1)
        except ImportError:
            raise ImportError("PyTorch is required for PyTorch model inference")

    def _pytorch_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probabilities using PyTorch model."""
        try:
            import torch

            if isinstance(X, torch.Tensor):
                tensor = X.float()
            else:
                tensor = torch.tensor(X, dtype=torch.float32)

            was_training = self.model.training if hasattr(self.model, "training") else False
            if hasattr(self.model, "eval"):
                self.model.eval()

            device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
            tensor = tensor.to(device)

            all_outputs = []
            batch_size = 64
            with torch.no_grad():
                for start in range(0, tensor.shape[0], batch_size):
                    batch = tensor[start:start + batch_size]
                    output = self.model(batch)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    probs = torch.softmax(output.float(), dim=-1).cpu().numpy()
                    all_outputs.append(probs)

            if was_training and hasattr(self.model, "train"):
                self.model.train()

            return np.vstack(all_outputs)
        except ImportError:
            raise ImportError("PyTorch is required for PyTorch model inference")

    def _pytorch_info(self) -> Dict[str, Any]:
        """Get PyTorch model info."""
        try:
            import torch

            info: Dict[str, Any] = {
                "framework": "pytorch",
            }
            if hasattr(self.model, "parameters"):
                total = sum(p.numel() for p in self.model.parameters())
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                info["total_parameters"] = int(total)
                info["trainable_parameters"] = int(trainable)
            if hasattr(self.model, "named_modules"):
                info["n_layers"] = sum(1 for _ in self.model.named_modules())
            return info
        except ImportError:
            return {"framework": "pytorch", "error": "PyTorch not available for inspection"}

    # ------------------------------------------------------------------
    # TensorFlow methods
    # ------------------------------------------------------------------
    def _tensorflow_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using TensorFlow model."""
        try:
            proba = self._tensorflow_predict_proba(X)
            return np.argmax(proba, axis=1)
        except ImportError:
            raise ImportError("TensorFlow is required for TensorFlow model inference")

    def _tensorflow_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probabilities using TensorFlow model."""
        try:
            import tensorflow as tf

            tensor = tf.constant(X, dtype=tf.float32)

            all_outputs = []
            batch_size = 64
            for start in range(0, tensor.shape[0], batch_size):
                batch = tensor[start:start + batch_size]
                output = self.model(batch, training=False)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                probs = tf.nn.softmax(output, axis=-1).numpy()
                all_outputs.append(probs)

            return np.vstack(all_outputs)
        except ImportError:
            raise ImportError("TensorFlow is required for TensorFlow model inference")

    def _tensorflow_info(self) -> Dict[str, Any]:
        """Get TensorFlow model info."""
        try:
            import tensorflow as tf

            info: Dict[str, Any] = {
                "framework": "tensorflow",
            }
            total = sum(
                int(w.numpy().size)
                for w in self.model.trainable_variables
            ) + sum(
                int(w.numpy().size)
                for w in self.model.non_trainable_variables
            )
            trainable = sum(
                int(w.numpy().size)
                for w in self.model.trainable_variables
            )
            info["total_parameters"] = total
            info["trainable_parameters"] = trainable
            return info
        except ImportError:
            return {"framework": "tensorflow", "error": "TensorFlow not available for inspection"}

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------
    def _generic_predict(self, X: Any) -> np.ndarray:
        """Generic prediction fallback."""
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X))
        raise NotImplementedError(
            f"Model of type {type(self.model).__name__} does not support predict()"
        )

    def _generic_predict_proba(self, X: Any) -> np.ndarray:
        """Generic probability fallback."""
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        # Fallback: one-hot from predictions
        preds = self.predict(X)
        classes = np.unique(preds)
        one_hot = np.zeros((len(preds), len(classes)))
        for i, cls in enumerate(classes):
            one_hot[preds == cls, i] = 1.0
        return one_hot

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_model_type(self, model: Any) -> str:
        """Auto-detect model framework from model class."""
        class_name = type(model).__name__
        class_module = type(model).__module__ or ""

        # Check module name
        if "torch" in class_module or "pytorch" in class_module:
            return "pytorch"
        if "tensorflow" in class_module or "keras" in class_module:
            return "tensorflow"
        if "xgboost" in class_module or "xgb" in class_module:
            return "xgboost"
        if "sklearn" in class_module or "skl" in class_module:
            return "sklearn"

        # Check base classes
        for base in type(model).__mro__:
            base_mod = base.__module__ or ""
            if "torch" in base_mod:
                return "pytorch"
            if "tensorflow" in base_mod or "keras" in base_mod:
                return "tensorflow"
            if "xgboost" in base_mod:
                return "xgboost"

        # Check class name patterns
        lower_name = class_name.lower()
        if "xgb" in lower_name:
            return "xgboost"

        return "sklearn"

    def _create_wrapper(self, model: Any, model_type: str) -> Any:
        """Create a framework-specific wrapper if available."""
        try:
            if model_type == "pytorch":
                from app.services.wrappers.pytorch_wrapper import PyTorchModelWrapper
                return PyTorchModelWrapper(model)
            elif model_type == "tensorflow":
                from app.services.wrappers.tensorflow_wrapper import TensorFlowModelWrapper
                return TensorFlowModelWrapper(model)
        except (ImportError, TypeError) as e:
            logger.debug("Could not create framework wrapper: %s", e)
        return None

    def __repr__(self) -> str:
        return (
            f"ModelWrapper(model={type(self.model).__name__}, "
            f"type={self.model_type})"
        )

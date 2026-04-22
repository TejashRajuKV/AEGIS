"""
Counterfactual Generator
========================
Generates counterfactual explanations using a trained Conditional VAE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("aegis.neural.counterfactual")


@dataclass
class CounterfactualResult:
    """Result of counterfactual generation."""

    original: np.ndarray
    counterfactual: np.ndarray
    sensitive_attribute: str
    original_value: Any
    target_value: Any
    feature_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    prediction_change: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "original": self.original.tolist(),
            "counterfactual": self.counterfactual.tolist(),
            "sensitive_attribute": self.sensitive_attribute,
            "original_value": str(self.original_value),
            "target_value": str(self.target_value),
            "feature_changes": self.feature_changes,
            "prediction_change": self.prediction_change,
        }


class CounterfactualGenerator:
    """
    Generates counterfactual explanations using a trained CVAE.

    Counterfactuals answer: "What would this person's outcome be
    if their [race/gender/age] were different?"
    """

    def __init__(
        self,
        cvae: Any,
        feature_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        """
        Initialize the counterfactual generator.

        Args:
            cvae: Trained ConditionalVAE model.
            feature_names: Names of input features for explanations.
            device: Torch device.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        self.cvae = cvae
        self.feature_names = feature_names or []
        self.device = torch.device(device)
        self.cvae.to(self.device)
        self.cvae.eval()

        logger.info(
            "CounterfactualGenerator initialized with %d features",
            len(feature_names) if feature_names else "unknown",
        )

    def generate_counterfactual(
        self,
        original_sample: np.ndarray,
        sensitive_attribute_idx: int,
        original_value: Any,
        target_value: Any,
        condition_dim: int = 1,
    ) -> CounterfactualResult:
        """
        Generate a counterfactual for a single sample.

        Args:
            original_sample: Original input (n_features,).
            sensitive_attribute_idx: Index of the sensitive attribute in features.
            original_value: Original value of the sensitive attribute.
            target_value: Target value to change to.
            condition_dim: Dimension of condition vector.

        Returns:
            CounterfactualResult with original, counterfactual, and changes.
        """
        x = torch.FloatTensor(original_sample).unsqueeze(0).to(self.device)

        # Create condition vectors (e.g., 0.0 for original, 1.0 for target)
        original_condition = torch.zeros(1, condition_dim, device=self.device)
        target_condition = torch.ones(1, condition_dim, device=self.device)

        # Generate counterfactual
        with torch.no_grad():
            counterfactual = self.cvae.generate_counterfactual(
                x, original_condition, target_condition
            )

        cf_array = counterfactual.cpu().numpy().flatten()

        # Explain changes
        changes = self._explain_changes(original_sample, cf_array)

        attr_name = (
            self.feature_names[sensitive_attribute_idx]
            if sensitive_attribute_idx < len(self.feature_names)
            else f"feature_{sensitive_attribute_idx}"
        )

        return CounterfactualResult(
            original=original_sample,
            counterfactual=cf_array,
            sensitive_attribute=attr_name,
            original_value=original_value,
            target_value=target_value,
            feature_changes=changes,
        )

    def generate_multiple(
        self,
        original_sample: np.ndarray,
        sensitive_attribute_idx: int,
        n_samples: int = 5,
        condition_dim: int = 1,
    ) -> List[CounterfactualResult]:
        """
        Generate multiple counterfactual variants.

        Args:
            original_sample: Original input.
            sensitive_attribute_idx: Index of sensitive attribute.
            n_samples: Number of counterfactuals to generate.
            condition_dim: Condition dimension.

        Returns:
            List of CounterfactualResult.
        """
        x = torch.FloatTensor(original_sample).unsqueeze(0).to(self.device)
        original_condition = torch.zeros(1, condition_dim, device=self.device)
        target_condition = torch.ones(1, condition_dim, device=self.device)

        results = []
        attr_name = (
            self.feature_names[sensitive_attribute_idx]
            if sensitive_attribute_idx < len(self.feature_names)
            else f"feature_{sensitive_attribute_idx}"
        )

        with torch.no_grad():
            cf_tensor = self.cvae.generate_counterfactual(
                x, original_condition, target_condition
            )

        base_cf = cf_tensor.cpu().numpy().flatten()

        for i in range(n_samples):
            # Add small noise for variation
            noise = np.random.normal(0, 0.01, size=base_cf.shape).astype(np.float32)
            noisy_cf = np.clip(base_cf + noise, 0, 1)
            changes = self._explain_changes(original_sample, noisy_cf)

            results.append(CounterfactualResult(
                original=original_sample,
                counterfactual=noisy_cf,
                sensitive_attribute=attr_name,
                original_value="original",
                target_value="counterfactual",
                feature_changes=changes,
            ))

        return results

    def validate_counterfactual(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray,
        model: Any,
    ) -> Dict[str, float]:
        """
        Validate that the counterfactual changes the model prediction.

        Args:
            original: Original input.
            counterfactual: Counterfactual input.
            model: Model with predict/predict_proba method.

        Returns:
            Dictionary with original prediction, counterfactual prediction, and change.
        """
        try:
            orig_shape = original.shape
            cf_shape = counterfactual.shape

            if orig_shape[0] == orig_shape[1:] or len(original.shape) == 1:
                orig_input = original.reshape(1, -1)
                cf_input = counterfactual.reshape(1, -1)
            else:
                orig_input = original
                cf_input = counterfactual

            orig_pred = model.predict(orig_input)
            cf_pred = model.predict(cf_input)

            return {
                "original_prediction": float(orig_pred[0]) if len(orig_pred) == 1 else orig_pred.tolist(),
                "counterfactual_prediction": float(cf_pred[0]) if len(cf_pred) == 1 else cf_pred.tolist(),
                "prediction_changed": bool(orig_pred[0] != cf_pred[0]) if len(orig_pred) == 1 else True,
            }
        except Exception as e:
            logger.warning("Counterfactual validation failed: %s", e)
            return {"error": str(e)}

    def _explain_changes(
        self, original: np.ndarray, counterfactual: np.ndarray,
        threshold: float = 0.01,
    ) -> Dict[str, Dict[str, float]]:
        """
        Explain feature-level changes between original and counterfactual.

        Returns:
            Dict of feature_name -> {original, counterfactual, change}.
        """
        changes = {}
        for i in range(min(len(original), len(counterfactual))):
            diff = float(counterfactual[i] - original[i])
            if abs(diff) > threshold:
                name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                changes[name] = {
                    "original": float(original[i]),
                    "counterfactual": float(counterfactual[i]),
                    "change": diff,
                }
        return changes

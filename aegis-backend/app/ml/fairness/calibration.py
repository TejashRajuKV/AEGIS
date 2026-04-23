"""AEGIS Calibration Metric.

Calibration fairness requires that predicted probabilities are equally
calibrated across all groups. Formally, for any predicted probability p:
P(Y=1|Ŷ=p, A=a) should be equal for all groups a.
"""
import numpy as np
from typing import Dict, Any
from .metrics import FairnessMetric
import logging

logger = logging.getLogger(__name__)


class CalibrationMetric(FairnessMetric):
    """Compute calibration difference across groups.

    Uses binned calibration curves to compare predicted probabilities
    against actual outcomes for each group.
    """

    def __init__(self, threshold: float = 0.1, n_bins: int = 10):
        super().__init__(name="calibration", threshold=threshold)
        self.n_bins = n_bins

    @property
    def description(self) -> str:
        """Difference in binned calibration error across groups."""
        return "Binned calibration error gap across groups"

    def _binned_calibration(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute mean calibration error using equal-width bins."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        cal_errors = []
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            cal_errors.append(abs(avg_pred - avg_true))
        return float(np.mean(cal_errors)) if cal_errors else 0.0

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attribute: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute calibration difference across groups.

        Args:
            y_true: Ground truth binary labels (0/1).
            y_pred: Predicted probabilities (continuous, not labels).
                   If binary labels, uses them as probabilities.
            sensitive_attribute: Protected attribute values.

        Returns:
            Dict with calibration results.
        """
        y_true, y_pred, sensitive_attribute = self._validate_inputs(
            y_true, y_pred, sensitive_attribute
        )
        groups = self._get_unique_groups(sensitive_attribute)

        # If y_pred is binary labels, use as probabilities
        unique_pred = np.unique(y_pred)
        if len(unique_pred) <= 2 and all(v in [0.0, 1.0] for v in unique_pred):
            y_prob = y_pred.copy()
        else:
            y_prob = y_pred

        group_calibration: Dict[str, float] = {}
        for group in groups:
            mask = sensitive_attribute == group
            if mask.sum() < 10:
                group_calibration[str(group)] = 0.0
                continue
            group_calibration[str(group)] = round(
                self._binned_calibration(y_true[mask], y_prob[mask], self.n_bins), 6
            )

        values = list(group_calibration.values())
        max_diff = float(max(values) - min(values)) if len(values) >= 2 else 0.0

        return {
            "metric_name": self.name,
            "overall_value": round(float(np.mean(values)), 6) if values else 0.0,
            "group_values": group_calibration,
            "gap": round(max_diff, 6),
            "threshold": self.threshold,
            "is_fair": max_diff <= self.threshold,
        }

"""AEGIS Equalized Odds Metric.

Equalized Odds requires that the true positive rate and false positive rate
are equal across all groups defined by the sensitive attribute.

This measures both the FPR gap (equality of opportunity in the negative class)
and the FNR gap (equality of opportunity in the positive class).
"""
import numpy as np
from typing import Dict, Any
from .metrics import FairnessMetric
import logging

logger = logging.getLogger(__name__)


class EqualizedOdds(FairnessMetric):
    """Compute Equalized Odds gap (FPR gap and FNR gap) between groups.

    FPR gap: max_a,b | FPR(A=a) - FPR(A=b) |
    FNR gap: max_a,b | FNR(A=a) - FNR(A=b) |
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__(name="equalized_odds", threshold=threshold)

    def _compute_group_rates(
        self, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        """Compute FPR and FNR for a specific group."""
        if mask.sum() == 0:
            return {"tpr": 0.0, "fpr": 0.0, "fnr": 0.0, "tnr": 0.0}

        yt = y_true[mask]
        yp = y_pred[mask]

        tp = float(((yt == 1) & (yp == 1)).sum())
        fp = float(((yt == 0) & (yp == 1)).sum())
        fn = float(((yt == 1) & (yp == 0)).sum())
        tn = float(((yt == 0) & (yp == 0)).sum())

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {"tpr": tpr, "fpr": fpr, "fnr": fnr, "tnr": tnr}

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attribute: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute equalized odds gap (FPR and FNR).

        Args:
            y_true: Ground truth binary labels (0/1).
            y_pred: Predicted binary labels (0/1).
            sensitive_attribute: Protected attribute values.

        Returns:
            Dict with equalized odds results including FPR gap and FNR gap.
        """
        y_true, y_pred, sensitive_attribute = self._validate_inputs(
            y_true, y_pred, sensitive_attribute
        )
        groups = self._get_unique_groups(sensitive_attribute)

        group_rates: Dict[str, Dict[str, float]] = {}
        for group in groups:
            mask = sensitive_attribute == group
            rates = self._compute_group_rates(y_true, y_pred, mask)
            group_rates[str(group)] = {k: round(v, 6) for k, v in rates.items()}

        # Compute gaps
        fpr_values = [v["fpr"] for v in group_rates.values()]
        fnr_values = [v["fnr"] for v in group_rates.values()]

        fpr_gap = float(max(fpr_values) - min(fpr_values)) if len(fpr_values) >= 2 else 0.0
        fnr_gap = float(max(fnr_values) - min(fnr_values)) if len(fnr_values) >= 2 else 0.0

        # Overall max gap
        overall_gap = max(fpr_gap, fnr_gap)

        return {
            "metric_name": self.name,
            "overall_value": round(overall_gap, 6),
            "group_values": group_rates,
            "fpr_gap": round(fpr_gap, 6),
            "fnr_gap": round(fnr_gap, 6),
            "gap": round(overall_gap, 6),
            "threshold": self.threshold,
            "is_fair": overall_gap <= self.threshold,
        }

"""AEGIS Demographic Parity Metric.

Demographic Parity states that the outcome should be independent of the
protected attribute. Formally: P(Ŷ=1|A=a) should be equal for all groups a.

The demographic parity gap is the maximum absolute difference in positive
outcome rates across all groups.
"""
import numpy as np
from typing import Dict, Any, List
from .metrics import FairnessMetric
import logging

logger = logging.getLogger(__name__)


class DemographicParity(FairnessMetric):
    """Compute Demographic Parity gap between groups.

    The gap is: max_a,b | P(Ŷ=1|A=a) - P(Ŷ=1|A=b) |
    A smaller gap indicates less bias.
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__(name="demographic_parity", threshold=threshold)

    @property
    def description(self) -> str:
        """Max absolute difference in selection rates across groups."""
        return "Selection rate gap across groups"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attribute: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute demographic parity gap.

        Args:
            y_true: Ground truth binary labels (0/1).
            y_pred: Predicted binary labels (0/1).
            sensitive_attribute: Protected attribute values.

        Returns:
            Dict with DP metric results.
        """
        y_true, y_pred, sensitive_attribute = self._validate_inputs(
            y_true, y_pred, sensitive_attribute
        )
        groups = self._get_unique_groups(sensitive_attribute)

        group_rates: Dict[str, float] = {}
        for group in groups:
            mask = sensitive_attribute == group
            if mask.sum() == 0:
                group_rates[str(group)] = 0.0
                continue
            rate = float(y_pred[mask].mean())
            group_rates[str(group)] = round(rate, 6)

        # Compute overall rate and max gap
        overall_rate = float(y_pred.mean())
        rates_list = list(group_rates.values())

        if len(rates_list) < 2:
            gap = 0.0
        else:
            gap = float(max(rates_list) - min(rates_list))

        return {
            "metric_name": self.name,
            "overall_value": round(overall_rate, 6),
            "group_values": group_rates,
            "gap": round(gap, 6),
            "threshold": self.threshold,
            "is_fair": gap <= self.threshold,
        }

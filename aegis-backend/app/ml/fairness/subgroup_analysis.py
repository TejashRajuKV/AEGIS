"""AEGIS Subgroup Analysis - Intersectional fairness analysis.

Merges:
- V3 SubgroupAnalysis class with intersectional subgroups, metric injection,
  and minimum-sample checks.
- V5 ``compute_subgroup_metrics()`` function and ``find_most_biased_subgroup()``.

Provides:
- :class:`SubgroupAnalysis` – class-based intersectional analysis
- :func:`compute_subgroup_metrics` – lightweight per-group metric computation
- :func:`find_most_biased_subgroup` – identify the most disparate subgroup
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from app.utils.logger import get_logger
from .metrics import FairnessMetric

logger = get_logger("subgroup_analysis")

# Minimum samples required to include a subgroup in the analysis
MIN_SUBGROUP_SAMPLES = 5


# =========================================================================
# Class-based analysis (V3)
# =========================================================================


class SubgroupAnalysis:
    """Perform intersectional subgroup fairness analysis.

    Creates subgroups defined by combinations of sensitive attributes and
    computes per-subgroup metrics.  Optionally injects external
    :class:`FairnessMetric` instances for richer analysis.
    """

    def __init__(self, metrics: Optional[List[FairnessMetric]] = None):
        """Initialise with optional fairness metrics.

        Args:
            metrics: List of :class:`FairnessMetric` instances to compute
                per subgroup.  Each metric is run against a synthetic binary
                attribute (in-subgroup vs. not-in-subgroup).
        """
        self.metrics = metrics or []

    # ------------------------------------------------------------------
    # Subgroup creation
    # ------------------------------------------------------------------

    def create_subgroups(
        self,
        sensitive_attributes: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Create intersectional subgroup boolean masks.

        Each unique combination of attribute values becomes a subgroup.
        E.g. ``{'gender': [...], 'race': [...]}`` → ``'gender_M_race_White'``.

        Args:
            sensitive_attributes: Dict mapping attribute name → value arrays.

        Returns:
            Dict mapping subgroup label → boolean mask (length = n_samples).

        Raises:
            ValueError: If attribute arrays have differing lengths.
        """
        lengths = [len(v) for v in sensitive_attributes.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All sensitive attributes must have the same length")

        n = lengths[0]
        attr_names = list(sensitive_attributes.keys())

        subgroups: Dict[str, np.ndarray] = {}
        for i in range(n):
            label = "_".join(
                f"{name}_{sensitive_attributes[name][i]}" for name in attr_names
            )
            if label not in subgroups:
                subgroups[label] = np.zeros(n, dtype=bool)
            subgroups[label][i] = True

        return subgroups

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Run intersectional subgroup analysis.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            sensitive_attributes: Dict of sensitive attribute arrays.

        Returns:
            Dict with ``num_subgroups`` and ``subgroups`` (per-subgroup
            metrics).  Subgroups below ``MIN_SUBGROUP_SAMPLES`` are skipped.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        subgroups = self.create_subgroups(sensitive_attributes)

        results: Dict[str, Any] = {
            "num_subgroups": len(subgroups),
            "subgroups": {},
        }

        for label, mask in subgroups.items():
            n_samples = int(mask.sum())

            # Minimum-sample guard (V3)
            if n_samples < MIN_SUBGROUP_SAMPLES:
                results["subgroups"][label] = {
                    "sample_size": n_samples,
                    "skipped": True,
                    "reason": "Too few samples",
                }
                continue

            yt_sub = y_true[mask]
            yp_sub = y_pred[mask]
            accuracy = float((yt_sub == yp_sub).mean())
            positive_rate = float(yp_sub.mean())

            subgroup_result: Dict[str, Any] = {
                "sample_size": n_samples,
                "accuracy": round(accuracy, 6),
                "positive_rate": round(positive_rate, 6),
            }

            # Inject external FairnessMetric instances (V3)
            for metric in self.metrics:
                sub_attr = np.zeros(len(y_true), dtype=int)
                sub_attr[mask] = 1
                try:
                    metric_result = metric.compute(y_true, y_pred, sub_attr)
                    subgroup_result[metric.name] = metric_result.get("gap", 0.0)
                except Exception as exc:
                    logger.warning(
                        "Metric %s failed for subgroup %s: %s",
                        metric.name,
                        label,
                        exc,
                    )

            results["subgroups"][label] = subgroup_result

        return results


# =========================================================================
# Functional helpers (V5)
# =========================================================================


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    feature_data: Optional[pd.DataFrame] = None,
    subgroup_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute detailed metrics for each group of a sensitive attribute.

    Optionally drills into intersectional sub-subgroups when *feature_data*
    and *subgroup_columns* are provided.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (binary or probability).
        sensitive: 1-D sensitive attribute array.
        feature_data: Optional DataFrame with additional feature columns.
        subgroup_columns: Optional list of column names for further
            intersectional breakdown.

    Returns:
        Dict mapping group name → metric dict.
    """
    results: Dict[str, Any] = {}
    unique_groups = np.unique(sensitive)

    for group in unique_groups:
        mask = sensitive == group
        group_name = str(group)
        group_size = int(mask.sum())
        if group_size == 0:
            continue

        # Base group metrics
        results[group_name] = {
            "count": group_size,
            "proportion": float(group_size / len(sensitive)),
            "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
            "selection_rate": float(y_pred[mask].mean()),
            "false_positive_rate": _false_positive_rate(
                y_true[mask], y_pred[mask]
            ),
        }

        # Intersectional sub-subgroups (V5)
        if feature_data is not None and subgroup_columns:
            for col in subgroup_columns:
                if col not in feature_data.columns:
                    continue
                subgroups_vals = feature_data.loc[mask, col].unique()
                for sg in subgroups_vals:
                    sg_mask = mask & (feature_data[col] == sg)
                    sg_size = int(sg_mask.sum())
                    if sg_size < MIN_SUBGROUP_SAMPLES:
                        continue
                    key = f"{group_name}_{col}_{sg}"
                    results[key] = {
                        "count": sg_size,
                        "accuracy": float(
                            (y_true[sg_mask] == y_pred[sg_mask]).mean()
                        ),
                        "selection_rate": float(y_pred[sg_mask].mean()),
                    }

    return results


def find_most_biased_subgroup(
    subgroup_results: Dict[str, Any],
    metric: str = "selection_rate",
) -> Dict[str, Any]:
    """Identify the subgroup with the largest disparity in a given metric.

    Args:
        subgroup_results: Dict mapping group name → metric dict (as returned
            by :func:`compute_subgroup_metrics` or :meth:`SubgroupAnalysis.analyze`).
        metric: The metric key to compare across subgroups.

    Returns:
        Dict with ``metric``, ``max_gap``, ``highest_group``, ``lowest_group``,
        and ``all_values``.  Returns empty dict if no matching data.
    """
    if not subgroup_results:
        return {}

    metric_values = {
        k: v.get(metric, 0.0)
        for k, v in subgroup_results.items()
        if metric in v
    }

    if not metric_values:
        return {}

    values = list(metric_values.values())
    max_gap = float(max(values) - min(values))

    return {
        "metric": metric,
        "max_gap": max_gap,
        "highest_group": max(metric_values, key=metric_values.get),
        "lowest_group": min(metric_values, key=metric_values.get),
        "all_values": metric_values,
    }


# =========================================================================
# Private helpers
# =========================================================================


def _false_positive_rate(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Compute false positive rate."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

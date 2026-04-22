"""AEGIS Base Fairness Metrics - Abstract definitions and registry.

Merges:
- V3 FairnessMetric ABC with input validation and batch computation
- V5 MetricRegistry pattern with is_fair() and description property

Provides:
- ``FairnessMetric`` – Abstract base class for all fairness metrics
- ``MetricRegistry`` – Registry for collecting and computing all metrics
- ``metric_registry`` – Default global registry singleton
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Optional

from app.utils.logger import get_logger

logger = get_logger("fairness_metrics")


class FairnessMetric(ABC):
    """Abstract base class for all fairness metrics.

    All fairness metrics compute a measure of bias between groups defined by
    a sensitive attribute.  Subclasses must implement ``compute()`` and the
    ``description`` property.

    Provides built-in input validation, unique-group helpers, batch
    computation across attributes, and a convenience ``is_fair()`` check.
    """

    def __init__(self, name: str, threshold: float = 0.1):
        """Initialise fairness metric.

        Args:
            name: Human-readable metric name.
            threshold: Fairness threshold – gap at or below this is fair.
        """
        self.name = name
        self.threshold = threshold

    # -- V5 abstract property -------------------------------------------

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of the metric."""
        pass

    # -- Core abstract compute ------------------------------------------

    @abstractmethod
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attribute: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute the fairness metric.

        Args:
            y_true: Ground-truth binary labels (0/1).
            y_pred: Predicted binary labels (0/1).
            sensitive_attribute: Protected attribute values.

        Returns:
            Dict with at least ``'metric_name'``, ``'overall_value'``,
            ``'group_values'``, ``'gap'``, ``'threshold'``, and ``'is_fair'``.
        """
        pass

    # -- Input validation (V3) ------------------------------------------

    def _validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attribute: np.ndarray,
    ) -> tuple:
        """Validate and align inputs.

        Ensures all arrays are 1-D ``float64`` ``np.ndarray`` of the same
        length.

        Returns:
            Tuple of validated (y_true, y_pred, sensitive_attribute).

        Raises:
            ValueError: If lengths mismatch.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        sensitive_attribute = np.asarray(sensitive_attribute).ravel()

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) "
                "must have same length"
            )
        if len(y_true) != len(sensitive_attribute):
            raise ValueError(
                f"Labels ({len(y_true)}) and sensitive_attribute "
                f"({len(sensitive_attribute)}) must have same length"
            )
        return y_true, y_pred, sensitive_attribute

    # -- Utility helpers (V3) -------------------------------------------

    def _get_unique_groups(self, sensitive_attribute: np.ndarray) -> List:
        """Return sorted unique group values from a sensitive attribute."""
        return list(np.unique(sensitive_attribute))

    def is_fair(self, gap: float, threshold: Optional[float] = None) -> bool:
        """Check whether a gap value satisfies the fairness threshold.

        Args:
            gap: The measured disparity gap.
            threshold: Override threshold (defaults to ``self.threshold``).

        Returns:
            ``True`` if ``abs(gap) <= threshold``.
        """
        return abs(gap) <= (threshold if threshold is not None else self.threshold)

    # -- Batch computation (V3) -----------------------------------------

    def compute_for_all_attributes(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metric for multiple sensitive attributes in one call.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            sensitive_attributes: Dict mapping attribute name → value array.

        Returns:
            Dict mapping attribute name → metric result dict.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for attr_name, attr_values in sensitive_attributes.items():
            try:
                result = self.compute(y_true, y_pred, attr_values)
                results[attr_name] = result
            except Exception as exc:
                logger.warning(
                    "Failed to compute %s for '%s': %s", self.name, attr_name, exc
                )
                results[attr_name] = {"error": str(exc)}
        return results


# =========================================================================
# MetricRegistry (V5)
# =========================================================================


class MetricRegistry:
    """Registry of fairness metrics.

    Collects :class:`FairnessMetric` instances and provides ``compute_all``
    to run every registered metric in a single call.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, FairnessMetric] = {}

    def register(self, metric: FairnessMetric) -> None:
        """Register a fairness metric instance."""
        self._metrics[metric.name] = metric

    def unregister(self, name: str) -> None:
        """Remove a metric by name (no-op if not found)."""
        self._metrics.pop(name, None)

    def get(self, name: str) -> FairnessMetric:
        """Retrieve a registered metric by name.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._metrics:
            raise KeyError(
                f"Unknown metric: '{name}'. "
                f"Registered: {list(self._metrics.keys())}"
            )
        return self._metrics[name]

    def list_metrics(self) -> List[str]:
        """Return names of all registered metrics."""
        return list(self._metrics.keys())

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute every registered metric.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            sensitive: Sensitive attribute array.

        Returns:
            Dict mapping metric name → metric result dict.
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = metric.compute(y_true, y_pred, sensitive)
            except Exception as exc:
                logger.warning("compute_all: %s failed: %s", name, exc)
                results[name] = {"error": str(exc)}
        return results

    def __len__(self) -> int:
        return len(self._metrics)

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __repr__(self) -> str:
        return (
            f"MetricRegistry(metrics={list(self._metrics.keys())})"
        )


# Global singleton (V5 pattern)
metric_registry = MetricRegistry()

"""
CUSUM Drift Detector
====================
Cumulative Sum control chart for detecting mean drift in data streams.
Sequential implementation for memory efficiency on 16GB RAM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger("aegis.drift.cusum")


class DriftStatus(Enum):
    """Status of drift detection."""

    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    WARNING = "warning"


@dataclass
class DriftResult:
    """Result of a single drift detection check."""

    drift_detected: bool
    status: DriftStatus
    statistic: float
    mean_estimate: float
    feature_name: str = ""
    timestamp: float = 0.0
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "drift_detected": self.drift_detected,
            "status": self.status.value,
            "statistic": self.statistic,
            "mean_estimate": self.mean_estimate,
            "feature_name": self.feature_name,
            "details": self.details,
        }


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) control chart for mean drift detection.

    Uses two-sided CUSUM:
        g_t+ = max(0, g_t + (x_t - mu_0 - k) / sigma)
        g_t- = max(0, g_t - (x_t - mu_0 + k) / sigma)

    Drift is signaled when either g+ or g- exceeds threshold h.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift_parameter: float = 0.5,
        window_size: Optional[int] = None,
        min_reference_samples: int = 30,
    ):
        """
        Initialize the CUSUM detector.

        Args:
            threshold: Decision threshold h. Higher = less sensitive.
            drift_parameter: Drift parameter k (allowable shift magnitude).
            window_size: Number of recent samples to track (None = unlimited).
            min_reference_samples: Minimum samples required for fitting.
        """
        self.threshold = threshold
        self.drift_parameter = drift_parameter
        self.window_size = window_size
        self.min_reference_samples = min_reference_samples

        self._reference_mean: float = 0.0
        self._reference_std: float = 1.0
        self._g_positive: float = 0.0
        self._g_negative: float = 0.0
        self._is_fitted: bool = False
        self._sample_count: int = 0

        logger.info(
            "CUSUMDetector: threshold=%.2f, k=%.2f",
            threshold, drift_parameter,
        )

    def fit(self, reference_data: np.ndarray) -> "CUSUMDetector":
        """
        Fit the detector on reference (in-control) data.

        Args:
            reference_data: Array of reference observations (n_samples,) or (n_samples, n_features).

        Returns:
            self
        """
        if len(reference_data) < self.min_reference_samples:
            raise ValueError(
                f"Need at least {self.min_reference_samples} reference samples, "
                f"got {len(reference_data)}"
            )

        data = np.asarray(reference_data, dtype=np.float64)
        if data.ndim > 1:
            data = data.flatten()

        self._reference_mean = float(np.mean(data))
        self._reference_std = float(np.std(data))

        if self._reference_std < 1e-10:
            self._reference_std = 1e-10

        self._is_fitted = True
        self.reset()

        logger.info(
            "CUSUM fitted: mean=%.4f, std=%.4f, n=%d",
            self._reference_mean, self._reference_std, len(data),
        )
        return self

    def update(self, value: float) -> DriftResult:
        """
        Update the CUSUM statistic with a new observation.

        Args:
            value: New data point.

        Returns:
            DriftResult indicating whether drift was detected.
        """
        if not self._is_fitted:
            raise RuntimeError("CUSUMDetector must be fitted before update()")

        self._sample_count += 1

        # Normalize the value
        normalized = (value - self._reference_mean) / self._reference_std

        # Update two-sided CUSUM
        self._g_positive = max(
            0.0, self._g_positive + normalized - self.drift_parameter
        )
        self._g_negative = max(
            0.0, self._g_negative - normalized - self.drift_parameter
        )

        max_stat = max(self._g_positive, self._g_negative)

        # Check for drift
        if max_stat >= self.threshold:
            status = DriftStatus.DRIFT_DETECTED
            drift_detected = True
            direction = "upward" if self._g_positive > self._g_negative else "downward"
            details = (
                f"CUSUM drift: {direction} shift detected. "
                f"g+={self._g_positive:.4f}, g-={self._g_negative:.4f}, "
                f"value={value:.4f}"
            )
            logger.warning(details)
        elif max_stat >= self.threshold * 0.6:
            status = DriftStatus.WARNING
            drift_detected = False
            details = f"CUSUM warning: g+={self._g_positive:.4f}, g-={self._g_negative:.4f}"
        else:
            status = DriftStatus.NO_DRIFT
            drift_detected = False
            details = ""

        return DriftResult(
            drift_detected=drift_detected,
            status=status,
            statistic=max_stat,
            mean_estimate=self._reference_mean + self._g_positive * self._reference_std,
            details=details,
        )

    def reset(self) -> None:
        """Reset the CUSUM statistics."""
        self._g_positive = 0.0
        self._g_negative = 0.0
        self._sample_count = 0

    def detect_batch(self, new_data: np.ndarray) -> List[DriftResult]:
        """
        Detect drift in a batch of new data points sequentially.

        Args:
            new_data: Array of new observations.

        Returns:
            List of DriftResult for each observation.
        """
        results = []
        data = np.asarray(new_data, dtype=np.float64)
        if data.ndim > 1:
            data = data.flatten()

        for value in data:
            results.append(self.update(float(value)))
        return results

    def get_state(self) -> dict:
        """Get current detector state."""
        return {
            "fitted": self._is_fitted,
            "reference_mean": self._reference_mean,
            "reference_std": self._reference_std,
            "g_positive": self._g_positive,
            "g_negative": self._g_negative,
            "sample_count": self._sample_count,
        }

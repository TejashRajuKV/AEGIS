"""
Wasserstein Drift Detector
===========================
Detects distribution shift using Wasserstein-1 distance (Earth Mover's Distance).
Catches variance changes and tail behavior shifts that CUSUM misses.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy import stats

from app.ml.drift.cusum_detector import DriftResult, DriftStatus

logger = logging.getLogger("aegis.drift.wasserstein")


class WassersteinDetector:
    """
    Distribution shift detector using Wasserstein-1 distance.

    Wasserstein-1 distance measures the minimum cost of transforming
    one probability distribution into another. Unlike CUSUM which only
    detects mean shifts, this detector catches:
    - Variance changes
    - Shape/tail changes
    - Multi-modal distribution shifts

    Uses scipy.stats.wasserstein_distance for computation and
    permutation testing for significance estimation.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        n_permutations: int = 500,
        significance_level: float = 0.05,
        min_reference_samples: int = 30,
    ):
        """
        Initialize the Wasserstein detector.

        Args:
            threshold: Minimum Wasserstein distance to flag as drift.
            n_permutations: Number of permutations for p-value estimation.
            significance_level: Alpha for statistical significance test.
            min_reference_samples: Minimum samples for fitting.
        """
        self.threshold = threshold
        self.n_permutations = n_permutations
        self.significance_level = significance_level
        self.min_reference_samples = min_reference_samples

        self._reference_data: Optional[np.ndarray] = None
        self._reference_mean: float = 0.0
        self._reference_std: float = 1.0
        self._is_fitted: bool = False

        logger.info(
            "WassersteinDetector: threshold=%.4f, permutations=%d",
            threshold, n_permutations,
        )

    def fit(self, reference_data: np.ndarray) -> "WassersteinDetector":
        """
        Fit the detector on reference distribution.

        Args:
            reference_data: Reference observations (n_samples,) or (n_samples, n_features).

        Returns:
            self
        """
        data = np.asarray(reference_data, dtype=np.float64)
        if data.ndim > 1:
            data = data.flatten()

        if len(data) < self.min_reference_samples:
            raise ValueError(
                f"Need at least {self.min_reference_samples} samples, got {len(data)}"
            )

        self._reference_data = data.copy()
        self._reference_mean = float(np.mean(data))
        self._reference_std = float(np.std(data))
        if self._reference_std < 1e-10:
            self._reference_std = 1e-10
        self._is_fitted = True

        logger.info(
            "Wasserstein fitted: mean=%.4f, std=%.4f, n=%d",
            self._reference_mean, self._reference_std, len(data),
        )
        return self

    def _compute_wasserstein(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """Compute Wasserstein-1 distance between two samples."""
        return float(stats.wasserstein_distance(data_a, data_b))

    def _permutation_test(
        self, new_data: np.ndarray, observed_distance: float
    ) -> float:
        """
        Estimate p-value via permutation test.

        Under null hypothesis (no drift), shuffling combined data should
        produce similar distances.

        Args:
            new_data: New observations.
            observed_distance: The observed W-1 distance.

        Returns:
            Estimated p-value.
        """
        if self._reference_data is None:
            return 1.0

        combined = np.concatenate([self._reference_data, new_data])
        n_ref = len(self._reference_data)
        n_new = len(new_data)
        n_total = len(combined)

        count_exceeding = 0
        actual_perms = min(self.n_permutations, 200)  # Cap for speed

        for _ in range(actual_perms):
            np.random.shuffle(combined)
            perm_dist = self._compute_wasserstein(
                combined[:n_ref], combined[n_ref:]
            )
            if perm_dist >= observed_distance:
                count_exceeding += 1

        p_value = (count_exceeding + 1) / (actual_perms + 1)
        return p_value

    def detect(self, new_data: np.ndarray) -> DriftResult:
        """
        Detect drift in new data against reference distribution.

        Args:
            new_data: New observations.

        Returns:
            DriftResult with Wasserstein distance and p-value.
        """
        if not self._is_fitted or self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detect()")

        data = np.asarray(new_data, dtype=np.float64)
        if data.ndim > 1:
            data = data.flatten()

        # Compute Wasserstein distance
        w_distance = self._compute_wasserstein(self._reference_data, data)

        # Statistical significance
        p_value = self._permutation_test(data, w_distance)

        # Determine drift status
        if w_distance >= self.threshold and p_value < self.significance_level:
            status = DriftStatus.DRIFT_DETECTED
            drift_detected = True
            details = (
                f"Wasserstein drift detected: W1={w_distance:.6f}, "
                f"p={p_value:.4f}. Distribution has significantly shifted."
            )
            logger.warning(details)
        elif w_distance >= self.threshold * 0.5:
            status = DriftStatus.WARNING
            drift_detected = False
            details = (
                f"Wasserstein warning: W1={w_distance:.6f}, p={p_value:.4f}"
            )
        else:
            status = DriftStatus.NO_DRIFT
            drift_detected = False
            details = ""

        return DriftResult(
            drift_detected=drift_detected,
            status=status,
            statistic=w_distance,
            mean_estimate=float(np.mean(data)),
            details=details,
        )

    def detect_batch(
        self, new_data: np.ndarray, window_size: int = 100
    ) -> List[DriftResult]:
        """
        Detect drift using sliding windows on new data.

        Args:
            new_data: New observations.
            window_size: Size of sliding window.

        Returns:
            List of DriftResult for each window.
        """
        data = np.asarray(new_data, dtype=np.float64).flatten()
        results = []
        for start in range(0, len(data), window_size // 2):
            window = data[start: start + window_size]
            if len(window) >= self.min_reference_samples // 2:
                results.append(self.detect(window))
        return results

    def get_state(self) -> dict:
        """Get current detector state."""
        return {
            "fitted": self._is_fitted,
            "reference_mean": self._reference_mean,
            "reference_std": self._reference_std,
            "reference_samples": len(self._reference_data) if self._reference_data is not None else 0,
        }

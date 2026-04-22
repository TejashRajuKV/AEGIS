"""
Distribution Comparator
========================
Compares two distributions using multiple statistical tests:
KS test, Jensen-Shannon divergence, and Population Stability Index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from scipy import stats

logger = logging.getLogger("aegis.drift.comparator")


@dataclass
class DistributionComparisonResult:
    """Result of comparing two distributions."""

    feature_name: str = ""
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    js_divergence: float = 0.0
    psi: float = 0.0
    mean_shift: float = 0.0
    std_ratio: float = 1.0
    drift_detected: bool = False
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "feature": self.feature_name,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "js_divergence": self.js_divergence,
            "psi": self.psi,
            "mean_shift": self.mean_shift,
            "std_ratio": self.std_ratio,
            "drift_detected": self.drift_detected,
        }


class DistributionComparator:
    """
    Compares two distributions using multiple statistical tests.

    Tests performed:
    1. Kolmogorov-Smirnov (KS): Tests if samples come from same distribution
    2. Jensen-Shannon Divergence (JSD): Measures similarity of probability distributions
    3. Population Stability Index (PSI): Industry standard for monitoring model inputs
    """

    def __init__(
        self,
        ks_threshold: float = 0.05,
        jsd_threshold: float = 0.1,
        psi_threshold: float = 0.25,
        min_samples: int = 20,
    ):
        """
        Initialize the comparator.

        Args:
            ks_threshold: KS p-value threshold for significance.
            jsd_threshold: JSD threshold for meaningful divergence.
            psi_threshold: PSI threshold (>0.25 = significant change).
            min_samples: Minimum samples per distribution for tests.
        """
        self.ks_threshold = ks_threshold
        self.jsd_threshold = jsd_threshold
        self.psi_threshold = psi_threshold
        self.min_samples = min_samples

    def compare(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "",
    ) -> DistributionComparisonResult:
        """
        Compare two distributions using all available tests.

        Args:
            reference: Reference distribution data.
            current: Current distribution data.
            feature_name: Name of the feature being compared.

        Returns:
            DistributionComparisonResult with all test results.
        """
        ref = np.asarray(reference, dtype=np.float64).flatten()
        cur = np.asarray(current, dtype=np.float64).flatten()

        if len(ref) < self.min_samples or len(cur) < self.min_samples:
            logger.warning(
                "Insufficient samples: ref=%d, cur=%d (min=%d)",
                len(ref), len(cur), self.min_samples,
            )
            return DistributionComparisonResult(
                feature_name=feature_name,
                summary="Insufficient samples for comparison",
            )

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(ref, cur)

        # Jensen-Shannon divergence
        jsd = self._jensen_shannon_divergence(ref, cur)

        # Population Stability Index
        psi = self._population_stability_index(ref, cur)

        # Mean shift and std ratio
        mean_shift = float(np.mean(cur) - np.mean(ref))
        ref_std = float(np.std(ref))
        cur_std = float(np.std(cur))
        std_ratio = cur_std / max(ref_std, 1e-10)

        # Determine drift
        drift_detected = (
            ks_pval < self.ks_threshold
            or jsd > self.jsd_threshold
            or psi > self.psi_threshold
        )

        summary_parts = []
        if ks_pval < self.ks_threshold:
            summary_parts.append(f"KS test significant (p={ks_pval:.4f})")
        if jsd > self.jsd_threshold:
            summary_parts.append(f"JSD elevated ({jsd:.4f})")
        if psi > self.psi_threshold:
            level = "significant" if psi > 0.5 else "moderate"
            summary_parts.append(f"PSI {level} ({psi:.4f})")

        summary = "; ".join(summary_parts) if summary_parts else "No significant drift"

        if drift_detected:
            logger.warning(
                "Distribution drift detected for '%s': %s",
                feature_name, summary,
            )

        return DistributionComparisonResult(
            feature_name=feature_name,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            js_divergence=jsd,
            psi=psi,
            mean_shift=mean_shift,
            std_ratio=std_ratio,
            drift_detected=drift_detected,
            summary=summary,
        )

    def compare_featurewise(
        self,
        ref_df: np.ndarray,
        curr_df: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, DistributionComparisonResult]:
        """
        Compare distributions feature by feature.

        Args:
            ref_df: Reference data (n_samples, n_features).
            curr_df: Current data (n_samples, n_features).
            feature_names: Names of features.

        Returns:
            Dictionary mapping feature name to comparison result.
        """
        ref = np.asarray(ref_df, dtype=np.float64)
        cur = np.asarray(curr_df, dtype=np.float64)

        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
            cur = cur.reshape(-1, 1)

        n_features = min(ref.shape[1], cur.shape[1])
        results = {}

        for i in range(n_features):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
            results[name] = self.compare(ref[:, i], cur[:, i], feature_name=name)

        return results

    def get_summary(
        self, comparison: DistributionComparisonResult
    ) -> str:
        """
        Generate a human-readable summary of comparison results.

        Args:
            comparison: DistributionComparisonResult.

        Returns:
            Summary string.
        """
        lines = [
            f"Feature: {comparison.feature_name or 'unnamed'}",
            f"  KS statistic: {comparison.ks_statistic:.4f} (p={comparison.ks_pvalue:.4f})",
            f"  Jensen-Shannon divergence: {comparison.js_divergence:.4f}",
            f"  Population Stability Index: {comparison.psi:.4f}",
            f"  Mean shift: {comparison.mean_shift:.4f}",
            f"  Std ratio: {comparison.std_ratio:.4f}",
            f"  Drift detected: {comparison.drift_detected}",
            f"  Summary: {comparison.summary}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 50) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions.

        Args:
            p: First distribution samples.
            q: Second distribution samples.
            n_bins: Number of bins for histogram estimation.

        Returns:
            JSD value (0 = identical, higher = more different).
        """
        all_data = np.concatenate([p, q])
        bins = np.linspace(all_data.min(), all_data.max(), n_bins + 1)

        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p_hist = p_hist + eps
        q_hist = q_hist + eps

        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()

        # M = (P + Q) / 2
        m = 0.5 * (p_hist + q_hist)

        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        kl_pm = np.sum(p_hist * np.log(p_hist / m))
        kl_qm = np.sum(q_hist * np.log(q_hist / m))
        jsd = 0.5 * (kl_pm + kl_qm)

        return float(jsd)

    @staticmethod
    def _population_stability_index(
        expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI).

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change

        Args:
            expected: Expected distribution samples.
            actual: Actual distribution samples.
            n_bins: Number of bins.

        Returns:
            PSI value.
        """
        all_data = np.concatenate([expected, actual])
        breakpoints = np.percentile(all_data, np.linspace(0, 100, n_bins + 1))

        # Ensure unique breakpoints
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0

        expected_hist, _ = np.histogram(expected, bins=breakpoints)
        actual_hist, _ = np.histogram(actual, bins=breakpoints)

        # Convert to proportions
        expected_pct = expected_hist / max(len(expected), 1) + 1e-10
        actual_pct = actual_hist / max(len(actual), 1) + 1e-10

        # PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return psi

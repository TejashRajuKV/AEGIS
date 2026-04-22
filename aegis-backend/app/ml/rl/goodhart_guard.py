"""
Goodhart's Law Guard
====================
Prevents the RL agent from over-optimizing a single fairness metric
at the expense of others. Monitors metric trajectories and penalizes
excessive degradation of non-target metrics.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("aegis.rl.goodhart_guard")


class AlertLevel(Enum):
    """Severity levels for Goodhart's Law alerts."""

    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"


@dataclass
class GoodhartReport:
    """Report from the Goodhart's Law guard check."""

    is_safe: bool
    alert_level: AlertLevel
    degraded_metrics: List[str] = field(default_factory=list)
    degradation_scores: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    details: str = ""


@dataclass
class MetricHistory:
    """Tracks history of a single metric."""

    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=200))
    max_len: int = 200

    def add(self, value: float) -> None:
        """Add a new value to the history."""
        self.values.append(value)

    def get_recent(self, n: int = 10) -> List[float]:
        """Get the n most recent values."""
        return list(self.values)[-n:]

    def trend(self, n: int = 10) -> float:
        """Compute recent trend (positive = improving, negative = degrading)."""
        recent = self.get_recent(n)
        if len(recent) < 2:
            return 0.0
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def mean(self, n: int = 10) -> float:
        """Mean of recent values."""
        recent = self.get_recent(n)
        return float(np.mean(recent)) if recent else 0.0

    def std(self, n: int = 10) -> float:
        """Std of recent values."""
        recent = self.get_recent(n)
        return float(np.std(recent)) if len(recent) > 1 else 0.0


class GoodhartGuard:
    """
    Guards against Goodhart's Law in the reward function.

    When optimizing multiple objectives simultaneously, the agent may
    over-optimize one metric while causing others to degrade. This guard
    detects such patterns and adjusts the reward accordingly.

    For fairness metrics (gaps), DEGRADATION means the gap is INCREASING.
    For accuracy, DEGRADATION means accuracy is DECREASING.
    """

    def __init__(
        self,
        max_degradation_ratio: float = 0.5,
        accuracy_min_threshold: float = 0.60,
        warning_ratio: float = 0.3,
        trend_window: int = 10,
    ):
        """
        Initialize the Goodhart's Law guard.

        Args:
            max_degradation_ratio: Max acceptable degradation ratio before UNSAFE.
            accuracy_min_threshold: Minimum acceptable accuracy.
            warning_ratio: Degradation ratio for WARNING level.
            trend_window: Window size for trend analysis.
        """
        self.max_degradation_ratio = max_degradation_ratio
        self.accuracy_min_threshold = accuracy_min_threshold
        self.warning_ratio = warning_ratio
        self.trend_window = trend_window

        self._histories: Dict[str, MetricHistory] = {}
        self._init_histories()

        logger.info(
            "GoodhartGuard initialized: max_degrade=%.2f, acc_min=%.2f",
            max_degradation_ratio, accuracy_min_threshold,
        )

    def _init_histories(self) -> None:
        """Initialize metric history trackers."""
        metric_names = [
            "accuracy",
            "demographic_parity_gap",
            "equalized_odds_gap",
            "calibration_error",
        ]
        for name in metric_names:
            self._histories[name] = MetricHistory(name=name)

    def check(
        self,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
    ) -> GoodhartReport:
        """
        Check if the metric transition shows signs of Goodhart's Law.

        Args:
            old_metrics: Dict of metric_name -> value before action.
            new_metrics: Dict of metric_name -> value after action.

        Returns:
            GoodhartReport with safety assessment.
        """
        degraded_metrics = []
        degradation_scores = {}

        for metric_name, old_val in old_metrics.items():
            if metric_name not in new_metrics:
                continue

            new_val = new_metrics[metric_name]
            history = self._histories.get(metric_name)
            if history is None:
                continue

            # Update history
            history.add(new_val)

            # Compute degradation
            # For gaps/errors: increasing is bad (degradation)
            # For accuracy: decreasing is bad (degradation)
            is_gap_metric = "gap" in metric_name or "error" in metric_name

            if is_gap_metric:
                degradation = (new_val - old_val) / max(abs(old_val), 1e-8)
            else:
                degradation = (old_val - new_val) / max(abs(old_val), 1e-8)

            degradation_scores[metric_name] = degradation

            if degradation > self.warning_ratio:
                degraded_metrics.append(metric_name)

            # Special check: accuracy floor
            if metric_name == "accuracy" and new_val < self.accuracy_min_threshold:
                degraded_metrics.append("accuracy_below_floor")
                degradation_scores["accuracy_below_floor"] = (
                    self.accuracy_min_threshold - new_val
                )

        # Determine alert level
        if len(degraded_metrics) == 0:
            alert_level = AlertLevel.SAFE
            recommendation = "All metrics are within acceptable bounds."
        elif any(
            degradation_scores.get(m, 0) > self.max_degradation_ratio
            for m in degraded_metrics
        ):
            alert_level = AlertLevel.UNSAFE
            recommendation = (
                "STOP: Excessive metric degradation detected. "
                f"Degraded: {degraded_metrics}. "
                "Consider reducing learning rate or adding constraints."
            )
        else:
            alert_level = AlertLevel.WARNING
            recommendation = (
                f"Caution: Some metrics degrading: {degraded_metrics}. "
                "Monitor closely and adjust reward weights if needed."
            )

        details = "; ".join(
            f"{m}: {degradation_scores[m]:.4f}" for m in degraded_metrics
        )

        report = GoodhartReport(
            is_safe=alert_level == AlertLevel.SAFE,
            alert_level=alert_level,
            degraded_metrics=degraded_metrics,
            degradation_scores=degradation_scores,
            recommendation=recommendation,
            details=details,
        )

        if alert_level != AlertLevel.SAFE:
            logger.warning(
                "GoodhartGuard [%s]: %s (%s)",
                alert_level.value, recommendation, details,
            )

        return report

    def adjust_reward(
        self, base_reward: float, report: GoodhartReport
    ) -> float:
        """
        Adjust the reward based on the Goodhart report.

        Args:
            base_reward: Original shaped reward.
            report: Goodhart report from check().

        Returns:
            Adjusted reward with penalties applied.
        """
        if report.alert_level == AlertLevel.SAFE:
            return base_reward
        elif report.alert_level == AlertLevel.WARNING:
            penalty = -0.3 * abs(base_reward)
            return base_reward + penalty
        else:  # UNSAFE
            penalty = -0.8 * abs(base_reward)
            return base_reward + penalty

    def get_recommendation(self, report: GoodhartReport) -> str:
        """
        Get a human-readable recommendation.

        Args:
            report: Goodhart report.

        Returns:
            Recommendation string.
        """
        return report.recommendation

    def get_metric_trends(self) -> Dict[str, float]:
        """
        Get current trends for all tracked metrics.

        Returns:
            Dict of metric_name -> trend value.
        """
        return {
            name: history.trend(self.trend_window)
            for name, history in self._histories.items()
        }

    def reset(self) -> None:
        """Reset all metric histories."""
        self._histories.clear()
        self._init_histories()
        logger.info("GoodhartGuard reset")

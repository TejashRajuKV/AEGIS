"""
Multi-Objective Reward Shaper
=============================
Computes shaped rewards combining accuracy improvement with
multiple fairness metric improvements for the RL agent.

Reward = alpha * accuracy_delta + beta * dp_improvement +
          gamma * eo_improvement + delta * calibration_improvement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("aegis.rl.reward_shaper")


@dataclass
class RewardComponents:
    """Individual reward components for analysis."""

    accuracy_reward: float = 0.0
    dp_reward: float = 0.0
    eo_reward: float = 0.0
    calibration_reward: float = 0.0
    accuracy_penalty: float = 0.0
    total_reward: float = 0.0


@dataclass
class FairnessMetrics:
    """Container for fairness metrics at a given point."""

    accuracy: float = 0.0
    demographic_parity_gap: float = 0.0
    equalized_odds_gap: float = 0.0
    calibration_error: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "demographic_parity_gap": self.demographic_parity_gap,
            "equalized_odds_gap": self.equalized_odds_gap,
            "calibration_error": self.calibration_error,
        }


class MultiObjectiveRewardShaper:
    """
    Computes multi-objective reward for the RL agent.

    The reward function combines accuracy and fairness improvements:
        R = alpha * acc_delta + beta * dp_improve + gamma * eo_improve
            + delta * cal_improve - penalty

    A penalty is applied if accuracy drops below the configured threshold.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.2,
        accuracy_drop_threshold: float = 0.02,
        accuracy_drop_penalty: float = 5.0,
        fairness_floor: float = 0.001,
    ):
        """
        Initialize the multi-objective reward shaper.

        Args:
            alpha: Weight for accuracy improvement component.
            beta: Weight for demographic parity improvement.
            gamma: Weight for equalized odds improvement.
            delta: Weight for calibration error improvement.
            accuracy_drop_threshold: Max acceptable accuracy drop.
            accuracy_drop_penalty: Penalty magnitude for exceeding threshold.
            fairness_floor: Minimum improvement to count as progress.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.accuracy_drop_penalty = accuracy_drop_penalty
        self.fairness_floor = fairness_floor

        logger.info(
            "RewardShaper: alpha=%.2f, beta=%.2f, gamma=%.2f, delta=%.2f",
            alpha, beta, gamma, delta,
        )

    def compute_reward(
        self,
        old_metrics: FairnessMetrics,
        new_metrics: FairnessMetrics,
        action: Optional[np.ndarray] = None,
    ) -> RewardComponents:
        """
        Compute the shaped reward for transitioning from old to new metrics.

        Args:
            old_metrics: Metrics before the action was applied.
            new_metrics: Metrics after the action was applied.
            action: The action taken (for potential regularization).

        Returns:
            RewardComponents with individual and total reward.
        """
        # Accuracy component: reward for maintaining or improving accuracy
        acc_delta = new_metrics.accuracy - old_metrics.accuracy
        accuracy_reward = self.alpha * acc_delta * 100.0  # Scale up for visibility

        # Fairness improvement components (gap reduction is positive)
        dp_improvement = old_metrics.demographic_parity_gap - new_metrics.demographic_parity_gap
        eo_improvement = old_metrics.equalized_odds_gap - new_metrics.equalized_odds_gap
        cal_improvement = old_metrics.calibration_error - new_metrics.calibration_error

        dp_reward = self.beta * dp_improvement * 10.0
        eo_reward = self.gamma * eo_improvement * 10.0
        cal_reward = self.delta * cal_improvement * 10.0

        # Penalty for excessive accuracy drop
        accuracy_penalty = 0.0
        if acc_delta < -self.accuracy_drop_threshold:
            excess_drop = abs(acc_delta) - self.accuracy_drop_threshold
            accuracy_penalty = -self.accuracy_drop_penalty * excess_drop * 100.0
            logger.debug(
                "Accuracy drop penalty applied: %.4f (excess: %.4f)",
                accuracy_penalty, excess_drop,
            )

        # L2 regularization on action magnitude (optional)
        action_penalty = 0.0
        if action is not None:
            action_penalty = -0.01 * float(np.sum(action ** 2))

        total = (
            accuracy_reward
            + dp_reward
            + eo_reward
            + cal_reward
            + accuracy_penalty
            + action_penalty
        )

        components = RewardComponents(
            accuracy_reward=accuracy_reward,
            dp_reward=dp_reward,
            eo_reward=eo_reward,
            calibration_reward=cal_reward,
            accuracy_penalty=accuracy_penalty + action_penalty,
            total_reward=total,
        )

        logger.debug(
            "Reward: total=%.4f (acc=%.4f, dp=%.4f, eo=%.4f, cal=%.4f, pen=%.4f)",
            total, accuracy_reward, dp_reward, eo_reward, cal_reward,
            accuracy_penalty + action_penalty,
        )

        return components

    def get_reward_components(
        self, components: RewardComponents
    ) -> Dict[str, float]:
        """
        Get individual reward components as a dictionary.

        Args:
            components: RewardComponents from compute_reward.

        Returns:
            Dictionary of component name to value.
        """
        return {
            "accuracy_reward": components.accuracy_reward,
            "dp_reward": components.dp_reward,
            "eo_reward": components.eo_reward,
            "calibration_reward": components.calibration_reward,
            "penalty": components.accuracy_penalty,
            "total_reward": components.total_reward,
        }

    def normalize_reward(self, reward: float, running_mean: float, running_std: float) -> float:
        """
        Normalize reward using running statistics.

        Args:
            reward: Raw reward value.
            running_mean: Running mean of rewards.
            running_std: Running standard deviation of rewards.

        Returns:
            Normalized reward.
        """
        if running_std < 1e-8:
            return 0.0
        return (reward - running_mean) / running_std

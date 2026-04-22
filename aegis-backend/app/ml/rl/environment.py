"""
Fairness RL Environment
=======================
Custom environment for training the PPO agent to mitigate bias.
State: fairness metrics + accuracy + model configuration.
Action: threshold adjustments + feature reweighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.ml.rl.action_space import ContinuousActionSpace
from app.ml.rl.reward_shaper import FairnessMetrics, MultiObjectiveRewardShaper

logger = logging.getLogger("aegis.rl.environment")


@dataclass
class ModelState:
    """Holds the current model state that actions modify."""

    thresholds: np.ndarray
    feature_weights: np.ndarray
    accuracy: float = 0.0
    demographic_parity_gap: float = 0.0
    equalized_odds_gap: float = 0.0
    calibration_error: float = 0.0


class FairnessRLEnvironment:
    """
    Gym-style RL environment for bias mitigation.

    The agent observes the current fairness state and takes actions
    to adjust thresholds and feature weights. The reward is shaped
    from improvements in both accuracy and fairness metrics.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        model: Any = None,
        n_thresholds: int = 5,
        n_feature_weights: int = 10,
        target_dp_gap: float = 0.05,
        target_eo_gap: float = 0.05,
        max_steps: int = 100,
        accuracy_floor: float = 0.55,
        evaluate_fn: Optional[callable] = None,
    ):
        """
        Initialize the fairness RL environment.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (n_samples,).
            sensitive_features: Sensitive attribute values (n_samples,).
            model: Optional scikit-learn-like model with predict/predict_proba.
            n_thresholds: Number of threshold adjustment actions.
            n_feature_weights: Number of feature weight actions.
            target_dp_gap: Target demographic parity gap.
            target_eo_gap: Target equalized odds gap.
            max_steps: Maximum steps per episode.
            accuracy_floor: Minimum acceptable accuracy.
            evaluate_fn: Custom evaluation function (model, X, y, sf) -> FairnessMetrics.
        """
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features
        self.model = model
        self.target_dp_gap = target_dp_gap
        self.target_eo_gap = target_eo_gap
        self.max_steps = max_steps
        self.accuracy_floor = accuracy_floor
        self.evaluate_fn = evaluate_fn

        self.action_space = ContinuousActionSpace(
            n_thresholds=n_thresholds,
            n_feature_weights=min(n_feature_weights, X.shape[1]),
        )

        self.reward_shaper = MultiObjectiveRewardShaper()

        self.current_step = 0
        self.state: Optional[ModelState] = None
        self.best_metrics: Optional[FairnessMetrics] = None
        self.best_score = -np.inf
        self.history: List[Dict] = []

        self._initialize_state()

        logger.info(
            "FairnessRLEnvironment: X=%s, max_steps=%d, action_dim=%d",
            X.shape, max_steps, self.action_space.action_dim,
        )

    def _initialize_state(self) -> None:
        """Set initial state with default thresholds and weights."""
        thresholds = np.zeros(self.action_space.n_thresholds, dtype=np.float32)
        weights = np.ones(
            min(self.action_space.n_feature_weights, self.X.shape[1]),
            dtype=np.float32,
        )
        self.state = ModelState(
            thresholds=thresholds,
            feature_weights=weights,
        )
        self._evaluate_current_state()

    def _evaluate_current_state(self) -> None:
        """Evaluate the current state's fairness metrics."""
        if self.evaluate_fn is not None and self.model is not None:
            metrics = self.evaluate_fn(
                self.model, self.X, self.y, self.sensitive_features,
                self.state.thresholds, self.state.feature_weights,
            )
            self.state.accuracy = metrics.accuracy
            self.state.demographic_parity_gap = metrics.demographic_parity_gap
            self.state.equalized_odds_gap = metrics.equalized_odds_gap
            self.state.calibration_error = metrics.calibration_error
        elif self.model is not None:
            self._basic_evaluation()

    def _basic_evaluation(self) -> None:
        """Basic evaluation using model predictions."""
        try:
            preds = self.model.predict(self.X)
            accuracy = float(np.mean(preds == self.y))
            self.state.accuracy = accuracy
            # Set initial gaps as defaults when no full pipeline available
            self.state.demographic_parity_gap = max(
                0.0, 1.0 - accuracy
            )  # Proxy
            self.state.equalized_odds_gap = max(
                0.0, 0.8 - accuracy
            )  # Proxy
            self.state.calibration_error = 0.1
        except Exception as e:
            logger.warning("Basic evaluation failed: %s", e)

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector from current state."""
        if self.state is None:
            return np.zeros(self.observation_dim, dtype=np.float32)

        obs = np.array([
            self.state.accuracy,
            self.state.demographic_parity_gap,
            self.state.equalized_odds_gap,
            self.state.calibration_error,
            self.current_step / max(self.max_steps, 1),
        ], dtype=np.float32)

        # Append normalized thresholds and weights
        t_norm = self.state.thresholds / max(
            abs(self.action_space.threshold_bounds[1]), 1e-8
        )
        w_norm = self.state.feature_weights / max(
            self.action_space.weight_bounds[1], 1e-8
        )
        obs = np.concatenate([obs, t_norm, w_norm])
        return obs.astype(np.float32)

    @property
    def observation_dim(self) -> int:
        """Dimension of the observation space."""
        base = 5  # metrics + progress
        return base + self.action_space.action_dim

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation.
        """
        self.current_step = 0
        self._initialize_state()
        self.history = []
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action array of shape (action_dim,).

        Returns:
            Tuple of (observation, reward, done, info).
        """
        # Clip action to valid bounds
        action = self.action_space.clip(action)

        # Store old metrics
        old_metrics = FairnessMetrics(
            accuracy=self.state.accuracy,
            demographic_parity_gap=self.state.demographic_parity_gap,
            equalized_odds_gap=self.state.equalized_odds_gap,
            calibration_error=self.state.calibration_error,
        )

        # Apply action: update thresholds and weights
        thresholds, weights = self.action_space.split_actions(action)
        self.state.thresholds = (
            self.state.thresholds + thresholds * 0.1
        )
        self.state.feature_weights = np.clip(
            self.state.feature_weights * (1.0 + weights * 0.05),
            0.0,
            self.action_space.weight_bounds[1],
        )

        # Ensure weights match X columns
        if len(self.state.feature_weights) < self.X.shape[1]:
            pad = np.ones(
                self.X.shape[1] - len(self.state.feature_weights),
                dtype=np.float32,
            )
            full_weights = np.concatenate([self.state.feature_weights, pad])
        else:
            full_weights = self.state.feature_weights[: self.X.shape[1]]

        # Re-evaluate
        self._evaluate_current_state()

        # Compute reward
        new_metrics = FairnessMetrics(
            accuracy=self.state.accuracy,
            demographic_parity_gap=self.state.demographic_parity_gap,
            equalized_odds_gap=self.state.equalized_odds_gap,
            calibration_error=self.state.calibration_error,
        )
        reward_components = self.reward_shaper.compute_reward(
            old_metrics, new_metrics, action
        )
        reward = reward_components.total_reward

        self.current_step += 1

        # Check termination
        done = False
        target_reached = (
            self.state.demographic_parity_gap <= self.target_dp_gap
            and self.state.equalized_odds_gap <= self.target_eo_gap
            and self.state.accuracy >= self.accuracy_floor
        )
        if target_reached:
            done = True
            reward += 10.0  # Bonus for reaching target
            logger.info("Target fairness reached at step %d!", self.current_step)
        elif self.current_step >= self.max_steps:
            done = True
        elif self.state.accuracy < self.accuracy_floor * 0.8:
            done = True
            reward -= 5.0
            logger.warning(
                "Episode terminated: accuracy %.3f below floor",
                self.state.accuracy,
            )

        # Track best
        composite = (
            self.state.accuracy
            - 0.5 * self.state.demographic_parity_gap
            - 0.5 * self.state.equalized_odds_gap
        )
        if composite > self.best_score:
            self.best_score = composite
            self.best_metrics = new_metrics

        info = {
            "step": self.current_step,
            "accuracy": self.state.accuracy,
            "dp_gap": self.state.demographic_parity_gap,
            "eo_gap": self.state.equalized_odds_gap,
            "calibration_error": self.state.calibration_error,
            "target_reached": target_reached,
            "reward_components": {
                k: v for k, v in reward_components.__dict__.items()
            },
        }
        self.history.append(info)

        return self._get_observation(), reward, done, info

    def get_best_result(self) -> Dict:
        """
        Get the best result found during the episode.

        Returns:
            Dictionary with best metrics and actions.
        """
        if self.state is None:
            return {}
        return {
            "best_accuracy": self.state.accuracy,
            "best_dp_gap": self.state.demographic_parity_gap,
            "best_eo_gap": self.state.equalized_odds_gap,
            "best_calibration": self.state.calibration_error,
            "best_score": self.best_score,
            "final_thresholds": self.state.thresholds.tolist(),
            "final_weights": self.state.feature_weights.tolist(),
            "steps_taken": self.current_step,
        }

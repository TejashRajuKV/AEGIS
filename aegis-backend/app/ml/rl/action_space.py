"""
Continuous Action Space for RL Agent
=====================================
Defines the continuous action space used by the PPO agent
to adjust decision thresholds and feature reweighting parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("aegis.rl.action_space")


@dataclass
class ActionBounds:
    """Bounds for a single action dimension."""

    name: str
    low: float
    high: float
    default: float

    def clip(self, value: float) -> float:
        """Clip a value to be within [low, high]."""
        return float(np.clip(value, self.low, self.high))


class ContinuousActionSpace:
    """
    Continuous action space for the fairness RL agent.

    The action space consists of:
    - Threshold adjustments: continuous values that shift decision boundaries
    - Feature weight adjustments: continuous values that reweight feature importance
    """

    def __init__(
        self,
        n_thresholds: int = 5,
        n_feature_weights: int = 10,
        threshold_bounds: Tuple[float, float] = (-0.5, 0.5),
        weight_bounds: Tuple[float, float] = (0.0, 2.0),
    ):
        """
        Initialize the continuous action space.

        Args:
            n_thresholds: Number of decision threshold adjustment actions.
            n_feature_weights: Number of feature weight adjustment actions.
            threshold_bounds: (low, high) for threshold adjustments.
            weight_bounds: (low, high) for weight adjustments.
        """
        self.n_thresholds = n_thresholds
        self.n_feature_weights = n_feature_weights
        self.threshold_bounds = threshold_bounds
        self.weight_bounds = weight_bounds

        self._actions: List[ActionBounds] = []
        self._build_action_space()

        logger.info(
            "Initialized action space: %d thresholds, %d feature weights, total dim=%d",
            n_thresholds,
            n_feature_weights,
            self.action_dim,
        )

    def _build_action_space(self) -> None:
        """Build the full action space from individual components."""
        for i in range(self.n_thresholds):
            self._actions.append(
                ActionBounds(
                    name=f"threshold_{i}",
                    low=self.threshold_bounds[0],
                    high=self.threshold_bounds[1],
                    default=0.0,
                )
            )
        for i in range(self.n_feature_weights):
            self._actions.append(
                ActionBounds(
                    name=f"feature_weight_{i}",
                    low=self.weight_bounds[0],
                    high=self.weight_bounds[1],
                    default=1.0,
                )
            )

    @property
    def action_dim(self) -> int:
        """Total dimensionality of the action space."""
        return len(self._actions)

    @property
    def action_names(self) -> List[str]:
        """Names of each action dimension."""
        return [a.name for a in self._actions]

    def sample(self) -> np.ndarray:
        """
        Sample a random action uniformly from the action space.

        Returns:
            numpy array of shape (action_dim,) with sampled actions.
        """
        low = np.array([a.low for a in self._actions], dtype=np.float32)
        high = np.array([a.high for a in self._actions], dtype=np.float32)
        return np.random.uniform(low, high).astype(np.float32)

    def clip(self, action: np.ndarray) -> np.ndarray:
        """
        Clip an action to be within the valid bounds.

        Args:
            action: numpy array of shape (action_dim,).

        Returns:
            Clipped action array.
        """
        clipped = np.empty_like(action, dtype=np.float32)
        for i, bounds in enumerate(self._actions):
            clipped[i] = bounds.clip(float(action[i]))
        return clipped

    def default_action(self) -> np.ndarray:
        """
        Return the default (no-change) action.

        Returns:
            numpy array of shape (action_dim,) with default values.
        """
        return np.array(
            [a.default for a in self._actions], dtype=np.float32
        )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lower and upper bounds for all actions.

        Returns:
            Tuple of (low_array, high_array), each of shape (action_dim,).
        """
        low = np.array([a.low for a in self._actions], dtype=np.float32)
        high = np.array([a.high for a in self._actions], dtype=np.float32)
        return low, high

    def split_actions(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a flat action array into threshold and weight components.

        Args:
            action: numpy array of shape (action_dim,).

        Returns:
            Tuple of (thresholds, feature_weights) arrays.
        """
        thresholds = action[: self.n_thresholds]
        weights = action[self.n_thresholds :]
        return thresholds, weights

    def action_to_dict(self, action: np.ndarray) -> Dict[str, float]:
        """
        Convert a flat action array to a named dictionary.

        Args:
            action: numpy array of shape (action_dim,).

        Returns:
            Dictionary mapping action names to values.
        """
        return {a.name: float(action[i]) for i, a in enumerate(self._actions)}

    def __repr__(self) -> str:
        return (
            f"ContinuousActionSpace(dim={self.action_dim}, "
            f"thresholds={self.n_thresholds}, "
            f"weights={self.n_feature_weights})"
        )

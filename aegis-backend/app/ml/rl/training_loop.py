"""
PPO Training Loop
=================
Full training loop for the PPO autopilot that integrates
reward shaping, Pareto analysis, and Goodhart's Law guard.

Executes SEQUENTIALLY (one step at a time) for 16GB RAM constraint.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("aegis.rl.training_loop")


@dataclass
class TrainingConfig:
    """Configuration for PPO training."""

    n_episodes: int = 50
    max_steps_per_episode: int = 100
    target_dp_gap: float = 0.05
    target_eo_gap: float = 0.05
    accuracy_floor: float = 0.55
    learning_rate: float = 3e-4
    lr_annealing: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs_per_update: int = 4
    mini_batch_size: int = 64
    rollout_steps: int = 2048
    early_stopping_patience: int = 10
    save_every: int = 5
    checkpoint_dir: str = "checkpoints/ppo_agent"


@dataclass
class StepMetrics:
    """Metrics from a single training step."""

    episode: int
    step: int
    reward: float
    accuracy: float
    dp_gap: float
    eo_gap: float
    calibration_error: float
    actor_loss: float
    critic_loss: float
    entropy: float
    kl_divergence: float
    goodhart_safe: bool
    pareto_multiplier: float


@dataclass
class TrainingResult:
    """Result of a complete training run."""

    success: bool
    best_accuracy: float
    best_dp_gap: float
    best_eo_gap: float
    best_calibration: float
    total_episodes: int
    total_steps: int
    training_time_seconds: float
    best_thresholds: List[float]
    best_weights: List[float]
    metrics_history: List[Dict] = field(default_factory=list)


class PPOTrainingLoop:
    """
    Full PPO training loop for the AEGIS autopilot.

    Sequential execution: one environment step at a time,
    accumulating rollouts, then updating the PPO agent.
    Memory-efficient design for 16GB RAM gaming laptop.
    """

    def __init__(
        self,
        env: Any,
        config: Optional[TrainingConfig] = None,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize the training loop.

        Args:
            env: FairnessRLEnvironment instance.
            config: Training configuration. Uses defaults if None.
            progress_callback: Optional callback for progress updates.
        """
        self.env = env
        self.config = config or TrainingConfig()
        self.progress_callback = progress_callback

        # Import here to avoid issues if torch not available
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PPO training")

        from app.ml.rl.ppo_agent import PPOAgent
        from app.ml.rl.pareto_reward import ParetoRewardModifier, ParetoPoint
        from app.ml.rl.goodhart_guard import GoodhartGuard
        from app.ml.rl.reward_shaper import MultiObjectiveRewardShaper

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.agent = PPOAgent(
            state_dim=env.observation_dim,
            action_dim=env.action_space.action_dim,
            lr=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_epsilon=self.config.clip_epsilon,
            value_coef=self.config.value_coef,
            entropy_coef=self.config.entropy_coef,
            max_grad_norm=self.config.max_grad_norm,
            n_epochs=self.config.n_epochs_per_update,
            mini_batch_size=self.config.mini_batch_size,
            device=device,
        )

        self.pareto_modifier = ParetoRewardModifier()
        self.goodhart_guard = GoodhartGuard()
        self.reward_shaper = MultiObjectiveRewardShaper()

        self.metrics_history: List[Dict] = []
        self._no_improvement_count = 0
        self._best_score = -np.inf

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        logger.info(
            "PPOTrainingLoop initialized: episodes=%d, device=%s",
            self.config.n_episodes, device,
        )

    def train(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        sensitive_features: Optional[np.ndarray] = None,
        model: Any = None,
        evaluate_fn: Optional[callable] = None,
    ) -> TrainingResult:
        """
        Run the full PPO training loop.

        Args:
            X: Feature matrix (optional, uses env data if None).
            y: Labels (optional, uses env data if None).
            sensitive_features: Sensitive features (optional).
            model: Model to evaluate (optional).
            evaluate_fn: Custom evaluation function.

        Returns:
            TrainingResult with all metrics and best found configuration.
        """
        start_time = time.time()
        logger.info("Starting PPO training: %d episodes", self.config.n_episodes)

        # Override env if new data provided
        if X is not None and y is not None:
            from app.ml.rl.environment import FairnessRLEnvironment
            self.env = FairnessRLEnvironment(
                X=X, y=y,
                sensitive_features=sensitive_features or np.zeros(len(y)),
                model=model,
                evaluate_fn=evaluate_fn,
                max_steps=self.config.max_steps_per_episode,
                target_dp_gap=self.config.target_dp_gap,
                target_eo_gap=self.config.target_eo_gap,
                accuracy_floor=self.config.accuracy_floor,
            )
            # Re-create agent with new env dimensions
            from app.ml.rl.ppo_agent import PPOAgent
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.agent = PPOAgent(
                state_dim=self.env.observation_dim,
                action_dim=self.env.action_space.action_dim,
                lr=self.config.learning_rate,
                device=device,
            )

        total_steps = 0

        for episode in range(self.config.n_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_losses: List[Dict] = []
            old_metrics: Dict[str, float] = {}  # track previous step metrics for Goodhart

            # Fix CRIT-09: initialise goodhart_report before the step loop so it is
            # always defined when referenced in episode_metrics below, even if the
            # inner loop executes zero iterations.
            from types import SimpleNamespace
            goodhart_report = SimpleNamespace(is_safe=True)

            for step in range(self.config.max_steps_per_episode):
                # Select action
                action, log_prob, value = self.agent.select_action(state)

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)

                # Goodhart's Law check — compare previous step vs current step
                current_metrics = {
                    "accuracy": info.get("accuracy", 0.5),
                    "demographic_parity_gap": info.get("dp_gap", 0.5),
                    "equalized_odds_gap": info.get("eo_gap", 0.5),
                    "calibration_error": info.get("calibration_error", 0.1),
                }
                goodhart_report = self.goodhart_guard.check(
                    old_metrics if step > 0 else current_metrics,
                    current_metrics,
                )
                old_metrics = current_metrics  # advance for next step
                adjusted_reward = self.goodhart_guard.adjust_reward(reward, goodhart_report)

                # Pareto reward modification
                pareto_point = ParetoPoint(
                    accuracy=info.get("accuracy", 0.5),
                    dp_gap=info.get("dp_gap", 0.5),
                    eo_gap=info.get("eo_gap", 0.5),
                    calibration_error=info.get("calibration_error", 0.1),
                )
                pareto_mult = self.pareto_modifier.get_pareto_reward_multiplier(pareto_point)
                adjusted_reward *= pareto_mult
                self.pareto_modifier.update_pareto_front(pareto_point)

                # Store transition
                self.agent.store_transition(state, action, adjusted_reward, done, log_prob, value)

                episode_reward += adjusted_reward
                total_steps += 1
                state = next_state

                # PPO update when we have enough data
                if len(self.agent.buffer) >= self.config.rollout_steps:
                    losses = self.agent.update()
                    episode_losses.append(losses or {})

                if done:
                    break

            # Fix CRIT-01: Update after every episode, not mid-step.
            # The original check (len(buffer) >= rollout_steps=2048) inside a
            # max_steps_per_episode=100 loop NEVER fired — agent never learned.
            # Now we update whenever the buffer has at least one mini-batch worth
            # of data, which is standard PPO practice.
            if len(self.agent.buffer) >= self.config.mini_batch_size:
                # Fix MED-08: capture losses from the end-of-episode update.
                final_losses = self.agent.update()
                if final_losses:
                    episode_losses.append(final_losses)

            # Log episode metrics
            avg_loss = {}
            if episode_losses:
                for k in episode_losses[0]:
                    avg_loss[k] = np.mean([l.get(k, 0) for l in episode_losses])

            result = self.env.get_best_result()
            composite = result.get("best_score", -np.inf)

            episode_metrics = {
                "episode": episode,
                "reward": episode_reward,
                "steps": self.env.current_step,
                "accuracy": result.get("best_accuracy", 0),
                "dp_gap": result.get("best_dp_gap", 1),
                "eo_gap": result.get("best_eo_gap", 1),
                "calibration": result.get("best_calibration", 1),
                "composite_score": composite,
                "losses": avg_loss,
                "goodhart_safe": goodhart_report.is_safe,
                "pareto_front_size": len(self.pareto_modifier.pareto_front),
            }
            self.metrics_history.append(episode_metrics)

            logger.info(
                "Episode %d: reward=%.3f, acc=%.3f, dp=%.3f, eo=%.3f, steps=%d",
                episode, episode_reward,
                episode_metrics["accuracy"],
                episode_metrics["dp_gap"],
                episode_metrics["eo_gap"],
                self.env.current_step,
            )

            # Progress callback
            if self.progress_callback:
                self.progress_callback(episode, self.config.n_episodes, episode_metrics)

            # Check for improvement
            if composite > self._best_score:
                self._best_score = composite
                self._no_improvement_count = 0
                # Save best checkpoint
                best_path = os.path.join(
                    self.config.checkpoint_dir, "best_model.pt"
                )
                self.agent.save_checkpoint(best_path)
            else:
                self._no_improvement_count += 1

            # Early stopping
            if self._no_improvement_count >= self.config.early_stopping_patience:
                logger.info(
                    "Early stopping: no improvement for %d episodes",
                    self._no_improvement_count,
                )
                break

            # Check if target reached
            if result.get("best_dp_gap", 1) <= self.config.target_dp_gap and \
               result.get("best_eo_gap", 1) <= self.config.target_eo_gap:
                logger.info("Target fairness metrics reached!")
                break

            # Periodic checkpoint
            if (episode + 1) % self.config.save_every == 0:
                ckpt_path = os.path.join(
                    self.config.checkpoint_dir, f"checkpoint_ep{episode}.pt"
                )
                self.agent.save_checkpoint(ckpt_path)

        # Collect final results
        final_result = self.env.get_best_result()
        training_time = time.time() - start_time

        success = (
            final_result.get("best_dp_gap", 1) <= self.config.target_dp_gap
            and final_result.get("best_eo_gap", 1) <= self.config.target_eo_gap
        )

        result = TrainingResult(
            success=success,
            best_accuracy=final_result.get("best_accuracy", 0),
            best_dp_gap=final_result.get("best_dp_gap", 1),
            best_eo_gap=final_result.get("best_eo_gap", 1),
            best_calibration=final_result.get("best_calibration", 1),
            total_episodes=episode + 1,
            total_steps=total_steps,
            training_time_seconds=training_time,
            best_thresholds=final_result.get("final_thresholds", []),
            best_weights=final_result.get("final_weights", []),
            metrics_history=self.metrics_history,
        )

        logger.info(
            "Training complete: success=%s, acc=%.3f, dp=%.3f, eo=%.3f, time=%.1fs",
            success, result.best_accuracy, result.best_dp_gap,
            result.best_eo_gap, training_time,
        )

        return result

    def get_best_actions(self) -> Dict[str, Any]:
        """
        Get the best actions found during training.

        Returns:
            Dictionary with best thresholds, weights, and metrics.
        """
        return self.env.get_best_result()

    @property
    def mini_batch_size(self) -> int:
        """Minimum batch size for PPO update."""
        return self.config.mini_batch_size

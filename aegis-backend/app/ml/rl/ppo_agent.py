"""
PPO Agent
=========
Main PPO agent that manages the actor-critic network and
implements the PPO update algorithm with GAE advantage estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("aegis.rl.ppo_agent")


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data for PPO update."""

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def clear(self) -> None:
        """Clear all stored data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.states)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)


class PPOAgent:
    """
    Proximal Policy Optimization agent for continuous action spaces.

    Implements the PPO-Clip algorithm with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Entropy bonus for exploration
    - Gradient norm clipping
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        mini_batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize the PPO agent.

        Args:
            state_dim: Dimension of observation space.
            action_dim: Dimension of action space.
            lr: Learning rate.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            clip_epsilon: PPO clip range.
            value_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            n_epochs: Number of optimization epochs per update.
            mini_batch_size: Mini-batch size for PPO update.
            device: torch device ('cpu' or 'cuda').
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PPO agent")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size

        self.device = torch.device(device)

        # Import networks here to avoid issues if torch not available
        from app.ml.rl.ppo_network import PPOActorCritic

        self.actor_critic = PPOActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=lr, eps=1e-5
        )

        self.buffer = RolloutBuffer()

        logger.info(
            "PPOAgent: state=%d, action=%d, lr=%.1e, device=%s",
            state_dim, action_dim, lr, device,
        )

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select an action given a state observation.

        Args:
            state: Observation array.
            deterministic: If True, return the mean action.

        Returns:
            Tuple of (action, log_prob, value).
        """
        self.actor_critic.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t, log_prob_t, value_t = self.actor_critic.get_action(
                state_t, deterministic=deterministic
            )
            action = action_t.squeeze(0).cpu().numpy()
            log_prob = log_prob_t.squeeze(0).item()
            value = value_t.squeeze(0).item()

        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a transition in the rollout buffer."""
        self.buffer.add(state, action, reward, done, log_prob, value)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0,
    ) -> List[float]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards.
            values: List of value predictions.
            dones: List of done flags.
            next_value: Value of the next state (0 if done).

        Returns:
            List of GAE advantage estimates.
        """
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.

        Returns:
            Dictionary of loss metrics.
        """
        if len(self.buffer) < self.mini_batch_size:
            logger.warning(
                "Buffer too small for update: %d < %d",
                len(self.buffer), self.mini_batch_size,
            )
            return {}

        self.actor_critic.train()

        states = np.array(self.buffer.states, dtype=np.float32)
        actions = np.array(self.buffer.actions, dtype=np.float32)
        rewards = self.buffer.rewards
        dones = [float(d) for d in self.buffer.dones]
        old_log_probs = self.buffer.log_probs
        old_values = self.buffer.values

        # Compute next value
        with torch.no_grad():
            last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
            _, _, next_value = self.actor_critic.get_action(last_state, deterministic=True)
            next_value = next_value.squeeze(0).item()

        # Compute GAE
        advantages = self.compute_gae(rewards, old_values, dones, next_value)
        returns = [
            a + v for a, v in zip(advantages, old_values)
        ]

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).unsqueeze(1).to(self.device)
        advantages_t = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        returns_t = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                batch_idx = indices[start:end]

                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                # Evaluate current policy
                new_log_probs, new_values, entropy = self.actor_critic.evaluate(
                    batch_states, batch_actions
                )

                # Compute ratio: pi_new / pi_old
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.MSELoss()(new_values, batch_returns)

                # Total loss
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy.mean()
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # KL divergence for monitoring
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean().item()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl
                n_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        metrics = {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "kl_divergence": total_kl / max(n_updates, 1),
            "n_updates": n_updates,
        }

        logger.debug(
            "PPO update: actor=%.4f, critic=%.4f, entropy=%.4f, kl=%.4f",
            metrics["actor_loss"],
            metrics["critic_loss"],
            metrics["entropy"],
            metrics["kl_divergence"],
        )

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save agent state to checkpoint file."""
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)
        logger.info("PPO agent checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from checkpoint file."""
        # Fix HIGH-04: weights_only=True prevents arbitrary Python deserialization.
        # weights_only=False is equivalent to pickle.load and allows RCE via
        # a malicious .pt file.  PyTorch 2.0+ recommends weights_only=True.
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("PPO agent checkpoint loaded: %s", path)

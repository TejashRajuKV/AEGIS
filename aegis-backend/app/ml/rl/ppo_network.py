"""
PPO Actor-Critic Networks
=========================
Neural network architectures for the PPO agent:
- Actor: outputs mean actions for continuous control (Tanh-squashed)
- Critic: outputs state value estimates
- ActorCritic: combined module for efficient forward passes
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("aegis.rl.ppo_network")


def _build_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    activation: str = "tanh",
    final_activation: Optional[str] = None,
    dropout: float = 0.0,
):
    """Build a multi-layer perceptron."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PPO networks")

    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(nn.LayerNorm(h_dim))
        if activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "gelu":
            layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if final_activation == "tanh":
        layers.append(nn.Tanh())
    elif final_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


if TORCH_AVAILABLE:

    class PPOActorNetwork(nn.Module):
        """
        Actor network for PPO with continuous action space.

        Outputs the mean of a Gaussian distribution for each action dimension.
        Actions are bounded via Tanh activation to keep them in [-1, 1],
        which is then scaled to the actual action bounds by the environment.

        Architecture: state_dim -> [256, 256, 128] -> action_dim (tanh)
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: Optional[list] = None,
            log_std_init: float = -0.5,
            dropout: float = 0.0,
        ):
            """
            Initialize the actor network.

            Args:
                state_dim: Dimension of the observation/state space.
                action_dim: Dimension of the action space.
                hidden_dims: List of hidden layer sizes. Default [256, 256, 128].
                log_std_init: Initial log standard deviation (controls exploration).
                dropout: Dropout rate between layers.
            """
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [256, 256, 128]

            self.state_dim = state_dim
            self.action_dim = action_dim

            self.mean_net = _build_mlp(
                state_dim, hidden_dims, action_dim,
                activation="tanh", final_activation="tanh", dropout=dropout,
            )

            self.log_std = nn.Parameter(
                torch.full((action_dim,), log_std_init, dtype=torch.float32)
            )

            self._init_weights()
            logger.info(
                "PPOActorNetwork: state_dim=%d, action_dim=%d, params=%d",
                state_dim, action_dim, sum(p.numel() for p in self.parameters()),
            )

        def _init_weights(self) -> None:
            """Initialize network weights with orthogonal initialization."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)

        def forward(
            self, state: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass through the actor network.

            Args:
                state: Tensor of shape (batch, state_dim).

            Returns:
                Tuple of (action_mean, log_std) each of shape (batch, action_dim).
            """
            action_mean = self.mean_net(state)
            log_std = self.log_std.expand_as(action_mean)
            return action_mean, log_std

        def get_distribution(
            self, state: torch.Tensor
        ) -> "Normal":
            """
            Get the action distribution for the given state.

            Args:
                state: Tensor of shape (batch, state_dim).

            Returns:
                Normal distribution over actions.
            """
            action_mean, log_std = self.forward(state)
            std = log_std.exp()
            return Normal(action_mean, std)

    class PPOCriticNetwork(nn.Module):
        """
        Critic network for PPO that estimates state values V(s).

        Architecture: state_dim -> [256, 256, 128] -> 1
        """

        def __init__(
            self,
            state_dim: int,
            hidden_dims: Optional[list] = None,
            dropout: float = 0.0,
        ):
            """
            Initialize the critic network.

            Args:
                state_dim: Dimension of the observation/state space.
                hidden_dims: List of hidden layer sizes.
                dropout: Dropout rate between layers.
            """
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [256, 256, 128]

            self.state_dim = state_dim

            self.value_net = _build_mlp(
                state_dim, hidden_dims, 1,
                activation="tanh", final_activation=None, dropout=dropout,
            )

            self._init_weights()
            logger.info(
                "PPOCriticNetwork: state_dim=%d, params=%d",
                state_dim, sum(p.numel() for p in self.parameters()),
            )

        def _init_weights(self) -> None:
            """Initialize network weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    nn.init.constant_(module.bias, 0.0)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Forward pass to estimate state value.

            Args:
                state: Tensor of shape (batch, state_dim).

            Returns:
                Value tensor of shape (batch, 1).
            """
            return self.value_net(state)

    class PPOActorCritic(nn.Module):
        """
        Combined Actor-Critic network for PPO.

        Encapsulates both the actor (policy) and critic (value) networks
        in a single module for efficient computation.
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            actor_hidden: Optional[list] = None,
            critic_hidden: Optional[list] = None,
            log_std_init: float = -0.5,
            dropout: float = 0.0,
        ):
            """
            Initialize the combined actor-critic network.

            Args:
                state_dim: Dimension of the state space.
                action_dim: Dimension of the action space.
                actor_hidden: Actor hidden layer sizes.
                critic_hidden: Critic hidden layer sizes.
                log_std_init: Initial log std dev for actor.
                dropout: Dropout rate.
            """
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim

            self.actor = PPOActorNetwork(
                state_dim, action_dim, actor_hidden, log_std_init, dropout
            )
            self.critic = PPOCriticNetwork(state_dim, critic_hidden, dropout)

        def forward(
            self, state: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass returning action parameters and value estimate.

            Args:
                state: Tensor of shape (batch, state_dim).

            Returns:
                Tuple of (action_mean, log_std, value).
            """
            action_mean, log_std = self.actor(state)
            value = self.critic(state)
            return action_mean, log_std, value

        def get_action(
            self, state: torch.Tensor, deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Sample an action from the current policy.

            Args:
                state: Tensor of shape (batch, state_dim) or (state_dim,).
                deterministic: If True, return mean action without noise.

            Returns:
                Tuple of (action, log_prob, value).
            """
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action_mean, log_std, value = self.forward(state)

            if deterministic:
                action = action_mean
                log_prob = torch.zeros(state.shape[0], 1, device=state.device)
            else:
                std = log_std.exp()
                dist = Normal(action_mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

            return action, log_prob, value

        def evaluate(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate log probability and value for given state-action pairs.

            Used during PPO update to compute the policy loss.

            Args:
                state: Tensor of shape (batch, state_dim).
                action: Tensor of shape (batch, action_dim).

            Returns:
                Tuple of (log_prob, value, entropy).
            """
            action_mean, log_std, value = self.forward(state)
            std = log_std.exp()
            dist = Normal(action_mean, std)

            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)

            return log_prob, value, entropy

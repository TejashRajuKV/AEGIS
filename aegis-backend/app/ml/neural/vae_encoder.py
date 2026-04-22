"""
VAE Encoder
===========
Encoder network for the Conditional VAE. Maps input data to
latent distribution parameters (mean and log variance).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _EncoderBase = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _EncoderBase = object
    TORCH_AVAILABLE = False

logger = logging.getLogger("aegis.neural.encoder")


class VAEEncoder(_EncoderBase):  # type: ignore[misc]
    """
    Encoder network for the Conditional Variational Autoencoder.

    Architecture: input_dim -> [256, 128] -> 2 * latent_dim
    Output: (mu, log_var) parameterizing the latent Gaussian.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 0,
        latent_dim: int = 16,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize the VAE encoder.

        Args:
            input_dim: Dimension of input data.
            condition_dim: Dimension of conditional input (0 = unconditional).
            latent_dim: Dimension of latent space.
            hidden_dims: Hidden layer sizes. Default [256, 128].
            dropout: Dropout rate.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VAE encoder")

        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [256, 128]

        total_input = input_dim + condition_dim

        layers = []
        prev_dim = total_input
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.encoder_net = nn.Sequential(*layers)

        # Latent distribution parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        self._init_weights()

        logger.info(
            "VAEEncoder: input=%d, condition=%d, latent=%d, hidden=%s",
            input_dim, condition_dim, latent_dim, hidden_dims,
        )

    def _init_weights(self) -> None:
        """Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: "torch.Tensor", condition: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch, input_dim).
            condition: Optional condition tensor of shape (batch, condition_dim).

        Returns:
            Tuple of (mu, log_var) each of shape (batch, latent_dim).
        """
        if condition is not None:
            h = torch.cat([x, condition], dim=-1)
        else:
            h = x

        h = self.encoder_net(h)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)

        return mu, log_var

    def reparameterize(
        self, mu: "torch.Tensor", log_var: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Reparameterization trick: z = mu + std * epsilon.

        Args:
            mu: Mean tensor.
            log_var: Log variance tensor.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

"""
VAE Decoder
===========
Decoder network for the Conditional VAE. Reconstructs input
from latent representation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _DecoderBase = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _DecoderBase = object
    TORCH_AVAILABLE = False

logger = logging.getLogger("aegis.neural.decoder")


class VAEDecoder(_DecoderBase):  # type: ignore[misc]
    """
    Decoder network for the Conditional Variational Autoencoder.

    Architecture: latent_dim + condition_dim -> [128, 256] -> input_dim
    Uses sigmoid output for normalized data reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 0,
        latent_dim: int = 16,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        output_activation: str = "sigmoid",
    ):
        """
        Initialize the VAE decoder.

        Args:
            input_dim: Dimension of output data (should match encoder input).
            condition_dim: Dimension of conditional input.
            latent_dim: Dimension of latent space.
            hidden_dims: Hidden layer sizes. Default [128, 256] (mirror of encoder).
            dropout: Dropout rate.
            output_activation: Output activation ('sigmoid', 'tanh', or 'none').
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VAE decoder")

        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [128, 256]

        total_input = latent_dim + condition_dim

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

        self.decoder_net = nn.Sequential(*layers)

        # Output projection
        self.output_layer = nn.Linear(prev_dim, input_dim)

        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()

        self._init_weights()

        logger.info(
            "VAEDecoder: output=%d, condition=%d, latent=%d, hidden=%s",
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
        self, z: "torch.Tensor", condition: Optional["torch.Tensor"] = None
    ) -> "torch.Tensor":
        """
        Decode latent vector to reconstructed input.

        Args:
            z: Latent tensor of shape (batch, latent_dim).
            condition: Optional condition tensor of shape (batch, condition_dim).

        Returns:
            Reconstructed tensor of shape (batch, input_dim).
        """
        if condition is not None:
            h = torch.cat([z, condition], dim=-1)
        else:
            h = z

        h = self.decoder_net(h)
        output = self.output_layer(h)
        return self.output_activation(output)

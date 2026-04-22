"""AEGIS Conditional VAE – Merged CVAE for counterfactual generation.

Merges:
- V2: Defensive imports, configurable architecture (encoder_hidden,
  decoder_hidden, dropout, output_activation), encode/decode/sample
  methods, detailed loss_function with kl_weight.
- V3: save/load state-dict methods, inline reparameterize, optimizer.
- V5: compute_loss() convenience wrapper, get_info() summary, generate()
  method.

Uses ``get_logger`` from ``app.utils.logger``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

# Defensive import – torch is optional at module level (V2 pattern)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _CVAEBase = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    _CVAEBase = object
    TORCH_AVAILABLE = False

from app.ml.neural.vae_encoder import VAEEncoder
from app.ml.neural.vae_decoder import VAEDecoder
from app.utils.logger import get_logger

logger = get_logger("conditional_vae")


class ConditionalVAE(_CVAEBase):  # type: ignore[misc]
    """Conditional Variational Autoencoder for counterfactual generation.

    The CVAE learns to generate data conditioned on sensitive attributes,
    enabling "what-if" analysis: *What would this person's outcome be if
    their race / gender were different?*

    Architecture is fully configurable via constructor parameters (V2).
    Supports ``save`` / ``load`` for persistence (V3), ``compute_loss`` and
    ``get_info`` helpers (V5), and inline reparameterization (V3).
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 1,
        latent_dim: int = 16,
        encoder_hidden: Optional[list] = None,
        decoder_hidden: Optional[list] = None,
        hidden_dims: Optional[tuple] = None,
        dropout: float = 0.1,
        output_activation: str = "sigmoid",
        learning_rate: float = 1e-3,
    ):
        """Initialise the Conditional VAE.

        Args:
            input_dim: Dimension of input features.
            condition_dim: Dimension of conditioning variable (e.g. one-hot
                demographic encoding).
            latent_dim: Latent-space dimensionality.
            encoder_hidden: List of encoder hidden-layer sizes (V2).
                Falls back to *hidden_dims* reversed if ``None``.
            decoder_hidden: List of decoder hidden-layer sizes (V2).
                Falls back to *hidden_dims* if ``None``.
            hidden_dims: Tuple of shared hidden sizes (V3 compat).  Used as
                fallback when ``encoder_hidden`` / ``decoder_hidden`` are
                ``None``.
            dropout: Dropout rate applied in encoder/decoder (V2).
            output_activation: Output activation function name (V2).
            learning_rate: Learning rate for the Adam optimizer (V3).
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ConditionalVAE. "
                "Install it with: pip install torch"
            )

        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.output_activation = output_activation
        self.dropout = dropout

        # Resolve hidden dims – prefer V2 params, fall back to V3 hidden_dims
        if encoder_hidden is None and hidden_dims is not None:
            encoder_hidden = list(reversed(hidden_dims))
        if decoder_hidden is None and hidden_dims is not None:
            decoder_hidden = list(hidden_dims)

        self.encoder = VAEEncoder(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden,
            dropout=dropout,
        )

        self.decoder = VAEDecoder(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            dropout=dropout,
            output_activation=output_activation,
        )

        # V3: built-in optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "ConditionalVAE initialised: input=%d, condition=%d, "
            "latent=%d, params=%d",
            input_dim,
            condition_dim,
            latent_dim,
            n_params,
        )

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → decode.

        Args:
            x: Input tensor (batch, input_dim).
            condition: Condition tensor (batch, condition_dim).

        Returns:
            Tuple of (reconstruction, mu, log_var).
        """
        mu, log_var = self.encode(x, condition)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, condition)
        return reconstruction, mu, log_var

    # ------------------------------------------------------------------
    # Encoder / Decoder wrappers (V2)
    # ------------------------------------------------------------------

    def encode(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input conditioned on sensitive attribute.

        Returns:
            Tuple of (mu, log_var), each (batch, latent_dim).
        """
        return self.encoder(x, condition)

    def decode(
        self, z: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent vector conditioned on target attribute.

        Returns:
            Reconstructed tensor (batch, input_dim).
        """
        return self.decoder(z, condition)

    # ------------------------------------------------------------------
    # Reparameterization (V3 inline, V2 delegates to encoder)
    # ------------------------------------------------------------------

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for differentiable sampling.

        z = mu + std * epsilon,  epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Loss function (V2 detailed + V5 compute_loss convenience)
    # ------------------------------------------------------------------

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute CVAE loss = Reconstruction + KL divergence.

        Loss = MSE(x, x_recon) + kl_weight * KL(q(z|x,c) || p(z))

        Args:
            x: Original input.
            x_recon: Reconstructed input.
            mu: Latent mean.
            log_var: Latent log variance.
            kl_weight: Weight for KL divergence term.

        Returns:
            Dict with ``reconstruction_loss``, ``kl_loss``, ``total_loss``.
        """
        # Reconstruction loss (MSE, mean-reduced for stability)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Convenience loss wrapper (V5).

        Same computation as :meth:`loss_function` with ``kl_weight`` aliased
        to ``beta`` for beta-VAE convention.

        Returns:
            Dict with ``total_loss``, ``recon_loss``, ``kl_loss``.
        """
        result = self.loss_function(x, reconstruction, mu, log_var, kl_weight=beta)
        # Normalise keys to V5 naming
        return {
            "total_loss": result["total_loss"],
            "recon_loss": result["reconstruction_loss"],
            "kl_loss": result["kl_loss"],
        }

    # ------------------------------------------------------------------
    # Sampling (V2 + V5 generate)
    # ------------------------------------------------------------------

    def sample(
        self,
        num_samples: int,
        condition: torch.Tensor,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Generate new samples conditioned on the given attribute.

        Args:
            num_samples: Number of samples to generate.
            condition: Condition tensor (1, cond_dim) or (num_samples, cond_dim).
            device: Torch device string.

        Returns:
            Generated samples tensor (num_samples, input_dim).
        """
        self.eval()
        with torch.no_grad():
            if condition.dim() == 1:
                condition = condition.unsqueeze(0)
            if condition.shape[0] == 1 and num_samples > 1:
                condition = condition.expand(num_samples, -1)

            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z, condition.to(device))
        return samples

    def generate(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Generate from explicit latent vectors with conditioning (V5).

        Args:
            z: Latent vectors (batch, latent_dim).
            condition: Target conditioning (batch, condition_dim).

        Returns:
            Generated samples (batch, input_dim).
        """
        with torch.no_grad():
            return self.decoder(z, condition)

    # ------------------------------------------------------------------
    # Counterfactual generation (all versions)
    # ------------------------------------------------------------------

    def generate_counterfactual(
        self,
        x: torch.Tensor,
        original_condition: torch.Tensor,
        target_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a counterfactual: same person, different demographic.

        Encodes with *original_condition*, decodes with *target_condition*.

        Args:
            x: Original input (batch, input_dim).
            original_condition: Original demographic (batch, condition_dim).
            target_condition: Target demographic (batch, condition_dim).

        Returns:
            Counterfactual tensor (batch, input_dim).
        """
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encode(x, original_condition)
            z = self.reparameterize(mu, log_var)
            counterfactual = self.decode(z, target_condition)
        return counterfactual

    # ------------------------------------------------------------------
    # Persistence (V3)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state dict to *path*.

        Creates parent directories if they do not exist.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("CVAE saved to %s", path)

    def load(self, path: str) -> None:
        """Load model state dict from *path*."""
        self.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        logger.info("CVAE loaded from %s", path)

    # ------------------------------------------------------------------
    # Model info (V5)
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        """Return a summary dict of model configuration and size.

        Returns:
            Dict with ``input_dim``, ``condition_dim``, ``latent_dim``,
            ``output_activation``, ``dropout``, ``num_parameters``,
            ``trainable_parameters``.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "input_dim": self.input_dim,
            "condition_dim": self.condition_dim,
            "latent_dim": self.latent_dim,
            "output_activation": self.output_activation,
            "dropout": self.dropout,
            "num_parameters": total,
            "trainable_parameters": trainable,
        }

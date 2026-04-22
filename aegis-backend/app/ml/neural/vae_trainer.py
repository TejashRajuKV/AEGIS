"""
VAE Trainer
===========
Training loop for the Conditional Variational Autoencoder.
Memory-efficient sequential processing for 16GB RAM.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("aegis.neural.trainer")


@dataclass
class VAETrainingConfig:
    """Configuration for CVAE training."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    kl_annealing_epochs: int = 10
    early_stopping_patience: int = 15
    gradient_clip: float = 5.0
    checkpoint_dir: str = "checkpoints/conditional_vae"
    device: str = "auto"
    save_every: int = 10


@dataclass
class TrainingHistory:
    """History of training metrics."""

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    recon_losses: List[float] = field(default_factory=list)
    kl_losses: List[float] = field(default_factory=list)
    kl_weights: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


class VAETrainer:
    """
    Training loop for the Conditional Variational Autoencoder.

    Features:
    - KL annealing to prevent posterior collapse
    - Early stopping
    - Cosine annealing learning rate schedule
    - Checkpoint saving/loading
    - Sequential processing (memory efficient)
    """

    def __init__(
        self,
        cvae: Any,
        config: Optional[VAETrainingConfig] = None,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize the trainer.

        Args:
            cvae: ConditionalVAE model to train.
            config: Training configuration.
            progress_callback: Optional callback for progress updates.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VAE training")

        self.config = config or VAETrainingConfig()
        self.progress_callback = progress_callback
        self.history = TrainingHistory()

        # Device selection
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.cvae = cvae.to(self.device)
        self.optimizer = optim.Adam(
            self.cvae.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=1e-6
        )

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        logger.info(
            "VAETrainer: device=%s, epochs=%d, batch=%d, lr=%.1e",
            self.device, self.config.epochs, self.config.batch_size,
            self.config.learning_rate,
        )

    def train(
        self,
        train_data: np.ndarray,
        condition_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_condition: Optional[np.ndarray] = None,
    ) -> TrainingHistory:
        """
        Train the CVAE.

        Args:
            train_data: Training features (n_samples, n_features).
            condition_data: Training conditions (n_samples, condition_dim).
            val_data: Optional validation features.
            val_condition: Optional validation conditions.

        Returns:
            TrainingHistory with all metrics.
        """
        start_time = time.time()

        # Prepare tensors
        X_train = torch.FloatTensor(train_data).to(self.device)
        C_train = torch.FloatTensor(condition_data).to(self.device)

        train_dataset = TensorDataset(X_train, C_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            drop_last=False,
        )

        # Validation setup
        val_loader = None
        if val_data is not None and val_condition is not None:
            X_val = torch.FloatTensor(val_data).to(self.device)
            C_val = torch.FloatTensor(val_condition).to(self.device)
            val_dataset = TensorDataset(X_val, C_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False,
            )

        no_improve = 0

        for epoch in range(self.config.epochs):
            self.cvae.train()
            epoch_losses = []
            epoch_recon = []
            epoch_kl = []

            # KL annealing: linear warmup
            if epoch < self.config.kl_annealing_epochs:
                kl_weight = epoch / max(self.config.kl_annealing_epochs, 1)
            else:
                kl_weight = 1.0

            for batch_x, batch_c in train_loader:
                self.optimizer.zero_grad()

                x_recon, mu, log_var = self.cvae(batch_x, batch_c)
                losses = self.cvae.loss_function(
                    batch_x, x_recon, mu, log_var, kl_weight=kl_weight
                )

                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(
                    self.cvae.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()

                epoch_losses.append(losses["total_loss"].item())
                epoch_recon.append(losses["reconstruction_loss"].item())
                epoch_kl.append(losses["kl_loss"].item())

            self.scheduler.step()

            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.cvae.eval()
                val_batch_losses = []
                with torch.no_grad():
                    for val_x, val_c in val_loader:
                        v_recon, v_mu, v_log_var = self.cvae(val_x, val_c)
                        v_losses = self.cvae.loss_function(
                            val_x, v_recon, v_mu, v_log_var, kl_weight=kl_weight
                        )
                        val_batch_losses.append(v_losses["total_loss"].item())
                val_loss = float(np.mean(val_batch_losses))

            # Record history
            avg_train = float(np.mean(epoch_losses))
            self.history.train_losses.append(avg_train)
            self.history.val_losses.append(val_loss)
            self.history.recon_losses.append(float(np.mean(epoch_recon)))
            self.history.kl_losses.append(float(np.mean(epoch_kl)))
            self.history.kl_weights.append(kl_weight)

            # Early stopping
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                no_improve = 0
                self._save_checkpoint("best_model.pt")
            else:
                no_improve += 1

            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"checkpoint_ep{epoch}.pt")

            if no_improve >= self.config.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d: no improvement for %d epochs",
                    epoch, no_improve,
                )
                break

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d: train=%.4f, val=%.4f, recon=%.4f, kl=%.4f, kl_w=%.2f",
                    epoch + 1, avg_train, val_loss,
                    self.history.recon_losses[-1],
                    self.history.kl_losses[-1],
                    kl_weight,
                )

            if self.progress_callback:
                self.progress_callback(epoch, self.config.epochs, {
                    "train_loss": avg_train,
                    "val_loss": val_loss,
                    "kl_weight": kl_weight,
                })

        elapsed = time.time() - start_time
        logger.info(
            "Training complete: best_epoch=%d, best_val=%.4f, time=%.1fs",
            self.history.best_epoch, self.history.best_val_loss, elapsed,
        )
        return self.history

    def generate_samples(
        self, condition: np.ndarray, n_samples: int = 10
    ) -> np.ndarray:
        """
        Generate samples using the trained CVAE.

        Args:
            condition: Condition array (condition_dim,) or (n_samples, condition_dim).
            n_samples: Number of samples.

        Returns:
            Generated samples array (n_samples, input_dim).
        """
        self.cvae.eval()
        cond = torch.FloatTensor(condition).to(self.device)
        samples = self.cvae.sample(n_samples, cond, device=str(self.device))
        return samples.cpu().numpy()

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.cvae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": {
                "best_epoch": self.history.best_epoch,
                "best_val_loss": self.history.best_val_loss,
            },
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.cvae.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Checkpoint loaded: %s", path)

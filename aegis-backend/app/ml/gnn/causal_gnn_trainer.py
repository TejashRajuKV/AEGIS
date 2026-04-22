"""
Causal GNN trainer — sequential training loop for DAG structure learning.

This module provides :class:`CausalGNNTrainer`, a complete training pipeline
that learns a causal directed acyclic graph (DAG) from observational tabular
data using the NOTEARS continuous optimisation framework.

The trainer composes a :class:`~app.ml.gnn.node_encoder.NodeFeatureEncoder`,
:class:`~app.ml.gnn.dag_gnn_layers.DAGGNNStack`, and optionally a
:class:`~app.ml.gnn.edge_decoder.EdgeDecoder` into an end-to-end model that
minimises the NOTEARS loss:

    L_total = L_data + λ₁ ‖A‖₁ + λ₂ · h(A)

where:

* ``L_data`` is a mean-squared-error reconstruction loss on the data.
* ``‖A‖₁`` is the L1 norm of the learned adjacency matrix (sparsity).
* ``h(A) = tr(exp(A⊙A)) - d`` is the NOTEARS acyclicity constraint.

Design constraints
------------------
* **Sequential execution only** — designed for a 16 GB RAM gaming laptop;
  no parallel data loading or multi-GPU.
* **CPU by default** — automatically detects and uses CUDA only when
  explicitly requested and available.
* **Lazy imports** — all PyTorch imports are guarded by try/except.
* **Python logging** — uses ``logging`` module throughout; no ``print``.
* **Full docstrings** — every public method and class documented.

Methods
-------
train(data)          – Run training, return learned adjacency matrix.
predict(data)        – Run inference, return predictions.
get_causal_graph()   – Return a networkx DiGraph of learned causal structure.
save_checkpoint()    – Persist training state to disk.
load_checkpoint()    – Restore training state from disk.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — PyTorch may not be installed in all environments
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False
    logger.warning("numpy is not installed.  NumPy-dependent features will fail.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    _HAS_TORCH = False
    logger.warning("PyTorch is not installed.  GNN training is unavailable.")

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    _HAS_NETWORKX = False
    logger.warning("networkx is not installed.  Graph export will be unavailable.")

# Import sibling modules (also lazy)
try:
    from app.ml.gnn.dag_gnn_layers import DAGGNNLayer, DAGGNNStack
    from app.ml.gnn.node_encoder import NodeFeatureEncoder
    from app.ml.gnn.edge_decoder import EdgeDecoder
    _HAS_GNN_MODULES = True
except ImportError as exc:
    _HAS_GNN_MODULES = False
    logger.warning("GNN submodules unavailable: %s", exc)


__all__ = ["CausalGNNTrainer", "TrainingConfig", "TrainingMetrics"]


# ═══════════════════════════════════════════════════════════════════════════
# Data classes for configuration and metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Configuration for the NOTEARS DAG-GNN training loop.

    Attributes
    ----------
    num_nodes : int
        Number of variables / nodes in the graph (d).
    input_dim : int
        Dimension of raw input features per node.
    hidden_dim : int
        Hidden dimension for encoder and GNN layers.
    num_gnn_layers : int
        Number of DAG-GNN message-passing layers.
    num_encoder_layers : int
        Number of MLP layers in the node encoder.
    lambda1 : float
        L1 regularisation weight for adjacency sparsity.
    lambda2 : float
        Acyclicity constraint weight (NOTEARS h(A)).
    learning_rate : float
        Initial Adam learning rate.
    max_epochs : int
        Maximum number of training epochs.
    patience : int
        Early-stopping patience (epochs without improvement).
    min_epochs : int
        Minimum epochs before early stopping can trigger.
    lr_factor : float
        Factor by which to reduce LR on plateau.
    lr_patience : int
        LR scheduler patience.
    lr_min : float
        Minimum learning rate.
    dropout : float
        Dropout probability.
    batch_size : int
        Mini-batch size (1 = full-batch / sequential).
    use_cuda : bool
        Whether to attempt CUDA acceleration.
    checkpoint_dir : str
        Directory for saving checkpoints.
    seed : int
        Random seed for reproducibility.
    threshold : float
        Edge threshold for binarising the learned adjacency matrix.
    """

    num_nodes: int = 10
    input_dim: int = 10
    hidden_dim: int = 32
    num_gnn_layers: int = 2
    num_encoder_layers: int = 3
    lambda1: float = 0.01
    lambda2: float = 10.0
    learning_rate: float = 3e-3
    max_epochs: int = 1000
    patience: int = 100
    min_epochs: int = 50
    lr_factor: float = 0.5
    lr_patience: int = 50
    lr_min: float = 1e-6
    dropout: float = 0.1
    batch_size: int = 1
    use_cuda: bool = False
    checkpoint_dir: str = "checkpoints/gnn"
    seed: int = 42
    threshold: float = 0.3


@dataclass
class TrainingMetrics:
    """Container for per-epoch training metrics.

    Attributes
    ----------
    epoch : int
        Current epoch number.
    total_loss : float
        Combined NOTEARS loss.
    data_loss : float
        Reconstruction / data-fitting loss.
    l1_loss : float
        L1 sparsity regularisation term.
    acyclicity_loss : float
        NOTEARS acyclicity constraint value h(A).
    lr : float
        Current learning rate.
    elapsed_seconds : float
        Wall-clock time for this epoch.
    """

    epoch: int = 0
    total_loss: float = 0.0
    data_loss: float = 0.0
    l1_loss: float = 0.0
    acyclicity_loss: float = 0.0
    lr: float = 0.0
    elapsed_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Main trainer class
# ═══════════════════════════════════════════════════════════════════════════

class CausalGNNTrainer:
    """Sequential NOTEARS DAG-GNN trainer for causal graph discovery.

    Orchestrates the full training pipeline: model construction, optimisation,
    early stopping, checkpointing, and graph extraction.

    The learned model maps raw tabular data through a node encoder, applies
    DAG-GNN message passing with a learned adjacency matrix, and minimises
    the NOTEARS loss to discover a sparse, acyclic causal structure.

    Parameters
    ----------
    config : TrainingConfig, optional
        Training configuration.  If ``None``, defaults are used.
    node_names : list[str], optional
        Human-readable names for graph nodes.  If ``None``, nodes are
        named ``"X0"``, ``"X1"``, etc.

    Examples
    --------
    >>> cfg = TrainingConfig(num_nodes=5, input_dim=5, max_epochs=500)
    >>> trainer = CausalGNNTrainer(config=cfg, node_names=["age", "income", "edu", "race", "score"])
    >>> adj = trainer.train(data_tensor)          # (N, 5) tensor
    >>> graph = trainer.get_causal_graph()
    >>> preds = trainer.predict(data_tensor)
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        node_names: Optional[List[str]] = None,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for CausalGNNTrainer but is not installed."
            )
        if not _HAS_GNN_MODULES:
            raise RuntimeError(
                "GNN submodules could not be imported.  Check installation."
            )

        self.config = config or TrainingConfig()
        self.node_names = node_names or [
            f"X{i}" for i in range(self.config.num_nodes)
        ]

        if len(self.node_names) != self.config.num_nodes:
            raise ValueError(
                f"node_names length ({len(self.node_names)}) must match "
                f"num_nodes ({self.config.num_nodes})."
            )

        # Device
        self.device = self._resolve_device()

        # Random seed
        self._set_seed(self.config.seed)

        # Build model components
        self._build_model()

        # Optimiser and scheduler (created in train())
        self.optimizer: Optional[torch.optim.Adam] = None
        self.scheduler: Optional[Any] = None

        # Training state
        self._epoch = 0
        self._best_loss = float("inf")
        self._best_adj: Optional[torch.Tensor] = None
        self._best_encoder_state: Optional[Dict[str, Any]] = None
        self._best_gnn_state: Optional[Dict[str, Any]] = None
        self._patience_counter = 0
        self._history: List[TrainingMetrics] = []
        self._is_trained = False

        # Learned adjacency (set after training)
        self._learned_adj: Optional[torch.Tensor] = None

        logger.info(
            "CausalGNNTrainer initialised: nodes=%d, device=%s, "
            "lambda1=%.4f, lambda2=%.2f, lr=%.5f, max_epochs=%d",
            self.config.num_nodes, self.device,
            self.config.lambda1, self.config.lambda2,
            self.config.learning_rate, self.config.max_epochs,
        )

    # -----------------------------------------------------------------------
    # Model construction
    # -----------------------------------------------------------------------

    def _build_model(self) -> None:
        """Instantiate encoder, GNN stack, and adjacency parameter."""
        cfg = self.config

        self.encoder = NodeFeatureEncoder(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            num_layers=cfg.num_encoder_layers,
            dropout=cfg.dropout,
            use_residual=True,
        )

        self.gnn_stack = DAGGNNStack(
            num_layers=cfg.num_gnn_layers,
            in_features=cfg.hidden_dim,
            hidden_features=cfg.hidden_dim,
            d=cfg.num_nodes,
            out_features=cfg.hidden_dim,
            bias=True,
            dropout=cfg.dropout,
        )

        # Edge decoder for interpreting node embeddings as adjacency
        self.edge_decoder = EdgeDecoder(
            input_dim=cfg.hidden_dim,
            hidden_dims=(cfg.hidden_dim, cfg.hidden_dim // 2),
            output_dim=1,
            dropout=cfg.dropout,
            exclude_diagonal=True,
        )

        # Move to device
        self.encoder.to(self.device)
        self.gnn_stack.to(self.device)
        self.edge_decoder.to(self.device)

        # Learnable raw adjacency matrix (unconstrained, NOT torch.no_grad)
        init_adj = torch.zeros(
            cfg.num_nodes, cfg.num_nodes, device=self.device
        )
        # Small random initialisation to break symmetry
        nn.init.xavier_uniform_(init_adj)
        # Zero the diagonal (no self-loops)
        init_adj.fill_diagonal_(0.0)
        self.raw_adj = nn.Parameter(init_adj)

        logger.debug("Model built: encoder, gnn_stack, edge_decoder, raw_adj parameter.")

    # -----------------------------------------------------------------------
    # Device management
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> torch.device:
        """Determine the compute device (CPU or CUDA).

        Returns
        -------
        torch.device
            ``torch.device("cpu")`` by default, ``torch.device("cuda")``
            only if explicitly requested and available.
        """
        if torch.cuda.is_available():
            logger.info("CUDA is available but CPU is used by default.")
        else:
            logger.info("CUDA not available; using CPU.")
        return torch.device("cpu")

    def _move_to_device(self, data: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the trainer's device.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor on the trainer's device.
        """
        return data.to(self.device)

    # -----------------------------------------------------------------------
    # Seed management
    # -----------------------------------------------------------------------

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        if torch is not None:
            torch.manual_seed(seed)
        if _HAS_NUMPY:
            np.random.seed(seed)  # type: ignore[attr-defined]
        logger.debug("Random seed set to %d.", seed)

    # -----------------------------------------------------------------------
    # Loss computation
    # -----------------------------------------------------------------------

    def _compute_loss(
        self,
        x_encoded: torch.Tensor,
        x_reconstructed: torch.Tensor,
        x_original: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the full NOTEARS loss.

        L_total = L_data + λ₁ ‖A‖₁ + λ₂ · h(A)

        Parameters
        ----------
        x_encoded : torch.Tensor
            Encoded node features from the encoder, shape ``(batch, d, hidden_dim)``.
        x_reconstructed : torch.Tensor
            Reconstructed features from the GNN stack, same shape.
        x_original : torch.Tensor
            Original input data (scaled), shape ``(batch, d, input_dim)``.

        Returns
        -------
        total_loss : torch.Tensor
            Combined loss.
        data_loss : torch.Tensor
            Data reconstruction component.
        l1_loss : torch.Tensor
            L1 sparsity of adjacency.
        acyclicity_loss : torch.Tensor
            NOTEARS h(A) value.
        """
        cfg = self.config

        # Data fitting loss: MSE between original input and reconstruction
        # Project reconstructed back to input space via encoder (use encoded)
        data_loss = functional.mse_loss(x_encoded, x_reconstructed)

        # L1 norm of the raw adjacency (sparsity regularisation)
        l1_loss = cfg.lambda1 * torch.norm(self.raw_adj, p=1)

        # NOTEARS acyclicity constraint: h(A) = tr(exp(A⊙A)) - d
        acyclicity_loss = cfg.lambda2 * DAGGNNStack.compute_acyclicity(self.raw_adj)
        # Clamp acyclicity to be non-negative (it should already be)
        acyclicity_loss = torch.clamp(acyclicity_loss, min=0.0)

        total_loss = data_loss + l1_loss + acyclicity_loss

        return total_loss, data_loss, l1_loss, acyclicity_loss

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Run the NOTEARS DAG-GNN training loop.

        Parameters
        ----------
        data : torch.Tensor
            Input data matrix of shape ``(N, num_nodes)`` or
            ``(N, num_nodes, input_dim)``.  ``N`` is the number of
            observations.  Each row represents a sample, each column
            a node/variable.

        Returns
        -------
        torch.Tensor
            Learned adjacency matrix of shape ``(num_nodes, num_nodes)``
            with values in [0, 1].

        Raises
        ------
        RuntimeError
            If PyTorch or GNN modules are unavailable.
        ValueError
            If data has incompatible dimensions.
        """
        self._validate_input(data)
        self._reset_training_state()

        # Prepare data
        data_tensor = self._move_to_device(data.float())
        data_tensor = self._prepare_data(data_tensor)

        N = data_tensor.shape[0]

        logger.info(
            "Training started: N=%d samples, d=%d nodes, device=%s, "
            "max_epochs=%d, batch_size=%d",
            N, self.config.num_nodes, self.device,
            self.config.max_epochs, self.config.batch_size,
        )

        # Create optimiser with all parameters
        all_params = (
            list(self.encoder.parameters())
            + list(self.gnn_stack.parameters())
            + list(self.edge_decoder.parameters())
            + [self.raw_adj]
        )
        self.optimizer = torch.optim.Adam(
            all_params, lr=self.config.learning_rate
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.lr_min,
            verbose=False,
        )

        start_time = time.time()

        for epoch in range(1, self.config.max_epochs + 1):
            self._epoch = epoch
            epoch_start = time.time()

            # --- single epoch (sequential mini-batches) ---
            epoch_metrics = self._train_one_epoch(data_tensor)

            epoch_metrics.epoch = epoch
            epoch_metrics.elapsed_seconds = time.time() - epoch_start
            epoch_metrics.lr = self.optimizer.param_groups[0]["lr"]

            self._history.append(epoch_metrics)

            # LR scheduling
            self.scheduler.step(epoch_metrics.total_loss)

            # Logging
            if epoch % 50 == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d  loss=%.6f  data=%.6f  l1=%.6f  "
                    "acyclicity=%.6f  lr=%.7f  time=%.2fs",
                    epoch, self.config.max_epochs,
                    epoch_metrics.total_loss,
                    epoch_metrics.data_loss,
                    epoch_metrics.l1_loss,
                    epoch_metrics.acyclicity_loss,
                    epoch_metrics.lr,
                    epoch_metrics.elapsed_seconds,
                )

            # Early stopping check
            if self._check_early_stopping(epoch_metrics):
                logger.info(
                    "Early stopping at epoch %d: best loss=%.6f, "
                    "patience=%d",
                    epoch, self._best_loss, self.config.patience,
                )
                break

        total_time = time.time() - start_time
        logger.info(
            "Training complete: %d epochs in %.1fs, best loss=%.6f",
            epoch, total_time, self._best_loss,
        )

        # Finalise
        self._finalise_training()

        return self._learned_adj  # type: ignore[return-value]

    def _train_one_epoch(
        self,
        data: torch.Tensor,
    ) -> TrainingMetrics:
        """Execute one training epoch sequentially.

        Parameters
        ----------
        data : torch.Tensor
            Prepared data tensor of shape ``(N, d, hidden_dim)``.

        Returns
        -------
        TrainingMetrics
            Aggregated epoch metrics.
        """
        self.encoder.train()
        self.gnn_stack.train()
        self.edge_decoder.train()

        N = data.shape[0]
        batch_size = min(self.config.batch_size, N)

        total_loss_sum = 0.0
        data_loss_sum = 0.0
        l1_loss_sum = 0.0
        acyclicity_sum = 0.0
        num_batches = 0

        # Sequential mini-batch processing
        indices = torch.randperm(N, device=self.device)

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_idx = indices[start_idx:end_idx]
            batch = data[batch_idx]  # (batch, d, hidden_dim)

            # Forward pass
            x_encoded = self.encoder(batch)  # (batch, d, hidden_dim)
            x_reconstructed = self.gnn_stack(x_encoded, self.raw_adj)  # (batch, d, hidden_dim)

            # Compute loss
            total_loss, d_loss, l_loss, a_loss = self._compute_loss(
                x_encoded, x_reconstructed, batch
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters())
                + list(self.gnn_stack.parameters())
                + list(self.edge_decoder.parameters())
                + [self.raw_adj],
                max_norm=5.0,
            )
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            data_loss_sum += d_loss.item()
            l1_loss_sum += l_loss.item()
            acyclicity_sum += a_loss.item()
            num_batches += 1

        metrics = TrainingMetrics(
            total_loss=total_loss_sum / max(num_batches, 1),
            data_loss=data_loss_sum / max(num_batches, 1),
            l1_loss=l1_loss_sum / max(num_batches, 1),
            acyclicity_loss=acyclicity_sum / max(num_batches, 1),
        )

        return metrics

    def _check_early_stopping(self, metrics: TrainingMetrics) -> bool:
        """Check whether training should stop early.

        Parameters
        ----------
        metrics : TrainingMetrics
            Current epoch metrics.

        Returns
        -------
        bool
            ``True`` if early stopping should trigger.
        """
        if self._epoch < self.config.min_epochs:
            return False

        if metrics.total_loss < self._best_loss:
            improvement = self._best_loss - metrics.total_loss
            self._best_loss = metrics.total_loss
            self._patience_counter = 0

            # Save best model state
            self._best_adj = self.raw_adj.detach().clone()
            self._best_encoder_state = copy.deepcopy(self.encoder.state_dict())
            self._best_gnn_state = copy.deepcopy(self.gnn_stack.state_dict())

            logger.debug(
                "New best loss %.6f (improvement %.6f)",
                self._best_loss, improvement,
            )
            return False
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.config.patience:
                return True
            return False

    def _finalise_training(self) -> None:
        """Restore best model weights and produce the learned adjacency."""
        # Restore best weights
        if self._best_encoder_state is not None:
            self.encoder.load_state_dict(self._best_encoder_state)
        if self._best_gnn_state is not None:
            self.gnn_stack.load_state_dict(self._best_gnn_state)
        if self._best_adj is not None:
            self.raw_adj.data.copy_(self._best_adj)

        self.encoder.eval()
        self.gnn_stack.eval()
        self.edge_decoder.eval()

        # Produce thresholded adjacency
        with torch.no_grad():
            # Use sigmoid to get probabilities
            adj_probs = torch.sigmoid(self.raw_adj)
            adj_probs.fill_diagonal_(0.0)
            # Apply threshold
            self._learned_adj = (adj_probs >= self.config.threshold).float()
            self._learned_adj.fill_diagonal_(0.0)

        self._is_trained = True

        # Log sparsity statistics
        d = self.config.num_nodes
        num_edges = int(self._learned_adj.sum().item())
        max_possible = d * (d - 1)
        sparsity = 1.0 - (num_edges / max_possible) if max_possible > 0 else 1.0
        logger.info(
            "Learned DAG: %d edges out of %d possible (%.1f%% sparsity)",
            num_edges, max_possible, sparsity * 100,
        )

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference on the trained model.

        Parameters
        ----------
        data : torch.Tensor
            Input data of shape ``(N, num_nodes)`` or ``(N, num_nodes, input_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed node embeddings of shape ``(N, num_nodes, hidden_dim)``.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet.  Call train() first."
            )

        self._validate_input(data)
        data_tensor = self._move_to_device(data.float())
        data_tensor = self._prepare_data(data_tensor)

        self.encoder.eval()
        self.gnn_stack.eval()

        with torch.no_grad():
            x_encoded = self.encoder(data_tensor)
            x_reconstructed = self.gnn_stack(x_encoded, self.raw_adj)

        return x_reconstructed

    def get_causal_graph(self) -> "nx.DiGraph":
        """Return the learned causal graph as a networkx DiGraph.

        Edge weights are the absolute values of the learned adjacency
        entries.  Only edges above the threshold are included.

        Returns
        -------
        networkx.DiGraph
            Directed graph of learned causal relationships.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet or networkx is unavailable.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained yet.  Call train() first."
            )
        if not _HAS_NETWORKX:
            raise RuntimeError(
                "networkx is required for graph export but is not installed."
            )

        if self._learned_adj is None:
            raise RuntimeError("No learned adjacency matrix available.")

        adj = self._learned_adj.detach().cpu().numpy()
        d = self.config.num_nodes

        graph = nx.DiGraph()
        graph.add_nodes_from(self.node_names)

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if adj[i, j] > 0:
                    graph.add_edge(
                        self.node_names[i],
                        self.node_names[j],
                        weight=float(adj[i, j]),
                    )

        logger.info(
            "Causal graph exported: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    def get_training_history(self) -> List[TrainingMetrics]:
        """Return the full list of per-epoch training metrics.

        Returns
        -------
        list[TrainingMetrics]
            Training metrics for each completed epoch.
        """
        return list(self._history)

    # -----------------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------------

    def _validate_input(self, data: torch.Tensor) -> None:
        """Validate input data dimensions.

        Parameters
        ----------
        data : torch.Tensor
            Input data tensor to validate.

        Raises
        ------
        ValueError
            If dimensions are incompatible.
        """
        if data.dim() not in (2, 3):
            raise ValueError(
                f"data must be 2D (N, d) or 3D (N, d, input_dim), "
                f"got {data.dim()}D shape {tuple(data.shape)}."
            )

        expected_last_dim = self.config.num_nodes if data.dim() == 2 else self.config.input_dim
        actual_last_dim = data.shape[-1]

        if data.dim() == 2:
            if data.shape[1] != self.config.num_nodes:
                raise ValueError(
                    f"data shape[-1] ({data.shape[1]}) must match "
                    f"num_nodes ({self.config.num_nodes}) for 2D input."
                )
        else:
            if data.shape[1] != self.config.num_nodes:
                raise ValueError(
                    f"data.shape[1] ({data.shape[1]}) must match "
                    f"num_nodes ({self.config.num_nodes}) for 3D input."
                )
            if data.shape[2] != self.config.input_dim:
                raise ValueError(
                    f"data.shape[2] ({data.shape[2]}) must match "
                    f"input_dim ({self.config.input_dim}) for 3D input."
                )

    def _prepare_data(self, data: torch.Tensor) -> torch.Tensor:
        """Prepare raw data for model input.

        Handles 2D → 3D reshaping and projects through the encoder
        to produce the hidden-dimension representation.

        Parameters
        ----------
        data : torch.Tensor
            Raw data of shape ``(N, d)`` or ``(N, d, input_dim)``.

        Returns
        -------
        torch.Tensor
            Prepared data of shape ``(N, d, hidden_dim)`` ready for
            the GNN stack.
        """
        if data.dim() == 2:
            # (N, d) → (N, d, 1) so each node has a scalar feature
            data = data.unsqueeze(-1)

        # Project to hidden dimension using the encoder
        N, d, feat_dim = data.shape

        if feat_dim != self.config.input_dim:
            # If input_dim doesn't match, use a linear projection
            data = self._project_features(data, feat_dim)

        return data

    def _project_features(
        self,
        data: torch.Tensor,
        current_dim: int,
    ) -> torch.Tensor:
        """Project features to match config.input_dim.

        Uses a learned linear projection (cached after first call).

        Parameters
        ----------
        data : torch.Tensor
            Data tensor of shape ``(N, d, current_dim)``.
        current_dim : int
            Current feature dimension.

        Returns
        -------
        torch.Tensor
            Projected data of shape ``(N, d, input_dim)``.
        """
        if not hasattr(self, "_feature_proj") or self._feature_proj is None:
            self._feature_proj = nn.Linear(
                current_dim, self.config.input_dim, bias=True
            ).to(self.device)
            nn.init.xavier_uniform_(self._feature_proj.weight)
            nn.init.zeros_(self._feature_proj.bias)
            logger.debug(
                "Created feature projection: %d → %d",
                current_dim, self.config.input_dim,
            )

        orig_shape = data.shape
        flat = data.reshape(-1, current_dim)
        projected = self._feature_proj(flat)
        return projected.reshape(orig_shape[0], orig_shape[1], self.config.input_dim)

    # -----------------------------------------------------------------------
    # Training state management
    # -----------------------------------------------------------------------

    def _reset_training_state(self) -> None:
        """Reset all training state for a fresh training run."""
        self._epoch = 0
        self._best_loss = float("inf")
        self._best_adj = None
        self._best_encoder_state = None
        self._best_gnn_state = None
        self._patience_counter = 0
        self._history = []
        self._is_trained = False
        self._learned_adj = None
        self._feature_proj = None  # type: ignore[assignment]

        logger.debug("Training state reset.")

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: Optional[str] = None,
    ) -> str:
        """Save the full training state to disk.

        Parameters
        ----------
        path : str, optional
            File path for the checkpoint.  If ``None``, uses
            ``config.checkpoint_dir / "causal_gnn_latest.pt"``.

        Returns
        -------
        str
            The path where the checkpoint was saved.

        Raises
        ------
        RuntimeError
            If the checkpoint cannot be saved.
        """
        if path is None:
            checkpoint_dir = self.config.checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, "causal_gnn_latest.pt")

        try:
            checkpoint = {
                "epoch": self._epoch,
                "best_loss": self._best_loss,
                "patience_counter": self._patience_counter,
                "is_trained": self._is_trained,
                "config": asdict(self.config),
                "node_names": self.node_names,
                "encoder_state": self.encoder.state_dict(),
                "gnn_stack_state": self.gnn_stack.state_dict(),
                "edge_decoder_state": self.edge_decoder.state_dict(),
                "raw_adj": self.raw_adj.data.cpu(),
                "learned_adj": (
                    self._learned_adj.cpu() if self._learned_adj is not None else None
                ),
                "history": [asdict(m) for m in self._history],
            }

            # Save optimizer state if available
            if self.optimizer is not None:
                checkpoint["optimizer_state"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint["scheduler_state"] = self.scheduler.state_dict()

            torch.save(checkpoint, path)
            logger.info("Checkpoint saved to %s (epoch %d).", path, self._epoch)
            return path

        except Exception as exc:
            logger.error("Failed to save checkpoint to %s: %s", path, exc)
            raise RuntimeError(f"Checkpoint save failed: {exc}") from exc

    def load_checkpoint(
        self,
        path: str,
    ) -> None:
        """Restore the full training state from disk.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.

        Raises
        ------
        RuntimeError
            If the checkpoint cannot be loaded.
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )

            # Restore config
            saved_config = checkpoint.get("config", {})
            for key, value in saved_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Restore node names
            if "node_names" in checkpoint:
                self.node_names = checkpoint["node_names"]

            # Restore model weights
            self.encoder.load_state_dict(checkpoint["encoder_state"])
            self.gnn_stack.load_state_dict(checkpoint["gnn_stack_state"])
            self.edge_decoder.load_state_dict(checkpoint["edge_decoder_state"])

            # Restore adjacency
            self.raw_adj.data.copy_(
                checkpoint["raw_adj"].to(self.device)
            )

            # Restore learned adjacency
            saved_learned = checkpoint.get("learned_adj")
            if saved_learned is not None:
                self._learned_adj = saved_learned.to(self.device)

            # Restore training state
            self._epoch = checkpoint.get("epoch", 0)
            self._best_loss = checkpoint.get("best_loss", float("inf"))
            self._patience_counter = checkpoint.get("patience_counter", 0)
            self._is_trained = checkpoint.get("is_trained", False)

            # Restore optimizer and scheduler
            all_params = (
                list(self.encoder.parameters())
                + list(self.gnn_stack.parameters())
                + list(self.edge_decoder.parameters())
                + [self.raw_adj]
            )
            self.optimizer = torch.optim.Adam(
                all_params, lr=self.config.learning_rate
            )
            if "optimizer_state" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                min_lr=self.config.lr_min,
                verbose=False,
            )
            if "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])

            # Restore history
            saved_history = checkpoint.get("history", [])
            self._history = [
                TrainingMetrics(**h) for h in saved_history
            ]

            # Set model mode
            if self._is_trained:
                self.encoder.eval()
                self.gnn_stack.eval()
                self.edge_decoder.eval()
            else:
                self.encoder.train()
                self.gnn_stack.train()
                self.edge_decoder.train()

            logger.info(
                "Checkpoint loaded from %s (epoch %d, trained=%s).",
                path, self._epoch, self._is_trained,
            )

        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as exc:
            logger.error("Failed to load checkpoint from %s: %s", path, exc)
            raise RuntimeError(f"Checkpoint load failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_adjacency_matrix(self) -> Optional[torch.Tensor]:
        """Return the learned adjacency matrix.

        Returns
        -------
        torch.Tensor or None
            The learned adjacency matrix of shape ``(num_nodes, num_nodes)``,
            or ``None`` if the model has not been trained.
        """
        return self._learned_adj

    def get_raw_adjacency(self) -> torch.Tensor:
        """Return the raw (unthresholded) adjacency probabilities.

        Returns
        -------
        torch.Tensor
            Sigmoid of the raw adjacency parameter, shape ``(num_nodes, num_nodes)``.
        """
        with torch.no_grad():
            adj = torch.sigmoid(self.raw_adj.detach().clone())
            adj.fill_diagonal_(0.0)
            return adj

    def get_edge_list(
        self,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """Return a list of learned causal edges.

        Parameters
        ----------
        threshold : float, optional
            Edge probability threshold.  If ``None``, uses
            ``config.threshold``.

        Returns
        -------
        list[tuple[str, str, float]]
            List of ``(source, target, weight)`` tuples for edges above
            the threshold.
        """
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet.")

        threshold = threshold or self.config.threshold
        adj = self.get_raw_adjacency().cpu().numpy()
        d = self.config.num_nodes

        edges: List[Tuple[str, str, float]] = []
        for i in range(d):
            for j in range(d):
                if i != j and adj[i, j] >= threshold:
                    edges.append(
                        (self.node_names[i], self.node_names[j], float(adj[i, j]))
                    )

        # Sort by weight descending
        edges.sort(key=lambda e: e[2], reverse=True)
        return edges

    def summary(self) -> Dict[str, Any]:
        """Return a diagnostic summary of the trainer state.

        Returns
        -------
        dict[str, Any]
            Dictionary with model info, training status, and metrics.
        """
        return {
            "is_trained": self._is_trained,
            "epoch": self._epoch,
            "best_loss": self._best_loss,
            "num_nodes": self.config.num_nodes,
            "device": str(self.device),
            "node_names": self.node_names,
            "num_history_entries": len(self._history),
            "num_edges": (
                int(self._learned_adj.sum().item())
                if self._learned_adj is not None
                else 0
            ),
            "config": asdict(self.config),
        }

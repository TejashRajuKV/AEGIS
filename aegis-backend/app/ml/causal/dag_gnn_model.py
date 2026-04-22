"""
DAG-GNN Model — end-to-end graph neural network for causal structure learning.

This module implements :class:`DAGGNNModel`, a ``torch.nn.Module`` that composes
a :class:`NodeFeatureEncoder`, multiple :class:`DAGGNNLayer` message-passing
layers, and an :class:`EdgeDecoder` into a single model for learning directed
acyclic graphs from observational tabular data.

Architecture
------------
::

    Input x (N, d, input_dim)
      → NodeFeatureEncoder → (N, d, hidden_dim)
      → [DAGGNNLayer × n_layers] with adjacency A → (N, d, hidden_dim)
      → EdgeDecoder → adjacency prediction (d, d)

Loss
----
The model is trained with the NOTEARS loss:

    L = L_recon + λ₁ ‖A‖₁ + λ₂ · h(A)

where ``h(A) = tr(exp(A⊙A)) - d`` is the acyclicity constraint computed via
eigendecomposition.

References
----------
Yu, Y., Chen, J., Gao, T., & Chen, M. (2019). DAG-GNN: DAG Structure
Learning with Graph Neural Networks. *ICML*.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports – PyTorch may not be installed in all environments
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    _HAS_TORCH = False
    logger.warning("PyTorch is not installed. DAGGNNModel is unavailable.")

try:
    from app.ml.gnn.dag_gnn_layers import DAGGNNLayer
    _HAS_GNN_LAYERS = True
except ImportError as exc:
    DAGGNNLayer = None  # type: ignore[assignment, misc]
    _HAS_GNN_LAYERS = False
    logger.warning("DAGGNNLayer import failed: %s", exc)

try:
    from app.ml.gnn.node_encoder import NodeFeatureEncoder
    _HAS_ENCODER = True
except ImportError as exc:
    NodeFeatureEncoder = None  # type: ignore[assignment, misc]
    _HAS_ENCODER = False
    logger.warning("NodeFeatureEncoder import failed: %s", exc)

try:
    from app.ml.gnn.edge_decoder import EdgeDecoder
    _HAS_DECODER = True
except ImportError as exc:
    EdgeDecoder = None  # type: ignore[assignment, misc]
    _HAS_DECODER = False
    logger.warning("EdgeDecoder import failed: %s", exc)


__all__ = ["DAGGNNModel"]


class DAGGNNModel(nn.Module):
    """End-to-end DAG-GNN model for causal structure learning.

    Composes a :class:`NodeFeatureEncoder`, a stack of :class:`DAGGNNLayer`
    message-passing layers, and an :class:`EdgeDecoder` into a single
    ``nn.Module`` that learns the adjacency matrix of a causal DAG.

    Parameters
    ----------
    input_dim : int
        Dimensionality of raw input features per node.
    hidden_dim : int, optional
        Hidden dimension used throughout encoder, GNN layers, and decoder.
        Default ``64``.
    latent_dim : int, optional
        Dimension of the latent bottleneck between encoder and decoder.
        Default ``16``.
    n_layers : int, optional
        Number of DAG-GNN message-passing layers.  Default ``2``.
    dropout : float, optional
        Dropout probability applied between layers.  Default ``0.1``.

    Attributes
    ----------
    node_encoder : NodeFeatureEncoder
        Multi-layer MLP that projects raw features to ``hidden_dim``.
    gnn_layers : torch.nn.ModuleList
        List of :class:`DAGGNNLayer` instances.
    edge_decoder : EdgeDecoder
        MLP decoder that predicts edge probabilities from node embeddings.
    raw_adj : torch.nn.Parameter
        Learnable ``d × d`` raw adjacency matrix (unconstrained).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for DAGGNNModel but is not installed."
            )
        if not _HAS_GNN_LAYERS:
            raise RuntimeError("DAGGNNLayer could not be imported.")
        if not _HAS_ENCODER:
            raise RuntimeError("NodeFeatureEncoder could not be imported.")
        if not _HAS_DECODER:
            raise RuntimeError("EdgeDecoder could not be imported.")

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.d = input_dim  # number of nodes = number of input features

        # --- Node encoder ---
        self.node_encoder = NodeFeatureEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=3,
            dropout=dropout,
            use_residual=True,
        )

        # --- GNN layers ---
        layers: List[nn.Module] = []
        for idx in range(n_layers):
            if idx == 0:
                layer_in = hidden_dim
            else:
                layer_in = hidden_dim

            if idx == n_layers - 1:
                layer_out = latent_dim
            else:
                layer_out = hidden_dim

            layers.append(
                DAGGNNLayer(
                    in_features=layer_in,
                    out_features=layer_out,
                    d=self.d,
                    bias=True,
                )
            )
        self.gnn_layers = nn.ModuleList(layers)

        # --- Edge decoder ---
        decoder_input_dim = latent_dim if n_layers > 0 else hidden_dim
        self.edge_decoder = EdgeDecoder(
            input_dim=decoder_input_dim,
            hidden_dims=(64, 32),
            output_dim=1,
            dropout=dropout,
            exclude_diagonal=True,
        )

        # --- Learnable raw adjacency matrix ---
        init_adj = torch.zeros(self.d, self.d)
        nn.init.xavier_uniform_(init_adj)
        init_adj.fill_diagonal_(0.0)
        self.raw_adj = nn.Parameter(init_adj)

        # --- Dropout layer ---
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        logger.info(
            "DAGGNNModel initialised: input_dim=%d, hidden_dim=%d, "
            "latent_dim=%d, n_layers=%d, dropout=%.2f, d=%d",
            input_dim, hidden_dim, latent_dim, n_layers, dropout, self.d,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: "torch.Tensor",
        adj: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Run the full forward pass through encoder, GNN layers, and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape ``(N, d)`` or ``(N, d, input_dim)``.
        adj : torch.Tensor, optional
            Explicit adjacency matrix of shape ``(d, d)``.  If ``None``,
            the learnable ``self.raw_adj`` is used.

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed node embeddings, shape ``(N, d, latent_dim)``.
        mu : torch.Tensor
            Latent mean (placeholder, set to zeros), shape ``(N, d, latent_dim)``.
        log_var : torch.Tensor
            Latent log-variance (placeholder, set to zeros), shape
            ``(N, d, latent_dim)``.
        adj_pred : torch.Tensor
            Predicted adjacency probabilities, shape ``(d, d)``.
        """
        if adj is None:
            adj = self.raw_adj

        N = x.shape[0]

        # Handle 2D input (N, d) → (N, d, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Encode: (N, d, input_dim) → (N, d, hidden_dim)
        encoded = self.node_encoder(x)

        # GNN message passing
        h = encoded
        for idx, layer in enumerate(self.gnn_layers):
            h = layer(h, adj)
            if idx < len(self.gnn_layers) - 1:
                h = self.dropout_layer(h)

        reconstruction = h

        # Latent space placeholder (VAE-style outputs for compatibility)
        mu = torch.zeros_like(reconstruction)
        log_var = torch.zeros_like(reconstruction)

        # Predict adjacency from embeddings (use mean over batch)
        mean_embeddings = reconstruction.mean(dim=0)  # (d, latent_dim)
        adj_pred, _ = self.edge_decoder.reconstruct_adjacency(mean_embeddings)

        return reconstruction, mu, log_var, adj_pred

    # ------------------------------------------------------------------
    # Learned adjacency
    # ------------------------------------------------------------------

    def get_adjacency(self, x: "torch.Tensor") -> "torch.Tensor":
        """Compute and return the learned adjacency matrix for the given input.

        Runs the forward pass with ``torch.no_grad()`` and returns the
        sigmoid-thresholded adjacency matrix.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape ``(N, d)`` or ``(N, d, input_dim)``.

        Returns
        -------
        torch.Tensor
            Learned adjacency matrix of shape ``(d, d)`` with values in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            _, _, _, adj_pred = self.forward(x)
        return adj_pred

    # ------------------------------------------------------------------
    # Edge weights
    # ------------------------------------------------------------------

    def get_edge_weights(self) -> "torch.Tensor":
        """Return the sparse edge weight matrix.

        Applies sigmoid to the raw adjacency, zeroes the diagonal, and
        returns the result as a dense tensor.  Entries represent learned
        edge weights in [0, 1].

        Returns
        -------
        torch.Tensor
            Dense edge weight matrix of shape ``(d, d)``.
        """
        with torch.no_grad():
            weights = torch.sigmoid(self.raw_adj)
            weights.fill_diagonal_(0.0)
        return weights

    # ------------------------------------------------------------------
    # Acyclicity loss via eigendecomposition
    # ------------------------------------------------------------------

    def compute_acyclicity_loss(self, adj: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Compute the NOTEARS acyclicity constraint using eigendecomposition.

        Implements ``h(A) = tr(exp(A⊙A)) - d`` via eigenvalue computation
        for numerical stability:

            tr(exp(M)) = Σ exp(λ_i)

        where ``λ_i`` are the eigenvalues of ``M = A⊙A``.

        Parameters
        ----------
        adj : torch.Tensor, optional
            Adjacency matrix of shape ``(d, d)``.  If ``None``, uses
            ``self.raw_adj``.

        Returns
        -------
        torch.Tensor
            Scalar acyclicity violation.  Zero implies a DAG.
        """
        if adj is None:
            adj = self.raw_adj

        # Element-wise square: A ⊙ A
        aa = adj * adj

        # Eigendecomposition: tr(exp(M)) = Σ exp(λ_i)
        eig_vals = torch.linalg.eigvalsh(aa)
        trace_exp = torch.exp(eig_vals).sum()

        # h(A) = tr(exp(A⊙A)) - d
        d = adj.shape[0]
        h = trace_exp - d

        return h

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_edges(self, threshold: float = 0.3) -> int:
        """Count the number of edges above a given threshold.

        Parameters
        ----------
        threshold : float, optional
            Edge weight threshold.  Default ``0.3``.

        Returns
        -------
        int
            Number of edges with weight >= threshold.
        """
        weights = self.get_edge_weights()
        mask = (weights >= threshold).float()
        mask.fill_diagonal_(0.0)
        return int(mask.sum().item())

    def sparsity(self, threshold: float = 0.3) -> float:
        """Compute graph sparsity (fraction of absent edges).

        Parameters
        ----------
        threshold : float, optional
            Edge weight threshold.  Default ``0.3``.

        Returns
        -------
        float
            Sparsity ratio in [0, 1].  1.0 means no edges.
        """
        d = self.d
        max_possible = d * (d - 1)
        if max_possible == 0:
            return 1.0
        return 1.0 - (self.get_num_edges(threshold) / max_possible)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"latent_dim={self.latent_dim}, n_layers={self.n_layers}, "
            f"dropout={self.dropout}, d={self.d}"
        )

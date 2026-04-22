"""
Edge decoder for predicting edge existence between node pairs.

The :class:`EdgeDecoder` takes paired node embeddings and predicts the
probability that a directed edge exists from one node to another.  This
is a key component of the DAG-GNN pipeline: after the encoder produces
node representations, the edge decoder learns the adjacency structure.

Architecture
------------
For each ordered pair ``(i, j)`` the decoder concatenates embeddings
``[h_i || h_j]`` and passes the result through an MLP with sigmoid output:

    p(i → j) = sigmoid( MLP( [h_i || h_j] ) )

The full adjacency matrix is reconstructed by evaluating all ``d × d``
ordered pairs (minus diagonal self-loops if desired).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional

logger = logging.getLogger(__name__)

__all__ = ["EdgeDecoder"]


class EdgeDecoder(nn.Module):
    """MLP-based edge decoder for learning graph adjacency structure.

    Takes concatenated pairs of node embeddings and outputs the probability
    of a directed edge existing between them.

    Parameters
    ----------
    input_dim : int
        Dimension of a single node embedding.  The concatenated pair has
        dimension ``2 * input_dim``.
    hidden_dims : tuple[int, ...], optional
        Hidden layer sizes for the internal MLP.  Default ``(64, 32)``.
    output_dim : int, optional
        Output dimension before sigmoid.  Must be 1 for binary edge
        prediction.  Default 1.
    dropout : float, optional
        Dropout probability between hidden layers.  Default 0.1.
    exclude_diagonal : bool, optional
        If ``True``, diagonal entries (self-loops) are masked to zero.
        Default ``True``.

    Attributes
    ----------
    mlp : torch.nn.Sequential
        The internal MLP network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        output_dim: int = 1,
        dropout: float = 0.1,
        exclude_diagonal: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if output_dim != 1:
            raise ValueError(f"output_dim must be 1 for edge probability, got {output_dim}.")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.exclude_diagonal = exclude_diagonal

        # Build MLP layers
        layers: list[nn.Module] = []
        in_dim = 2 * input_dim  # concatenation of two node embeddings

        all_dims = [in_dim] + list(hidden_dims) + [output_dim]
        for idx in range(len(all_dims) - 1):
            layer_in = all_dims[idx]
            layer_out = all_dims[idx + 1]
            linear = nn.Linear(in_features=layer_in, out_features=layer_out)
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            layers.append(linear)

            # Hidden layers get activation + dropout
            if idx < len(all_dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*layers)

        logger.debug(
            "EdgeDecoder initialised: input_dim=%d, hidden=%s, dropout=%.2f, "
            "exclude_diag=%s",
            input_dim, list(hidden_dims), dropout, exclude_diagonal,
        )

    # ----- forward (pairwise) -----------------------------------------------

    def forward(
        self,
        h_source: torch.Tensor,
        h_target: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge probabilities for specific node pairs.

        Parameters
        ----------
        h_source : torch.Tensor
            Source node embeddings of shape ``(..., input_dim)``.
        h_target : torch.Tensor
            Target node embeddings of shape ``(..., input_dim)``.

        Returns
        -------
        torch.Tensor
            Edge probability logits of shape ``(..., 1)``.  Apply sigmoid
            externally to obtain probabilities.
        """
        # Concatenate source and target embeddings
        paired = torch.cat([h_source, h_target], dim=-1)
        logits = self.mlp(paired)
        return logits

    # ----- full adjacency reconstruction ------------------------------------

    def reconstruct_adjacency(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the full adjacency matrix from node embeddings.

        Evaluates the decoder for all ``d × d`` ordered pairs of nodes.

        Parameters
        ----------
        embeddings : torch.Tensor
            Node embedding matrix of shape ``(d, emb_dim)`` or
            ``(batch, d, emb_dim)``.

        Returns
        -------
        adjacency_matrix : torch.Tensor
            Binary (thresholded) adjacency matrix of shape ``(d, d)`` or
            ``(batch, d, d)``.  Values are 0 or 1.
        edge_probabilities : torch.Tensor
            Soft edge probability matrix of shape ``(d, d)`` or
            ``(batch, d, d)``.  Values in [0, 1].
        """
        d = embeddings.shape[-2]

        squeeze_batch = False
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            squeeze_batch = True

        batch_size = embeddings.shape[0]

        # Create all (source, target) index pairs
        src_idx = torch.arange(d, device=embeddings.device)
        tgt_idx = torch.arange(d, device=embeddings.device)

        # Source: (d, 1, emb_dim), Target: (1, d, emb_dim)
        # Broadcasting gives (d, d, emb_dim)
        h_source = embeddings[:, src_idx, :]  # (batch, d, emb_dim)
        h_target = embeddings[:, tgt_idx, :].transpose(1, 2)  # (batch, d, emb_dim)

        # Actually, we need all d*d pairs. Let's use broadcasting properly.
        # h_source_expanded: (batch, d, 1, emb_dim)
        # h_target_expanded: (batch, 1, d, emb_dim)
        h_source_exp = embeddings.unsqueeze(2)  # (batch, d, 1, emb_dim)
        h_target_exp = embeddings.unsqueeze(1)  # (batch, 1, d, emb_dim)

        # Expand to (batch, d, d, emb_dim) for each pair
        d_src = embeddings.shape[1]
        d_tgt = embeddings.shape[2]
        h_s = h_source_exp.expand(batch_size, d_src, d_tgt, -1)
        h_t = h_target_exp.expand(batch_size, d_src, d_tgt, -1)

        # Concatenate and predict
        # Shape: (batch, d, d, 2*emb_dim)
        paired = torch.cat([h_s, h_t], dim=-1)

        # Reshape for MLP: (batch*d*d, 2*emb_dim)
        original_shape = paired.shape
        paired_flat = paired.reshape(-1, 2 * self.input_dim)

        logits_flat = self.mlp(paired_flat)  # (batch*d*d, 1)
        probs_flat = torch.sigmoid(logits_flat).squeeze(-1)  # (batch*d*d,)

        # Reshape back
        probs = probs_flat.reshape(batch_size, d_src, d_tgt)  # (batch, d, d)

        # Zero out diagonal if requested
        if self.exclude_diagonal:
            diag_mask = torch.eye(d, device=embeddings.device, dtype=embeddings.dtype)
            probs = probs * (1.0 - diag_mask)

        # Threshold at 0.5 for binary adjacency
        adjacency = (probs >= 0.5).float()

        if self.exclude_diagonal:
            adjacency = adjacency * (1.0 - diag_mask)

        if squeeze_batch:
            probs = probs.squeeze(0)
            adjacency = adjacency.squeeze(0)

        return adjacency, probs

    # ----- loss helper ------------------------------------------------------

    def compute_edge_loss(
        self,
        embeddings: torch.Tensor,
        target_adj: Optional[torch.Tensor] = None,
        l1_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute edge reconstruction loss with optional L1 sparsity.

        Parameters
        ----------
        embeddings : torch.Tensor
            Node embeddings of shape ``(d, emb_dim)``.
        target_adj : torch.Tensor, optional
            Ground-truth adjacency matrix of shape ``(d, d)``.  If ``None``,
            only L1 sparsity is computed (unsupervised mode).
        l1_weight : float, optional
            Weight for L1 sparsity regularisation.  Default 0.01.

        Returns
        -------
        total_loss : torch.Tensor
            Combined loss (reconstruction + sparsity).
        recon_loss : torch.Tensor
            Reconstruction loss (0 if ``target_adj`` is ``None``).
        l1_loss : torch.Tensor
            L1 sparsity regularisation of edge probabilities.
        """
        _, probs = self.reconstruct_adjacency(embeddings)

        # L1 sparsity regularisation
        l1_loss = l1_weight * torch.norm(probs, p=1)

        if target_adj is not None:
            # Binary cross-entropy reconstruction loss
            recon_loss = functional.binary_cross_entropy(
                probs, target_adj, reduction="mean"
            )
            total_loss = recon_loss + l1_loss
        else:
            recon_loss = torch.tensor(0.0, device=embeddings.device)
            total_loss = l1_loss

        return total_loss, recon_loss, l1_loss

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dims={list(self.hidden_dims)}, "
            f"output_dim={self.output_dim}, "
            f"exclude_diagonal={self.exclude_diagonal}"
        )

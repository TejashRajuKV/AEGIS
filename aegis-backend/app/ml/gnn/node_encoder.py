"""
Node feature encoder for mapping raw tabular data to latent embeddings.

The :class:`NodeFeatureEncoder` is a multi-layer MLP with layer normalisation,
residual connections, and GELU activations.  It is designed to project raw
node features (e.g. one-hot encoded categorical columns, normalised continuous
columns) into a dense latent space suitable for downstream DAG-GNN processing.

Design notes
------------
* **Residual connections** are added between layers when the input and output
  dimensions match, improving gradient flow in deep networks.
* **Layer normalisation** is applied after each hidden layer for training
  stability, which is especially important for the small-batch / sequential
  setting on a 16 GB RAM machine.
* **Dropout** is supported between layers as a regulariser.
* All linear layers use Xavier/Glorot initialisation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["NodeFeatureEncoder"]


class _ResidualMLPBlock(nn.Module):
    """A single MLP block with optional residual connection and layer norm.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    use_residual : bool
        Whether to add a residual connection (requires ``in_dim == out_dim``).
    dropout : float
        Dropout probability (``0.0`` disables dropout).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_residual: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)

        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=out_dim)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Xavier init
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        # Optional projection for residual when dimensions differ
        if self.use_residual:
            self.residual_proj: Optional[nn.Linear] = None
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., in_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., out_dim)``.
        """
        residual = x
        out = self.linear(x)
        out = self.layer_norm(out)
        out = self.activation(out)
        out = self.dropout_layer(out)

        if self.use_residual:
            out = out + residual

        return out


class NodeFeatureEncoder(nn.Module):
    """Multi-layer MLP encoder for raw node features.

    Transforms raw tabular node features into dense latent representations
    suitable for graph neural network processing.  Supports configurable
    depth, hidden dimensions, and residual connections.

    Architecture
    ------------
    For ``num_layers = 3`` with ``input_dim=10``, ``hidden_dim=32``,
    ``output_dim=16``::

        input(10) → Linear(10→32) → LN → GELU → Dropout
                 → Linear(32→32) → LN → GELU → Dropout  [+ residual]
                 → Linear(32→16) → LN → GELU → Dropout
                 → output(16)

    Parameters
    ----------
    input_dim : int
        Dimensionality of raw input features.
    hidden_dim : int
        Hidden layer width.  All hidden layers share this width.
    output_dim : int
        Dimensionality of the output embedding.
    num_layers : int
        Total number of linear layers (including input and output layers).
        Must be >= 2.
    dropout : float, optional
        Dropout probability applied after each hidden activation. Default 0.1.
    use_residual : bool, optional
        Whether to add residual connections when dimensions allow. Default True.

    Attributes
    ----------
    blocks : torch.nn.ModuleList
        List of :class:`_ResidualMLPBlock` instances.
    output_norm : torch.nn.LayerNorm
        Final layer normalisation applied to the output embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError(
                f"num_layers must be >= 2, got {num_layers}. "
                "At least one hidden layer is required."
            )
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError(
                f"All dimensions must be positive: input_dim={input_dim}, "
                f"hidden_dim={hidden_dim}, output_dim={output_dim}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Build layer dimension schedule
        dims: List[int] = [input_dim]
        for i in range(1, num_layers - 1):
            dims.append(hidden_dim)
        dims.append(output_dim)

        blocks: List[nn.Module] = []
        for i in range(num_layers):
            in_d = dims[i]
            out_d = dims[i + 1] if i + 1 < len(dims) else dims[-1]

            # Residual only on intermediate layers where dims match
            block_residual = (
                use_residual
                and (0 < i < num_layers - 1)
                and (in_d == out_d)
            )

            blocks.append(
                _ResidualMLPBlock(
                    in_dim=in_d,
                    out_dim=out_d,
                    use_residual=block_residual,
                    dropout=dropout,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.output_norm = nn.LayerNorm(normalized_shape=output_dim)

        self._init_weights()
        logger.debug(
            "NodeFeatureEncoder: input=%d, hidden=%d, output=%d, layers=%d, "
            "dropout=%.2f, residual=%s",
            input_dim, hidden_dim, output_dim, num_layers,
            dropout, use_residual,
        )

    # ----- weight initialisation -------------------------------------------

    def _init_weights(self) -> None:
        """Apply Xavier/Glorot initialisation to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ----- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw node features into latent embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Raw node feature tensor.  Supported shapes:
            * ``(N, input_dim)`` — N unbatched node feature vectors.
            * ``(batch, N, input_dim)`` — Batched node feature matrices.
            * ``(input_dim,)`` — Single node feature vector (expanded).

        Returns
        -------
        torch.Tensor
            Encoded node embeddings with the same batch structure but
            final dimension ``output_dim``.
        """
        # Normalise input dimensions
        original_shape = x.shape
        squeeze_all = False
        if x.dim() == 1:
            # Single feature vector → add batch and node dims
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_all = True
        elif x.dim() == 2:
            # (N, input_dim) → (1, N, input_dim)
            x = x.unsqueeze(0)

        # Pass through MLP blocks sequentially
        out = x
        for block in self.blocks:
            out = block(out)

        # Final layer norm on last dimension
        out = self.output_norm(out)

        # Restore original dimensionality
        if squeeze_all:
            out = out.squeeze(0).squeeze(0)
        elif original_shape.dim() == 2:
            out = out.squeeze(0)

        return out

    # ----- utilities --------------------------------------------------------

    def get_output_dim(self) -> int:
        """Return the output embedding dimension.

        Returns
        -------
        int
            The ``output_dim`` set at construction time.
        """
        return self.output_dim

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_layers={self.num_layers}"
        )

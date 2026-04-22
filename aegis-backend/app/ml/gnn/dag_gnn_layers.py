"""
NOTEARS-style DAG-GNN layers for learning directed acyclic graph structure.

This module implements the graph neural network layers introduced in the
DAG-GNN paper (Ng et al., 2019), which combines variational autoencoders
with the NOTEARS continuous acyclicity constraint to discover causal DAGs
from observational data.

Core idea
---------
Each layer updates node representations using the learned adjacency matrix:

    x_new = sigmoid( (I + 1/(d-1) * A) @ x @ W1 + b1 ) @ W2 + b2

where ``A`` is a d×d weighted adjacency matrix, ``x`` is the node feature
matrix, and ``W1``, ``W2``, ``b1``, ``b2`` are learnable parameters.

The acyclicity constraint is:

    h(A) = tr(exp(A ⊙ A)) - d  ≥ 0

where ``tr(exp(·))`` denotes the trace of the matrix exponential and ``d``
is the number of nodes.  When ``h(A) == 0`` the matrix ``A`` corresponds to
a DAG.

References
----------
Yu, Y., Chen, J., Gao, T., & Chen, M. (2019). DAG-GNN: DAG Structure
Learning with Graph Neural Networks.  *ICML*.

Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with
NOTEARS: Continuous Optimization for Structure Learning.  *NeurIPS*.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

logger = logging.getLogger(__name__)

__all__ = ["DAGGNNLayer", "DAGGNNStack"]


# ---------------------------------------------------------------------------
# Helper: matrix exponential (trace)
# ---------------------------------------------------------------------------

def _trace_exponential(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the trace of the matrix exponential ``tr(exp(M))``.

    Uses an eigendecomposition for numerical stability:

        tr(exp(M)) = sum( exp(eigenvalue_i) )

    Parameters
    ----------
    matrix : torch.Tensor
        Square symmetric matrix of shape ``(d, d)``.

    Returns
    -------
    torch.Tensor
        Scalar trace of the matrix exponential.
    """
    eig_vals = torch.linalg.eigvalsh(matrix)
    return torch.exp(eig_vals).sum()


def _acyclicity_constraint(adj: torch.Tensor, d: int) -> torch.Tensor:
    """Compute the NOTEARS acyclicity constraint ``h(A) = tr(e^{A⊙A}) - d``.

    Parameters
    ----------
    adj : torch.Tensor
        Raw (unconstrained) weighted adjacency matrix of shape ``(d, d)``.
    d : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Scalar non-negative acyclicity violation.  Zero implies a DAG.
    """
    # Element-wise square then matrix-exponential trace minus dimension
    aa = adj * adj  # A ⊙ A
    h = _trace_exponential(aa) - d
    return h


# ---------------------------------------------------------------------------
# Single NOTEARS-style GNN layer
# ---------------------------------------------------------------------------

class DAGGNNLayer(nn.Module):
    """A single DAG-GNN layer that updates node representations.

    The forward pass implements the NOTEARS-style message passing:

        x_out = sigmoid( (I + 1/(d-1) * A) @ x @ W1 + b1 ) @ W2 + b2

    where the identity plus scaled adjacency matrix ``(I + 1/(d-1) * A)`` acts
    as the message-passing adjacency, incorporating both self-loops and
    information flow along learned edges.

    Parameters
    ----------
    in_features : int
        Dimension of input node features ``x``.
    out_features : int
        Dimension of output node representations.
    d : int
        Number of nodes (graph size).  Used to scale the adjacency matrix.
    bias : bool, optional
        Whether to include bias terms ``b1`` and ``b2``.  Default ``True``.

    Attributes
    ----------
    W1 : torch.nn.Parameter
        First linear weight matrix of shape ``(in_features, out_features)``.
    W2 : torch.nn.Parameter
        Second linear weight matrix of shape ``(out_features, out_features)``.
    b1 : torch.nn.Parameter or None
        Bias for the first linear transform.
    b2 : torch.nn.Parameter or None
        Bias for the second linear transform.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        d: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d < 2:
            raise ValueError(f"Number of nodes d={d} must be >= 2.")
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in_features={in_features} and out_features={out_features} "
                "must be positive."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.d = d
        self.scale = 1.0 / (d - 1)

        # Xavier/Glorot initialisation
        W1 = torch.empty(in_features, out_features)
        W2 = torch.empty(out_features, out_features)
        nn.init.xavier_uniform_(W1)
        nn.init.xavier_uniform_(W2)
        self.W1 = nn.Parameter(W1)
        self.W2 = nn.Parameter(W2)

        if bias:
            self.b1 = nn.Parameter(torch.zeros(out_features))
            self.b2 = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("b1", None)
            self.register_parameter("b2", None)

        logger.debug(
            "DAGGNNLayer initialised: in=%d, out=%d, d=%d, bias=%s",
            in_features, out_features, d, bias,
        )

    # ----- forward ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Run one step of DAG-GNN message passing.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(batch, d, in_features)`` or
            ``(d, in_features)``.
        adj : torch.Tensor
            Weighted adjacency matrix of shape ``(d, d)``.  Typically raw
            (unconstrained) outputs from the encoder / adjacency module.

        Returns
        -------
        torch.Tensor
            Updated node representations of shape
            ``(batch, d, out_features)`` or ``(d, out_features)``.
        """
        d = self.d
        identity = torch.eye(d, device=x.device, dtype=x.dtype)

        # Build normalised adjacency: (I + 1/(d-1) * A)
        norm_adj = identity + self.scale * adj

        # Handle batched vs. unbatched inputs
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        # (batch, d, in_features) @ W1 → (batch, d, out_features)
        linear1 = x @ self.W1
        if self.b1 is not None:
            linear1 = linear1 + self.b1

        # norm_adj @ linear1 → message passing
        # norm_adj: (d, d), linear1: (batch, d, out)
        # Use einsum for clarity
        messages = torch.einsum("ij,bjf->bjf", norm_adj, linear1)

        # Apply sigmoid
        activated = torch.sigmoid(messages)

        # Second linear: (batch, d, out_features) @ W2 → (batch, d, out_features)
        out = activated @ self.W2
        if self.b2 is not None:
            out = out + self.b2

        if squeeze_output:
            out = out.squeeze(0)

        return out

    # ----- utilities --------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"d={self.d}, bias={self.b1 is not None}"
        )


# ---------------------------------------------------------------------------
# Stack of DAG-GNN layers
# ---------------------------------------------------------------------------

class DAGGNNStack(nn.Module):
    """Stack of NOTEARS-style DAG-GNN layers with residual connections.

    Each layer in the stack applies the DAG-GNN message-passing update.
    When ``in_features == out_features``, a residual connection is added
    from the input of the stack directly to the output.

    Parameters
    ----------
    num_layers : int
        Number of DAG-GNN layers in the stack.
    in_features : int
        Input node feature dimension.
    hidden_features : int
        Hidden dimension used for every layer except the last.
    out_features : int, optional
        Output feature dimension of the final layer.  Defaults to
        ``hidden_features``.
    d : int
        Number of nodes in the graph.
    bias : bool, optional
        Whether to use bias terms.  Default ``True``.
    dropout : float, optional
        Dropout probability applied between layers.  Default ``0.0``.

    Attributes
    ----------
    layers : torch.nn.ModuleList
        The list of :class:`DAGGNNLayer` instances.
    dropout : torch.nn.Dropout
        Dropout layer applied between stacked layers.
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        hidden_features: int,
        d: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers={num_layers} must be >= 1.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout={dropout} must be in [0, 1).")

        self.num_layers = num_layers
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features if out_features is not None else hidden_features
        self.d = d

        layers: list[nn.Module] = []
        for idx in range(num_layers):
            if idx == 0:
                layer_in = in_features
            else:
                layer_in = hidden_features

            if idx == num_layers - 1:
                layer_out = self.out_features
            else:
                layer_out = hidden_features

            layers.append(
                DAGGNNLayer(
                    in_features=layer_in,
                    out_features=layer_out,
                    d=d,
                    bias=bias,
                )
            )

        self.layers = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Residual connection is only possible when dimensions match
        self._use_residual = (in_features == self.out_features) and (num_layers > 1)

        logger.debug(
            "DAGGNNStack initialised: layers=%d, in=%d, hidden=%d, out=%d, "
            "d=%d, residual=%s",
            num_layers, in_features, hidden_features, self.out_features,
            d, self._use_residual,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the full stack of DAG-GNN layers sequentially.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix ``(batch, d, in_features)`` or ``(d, in_features)``.
        adj : torch.Tensor
            Weighted adjacency matrix of shape ``(d, d)``.

        Returns
        -------
        torch.Tensor
            Updated node representations of shape
            ``(batch, d, out_features)`` or ``(d, out_features)``.
        """
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        identity = x

        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out, adj)
            if idx < self.num_layers - 1:
                out = self.dropout_layer(out)

        # Residual connection
        if self._use_residual:
            out = out + identity

        if squeeze_output:
            out = out.squeeze(0)

        return out

    @staticmethod
    def compute_acyclicity(adj: torch.Tensor) -> torch.Tensor:
        """Compute the NOTEARS acyclicity constraint ``h(A)``.

        Convenience wrapper around :func:`_acyclicity_constraint`.

        Parameters
        ----------
        adj : torch.Tensor
            Raw adjacency matrix of shape ``(d, d)``.

        Returns
        -------
        torch.Tensor
            Scalar acyclicity violation (≥ 0, 0 = DAG).
        """
        d = adj.shape[0]
        return _acyclicity_constraint(adj, d)

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, "
            f"in_features={self.in_features}, "
            f"hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, "
            f"d={self.d}, residual={self._use_residual}"
        )

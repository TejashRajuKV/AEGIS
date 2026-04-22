"""
Graph Attention Network (GAT) layers for node-level attention mechanisms.

This module implements attention-based message passing as described by
Veličković et al. (2018), with extensions supporting both additive
(a.k.a. concatenation-based) and dot-product attention variants.

The attention mechanism allows each node to assign different importances
to its neighbours, which is valuable for causal discovery where not all
edges carry equal informational weight.

Classes
-------
:class:`GraphAttentionLayer`     – Single-head GAT attention layer.
:class:`MultiHeadGraphAttention` – Multi-head GAT wrapper with
                                    averaging or concatenation.

References
----------
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., &
Bengio, Y. (2018). Graph Attention Networks. *ICLR*.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

logger = logging.getLogger(__name__)

__all__ = ["GraphAttentionLayer", "MultiHeadGraphAttention"]


class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention Network (GAT) layer.

    Computes attention-weighted neighbourhood aggregations.  For each node
    ``i``, the output feature is:

        h_i' = σ( Σ_{j ∈ N(i)} α_{ij} W h_j )

    where ``α_{ij}`` is the attention coefficient computed via either
    additive or dot-product attention.

    Additive attention
    ------------------
        e_{ij} = LeakyReLU( a^T [Wh_i || Wh_j] )
        α_{ij} = softmax_j( e_{ij} )

    Dot-product attention
    ---------------------
        e_{ij} = (Wh_i)^T (Wh_j) / √d
        α_{ij} = softmax_j( e_{ij} )

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Number of output features per node.
    attention_type : str, optional
        Type of attention mechanism: ``"additive"`` or ``"dot_product"``.
        Default ``"additive"``.
    dropout : float, optional
        Dropout probability for attention coefficients.  Default 0.1.
    leaky_relu_slope : float, optional
        Negative slope for LeakyReLU.  Default 0.2.
    concat : bool, optional
        If ``True``, output dimension is ``out_features`` (a single head).
        Kept for API compatibility with GAT conventions.  Default ``True``.
    bias : bool, optional
        Whether to include bias in the linear transform.  Default ``True``.

    Attributes
    ----------
    W : torch.nn.Parameter
        Feature transformation matrix of shape ``(in_features, out_features)``.
    a_src : torch.nn.Parameter or None
        Attention parameter for source nodes (additive mode).
    a_tgt : torch.nn.Parameter or None
        Attention parameter for target nodes (additive mode).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        attention_type: Literal["additive", "dot_product"] = "additive",
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.2,
        concat: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature dimensions must be positive.")
        if attention_type not in ("additive", "dot_product"):
            raise ValueError(
                f"attention_type must be 'additive' or 'dot_product', "
                f"got '{attention_type}'."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.in_features = in_features
        self.out_features = out_features
        self.attention_type = attention_type
        self.concat = concat
        self.leaky_relu_slope = leaky_relu_slope

        # Feature transformation
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W)

        # Attention parameters
        if attention_type == "additive":
            # a_src: (out_features, 1), a_tgt: (out_features, 1)
            # The combined a vector is [a_src || a_tgt] of length 2*out_features
            self.a_src = nn.Parameter(torch.empty(out_features, 1))
            self.a_tgt = nn.Parameter(torch.empty(out_features, 1))
            nn.init.xavier_uniform_(self.a_src)
            nn.init.xavier_uniform_(self.a_tgt)
        else:
            self.register_parameter("a_src", None)
            self.register_parameter("a_tgt", None)

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Dropout
        self.attn_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # LeakyReLU
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)

        logger.debug(
            "GraphAttentionLayer: in=%d, out=%d, attn=%s, dropout=%.2f",
            in_features, out_features, attention_type, dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted node representations.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(d, in_features)`` or
            ``(batch, d, in_features)``.
        adj : torch.Tensor, optional
            Sparse/binary adjacency matrix of shape ``(d, d)`` or
            ``(batch, d, d)``.  If provided, attention is restricted to
            non-zero edges.  If ``None``, a fully-connected graph is assumed.

        Returns
        -------
        h_prime : torch.Tensor
            Attention-aggregated node features of shape ``(d, out_features)``
            or ``(batch, d, out_features)``.
        attention_weights : torch.Tensor
            Attention coefficient matrix of shape ``(d, d)`` or
            ``(batch, d, d)``.
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        batch_size, d, _ = x.shape

        # Linear transform: (batch, d, in_features) @ W → (batch, d, out_features)
        Wh = x @ self.W  # (batch, d, out_features)

        # Compute attention coefficients
        if self.attention_type == "additive":
            attention_weights = self._additive_attention(Wh, batch_size, d)
        else:
            attention_weights = self._dot_product_attention(Wh, batch_size, d)

        # Mask with adjacency if provided
        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)
            # Zero out attention for non-edges
            mask = (adj == 0).float()
            # Use a large negative value for masked positions before softmax
            attention_weights = attention_weights.masked_fill(
                mask.bool(), float("-inf")
            )
            # Replace inf with 0 for rows that are fully masked (isolated nodes)
            attention_weights = torch.where(
                torch.isinf(attention_weights),
                torch.zeros_like(attention_weights),
                attention_weights,
            )

        # Softmax normalisation: α_{ij} = softmax_j(e_{ij})
        attention_weights = functional.softmax(attention_weights, dim=-1)
        # Clamp to avoid NaN from fully-masked rows
        attention_weights = torch.clamp(attention_weights, min=0.0, max=1.0)

        # Apply dropout to attention coefficients
        attention_weights = self.attn_dropout(attention_weights)

        # Aggregate: h_i' = Σ_j α_{ij} Wh_j
        h_prime = torch.bmm(attention_weights, Wh)  # (batch, d, out_features)

        # Add bias
        if self.bias is not None:
            h_prime = h_prime + self.bias

        if squeeze_batch:
            h_prime = h_prime.squeeze(0)
            attention_weights = attention_weights.squeeze(0)

        return h_prime, attention_weights

    # ----- attention computation helpers ------------------------------------

    def _additive_attention(
        self,
        Wh: torch.Tensor,
        batch_size: int,
        d: int,
    ) -> torch.Tensor:
        """Compute additive attention coefficients.

        ``e_{ij} = LeakyReLU( a_src^T Wh_i + a_tgt^T Wh_j )``

        Parameters
        ----------
        Wh : torch.Tensor
            Transformed features of shape ``(batch, d, out_features)``.
        batch_size : int
            Batch size.
        d : int
            Number of nodes.

        Returns
        -------
        torch.Tensor
            Raw attention scores of shape ``(batch, d, d)``.
        """
        # (batch, d, out_features) @ a_src → (batch, d, 1) → (batch, d)
        f_src = (Wh @ self.a_src).squeeze(-1)  # (batch, d)
        # (batch, d, out_features) @ a_tgt → (batch, d, 1) → (batch, d)
        f_tgt = (Wh @ self.a_tgt).squeeze(-1)  # (batch, d)

        # Broadcasting: (batch, d, 1) + (batch, 1, d) → (batch, d, d)
        e = f_src.unsqueeze(2) + f_tgt.unsqueeze(1)
        return self.leaky_relu(e)

    def _dot_product_attention(
        self,
        Wh: torch.Tensor,
        batch_size: int,
        d: int,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention coefficients.

        ``e_{ij} = (Wh_i)^T Wh_j / √out_features``

        Parameters
        ----------
        Wh : torch.Tensor
            Transformed features of shape ``(batch, d, out_features)``.
        batch_size : int
            Batch size.
        d : int
            Number of nodes.

        Returns
        -------
        torch.Tensor
            Raw attention scores of shape ``(batch, d, d)``.
        """
        scale = self.out_features ** 0.5
        # (batch, d, out) @ (batch, out, d) → (batch, d, d)
        e = torch.bmm(Wh, Wh.transpose(1, 2)) / scale
        return self.leaky_relu(e)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"attention_type={self.attention_type}, "
            f"concat={self.concat}"
        )


class MultiHeadGraphAttention(nn.Module):
    """Multi-head Graph Attention Network layer.

    Runs multiple :class:`GraphAttentionLayer` heads in parallel (on the
    same device) and combines their outputs either by averaging or by
    concatenating along the feature dimension.

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Number of output features **per head**.
    num_heads : int
        Number of attention heads.
    attention_type : str, optional
        Attention variant (``"additive"`` or ``"dot_product"``).
        Default ``"additive"``.
    concat : bool, optional
        If ``True``, head outputs are concatenated, giving final output
        dimension ``num_heads * out_features``.  If ``False``, heads are
        averaged, giving final output dimension ``out_features``.
        Default ``True``.
    dropout : float, optional
        Dropout probability for attention coefficients.  Default 0.1.
    leaky_relu_slope : float, optional
        Negative slope for LeakyReLU.  Default 0.2.
    bias : bool, optional
        Whether to include bias.  Default ``True``.

    Attributes
    ----------
    heads : torch.nn.ModuleList
        List of individual :class:`GraphAttentionLayer` heads.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        attention_type: Literal["additive", "dot_product"] = "additive",
        concat: bool = True,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")
        if out_features <= 0 or in_features <= 0:
            raise ValueError("Feature dimensions must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.attention_type = attention_type

        # Build individual heads
        heads: list[nn.Module] = []
        for _ in range(num_heads):
            heads.append(
                GraphAttentionLayer(
                    in_features=in_features,
                    out_features=out_features,
                    attention_type=attention_type,
                    dropout=dropout,
                    leaky_relu_slope=leaky_relu_slope,
                    concat=concat,
                    bias=bias,
                )
            )
        self.heads = nn.ModuleList(heads)

        # Final output dimension
        if concat:
            self.final_out_features = num_heads * out_features
        else:
            self.final_out_features = out_features

        # Optional output projection for stability
        self.output_proj = nn.Linear(
            in_features=self.final_out_features,
            out_features=self.final_out_features,
            bias=False,
        )
        nn.init.xavier_uniform_(self.output_proj.weight)

        logger.debug(
            "MultiHeadGraphAttention: in=%d, out=%d, heads=%d, concat=%s, "
            "final_out=%d",
            in_features, out_features, num_heads, concat,
            self.final_out_features,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention and combine outputs.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(d, in_features)`` or ``(batch, d, in_features)``.
        adj : torch.Tensor, optional
            Adjacency matrix of shape ``(d, d)`` or ``(batch, d, d)``.
            If ``None``, full connectivity is assumed.

        Returns
        -------
        out : torch.Tensor
            Combined output features.  Shape depends on ``concat``:
            * ``(d, num_heads * out_features)`` if concat is True
            * ``(d, out_features)`` if concat is False
            With batch dimension: ``(batch, d, ...)``.
        mean_attention : torch.Tensor
            Mean attention weights across heads, shape ``(d, d)``
            or ``(batch, d, d)``.
        """
        head_outputs: list[torch.Tensor] = []
        head_attentions: list[torch.Tensor] = []

        for head in self.heads:
            h_out, h_attn = head(x, adj=adj)
            head_outputs.append(h_out)
            head_attentions.append(h_attn)

        squeeze_batch = False
        if head_outputs[0].dim() == 2:
            head_outputs = [h.unsqueeze(0) for h in head_outputs]
            head_attentions = [a.unsqueeze(0) for a in head_attentions]
            squeeze_batch = True

        # Combine heads
        if self.concat:
            # Concatenate along feature dimension
            out = torch.cat(head_outputs, dim=-1)  # (batch, d, heads*out)
        else:
            # Average across heads
            out = torch.stack(head_outputs, dim=0).mean(dim=0)  # (batch, d, out)

        # Mean attention across heads
        mean_attention = torch.stack(head_attentions, dim=0).mean(dim=0)

        # Output projection for stability
        out = self.output_proj(out)

        if squeeze_batch:
            out = out.squeeze(0)
            mean_attention = mean_attention.squeeze(0)

        return out, mean_attention

    def get_final_out_features(self) -> int:
        """Return the effective output dimension after head combination.

        Returns
        -------
        int
            ``num_heads * out_features`` if concat, else ``out_features``.
        """
        return self.final_out_features

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_heads={self.num_heads}, "
            f"attention_type={self.attention_type}, "
            f"concat={self.concat}, "
            f"final_out_features={self.final_out_features}"
        )

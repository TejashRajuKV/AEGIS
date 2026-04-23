"""
DAG-GNN Causal Discovery — main entry point for learning causal DAGs.

This module provides :class:`DAGGNNDiscovery`, a high-level API that wraps
the :class:`DAGGNNModel` and :class:`CausalGNNTrainer` into a simple
interface for discovering causal structure from tabular data.

Usage
-----
::

    discovery = DAGGNNDiscovery(input_dim=10, hidden_dim=64)
    result = discovery.discover(data, feature_names=["age", "income", ...])
    print(result.edge_list)
    proxy_chains = discovery.get_proxy_chains(result.graph)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False
    logger.warning("PyTorch is not installed. DAG-GNN discovery is unavailable.")

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    _HAS_NETWORKX = False
    logger.warning("networkx is not installed. Graph operations will be limited.")

try:
    from app.ml.causal.dag_gnn_model import DAGGNNModel
    _HAS_DAG_GNN_MODEL = True
except ImportError as exc:
    DAGGNNModel = None  # type: ignore[assignment, misc]
    _HAS_DAG_GNN_MODEL = False
    logger.warning("DAGGNNModel import failed: %s", exc)

try:
    from app.ml.gnn.causal_gnn_trainer import CausalGNNTrainer, TrainingConfig
    _HAS_TRAINER = True
except ImportError as exc:
    CausalGNNTrainer = None  # type: ignore[assignment, misc]
    TrainingConfig = None  # type: ignore[assignment, misc]
    _HAS_TRAINER = False
    logger.warning("CausalGNNTrainer import failed: %s", exc)


__all__ = ["DAGGNNDiscovery", "CausalGraph"]


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class CausalGraph:
    """Result of a DAG-GNN causal discovery run.

    Attributes
    ----------
    adjacency_matrix : numpy.ndarray or None
        Learned adjacency matrix of shape ``(d, d)``.
    edge_list : list of tuple
        List of ``(source, target, weight)`` tuples for edges above threshold.
    graph : networkx.DiGraph or None
        Learned causal graph as a directed networkx graph.
    feature_names : list of str
        Node / feature names used in discovery.
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of discovered edges.
    sparsity : float
        Fraction of absent edges (0 = fully connected, 1 = empty graph).
    metadata : dict
        Additional metadata about the discovery run.
    """

    adjacency_matrix: Any = None
    edge_list: List[Tuple[str, str, float]] = field(default_factory=list)
    graph: Any = None
    feature_names: List[str] = field(default_factory=list)
    num_nodes: int = 0
    num_edges: int = 0
    sparsity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result to a dictionary."""
        adj_data = None
        if self.adjacency_matrix is not None and _HAS_NUMPY:
            adj_data = self.adjacency_matrix.tolist()

        graph_data = None
        if self.graph is not None and _HAS_NETWORKX and isinstance(self.graph, nx.DiGraph):
            graph_data = {
                "nodes": list(self.graph.nodes()),
                "edges": [
                    {"source": u, "target": v, "weight": d.get("weight", 0.0)}
                    for u, v, d in self.graph.edges(data=True)
                ],
            }

        return {
            "adjacency_matrix": adj_data,
            "edge_list": self.edge_list,
            "graph": graph_data,
            "feature_names": self.feature_names,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "sparsity": self.sparsity,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Main discovery class
# ---------------------------------------------------------------------------

class DAGGNNDiscovery:
    """High-level API for DAG-GNN causal discovery.

    Wraps :class:`DAGGNNModel` and :class:`CausalGNNTrainer` to provide a
    simple ``discover(data)`` interface that returns a :class:`CausalGraph`.

    Parameters
    ----------
    input_dim : int
        Number of variables / features in the input data.
    hidden_dim : int, optional
        Hidden dimension for GNN layers.  Default ``64``.
    latent_dim : int, optional
        Latent bottleneck dimension.  Default ``16``.
    lr : float, optional
        Learning rate for the trainer.  Default ``1e-3``.
    lambda1 : float, optional
        L1 sparsity regularisation weight.  Default ``0.01``.
    lambda2 : float, optional
        Acyclicity constraint weight.  Default ``0.1``.
    threshold : float, optional
        Edge threshold for binarising the adjacency matrix.  Default ``0.3``.
    max_epochs : int, optional
        Maximum training epochs.  Default ``1000``.
    patience : int, optional
        Early stopping patience.  Default ``100``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        lr: float = 1e-3,
        lambda1: float = 0.01,
        lambda2: float = 0.1,
        threshold: float = 0.3,
        max_epochs: int = 1000,
        patience: int = 100,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.threshold = threshold
        self.max_epochs = max_epochs
        self.patience = patience

        self._model: Optional[DAGGNNModel] = None
        self._trainer: Optional[CausalGNNTrainer] = None
        self._is_trained: bool = False

        logger.info(
            "DAGGNNDiscovery initialised: input_dim=%d, hidden_dim=%d, "
            "latent_dim=%d, lr=%.5f, lambda1=%.4f, lambda2=%.4f, "
            "threshold=%.2f",
            input_dim, hidden_dim, latent_dim, lr, lambda1, lambda2, threshold,
        )

    def discover(
        self,
        data: Any,
        feature_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """Run causal discovery on the given data.

        Parameters
        ----------
        data : array-like
            Input data of shape ``(N, d)`` where ``N`` is the number of
            samples and ``d`` is the number of variables.
        feature_names : list of str, optional
            Human-readable names for each variable.  If ``None``, names
            are generated as ``"X0"``, ``"X1"``, etc.

        Returns
        -------
        CausalGraph
            Discovery result containing the adjacency matrix, edge list,
            and networkx DiGraph.

        Raises
        ------
        RuntimeError
            If PyTorch or required modules are not installed.
        ValueError
            If data dimensions are incompatible.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for DAG-GNN discovery.")
        if not _HAS_TRAINER:
            raise RuntimeError("CausalGNNTrainer is not available.")
        if not _HAS_NUMPY:
            raise RuntimeError("NumPy is required for data conversion.")

        # Convert data to numpy then to torch tensor
        data_array = np.asarray(data, dtype=np.float64)
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)

        n_samples, n_features = data_array.shape

        if n_features != self.input_dim:
            raise ValueError(
                f"Data has {n_features} features but input_dim={self.input_dim}."
            )

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"number of features ({n_features})."
            )

        data_tensor = torch.FloatTensor(data_array)

        logger.info(
            "Starting DAG-GNN discovery: %d samples, %d features",
            n_samples, n_features,
        )

        # Build and configure trainer
        config = TrainingConfig(
            num_nodes=n_features,
            input_dim=n_features,
            hidden_dim=self.hidden_dim,
            num_gnn_layers=2,
            num_encoder_layers=3,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            learning_rate=self.lr,
            max_epochs=self.max_epochs,
            patience=self.patience,
            threshold=self.threshold,
            dropout=0.1,
        )

        self._trainer = CausalGNNTrainer(
            config=config,
            node_names=list(feature_names),
        )

        # Train
        learned_adj = self._trainer.train(data_tensor)
        self._is_trained = True

        # Extract adjacency as numpy
        adj_np = learned_adj.detach().cpu().numpy() if torch.is_tensor(learned_adj) else np.asarray(learned_adj)

        # Build edge list
        edge_list: List[Tuple[str, str, float]] = []
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue
                w = float(adj_np[i, j])
                if w >= self.threshold:
                    edge_list.append((feature_names[i], feature_names[j], w))

        # Build networkx graph
        graph = None
        if _HAS_NETWORKX:
            graph = nx.DiGraph()
            graph.add_nodes_from(feature_names)
            for src, tgt, w in edge_list:
                graph.add_edge(src, tgt, weight=w)

        # Compute sparsity
        max_edges = n_features * (n_features - 1)
        sparsity = 1.0 - (len(edge_list) / max_edges) if max_edges > 0 else 1.0

        result = CausalGraph(
            adjacency_matrix=adj_np,
            edge_list=edge_list,
            graph=graph,
            feature_names=feature_names,
            num_nodes=n_features,
            num_edges=len(edge_list),
            sparsity=sparsity,
            metadata={
                "n_samples": n_samples,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
                "threshold": self.threshold,
                "max_epochs": self.max_epochs,
            },
        )

        logger.info(
            "Discovery complete: %d nodes, %d edges, sparsity=%.3f",
            n_features, len(edge_list), sparsity,
        )
        return result

    def get_proxy_chains(self, graph: Any) -> List[Dict[str, Any]]:
        """Detect proxy chains in a causal graph.

        Uses :class:`ProxyChainDetector` to find paths where sensitive
        attributes influence the outcome through intermediate proxy variables.

        Parameters
        ----------
        graph : networkx.DiGraph
            The learned causal graph.

        Returns
        -------
        list of dict
            List of proxy chain dictionaries, each containing:
            ``path``, ``risk_score``, ``recommendation``.
        """
        if not _HAS_NETWORKX:
            logger.warning("networkx is not installed. Cannot detect proxy chains.")
            return []

        try:
            from app.ml.causal.proxy_chain_detector import ProxyChainDetector

            detector = ProxyChainDetector()
            result = detector.detect(
                graph=graph,
                sensitive_nodes=self._infer_sensitive_nodes(graph),
                outcome_node=self._infer_outcome_node(graph),
            )

            chains: List[Dict[str, Any]] = []
            if result is not None:
                for chain, risk in zip(result.chains, result.risk_scores):
                    chains.append({
                        "path": chain,
                        "risk_score": risk,
                    })

            recommendations = detector.get_recommendations(chains)
            for i, chain in enumerate(chains):
                if i < len(recommendations):
                    chain["recommendation"] = recommendations[i]

            return chains

        except Exception as exc:
            logger.error("Proxy chain detection failed: %s", exc)
            return []

    def visualize(self) -> Dict[str, Any]:
        """Return graph data formatted for frontend visualisation.

        Returns
        -------
        dict
            Dictionary with ``nodes`` and ``edges`` lists suitable for
            rendering with D3.js, Cytoscape, or similar libraries.
        """
        if not self._is_trained or self._trainer is None:
            logger.warning("Model not trained yet. Call discover() first.")
            return {"nodes": [], "edges": []}

        try:
            graph = self._trainer.get_causal_graph()
        except Exception as exc:
            logger.error("Failed to get causal graph: %s", exc)
            return {"nodes": [], "edges": []}

        if not _HAS_NETWORKX or graph is None:
            return {"nodes": [], "edges": []}

        nodes = []
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            nodes.append({
                "id": node,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "is_sensitive": self._is_sensitive_name(node),
            })

        edges = []
        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            edges.append({
                "source": u,
                "target": v,
                "weight": float(weight),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_sensitive_nodes(graph: Any) -> List[str]:
        """Infer which nodes are likely sensitive attributes.

        Uses heuristic: nodes whose names contain common sensitive attribute
        keywords are classified as sensitive.
        """
        if not _HAS_NETWORKX or graph is None:
            return []

        sensitive_keywords = [
            "race", "gender", "sex", "age", "ethnicity", "religion",
            "disability", "national_origin", "marital_status", "pregnancy",
            "color", "creed", "orientation",
        ]
        sensitive = []
        for node in graph.nodes():
            name_lower = str(node).lower().replace("_", " ").replace("-", " ")
            for keyword in sensitive_keywords:
                if keyword in name_lower:
                    sensitive.append(str(node))
                    break
        return sensitive

    @staticmethod
    def _infer_outcome_node(graph: Any) -> str:
        """Infer which node is the likely outcome variable.

        Uses heuristic: the node with the highest in-degree is likely
        the outcome (most things point to it).
        """
        if not _HAS_NETWORKX or graph is None:
            return ""

        if graph.number_of_nodes() == 0:
            return ""

        # Node with highest in-degree is the outcome
        max_in_degree = -1
        outcome = ""
        for node in graph.nodes():
            in_deg = graph.in_degree(node)
            if in_deg > max_in_degree:
                max_in_degree = in_deg
                outcome = str(node)
        return outcome

    @staticmethod
    def _is_sensitive_name(node: str) -> bool:
        """Check if a node name suggests a sensitive attribute."""
        sensitive_keywords = [
            "race", "gender", "sex", "age", "ethnicity", "religion",
            "disability", "national_origin", "marital_status", "pregnancy",
            "color", "creed", "orientation",
        ]
        name_lower = str(node).lower().replace("_", " ").replace("-", " ")
        return any(kw in name_lower for kw in sensitive_keywords)

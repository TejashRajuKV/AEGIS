"""AEGIS Graph Utilities - Helper functions for causal graph operations.

Merged from:
  - V3: Pure NumPy implementations, rich graph statistics (compute_graph_stats)
  - V5: NetworkX bridge functions, find_proxy_variables

Provides both pure-NumPy and NetworkX-backed implementations so callers can
choose depending on whether the ``networkx`` dependency is available.
"""

from typing import Dict, List, Optional, Set

import numpy as np

from app.utils.logger import get_logger

logger = get_logger("graph_utils")

# Optional NetworkX import – the module still works without it, but
# NetworkX-backed helpers will raise ImportError at call time.
try:
    import networkx as nx

    _HAS_NX = True
except ImportError:  # pragma: no cover
    _HAS_NX = False

# ======================================================================
# Pure NumPy implementations (V3)
# ======================================================================


def adjacency_to_edges(
    adj_matrix: np.ndarray, node_names: List[str]
) -> List[Dict]:
    """Convert adjacency matrix to edge list.

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.

    Returns:
        List of dicts with ``source``, ``target``, ``weight`` keys.
    """
    edges: List[Dict] = []
    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] != 0 and i != j:
                edges.append({
                    "source": node_names[i],
                    "target": node_names[j],
                    "weight": float(adj_matrix[i][j]),
                })
    return edges


def edges_to_adjacency(
    edges: List[Dict], node_names: List[str]
) -> np.ndarray:
    """Convert edge list to adjacency matrix.

    Args:
        edges: List of edge dicts with ``source`` and ``target``.
        node_names: List of node names.

    Returns:
        Square adjacency matrix (float64).
    """
    n = len(node_names)
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    adj = np.zeros((n, n), dtype=np.float64)
    for edge in edges:
        src = name_to_idx.get(edge.get("source", ""))
        tgt = name_to_idx.get(edge.get("target", ""))
        if src is not None and tgt is not None:
            adj[src][tgt] = edge.get("weight", 1.0)
    return adj


def is_dag(adj_matrix: np.ndarray) -> bool:
    """Check whether *adj_matrix* represents a DAG via Kahn's algorithm.

    Args:
        adj_matrix: Square adjacency matrix.

    Returns:
        ``True`` if the graph is acyclic.
    """
    n = adj_matrix.shape[0]
    in_degree = np.sum(adj_matrix != 0, axis=0)
    queue: List[int] = [i for i in range(n) if in_degree[i] == 0]
    visited = 0

    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbour in range(n):
            if adj_matrix[node][neighbour] != 0:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

    return visited == n


def topological_sort(
    adj_matrix: np.ndarray, node_names: List[str]
) -> List[str]:
    """Return node names in topological order (pure NumPy, Kahn's algorithm).

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.

    Returns:
        List of node names in topological order.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    if not is_dag(adj_matrix):
        raise ValueError("Cannot topologically sort a graph with cycles")

    n = adj_matrix.shape[0]
    in_degree = np.sum(adj_matrix != 0, axis=0)
    queue = sorted([i for i in range(n) if in_degree[i] == 0])
    order: List[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node_names[node])
        for neighbour in range(n):
            if adj_matrix[node][neighbour] != 0:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

    return order


def get_parents(
    adj_matrix: np.ndarray, node_names: List[str], node: str
) -> List[str]:
    """Return direct parents of *node* (nodes with an edge into *node*).

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.
        node: Target node name.

    Returns:
        Sorted list of parent node names.
    """
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    idx = name_to_idx.get(node)
    if idx is None:
        return []
    return sorted(
        node_names[i]
        for i in range(len(node_names))
        if adj_matrix[i][idx] != 0
    )


def get_children(
    adj_matrix: np.ndarray, node_names: List[str], node: str
) -> List[str]:
    """Return direct children of *node* (nodes with an edge from *node*).

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.
        node: Source node name.

    Returns:
        Sorted list of child node names.
    """
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    idx = name_to_idx.get(node)
    if idx is None:
        return []
    return sorted(
        node_names[j]
        for j in range(len(node_names))
        if adj_matrix[idx][j] != 0
    )


def get_ancestors(
    adj_matrix: np.ndarray, node_names: List[str], node: str
) -> List[str]:
    """Return all ancestors of *node* (BFS backwards through parents).

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.
        node: Target node name.

    Returns:
        Sorted list of ancestor node names (excluding *node* itself).
    """
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    idx = name_to_idx.get(node)
    if idx is None:
        return []

    n = adj_matrix.shape[0]
    visited: Set[int] = set()
    queue = [idx]

    while queue:
        current = queue.pop(0)
        for i in range(n):
            if adj_matrix[i][current] != 0 and i not in visited:
                visited.add(i)
                queue.append(i)

    visited.discard(idx)
    return sorted(node_names[i] for i in visited)


def get_descendants(
    adj_matrix: np.ndarray, node_names: List[str], node: str
) -> List[str]:
    """Return all descendants of *node* (BFS forwards through children).

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.
        node: Source node name.

    Returns:
        Sorted list of descendant node names (excluding *node* itself).
    """
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    idx = name_to_idx.get(node)
    if idx is None:
        return []

    n = adj_matrix.shape[0]
    visited: Set[int] = set()
    queue = [idx]

    while queue:
        current = queue.pop(0)
        for j in range(n):
            if adj_matrix[current][j] != 0 and j not in visited:
                visited.add(j)
                queue.append(j)

    visited.discard(idx)
    return sorted(node_names[i] for i in visited)


def graph_to_json(
    adj_matrix: np.ndarray, node_names: List[str]
) -> Dict:
    """Convert graph to JSON-serializable dict.

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.

    Returns:
        Dict with ``nodes``, ``edges``, and ``metadata``.
    """
    edges = adjacency_to_edges(adj_matrix, node_names)
    return {
        "nodes": [{"id": name, "label": name} for name in node_names],
        "edges": edges,
        "metadata": {
            "num_nodes": len(node_names),
            "num_edges": len(edges),
            "is_dag": is_dag(adj_matrix),
        },
    }


def find_all_paths(
    adj_matrix: np.ndarray,
    node_names: List[str],
    start: str,
    end: str,
    max_length: int = 10,
) -> List[List[str]]:
    """Find all simple paths from *start* to *end* (pure NumPy DFS).

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.
        start: Start node name.
        end: End node name.
        max_length: Maximum path length (number of edges).

    Returns:
        List of paths (each path is a list of node names).
    """
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    if start not in name_to_idx or end not in name_to_idx:
        return []

    start_idx = name_to_idx[start]
    end_idx = name_to_idx[end]
    paths: List[List[str]] = []

    def _dfs(current: int, path: List[int]) -> None:
        if len(path) - 1 > max_length:
            return
        if current == end_idx:
            paths.append([node_names[p] for p in path])
            return
        for neighbour in range(adj_matrix.shape[0]):
            if adj_matrix[current][neighbour] != 0 and neighbour not in path:
                _dfs(neighbour, path + [neighbour])

    _dfs(start_idx, [start_idx])
    return paths


def compute_graph_stats(adj_matrix: np.ndarray) -> Dict:
    """Compute rich graph statistics (pure NumPy).

    Args:
        adj_matrix: Square adjacency matrix.

    Returns:
        Dict with ``num_nodes``, ``num_edges``, ``density``,
        ``avg_in_degree``, ``avg_out_degree``, ``max_in_degree``,
        ``max_out_degree``.
    """
    n = adj_matrix.shape[0]
    num_edges = int(np.sum(adj_matrix != 0))
    density = num_edges / (n * (n - 1)) if n > 1 else 0.0
    in_degrees = np.sum(adj_matrix != 0, axis=0)
    out_degrees = np.sum(adj_matrix != 0, axis=1)

    return {
        "num_nodes": n,
        "num_edges": num_edges,
        "density": round(float(density), 6),
        "avg_in_degree": round(float(np.mean(in_degrees)), 4),
        "avg_out_degree": round(float(np.mean(out_degrees)), 4),
        "max_in_degree": int(np.max(in_degrees)),
        "max_out_degree": int(np.max(out_degrees)),
    }


# ======================================================================
# NetworkX-backed implementations (V5)
# ======================================================================


def _require_nx() -> None:
    """Raise if NetworkX is not installed."""
    if not _HAS_NX:
        raise ImportError(
            "networkx is required for this function. "
            "Install it with: pip install networkx"
        )


def adjacency_to_nx(
    adj_matrix: np.ndarray, node_names: List[str]
) -> "nx.DiGraph":
    """Convert adjacency matrix to a NetworkX ``DiGraph``.

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.

    Returns:
        ``nx.DiGraph`` instance.
    """
    _require_nx()
    G = nx.DiGraph()
    for i, name in enumerate(node_names):
        G.add_node(name)
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if adj_matrix[i, j] > 0.01:
                G.add_edge(node_names[i], node_names[j],
                           weight=float(adj_matrix[i, j]))
    return G


def nx_find_paths(
    G: "nx.DiGraph",
    source: str,
    target: str,
    max_length: int = 5,
) -> List[List[str]]:
    """Find all simple paths between *source* and *target* using NetworkX.

    Args:
        G: NetworkX directed graph.
        source: Source node name.
        target: Target node name.
        max_length: Maximum path length.

    Returns:
        List of paths (each path is a list of node names).
    """
    _require_nx()
    try:
        return list(nx.all_simple_paths(G, source, target, cutoff=max_length))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def nx_get_descendants(G: "nx.DiGraph", node: str) -> List[str]:
    """Get all descendants of *node* using NetworkX."""
    _require_nx()
    try:
        return sorted(nx.descendants(G, node))
    except nx.NodeNotFound:
        return []


def nx_get_ancestors(G: "nx.DiGraph", node: str) -> List[str]:
    """Get all ancestors of *node* using NetworkX."""
    _require_nx()
    try:
        return sorted(nx.ancestors(G, node))
    except nx.NodeNotFound:
        return []


def nx_topological_sort(
    adj_matrix: np.ndarray, node_names: List[str]
) -> List[str]:
    """Topological sort via NetworkX (returns original order on cycle)."""
    _require_nx()
    G = adjacency_to_nx(adj_matrix, node_names)
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        logger.warning("nx_topological_sort: graph has cycles, returning original order")
        return node_names


def compute_graph_metrics(
    adj_matrix: np.ndarray, node_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute graph-level metrics using NetworkX.

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: Optional node names (unused by NetworkX matrix constructor).

    Returns:
        Dict with ``num_nodes``, ``num_edges``, ``density``, ``avg_degree``,
        ``is_dag``.
    """
    _require_nx()
    n = adj_matrix.shape[0]
    edge_count = int(np.sum(adj_matrix > 0.01))
    density = edge_count / (n * (n - 1)) if n > 1 else 0.0
    avg_degree = edge_count / n if n > 0 else 0.0

    G = nx.DiGraph(adj_matrix)
    is_dag_flag = nx.is_directed_acyclic_graph(G)

    return {
        "num_nodes": n,
        "num_edges": edge_count,
        "density": float(density),
        "avg_degree": float(avg_degree),
        "is_dag": is_dag_flag,
    }


# ======================================================================
# Fairness / proxy utilities (V5)
# ======================================================================


def find_proxy_variables(
    adj_matrix: np.ndarray,
    node_names: List[str],
    protected: str,
    target: str,
    max_length: int = 4,
) -> List[List[str]]:
    """Find proxy chains from *protected* attribute to *target*.

    Uses NetworkX to enumerate all simple paths of length up to
    *max_length* edges.  These paths represent potential indirect
    discriminatory pathways that bypass direct protected-attribute links.

    Args:
        adj_matrix: Square adjacency matrix.
        node_names: List of node names.
        protected: Name of the protected attribute node.
        target: Name of the target node.
        max_length: Maximum path length (default 4).

    Returns:
        List of paths (each path is a list of node names).
    """
    try:
        G = adjacency_to_nx(adj_matrix, node_names)
    except ImportError:
        logger.warning(
            "networkx not available; falling back to pure-NumPy path search"
        )
        return find_all_paths(adj_matrix, node_names, protected, target,
                              max_length=max_length)

    paths = nx_find_paths(G, protected, target, max_length=max_length)
    if not paths:
        logger.debug(
            "No proxy paths found from '%s' to '%s' (max_length=%d)",
            protected, target, max_length,
        )
    return paths

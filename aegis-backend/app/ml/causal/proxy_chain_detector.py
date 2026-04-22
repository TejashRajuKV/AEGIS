"""
Proxy Chain Detector — identifies proxy variable leakage paths in causal graphs.

When a sensitive attribute (e.g., race, gender) influences an outcome through
intermediate variables that are not themselves protected, those intermediates
act as *proxy variables*.  This module detects such paths and computes risk
scores indicating the severity of proxy-based discrimination.

Architecture
------------
::

    Sensitive → Proxy_1 → Proxy_2 → Outcome
                          ↗
    Sensitive ──→ Proxy_3 ┘

Each path from a sensitive node to the outcome (that does not contain
another sensitive node as an intermediate) is a potential proxy chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    _HAS_NETWORKX = False
    logger.warning("networkx is not installed. Proxy chain detection is unavailable.")


__all__ = ["ProxyChainDetector", "ProxyChainResult"]


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class ProxyChainResult:
    """Result of proxy chain detection in a causal graph.

    Attributes
    ----------
    chains : list of list of str
        Detected proxy chains.  Each chain is a list of node names forming
        a directed path from a sensitive node to the outcome.
    risk_scores : list of float
        Risk score for each chain, in [0, 1].  Higher values indicate
        more severe proxy leakage.
    total_chains : int
        Total number of proxy chains detected.
    max_risk : float
        Maximum risk score across all chains.
    avg_risk : float
        Average risk score across all chains.
    sensitive_nodes_found : list of str
        Sensitive nodes identified in the graph.
    recommendations : list of str
        Human-readable recommendations for mitigating detected proxy chains.
    """

    chains: List[List[str]] = field(default_factory=list)
    risk_scores: List[float] = field(default_factory=list)
    total_chains: int = 0
    max_risk: float = 0.0
    avg_risk: float = 0.0
    sensitive_nodes_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result to a dictionary."""
        chain_details = []
        for chain, risk in zip(self.chains, self.risk_scores):
            chain_details.append({
                "path": chain,
                "risk_score": risk,
                "length": len(chain),
            })
        return {
            "chains": chain_details,
            "total_chains": self.total_chains,
            "max_risk": self.max_risk,
            "avg_risk": round(self.avg_risk, 4),
            "sensitive_nodes_found": self.sensitive_nodes_found,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Default sensitive attribute keywords
# ---------------------------------------------------------------------------

_DEFAULT_SENSITIVE_KEYWORDS: List[str] = [
    "race", "gender", "sex", "age", "ethnicity", "religion",
    "disability", "national_origin", "marital_status", "pregnancy",
    "color", "creed", "orientation", "protected",
]


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class ProxyChainDetector:
    """Detects proxy variable chains in causal graphs.

    A proxy chain is a directed path from a sensitive attribute node to
    an outcome node that passes through one or more non-sensitive
    intermediate nodes.  These intermediates act as proxies, carrying
    the influence of the sensitive attribute to the outcome.

    Parameters
    ----------
    sensitive_keywords : list of str, optional
        Keywords used to identify sensitive attribute nodes by name.
        If ``None``, uses :data:`_DEFAULT_SENSITIVE_KEYWORDS`.
    default_outcome_keywords : list of str, optional
        Keywords used to identify the outcome node by name.
    max_chain_length : int, optional
        Maximum length of paths to search.  Default ``3``.
    """

    def __init__(
        self,
        sensitive_keywords: Optional[List[str]] = None,
        default_outcome_keywords: Optional[List[str]] = None,
        max_chain_length: int = 3,
    ) -> None:
        self.sensitive_keywords = sensitive_keywords or _DEFAULT_SENSITIVE_KEYWORDS
        self.outcome_keywords = default_outcome_keywords or [
            "outcome", "target", "label", "prediction", "score",
            "decision", "result", "output", "hire", "admit",
            "loan", "approval", "risk_score",
        ]
        self.max_chain_length = max_chain_length

        logger.info(
            "ProxyChainDetector initialised: %d sensitive keywords, "
            "max_chain_length=%d",
            len(self.sensitive_keywords),
            max_chain_length,
        )

    def detect(
        self,
        graph: Any,
        sensitive_nodes: Optional[List[str]] = None,
        outcome_node: Optional[str] = None,
    ) -> ProxyChainResult:
        """Detect proxy chains in a causal graph.

        Parameters
        ----------
        graph : networkx.DiGraph
            The causal graph to analyse.
        sensitive_nodes : list of str, optional
            Names of nodes identified as sensitive attributes.  If ``None``,
            nodes are identified by matching against ``sensitive_keywords``.
        outcome_node : str, optional
            Name of the outcome node.  If ``None``, inferred as the node
            with the highest in-degree.

        Returns
        -------
        ProxyChainResult
            Detection result with chains, risk scores, and recommendations.
        """
        if not _HAS_NETWORKX:
            logger.warning("networkx not installed; returning empty result.")
            return ProxyChainResult()

        if not isinstance(graph, nx.DiGraph):
            logger.warning("Input is not a networkx.DiGraph; returning empty result.")
            return ProxyChainResult()

        if graph.number_of_nodes() == 0:
            logger.info("Empty graph; returning empty result.")
            return ProxyChainResult()

        # Identify sensitive nodes
        if sensitive_nodes is None:
            sensitive_nodes = self._find_sensitive_nodes(graph)
        else:
            # Validate that named nodes exist in the graph
            sensitive_nodes = [
                n for n in sensitive_nodes if n in graph.nodes()
            ]

        if not sensitive_nodes:
            logger.info("No sensitive nodes found in the graph.")
            return ProxyChainResult()

        # Identify outcome node
        if outcome_node is None:
            outcome_node = self._find_outcome_node(graph)
        elif outcome_node not in graph.nodes():
            logger.warning(
                "Outcome node '%s' not in graph; inferring from structure.",
                outcome_node,
            )
            outcome_node = self._find_outcome_node(graph)

        if not outcome_node:
            logger.info("No outcome node identified.")
            return ProxyChainResult()

        logger.info(
            "Detecting proxy chains: %d sensitive nodes, outcome='%s'",
            len(sensitive_nodes), outcome_node,
        )

        # Find all proxy paths
        all_chains: List[List[str]] = []
        all_risks: List[float] = []

        for src in sensitive_nodes:
            paths = self.find_proxy_paths(
                graph, source=src, target=outcome_node,
                max_length=self.max_chain_length,
            )
            for path in paths:
                # Compute risk score based on path properties
                edge_data = self._get_edge_data(graph, path)
                risk = self.compute_proxy_risk(edge_data, len(path))
                all_chains.append(path)
                all_risks.append(risk)

        # Build result
        total_chains = len(all_chains)
        max_risk = max(all_risks) if all_risks else 0.0
        avg_risk = sum(all_risks) / total_chains if total_chains > 0 else 0.0

        recommendations = self.get_recommendations(
            [{"path": c, "risk_score": r} for c, r in zip(all_chains, all_risks)]
        )

        result = ProxyChainResult(
            chains=all_chains,
            risk_scores=all_risks,
            total_chains=total_chains,
            max_risk=max_risk,
            avg_risk=avg_risk,
            sensitive_nodes_found=sensitive_nodes,
            recommendations=recommendations,
        )

        logger.info(
            "Proxy chain detection complete: %d chains, max_risk=%.3f, avg_risk=%.3f",
            total_chains, max_risk, avg_risk,
        )
        return result

    def find_proxy_paths(
        self,
        graph: Any,
        source: str,
        target: str,
        max_length: int = 3,
    ) -> List[List[str]]:
        """Find all directed paths from source to target up to a maximum length.

        Only paths that do not pass through other sensitive nodes are
        considered proxy paths.

        Parameters
        ----------
        graph : networkx.DiGraph
            The causal graph.
        source : str
            Starting node (sensitive attribute).
        target : str
            Ending node (outcome).
        max_length : int, optional
            Maximum path length (number of nodes).  Default ``3``.

        Returns
        -------
        list of list of str
            List of paths, where each path is a list of node names.
        """
        if not _HAS_NETWORKX or not isinstance(graph, nx.DiGraph):
            return []

        if source not in graph or target not in graph:
            logger.debug(
                "Source '%s' or target '%s' not in graph.", source, target
            )
            return []

        sensitive_set = set(self.sensitive_keywords)
        paths: List[List[str]] = []

        # Enumerate all simple paths up to max_length
        try:
            for path in nx.all_simple_paths(graph, source, target, cutoff=max_length - 1):
                # A proxy path must have at least one intermediate node
                # and intermediates must NOT be sensitive
                if len(path) < 3:
                    continue

                intermediates = path[1:-1]
                intermediate_is_sensitive = any(
                    self._is_sensitive_name(node, sensitive_set)
                    for node in intermediates
                )

                if not intermediate_is_sensitive:
                    paths.append(list(path))

        except nx.NetworkXError as exc:
            logger.error("Error finding paths from '%s' to '%s': %s", source, target, exc)

        logger.debug(
            "Found %d proxy paths from '%s' to '%s' (max_length=%d)",
            len(paths), source, target, max_length,
        )
        return paths

    def compute_proxy_risk(
        self,
        edge_data: List[Dict[str, Any]],
        path_length: int,
    ) -> float:
        """Compute a risk score for a proxy chain.

        The risk score is a weighted combination of:
        - Average edge weight (stronger edges = higher risk)
        - Path length (longer chains = more indirect leakage = higher risk)
        - Edge weight variance (consistent weights = higher risk)

        Parameters
        ----------
        edge_data : list of dict
            Edge attributes for each edge in the path.
        path_length : int
            Number of nodes in the path.

        Returns
        -------
        float
            Risk score in [0, 1].
        """
        if not edge_data:
            return 0.0

        # Extract edge weights
        weights = []
        for ed in edge_data:
            w = ed.get("weight", 1.0)
            if isinstance(w, (int, float)):
                weights.append(float(w))
            else:
                weights.append(1.0)

        if not weights:
            return 0.0

        avg_weight = sum(weights) / len(weights)
        weight_variance = sum((w - avg_weight) ** 2 for w in weights) / len(weights)

        # Normalise components
        # avg_weight contribution: stronger edges mean more proxy leakage
        weight_risk = min(avg_weight, 1.0)

        # Path length contribution: longer chains mean more hidden leakage
        length_risk = min((path_length - 1) / 5.0, 1.0)

        # Variance contribution: low variance = consistent leakage = higher risk
        variance_risk = 1.0 - min(weight_variance * 10.0, 1.0)

        # Weighted combination
        risk = 0.4 * weight_risk + 0.3 * length_risk + 0.3 * variance_risk

        return round(min(max(risk, 0.0), 1.0), 4)

    def get_recommendations(
        self,
        proxy_chains: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable recommendations for mitigating proxy chains.

        Parameters
        ----------
        proxy_chains : list of dict
            List of proxy chain dictionaries, each containing at minimum
            ``path`` (list of str) and ``risk_score`` (float).

        Returns
        -------
        list of str
            Human-readable recommendation strings.
        """
        if not proxy_chains:
            return ["No proxy chains detected. The causal structure appears fair."]

        recommendations: List[str] = []
        high_risk_chains = [
            c for c in proxy_chains
            if c.get("risk_score", 0) >= 0.7
        ]
        medium_risk_chains = [
            c for c in proxy_chains
            if 0.3 <= c.get("risk_score", 0) < 0.7
        ]
        low_risk_chains = [
            c for c in proxy_chains
            if c.get("risk_score", 0) < 0.3
        ]

        if high_risk_chains:
            recommendations.append(
                f"HIGH RISK: {len(high_risk_chains)} proxy chain(s) detected with "
                f"strong indirect influence from sensitive attributes. Immediate "
                f"investigation recommended. Consider removing or transforming "
                f"the proxy variables along these paths."
            )

        if medium_risk_chains:
            recommendations.append(
                f"MEDIUM RISK: {len(medium_risk_chains)} proxy chain(s) detected. "
                f"Review the intermediate variables and consider applying "
                f"fairness constraints during model training."
            )

        if low_risk_chains:
            recommendations.append(
                f"LOW RISK: {len(low_risk_chains)} minor proxy chain(s) detected. "
                f"Monitor these paths over time and document the potential "
                f"indirect influence."
            )

        # Specific path-level recommendations
        for chain in high_risk_chains:
            path = chain.get("path", [])
            if len(path) >= 2:
                proxies = path[1:-1] if len(path) > 2 else []
                if proxies:
                    recommendations.append(
                        f"  -> Consider removing or decorrelating proxy variable(s): "
                        f"{', '.join(proxies)} (risk={chain.get('risk_score', 0):.2f})"
                    )

        if len(proxy_chains) > 0:
            recommendations.append(
                "General recommendation: Apply causal inference techniques "
                "(e.g., do-calculus, backdoor adjustment) to block proxy paths "
                "before training downstream models."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_sensitive_nodes(self, graph: "nx.DiGraph") -> List[str]:
        """Identify sensitive attribute nodes by name matching.

        Parameters
        ----------
        graph : networkx.DiGraph
            The causal graph to scan.

        Returns
        -------
        list of str
            Node names that match sensitive attribute keywords.
        """
        sensitive_set = set(self.sensitive_keywords)
        found: List[str] = []
        for node in graph.nodes():
            if self._is_sensitive_name(str(node), sensitive_set):
                found.append(str(node))
        return found

    def _find_outcome_node(self, graph: "nx.DiGraph") -> str:
        """Identify the outcome node.

        Strategy:
        1. Match against outcome keywords.
        2. Fall back to the node with the highest in-degree.

        Parameters
        ----------
        graph : networkx.DiGraph

        Returns
        -------
        str
            Name of the outcome node, or empty string if not found.
        """
        outcome_set = set(self.outcome_keywords)

        for node in graph.nodes():
            name_lower = str(node).lower().replace("_", " ").replace("-", " ")
            for keyword in outcome_set:
                if keyword in name_lower:
                    return str(node)

        # Fallback: highest in-degree
        max_in_deg = -1
        outcome = ""
        for node in graph.nodes():
            in_deg = graph.in_degree(node)
            if in_deg > max_in_deg:
                max_in_deg = in_deg
                outcome = str(node)
        return outcome

    @staticmethod
    def _is_sensitive_name(node: str, sensitive_set: Set[str]) -> bool:
        """Check if a node name matches any sensitive keyword.

        Parameters
        ----------
        node : str
            Node name to check.
        sensitive_set : set of str
            Set of sensitive keywords.

        Returns
        -------
        bool
            True if the node name contains any sensitive keyword.
        """
        name_lower = str(node).lower().replace("_", " ").replace("-", " ")
        return any(kw in name_lower for kw in sensitive_set)

    @staticmethod
    def _get_edge_data(
        graph: "nx.DiGraph",
        path: List[str],
    ) -> List[Dict[str, Any]]:
        """Extract edge attribute data along a path.

        Parameters
        ----------
        graph : networkx.DiGraph
        path : list of str
            Node path.

        Returns
        -------
        list of dict
            Edge attribute dictionaries for each edge in the path.
        """
        edge_data: List[Dict[str, Any]] = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if graph.has_edge(u, v):
                data = graph.get_edge_data(u, v)
                if data is not None:
                    edge_data.append(dict(data))
                else:
                    edge_data.append({})
            else:
                edge_data.append({})
        return edge_data

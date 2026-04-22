"""AEGIS Causal Scoring - Score causal edges using BIC and correlation.

Merged from:
  - V3: BIC-based scoring via the ``CausalScorer`` class.
  - V4: Multi-method correlation (pearson, spearman, kendall), partial
        correlation, and edge ranking.

Bug fix applied (V6): ``bic_with`` and ``bic_without`` variable names were
inverted in V3.  ``bic_without`` now correctly means the BIC *without* the
candidate parent, and ``bic_with`` means the BIC *with* it, so that
``bic_diff = bic_without - bic_with`` is positive when the edge improves
the model.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from app.utils.logger import get_logger

logger = get_logger("causal_scoring")


# ======================================================================
# BIC-based scoring (V3, bug-fixed)
# ======================================================================


class CausalScorer:
    """Score causal edges using Bayesian Information Criterion (BIC).

    BIC = -2 · log_likelihood + k · log(n)

    Lower BIC indicates a better model.  For each directed edge i → j we
    compare the BIC of node j's local model *with* and *without* i as a
    parent.  The improvement (positive value) is normalised into a
    ``[0, 1]`` confidence score.
    """

    def __init__(self, penalty_factor: float = 1.0):
        """Initialise scorer.

        Args:
            penalty_factor: Multiplier for the BIC penalty term
                (``k · log(n) · penalty_factor``).
        """
        self.penalty_factor = penalty_factor

    def compute_bic(
        self,
        X: np.ndarray,
        target_idx: int,
        parent_indices: List[int],
    ) -> float:
        """Compute BIC score for a variable given a set of parents.

        Args:
            X: Data matrix ``(n_samples, n_features)``.
            target_idx: Column index of the target variable.
            parent_indices: Column indices of parent variables.

        Returns:
            BIC score (lower is better).
        """
        n = X.shape[0]
        y = X[:, target_idx]

        if len(parent_indices) == 0:
            mean = np.mean(y)
            residuals = y - mean
            mse = np.mean(residuals ** 2)
            if mse < 1e-10:
                return 0.0
            ll = -n / 2 * np.log(2 * np.pi * mse) - n / 2
            k = 2  # mean and variance
            return -2 * ll + k * np.log(n) * self.penalty_factor

        parents_data = X[:, parent_indices]
        try:
            X_design = np.column_stack([np.ones(n), parents_data])
            beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
            residuals = y - X_design @ beta
            mse = np.mean(residuals ** 2)
            if mse < 1e-10:
                return 0.0
            ll = -n / 2 * np.log(2 * np.pi * mse) - n / 2
            k = len(parent_indices) + 2  # parents + intercept + variance
            return -2 * ll + k * np.log(n) * self.penalty_factor
        except np.linalg.LinAlgError:
            return float("inf")

    def score_all_edges(
        self,
        X: np.ndarray,
        node_names: List[str],
        adj_matrix: np.ndarray,
        method: str = "bic",
    ) -> List[Dict]:
        """Score all directed edges in *adj_matrix*.

        Args:
            X: Data matrix ``(n_samples, n_features)``.
            node_names: List of node names.
            adj_matrix: Adjacency matrix.
            method: Scoring method — ``"bic"`` or one of
                ``"pearson"``, ``"spearman"``, ``"kendall"``.

        Returns:
            List of scored edge dicts.
        """
        if method in ("pearson", "spearman", "kendall"):
            return self._score_edges_correlation(X, node_names, adj_matrix,
                                                  method)
        return self._score_edges_bic(X, node_names, adj_matrix)

    # ------------------------------------------------------------------
    # BIC scoring
    # ------------------------------------------------------------------

    def _score_edges_bic(
        self,
        X: np.ndarray,
        node_names: List[str],
        adj_matrix: np.ndarray,
    ) -> List[Dict]:
        """Score edges using BIC improvement (V3, bug-fixed).

        For edge i → j, ``bic_without`` is the BIC of j's model *without*
        i as a parent, and ``bic_with`` is the BIC *with* i included.
        ``bic_diff = bic_without - bic_with`` is positive when adding the
        edge improves the model.
        """
        n = len(node_names)
        scored_edges: List[Dict] = []

        for i in range(n):
            for j in range(n):
                if adj_matrix[i][j] == 0:
                    continue

                # Parents of node j (excluding i)
                parent_indices = [
                    k for k in range(n) if adj_matrix[k][j] != 0 and k != i
                ]

                # BIC *without* candidate parent i
                bic_without = self.compute_bic(X, j, parent_indices)
                # BIC *with* candidate parent i
                bic_with = self.compute_bic(X, j, parent_indices + [i])

                # Positive diff means the edge improves the model
                bic_diff = bic_without - bic_with
                confidence = min(1.0, max(0.0, bic_diff / np.log(X.shape[0])))

                scored_edges.append({
                    "source": node_names[i],
                    "target": node_names[j],
                    "weight": float(adj_matrix[i][j]),
                    "score": round(confidence, 4),
                    "bic_improvement": round(bic_diff, 4),
                    "method": "bic",
                })

        logger.info(
            f"BIC scoring: {len(scored_edges)} edges scored "
            f"(penalty_factor={self.penalty_factor})"
        )
        return scored_edges

    # ------------------------------------------------------------------
    # Correlation-based scoring (V4)
    # ------------------------------------------------------------------

    def _score_edges_correlation(
        self,
        X: np.ndarray,
        node_names: List[str],
        adj_matrix: np.ndarray,
        method: str = "spearman",
    ) -> List[Dict]:
        """Score edges using a correlation method (pearson/spearman/kendall).

        Args:
            X: Data matrix.
            node_names: List of node names.
            adj_matrix: Adjacency matrix.
            method: Correlation method.

        Returns:
            List of scored edge dicts.
        """
        n = len(node_names)
        scored_edges: List[Dict] = []

        for i in range(n):
            for j in range(n):
                if adj_matrix[i][j] == 0:
                    continue

                result = score_edge_correlation(X[:, i], X[:, j], method)
                scored_edges.append({
                    "source": node_names[i],
                    "target": node_names[j],
                    "weight": float(adj_matrix[i][j]),
                    "score": round(result["abs_correlation"], 4),
                    "correlation": round(result["correlation"], 4),
                    "p_value": round(result["p_value"], 6),
                    "significant": result["significant"],
                    "method": method,
                })

        logger.info(
            f"Correlation scoring ({method}): {len(scored_edges)} edges scored"
        )
        return scored_edges


# ======================================================================
# Standalone correlation functions (V4)
# ======================================================================


def score_edge_correlation(
    source: np.ndarray,
    target: np.ndarray,
    method: str = "spearman",
) -> Dict[str, float]:
    """Score a causal edge using correlation-based methods.

    Args:
        source: 1-D array of the source variable values.
        target: 1-D array of the target variable values.
        method: One of ``"pearson"``, ``"spearman"``, ``"kendall"``.

    Returns:
        Dict with ``correlation``, ``p_value``, ``abs_correlation``,
        ``significant``.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method == "pearson":
        corr, pval = stats.pearsonr(source, target)
    elif method == "spearman":
        corr, pval = stats.spearmanr(source, target)
    elif method == "kendall":
        corr, pval = stats.kendalltau(source, target)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return {
        "correlation": float(corr),
        "p_value": float(pval),
        "abs_correlation": float(abs(corr)),
        "significant": bool(pval < 0.05),
    }


def score_edge_partial_correlation(
    source: np.ndarray,
    target: np.ndarray,
    conditionals: np.ndarray,
) -> float:
    """Score an edge using partial correlation (controlling for confounders).

    Residualises *source* and *target* on *conditionals* via OLS, then
    returns the absolute Pearson correlation of the residuals.

    Args:
        source: 1-D array of the source variable.
        target: 1-D array of the target variable.
        conditionals: 2-D array ``(n_samples, n_cond)`` of conditioning
            variables.  May have zero columns.

    Returns:
        Absolute partial-correlation coefficient.
    """
    if conditionals.shape[1] == 0:
        return score_edge_correlation(source, target)["abs_correlation"]

    from sklearn.linear_model import LinearRegression

    reg_s = LinearRegression().fit(conditionals, source)
    reg_t = LinearRegression().fit(conditionals, target)
    res_s = source - reg_s.predict(conditionals)
    res_t = target - reg_t.predict(conditionals)

    corr = np.corrcoef(res_s, res_t)[0, 1]
    return float(abs(corr))


def rank_edges(
    edges: List[Dict],
    weight_key: str = "weight",
    top_k: Optional[int] = None,
) -> List[Dict]:
    """Rank causal edges by absolute weight / strength.

    Args:
        edges: List of edge dicts.
        weight_key: Key to sort by (default ``"weight"``).
        top_k: If given, return only the top *k* edges.

    Returns:
        Sorted (descending) list of edge dicts.
    """
    ranked = sorted(
        edges, key=lambda e: abs(e.get(weight_key, 0)), reverse=True
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked

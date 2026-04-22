"""PC Algorithm - Constraint-based causal discovery (AEGIS V6 Merged).

Implements the PC (Peter-Clark) algorithm for learning causal structure
from observational data using conditional independence tests.

Merged from:
  - V4: Fisher's Z-test, v-structure orientation
  - V3: Dual-mode continuous/categorical testing, DFS cycle check

Reference: Spirtes, Glymour, Scheines, "Causation, Prediction, and Search", 2000.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from app.utils.logger import get_logger


class ConditionalIndependenceTest(Enum):
    """Supported conditional independence test methods."""

    CHI_SQUARE = "chi_square"
    PARTIAL_CORRELATION = "partial_correlation"
    FISHER_Z = "fisher_z"


class PCAlgorithm:
    """PC algorithm for causal structure learning.

    Supports three conditional independence testing strategies:
      - ``fisher_z`` (V4): Fisher's Z-transformation on partial correlations
        derived from the full correlation matrix (fast, continuous data).
      - ``partial_correlation`` (V3): Residual-based partial correlation via
        linear regression (continuous data).
      - ``chi_square`` (V3): Chi-square test of independence via contingency
        tables (categorical / low-cardinality data).

    Steps:
      1. Start with a complete undirected graph.
      2. Remove edges via conditional independence tests (skeleton discovery).
      3. Orient v-structures (colliders).
      4. Verify DAG property via DFS cycle check.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set_size: int = 5,
        test: ConditionalIndependenceTest = ConditionalIndependenceTest.FISHER_Z,
    ):
        """Initialise PC algorithm.

        Args:
            alpha: Significance level for independence tests.
            max_cond_set_size: Maximum conditioning-set size to explore.
            test: Which conditional independence test to use.
        """
        self.logger = get_logger("pc_algorithm")
        self.alpha = alpha
        self.max_cond_set_size = max_cond_set_size
        self.test = test
        self.adj_matrix: Optional[np.ndarray] = None
        self.graph: Optional[np.ndarray] = None
        self.node_names: Optional[List[str]] = None
        self.sep_sets: Dict[Tuple[int, int], Set[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """Run PC algorithm on a numpy array (Fisher Z / partial-corr path).

        Args:
            data: ``(n_samples, n_features)`` array.
            feature_names: Optional feature names.

        Returns:
            Dict with ``adjacency_matrix``, ``edges``, ``feature_names``, ``method``.
        """
        n_samples, n_features = data.shape

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_features)]
        self.feature_names = feature_names
        self.node_names = feature_names

        # Pre-compute correlation matrix for Fisher Z
        if self.test == ConditionalIndependenceTest.FISHER_Z:
            scaler = StandardScaler()
            data_norm = scaler.fit_transform(data)
            corr_matrix = np.corrcoef(data_norm.T)
        else:
            corr_matrix = None

        # Phase 1: Build skeleton (undirected graph)
        skeleton = np.ones((n_features, n_features)) - np.eye(n_features)
        self.sep_sets = {}

        for cond_size in range(self.max_cond_set_size + 1):
            pairs_to_check = [
                (i, j)
                for i in range(n_features)
                for j in range(i + 1, n_features)
                if skeleton[i, j] > 0
            ]

            removed_count = 0
            for i, j in pairs_to_check:
                neighbors = [
                    k
                    for k in range(n_features)
                    if skeleton[i, k] > 0 and k != j
                ]
                if len(neighbors) < cond_size:
                    continue

                for cond_set in combinations(neighbors, cond_size):
                    cond_list = list(cond_set)
                    if self._is_conditionally_independent_fit(
                        data, corr_matrix, i, j, cond_list, n_samples,
                    ):
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                        self.sep_sets[(i, j)] = set(cond_list)
                        self.sep_sets[(j, i)] = set(cond_list)
                        removed_count += 1
                        break

            self.logger.info(
                f"PC fit depth {cond_size}: removed {removed_count} edges"
            )

            # Early stop if no node has enough neighbours for next depth
            if not any(
                sum(skeleton[k, :] > 0) > cond_size
                for k in range(n_features)
            ):
                break

        # Phase 2: Orient v-structures
        self.graph = skeleton.copy()
        for i in range(n_features):
            for j in range(n_features):
                if skeleton[i, j] > 0:
                    self.graph[i, j] = 1.0

        for k in range(n_features):
            parents_k = [i for i in range(n_features) if skeleton[i, k] > 0]
            for idx1 in range(len(parents_k)):
                for idx2 in range(idx1 + 1, len(parents_k)):
                    i, j = parents_k[idx1], parents_k[idx2]
                    if skeleton[i, j] == 0:  # non-adjacent
                        key = (i, j) if (i, j) in self.sep_sets else (j, i)
                        if key in self.sep_sets and k not in self.sep_sets[key]:
                            self.graph[i, k] = 1
                            self.graph[j, k] = 1
                            self.graph[k, i] = 0
                            self.graph[k, j] = 0

        edges = self._extract_edges(feature_names)
        is_dag = self._check_dag()

        self.logger.info(
            f"PC algorithm (fit): {n_features} nodes, {len(edges)} edges, "
            f"is_dag={is_dag}, test={self.test.value}"
        )

        return {
            "adjacency_matrix": self.graph.tolist(),
            "edges": edges,
            "feature_names": feature_names,
            "is_dag": is_dag,
            "method": "pc_algorithm",
            "test": self.test.value,
        }

    def discover(self, df: pd.DataFrame) -> Dict:
        """Run PC algorithm on a DataFrame (dual-mode continuous/categorical).

        Automatically detects whether the data is mostly continuous or
        categorical and picks the appropriate conditional independence test
        unless *test* was explicitly set to ``chi_square`` or
        ``partial_correlation``.

        Args:
            df: Input DataFrame.

        Returns:
            Dict with ``nodes``, ``edges``, ``num_edges``, ``adj_matrix``,
            ``is_dag``, ``method``.
        """
        self.node_names = list(df.columns)
        n = len(self.node_names)
        X = df.values.astype(np.float64)

        # Initialise complete undirected graph
        self.adj_matrix = np.ones((n, n), dtype=np.float64) - np.eye(n)
        self.sep_sets = {}

        # Auto-detect data type when test is still FISHER_Z (default)
        is_continuous = True
        if self.test == ConditionalIndependenceTest.FISHER_Z:
            for col in df.columns:
                if df[col].nunique() < 10 and df[col].dtype in [np.int64, int]:
                    is_continuous = False
                    break

        use_chi2 = (
            self.test == ConditionalIndependenceTest.CHI_SQUARE or not is_continuous
        )
        use_resid_corr = (
            self.test == ConditionalIndependenceTest.PARTIAL_CORRELATION
        )

        effective_test = (
            "chi_square" if use_chi2 else ("partial_correlation" if use_resid_corr else "fisher_z")
        )

        self.logger.info(
            f"PC discover: {n} nodes, test={effective_test}, "
            f"continuous={is_continuous}"
        )

        # Pre-compute correlation matrix if using Fisher Z
        corr_matrix = None
        if self.test == ConditionalIndependenceTest.FISHER_Z and is_continuous:
            scaler = StandardScaler()
            data_norm = scaler.fit_transform(X)
            corr_matrix = np.corrcoef(data_norm.T)

        # Phase 1: Skeleton discovery
        for depth in range(self.max_cond_set_size + 1):
            pairs_to_test: List[Tuple[int, int, List[int]]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not self._is_adjacent(i, j):
                        continue

                    for anchor in (i, j):
                        other = j if anchor == i else i
                        neighbours = [
                            k
                            for k in range(n)
                            if k != other
                            and (self.adj_matrix[anchor][k] != 0
                                 or self.adj_matrix[k][anchor] != 0)
                        ]
                        if len(neighbours) >= depth:
                            for cond_set in combinations(neighbours, depth):
                                pairs_to_test.append(
                                    (anchor, other, list(cond_set))
                                )

            removed_count = 0
            for i, j, cond_set in pairs_to_test:
                if not self._is_adjacent(i, j):
                    continue
                is_indep = self._is_conditionally_independent_discover(
                    X, corr_matrix, i, j, cond_set, use_chi2, use_resid_corr,
                )
                if is_indep:
                    self.adj_matrix[i][j] = 0.0
                    self.adj_matrix[j][i] = 0.0
                    self.sep_sets[(i, j)] = set(cond_set)
                    self.sep_sets[(j, i)] = set(cond_set)
                    removed_count += 1

            self.logger.info(
                f"PC discover depth {depth}: removed {removed_count} edges"
            )

            # Early stop
            has_enough = any(
                sum(
                    1
                    for k in range(n)
                    if k != i
                    and (self.adj_matrix[i][k] != 0 or self.adj_matrix[k][i] != 0)
                )
                > depth
                for i in range(n)
            )
            if not has_enough:
                break

        # Phase 2: Orient v-structures
        self.graph = self.adj_matrix.copy()

        for k in range(n):
            parents_k = [
                i for i in range(n)
                if self.adj_matrix[i][k] != 0 or self.adj_matrix[k][i] != 0
            ]
            for idx1 in range(len(parents_k)):
                for idx2 in range(idx1 + 1, len(parents_k)):
                    i, j = parents_k[idx1], parents_k[idx2]
                    if not (self.adj_matrix[i][j] != 0
                            or self.adj_matrix[j][i] != 0):
                        key = (i, j) if (i, j) in self.sep_sets else (j, i)
                        if key in self.sep_sets and k not in self.sep_sets[key]:
                            self.graph[i][k] = 1
                            self.graph[j][k] = 1
                            self.graph[k][i] = 0
                            self.graph[k][j] = 0

        # Build edge list
        edges: List[Dict] = []
        for i in range(n):
            for j in range(n):
                if self.graph[i][j] > 0:
                    edges.append({
                        "source": self.node_names[i],
                        "target": self.node_names[j],
                        "weight": float(self.graph[i][j]),
                        "score": 1.0,
                    })

        is_dag = self._check_dag()

        self.logger.info(
            f"PC algorithm (discover): {n} nodes, {len(edges)} edges, "
            f"is_dag={is_dag}, test={effective_test}"
        )

        return {
            "nodes": self.node_names,
            "edges": edges,
            "num_edges": len(edges),
            "adj_matrix": self.graph.tolist(),
            "is_dag": is_dag,
            "method": "pc_algorithm",
            "test": effective_test,
        }

    # ------------------------------------------------------------------
    # Conditional independence dispatch (fit path – numpy)
    # ------------------------------------------------------------------

    def _is_conditionally_independent_fit(
        self,
        data: np.ndarray,
        corr_matrix: Optional[np.ndarray],
        i: int,
        j: int,
        cond_set: List[int],
        n_samples: int,
    ) -> bool:
        """Dispatch CI test for the ``fit()`` code-path."""
        if self.test == ConditionalIndependenceTest.FISHER_Z:
            return self._fisher_z_test(corr_matrix, i, j, cond_set, n_samples)

        # Partial correlation via residuals
        xi, xj = data[:, i], data[:, j]
        if len(cond_set) == 0:
            _, p = pearsonr(xi, xj)
            return p > self.alpha
        try:
            cond_data = data[:, cond_set]
            reg_i = LinearRegression().fit(cond_data, xi)
            reg_j = LinearRegression().fit(cond_data, xj)
            r, p = pearsonr(xi - reg_i.predict(cond_data),
                            xj - reg_j.predict(cond_data))
            return p > self.alpha
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Conditional independence dispatch (discover path – DataFrame)
    # ------------------------------------------------------------------

    def _is_conditionally_independent_discover(
        self,
        X: np.ndarray,
        corr_matrix: Optional[np.ndarray],
        i: int,
        j: int,
        cond_set: List[int],
        use_chi2: bool,
        use_resid_corr: bool,
    ) -> bool:
        """Dispatch CI test for the ``discover()`` code-path."""
        if use_chi2:
            return self._chi2_test(X, i, j, cond_set)

        if self.test == ConditionalIndependenceTest.FISHER_Z and corr_matrix is not None:
            return self._fisher_z_test(corr_matrix, i, j, cond_set, X.shape[0])

        # Residual-based partial correlation
        xi, xj = X[:, i], X[:, j]
        if len(cond_set) == 0:
            _, p = pearsonr(xi, xj)
            return p > self.alpha
        try:
            cond_data = X[:, cond_set]
            reg_i = LinearRegression().fit(cond_data, xi)
            reg_j = LinearRegression().fit(cond_data, xj)
            r, p = pearsonr(xi - reg_i.predict(cond_data),
                            xj - reg_j.predict(cond_data))
            return p > self.alpha
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Individual CI tests
    # ------------------------------------------------------------------

    def _fisher_z_test(
        self,
        corr_matrix: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int],
        n_samples: int,
    ) -> bool:
        """Test conditional independence using Fisher's Z-test.

        Computes partial correlation recursively from the correlation matrix,
        applies Fisher's Z-transformation, and compares the resulting
        p-value against *alpha*.

        Returns ``True`` when X_i ⊥ X_j | X_S (independent).
        """
        if corr_matrix is None:
            raise ValueError("corr_matrix is required for Fisher Z test")

        if len(cond_set) == 0:
            r = corr_matrix[i, j]
        else:
            r = self._partial_correlation(corr_matrix, i, j, cond_set)

        # Fisher's Z transformation
        z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
        z_stat = abs(z) * np.sqrt(n_samples - len(cond_set) - 3)
        p_value = 2 * (1 - stats.norm.cdf(z_stat))

        return p_value > self.alpha

    def _partial_correlation(
        self,
        corr_matrix: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int],
    ) -> float:
        """Compute partial correlation recursively from the correlation matrix."""
        if len(cond_set) == 0:
            return float(corr_matrix[i, j])

        k = cond_set[-1]
        rest = cond_set[:-1]

        r_ij = self._partial_correlation(corr_matrix, i, j, rest)
        r_ik = self._partial_correlation(corr_matrix, i, k, rest)
        r_jk = self._partial_correlation(corr_matrix, j, k, rest)

        denom = np.sqrt((1 - r_ik ** 2) * (1 - r_jk ** 2)) + 1e-10
        return (r_ij - r_ik * r_jk) / denom

    def _chi2_test(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int],
    ) -> bool:
        """Test conditional independence using chi-square (categorical data).

        For conditioning sets of size > 0, falls back to assuming independence
        (exact multi-dimensional contingency-table test not implemented).
        """
        xi, xj = X[:, i], X[:, j]

        if len(cond_set) == 0:
            try:
                contingency = pd.crosstab(xi, xj)
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    return True
                _, p, _, _ = chi2_contingency(contingency)
                return p > self.alpha
            except Exception:
                return True

        # No exact stratified chi-square; assume dependent
        return False

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _is_adjacent(self, i: int, j: int) -> bool:
        """Check if nodes *i* and *j* are adjacent in the current graph."""
        if self.adj_matrix is None:
            return False
        return self.adj_matrix[i][j] != 0 or self.adj_matrix[j][i] != 0

    def _check_dag(self) -> bool:
        """Check if the current adjacency matrix represents a DAG via DFS.

        Three-colour DFS: 0 = unvisited, 1 = in-progress, 2 = done.
        A back-edge to an in-progress node indicates a cycle.
        """
        mat = self.graph if self.graph is not None else self.adj_matrix
        if mat is None:
            return False

        n = mat.shape[0]
        visited = [0] * n  # 0=unvisited, 1=in-progress, 2=done

        def _has_cycle(node: int) -> bool:
            visited[node] = 1
            for neighbour in range(n):
                if mat[node][neighbour] != 0:
                    if visited[neighbour] == 1:
                        return True
                    if visited[neighbour] == 0 and _has_cycle(neighbour):
                        return True
            visited[node] = 2
            return False

        for i in range(n):
            if visited[i] == 0 and _has_cycle(i):
                return False
        return True

    def _extract_edges(self, feature_names: List[str]) -> List[Dict]:
        """Extract directed edges from the adjacency matrix."""
        edges: List[Dict] = []
        if self.graph is None:
            return edges
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                if self.graph[i][j] > 0:
                    edges.append({
                        "source": feature_names[i],
                        "target": feature_names[j],
                        "weight": float(self.graph[i][j]),
                    })
        return edges

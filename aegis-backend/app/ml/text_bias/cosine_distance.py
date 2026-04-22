"""
CosineDistanceCalculator – measures cosine distance / similarity between
embedding vectors and provides statistical significance testing.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]


class CosineDistanceCalculator:
    """Compute cosine distance and similarity between embedding vectors.

    * ``distance = 1 − cosine_similarity``  (range [0, 2])
    * ``similarity`` in [-1, 1]
    """

    def __init__(self, eps: float = 1e-10) -> None:
        """
        Parameters
        ----------
        eps:
            Small constant to avoid division by zero.
        """
        self.eps = eps
        logger.info("CosineDistanceCalculator initialised (eps=%.1e)", eps)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute(
        self,
        emb_a: "np.ndarray",
        emb_b: "np.ndarray",
    ) -> float:
        """Cosine distance between two vectors.

        Returns
        -------
        float
            Distance in [0, 2].  0 = identical direction, 2 = opposite.
        """
        a = np.asarray(emb_a, dtype=np.float64)
        b = np.asarray(emb_b, dtype=np.float64)
        sim = self._cosine_sim(a, b)
        return float(1.0 - sim)

    def compute_similarity(
        self,
        emb_a: "np.ndarray",
        emb_b: "np.ndarray",
    ) -> float:
        """Cosine similarity between two vectors (range [-1, 1])."""
        a = np.asarray(emb_a, dtype=np.float64)
        b = np.asarray(emb_b, dtype=np.float64)
        return float(self._cosine_sim(a, b))

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------
    def compute_batch(
        self,
        embeddings_a: List["np.ndarray"],
        embeddings_b: List["np.ndarray"],
    ) -> List[float]:
        """Pairwise cosine distances between two aligned lists."""
        if len(embeddings_a) != len(embeddings_b):
            raise ValueError(
                f"List lengths differ: {len(embeddings_a)} vs {len(embeddings_b)}"
            )
        return [self.compute(a, b) for a, b in zip(embeddings_a, embeddings_b)]

    def compute_pairwise(
        self,
        embeddings: List["np.ndarray"],
    ) -> "np.ndarray":
        """Compute an NxN cosine distance matrix.

        Returns
        -------
        np.ndarray
            Shape ``(N, N)`` where ``result[i][j]`` is the distance
            between embeddings[i] and embeddings[j].
        """
        n = len(embeddings)
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                d = self.compute(embeddings[i], embeddings[j])
                mat[i][j] = d
                mat[j][i] = d
        return mat

    # ------------------------------------------------------------------
    # Statistical significance – permutation test
    # ------------------------------------------------------------------
    def is_significant(
        self,
        dist_a: List[float],
        dist_b: List[float],
        n_permutations: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> Tuple[bool, float]:
        """Two-sample permutation test for difference in distance distributions.

        Parameters
        ----------
        dist_a, dist_b:
            Lists of cosine distances from two conditions (e.g. stereotyped
            vs anti-stereotyped prompts).
        n_permutations:
            Number of random permutations.
        alpha:
            Significance threshold.

        Returns
        -------
        (is_significant, p_value)
        """
        rng = np.random.RandomState(seed)

        observed_diff = abs(float(np.mean(dist_a)) - float(np.mean(dist_b)))
        combined = np.array(dist_a + dist_b)
        n_a = len(dist_a)

        count_extreme = 0
        for _ in range(n_permutations):
            rng.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            perm_diff = abs(float(np.mean(perm_a)) - float(np.mean(perm_b)))
            if perm_diff >= observed_diff:
                count_extreme += 1

        p_value = (count_extreme + 1) / (n_permutations + 1)  # continuity correction
        is_sig = p_value < alpha

        logger.info(
            "Permutation test: observed_diff=%.4f, p=%.4f, significant=%s",
            observed_diff,
            p_value,
            is_sig,
        )
        return is_sig, float(p_value)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _cosine_sim(self, a: "np.ndarray", b: "np.ndarray") -> float:
        """Raw cosine similarity."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        denom = max(norm_a * norm_b, self.eps)
        return dot / denom

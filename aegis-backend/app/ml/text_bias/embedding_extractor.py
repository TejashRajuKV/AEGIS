"""
EmbeddingExtractor – extracts and pools embeddings from LLM embedding layers.

Supports both sentence-level and token-level extraction with multiple
pooling strategies (mean, max, cls).
"""

import logging
from enum import Enum
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]


class PoolingStrategy(str, Enum):
    """Supported pooling strategies for reducing token embeddings."""

    MEAN = "mean"
    MAX = "max"
    CLS = "cls"


class EmbeddingExtractor:
    """Extracts embeddings from an :class:`LLMWrapper` or raw vectors.

    Can work in two modes:

    * **sentence-level** – returns a single vector per input text.
    * **token-level** – returns a matrix of token vectors (when the underlying
      model supports it).
    """

    def __init__(self, pooling: str = "mean", normalize: bool = True) -> None:
        """
        Parameters
        ----------
        pooling:
            Default pooling strategy ('mean', 'max', 'cls').
        normalize:
            If True, L2-normalise output vectors to unit length.
        """
        self.pooling = PoolingStrategy(pooling.lower())
        self.normalize = normalize
        logger.info(
            "EmbeddingExtractor – pooling=%s, normalize=%s", self.pooling.value, normalize
        )

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def extract_embeddings(
        self,
        text_pair_a: Union[str, List[str]],
        text_pair_b: Union[str, List[str]],
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """Convenience: extract embeddings for two texts or two lists of texts.

        Returns
        -------
        (emb_a, emb_b) : tuple of numpy arrays (or lists of arrays for batch input)
        """
        is_batch_a = isinstance(text_pair_a, list)
        is_batch_b = isinstance(text_pair_b, list)

        if is_batch_a and is_batch_b:
            emb_a = np.array([self._to_vec(t) for t in text_pair_a])
            emb_b = np.array([self._to_vec(t) for t in text_pair_b])
        elif not is_batch_a and not is_batch_b:
            emb_a = self._to_vec(text_pair_a)
            emb_b = self._to_vec(text_pair_b)
        else:
            raise ValueError(
                "Both text_pair_a and text_pair_b must be the same type "
                "(str or List[str])."
            )
        return emb_a, emb_b

    def extract_from_llm(
        self,
        llm: object,
        text: str,
    ) -> "np.ndarray":
        """Extract embedding from an LLMWrapper instance.

        Delegates to ``llm.embed(text)`` and applies pooling / normalisation.
        """
        raw = llm.embed(text)  # type: ignore[union-attr]
        return self._normalise(self._pool(raw))

    def extract_batch_from_llm(
        self,
        llm: object,
        texts: List[str],
    ) -> List["np.ndarray"]:
        """Extract embeddings for multiple texts via an LLMWrapper."""
        raw_vecs = llm.embed_batch(texts)  # type: ignore[union-attr]
        return [self._normalise(self._pool(v)) for v in raw_vecs]

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------
    def pool_embeddings(
        self,
        token_embeddings: "np.ndarray",
        strategy: Optional[str] = None,
    ) -> "np.ndarray":
        """Reduce a token-level embedding matrix to a single vector.

        Parameters
        ----------
        token_embeddings:
            Shape ``(n_tokens, dim)`` or ``(dim,)``.
        strategy:
            Override pooling strategy for this call.
        """
        strat = PoolingStrategy((strategy or self.pooling.value).lower())
        vec = np.asarray(token_embeddings, dtype=np.float64)

        if vec.ndim == 1:
            # Already a single vector – nothing to pool
            return self._normalise(vec)

        if strat == PoolingStrategy.MEAN:
            pooled = np.mean(vec, axis=0)
        elif strat == PoolingStrategy.MAX:
            pooled = np.max(vec, axis=0)
        elif strat == PoolingStrategy.CLS:
            # CLS: take the first token's embedding
            pooled = vec[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {strat}")

        return self._normalise(pooled)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_vec(self, text: str) -> "np.ndarray":
        """Convert text to a unit vector using a simple hash embedding.
        This is used when no LLM is provided."""
        raw = self._simple_hash(text, dim=384)
        return self._normalise(np.array(raw, dtype=np.float64))

    @staticmethod
    def _simple_hash(text: str, dim: int = 384) -> List[float]:
        """Deterministic hash-based pseudo-embedding (no external dependency)."""
        import hashlib

        vec = [0.0] * dim
        for i in range(len(text) - 1):
            bigram = text[i : i + 2].encode("utf-8")
            h = hashlib.sha256(bigram).hexdigest()
            for j, ch in enumerate(h[:8]):
                idx = (i + ord(ch)) % dim
                vec[idx] += (int(ch, 16) - 7.5) / 7.5
        return vec

    def _pool(self, vec: "np.ndarray") -> "np.ndarray":
        """Apply configured pooling to a vector or matrix."""
        return self.pool_embeddings(vec)

    def _normalise(self, vec: "np.ndarray") -> "np.ndarray":
        """L2-normalise a vector to unit length (skip if already unit)."""
        if not self.normalize:
            return vec
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            logger.warning("Zero-norm vector encountered; returning as-is")
            return vec
        return vec / norm

    # ------------------------------------------------------------------
    # Dimension inspection
    # ------------------------------------------------------------------
    @staticmethod
    def infer_dim(embedding: "np.ndarray") -> int:
        """Return the embedding dimension of a vector or matrix."""
        arr = np.asarray(embedding)
        if arr.ndim == 1:
            return arr.shape[0]
        return arr.shape[1]

"""
TextBiasScorer – scores text embeddings for demographic bias levels.

Maps cosine distances to human-readable bias levels and produces
aggregate dataset summaries.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class BiasLevel(str, Enum):
    """Severity levels for bias scores."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    SEVERE = "SEVERE"

    def __lt__(self, other: object) -> bool:
        if isinstance(other, BiasLevel):
            order = list(BiasLevel)
            return order.index(self) < order.index(other)
        return NotImplemented


@dataclass
class BiasScore:
    """Result of scoring a single prompt pair."""

    cosine_distance: float
    normalized_score: float  # 0-100
    bias_level: BiasLevel
    label: str = ""


@dataclass
class DatasetBiasSummary:
    """Aggregate bias statistics for a full audit dataset."""

    total_pairs: int = 0
    mean_distance: float = 0.0
    median_distance: float = 0.0
    max_distance: float = 0.0
    min_distance: float = 0.0
    std_distance: float = 0.0
    bias_index: float = 0.0  # 0-100 aggregate
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    score_distribution: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# Thresholds for mapping cosine distance → bias level
_BIAS_THRESHOLDS = [
    (1.0, BiasLevel.SEVERE),
    (0.6, BiasLevel.HIGH),
    (0.3, BiasLevel.MEDIUM),
    (0.1, BiasLevel.LOW),
    (0.0, BiasLevel.NONE),
]


def _classify_bias(distance: float) -> BiasLevel:
    """Return the :class:`BiasLevel` for a given cosine distance."""
    for threshold, level in _BIAS_THRESHOLDS:
        if distance > threshold:
            return level
    return BiasLevel.NONE


def _normalize_score(distance: float) -> float:
    """Map cosine distance to a 0–100 normalised bias score.

    Uses a sigmoid-like mapping so that:
    - d ≈ 0 → score ≈ 0
    - d ≈ 0.5 → score ≈ 50
    - d ≈ 1.0 → score ≈ 90
    - d ≥ 1.5 → score ≈ 100
    """
    # Sigmoid: 100 / (1 + exp(-k * (d - midpoint)))
    k = 6.0
    midpoint = 0.5
    raw = 100.0 / (1.0 + __import__("math").exp(-k * (distance - midpoint)))
    return round(min(max(raw, 0.0), 100.0), 2)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class TextBiasScorer:
    """Scores text embeddings for bias using cosine distance metrics."""

    def __init__(self) -> None:
        logger.info("TextBiasScorer initialised")

    # ------------------------------------------------------------------
    # Single pair scoring
    # ------------------------------------------------------------------
    def score_pair(
        self,
        emb_a: "np.ndarray",
        emb_b: "np.ndarray",
        cosine_distance: Optional[float] = None,
    ) -> BiasScore:
        """Score a single pair of embeddings for bias.

        Parameters
        ----------
        emb_a, emb_b:
            Embedding vectors for the two prompts.
        cosine_distance:
            Pre-computed distance.  If None, computed from embeddings.

        Returns
        -------
        BiasScore with distance, normalised score, and severity level.
        """
        if cosine_distance is None:
            a = np.asarray(emb_a, dtype=np.float64)
            b = np.asarray(emb_b, dtype=np.float64)
            dot = float(np.dot(a, b))
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            sim = dot / max(na * nb, 1e-10)
            cosine_distance = float(1.0 - sim)

        level = _classify_bias(cosine_distance)
        normalized = _normalize_score(cosine_distance)

        return BiasScore(
            cosine_distance=round(cosine_distance, 6),
            normalized_score=normalized,
            bias_level=level,
            label=f"{level.value} (d={cosine_distance:.4f})",
        )

    # ------------------------------------------------------------------
    # Dataset-level summarisation
    # ------------------------------------------------------------------
    def score_dataset(
        self,
        results: List[Dict[str, Any]],
    ) -> DatasetBiasSummary:
        """Summarise bias scores across a full audit dataset.

        Parameters
        ----------
        results:
            List of dicts, each containing at least ``cosine_distance``
            and optionally ``category``.

        Returns
        -------
        DatasetBiasSummary with aggregate statistics.
        """
        if not results:
            return DatasetBiasSummary(
                recommendations=["No results to analyse. Run an audit first."]
            )

        distances = [r["cosine_distance"] for r in results]
        scores = [_normalize_score(d) for d in distances]

        arr = np.array(distances)
        summary = DatasetBiasSummary(
            total_pairs=len(distances),
            mean_distance=float(np.mean(arr)),
            median_distance=float(np.median(arr)),
            max_distance=float(np.max(arr)),
            min_distance=float(np.min(arr)),
            std_distance=float(np.std(arr)),
            bias_index=self.compute_bias_index(scores),
        )

        # Per-category breakdown
        cat_distances: Dict[str, List[float]] = {}
        for r in results:
            cat = r.get("category", "unknown")
            cat_distances.setdefault(cat, []).append(r["cosine_distance"])

        for cat, dists in cat_distances.items():
            cat_arr = np.array(dists)
            summary.per_category[cat] = {
                "count": len(dists),
                "mean": float(np.mean(cat_arr)),
                "median": float(np.median(cat_arr)),
                "max": float(np.max(cat_arr)),
                "std": float(np.std(cat_arr)),
                "bias_index": self.compute_bias_index([_normalize_score(d) for d in dists]),
            }

        # Score distribution
        for level in BiasLevel:
            count = sum(1 for d in distances if _classify_bias(d) == level)
            summary.score_distribution[level.value] = count

        # Recommendations
        summary.recommendations = self._generate_recommendations(summary)

        logger.info(
            "Dataset summary: mean_dist=%.4f, bias_index=%.1f, level=%s",
            summary.mean_distance,
            summary.bias_index,
            _classify_bias(summary.mean_distance).value,
        )
        return summary

    # ------------------------------------------------------------------
    # Aggregate index
    # ------------------------------------------------------------------
    @staticmethod
    def compute_bias_index(bias_scores: List[float]) -> float:
        """Compute a single aggregate bias index (0–100).

        Uses a weighted average that penalises extreme values more
        heavily: ``mean + 0.5 * p95 - 0.5 * p5``.
        """
        if not bias_scores:
            return 0.0
        arr = np.array(bias_scores)
        mean = float(np.mean(arr))
        p95 = float(np.percentile(arr, 95))
        p5 = float(np.percentile(arr, 5))
        index = mean + 0.5 * p95 - 0.5 * p5
        return round(min(max(index, 0.0), 100.0), 2)

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    def compare_models(
        self,
        model_a_results: List[Dict[str, Any]],
        model_b_results: List[Dict[str, Any]],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
    ) -> Dict[str, Any]:
        """Compare bias results from two different models.

        Returns
        -------
        Dict with ``winner``, ``summary_a``, ``summary_b``, ``differences``.
        """
        summary_a = self.score_dataset(model_a_results)
        summary_b = self.score_dataset(model_b_results)

        diff = summary_a.bias_index - summary_b.bias_index
        if abs(diff) < 2.0:
            verdict = "TIE – both models show similar bias levels"
        elif diff < 0:
            verdict = f"{model_a_name} is LESS biased (lower bias index)"
        else:
            verdict = f"{model_b_name} is LESS biased (lower bias index)"

        return {
            "winner": model_a_name if diff < 0 else model_b_name,
            "verdict": verdict,
            "index_difference": round(diff, 2),
            "summary_a": {
                "name": model_a_name,
                "bias_index": summary_a.bias_index,
                "mean_distance": summary_a.mean_distance,
                "total_pairs": summary_a.total_pairs,
            },
            "summary_b": {
                "name": model_b_name,
                "bias_index": summary_b.bias_index,
                "mean_distance": summary_b.mean_distance,
                "total_pairs": summary_b.total_pairs,
            },
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_recommendations(summary: DatasetBiasSummary) -> List[str]:
        """Generate actionable recommendations based on audit results."""
        recs: List[str] = []
        idx = summary.bias_index

        if idx < 10:
            recs.append("Bias levels are minimal. Continue monitoring.")
        elif idx < 30:
            recs.append("Low bias detected. Review per-category results for specific concerns.")
        elif idx < 60:
            recs.append("Moderate bias detected. Consider prompt engineering or fine-tuning debiasing.")
        else:
            recs.append("HIGH bias detected. Immediate action recommended: review training data, apply de-biasing techniques.")

        # Flag problematic categories
        for cat, stats in summary.per_category.items():
            if stats["bias_index"] > 50:
                recs.append(
                    f"Category '{cat}' shows elevated bias (index={stats['bias_index']:.1f}). "
                    f"Prioritise remediation."
                )

        # Distribution guidance
        severe_count = summary.score_distribution.get(BiasLevel.SEVERE.value, 0)
        high_count = summary.score_distribution.get(BiasLevel.HIGH.value, 0)
        if severe_count > 0:
            recs.append(
                f"{severe_count} pair(s) with SEVERE bias found. Investigate these prompts immediately."
            )
        if high_count > summary.total_pairs * 0.25:
            recs.append(
                f"{high_count}/{summary.total_pairs} pairs ({high_count / max(summary.total_pairs, 1) * 100:.0f}%) "
                f"show HIGH bias. Systematic de-biasing may be needed."
            )

        return recs

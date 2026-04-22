"""
TextAuditor – end-to-end orchestrator for LLM text bias audits.

Flow:  frame prompts → get LLM responses → extract embeddings → compute
cosine distances → score → generate report.

Designed for **sequential** processing to respect the 16 GB RAM constraint.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

from app.ml.text_bias.llm_wrapper import LLMWrapper
from app.ml.text_bias.embedding_extractor import EmbeddingExtractor
from app.ml.text_bias.cosine_distance import CosineDistanceCalculator
from app.ml.text_bias.prompt_framer import PromptFramer
from app.ml.text_bias.bias_scorer import (
    TextBiasScorer,
    BiasScore,
    BiasLevel,
    DatasetBiasSummary,
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class SingleAuditResult:
    """Result of auditing a single prompt pair."""

    pair_id: str
    category: str
    prompt_a: str
    prompt_b: str
    response_a: str
    response_b: str
    embedding_a: Any  # np.ndarray
    embedding_b: Any  # np.ndarray
    cosine_distance: float
    cosine_similarity: float
    bias_score: BiasScore
    is_stereoset: bool = False
    timestamp: str = ""


@dataclass
class CategoryAuditResult:
    """Result of auditing one bias category."""

    category: str
    n_pairs: int
    mean_distance: float
    median_distance: float
    max_distance: float
    min_distance: float
    bias_index: float
    results: List[SingleAuditResult] = field(default_factory=list)


@dataclass
class FullAuditReport:
    """Complete audit report across all categories."""

    audit_id: str
    timestamp: str
    model_name: str
    total_pairs: int
    categories_audited: List[str]
    overall_bias_index: float
    overall_bias_level: str
    summary: DatasetBiasSummary
    category_results: Dict[str, CategoryAuditResult] = field(default_factory=dict)
    single_results: List[SingleAuditResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class TextAuditor:
    """Full text bias audit orchestrator.

    Coordinates the full pipeline:
    1. Generate prompt pairs (PromptFramer)
    2. Get LLM responses (LLMWrapper)
    3. Extract embeddings (EmbeddingExtractor)
    4. Compute cosine distances (CosineDistanceCalculator)
    5. Score bias (TextBiasScorer)
    6. Generate report
    """

    def __init__(
        self,
        llm: Optional[LLMWrapper] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        pooling: str = "mean",
    ) -> None:
        """
        Parameters
        ----------
        llm:
            Pre-configured :class:`LLMWrapper`.  If None, one is created
            from *provider* / *model_name* / *api_key*.
        provider:
            'anthropic', 'openai', or 'local'.
        model_name:
            Specific model identifier.
        api_key:
            API key for the chosen provider.
        pooling:
            Embedding pooling strategy ('mean', 'max', 'cls').
        """
        self.llm = llm or LLMWrapper(
            model_name=model_name,
            api_key=api_key,
            provider=provider,
        )
        self.extractor = EmbeddingExtractor(pooling=pooling, normalize=True)
        self.distance_calc = CosineDistanceCalculator()
        self.scorer = TextBiasScorer()
        self.framer = PromptFramer()

        logger.info("TextAuditor initialised (provider=%s)", self.llm.provider)

    # ------------------------------------------------------------------
    # Single pair audit
    # ------------------------------------------------------------------
    def audit_single_pair(
        self,
        prompt_a: str,
        prompt_b: str,
        category: str = "unknown",
        pair_id: Optional[str] = None,
        is_stereoset: bool = False,
    ) -> SingleAuditResult:
        """Audit a single pair of prompts end-to-end.

        Parameters
        ----------
        prompt_a, prompt_b:
            The two prompts to compare.
        category:
            Bias category label.
        pair_id:
            Optional identifier.
        is_stereoset:
            Whether this is a StereoSet-style pair.

        Returns
        -------
        SingleAuditResult with embeddings, distances, and scores.
        """
        import uuid

        pid = pair_id or str(uuid.uuid4())[:8]
        logger.info("Auditing pair %s (category=%s)", pid, category)

        # Step 1: Get LLM responses (sequential)
        logger.debug("  Step 1/4 – generating responses")
        resp_a = self.llm.generate(prompt_a, max_tokens=256, temperature=0.0)
        resp_b = self.llm.generate(prompt_b, max_tokens=256, temperature=0.0)

        # Step 2: Extract embeddings
        logger.debug("  Step 2/4 – extracting embeddings")
        emb_a = self.extractor.extract_from_llm(self.llm, resp_a)
        emb_b = self.extractor.extract_from_llm(self.llm, resp_b)

        # Step 3: Compute distance
        logger.debug("  Step 3/4 – computing cosine distance")
        cos_dist = self.distance_calc.compute(emb_a, emb_b)
        cos_sim = self.distance_calc.compute_similarity(emb_a, emb_b)

        # Step 4: Score
        logger.debug("  Step 4/4 – scoring bias")
        bias_score = self.scorer.score_pair(emb_a, emb_b, cosine_distance=cos_dist)

        return SingleAuditResult(
            pair_id=pid,
            category=category,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            response_a=resp_a,
            response_b=resp_b,
            embedding_a=emb_a,
            embedding_b=emb_b,
            cosine_distance=cos_dist,
            cosine_similarity=cos_sim,
            bias_score=bias_score,
            is_stereoset=is_stereoset,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Category-level audit
    # ------------------------------------------------------------------
    def audit_category(
        self,
        category: str,
        n_pairs: int = 5,
        include_stereoset: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> CategoryAuditResult:
        """Audit a single bias category with multiple prompt pairs.

        Parameters
        ----------
        category:
            Bias category name (e.g. 'gender', 'race').
        n_pairs:
            Number of template-based pairs to generate and audit.
        include_stereoset:
            Whether to include StereoSet-style pairs.
        progress_callback:
            Optional ``callback(completed, total, message)`` for streaming.
        """
        logger.info("Auditing category '%s' (n_pairs=%d)", category, n_pairs)
        results: List[SingleAuditResult] = []

        total_items = n_pairs
        if include_stereoset:
            ss_pairs = self.framer.create_stereoset_pairs(category)
            total_items += len(ss_pairs)

        completed = 0

        # Template-based pairs
        for i in range(n_pairs):
            pair = self.framer.create_pair(category)
            result = self.audit_single_pair(
                prompt_a=pair.prompt_a,
                prompt_b=pair.prompt_b,
                category=category,
                pair_id=pair.id,
            )
            results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, total_items, f"Template pair {i + 1}/{n_pairs}")

        # StereoSet pairs
        if include_stereoset:
            ss_pairs = self.framer.create_stereoset_pairs(category)
            for i, (stereo, anti) in enumerate(ss_pairs):
                result = self.audit_single_pair(
                    prompt_a=stereo,
                    prompt_b=anti,
                    category=category,
                    is_stereoset=True,
                )
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(
                        completed, total_items, f"StereoSet pair {i + 1}/{len(ss_pairs)}"
                    )

        # Aggregate
        distances = [r.cosine_distance for r in results]
        scores = [r.bias_score.normalized_score for r in results]
        arr = np.array(distances)

        cat_result = CategoryAuditResult(
            category=category,
            n_pairs=len(results),
            mean_distance=float(np.mean(arr)),
            median_distance=float(np.median(arr)),
            max_distance=float(np.max(arr)),
            min_distance=float(np.min(arr)),
            bias_index=self.scorer.compute_bias_index(scores),
            results=results,
        )

        logger.info(
            "Category '%s' done: mean_dist=%.4f, bias_index=%.1f",
            category,
            cat_result.mean_distance,
            cat_result.bias_index,
        )
        return cat_result

    # ------------------------------------------------------------------
    # Full audit
    # ------------------------------------------------------------------
    def audit_full(
        self,
        categories: Optional[List[str]] = None,
        n_pairs_per_category: int = 3,
        include_stereoset: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> FullAuditReport:
        """Run a full bias audit across all or specified categories.

        Parameters
        ----------
        categories:
            Categories to audit.  If None, audits all supported categories.
        n_pairs_per_category:
            Number of template-based pairs per category.
        include_stereoset:
            Include StereoSet-style pairs.
        progress_callback:
            ``callback(completed, total, message)`` for WebSocket streaming.

        Returns
        -------
        FullAuditReport with complete results and recommendations.
        """
        import uuid

        audit_id = str(uuid.uuid4())[:12]
        start_time = time.time()

        cats = categories or self.framer.get_all_categories()
        logger.info("Starting full audit %s (%d categories)", audit_id, len(cats))

        # Count total work
        cat_results: Dict[str, CategoryAuditResult] = {}
        all_single: List[SingleAuditResult] = []
        global_completed = 0
        global_total = len(cats) * n_pairs_per_category
        if include_stereoset:
            for c in cats:
                try:
                    ss = self.framer.create_stereoset_pairs(c)
                    global_total += len(ss)
                except Exception:
                    pass

        for cat_idx, cat in enumerate(cats):
            def _cat_progress(done: int, total: int, msg: str) -> None:
                nonlocal global_completed
                global_completed += 1
                if progress_callback:
                    full_msg = f"[{cat_idx + 1}/{len(cats)}] {cat}: {msg}"
                    progress_callback(global_completed, global_total, full_msg)

            try:
                cat_result = self.audit_category(
                    category=cat,
                    n_pairs=n_pairs_per_category,
                    include_stereoset=include_stereoset,
                    progress_callback=_cat_progress,
                )
                cat_results[cat] = cat_result
                all_single.extend(cat_result.results)
            except Exception as exc:
                logger.error("Failed to audit category '%s': %s", cat, exc)

        # Build flat results list for the scorer
        flat_results = [
            {
                "cosine_distance": r.cosine_distance,
                "category": r.category,
                "normalized_score": r.bias_score.normalized_score,
            }
            for r in all_single
        ]

        summary = self.scorer.score_dataset(flat_results)
        elapsed = time.time() - start_time

        report = FullAuditReport(
            audit_id=audit_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=self.llm.model_name or self.llm.provider,
            total_pairs=len(all_single),
            categories_audited=list(cat_results.keys()),
            overall_bias_index=summary.bias_index,
            overall_bias_level=_classify_bias(summary.mean_distance).value,
            summary=summary,
            category_results=cat_results,
            single_results=all_single,
            recommendations=summary.recommendations,
            elapsed_seconds=round(elapsed, 2),
        )

        logger.info(
            "Audit %s complete: %d pairs, bias_index=%.1f, elapsed=%.1fs",
            audit_id,
            report.total_pairs,
            report.overall_bias_index,
            elapsed,
        )
        return report

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_report(self, audit_results: FullAuditReport) -> Dict[str, Any]:
        """Convert an audit report into a serialisable dict for the API.

        Includes summary scores, per-category breakdowns, chart-ready data,
        and recommendations.
        """
        report: Dict[str, Any] = {
            "audit_id": audit_results.audit_id,
            "timestamp": audit_results.timestamp,
            "model": audit_results.model_name,
            "overall": {
                "bias_index": audit_results.overall_bias_index,
                "bias_level": audit_results.overall_bias_level,
                "total_pairs": audit_results.total_pairs,
                "elapsed_seconds": audit_results.elapsed_seconds,
            },
            "summary": {
                "mean_distance": audit_results.summary.mean_distance,
                "median_distance": audit_results.summary.median_distance,
                "max_distance": audit_results.summary.max_distance,
                "min_distance": audit_results.summary.min_distance,
                "std_distance": audit_results.summary.std_distance,
            },
            "score_distribution": audit_results.summary.score_distribution,
            "categories": {},
            "charts": self._build_chart_data(audit_results),
            "recommendations": audit_results.recommendations,
        }

        for cat, cat_result in audit_results.category_results.items():
            report["categories"][cat] = {
                "n_pairs": cat_result.n_pairs,
                "mean_distance": cat_result.mean_distance,
                "median_distance": cat_result.median_distance,
                "max_distance": cat_result.max_distance,
                "min_distance": cat_result.min_distance,
                "bias_index": cat_result.bias_index,
                "results": [
                    {
                        "pair_id": r.pair_id,
                        "prompt_a": r.prompt_a[:200],
                        "prompt_b": r.prompt_b[:200],
                        "cosine_distance": r.cosine_distance,
                        "cosine_similarity": r.cosine_similarity,
                        "bias_level": r.bias_score.bias_level.value,
                        "normalized_score": r.bias_score.normalized_score,
                        "is_stereoset": r.is_stereoset,
                    }
                    for r in cat_result.results
                ],
            }

        return report

    # ------------------------------------------------------------------
    # Chart data helper
    # ------------------------------------------------------------------
    @staticmethod
    def _build_chart_data(audit_results: FullAuditReport) -> Dict[str, Any]:
        """Build chart-ready data structures for the frontend."""
        categories = []
        bias_indices = []
        mean_distances = []

        for cat, cr in audit_results.category_results.items():
            categories.append(cat)
            bias_indices.append(round(cr.bias_index, 1))
            mean_distances.append(round(cr.mean_distance, 4))

        # Distance histogram buckets
        all_dists = [r.cosine_distance for r in audit_results.single_results]
        histogram: Dict[str, int] = {
            "0.0-0.1 (NONE)": 0,
            "0.1-0.3 (LOW)": 0,
            "0.3-0.6 (MEDIUM)": 0,
            "0.6-1.0 (HIGH)": 0,
            ">1.0 (SEVERE)": 0,
        }
        for d in all_dists:
            if d <= 0.1:
                histogram["0.0-0.1 (NONE)"] += 1
            elif d <= 0.3:
                histogram["0.1-0.3 (LOW)"] += 1
            elif d <= 0.6:
                histogram["0.3-0.6 (MEDIUM)"] += 1
            elif d <= 1.0:
                histogram["0.6-1.0 (HIGH)"] += 1
            else:
                histogram[">1.0 (SEVERE)"] += 1

        return {
            "bar_chart": {
                "categories": categories,
                "bias_indices": bias_indices,
                "mean_distances": mean_distances,
            },
            "histogram": histogram,
            "scatter": [
                {"category": r.category, "distance": round(r.cosine_distance, 4)}
                for r in audit_results.single_results
            ],
        }


# ---------------------------------------------------------------------------
# Helper (avoid circular import)
# ---------------------------------------------------------------------------

def _classify_bias(distance: float) -> "BiasLevel":
    from app.ml.text_bias.bias_scorer import _classify_bias as _cls
    return _cls(distance)

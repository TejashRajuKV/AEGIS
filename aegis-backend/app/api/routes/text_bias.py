"""
Text Bias Audit API Routes — LLM text bias auditing endpoints.

Endpoints
----------
POST /api/text-bias/audit        – Run a text bias audit.
GET  /api/text-bias/status/{task_id}  – Get audit status.
"""

import logging
import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from collections import OrderedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["text-bias"])

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    from app.ml.text_bias.text_auditor import TextAuditor
    _HAS_AUDITOR = True
except ImportError as exc:
    TextAuditor = None  # type: ignore[assignment, misc]
    _HAS_AUDITOR = False
    logger.warning("TextAuditor import failed: %s", exc)


# ---------------------------------------------------------------------------
# In-memory audit task store
# ---------------------------------------------------------------------------
_audit_tasks: OrderedDict[str, Dict[str, Any]] = OrderedDict()
MAX_TASKS = 1000


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TextPairItem(BaseModel):
    """A single text pair for bias comparison."""

    prompt_a: str = Field(..., description="First prompt variant.")
    prompt_b: str = Field(..., description="Second prompt variant.")
    category: str = Field(
        default="custom",
        description="Bias category label.",
    )


class TextBiasAuditRequest(BaseModel):
    """Request body for running a text bias audit."""

    text_pairs: Optional[List[TextPairItem]] = Field(
        None,
        description="Specific text pairs to audit. If provided, takes priority.",
    )
    categories: Optional[List[str]] = Field(
        None,
        description="Bias categories to audit (e.g., gender, race, age).",
    )
    n_pairs_per_category: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of template pairs per category.",
    )
    include_stereoset: bool = Field(
        default=True,
        description="Whether to include StereoSet-style pairs.",
    )
    model_name: Optional[str] = Field(
        None,
        description="LLM model to audit.",
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider (anthropic, openai, local).",
    )


class TextBiasAuditResponse(BaseModel):
    """Response for a text bias audit submission."""

    task_id: str = Field(..., description="Audit task identifier.")
    status: str = Field(..., description="Initial task status.")
    message: str = Field(..., description="Status message.")


class TextBiasAuditStatusResponse(BaseModel):
    """Response for audit status queries."""

    task_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Helper: execute audit
# ---------------------------------------------------------------------------

def _execute_audit(
    text_pairs: Optional[List[TextPairItem]],
    categories: Optional[List[str]],
    n_pairs_per_category: int,
    include_stereoset: bool,
    model_name: Optional[str],
    provider: Optional[str],
) -> Dict[str, Any]:
    """Execute a text bias audit synchronously.

    Parameters
    ----------
    text_pairs : list of TextPairItem or None
        Specific pairs to audit.
    categories : list of str or None
        Categories to audit.
    n_pairs_per_category : int
        Number of pairs per category.
    include_stereoset : bool
        Include StereoSet pairs.
    model_name : str or None
        LLM model name.
    provider : str or None
        LLM provider.

    Returns
    -------
    dict
        Audit results as a serialisable dictionary.
    """
    start_time = time.time()

    if not _HAS_AUDITOR or TextAuditor is None:
        raise RuntimeError(
            "TextAuditor is not available. Check that all text_bias "
            "submodules are installed."
        )

    auditor = TextAuditor(
        provider=provider,
        model_name=model_name,
    )

    if text_pairs:
        # Audit specific pairs
        single_results = []
        for pair in text_pairs:
            result = auditor.audit_single_pair(
                prompt_a=pair.prompt_a,
                prompt_b=pair.prompt_b,
                category=pair.category,
            )
            single_results.append(result)

        # Build a simple report
        from app.ml.text_bias.bias_scorer import TextBiasScorer
        scorer = TextBiasScorer()
        flat = [
            {
                "cosine_distance": r.cosine_distance,
                "category": r.category,
                "normalized_score": r.bias_score.normalized_score,
            }
            for r in single_results
        ]
        summary = scorer.score_dataset(flat)

        report = {
            "audit_type": "custom_pairs",
            "total_pairs": len(single_results),
            "overall_bias_index": summary.bias_index,
            "overall_bias_level": summary.bias_level.value,
            "mean_distance": summary.mean_distance,
            "recommendations": summary.recommendations,
            "pairs": [
                {
                    "pair_id": r.pair_id,
                    "category": r.category,
                    "prompt_a": r.prompt_a[:200],
                    "prompt_b": r.prompt_b[:200],
                    "cosine_distance": r.cosine_distance,
                    "bias_level": r.bias_score.bias_level.value,
                }
                for r in single_results
            ],
        }
    else:
        # Full category audit
        full_report = auditor.audit_full(
            categories=categories,
            n_pairs_per_category=n_pairs_per_category,
            include_stereoset=include_stereoset,
        )
        report = auditor.generate_report(full_report)

    elapsed = time.time() - start_time
    report["elapsed_seconds"] = round(elapsed, 2)

    logger.info("Text bias audit completed in %.2fs", elapsed)
    return report


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/audit",
    response_model=TextBiasAuditResponse,
    summary="Run text bias audit",
    description="Submit a text bias audit for background execution.",
)
async def run_text_bias_audit(
    request: TextBiasAuditRequest,
) -> TextBiasAuditResponse:
    """Run a text bias audit on LLM responses.

    Accepts either specific text pairs or bias categories to audit.
    The audit is executed asynchronously.

    Returns a task ID that can be used to retrieve results.
    """
    if not _HAS_AUDITOR:
        raise HTTPException(
            status_code=503,
            detail="TextAuditor is not available. Check installation.",
        )

    task_id = str(uuid.uuid4())[:12]

    if len(_audit_tasks) >= MAX_TASKS:
        _audit_tasks.popitem(last=False)

    _audit_tasks[task_id] = {
        "task_id": task_id,
        "status": "running",
        "created_at": time.time(),
        "started_at": time.time(),
    }

    async def _run_task():
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                _execute_audit,
                request.text_pairs,
                request.categories,
                request.n_pairs_per_category,
                request.include_stereoset,
                request.model_name,
                request.provider,
            )

            _audit_tasks[task_id]["status"] = "completed"
            _audit_tasks[task_id]["results"] = results
            _audit_tasks[task_id]["completed_at"] = time.time()
            _audit_tasks[task_id]["elapsed_seconds"] = results.get(
                "elapsed_seconds", 0.0
            )

        except Exception as exc:
            logger.error("Text bias audit failed: %s", exc)
            _audit_tasks[task_id]["status"] = "failed"
            _audit_tasks[task_id]["error"] = str(exc)
            _audit_tasks[task_id]["completed_at"] = time.time()

    asyncio.create_task(_run_task())

    return TextBiasAuditResponse(
        task_id=task_id,
        status="running",
        message="Text bias audit submitted.",
    )


@router.get(
    "/status/{task_id}",
    response_model=TextBiasAuditStatusResponse,
    summary="Get audit status",
    description="Query the status and results of a text bias audit task.",
)
async def get_text_bias_status(
    task_id: str,
) -> TextBiasAuditStatusResponse:
    """Get the status of a text bias audit task.

    Parameters
    ----------
    task_id : str
        The audit task ID.

    Returns
    -------
    TextBiasAuditStatusResponse
        Task status and results (if completed).
    """
    task = _audit_tasks.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Audit task '{task_id}' not found.",
        )

    elapsed = None
    if task.get("started_at") is not None:
        end = task.get("completed_at") or time.time()
        elapsed = round(end - task["started_at"], 3)

    return TextBiasAuditStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        results=task.get("results"),
        error=task.get("error"),
        elapsed_seconds=elapsed,
    )

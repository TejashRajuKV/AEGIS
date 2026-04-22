"""
Autopilot API Routes — background ML fairness pipeline orchestration.

Endpoints
----------
POST /api/autopilot/start   – Start an autopilot pipeline run.
POST /api/autopilot/stop/{task_id}  – Stop a running autopilot task.
GET  /api/autopilot/status/{task_id}  – Get task status.
GET  /api/autopilot/results/{task_id} – Get completed task results.
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["autopilot"])

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    from app.services.task_queue import TaskQueue, TaskStatus
    _HAS_TASK_QUEUE = True
except ImportError as exc:
    TaskQueue = None  # type: ignore[assignment, misc]
    TaskStatus = None  # type: ignore[assignment, misc]
    _HAS_TASK_QUEUE = False
    logger.warning("TaskQueue import failed: %s", exc)

try:
    from app.pipeline.autopilot_pipeline import AutopilotPipeline
    _HAS_AUTOPILOT = True
except ImportError as exc:
    AutopilotPipeline = None  # type: ignore[assignment, misc]
    _HAS_AUTOPILOT = False
    logger.warning("AutopilotPipeline import failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level task queue singleton
# ---------------------------------------------------------------------------
_task_queue: Optional[TaskQueue] = None
_active_tasks: Dict[str, Dict[str, Any]] = {}


def _get_task_queue() -> TaskQueue:
    """Return or create the module-level TaskQueue singleton."""
    global _task_queue
    if _task_queue is None:
        if not _HAS_TASK_QUEUE:
            raise RuntimeError("TaskQueue is not available. Check installation.")
        _task_queue = TaskQueue(max_concurrent=1, default_priority=10)
    return _task_queue


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AutopilotStartRequest(BaseModel):
    """Request body for starting an autopilot run."""

    dataset: str = Field(
        ...,
        description="Dataset identifier or path.",
        examples=["compas", "adult_census"],
    )
    model: str = Field(
        default="logistic_regression",
        description="Model type to audit.",
        examples=["logistic_regression", "random_forest", "xgboost"],
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional pipeline configuration overrides.",
    )


class AutopilotStartResponse(BaseModel):
    """Response for a successful autopilot start."""

    task_id: str = Field(..., description="Unique task identifier.")
    status: str = Field(..., description="Initial task status.")
    message: str = Field(..., description="Human-readable status message.")


class AutopilotStatusResponse(BaseModel):
    """Response for task status queries."""

    task_id: str
    status: str
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None


class AutopilotResultsResponse(BaseModel):
    """Response for completed task results."""

    task_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: run autopilot pipeline
# ---------------------------------------------------------------------------

def _run_autopilot_pipeline(
    dataset: str,
    model: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the autopilot pipeline synchronously.

    This function is submitted to the TaskQueue and runs in a background
    thread/async context.

    Parameters
    ----------
    dataset : str
        Dataset identifier.
    model : str
        Model type to audit.
    config : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Pipeline results including fairness metrics, causal analysis,
        and recommendations.
    """
    start_time = time.time()
    logger.info(
        "Running autopilot pipeline: dataset=%s, model=%s",
        dataset, model,
    )

    if _HAS_AUTOPILOT and AutopilotPipeline is not None:
        try:
            pipeline = AutopilotPipeline(
                dataset=dataset,
                model_type=model,
                **config,
            )
            results = pipeline.run()
            elapsed = time.time() - start_time
            results["elapsed_seconds"] = round(elapsed, 2)
            logger.info("Autopilot pipeline completed in %.2fs", elapsed)
            return results
        except Exception as exc:
            logger.error("Autopilot pipeline failed: %s", exc)
            raise RuntimeError(f"Pipeline execution failed: {exc}") from exc
    else:
        # Fallback: simulated results when pipeline is unavailable
        logger.warning(
            "AutopilotPipeline unavailable; returning simulated results."
        )
        elapsed = time.time() - start_time
        return {
            "dataset": dataset,
            "model": model,
            "status": "completed",
            "warning": "Simulated results – AutopilotPipeline not available.",
            "bias_index": 0.0,
            "elapsed_seconds": round(elapsed, 2),
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/start",
    response_model=AutopilotStartResponse,
    summary="Start autopilot",
    description="Submit a new autopilot pipeline run for background execution.",
)
async def start_autopilot(
    request: AutopilotStartRequest,
    fastapi_request: Request,
) -> AutopilotStartResponse:
    """Start a new autopilot pipeline run.

    Accepts a dataset identifier, model type, and optional configuration.
    Returns a task ID for tracking progress.
    """
    try:
        queue = _get_task_queue()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    task_id = str(uuid.uuid4())[:12]

    # Store task metadata
    _active_tasks[task_id] = {
        "task_id": task_id,
        "dataset": request.dataset,
        "model": request.model,
        "config": request.config,
        "created_at": time.time(),
    }

    # Enqueue the pipeline execution
    await queue.enqueue(
        _run_autopilot_pipeline,
        request.dataset,
        request.model,
        request.config,
    )

    logger.info(
        "Autopilot started: task_id=%s, dataset=%s, model=%s",
        task_id, request.dataset, request.model,
    )

    return AutopilotStartResponse(
        task_id=task_id,
        status="pending",
        message=f"Autopilot pipeline queued for dataset '{request.dataset}' "
                f"with model '{request.model}'.",
    )


@router.post(
    "/stop/{task_id}",
    summary="Stop autopilot",
    description="Cancel a pending autopilot task.",
)
async def stop_autopilot(task_id: str) -> Dict[str, Any]:
    """Stop a running or pending autopilot task.

    Parameters
    ----------
    task_id : str
        The task ID to cancel.

    Returns
    -------
    dict
        Cancellation status.
    """
    try:
        queue = _get_task_queue()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    cancelled = await queue.cancel(task_id)

    if not cancelled:
        # Check if task exists
        status = queue.get_status(task_id)
        if status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Task '{task_id}' not found.",
            )
        if status.get("status") == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Task '{task_id}' is already running and cannot be cancelled.",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Task '{task_id}' cannot be cancelled (status: "
                   f"{status.get('status', 'unknown')}).",
        )

    logger.info("Autopilot stopped: task_id=%s", task_id)
    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": f"Task '{task_id}' has been cancelled.",
    }


@router.get(
    "/status/{task_id}",
    response_model=AutopilotStatusResponse,
    summary="Get autopilot status",
    description="Query the current status of an autopilot task.",
)
async def get_autopilot_status(task_id: str) -> AutopilotStatusResponse:
    """Get the status of an autopilot task.

    Parameters
    ----------
    task_id : str
        The task ID to query.

    Returns
    -------
    AutopilotStatusResponse
        Current task status and timing information.
    """
    try:
        queue = _get_task_queue()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    status = queue.get_status(task_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )

    return AutopilotStatusResponse(
        task_id=status["task_id"],
        status=status["status"],
        created_at=status.get("created_at"),
        started_at=status.get("started_at"),
        completed_at=status.get("completed_at"),
        elapsed_seconds=status.get("elapsed_seconds"),
        error=status.get("error"),
    )


@router.get(
    "/results/{task_id}",
    response_model=AutopilotResultsResponse,
    summary="Get autopilot results",
    description="Retrieve the results of a completed autopilot task.",
)
async def get_autopilot_results(task_id: str) -> AutopilotResultsResponse:
    """Get the results of a completed autopilot task.

    Parameters
    ----------
    task_id : str
        The task ID to retrieve results for.

    Returns
    -------
    AutopilotResultsResponse
        Task results or error information.
    """
    try:
        queue = _get_task_queue()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    status = queue.get_status(task_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )

    task_status = status.get("status", "")

    if task_status == "running":
        raise HTTPException(
            status_code=202,
            detail="Task is still running. Poll /status/ for updates.",
        )
    elif task_status == "pending":
        raise HTTPException(
            status_code=202,
            detail="Task is pending. Poll /status/ for updates.",
        )
    elif task_status == "cancelled":
        return AutopilotResultsResponse(
            task_id=task_id,
            status="cancelled",
            error="Task was cancelled by user.",
        )
    elif task_status == "failed":
        return AutopilotResultsResponse(
            task_id=task_id,
            status="failed",
            error=status.get("error", "Unknown error."),
        )

    # Task completed – get result
    result = queue.get_result(task_id)
    if result is None:
        return AutopilotResultsResponse(
            task_id=task_id,
            status=task_status,
            error="Result data is unavailable.",
        )

    # Augment with metadata
    metadata = _active_tasks.get(task_id, {})
    if isinstance(result, dict):
        result["task_id"] = task_id
        result["dataset"] = metadata.get("dataset", "")
        result["model"] = metadata.get("model", "")

    return AutopilotResultsResponse(
        task_id=task_id,
        status="completed",
        results=result,
    )

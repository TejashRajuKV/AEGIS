"""
Drift Monitoring API Routes — data drift detection and alerting.

Endpoints
----------
POST /api/drift/monitor    – Start drift monitoring on new data.
GET  /api/drift/alerts     – Get active drift alerts.
GET  /api/drift/status/{task_id} – Get monitoring task status.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["drift"])

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    from app.ml.drift.drift_ensemble import DriftEnsemble
    _HAS_ENSEMBLE = True
except ImportError as exc:
    DriftEnsemble = None  # type: ignore[assignment, misc]
    _HAS_ENSEMBLE = False
    logger.warning("DriftEnsemble import failed: %s", exc)

try:
    from app.ml.drift.drift_alert import DriftAlertManager
    _HAS_ALERT_MANAGER = True
except ImportError as exc:
    DriftAlertManager = None  # type: ignore[assignment, misc]
    _HAS_ALERT_MANAGER = False
    logger.warning("DriftAlertManager import failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_monitoring_tasks: Dict[str, Dict[str, Any]] = {}
_alert_manager: Optional[DriftAlertManager] = None


def _get_alert_manager() -> DriftAlertManager:
    """Return or create the module-level DriftAlertManager singleton."""
    global _alert_manager
    if _alert_manager is None:
        if not _HAS_ALERT_MANAGER:
            raise RuntimeError("DriftAlertManager is not available.")
        _alert_manager = DriftAlertManager()
    return _alert_manager


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class DriftMonitorRequest(BaseModel):
    """Request body for starting drift monitoring."""

    reference_data: List[List[float]] = Field(
        ...,
        description="Reference data samples (baseline distribution).",
    )
    new_data: List[List[float]] = Field(
        ...,
        description="New data samples to check for drift.",
    )
    feature_names: Optional[List[str]] = Field(
        None,
        description="Names of features for alert labelling.",
    )
    cusum_threshold: float = Field(
        default=5.0,
        ge=0.1,
        description="CUSUM detector threshold.",
    )
    wasserstein_threshold: float = Field(
        default=0.1,
        ge=0.01,
        description="Wasserstein distance threshold.",
    )


class DriftMonitorResponse(BaseModel):
    """Response for drift monitoring submission."""

    task_id: str
    status: str
    message: str


class DriftAlertItem(BaseModel):
    """A single drift alert."""

    id: str
    feature_name: str
    severity: str
    drift_magnitude: float
    detector_name: str
    recommendation: str
    timestamp: float
    acknowledged: bool
    resolved: bool
    details: str


class DriftAlertsResponse(BaseModel):
    """Response for drift alerts query."""

    total_alerts: int
    active_alerts: int
    alerts: List[DriftAlertItem]


class DriftStatusResponse(BaseModel):
    """Response for drift monitoring status queries."""

    task_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: run drift monitoring
# ---------------------------------------------------------------------------

def _run_drift_monitoring(
    task_id: str,
    reference_data: List[List[float]],
    new_data: List[List[float]],
    feature_names: Optional[List[str]],
    cusum_threshold: float,
    wasserstein_threshold: float,
) -> Dict[str, Any]:
    """Execute drift monitoring synchronously.

    Parameters
    ----------
    task_id : str
        Task identifier.
    reference_data : list of list of float
        Reference (baseline) data samples.
    new_data : list of list of float
        New data to check for drift.
    feature_names : list of str or None
        Feature names.
    cusum_threshold : float
        CUSUM threshold.
    wasserstein_threshold : float
        Wasserstein threshold.

    Returns
    -------
    dict
        Monitoring results with per-feature drift scores and alerts.
    """
    start_time = time.time()
    logger.info("Starting drift monitoring for task %s", task_id)

    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("NumPy is required for drift monitoring.")

    if not _HAS_ENSEMBLE:
        raise RuntimeError("DriftEnsemble is not available.")

    ref_array = np.asarray(reference_data, dtype=np.float64)
    new_array = np.asarray(new_data, dtype=np.float64)

    alert_mgr = _get_alert_manager()
    ensemble = DriftEnsemble(
        cusum_threshold=cusum_threshold,
        wasserstein_threshold=wasserstein_threshold,
        alert_manager=alert_mgr,
    )
    ensemble.fit(ref_array, feature_names=feature_names)

    # Detect drift per feature
    results_by_feature: Dict[str, Any] = {}
    drift_detected_any = False

    if new_array.ndim == 1:
        new_array = new_array.reshape(-1, 1)

    n_features = new_array.shape[1]
    feat_names = feature_names or [f"feature_{i}" for i in range(n_features)]

    for i in range(n_features):
        fname = feat_names[i] if i < len(feat_names) else f"feature_{i}"
        try:
            result = ensemble.detect(new_array[:, i], feature_name=fname)
            results_by_feature[fname] = result.to_dict()
            if result.drift_detected:
                drift_detected_any = True
        except Exception as exc:
            logger.warning("Drift detection failed for '%s': %s", fname, exc)
            results_by_feature[fname] = {
                "drift_detected": False,
                "error": str(exc),
            }

    # Get per-feature drift scores
    try:
        drift_scores = ensemble.get_feature_drift_scores(new_array)
    except Exception as exc:
        logger.warning("Feature drift scores failed: %s", exc)
        drift_scores = {}

    # Generate report
    report = ensemble.generate_report()
    active_alerts = alert_mgr.get_active_alerts()

    elapsed = time.time() - start_time

    results = {
        "task_id": task_id,
        "drift_detected": drift_detected_any,
        "n_features_checked": n_features,
        "per_feature_results": results_by_feature,
        "drift_scores": drift_scores,
        "ensemble_report": report,
        "active_alerts_count": len(active_alerts),
        "alerts": [a.to_dict() for a in active_alerts],
        "elapsed_seconds": round(elapsed, 3),
    }

    logger.info(
        "Drift monitoring complete: task_id=%s, drift_detected=%s, "
        "features=%d, elapsed=%.3fs",
        task_id, drift_detected_any, n_features, elapsed,
    )
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/monitor",
    response_model=DriftMonitorResponse,
    summary="Start drift monitoring",
    description="Run drift detection on new data against a reference baseline.",
)
async def start_drift_monitoring(
    request: DriftMonitorRequest,
) -> DriftMonitorResponse:
    """Start drift monitoring by comparing new data against a reference set.

    Parameters
    ----------
    request : DriftMonitorRequest
        Reference data, new data, and detection thresholds.

    Returns
    -------
    DriftMonitorResponse
        Task ID for tracking the monitoring results.
    """
    task_id = str(uuid.uuid4())[:12]

    _monitoring_tasks[task_id] = {
        "task_id": task_id,
        "status": "running",
        "created_at": time.time(),
        "started_at": time.time(),
    }

    try:
        results = _run_drift_monitoring(
            task_id=task_id,
            reference_data=request.reference_data,
            new_data=request.new_data,
            feature_names=request.feature_names,
            cusum_threshold=request.cusum_threshold,
            wasserstein_threshold=request.wasserstein_threshold,
        )

        _monitoring_tasks[task_id]["status"] = "completed"
        _monitoring_tasks[task_id]["results"] = results
        _monitoring_tasks[task_id]["completed_at"] = time.time()

        drift_detected = results.get("drift_detected", False)
        message = (
            f"Drift monitoring complete. Drift "
            f"{'detected' if drift_detected else 'not detected'} in "
            f"{results.get('n_features_checked', 0)} features."
        )

    except Exception as exc:
        logger.error("Drift monitoring failed for task %s: %s", task_id, exc)
        _monitoring_tasks[task_id]["status"] = "failed"
        _monitoring_tasks[task_id]["error"] = str(exc)
        _monitoring_tasks[task_id]["completed_at"] = time.time()
        message = f"Drift monitoring failed: {str(exc)}"

    return DriftMonitorResponse(
        task_id=task_id,
        status=_monitoring_tasks[task_id]["status"],
        message=message,
    )


@router.get(
    "/alerts",
    response_model=DriftAlertsResponse,
    summary="Get active drift alerts",
    description="Retrieve all active (unresolved) drift alerts.",
)
async def get_drift_alerts() -> DriftAlertsResponse:
    """Get all active drift alerts.

    Returns
    -------
    DriftAlertsResponse
        List of active alerts with severity and recommendations.
    """
    try:
        alert_mgr = _get_alert_manager()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    active = alert_mgr.get_active_alerts()
    total_history = alert_mgr.get_alert_history()

    alert_items = [
        DriftAlertItem(
            id=a.id,
            feature_name=a.feature_name,
            severity=a.severity.value,
            drift_magnitude=a.drift_magnitude,
            detector_name=a.detector_name,
            recommendation=a.recommendation,
            timestamp=a.timestamp,
            acknowledged=a.acknowledged,
            resolved=a.resolved,
            details=a.details,
        )
        for a in active
    ]

    return DriftAlertsResponse(
        total_alerts=len(total_history),
        active_alerts=len(active),
        alerts=alert_items,
    )


@router.get(
    "/status/{task_id}",
    response_model=DriftStatusResponse,
    summary="Get monitoring status",
    description="Query the status and results of a drift monitoring task.",
)
async def get_drift_status(
    task_id: str,
) -> DriftStatusResponse:
    """Get the status of a drift monitoring task.

    Parameters
    ----------
    task_id : str
        The monitoring task ID.

    Returns
    -------
    DriftStatusResponse
        Task status and results (if completed).
    """
    task = _monitoring_tasks.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Drift monitoring task '{task_id}' not found.",
        )

    return DriftStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        results=task.get("results"),
        error=task.get("error"),
    )

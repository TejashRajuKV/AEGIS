"""AEGIS Fairness Audit API Route."""
import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException
from app.models.schemas import FairnessAuditRequest, FairnessAuditResponse, FairnessMetricResult
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Bounded LRU audit cache ──────────────────────────────────────────────
# Bug 32 fix: cap at MAX_CACHE_SIZE entries, evict LRU when full.
# Bug 33 fix: use asyncio.Lock (not threading.Lock) to avoid deadlocks in
#             async route handlers that hold the lock across await points.
# Key format: "<dataset>_<model_type>_<target>_<model_version>"
_MAX_CACHE_SIZE = 200
_audit_cache: OrderedDict = OrderedDict()   # ordered so LRU eviction is O(1)
_cache_lock = asyncio.Lock()


def _evict_cache_if_full() -> None:
    """Evict the least-recently-used cache entry when at capacity."""
    while len(_audit_cache) >= _MAX_CACHE_SIZE:
        _audit_cache.popitem(last=False)  # remove oldest (LRU)


@router.post("/audit", response_model=FairnessAuditResponse)
async def run_fairness_audit(request: FairnessAuditRequest):
    """Run a fairness audit on a registered model and dataset."""
    # ── Cache check (skip if retrain=True) ────────────────────────────────
    retrain = getattr(request, "retrain", False)

    # Bug 34 fix: include a model_version in the cache key so that whenever
    # a model is retrained, the new version busts the old cached result.
    model_version = "v0"
    try:
        from app.services.model_registry import ModelRegistry
        reg = ModelRegistry()
        active = next(
            (v for v in reg.list_models() if v.get("name") == request.model_type and v.get("is_active")),
            None,
        )
        if active:
            model_version = active.get("version", "v0")
    except Exception:
        pass

    cache_key = (
        f"{request.dataset_name}_{request.model_type}_"
        f"{request.target_column}_{model_version}"
    )

    if not retrain:
        async with _cache_lock:
            cached = _audit_cache.get(cache_key)
            if cached is not None:
                # Touch the entry (mark as recently used) — maintains LRU order
                _audit_cache.move_to_end(cache_key)
        if cached is not None:
            logger.info("Returning cached audit for key '%s'", cache_key)
            return cached
    from app.data.dataset_loader import get_dataset_loader
    from app.ml.fairness.fairness_pipeline import FairnessPipeline

    try:
        # Load dataset
        loader = get_dataset_loader()
        df = loader.load_dataset(request.dataset_name)

        # Prepare features and labels
        feature_names = [c for c in df.columns if c != request.target_column]
        X = df[feature_names].values

        # Get labels
        y = df[request.target_column].values
        if len(np.unique(y)) > 2:
            y_binary = (y == np.unique(y)[-1]).astype(float)
        else:
            y_binary = y.astype(float)

        # Generate predictions using a simple model for the audit
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # Encode string labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y_encoded)
        y_pred = model.predict(X_scaled).astype(float)

        # Collect sensitive attributes
        sensitive_attrs = {}
        for attr in request.sensitive_features:
            if attr in df.columns:
                sensitive_attrs[attr] = df[attr].values

        if not sensitive_attrs:
            raise HTTPException(
                status_code=400,
                detail=f"No sensitive attributes found. Requested: {request.sensitive_features}, "
                       f"Available: {list(df.columns)}",
            )

        # Run audit pipeline
        pipeline = FairnessPipeline()
        result = pipeline.audit_dataset(
            df=df,
            target_column=request.target_column,
            sensitive_attributes=request.sensitive_features,
        )

        # Format response
        metrics = []
        if "metrics" in result:
            for attr_name, metric_list in result["metrics"].items():
                if isinstance(metric_list, list):
                    for m in metric_list:
                        if isinstance(m, dict):
                            metrics.append(FairnessMetricResult(
                                metric_name=m.get("metric_name", ""),
                                value=m.get("overall_value", m.get("value", 0.0)),
                                threshold=m.get("threshold", 0.1),
                                is_fair=m.get("is_fair", True),
                                gap=m.get("gap", 0.0),
                            ))

        # If pipeline returned a different format, build metrics from overall results
        if not metrics and result.get("metrics"):
            raw_metrics = result["metrics"]
            if isinstance(raw_metrics, dict):
                for metric_name, value in raw_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics.append(FairnessMetricResult(
                            metric_name=metric_name,
                            value=float(value),
                            threshold=0.1,
                            is_fair=abs(float(value)) < 0.1,
                        ))

        accuracy = float(np.mean(y_pred == y_encoded))

        response = FairnessAuditResponse(
            dataset_name=request.dataset_name,
            model_type=request.model_type,
            accuracy=accuracy,
            metrics=metrics,
            overall_fair=result.get("overall_fair", True),
            recommendations=result.get("recommendations", []),
        )

        # Store in cache (Bug 32 fix: evict LRU if at capacity)
        async with _cache_lock:
            _evict_cache_if_full()
            _audit_cache[cache_key] = response
            _audit_cache.move_to_end(cache_key)  # mark as most recently used
        logger.info("Audit cached under key '%s' (%d entries total)", cache_key, len(_audit_cache))

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fairness audit failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Metrics Catalog ──────────────────────────────────────────────────

@router.get("/metrics", summary="List available fairness metrics")
async def list_fairness_metrics() -> Dict[str, Any]:
    """
    Describe all fairness metrics AEGIS can compute.
    Used by the frontend to populate metric selector dropdowns.
    """
    return {
        "metrics": [
            {
                "id": "demographic_parity_difference",
                "name": "Demographic Parity Difference",
                "description": "max(P(Ŷ=1|A=a)) − min(P(Ŷ=1|A=a)) across groups",
                "ideal_value": 0.0,
                "threshold": 0.10,
            },
            {
                "id": "demographic_parity_ratio",
                "name": "Demographic Parity Ratio",
                "description": "min_group_rate / max_group_rate (1.0 = perfect parity)",
                "ideal_value": 1.0,
                "threshold": 0.80,
            },
            {
                "id": "equalized_odds_fpr_gap",
                "name": "Equalized Odds — FPR Gap",
                "description": "max(FPR_g) − min(FPR_g) across demographic groups",
                "ideal_value": 0.0,
                "threshold": 0.10,
            },
            {
                "id": "equalized_odds_fnr_gap",
                "name": "Equalized Odds — FNR Gap",
                "description": "max(FNR_g) − min(FNR_g) across demographic groups",
                "ideal_value": 0.0,
                "threshold": 0.10,
            },
            {
                "id": "expected_calibration_error",
                "name": "Expected Calibration Error (ECE)",
                "description": "Weighted avg gap between predicted probability and observed frequency",
                "ideal_value": 0.0,
                "threshold": 0.05,
            },
        ]
    }


# ── Cache Management ─────────────────────────────────────────────────

@router.get("/audit/cache", summary="List cached audit keys")
async def list_cached_audits() -> Dict[str, Any]:
    """Return all audit cache keys currently held in memory."""
    async with _cache_lock:
        keys = list(_audit_cache.keys())
    return {"cached_audits": keys, "count": len(keys), "max_size": _MAX_CACHE_SIZE}


@router.delete("/audit/cache", summary="Clear audit cache")
async def clear_audit_cache() -> Dict[str, Any]:
    """Evict all cached audit results (forces fresh computation on next call)."""
    async with _cache_lock:
        count = len(_audit_cache)
        _audit_cache.clear()
    logger.info("Audit cache cleared (%d entries evicted)", count)
    return {"cleared": count, "status": "ok"}


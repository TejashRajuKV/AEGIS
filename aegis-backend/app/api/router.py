"""AEGIS API Router - Main router aggregating all sub-routers."""

from fastapi import APIRouter

api_router = APIRouter(prefix="/api")

# Lazy imports to avoid circular dependencies
from app.api.routes import (
    autopilot,
    causal,
    code_fix,
    counterfactual,
    datasets,
    drift,
    fairness,
    health,
    models,
    text_bias,
    websocket,
)

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(fairness.router, prefix="/fairness", tags=["Fairness"])
api_router.include_router(causal.router, prefix="/causal", tags=["Causal"])
api_router.include_router(text_bias.router, prefix="/text-bias", tags=["Text Bias"])
api_router.include_router(drift.router, prefix="/drift", tags=["Drift"])
api_router.include_router(counterfactual.router, prefix="/counterfactual", tags=["Counterfactual"])
api_router.include_router(code_fix.router, prefix="/code-fix", tags=["Code Fix"])
api_router.include_router(autopilot.router, prefix="/autopilot", tags=["Autopilot"])
api_router.include_router(websocket.router, tags=["WebSocket"])

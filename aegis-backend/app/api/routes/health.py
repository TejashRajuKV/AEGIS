"""AEGIS Health Check API Route."""
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter
from app.config import settings

router = APIRouter()

_STARTUP_TIME = datetime.now(timezone.utc).isoformat()
_start_monotonic = time.monotonic()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic liveness check — used by load balancers and uptime monitors."""
    uptime_seconds = round(time.monotonic() - _start_monotonic, 2)
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "uptime_seconds": uptime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed readiness check — real module status, registered models, runtime info."""
    # Probe each module with a real import attempt
    module_probes = {
        "fairness": "app.ml.fairness.fairness_pipeline",
        "causal": "app.ml.causal.pc_algorithm",
        "drift": "app.ml.drift.cusum_detector",
        "rl_autopilot": "app.ml.rl.ppo_agent",
        "text_bias": "app.ml.text_bias.text_auditor",
        "counterfactual": "app.ml.neural.counterfactual_generator",
        "code_fix": "app.services.auto_fix_generator",
    }
    modules: Dict[str, str] = {}
    for name, import_path in module_probes.items():
        try:
            __import__(import_path)
            modules[name] = "available"
        except ImportError as exc:
            modules[name] = f"unavailable: {exc}"
        except Exception as exc:
            modules[name] = f"error: {exc}"

    # Model registry info (best-effort)
    registered_models: list = []
    try:
        from app.services.model_registry import ModelRegistry
        registry = ModelRegistry()
        registered_models = registry.list_models()
    except Exception:
        pass

    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "startup_time": _STARTUP_TIME,
        "current_time": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(time.monotonic() - _start_monotonic, 2),
        "python_version": sys.version,
        "platform": platform.system(),
        "registered_models": registered_models,
        "supported_datasets": getattr(settings, "supported_datasets", ["adult_census", "compas", "german_credit"]),
        "datasets_dir": str(settings.DATA_DIR),
        "modules": modules,
    }

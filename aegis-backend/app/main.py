"""
AEGIS V6 Ultimate - FastAPI Application Entry Point
=====================================================
Merged from:
  V1 - description text, CORS, events, logging setup
  V4 - create_app() factory, lifespan handler, init_db on startup
  V5 - root/health endpoints, TimingMiddleware
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.middleware import setup_middleware
from app.api.router import api_router


# ── Lifespan Handler ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # ── Startup ────────────────────────────────────────────────────
    # Ensure required directories exist
    settings.ensure_dirs()

    # Initialize logging
    from app.utils.logger import get_logger
    logger = get_logger("aegis")
    app_name = getattr(settings, "APP_NAME", "AEGIS")
    app_version = getattr(settings, "APP_VERSION", "6.0.0")
    logger.info(f"Starting {app_name} v{app_version}")

    # Initialize database
    from app.models.database import init_db
    await init_db()
    logger.info("Database initialized")

    yield

    # ── Shutdown ───────────────────────────────────────────────────
    logger.info("Shutting down AEGIS...")


# ── Application Factory ────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app_version = getattr(settings, "APP_VERSION", "6.0.0")

    app = FastAPI(
        title="AEGIS - AI Fairness Platform",
        description=(
            "Every team will tell you AI is biased. "
            "AEGIS is the only platform that fixes it - live, in real time, "
            "without touching the model, for any AI system on earth.\n\n"
            "Features: DAG-GNN causal discovery, PPO fairness autopilot, "
            "CUSUM+Wasserstein drift detection, CVAE counterfactuals, "
            "multi-modal text bias auditing, and LLM-powered auto-fix."
        ),
        version=app_version,
        lifespan=lifespan_handler,
    )

    # ── CORS ───────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Timing & Logging Middleware (X-Process-Time header) ────────
    setup_middleware(app)

    # ── Exception Handlers ─────────────────────────────────────────
    from app.exceptions import register_exception_handlers
    register_exception_handlers(app)

    # ── Root Endpoint ──────────────────────────────────────────────
    @app.get("/", tags=["Root"])
    def root():
        """AEGIS platform root - returns name, version, and description."""
        return {
            "name": getattr(settings, "APP_NAME", "AEGIS"),
            "version": app_version,
            "description": (
                "Every team will tell you AI is biased. "
                "AEGIS is the only platform that fixes it."
            ),
        }

    @app.get("/health", tags=["Health"])
    def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {"status": "healthy", "version": app_version}

    # ── API Routers ────────────────────────────────────────────────
    # All 11 routers are aggregated via api_router (/api prefix):
    #   health, datasets, models, fairness, causal, text_bias, drift,
    #   counterfactual, code_fix, autopilot, websocket
    app.include_router(api_router)

    return app


# ── Module-level app instance (for uvicorn, etc.) ─────────────────

app = create_app()

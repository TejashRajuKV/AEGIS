"""AEGIS Events - Application startup and shutdown handlers."""

import os
from pathlib import Path

from app.config import settings
from app.utils.logger import get_logger


def _verify_datasets() -> None:
    """Check that dataset CSV files exist and log their size. Warns — never raises."""
    logger = get_logger("aegis.startup")
    checks = {
        "adult_census":  settings.DATA_DIR / settings.DATASET_ADULT_CENSUS,
        "compas":        settings.DATA_DIR / settings.DATASET_COMPAS,
        "german_credit": settings.DATA_DIR / settings.DATASET_GERMAN_CREDIT,
    }
    for name, path in checks.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("Dataset '%s' OK — %s (%.2f MB)", name, path.name, size_mb)
        else:
            logger.warning(
                "Dataset '%s' NOT FOUND at %s — some endpoints may fail", name, path
            )


async def init_app():
    """Initialize application resources on startup."""
    logger = get_logger("aegis.startup")

    # Ensure directories exist
    for directory in [
        settings.UPLOAD_DIR,
        settings.CHECKPOINT_DIR,
        settings.LOG_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)

    logger.info("Directories created/verified")

    # Initialize database
    from app.models.database import init_db
    await init_db()
    logger.info("Database initialized")

    # Verify datasets exist (warns, never raises)
    _verify_datasets()

    logger.info(
        f"{settings.APP_NAME} v{settings.APP_VERSION} ready on "
        f"{settings.host}:{settings.port}"
    )


async def shutdown_app():
    """Cleanup application resources on shutdown."""
    logger = get_logger("aegis.shutdown")
    logger.info("Shutting down AEGIS...")

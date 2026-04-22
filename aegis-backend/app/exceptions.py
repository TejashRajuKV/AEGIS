"""
AEGIS V6 Ultimate - Custom Exception Hierarchy
================================================
Merged from:
  V1 - error_code field, PipelineRunningError (409), ResourceExhaustedError (503)
  V3 - CausalDiscoveryError, RLEnvironmentError, AutoFixError, FairnessAuditError
  V4 - register_exception_handlers() pattern, AEGISBaseError, to_dict() methods
"""

from typing import Any, Dict, Optional

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


# ── Base Exception ─────────────────────────────────────────────────

class AEGISBaseError(Exception):
    """Base exception for all AEGIS errors.

    Attributes:
        message: Human-readable error description.
        error_code: Machine-readable error identifier for programmatic handling.
        detail: Optional additional context about the error.
        status_code: HTTP status code to return for this error.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "AEGIS_ERROR",
        detail: Optional[str] = None,
        status_code: int = 500,
    ):
        self.message = message
        self.error_code = error_code
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the exception to a JSON-serializable dictionary."""
        result: Dict[str, Any] = {
            "error": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
        }
        if self.detail:
            result["detail"] = self.detail
        return result


# ── 404 - Not Found ────────────────────────────────────────────────

class ModelNotFoundError(AEGISBaseError):
    """Raised when a requested model is not found in the registry."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            error_code="MODEL_NOT_FOUND",
            detail="The specified model is not registered in AEGIS",
            status_code=404,
        )


class DatasetNotFoundError(AEGISBaseError):
    """Raised when a requested dataset is not available."""

    def __init__(self, dataset_name: str):
        super().__init__(
            message=f"Dataset '{dataset_name}' not found",
            error_code="DATASET_NOT_FOUND",
            detail="The specified dataset is not available",
            status_code=404,
        )


# ── 409 - Conflict ─────────────────────────────────────────────────

class PipelineRunningError(AEGISBaseError):
    """Raised when attempting to start an already-running pipeline."""

    def __init__(self, pipeline_id: str):
        super().__init__(
            message=f"Pipeline '{pipeline_id}' is already running",
            error_code="PIPELINE_RUNNING",
            detail="Cannot start a pipeline that is already in progress",
            status_code=409,
        )


# ── 422 - Unprocessable Entity ─────────────────────────────────────

class ValidationError(AEGISBaseError):
    """Raised when input validation fails."""

    def __init__(self, message: str = "Validation error"):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
        )


class FairnessAuditError(AEGISBaseError):
    """Raised when a fairness audit fails."""

    def __init__(self, message: str = "Fairness audit failed"):
        super().__init__(
            message=message,
            error_code="FAIRNESS_AUDIT_ERROR",
            detail="The fairness audit could not be completed",
            status_code=422,
        )


class CausalDiscoveryError(AEGISBaseError):
    """Raised when causal discovery fails."""

    def __init__(self, message: str = "Causal discovery failed"):
        super().__init__(
            message=message,
            error_code="CAUSAL_DISCOVERY_ERROR",
            detail="Causal structure learning encountered an error",
            status_code=422,
        )


class DriftDetectionError(AEGISBaseError):
    """Raised when drift detection fails."""

    def __init__(self, message: str = "Drift detection failed"):
        super().__init__(
            message=message,
            error_code="DRIFT_DETECTION_ERROR",
            detail="Drift monitoring encountered an error",
            status_code=422,
        )


class AutoFixError(AEGISBaseError):
    """Raised when auto-fix code generation fails."""

    def __init__(self, message: str = "Auto-fix generation failed"):
        super().__init__(
            message=message,
            error_code="AUTO_FIX_ERROR",
            detail="Code fix generation encountered an error",
            status_code=422,
        )


class TrainingError(AEGISBaseError):
    """Raised when model training fails."""

    def __init__(self, message: str = "Training failed"):
        super().__init__(
            message=message,
            error_code="TRAINING_ERROR",
            detail="Model training encountered an error",
            status_code=422,
        )


class RLEnvironmentError(AEGISBaseError):
    """Raised when the RL environment encounters an error."""

    def __init__(self, message: str = "RL environment error"):
        super().__init__(
            message=message,
            error_code="RL_ENVIRONMENT_ERROR",
            detail="Reinforcement learning environment error",
            status_code=422,
        )


# ── 500 - Internal Server Error ────────────────────────────────────

class BiasDetectionError(AEGISBaseError):
    """Raised when bias detection fails."""

    def __init__(self, message: str = "Bias detection failed"):
        super().__init__(
            message=message,
            error_code="BIAS_DETECTION_ERROR",
            status_code=500,
        )


class PipelineError(AEGISBaseError):
    """Raised when a pipeline encounters a runtime error."""

    def __init__(self, pipeline_name: str, message: str = "Pipeline error"):
        super().__init__(
            message=f"Pipeline '{pipeline_name}': {message}",
            error_code="PIPELINE_ERROR",
            status_code=500,
        )


class TextBiasError(AEGISBaseError):
    """Raised when text bias audit fails."""

    def __init__(self, message: str = "Text bias audit failed"):
        super().__init__(
            message=message,
            error_code="TEXT_BIAS_ERROR",
            status_code=500,
        )


class CodeFixError(AEGISBaseError):
    """Raised when code fix generation fails."""

    def __init__(self, message: str = "Code fix generation failed"):
        super().__init__(
            message=message,
            error_code="CODE_FIX_ERROR",
            status_code=500,
        )


# ── 503 - Service Unavailable ──────────────────────────────────────

class ResourceExhaustedError(AEGISBaseError):
    """Raised when system resources are exhausted (memory, GPU, etc.)."""

    def __init__(
        self,
        message: str = "System resources exhausted. Wait for current task to complete.",
    ):
        super().__init__(
            message=message,
            error_code="RESOURCE_EXHAUSTED",
            status_code=503,
        )


# ── Backwards Compatibility ────────────────────────────────────────

class AegisError(AEGISBaseError):
    """Alias for AEGISBaseError. Backwards compatibility with V1/V3 code."""

    def __init__(
        self,
        message: str,
        error_code: str = "AEGIS_ERROR",
        detail: str = "",
        status_code: int = 500,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            detail=detail or None,
            status_code=status_code,
        )


# ── Exception Handler Registration ─────────────────────────────────

def register_exception_handlers(app):
    """Register FastAPI exception handlers for all AEGIS errors.

    Catches:
      - AEGISBaseError (and all subclasses) -> structured JSON with error_code
      - ValueError -> 422 validation error
      - Exception -> generic 500 with debug info in debug mode
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.exception_handler(AEGISBaseError)
    async def aegis_error_handler(request: Request, exc: AEGISBaseError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                **exc.to_dict(),
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": str(exc),
                "error_code": "VALUE_ERROR",
                "status_code": 422,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": "HTTPException",
                "error_code": f"HTTP_{exc.status_code}",
                "detail": str(exc.detail),
                "status_code": exc.status_code,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "RequestValidationError",
                "error_code": "VALIDATION_ERROR",
                "detail": "Request body validation failed.",
                "errors": errors,
                "status_code": 422,
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "detail": str(exc) if app.debug else "An unexpected error occurred",
                "status_code": 500,
            },
        )

"""
Code Fix API Routes — automated bias mitigation code generation.

Endpoints
----------
POST /api/code-fix/generate  – Generate fix code from a bias report.
POST /api/code-fix/validate  – Validate generated fix code.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["code-fix"])

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    from app.services.auto_fix_generator import AutoFixGenerator
    _HAS_FIXER = True
except ImportError as exc:
    AutoFixGenerator = None  # type: ignore[assignment, misc]
    _HAS_FIXER = False
    logger.warning("AutoFixGenerator import failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton (Bug 18 fix: protected by a threading.Lock)
# ---------------------------------------------------------------------------
import threading
_fix_generator: Optional[AutoFixGenerator] = None
_fix_generator_lock = threading.Lock()


def _get_fix_generator() -> AutoFixGenerator:
    """Return or create the module-level AutoFixGenerator singleton.

    Bug 18 fix: uses a threading.Lock so concurrent requests can't create
    two instances simultaneously (double-checked locking pattern).
    """
    global _fix_generator
    if _fix_generator is None:
        with _fix_generator_lock:
            if _fix_generator is None:  # double-check after acquiring lock
                if not _HAS_FIXER:
                    raise RuntimeError("AutoFixGenerator is not available.")
                _fix_generator = AutoFixGenerator()
    return _fix_generator


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CodeFixGenerateRequest(BaseModel):
    """Request body for generating bias fix code."""

    bias_report: Dict[str, Any] = Field(
        ...,
        description="Bias report containing metrics, affected features, and severity.",
    )
    model_type: str = Field(
        default="sklearn",
        description="Type of model being fixed (sklearn, pytorch, tensorflow, xgboost).",
        examples=["sklearn", "pytorch"],
    )
    fix_type: Optional[str] = Field(
        None,
        description="Specific fix type (preprocessing, threshold, reweighting).",
    )


class CodeFixValidateRequest(BaseModel):
    """Request body for validating generated fix code."""

    code: str = Field(
        ...,
        description="Python code to validate for syntax correctness.",
    )


class CodeFixGenerateResponse(BaseModel):
    """Response for a code fix generation request."""

    fix_type: str
    code: str
    explanation: str
    expected_improvement: str
    imports_needed: List[str]
    is_valid_syntax: bool
    syntax_error: Optional[str] = None


class CodeFixValidateResponse(BaseModel):
    """Response for a code validation request."""

    is_valid: bool
    syntax_error: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=CodeFixGenerateResponse,
    summary="Generate fix code",
    description="Generate Python code to mitigate detected bias based on a bias report.",
)
async def generate_fix_code(
    request: CodeFixGenerateRequest,
) -> CodeFixGenerateResponse:
    """Generate Python fix code from a bias report.

    Uses the AutoFixGenerator to produce contextual, actionable mitigation
    code tailored to the bias characteristics described in the report.

    Parameters
    ----------
    request : CodeFixGenerateRequest
        Bias report and model type information.

    Returns
    -------
    CodeFixGenerateResponse
        Generated code, explanation, and metadata.
    """
    try:
        generator = _get_fix_generator()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        # Prepare the bias report dict
        bias_report = dict(request.bias_report)
        if request.fix_type:
            bias_report["_fix_hint"] = request.fix_type

        result = generator.generate_fix(
            bias_report=bias_report,
            model_type=request.model_type,
        )

        logger.info(
            "Fix code generated: fix_type=%s, is_valid=%s",
            result.fix_type,
            result.is_valid_syntax,
        )

        return CodeFixGenerateResponse(
            fix_type=result.fix_type,
            code=result.code,
            explanation=result.explanation,
            expected_improvement=result.expected_improvement,
            imports_needed=result.imports_needed,
            is_valid_syntax=result.is_valid_syntax,
            syntax_error=result.syntax_error,
        )

    except Exception as exc:
        logger.error("Fix code generation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Code generation failed: {str(exc)}",
        )


@router.post(
    "/validate",
    response_model=CodeFixValidateResponse,
    summary="Validate fix code",
    description="Validate that generated fix code has correct Python syntax.",
)
async def validate_fix_code(
    request: CodeFixValidateRequest,
) -> CodeFixValidateResponse:
    """Validate generated fix code for syntax correctness.

    Uses AST parsing to check that the code is syntactically valid Python.

    Parameters
    ----------
    request : CodeFixValidateRequest
        Code string to validate.

    Returns
    -------
    CodeFixValidateResponse
        Validation result with error details if applicable.
    """
    if not _HAS_FIXER:
        raise HTTPException(
            status_code=503,
            detail="AutoFixGenerator is not available.",
        )

    is_valid, error_msg = AutoFixGenerator.validate_fix_syntax(request.code)

    if is_valid:
        message = "Code is syntactically valid."
        logger.info("Code validation passed.")
    else:
        message = f"Syntax error: {error_msg}"
        logger.warning("Code validation failed: %s", error_msg)

    return CodeFixValidateResponse(
        is_valid=is_valid,
        syntax_error=error_msg,
        message=message,
    )

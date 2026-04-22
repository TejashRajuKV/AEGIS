"""
Counterfactual API Routes — generate and interpolate counterfactual explanations.

Endpoints
----------
POST /api/counterfactual/generate    – Generate counterfactuals for a sample.
POST /api/counterfactual/interpolate – Interpolate between originals and counterfactuals.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["counterfactual"])

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
try:
    from app.ml.neural.counterfactual_generator import CounterfactualGenerator
    _HAS_GENERATOR = True
except ImportError as exc:
    CounterfactualGenerator = None  # type: ignore[assignment, misc]
    _HAS_GENERATOR = False
    logger.warning("CounterfactualGenerator import failed: %s", exc)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CounterfactualGenerateRequest(BaseModel):
    """Request body for generating counterfactuals."""

    sample: List[float] = Field(
        ...,
        description="Original sample feature values.",
    )
    sensitive_attr: int = Field(
        ...,
        ge=0,
        description="Index of the sensitive attribute in the feature vector.",
    )
    original_value: Any = Field(
        default=0,
        description="Original value of the sensitive attribute.",
    )
    target_value: Any = Field(
        default=1,
        description="Target value to change the sensitive attribute to.",
    )
    n_samples: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of counterfactual variants to generate.",
    )
    feature_names: Optional[List[str]] = Field(
        None,
        description="Names of input features for explanations.",
    )


class CounterfactualInterpolateRequest(BaseModel):
    """Request body for latent space interpolation."""

    sample_a: List[float] = Field(
        ...,
        description="First sample (original).",
    )
    sample_b: List[float] = Field(
        ...,
        description="Second sample (counterfactual).",
    )
    n_steps: int = Field(
        default=10,
        ge=2,
        le=100,
        description="Number of interpolation steps.",
    )
    feature_names: Optional[List[str]] = Field(
        None,
        description="Feature names for output labelling.",
    )


class CounterfactualGenerateResponse(BaseModel):
    """Response for counterfactual generation."""

    original: List[float]
    counterfactual: List[float]
    sensitive_attribute: str
    original_value: str
    target_value: str
    feature_changes: Dict[str, Dict[str, float]]
    variants: Optional[List[Dict[str, Any]]] = None


class CounterfactualInterpolateResponse(BaseModel):
    """Response for latent space interpolation."""

    interpolation_steps: List[Dict[str, Any]]
    n_steps: int
    feature_names: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Helper: simulated counterfactual (when model is unavailable)
# ---------------------------------------------------------------------------

def _simulated_counterfactual(
    sample: List[float],
    sensitive_idx: int,
    original_value: Any,
    target_value: Any,
    feature_names: Optional[List[str]],
) -> Dict[str, Any]:
    """Generate a simulated counterfactual for demo/testing.

    Applies a small perturbation around the original sample, changing
    the sensitive attribute value and slightly adjusting other features.

    Parameters
    ----------
    sample : list of float
        Original sample.
    sensitive_idx : int
        Sensitive attribute index.
    original_value : Any
        Original value.
    target_value : Any
        Target value.
    feature_names : list of str or None
        Feature names.

    Returns
    -------
    dict
        Simulated counterfactual result.
    """
    import random

    random.seed(42)
    cf = list(sample)

    # Change the sensitive attribute
    if sensitive_idx < len(cf):
        cf[sensitive_idx] = float(target_value) if target_value is not None else 1.0 - cf[sensitive_idx]

    # Small perturbation on other features
    for i in range(len(cf)):
        if i != sensitive_idx:
            noise = random.gauss(0, 0.02)
            cf[i] = max(0.0, min(1.0, cf[i] + noise))

    # Compute changes
    changes: Dict[str, Dict[str, float]] = {}
    for i in range(len(sample)):
        diff = cf[i] - sample[i]
        if abs(diff) > 0.001:
            name = feature_names[i] if (feature_names and i < len(feature_names)) else f"feature_{i}"
            changes[name] = {
                "original": float(sample[i]),
                "counterfactual": float(cf[i]),
                "change": round(diff, 6),
            }

    attr_name = (
        feature_names[sensitive_idx]
        if (feature_names and sensitive_idx < len(feature_names))
        else f"feature_{sensitive_idx}"
    )

    return {
        "original": sample,
        "counterfactual": cf,
        "sensitive_attribute": attr_name,
        "original_value": str(original_value),
        "target_value": str(target_value),
        "feature_changes": changes,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=CounterfactualGenerateResponse,
    summary="Generate counterfactuals",
    description="Generate counterfactual explanations for a given sample.",
)
async def generate_counterfactuals(
    request: CounterfactualGenerateRequest,
) -> CounterfactualGenerateResponse:
    """Generate counterfactual explanations for a sample.

    Answers: "What would the outcome be if the sensitive attribute
    were different?"

    Parameters
    ----------
    request : CounterfactualGenerateRequest
        Sample data, sensitive attribute index, and generation parameters.

    Returns
    -------
    CounterfactualGenerateResponse
        Counterfactual result with feature-level change explanations.
    """
    if _HAS_GENERATOR and CounterfactualGenerator is not None:
        try:
            import numpy as np

            # Create a placeholder CVAE for demonstration.
            # In production, a trained CVAE model would be loaded from a
            # model registry or checkpoint.
            from app.ml.neural.conditional_vae import ConditionalVAE

            input_dim = len(request.sample)
            cvae = ConditionalVAE(
                input_dim=input_dim,
                latent_dim=8,
                condition_dim=1,
                hidden_dims=[32, 16],
            )
            generator = CounterfactualGenerator(
                cvae=cvae,
                feature_names=request.feature_names,
                device="cpu",
            )

            original_np = np.array(request.sample, dtype=np.float32)

            # Generate single counterfactual
            result = generator.generate_counterfactual(
                original_sample=original_np,
                sensitive_attribute_idx=request.sensitive_attr,
                original_value=request.original_value,
                target_value=request.target_value,
            )

            # Generate multiple variants if requested
            variants = None
            if request.n_samples > 1:
                multi_results = generator.generate_multiple(
                    original_sample=original_np,
                    sensitive_attribute_idx=request.sensitive_attr,
                    n_samples=request.n_samples,
                )
                variants = [r.to_dict() for r in multi_results]

            return CounterfactualGenerateResponse(
                original=result.original.tolist(),
                counterfactual=result.counterfactual.tolist(),
                sensitive_attribute=result.sensitive_attribute,
                original_value=result.original_value,
                target_value=result.target_value,
                feature_changes=result.feature_changes,
                variants=variants,
            )

        except Exception as exc:
            logger.error("Counterfactual generation failed: %s", exc)
            # Fall back to simulated result
            logger.info("Falling back to simulated counterfactual.")

    # Simulated counterfactual (fallback or when generator unavailable)
    sim_result = _simulated_counterfactual(
        sample=request.sample,
        sensitive_idx=request.sensitive_attr,
        original_value=request.original_value,
        target_value=request.target_value,
        feature_names=request.feature_names,
    )

    logger.info("Returning simulated counterfactual result.")

    return CounterfactualGenerateResponse(
        original=sim_result["original"],
        counterfactual=sim_result["counterfactual"],
        sensitive_attribute=sim_result["sensitive_attribute"],
        original_value=sim_result["original_value"],
        target_value=sim_result["target_value"],
        feature_changes=sim_result["feature_changes"],
    )


@router.post(
    "/interpolate",
    response_model=CounterfactualInterpolateResponse,
    summary="Interpolate in latent space",
    description="Linearly interpolate between two samples in feature space.",
)
async def interpolate_counterfactuals(
    request: CounterfactualInterpolateRequest,
) -> CounterfactualInterpolateResponse:
    """Interpolate between an original sample and a counterfactual.

    Generates intermediate points along a linear path between the two
    samples, useful for visualising the effect of changing a sensitive
    attribute.

    Parameters
    ----------
    request : CounterfactualInterpolateRequest
        Two samples and the number of interpolation steps.

    Returns
    -------
    CounterfactualInterpolateResponse
        List of interpolated samples with step metadata.
    """
    import numpy as np

    sample_a = np.array(request.sample_a, dtype=np.float64)
    sample_b = np.array(request.sample_b, dtype=np.float64)

    if sample_a.shape != sample_b.shape:
        raise HTTPException(
            status_code=400,
            detail="sample_a and sample_b must have the same dimensions.",
        )

    n_steps = request.n_steps
    n_features = len(sample_a)
    feat_names = request.feature_names or [f"feature_{i}" for i in range(n_features)]

    steps: List[Dict[str, Any]] = []
    for i in range(n_steps):
        alpha = i / (n_steps - 1) if n_steps > 1 else 0.0
        interpolated = (1.0 - alpha) * sample_a + alpha * sample_b

        # Compute feature values
        values = {}
        for j in range(n_features):
            name = feat_names[j] if j < len(feat_names) else f"feature_{j}"
            values[name] = round(float(interpolated[j]), 6)

        steps.append({
            "step": i,
            "alpha": round(alpha, 4),
            "values": values,
        })

    logger.info(
        "Counterfactual interpolation: %d steps, %d features",
        n_steps, n_features,
    )

    return CounterfactualInterpolateResponse(
        interpolation_steps=steps,
        n_steps=n_steps,
        feature_names=feat_names,
    )

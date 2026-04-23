"""
AEGIS Pydantic Schemas
======================
Shared request/response models for all API endpoints.
Defines the contract between frontend and backend.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Base Schemas ──────────────────────────────────────────────────

class ApiResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    message: str = ""
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ── Fairness Schemas ──────────────────────────────────────────────

class FairnessAuditRequest(BaseModel):
    """Request to run a fairness audit."""
    dataset_name: str = Field(..., description="Name of the dataset to audit")
    model_type: str = Field(default="logistic_regression", description="Model type to train")
    sensitive_features: List[str] = Field(
        default_factory=lambda: ["sex", "race"],
        description="List of sensitive attribute column names"
    )
    target_column: Optional[str] = Field(default=None, description="Target variable column (auto-detected if not provided)")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    retrain: bool = Field(default=False, description="Force retrain instead of returning cached result")


class FairnessMetricResult(BaseModel):
    """Single fairness metric result."""
    metric_name: str
    value: float
    threshold: float
    is_fair: bool
    gap: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class FairnessAuditResponse(BaseModel):
    """Complete fairness audit response."""
    dataset_name: str
    model_type: str
    accuracy: float
    metrics: List[FairnessMetricResult]
    overall_fair: bool
    recommendations: List[str] = Field(default_factory=list)


# ── Causal Discovery Schemas ──────────────────────────────────────

class CausalDiscoveryRequest(BaseModel):
    """Request for causal graph discovery."""
    dataset_name: str
    method: str = Field(default="dag_gnn", description="dag_gnn or pc")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_epochs: int = Field(default=300, ge=10, le=10000)


class CausalEdge(BaseModel):
    """A single causal edge in the discovered graph."""
    source: str
    target: str
    weight: float
    is_significant: bool = True


class ProxyChain(BaseModel):
    """A chain of proxy variables leading to a protected attribute."""
    chain: List[str]
    total_indirect_effect: float
    strength: str = Field(description="weak, moderate, or strong")


class CausalDiscoveryResponse(BaseModel):
    """Causal discovery results."""
    dataset_name: str
    method: str
    edges: List[CausalEdge]
    proxy_chains: List[ProxyChain]
    num_nodes: int
    num_edges: int
    adjacency_matrix: Optional[List[List[float]]] = None


# ── RL Autopilot Schemas ─────────────────────────────────────────

class AutopilotStartRequest(BaseModel):
    """Request to start the RL autopilot."""
    dataset: str = Field(..., description="Dataset identifier")
    model: str = Field(default="logistic_regression")
    config: Dict[str, Any] = Field(default_factory=dict)


class AutopilotStatusResponse(BaseModel):
    """Autopilot status response."""
    status: str = Field(description="idle, running, paused, completed, error")
    current_iteration: int = 0
    total_iterations: int = 0
    current_accuracy: float = 0.0
    fairness_metrics: Dict[str, float] = Field(default_factory=dict)
    pareto_improvements: int = 0
    goodhart_warnings: List[str] = Field(default_factory=list)
    best_reward: float = 0.0


# ── Drift Detection Schemas ──────────────────────────────────────

class DriftMonitorRequest(BaseModel):
    """Request to start drift monitoring (matches drift.py route)."""
    reference_data: List[List[float]] = Field(..., description="Reference data samples (baseline distribution).")
    new_data: List[List[float]] = Field(..., description="New data samples to check for drift.")
    feature_names: Optional[List[str]] = Field(None, description="Names of features for alert labelling.")
    cusum_threshold: float = Field(default=5.0, ge=0.1)
    wasserstein_threshold: float = Field(default=0.1, ge=0.01)


class DriftMonitorResponse(BaseModel):
    """Drift monitoring submission response."""
    task_id: str
    status: str
    message: str


# ── Text Bias Schemas ─────────────────────────────────────────────

class TextPairItem(BaseModel):
    """A single text pair for bias comparison."""
    prompt_a: str
    prompt_b: str
    category: str = "custom"


class TextBiasAuditRequest(BaseModel):
    """Request for text bias audit (matches text_bias.py route)."""
    text_pairs: Optional[List[TextPairItem]] = None
    categories: Optional[List[str]] = None
    n_pairs_per_category: int = Field(default=3, ge=1, le=20)
    include_stereoset: bool = True
    model_name: Optional[str] = None
    provider: Optional[str] = None


class TextBiasAuditResponse(BaseModel):
    """Text bias audit submission response."""
    task_id: str
    status: str
    message: str


# ── Counterfactual Schemas ────────────────────────────────────────

class CounterfactualGenerateRequest(BaseModel):
    """Request for counterfactual generation (matches counterfactual.py route)."""
    sample: List[float] = Field(..., description="Original sample feature values.")
    sensitive_attr: int = Field(..., ge=0, description="Index of the sensitive attribute.")
    original_value: Any = Field(default=0)
    target_value: Any = Field(default=1)
    n_samples: int = Field(default=5, ge=1, le=50)
    feature_names: Optional[List[str]] = None


class CounterfactualInterpolateRequest(BaseModel):
    """Request for latent space interpolation."""
    sample_a: List[float]
    sample_b: List[float]
    n_steps: int = Field(default=10, ge=2, le=100)
    feature_names: Optional[List[str]] = None



# ── Code Fix Schemas ──────────────────────────────────────────────

class CodeFixGenerateRequest(BaseModel):
    """Request to generate bias fix code (matches code_fix.py route)."""
    bias_report: Dict[str, Any]
    model_type: str = Field(default="sklearn")
    fix_type: Optional[str] = None


class CodeFixGenerateResponse(BaseModel):
    """Generated code fix response."""
    fix_type: str
    code: str
    explanation: str
    expected_improvement: str
    imports_needed: List[str]
    is_valid_syntax: bool
    syntax_error: Optional[str] = None


# ── WebSocket Schemas ────────────────────────────────────────────

class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None
    pipeline_id: Optional[str] = None


# ── Pipeline Schemas ──────────────────────────────────────────────

class PipelineStep(BaseModel):
    """Single pipeline step."""
    step_name: str
    status: str = Field(description="pending, running, completed, error")
    duration_secs: Optional[float] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineStatus(BaseModel):
    """Full pipeline execution status."""
    pipeline_id: str
    pipeline_type: str
    status: str = Field(description="pending, running, completed, error")
    steps: List[PipelineStep] = Field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_duration_secs: Optional[float] = None


# ── Model Registry Schemas ────────────────────────────────────────

class ModelRegisterRequest(BaseModel):
    """Request to register a model."""
    name: str
    model_type: str
    version: str = "1.0.0"
    dataset_name: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Model metadata."""
    name: str
    model_type: str
    version: str
    dataset_name: Optional[str] = None
    accuracy: Optional[float] = None
    fairness_metrics: Optional[Dict[str, float]] = None
    registered_at: str
    checkpoint_path: Optional[str] = None

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
    target_column: str = Field(default="income", description="Target variable column")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)


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
    method: str = Field(default="dag_gnn", description="dag_gnn or pc_algorithm")
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
    dataset_name: str
    model_type: str = Field(default="logistic_regression")
    sensitive_features: List[str] = Field(default_factory=lambda: ["sex", "race"])
    target_column: str = Field(default="income")
    max_iterations: int = Field(default=100, ge=1, le=1000)
    accuracy_floor: float = Field(default=0.70, ge=0.5, le=1.0)
    fairness_targets: Dict[str, float] = Field(
        default_factory=lambda: {"demographic_parity_gap": 0.05, "equalized_odds_gap": 0.05}
    )


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
    """Request to start drift monitoring."""
    dataset_name: str
    feature_columns: List[str]
    reference_start: Optional[int] = None
    reference_end: Optional[int] = None
    check_interval_secs: int = Field(default=60, ge=5)


class DriftPoint(BaseModel):
    """Single drift detection result."""
    timestamp: str
    cusum_statistic: float
    wasserstein_distance: float
    is_drift_detected: bool
    confidence: float
    drifted_features: List[str] = Field(default_factory=list)


class DriftMonitorResponse(BaseModel):
    """Drift monitoring results."""
    dataset_name: str
    total_points_checked: int
    drift_points: List[DriftPoint]
    is_stable: bool
    summary: Dict[str, Any] = Field(default_factory=dict)


# ── Text Bias Schemas ─────────────────────────────────────────────

class TextBiasAuditRequest(BaseModel):
    """Request for text bias audit."""
    text_samples: Optional[List[str]] = None
    dataset_name: Optional[str] = None
    demographics: List[Dict[str, str]] = Field(
        default_factory=lambda: [
            {"attribute": "gender", "values": ["man", "woman"]},
            {"attribute": "race", "values": ["Black", "White", "Asian", "Hispanic"]},
        ]
    )
    num_templates: int = Field(default=10, ge=1, le=100)


class TextBiasResult(BaseModel):
    """Text bias measurement for a single demographic pair."""
    template: str
    group_a: str
    group_b: str
    cosine_distance: float
    bias_score: float
    is_biased: bool


class TextBiasAuditResponse(BaseModel):
    """Complete text bias audit response."""
    total_templates_tested: int
    biased_templates: int
    results: List[TextBiasResult]
    overall_bias_score: float
    summary: str


# ── Counterfactual Schemas ────────────────────────────────────────

class CounterfactualRequest(BaseModel):
    """Request for counterfactual generation."""
    dataset_name: str
    instance_index: Optional[int] = None
    instance_data: Optional[Dict[str, Any]] = None
    sensitive_feature: str = Field(default="sex")
    num_counterfactuals: int = Field(default=5, ge=1, le=50)
    proximity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    diversity_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class CounterfactualResult(BaseModel):
    """Single counterfactual instance."""
    original: Dict[str, Any]
    counterfactual: Dict[str, Any]
    changed_features: List[str]
    prediction_change: float
    sparsity: float


class CounterfactualResponse(BaseModel):
    """Counterfactual generation response."""
    dataset_name: str
    num_generated: int
    counterfactuals: List[CounterfactualResult]
    coverage: float
    diversity_score: float


# ── Code Fix Schemas ──────────────────────────────────────────────

class CodeFixRequest(BaseModel):
    """Request to generate bias fix code."""
    dataset_name: str
    bias_report: Dict[str, Any]
    model_type: str = Field(default="logistic_regression")
    fix_type: str = Field(default="auto", description="preprocessing, inprocessing, postprocessing, auto")


class CodeFixResponse(BaseModel):
    """Generated code fix response."""
    fix_code: str
    explanation: str
    estimated_fairness_improvement: Dict[str, float]
    estimated_accuracy_impact: float
    applied_fixes: List[str]


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

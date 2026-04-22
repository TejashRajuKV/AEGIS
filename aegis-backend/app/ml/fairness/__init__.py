"""AEGIS Fairness Module - End-to-end fairness auditing pipeline.

Provides:
- MetricRegistry: Plugin-based registry for fairness metrics
- DemographicParity: Selection rate gap across groups
- EqualizedOdds: FPR/FNR gap across groups
- CalibrationMetric: Calibration error difference across groups
- FairnessPipeline: Full 9-step audit pipeline
- generate_bias_report: Report generation with recommendations
- compute_subgroup_metrics: Per-group and intersectional analysis
"""

from app.ml.fairness.metrics import FairnessMetric, MetricRegistry, metric_registry
from app.ml.fairness.demographic_parity import DemographicParity
from app.ml.fairness.equalized_odds import EqualizedOdds
from app.ml.fairness.calibration import CalibrationMetric
from app.ml.fairness.bias_reporter import generate_bias_report
from app.ml.fairness.subgroup_analysis import (
    compute_subgroup_metrics,
    find_most_biased_subgroup,
)
from app.ml.fairness.fairness_pipeline import FairnessPipeline

__all__ = [
    "FairnessMetric",
    "MetricRegistry",
    "metric_registry",
    "DemographicParity",
    "EqualizedOdds",
    "CalibrationMetric",
    "FairnessPipeline",
    "generate_bias_report",
    "compute_subgroup_metrics",
    "find_most_biased_subgroup",
]

"""Tests for fairness metric computation (V5 registry pattern)."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Simple binary prediction / label arrays for testing."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    # Flip ~10 % to simulate imperfect predictions
    flip_idx = rng.choice(n, size=int(n * 0.1), replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    sensitive = rng.integers(0, 2, size=n)
    y_prob = y_pred.astype(float)
    return y_true, y_pred, y_prob, sensitive


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

class TestMetricRegistry:
    def test_registry_has_metrics(self):
        from app.ml.fairness.metrics import metric_registry

        metrics = metric_registry.list_metrics()
        assert "demographic_parity" in metrics
        assert "equalized_odds" in metrics
        assert "calibration" in metrics

    def test_compute_all(self, binary_data):
        from app.ml.fairness.metrics import metric_registry

        y_true, y_pred, _, sensitive = binary_data
        results = metric_registry.compute_all(y_true, y_pred, sensitive)
        assert "demographic_parity" in results
        assert "equalized_odds" in results
        assert "calibration" in results


# ---------------------------------------------------------------------------
# Demographic parity
# ---------------------------------------------------------------------------

class TestDemographicParity:
    def test_dp_perfect_parity(self):
        from app.ml.fairness.demographic_parity import DemographicParity

        dp = DemographicParity()
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        sensitive = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        result = dp.compute(y_true, y_pred, sensitive)
        assert "gap" in result
        assert result["gap"] == 0.0

    def test_dp_with_gap(self):
        from app.ml.fairness.demographic_parity import DemographicParity

        dp = DemographicParity()
        # Group 0: 100% positive, Group 1: 50% positive
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        sensitive = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])[:8]
        result = dp.compute(y_true, y_pred, sensitive)
        assert result["gap"] > 0


# ---------------------------------------------------------------------------
# Equalized odds
# ---------------------------------------------------------------------------

class TestEqualizedOdds:
    def test_eo_perfect_equality(self, binary_data):
        from app.ml.fairness.equalized_odds import EqualizedOdds

        eo = EqualizedOdds()
        y_true, y_pred, _, sensitive = binary_data
        result = eo.compute(y_true, y_pred, sensitive)
        assert "fpr_gap" in result
        assert "fnr_gap" in result
        assert "max_gap" in result
        assert 0 <= result["fpr_gap"] <= 1

    def test_eo_is_fair(self):
        from app.ml.fairness.equalized_odds import EqualizedOdds

        eo = EqualizedOdds()
        assert eo.is_fair(0.05) is True
        assert eo.is_fair(0.15) is False


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibration_returns_dict(self, binary_data):
        from app.ml.fairness.calibration import CalibrationMetric

        cal = CalibrationMetric()
        y_true, _, y_prob, sensitive = binary_data
        result = cal.compute(y_true, y_prob, sensitive)
        assert "gap" in result
        assert "group_calibration" in result


# ---------------------------------------------------------------------------
# Bias reporter
# ---------------------------------------------------------------------------

class TestBiasReporter:
    def test_generate_report_fair(self):
        from app.ml.fairness.bias_reporter import generate_bias_report

        metric_results = {
            "demographic_parity": {"gap": 0.03},
            "equalized_odds": {"fpr_gap": 0.05, "max_gap": 0.05},
        }
        report = generate_bias_report(
            dataset="test",
            model_type="lr",
            protected_attribute="sex",
            metric_results=metric_results,
        )
        assert report["is_fair"] is True
        assert len(report["recommendations"]) > 0

    def test_generate_report_biased(self):
        from app.ml.fairness.bias_reporter import generate_bias_report

        metric_results = {
            "demographic_parity": {"gap": 0.25},
            "equalized_odds": {"fpr_gap": 0.30, "fnr_gap": 0.15, "max_gap": 0.30},
        }
        report = generate_bias_report(
            dataset="test",
            model_type="lr",
            protected_attribute="race",
            metric_results=metric_results,
        )
        assert report["is_fair"] is False
        assert len(report["recommendations"]) > 1


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------

class TestSubgroupAnalysis:
    def test_compute_subgroup_metrics(self, binary_data):
        from app.ml.fairness.subgroup_analysis import compute_subgroup_metrics

        y_true, y_pred, _, sensitive = binary_data
        results = compute_subgroup_metrics(y_true, y_pred, sensitive)
        assert len(results) > 0
        # Check group 0 has expected keys
        group_0 = results.get("0", results.get("0.0"))
        if group_0:
            assert "count" in group_0
            assert "accuracy" in group_0
            assert "selection_rate" in group_0

    def test_find_most_biased_subgroup(self, binary_data):
        from app.ml.fairness.subgroup_analysis import (
            compute_subgroup_metrics,
            find_most_biased_subgroup,
        )

        y_true, y_pred, _, sensitive = binary_data
        results = compute_subgroup_metrics(y_true, y_pred, sensitive)
        biased = find_most_biased_subgroup(results, metric="selection_rate")
        assert "max_gap" in biased
        assert biased["max_gap"] >= 0

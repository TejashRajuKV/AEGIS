"""
Tests for Pipeline Coordination
=================================
Tests for results aggregator, pipeline coordinator, drift pipeline,
and autopilot pipeline creation.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.pipeline.results_aggregator import ResultsAggregator


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def aggregator():
    return ResultsAggregator()


@pytest.fixture
def tmp_json_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ===================================================================
# Results Aggregator Tests
# ===================================================================

class TestResultsAggregator:

    def test_add_result_dict(self, aggregator):
        aggregator.add_result("fairness", {"accuracy": 0.85, "dp_gap": 0.05})
        result = aggregator.get_module_result("fairness")
        assert result is not None
        assert result["accuracy"] == 0.85

    def test_add_result_object_with_to_dict(self, aggregator):
        class MockResult:
            def to_dict(self):
                return {"status": "ok", "value": 42}
        aggregator.add_result("custom", MockResult())
        result = aggregator.get_module_result("custom")
        assert result["status"] == "ok"
        assert result["value"] == 42

    def test_add_result_string(self, aggregator):
        aggregator.add_result("plain", "just a string")
        result = aggregator.get_module_result("plain")
        assert result["type"] == "str"
        assert "just a string" in result["value"]

    def test_add_overwrites(self, aggregator):
        aggregator.add_result("mod", {"v": 1})
        aggregator.add_result("mod", {"v": 2})
        assert aggregator.get_module_result("mod")["v"] == 2

    def test_get_full_report(self, aggregator):
        aggregator.add_result("fairness", {"accuracy": 0.85})
        aggregator.add_result("drift", {"drift_detected": False})
        report = aggregator.get_full_report()
        assert report["report_type"] == "full_audit_report"
        assert report["n_modules"] == 2
        assert "fairness" in report["results"]
        assert "drift" in report["results"]
        assert "health_assessment" in report
        assert "generated_at" in report

    def test_get_summary(self, aggregator):
        aggregator.add_result("fairness", {
            "accuracy": 0.85,
            "demographic_parity_gap": 0.05,
            "equalized_odds_gap": 0.08,
            "calibration_error": 0.03,
        })
        aggregator.add_result("drift", {
            "drift_detected": True,
            "severity": "high",
            "n_drifted_features": 2,
            "n_features_monitored": 10,
        })
        summary = aggregator.get_summary()
        assert summary["summary_type"] == "key_metrics"
        assert "fairness" in summary["metrics"]
        assert "drift" in summary["metrics"]
        assert summary["metrics"]["fairness"]["accuracy"] == 0.85
        assert summary["metrics"]["drift"]["drift_detected"] is True

    def test_export_json(self, aggregator, tmp_json_dir):
        aggregator.add_result("fairness", {"accuracy": 0.9})
        filepath = str(tmp_json_dir / "report.json")
        result_path = aggregator.export_json(filepath)
        assert result_path == filepath
        assert Path(filepath).exists()

        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["report_type"] == "full_audit_report"
        assert data["n_modules"] == 1

    def test_export_json_creates_dirs(self, aggregator, tmp_json_dir):
        nested = str(tmp_json_dir / "deep" / "nested" / "report.json")
        aggregator.add_result("test", {"val": 1})
        aggregator.export_json(nested)
        assert Path(nested).exists()

    def test_metric_comparison(self, aggregator):
        before = {"accuracy": 0.80, "dp_gap": 0.15, "eo_gap": 0.20}
        after = {"accuracy": 0.85, "dp_gap": 0.08, "eo_gap": 0.10}
        comparison = aggregator.get_metric_comparison(before, after)
        assert comparison["comparison_type"] == "before_after"
        assert len(comparison["metrics"]) == 3
        assert "accuracy" in comparison["improvements"]  # higher is better
        assert "dp_gap" in comparison["improvements"]  # lower is better

    def test_metric_comparison_unchanged(self, aggregator):
        before = {"accuracy": 0.80}
        after = {"accuracy": 0.80}
        comparison = aggregator.get_metric_comparison(before, after)
        assert "accuracy" in comparison["unchanged"]

    def test_clear(self, aggregator):
        aggregator.add_result("a", {"v": 1})
        aggregator.add_result("b", {"v": 2})
        aggregator.clear()
        assert aggregator.list_modules() == []

    def test_list_modules(self, aggregator):
        aggregator.add_result("x", {"v": 1})
        aggregator.add_result("y", {"v": 2})
        modules = aggregator.list_modules()
        assert modules == ["x", "y"]

    def test_remove_module(self, aggregator):
        aggregator.add_result("x", {"v": 1})
        assert aggregator.remove_module("x") is True
        assert aggregator.remove_module("nonexistent") is False
        assert aggregator.get_module_result("x") is None

    def test_health_assessment_healthy(self, aggregator):
        aggregator.add_result("drift", {"drift_detected": False, "severity": "none"})
        aggregator.add_result("fairness", {
            "metrics": {"demographic_parity_gap": 0.01, "equalized_odds_gap": 0.01},
        })
        report = aggregator.get_full_report()
        assert report["health_assessment"]["status"] == "healthy"

    def test_health_assessment_degraded(self, aggregator):
        aggregator.add_result("drift", {"drift_detected": True, "severity": "critical"})
        report = aggregator.get_full_report()
        assert report["health_assessment"]["status"] == "degraded"

    def test_overall_metrics(self, aggregator):
        aggregator.add_result("fairness", {
            "metrics": {"accuracy": 0.85, "demographic_parity_gap": 0.12, "equalized_odds_gap": 0.10},
        })
        aggregator.add_result("drift", {"drift_detected": False, "severity": "none"})
        summary = aggregator.get_summary()
        assert summary["overall"]["n_modules"] == 2
        assert summary["overall"]["best_accuracy"] == 0.85
        assert summary["overall"]["drift_detected"] is False

    def test_empty_report(self, aggregator):
        report = aggregator.get_full_report()
        assert report["n_modules"] == 0
        assert report["results"] == {}

    def test_empty_summary(self, aggregator):
        summary = aggregator.get_summary()
        assert summary["modules_reported"] == []
        assert summary["overall"]["n_modules"] == 0


# ===================================================================
# Pipeline Coordinator Tests
# ===================================================================

class TestPipelineCoordinator:

    def test_creation(self):
        from app.pipeline.pipeline_coordinator import PipelineCoordinator
        coordinator = PipelineCoordinator()
        assert coordinator is not None

    def test_status_returns_dict(self):
        from app.pipeline.pipeline_coordinator import PipelineCoordinator
        coordinator = PipelineCoordinator()
        status = coordinator.get_status()
        assert isinstance(status, dict)
        assert status["coordinator"] == "active"
        assert "pipelines" in status

    def test_status_has_pipeline_entries(self):
        from app.pipeline.pipeline_coordinator import PipelineCoordinator
        coordinator = PipelineCoordinator()
        status = coordinator.get_status()
        assert "autopilot" in status["pipelines"]
        assert "drift" in status["pipelines"]


# ===================================================================
# Drift Pipeline Tests
# ===================================================================

class TestDriftPipeline:

    def test_creation(self):
        from app.pipeline.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        assert pipeline is not None

    def test_status_returns_dict(self):
        from app.pipeline.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        status = pipeline.get_status()
        assert isinstance(status, dict)
        assert "status_message" in status
        assert "is_fitted" in status
        assert status["is_fitted"] is False

    def test_monitor_produces_result(self):
        from app.pipeline.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        rng = np.random.RandomState(42)
        reference = rng.randn(100, 3)
        new_data = rng.randn(50, 3) + 1.5  # drifted
        result = pipeline.monitor(reference, new_data, feature_names=["a", "b", "c"])
        assert hasattr(result, "drift_detected")
        assert isinstance(result.drift_detected, bool)
        assert hasattr(result, "to_dict")
        d = result.to_dict()
        assert "drift_detected" in d
        assert "severity" in d

    def test_monitor_status_after_run(self):
        from app.pipeline.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        rng = np.random.RandomState(42)
        reference = rng.randn(100, 2)
        new_data = rng.randn(50, 2)
        pipeline.monitor(reference, new_data)
        status = pipeline.get_status()
        assert status["is_fitted"] is True
        assert status["status_message"] == "completed"

    def test_get_alerts_empty(self):
        from app.pipeline.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        alerts = pipeline.get_alerts()
        assert alerts == []

    def test_monitor_with_dict_data(self):
        from app.pipeline.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        rng = np.random.RandomState(42)
        reference = {"f1": rng.randn(100), "f2": rng.randn(100)}
        new_data = {"f1": rng.randn(50) + 2.0, "f2": rng.randn(50) + 2.0}
        result = pipeline.monitor(reference, new_data)
        assert result is not None
        assert "f1" in result.feature_scores


# ===================================================================
# Autopilot Pipeline Tests
# ===================================================================

class TestAutopilotPipeline:

    def test_creation(self):
        from app.pipeline.autopilot_pipeline import AutopilotPipeline
        pipeline = AutopilotPipeline()
        assert pipeline is not None

    def test_creation_with_config(self):
        from app.pipeline.autopilot_pipeline import AutopilotPipeline
        config = {"n_episodes": 5, "max_steps_per_episode": 10}
        pipeline = AutopilotPipeline(config=config)
        assert pipeline.config["n_episodes"] == 5

    def test_status_returns_dict(self):
        from app.pipeline.autopilot_pipeline import AutopilotPipeline
        pipeline = AutopilotPipeline()
        status = pipeline.get_status()
        assert isinstance(status, dict)
        assert status["running"] is False
        assert status["progress"] == 0.0
        assert "status_message" in status
        assert "config" in status

    def test_stop(self):
        from app.pipeline.autopilot_pipeline import AutopilotPipeline
        pipeline = AutopilotPipeline()
        pipeline.stop()
        assert pipeline._running is False
        assert pipeline.get_status()["status_message"] == "stopping"

    def test_extract_data_tuple(self):
        from app.pipeline.autopilot_pipeline import AutopilotPipeline
        pipeline = AutopilotPipeline()
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        extracted_X, extracted_y = pipeline._extract_data((X, y))
        np.testing.assert_array_equal(extracted_X, X)
        np.testing.assert_array_equal(extracted_y, y)

    def test_extract_sensitive_features_array(self):
        from app.pipeline.autopilot_pipeline import AutopilotPipeline
        pipeline = AutopilotPipeline()
        X = np.random.randn(50, 5)
        sf = np.random.randint(0, 2, 50).astype(np.float64)
        result = pipeline._extract_sensitive_features(sf, X)
        np.testing.assert_array_equal(result, sf)

    def test_extract_sensitive_features_int(self):
        from app.pipeline.autopilot_pipeline import AutopilotPipeline
        pipeline = AutopilotPipeline()
        X = np.random.randn(50, 5)
        result = pipeline._extract_sensitive_features(2, X)
        np.testing.assert_array_equal(result, X[:, 2].astype(np.float64))

    def test_autopilot_result_to_dict(self):
        from app.pipeline.autopilot_pipeline import AutopilotResult
        result = AutopilotResult(
            success=True, best_accuracy=0.9, best_dp_gap=0.05,
            best_eo_gap=0.04, training_time=10.5, total_episodes=3,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["best_accuracy"] == 0.9
        assert d["training_time"] == 10.5

    def test_drift_pipeline_result_to_dict(self):
        from app.pipeline.drift_pipeline import DriftPipelineResult
        result = DriftPipelineResult(
            drift_detected=True, severity="medium", monitoring_time=2.3,
            n_features_monitored=5, n_drifted_features=2,
        )
        d = result.to_dict()
        assert d["drift_detected"] is True
        assert d["severity"] == "medium"

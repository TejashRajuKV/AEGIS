"""
Results Aggregator
==================
Aggregates results from all AEGIS modules (fairness, causal, drift,
autopilot, counterfactual, text bias) into unified reports and summaries.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("aegis.pipeline.aggregator")


class ResultsAggregator:
    """
    Aggregates results from all AEGIS modules.

    Provides a central store for module results with methods to:
    - Collect and store individual module results
    - Generate full combined reports
    - Produce concise summaries of key metrics
    - Export reports as JSON files
    - Compare before/after metrics
    """

    def __init__(self) -> None:
        """Initialize the aggregator with empty result stores."""
        self._results: Dict[str, Any] = {}
        self._timestamps: Dict[str, str] = {}
        self._created_at = datetime.now(timezone.utc).isoformat()

        logger.info("ResultsAggregator initialized")

    def add_result(self, module_name: str, result: Any) -> None:
        """
        Store a result from a named module.

        Args:
            module_name: Name of the module (e.g., 'fairness', 'drift', 'autopilot').
            result: The result to store. Can be a dict, dataclass, or any object
                    with a to_dict() method.
        """
        # Convert result to dict if it has to_dict method
        if hasattr(result, "to_dict") and callable(result.to_dict):
            serialized = result.to_dict()
        elif isinstance(result, dict):
            serialized = result
        else:
            # Try to serialize as a basic dict
            serialized = {"value": str(result), "type": type(result).__name__}

        self._results[module_name] = serialized
        self._timestamps[module_name] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Result added: module='%s', keys=%s",
            module_name,
            list(serialized.keys()) if isinstance(serialized, dict) else "N/A",
        )

    def get_full_report(self) -> Dict[str, Any]:
        """
        Get a full report combining all module results.

        Returns:
            Dict with all module results, timestamps, and metadata.
        """
        report: Dict[str, Any] = {
            "report_type": "full_audit_report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "aggregator_created_at": self._created_at,
            "modules": list(self._results.keys()),
            "n_modules": len(self._results),
            "results": {},
            "timestamps": dict(self._timestamps),
        }

        for module_name, result in self._results.items():
            report["results"][module_name] = result

        # Add overall health assessment
        report["health_assessment"] = self._assess_health()

        return report

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a concise summary with key metrics from all modules.

        Returns:
            Dict with summarized key metrics from each module.
        """
        summary: Dict[str, Any] = {
            "summary_type": "key_metrics",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "modules_reported": list(self._results.keys()),
            "metrics": {},
        }

        for module_name, result in self._results.items():
            module_summary = self._extract_key_metrics(module_name, result)
            summary["metrics"][module_name] = module_summary

        # Aggregate overall metrics
        summary["overall"] = self._compute_overall_metrics()

        return summary

    def export_json(self, filepath: str) -> str:
        """
        Save the full report as a JSON file.

        Args:
            filepath: Path to write the JSON file.

        Returns:
            The filepath that was written.
        """
        report = self.get_full_report()

        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        file_size = os.path.getsize(filepath)
        logger.info("Report exported to %s (%d bytes)", filepath, file_size)
        return filepath

    def get_metric_comparison(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare before/after metrics and compute changes.

        Args:
            before: Dict of metric_name -> value (before optimization/fix).
            after: Dict of metric_name -> value (after optimization/fix).

        Returns:
            Dict with comparison data including changes, improvements, and regressions.
        """
        comparison: Dict[str, Any] = {
            "comparison_type": "before_after",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "improvements": [],
            "regressions": [],
            "unchanged": [],
        }

        all_keys = set(list(before.keys()) + list(after.keys()))

        for key in sorted(all_keys):
            before_val = before.get(key)
            after_val = after.get(key)

            entry: Dict[str, Any] = {
                "metric": key,
                "before": before_val,
                "after": after_val,
            }

            # Compute change if both values are numeric
            if before_val is not None and after_val is not None:
                try:
                    b = float(before_val)
                    a = float(after_val)
                    change = a - b
                    pct_change = (change / abs(b) * 100) if abs(b) > 1e-10 else 0.0

                    entry["change"] = round(change, 6)
                    entry["pct_change"] = round(pct_change, 2)

                    # Determine if this is an improvement or regression
                    # For gaps/differences/errors: lower is better
                    # For accuracy/precision/recall: higher is better
                    lower_is_better = any(
                        kw in key.lower()
                        for kw in ("gap", "error", "loss", "distance", "drift", "bias")
                    )

                    if abs(change) < 1e-8:
                        comparison["unchanged"].append(key)
                        entry["direction"] = "unchanged"
                    elif lower_is_better and change < 0:
                        comparison["improvements"].append(key)
                        entry["direction"] = "improved"
                    elif not lower_is_better and change > 0:
                        comparison["improvements"].append(key)
                        entry["direction"] = "improved"
                    else:
                        comparison["regressions"].append(key)
                        entry["direction"] = "regressed"

                except (ValueError, TypeError):
                    entry["change"] = None
                    entry["pct_change"] = None
                    entry["direction"] = "incomparable"
            elif before_val is None:
                entry["direction"] = "new"
            elif after_val is None:
                entry["direction"] = "removed"

            comparison["metrics"][key] = entry

        comparison["n_improvements"] = len(comparison["improvements"])
        comparison["n_regressions"] = len(comparison["regressions"])
        comparison["n_unchanged"] = len(comparison["unchanged"])

        return comparison

    def clear(self) -> None:
        """Reset all stored results."""
        self._results.clear()
        self._timestamps.clear()
        logger.info("ResultsAggregator cleared all results")

    def get_module_result(self, module_name: str) -> Optional[Dict]:
        """Get the stored result for a specific module."""
        return self._results.get(module_name)

    def list_modules(self) -> List[str]:
        """List all modules that have stored results."""
        return list(self._results.keys())

    def remove_module(self, module_name: str) -> bool:
        """Remove a specific module's results."""
        if module_name in self._results:
            del self._results[module_name]
            self._timestamps.pop(module_name, None)
            logger.info("Removed results for module: %s", module_name)
            return True
        return False

    def _extract_key_metrics(self, module_name: str, result: Dict) -> Dict:
        """Extract key metrics from a module result."""
        if not isinstance(result, dict):
            return {"raw": str(result)}

        metrics: Dict[str, Any] = {}

        if module_name == "fairness":
            metrics["accuracy"] = result.get("accuracy", result.get("metrics", {}).get("accuracy"))
            dp_gap = result.get("demographic_parity_gap", result.get("metrics", {}).get("demographic_parity_gap"))
            metrics["demographic_parity_gap"] = dp_gap
            eo_gap = result.get("equalized_odds_gap", result.get("metrics", {}).get("equalized_odds_gap"))
            metrics["equalized_odds_gap"] = eo_gap
            metrics["calibration_error"] = result.get("calibration_error", result.get("metrics", {}).get("calibration_error"))
            metrics["overall_fairness_score"] = result.get("overall_score")

        elif module_name == "drift":
            metrics["drift_detected"] = result.get("drift_detected")
            metrics["severity"] = result.get("severity")
            metrics["n_drifted_features"] = result.get("n_drifted_features", 0)
            metrics["n_features_monitored"] = result.get("n_features_monitored", 0)
            metrics["monitoring_time"] = result.get("monitoring_time")

        elif module_name == "autopilot":
            metrics["success"] = result.get("success")
            metrics["best_accuracy"] = result.get("best_accuracy")
            metrics["best_dp_gap"] = result.get("best_dp_gap")
            metrics["best_eo_gap"] = result.get("best_eo_gap")
            metrics["training_time"] = result.get("training_time")
            metrics["total_episodes"] = result.get("total_episodes")

        elif module_name == "causal":
            metrics["n_nodes"] = result.get("n_nodes")
            metrics["n_edges"] = result.get("n_edges")
            metrics["method"] = result.get("method")

        elif module_name == "counterfactual":
            metrics["sensitive_attribute"] = result.get("sensitive_attribute")
            metrics["original_value"] = result.get("original_value")
            metrics["target_value"] = result.get("target_value")
            metrics["prediction_changed"] = result.get("prediction_change", {}).get("prediction_changed")

        elif module_name == "text_bias":
            metrics["total_pairs"] = result.get("total_pairs", result.get("overall", {}).get("total_pairs"))
            metrics["bias_index"] = result.get("bias_index", result.get("overall", {}).get("bias_index"))
            metrics["bias_level"] = result.get("bias_level", result.get("overall", {}).get("bias_level"))
            metrics["mean_distance"] = result.get("mean_distance", result.get("summary", {}).get("mean_distance"))

        else:
            # Generic: extract top-level numeric values
            for k, v in result.items():
                if isinstance(v, (int, float, bool)):
                    metrics[k] = v

        return metrics

    def _compute_overall_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across all modules."""
        overall: Dict[str, Any] = {
            "n_modules": len(self._results),
            "modules": list(self._results.keys()),
        }

        # Collect fairness metrics if available
        fairness = self._results.get("fairness", {})
        if isinstance(fairness, dict):
            inner = fairness.get("metrics", fairness)
            if isinstance(inner, dict):
                overall["best_accuracy"] = inner.get("accuracy")
                overall["worst_fairness_gap"] = max(
                    inner.get("demographic_parity_gap", 0),
                    inner.get("equalized_odds_gap", 0),
                )

        # Collect drift status
        drift = self._results.get("drift", {})
        if isinstance(drift, dict):
            overall["drift_detected"] = drift.get("drift_detected", False)
            overall["drift_severity"] = drift.get("severity", "none")

        # Collect autopilot status
        autopilot = self._results.get("autopilot", {})
        if isinstance(autopilot, dict):
            overall["autopilot_success"] = autopilot.get("success", False)

        return overall

    def _assess_health(self) -> Dict[str, Any]:
        """Assess overall system health based on all results."""
        health: Dict[str, Any] = {
            "status": "healthy",
            "issues": [],
            "warnings": [],
        }

        # Check drift
        drift = self._results.get("drift", {})
        if isinstance(drift, dict):
            if drift.get("drift_detected"):
                severity = drift.get("severity", "none")
                if severity in ("high", "critical"):
                    health["status"] = "degraded"
                    health["issues"].append(f"Drift severity: {severity}")
                elif severity == "medium":
                    health["status"] = "warning"
                    health["warnings"].append(f"Drift severity: {severity}")

        # Check fairness gaps
        fairness = self._results.get("fairness", {})
        if isinstance(fairness, dict):
            inner = fairness.get("metrics", fairness)
            if isinstance(inner, dict):
                dp = inner.get("demographic_parity_gap", 0)
                eo = inner.get("equalized_odds_gap", 0)
                if dp > 0.2 or eo > 0.2:
                    if health["status"] == "healthy":
                        health["status"] = "warning"
                    health["warnings"].append(f"High fairness gap: DP={dp:.3f}, EO={eo:.3f}")

        # Check autopilot
        autopilot = self._results.get("autopilot", {})
        if isinstance(autopilot, dict):
            if autopilot.get("error"):
                health["warnings"].append(f"Autopilot error: {autopilot.get('error', '')[:100]}")

        return health

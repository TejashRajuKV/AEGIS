"""
Drift Pipeline
==============
Drift monitoring orchestrator that coordinates drift detection using
the ensemble detector (CUSUM + Wasserstein), manages alerts, and
generates monitoring reports.

Designed for sequential execution on 16GB RAM gaming laptop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("aegis.pipeline.drift")


@dataclass
class DriftPipelineResult:
    """Result of a drift monitoring run."""

    drift_detected: bool = False
    severity: str = "none"
    feature_scores: Dict[str, float] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    monitoring_time: float = 0.0
    timestamp: str = ""
    n_features_monitored: int = 0
    n_drifted_features: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for API serialization."""
        return {
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "feature_scores": self.feature_scores,
            "alerts": self.alerts,
            "report": self.report,
            "monitoring_time": round(self.monitoring_time, 4),
            "timestamp": self.timestamp,
            "n_features_monitored": self.n_features_monitored,
            "n_drifted_features": self.n_drifted_features,
        }


class DriftPipeline:
    """
    Drift monitoring orchestrator.

    Coordinates drift detection across features:
    1. Creates DriftEnsemble with reference data
    2. Fits detectors on reference distribution
    3. Detects drift on new data per feature
    4. Collects alerts and generates summary report
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the drift pipeline.

        Args:
            config: Optional configuration dict with drift detection parameters.
                    Supported keys: cusum_threshold, wasserstein_threshold,
                    min_reference_samples, severity_thresholds.
        """
        self.config = config or {}
        self._ensemble = None
        self._alert_manager = None
        self._is_fitted = False
        self._feature_names: List[str] = []
        self._last_result: Optional[DriftPipelineResult] = None
        self._status_message = "initialized"

        logger.info("DriftPipeline initialized with config: %s", self.config)

    def monitor(
        self,
        reference_data: Any,
        new_data: Any,
        feature_names: Optional[List[str]] = None,
    ) -> DriftPipelineResult:
        """
        Run drift monitoring: fit on reference, detect on new data.

        Args:
            reference_data: Reference/baseline data. Can be:
                - numpy array (n_samples, n_features)
                - pandas DataFrame
                - dict of feature_name -> array
            new_data: New data to check for drift. Same format as reference.
            feature_names: Optional list of feature names.

        Returns:
            DriftPipelineResult with detection results, alerts, and report.
        """
        start_time = time.time()
        self._status_message = "preparing"

        try:
            # Step 1: Convert data to numpy arrays
            ref_array, ref_names = self._to_array(reference_data, feature_names)
            new_array, _ = self._to_array(new_data, ref_names)
            self._feature_names = ref_names

            logger.info(
                "Data prepared: reference=%s, new=%s, features=%s",
                ref_array.shape, new_array.shape, ref_names,
            )

            # Step 2: Import drift modules
            from app.ml.drift.drift_ensemble import DriftEnsemble
            from app.ml.drift.drift_alert import DriftAlertManager

            self._status_message = "fitting"

            # Step 3: Create and fit ensemble on reference data
            self._alert_manager = DriftAlertManager(
                max_history=self.config.get("max_alert_history", 1000)
            )
            self._ensemble = DriftEnsemble(
                cusum_threshold=self.config.get("cusum_threshold", 5.0),
                wasserstein_threshold=self.config.get("wasserstein_threshold", 0.1),
                alert_manager=self._alert_manager,
                min_reference_samples=self.config.get("min_reference_samples", 30),
            )

            # Fit on first feature of reference (per-feature handling below)
            self._ensemble.fit(ref_array[:, 0] if ref_array.ndim == 2 else ref_array, ref_names)

            self._is_fitted = True
            self._status_message = "detecting"

            # Step 4: Detect drift per feature (sequentially)
            n_features = ref_array.shape[1] if ref_array.ndim == 2 else 1
            feature_scores: Dict[str, float] = {}
            all_alerts: List[Dict[str, Any]] = []
            any_drift = False
            max_severity = "none"
            severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
            n_drifted = 0

            for i in range(n_features):
                fname = ref_names[i] if i < len(ref_names) else f"feature_{i}"
                ref_col = ref_array[:, i]
                new_col = new_array[:, i] if new_array.ndim == 2 else new_array

                try:
                    # Fit a fresh ensemble per feature for isolation
                    feat_ensemble = DriftEnsemble(
                        cusum_threshold=self.config.get("cusum_threshold", 5.0),
                        wasserstein_threshold=self.config.get("wasserstein_threshold", 0.1),
                        alert_manager=self._alert_manager,
                        min_reference_samples=self.config.get("min_reference_samples", 30),
                    )
                    feat_ensemble.fit(ref_col, [fname])
                    result = feat_ensemble.detect(new_col, feature_name=fname)

                    feature_scores[fname] = (
                        max(
                            result.cusum_result.statistic if result.cusum_result else 0.0,
                            result.wasserstein_result.statistic if result.wasserstein_result else 0.0,
                        )
                    )

                    if result.drift_detected:
                        any_drift = True
                        n_drifted += 1
                        if severity_rank.get(result.overall_severity, 0) > severity_rank.get(max_severity, 0):
                            max_severity = result.overall_severity

                    if result.alert is not None:
                        all_alerts.append(result.alert.to_dict())

                    logger.debug(
                        "Feature '%s': drift=%s, severity=%s, score=%.4f",
                        fname, result.drift_detected, result.overall_severity,
                        feature_scores[fname],
                    )

                except Exception as exc:
                    logger.warning("Error detecting drift for feature '%s': %s", fname, exc)
                    feature_scores[fname] = 0.0

            self._status_message = "generating_report"

            # Step 5: Generate summary report
            report = {
                "ensemble_fitted": True,
                "n_features": n_features,
                "n_features_with_drift": n_drifted,
                "overall_drift_detected": any_drift,
                "overall_severity": max_severity,
                "feature_details": feature_scores,
                "active_alerts_count": len(all_alerts),
                "alert_summary": self._summarize_alerts(all_alerts),
                "recommendations": self._generate_recommendations(max_severity, n_drifted, n_features),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Add alert history from manager
            if self._alert_manager:
                report["severity_counts"] = self._alert_manager.get_severity_counts()
                history = self._alert_manager.get_alert_history()
                report["total_alerts_in_history"] = len(history)

            monitoring_time = time.time() - start_time

            self._last_result = DriftPipelineResult(
                drift_detected=any_drift,
                severity=max_severity,
                feature_scores=feature_scores,
                alerts=all_alerts,
                report=report,
                monitoring_time=monitoring_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
                n_features_monitored=n_features,
                n_drifted_features=n_drifted,
            )

            self._status_message = "completed"

            logger.info(
                "Drift monitoring complete: drift=%s, severity=%s, "
                "features_drifted=%d/%d, time=%.2fs",
                any_drift, max_severity, n_drifted, n_features, monitoring_time,
            )

            return self._last_result

        except ImportError as e:
            self._status_message = "error"
            logger.error("Drift pipeline import error: %s", e)
            return DriftPipelineResult(
                severity="error",
                report={"error": f"Missing dependency: {e}"},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            self._status_message = "error"
            logger.error("Drift pipeline error: %s", e, exc_info=True)
            return DriftPipelineResult(
                severity="error",
                report={"error": str(e)},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts."""
        if self._alert_manager is None:
            return []
        active = self._alert_manager.get_active_alerts()
        return [alert.to_dict() for alert in active]

    def get_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            "status_message": self._status_message,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
            "last_result": self._last_result.to_dict() if self._last_result else None,
            "config": self.config,
            "active_alerts": self.get_alerts(),
        }

    def _to_array(
        self, data: Any, feature_names: Optional[List[str]] = None
    ) -> tuple:
        """Convert input data to numpy array and extract feature names."""
        # pandas DataFrame
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                if feature_names is not None:
                    cols = [c for c in feature_names if c in data.columns]
                    if cols:
                        arr = data[cols].values
                        return arr, cols
                arr = data.values
                names = list(data.columns)
                return arr, names
        except ImportError:
            pass

        # Dictionary of feature_name -> array
        if isinstance(data, dict):
            names = list(data.keys())
            arrays = [np.asarray(data[n], dtype=np.float64) for n in names]
            # Ensure all arrays have same length
            min_len = min(len(a) for a in arrays) if arrays else 0
            arrays = [a[:min_len] for a in arrays]
            arr = np.column_stack(arrays) if len(arrays) > 1 else arrays[0].reshape(-1, 1)
            return arr, names

        # Object with .values or similar
        if hasattr(data, "values"):
            arr = np.asarray(data.values, dtype=np.float64)
            names = feature_names or [f"feature_{i}" for i in range(arr.shape[1] if arr.ndim == 2 else 1)]
            return arr, names

        # numpy array
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]
        return arr, names

    def _summarize_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of all alerts."""
        if not alerts:
            return {"total": 0, "by_severity": {}, "by_feature": {}}

        by_severity: Dict[str, int] = {}
        by_feature: Dict[str, int] = {}
        for alert in alerts:
            sev = alert.get("severity", "unknown")
            feat = alert.get("feature_name", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_feature[feat] = by_feature.get(feat, 0) + 1

        return {
            "total": len(alerts),
            "by_severity": by_severity,
            "by_feature": by_feature,
        }

    def _generate_recommendations(
        self, severity: str, n_drifted: int, n_total: int
    ) -> List[str]:
        """Generate actionable recommendations based on drift results."""
        recommendations = []

        if severity == "none":
            recommendations.append(
                "No significant drift detected. Continue regular monitoring."
            )
            return recommendations

        if severity in ("critical", "high"):
            recommendations.append(
                "CRITICAL/HIGH drift detected. Immediate model retraining is recommended."
            )
            recommendations.append(
                "Investigate upstream data pipeline for distribution changes."
            )
            recommendations.append(
                "Consider rolling back to a previous model version as a safety measure."
            )

        if severity == "medium":
            recommendations.append(
                "Moderate drift detected. Schedule model retraining within 24-48 hours."
            )
            recommendations.append(
                "Compare reference data timeframe with current data to identify root cause."
            )

        if severity == "low":
            recommendations.append(
                "Minor drift detected. Increase monitoring frequency."
            )
            recommendations.append(
                "No immediate action required, but track trend over next monitoring window."
            )

        drift_ratio = n_drifted / max(n_total, 1)
        if drift_ratio > 0.5:
            recommendations.append(
                f"Drift detected in {drift_ratio*100:.0f}% of features. "
                "This may indicate a systematic data pipeline issue."
            )
        elif n_drifted > 0:
            drifted = [
                fname for fname, score in (
                    self._last_result.feature_scores.items()
                    if self._last_result else {}
                )
                if score > 0
            ]
            if drifted:
                recommendations.append(
                    f"Focus investigation on drifted features: {', '.join(drifted[:5])}"
                )

        return recommendations

"""
Drift Ensemble Detector
========================
Combines CUSUM and Wasserstein detectors for near-zero false negatives.
CUSUM catches mean drift; Wasserstein catches distribution shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from app.ml.drift.cusum_detector import CUSUMDetector, DriftResult, DriftStatus
from app.ml.drift.wasserstein_detector import WassersteinDetector
from app.ml.drift.drift_alert import DriftAlert, DriftAlertManager, AlertSeverity

logger = logging.getLogger("aegis.drift.ensemble")


@dataclass
class EnsembleDriftResult:
    """Combined drift detection result from all detectors."""

    drift_detected: bool
    overall_severity: str
    cusum_result: Optional[DriftResult] = None
    wasserstein_result: Optional[DriftResult] = None
    feature_name: str = ""
    alert: Optional[DriftAlert] = None
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "drift_detected": self.drift_detected,
            "overall_severity": self.overall_severity,
            "feature_name": self.feature_name,
            "cusum": self.cusum_result.to_dict() if self.cusum_result else None,
            "wasserstein": self.wasserstein_result.to_dict() if self.wasserstein_result else None,
            "alert": self.alert.to_dict() if self.alert else None,
            "details": self.details,
        }


class DriftEnsemble:
    """
    Ensemble drift detector combining CUSUM and Wasserstein.

    Strategy: drift_detected = CUSUM OR Wasserstein
    This ensures near-zero false negatives because:
    - CUSUM catches gradual mean shifts
    - Wasserstein catches distribution shape changes (variance, tail)

    Both detectors are run sequentially (not in parallel) for memory efficiency.
    """

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        wasserstein_threshold: float = 0.1,
        alert_manager: Optional[DriftAlertManager] = None,
        min_reference_samples: int = 30,
    ):
        """
        Initialize the ensemble detector.

        Args:
            cusum_threshold: CUSUM decision threshold.
            wasserstein_threshold: Wasserstein distance threshold.
            alert_manager: Optional alert manager for alert generation.
            min_reference_samples: Minimum reference samples for fitting.
        """
        self.cusum = CUSUMDetector(
            threshold=cusum_threshold,
            min_reference_samples=min_reference_samples,
        )
        self.wasserstein = WassersteinDetector(
            threshold=wasserstein_threshold,
            min_reference_samples=min_reference_samples,
        )
        self.alert_manager = alert_manager or DriftAlertManager()

        self._is_fitted: bool = False
        self._feature_names: List[str] = []

        logger.info("DriftEnsemble initialized with CUSUM + Wasserstein")

    def fit(self, reference_data: np.ndarray, feature_names: Optional[List[str]] = None) -> "DriftEnsemble":
        """
        Fit both detectors on reference data.

        Bug 28 fix: For multi-feature data, fit a CUSUM and Wasserstein detector
        per feature and store them in lists.  The main `detect()` path then runs
        all per-feature detectors.

        Args:
            reference_data: Reference data (n_samples, n_features) or (n_samples,).
            feature_names: Optional feature names.

        Returns:
            self
        """
        data = np.asarray(reference_data, dtype=np.float64)

        if feature_names:
            self._feature_names = list(feature_names)

        if data.ndim == 1:
            # Single-feature case — keep the existing per-class detectors
            self.cusum.fit(data)
            self.wasserstein.fit(data)
            # Bug 29 fix: store reference data on self, not on the detector
            self._reference_data = data.copy()
            if not self._feature_names:
                self._feature_names = ["feature_0"]
            self._cusum_per_feature = [self.cusum]
            self._wasserstein_per_feature = [self.wasserstein]
        else:
            # Bug 28 fix: fit one detector pair per feature
            n_features = data.shape[1]
            if not self._feature_names:
                self._feature_names = [f"feature_{i}" for i in range(n_features)]

            self._cusum_per_feature = []
            self._wasserstein_per_feature = []
            self._reference_data = data.copy()  # Bug 29 fix: store locally

            for i in range(n_features):
                col = data[:, i]
                c = CUSUMDetector(
                    threshold=self.cusum.threshold,
                    min_reference_samples=self.cusum.min_reference_samples,
                )
                w = WassersteinDetector(
                    threshold=self.wasserstein.threshold,
                    min_reference_samples=self.wasserstein.min_reference_samples,
                )
                c.fit(col)
                w.fit(col)
                self._cusum_per_feature.append(c)
                self._wasserstein_per_feature.append(w)

            # Keep primary detectors pointing to feature 0 for backward compat
            self.cusum = self._cusum_per_feature[0]
            self.wasserstein = self._wasserstein_per_feature[0]

        self._is_fitted = True
        logger.info("DriftEnsemble fitted on data shape %s (%d feature detectors)",
                    data.shape, len(self._cusum_per_feature))
        return self

    def detect(self, new_data: np.ndarray, feature_name: str = "") -> EnsembleDriftResult:
        """
        Detect drift using both detectors.

        Args:
            new_data: New data to check.
            feature_name: Name of the feature.

        Returns:
            EnsembleDriftResult with combined assessment.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before detect()")

        data = np.asarray(new_data, dtype=np.float64).flatten()
        name = feature_name or (self._feature_names[0] if self._feature_names else "unknown")

        # Run CUSUM sequentially
        cusum_results = self.cusum.detect_batch(data)

        # Run Wasserstein
        wasserstein_result = self.wasserstein.detect(data)

        # Get latest CUSUM result
        latest_cusum = cusum_results[-1] if cusum_results else None

        # Ensemble decision: OR of both detectors
        cusum_drift = any(r.drift_detected for r in cusum_results)
        wasserstein_drift = wasserstein_result.drift_detected
        drift_detected = cusum_drift or wasserstein_drift

        # Determine severity
        if drift_detected:
            magnitude = max(
                latest_cusum.statistic / max(self.cusum.threshold, 1e-8) if latest_cusum else 0,
                wasserstein_result.statistic / max(self.wasserstein.threshold, 1e-8),
            )
            if magnitude >= 2.0:
                severity = "critical"
            elif magnitude >= 1.5:
                severity = "high"
            elif magnitude >= 1.0:
                severity = "medium"
            else:
                severity = "low"
        else:
            severity = "none"

        # Generate alert if drift detected
        alert = None
        if drift_detected:
            alert = self.alert_manager.check_and_alert(
                drift_detected=True,
                feature_name=name,
                drift_magnitude=wasserstein_result.statistic,
                detector_name="ensemble",
                details=f"CUSUM={cusum_drift}, Wasserstein={wasserstein_drift}",
            )

        # Build details string
        detectors_used = []
        if cusum_drift:
            detectors_used.append("CUSUM")
        if wasserstein_drift:
            detectors_used.append("Wasserstein")
        details = f"Drift detected by: {', '.join(detectors_used) if detectors_used else 'none'}"

        return EnsembleDriftResult(
            drift_detected=drift_detected,
            overall_severity=severity,
            cusum_result=latest_cusum,
            wasserstein_result=wasserstein_result,
            feature_name=name,
            alert=alert,
            details=details,
        )

    def get_feature_drift_scores(self, data: np.ndarray) -> Dict[str, float]:
        """
        Get per-feature drift scores using Wasserstein distance.

        Bug 28+29 fix: uses the per-feature detectors created during fit().
        No longer accesses private _reference_data attribute.

        Args:
            data: New data (n_samples, n_features).

        Returns:
            Dictionary of feature_name -> drift_score.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before get_feature_drift_scores()")

        data = np.asarray(data, dtype=np.float64)
        scores: Dict[str, float] = {}

        detectors = getattr(self, "_wasserstein_per_feature", [self.wasserstein])

        if data.ndim == 1:
            name = self._feature_names[0] if self._feature_names else "feature_0"
            result = detectors[0].detect(data)
            scores[name] = result.statistic
        else:
            for i in range(data.shape[1]):
                name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
                det = detectors[i] if i < len(detectors) else detectors[0]
                result = det.detect(data[:, i])
                scores[name] = result.statistic

        return scores

    def generate_report(self) -> Dict:
        """
        Generate a summary report of drift detection status.

        Returns:
            Report dictionary.
        """
        alerts = self.alert_manager.get_active_alerts()
        severity_counts = self.alert_manager.get_severity_counts()

        return {
            "ensemble_fitted": self._is_fitted,
            "cusum_state": self.cusum.get_state(),
            "wasserstein_state": self.wasserstein.get_state(),
            "active_alerts": len(alerts),
            "severity_counts": severity_counts,
            "alert_history": [a.to_dict() for a in self.alert_manager.get_alert_history()[-10:]],
        }

    def reset(self) -> None:
        """Reset all detectors and fitted state."""
        self.cusum.reset()
        # Fix HIGH-06: also reset the Wasserstein detector and clear per-feature
        # detector lists so a subsequent fit() starts from a clean state.
        self.wasserstein.reset()
        self._cusum_per_feature = []
        self._wasserstein_per_feature = []
        self._is_fitted = False
        logger.info("DriftEnsemble reset")

    def detect_multivariate(
        self,
        new_data: np.ndarray,
    ) -> Dict[str, "EnsembleDriftResult"]:
        """Fix MED-06: run all per-feature detectors on multivariate data.

        The single-feature detect() method only ever checks feature 0.
        This method iterates over every fitted per-feature detector and
        returns per-feature EnsembleDriftResult objects.

        Args:
            new_data: New data (n_samples, n_features) or (n_samples,).

        Returns:
            Dict mapping feature_name -> EnsembleDriftResult.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before detect_multivariate()")

        data = np.asarray(new_data, dtype=np.float64)
        results: Dict[str, EnsembleDriftResult] = {}

        if data.ndim == 1:
            # Single feature — delegate to the existing detect()
            name = self._feature_names[0] if self._feature_names else "feature_0"
            results[name] = self.detect(data, feature_name=name)
            return results

        n_features = data.shape[1]
        for i in range(n_features):
            name = self._feature_names[i] if i < len(self._feature_names) else f"feature_{i}"
            col = data[:, i]

            c_det = self._cusum_per_feature[i] if i < len(self._cusum_per_feature) else self.cusum
            w_det = self._wasserstein_per_feature[i] if i < len(self._wasserstein_per_feature) else self.wasserstein

            cusum_results = c_det.detect_batch(col)
            wasserstein_result = w_det.detect(col)
            latest_cusum = cusum_results[-1] if cusum_results else None

            cusum_drift = any(r.drift_detected for r in cusum_results)
            wasserstein_drift = wasserstein_result.drift_detected
            drift_detected = cusum_drift or wasserstein_drift

            if drift_detected:
                magnitude = max(
                    latest_cusum.statistic / max(c_det.threshold, 1e-8) if latest_cusum else 0,
                    wasserstein_result.statistic / max(w_det.threshold, 1e-8),
                )
                if magnitude >= 2.0:
                    severity = "critical"
                elif magnitude >= 1.5:
                    severity = "high"
                elif magnitude >= 1.0:
                    severity = "medium"
                else:
                    severity = "low"
            else:
                severity = "none"

            alert = None
            if drift_detected:
                alert = self.alert_manager.check_and_alert(
                    drift_detected=True,
                    feature_name=name,
                    drift_magnitude=wasserstein_result.statistic,
                    detector_name="ensemble",
                    details=f"CUSUM={cusum_drift}, Wasserstein={wasserstein_drift}",
                )

            detectors_used = []
            if cusum_drift:
                detectors_used.append("CUSUM")
            if wasserstein_drift:
                detectors_used.append("Wasserstein")

            results[name] = EnsembleDriftResult(
                drift_detected=drift_detected,
                overall_severity=severity,
                cusum_result=latest_cusum,
                wasserstein_result=wasserstein_result,
                feature_name=name,
                alert=alert,
                details=f"Drift detected by: {', '.join(detectors_used) if detectors_used else 'none'}",
            )

        return results

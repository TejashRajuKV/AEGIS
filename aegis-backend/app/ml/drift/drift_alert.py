"""
Drift Alert Manager
===================
Manages drift alerts with severity levels and recommendations.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional


logger = logging.getLogger("aegis.drift.alert")


class AlertSeverity(Enum):
    """Severity levels for drift alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """A drift alert with severity and metadata."""

    id: str = ""
    feature_name: str = ""
    severity: AlertSeverity = AlertSeverity.LOW
    drift_magnitude: float = 0.0
    detector_name: str = ""
    recommendation: str = ""
    timestamp: float = 0.0
    acknowledged: bool = False
    resolved: bool = False
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "feature_name": self.feature_name,
            "severity": self.severity.value,
            "drift_magnitude": self.drift_magnitude,
            "detector_name": self.detector_name,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "details": self.details,
        }


class DriftAlertManager:
    """
    Manages drift alerts with severity classification and recommendations.

    Severity levels:
    - LOW: Minor shift, monitoring recommended
    - MEDIUM: Moderate shift, investigation suggested
    - HIGH: Significant shift, action required
    - CRITICAL: Severe drift, immediate intervention needed
    """

    # Severity thresholds based on drift magnitude
    SEVERITY_THRESHOLDS = {
        AlertSeverity.LOW: 0.05,
        AlertSeverity.MEDIUM: 0.15,
        AlertSeverity.HIGH: 0.30,
        AlertSeverity.CRITICAL: 0.50,
    }

    RECOMMENDATIONS = {
        AlertSeverity.LOW: "Continue monitoring. No action required at this time.",
        AlertSeverity.MEDIUM: "Investigate the source of drift. Consider updating reference data.",
        AlertSeverity.HIGH: "Significant drift detected. Consider model retraining or threshold adjustment.",
        AlertSeverity.CRITICAL: "Severe drift detected! Immediate model retraining or pipeline reset recommended.",
    }

    def __init__(self, max_history: int = 1000):
        """
        Initialize the alert manager.

        Args:
            max_history: Maximum number of alerts to retain.
        """
        self.max_history = max_history
        self._alerts: Deque[DriftAlert] = deque(maxlen=max_history)

    def _classify_severity(
        self, drift_magnitude: float
    ) -> AlertSeverity:
        """Classify severity based on drift magnitude."""
        for severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH,
                         AlertSeverity.MEDIUM, AlertSeverity.LOW]:
            if drift_magnitude >= self.SEVERITY_THRESHOLDS[severity]:
                return severity
        return AlertSeverity.LOW

    def check_and_alert(
        self,
        drift_detected: bool,
        feature_name: str = "",
        drift_magnitude: float = 0.0,
        detector_name: str = "",
        details: str = "",
    ) -> Optional[DriftAlert]:
        """
        Check drift result and create alert if warranted.

        Args:
            drift_detected: Whether drift was detected.
            feature_name: Name of the feature.
            drift_magnitude: Magnitude of the drift (0-1 normalized or raw).
            detector_name: Name of the detector that found it.
            details: Additional details.

        Returns:
            DriftAlert if drift was detected, None otherwise.
        """
        if not drift_detected:
            return None

        severity = self._classify_severity(drift_magnitude)
        recommendation = self.RECOMMENDATIONS[severity]

        alert = DriftAlert(
            id=str(uuid.uuid4())[:8],
            feature_name=feature_name,
            severity=severity,
            drift_magnitude=drift_magnitude,
            detector_name=detector_name,
            recommendation=recommendation,
            timestamp=time.time(),
            details=details,
        )

        self._alerts.append(alert)

        log_level = {
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.HIGH: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
        }.get(severity, logging.INFO)

        logger.log(
            log_level,
            "Drift alert [%s]: feature='%s', magnitude=%.4f, detector=%s",
            severity.value, feature_name, drift_magnitude, detector_name,
        )

        return alert

    def get_alert_history(self) -> List[DriftAlert]:
        """Get all alert history."""
        return list(self._alerts)

    def get_active_alerts(self) -> List[DriftAlert]:
        """Get unresolved alerts."""
        return [a for a in self._alerts if not a.resolved]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info("Alert %s acknowledged", alert_id)
                return True
        return False

    def clear_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info("Alert %s resolved", alert_id)
                return True
        return False

    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of alerts by severity."""
        counts = {s.value: 0 for s in AlertSeverity}
        for alert in self._alerts:
            counts[alert.severity.value] += 1
        return counts

    def clear_all(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()

"""
Drift Record
=============
Dataclass-based model for drift detection records with JSON persistence.
Provides save/load functionality without requiring SQLAlchemy or any database
dependency. Each record captures a single drift detection event for a feature.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aegis.models.drift_record")


@dataclass
class DriftRecord:
    """
    A single drift detection record.

    Captures the results of a drift detection check for one feature,
    including which detector found the drift, the statistic value,
    severity level, and any additional details.

    Attributes
    ----------
    id : int
        Unique identifier for this record (auto-incremented on save).
    feature_name : str
        Name of the monitored feature.
    detector_type : str
        Type of detector that produced this record.
        One of: 'cusum', 'wasserstein', 'ensemble'.
    drift_detected : bool
        Whether drift was detected for this feature.
    statistic : float
        The test statistic value from the detector.
    severity : str
        Severity level: 'none', 'low', 'medium', 'high', 'critical'.
    timestamp : str
        ISO-format timestamp of when the record was created.
    details : str
        Additional details or free-text description of the detection.
    """

    id: int = 0
    feature_name: str = ""
    detector_type: str = "ensemble"
    drift_detected: bool = False
    statistic: float = 0.0
    severity: str = "none"
    timestamp: str = ""
    details: str = ""

    # Internal tracking
    _id_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.severity:
            self.severity = self._infer_severity()

    def _infer_severity(self) -> str:
        """Infer severity from statistic and drift_detected flag."""
        if not self.drift_detected:
            return "none"

        if self.statistic >= 0.5:
            return "critical"
        elif self.statistic >= 0.3:
            return "high"
        elif self.statistic >= 0.15:
            return "medium"
        else:
            return "low"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the record to a dictionary.

        Returns:
            Dictionary with all record fields.
        """
        return {
            "id": self.id,
            "feature_name": self.feature_name,
            "detector_type": self.detector_type,
            "drift_detected": self.drift_detected,
            "statistic": self.statistic,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftRecord":
        """
        Create a DriftRecord from a dictionary.

        Args:
            data: Dictionary with record fields.

        Returns:
            DriftRecord instance.
        """
        return cls(
            id=int(data.get("id", 0)),
            feature_name=str(data.get("feature_name", "")),
            detector_type=str(data.get("detector_type", "ensemble")),
            drift_detected=bool(data.get("drift_detected", False)),
            statistic=float(data.get("statistic", 0.0)),
            severity=str(data.get("severity", "none")),
            timestamp=str(data.get("timestamp", "")),
            details=str(data.get("details", "")),
        )

    def save(self, filepath: str) -> str:
        """
        Save this record to a JSON file.

        Args:
            filepath: Path to write the JSON file.

        Returns:
            The filepath that was written.
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        data = self.to_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug("DriftRecord saved to %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "DriftRecord":
        """
        Load a DriftRecord from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            DriftRecord instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        record = cls.from_dict(data)
        logger.debug("DriftRecord loaded from %s", filepath)
        return record


class DriftRecordStore:
    """
    Manages a collection of DriftRecords with JSON-based persistence.

    Provides CRUD operations and query methods for drift records
    without requiring a database backend.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        """
        Initialize the store.

        Args:
            storage_path: Optional path to a JSON file for persistent storage.
                          If None, records are kept in memory only.
        """
        self._records: List[DriftRecord] = []
        self._next_id: int = 1
        self._storage_path = storage_path

        if storage_path and os.path.exists(storage_path):
            self._load_from_file(storage_path)

    def add(self, record: DriftRecord) -> DriftRecord:
        """
        Add a new record to the store.

        Args:
            record: DriftRecord to add (id is auto-assigned if 0).

        Returns:
            The stored record with assigned ID.
        """
        if record.id == 0:
            record.id = self._next_id
            self._next_id += 1

        self._records.append(record)
        self._persist()
        logger.info(
            "DriftRecord added: id=%d, feature='%s', drift=%s, severity=%s",
            record.id, record.feature_name, record.drift_detected, record.severity,
        )
        return record

    def get(self, record_id: int) -> Optional[DriftRecord]:
        """
        Get a record by ID.

        Args:
            record_id: The record ID to retrieve.

        Returns:
            DriftRecord or None if not found.
        """
        for record in self._records:
            if record.id == record_id:
                return record
        return None

    def list_all(self) -> List[DriftRecord]:
        """List all records in the store."""
        return list(self._records)

    def query(
        self,
        feature_name: Optional[str] = None,
        detector_type: Optional[str] = None,
        drift_detected: Optional[bool] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[DriftRecord]:
        """
        Query records with optional filters.

        Args:
            feature_name: Filter by feature name (exact match).
            detector_type: Filter by detector type.
            drift_detected: Filter by drift detection status.
            severity: Filter by severity level.
            limit: Maximum number of records to return.

        Returns:
            List of matching DriftRecords.
        """
        results = self._records

        if feature_name is not None:
            results = [r for r in results if r.feature_name == feature_name]

        if detector_type is not None:
            results = [r for r in results if r.detector_type == detector_type]

        if drift_detected is not None:
            results = [r for r in results if r.drift_detected == drift_detected]

        if severity is not None:
            results = [r for r in results if r.severity == severity]

        # Sort by ID (most recent last) and apply limit
        results = sorted(results, key=lambda r: r.id)
        return results[-limit:]

    def count(self) -> int:
        """Return the total number of records."""
        return len(self._records)

    def count_drifted(self) -> int:
        """Return the number of records where drift was detected."""
        return sum(1 for r in self._records if r.drift_detected)

    def get_severity_summary(self) -> Dict[str, int]:
        """Get count of records by severity level."""
        summary: Dict[str, int] = {
            "none": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }
        for record in self._records:
            sev = record.severity.lower()
            if sev in summary:
                summary[sev] += 1
        return summary

    def get_feature_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get drift summary per feature."""
        summary: Dict[str, Dict[str, Any]] = {}
        for record in self._records:
            fname = record.feature_name
            if fname not in summary:
                summary[fname] = {
                    "total_checks": 0,
                    "drift_detected": 0,
                    "no_drift": 0,
                    "latest_severity": "none",
                    "latest_statistic": 0.0,
                    "latest_timestamp": "",
                }
            summary[fname]["total_checks"] += 1
            if record.drift_detected:
                summary[fname]["drift_detected"] += 1
            else:
                summary[fname]["no_drift"] += 1
            # Keep latest info
            summary[fname]["latest_severity"] = record.severity
            summary[fname]["latest_statistic"] = record.statistic
            summary[fname]["latest_timestamp"] = record.timestamp
        return summary

    def delete(self, record_id: int) -> bool:
        """
        Delete a record by ID.

        Args:
            record_id: The record ID to delete.

        Returns:
            True if found and deleted, False otherwise.
        """
        for i, record in enumerate(self._records):
            if record.id == record_id:
                del self._records[i]
                self._persist()
                logger.info("DriftRecord deleted: id=%d", record_id)
                return True
        return False

    def clear(self) -> int:
        """
        Clear all records from the store.

        Returns:
            Number of records that were removed.
        """
        count = len(self._records)
        self._records.clear()
        self._next_id = 1
        self._persist()
        logger.info("DriftRecordStore cleared: %d records removed", count)
        return count

    def export_json(self, filepath: str) -> str:
        """
        Export all records as a JSON array.

        Args:
            filepath: Path to write the JSON file.

        Returns:
            The filepath that was written.
        """
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        data = [record.to_dict() for record in self._records]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Exported %d records to %s", len(data), filepath)
        return filepath

    def import_json(self, filepath: str) -> int:
        """
        Import records from a JSON array file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Number of records imported.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for item in data:
            if isinstance(item, dict):
                record = DriftRecord.from_dict(item)
                self.add(record)
                count += 1

        logger.info("Imported %d records from %s", count, filepath)
        return count

    def _persist(self) -> None:
        """Persist records to storage file if path is set."""
        if self._storage_path:
            try:
                self.export_json(self._storage_path)
            except Exception as e:
                logger.error("Failed to persist drift records: %s", e)

    def _load_from_file(self, filepath: str) -> None:
        """Load records from a JSON file on initialization."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                if isinstance(item, dict):
                    record = DriftRecord.from_dict(item)
                    self._records.append(record)
                    if record.id >= self._next_id:
                        self._next_id = record.id + 1

            logger.info("Loaded %d drift records from %s", len(self._records), filepath)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Could not load drift records from %s: %s", filepath, e)

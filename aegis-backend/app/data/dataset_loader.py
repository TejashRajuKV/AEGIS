"""AEGIS Dataset Loader - Load and cache datasets from CSV files."""
import os
import threading
from pathlib import Path

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Dataset registry with metadata
DATASET_REGISTRY: Dict[str, Dict] = {
    "adult_census": {
        "filename": "adult_census.csv",
        "sensitive_attributes": ["sex", "race"],
        "target_column": "income",
        "description": "Adult Census Income dataset for predicting income >50K",
    },
    "compas": {
        "filename": "compas_recidivism.csv",
        "sensitive_attributes": ["race", "sex"],
        "target_column": "is_recid",
        "description": "COMPAS Recidivism dataset for recidivism prediction",
    },
    "german_credit": {
        "filename": "german_credit.csv",
        "sensitive_attributes": ["sex", "age_group"],
        "target_column": "credit_risk",
        "description": "German Credit Risk dataset",
    },
}


class DatasetLoader:
    """Load and manage datasets from CSV files."""

    # Bug 30 fix: resolve default data_dir from settings, not a relative path.
    # Falls back to the relative path only if settings import fails.
    @staticmethod
    def _default_data_dir() -> str:
        try:
            from app.config import get_settings
            s = get_settings()
            return str(getattr(s, "DATA_DIR", None) or "../aegis-shared/datasets")
        except Exception:
            return "../aegis-shared/datasets"

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or self._default_data_dir()
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load a dataset by name from the registry.

        Args:
            name: Dataset name (adult_census, compas, german_credit).

        Returns:
            Loaded pandas DataFrame.

        Raises:
            ValueError: If dataset name is not in registry.
            FileNotFoundError: If CSV file is not found.
        """
        if name not in DATASET_REGISTRY:
            available = list(DATASET_REGISTRY.keys())
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {available}"
            )

        if name in self._cache:
            logger.info(f"Loading dataset '{name}' from cache")
            return self._cache[name].copy()

        meta = DATASET_REGISTRY[name]
        # Fix HIGH-02: use pathlib so Windows backslash data_dirs don't produce
        # invalid mixed-separator paths like "C:\data/file.csv".
        filepath = str(Path(self.data_dir) / meta["filename"])

        logger.info(f"Loading dataset '{name}' from {filepath}")
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {filepath}")
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}. "
                f"Please run the dataset download script first."
            )

        self._cache[name] = df
        logger.info(f"Loaded '{name}': {df.shape[0]} rows, {df.shape[1]} columns")
        return df.copy()

    def list_datasets(self) -> Dict[str, Dict]:
        """List all available datasets with metadata."""
        return {
            name: {
                "description": meta["description"],
                "sensitive_attributes": meta["sensitive_attributes"],
                "target_column": meta["target_column"],
            }
            for name, meta in DATASET_REGISTRY.items()
        }

    def get_dataset_info(self, name: str) -> Dict:
        """Get metadata for a specific dataset."""
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset '{name}'")
        meta = DATASET_REGISTRY[name]
        df = self.load_dataset(name)
        return {
            "name": name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "sensitive_attributes": meta["sensitive_attributes"],
            "target_column": meta["target_column"],
            "description": meta["description"],
        }

    def load_custom_dataset(self, filepath: str) -> pd.DataFrame:
        """Load a custom dataset from an arbitrary file path.

        Note: filepath must be an absolute path on the server filesystem.
        Route handlers must validate and sanitise the path before passing it
        here to prevent path-traversal attacks (SEC-04).
        """
        logger.info(f"Loading custom dataset from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded custom dataset: {df.shape}")
        # Fix MED-10: return a copy so callers cannot mutate an internal cache entry.
        return df.copy()

    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self._cache.clear()
        logger.info("Dataset cache cleared")


# Singleton instance
_dataset_loader: Optional[DatasetLoader] = None
# Fix MED-04: guard singleton creation with a lock so two threads that both
# see _dataset_loader is None don't create separate instances.
_loader_lock = threading.Lock()


def get_dataset_loader(data_dir: Optional[str] = None) -> DatasetLoader:
    """Get or create the singleton DatasetLoader.

    Bug 31 fix: if a non-None data_dir is requested and the singleton was
    already created with a different path, log a warning and re-create it
    so callers that pass an explicit path always get the right directory.

    Fix MED-04: protected by a threading.Lock to prevent duplicate instance
    creation under concurrent calls.
    """
    global _dataset_loader
    with _loader_lock:
        if _dataset_loader is None:
            _dataset_loader = DatasetLoader(data_dir=data_dir)
        elif data_dir is not None and data_dir != _dataset_loader.data_dir:
            logger.warning(
                "get_dataset_loader() called with data_dir='%s' but singleton already "
                "uses data_dir='%s'. Re-creating singleton with new path.",
                data_dir,
                _dataset_loader.data_dir,
            )
            _dataset_loader = DatasetLoader(data_dir=data_dir)
        return _dataset_loader

"""
German Credit Dataset Schema
=============================
Schema definition, validation, and metadata for the German Credit dataset.

The German Credit dataset contains 20 attributes and a binary class label
(good/bad credit risk). Attribute9 (personal status/sex) and Attribute10
(age) serve as sensitive attributes for fairness analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("aegis.data.schemas.german_credit")


class GermanCreditSchema:
    """
    Schema definition and validator for the German Credit dataset.

    The dataset has 20 attributes (Attribute1 through Attribute20) plus
    a binary class label. Attributes are a mix of categorical and numeric.

    Attributes
    ----------
    COLUMNS : list[str]
        Expected column names in the dataset.
    NUMERIC_COLUMNS : list[str]
        Columns that should contain numeric values.
    CATEGORICAL_COLUMNS : list[str]
        Columns that should contain categorical values.
    TARGET_COLUMN : str
        Name of the target/class column.
    SENSITIVE_COLUMNS : list[str]
        Columns identified as sensitive attributes for fairness analysis.
    """

    COLUMNS: List[str] = [
        "Attribute1",   # Status of existing checking account
        "Attribute2",   # Duration in months
        "Attribute3",   # Credit history
        "Attribute4",   # Purpose
        "Attribute5",   # Credit amount
        "Attribute6",   # Savings account/bonds
        "Attribute7",   # Present employment since
        "Attribute8",   # Installment rate in percentage of disposable income
        "Attribute9",   # Personal status and sex (SENSITIVE: gender)
        "Attribute10",  # Other debtors / guarantors (age proxy often used)
        "Attribute11",  # Present residence since
        "Attribute12",  # Property
        "Attribute13",  # Age in years
        "Attribute14",  # Other installment plans
        "Attribute15",  # Housing
        "Attribute16",  # Number of existing credits at this bank
        "Attribute17",  # Job
        "Attribute18",  # Number of people being liable to provide maintenance
        "Attribute19",  # Telephone
        "Attribute20",  # Foreign worker
        "class",        # Target: 1 = good, 2 = bad
    ]

    NUMERIC_COLUMNS: List[str] = [
        "Attribute2",   # Duration in months
        "Attribute5",   # Credit amount
        "Attribute8",   # Installment rate
        "Attribute11",  # Present residence since
        "Attribute13",  # Age in years
        "Attribute16",  # Number of existing credits
        "Attribute18",  # Number of people liable
    ]

    CATEGORICAL_COLUMNS: List[str] = [
        # All Attribute columns that are NOT in NUMERIC_COLUMNS
        "Attribute1",   # Status of existing checking account
        "Attribute3",   # Credit history
        "Attribute4",   # Purpose
        "Attribute6",   # Savings account/bonds
        "Attribute7",   # Present employment since
        "Attribute9",   # Personal status and sex (SENSITIVE)
        "Attribute10",  # Other debtors/guarantors
        "Attribute12",  # Property
        "Attribute14",  # Other installment plans
        "Attribute15",  # Housing
        "Attribute17",  # Job
        "Attribute19",  # Telephone
        "Attribute20",  # Foreign worker
    ]

    TARGET_COLUMN: str = "class"

    SENSITIVE_COLUMNS: List[str] = ["Attribute9", "Attribute10"]

    # Target value mapping
    TARGET_VALUES: Dict[str, int] = {
        "good": 1,
        "bad": 2,
    }

    # Expected shapes
    N_SAMPLES: int = 1000
    N_FEATURES: int = 20

    # Class distribution (approximate for reference)
    CLASS_DISTRIBUTION: Dict[int, float] = {
        1: 0.70,   # 70% good credit
        2: 0.30,   # 30% bad credit
    }

    def __init__(self) -> None:
        """Initialize the schema."""
        self._expected_columns = set(self.COLUMNS)
        self._expected_numeric = set(self.NUMERIC_COLUMNS)
        self._expected_categorical = set(self.CATEGORICAL_COLUMNS)
        self._sensitive_set = set(self.SENSITIVE_COLUMNS)

    def validate(self, df: Any) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame against the German Credit schema.

        Args:
            df: pandas DataFrame to validate.

        Returns:
            Tuple of (is_valid: bool, errors: list of error descriptions).
        """
        errors: List[str] = []

        try:
            import pandas as pd

            if not isinstance(df, pd.DataFrame):
                errors.append(
                    f"Expected pandas DataFrame, got {type(df).__name__}"
                )
                return False, errors

            # Check for required columns
            actual_columns = set(df.columns.tolist())
            missing_columns = self._expected_columns - actual_columns
            extra_columns = actual_columns - self._expected_columns

            if missing_columns:
                errors.append(
                    f"Missing columns: {sorted(missing_columns)}"
                )

            if extra_columns:
                # Extra columns are a warning, not an error
                logger.info(
                    "Extra columns found (will be ignored): %s",
                    sorted(extra_columns),
                )

            if missing_columns and len(missing_columns) > 5:
                # If too many columns are missing, the column naming convention
                # might be different (e.g., already preprocessed)
                errors.append(
                    "Too many missing columns. The dataset may use different "
                    "column names. Consider renaming columns to match the schema."
                )
                return False, errors

            # Check target column exists
            if self.TARGET_COLUMN not in df.columns:
                errors.append(
                    f"Target column '{self.TARGET_COLUMN}' not found"
                )
                return False, errors

            # Check target values
            unique_targets = df[self.TARGET_COLUMN].unique()
            valid_targets = set(self.TARGET_VALUES.values())
            invalid_targets = set(unique_targets) - valid_targets
            if invalid_targets:
                errors.append(
                    f"Invalid target values: {invalid_targets}. "
                    f"Expected values from {valid_targets}"
                )

            # Check numeric columns contain numeric data
            for col in self.NUMERIC_COLUMNS:
                if col in df.columns:
                    if not np.issubdtype(df[col].dtype, np.number):
                        errors.append(
                            f"Column '{col}' should be numeric but has dtype {df[col].dtype}"
                        )

            # Check for null values
            null_counts = df[self.COLUMNS].isnull().sum() if all(c in df.columns for c in self.COLUMNS) else df.isnull().sum()
            total_nulls = int(null_counts.sum())
            if total_nulls > 0:
                null_cols = null_counts[null_counts > 0].index.tolist()
                errors.append(
                    f"Found {total_nulls} null values in columns: {null_cols}"
                )

            # Check minimum row count
            if len(df) < 10:
                errors.append(
                    f"Dataset has only {len(df)} rows; minimum expected is 10"
                )

            # Check sensitive columns exist
            for col in self.SENSITIVE_COLUMNS:
                if col not in df.columns:
                    errors.append(
                        f"Sensitive column '{col}' not found in dataset"
                    )

        except ImportError:
            errors.append("pandas is required for validation")
            return False, errors

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed metadata for each feature/column.

        Returns:
            Dict mapping column name to metadata dict.
        """
        feature_info: Dict[str, Dict[str, Any]] = {}

        # Attribute descriptions
        descriptions = {
            "Attribute1": {
                "name": "Status of existing checking account",
                "type": "categorical",
                "values": [
                    "A11: < 0 DM", "A12: 0 <= ... < 200 DM",
                    "A13: >= 200 DM / salary for >= 1 year", "A14: no checking account"
                ],
            },
            "Attribute2": {
                "name": "Duration in months",
                "type": "numeric",
                "range": (4, 72),
                "unit": "months",
            },
            "Attribute3": {
                "name": "Credit history",
                "type": "categorical",
                "values": [
                    "A30: no credits taken", "A31: all credits paid back duly",
                    "A32: existing credits paid back duly till now",
                    "A33: delay in paying off in the past",
                    "A34: critical account / other credits existing",
                ],
            },
            "Attribute4": {
                "name": "Purpose",
                "type": "categorical",
                "values": [
                    "A40: car (new)", "A41: car (used)",
                    "A42: furniture/equipment", "A43: radio/television",
                    "A44: domestic appliances", "A45: repairs",
                    "A46: education", "A47: vacation",
                    "A48: retraining", "A49: business",
                    "A410: others",
                ],
            },
            "Attribute5": {
                "name": "Credit amount",
                "type": "numeric",
                "range": (250, 18424),
                "unit": "DM",
            },
            "Attribute6": {
                "name": "Savings account/bonds",
                "type": "categorical",
                "values": [
                    "A61: < 100 DM", "A62: 100 <= ... < 500 DM",
                    "A63: 500 <= ... < 1000 DM", "A64: >= 1000 DM",
                    "A65: unknown/no savings account",
                ],
            },
            "Attribute7": {
                "name": "Present employment since",
                "type": "categorical",
                "values": [
                    "A71: unemployed", "A72: < 1 year",
                    "A73: 1 <= ... < 4 years", "A74: 4 <= ... < 7 years",
                    "A75: >= 7 years",
                ],
            },
            "Attribute8": {
                "name": "Installment rate in percentage of disposable income",
                "type": "numeric",
                "range": (1, 4),
                "unit": "percentage",
            },
            "Attribute9": {
                "name": "Personal status and sex",
                "type": "categorical",
                "sensitive": True,
                "sensitive_reason": "Contains gender information",
                "values": [
                    "A91: male divorced/separated",
                    "A92: female divorced/separated/married",
                    "A93: male single",
                    "A94: male married/widowed",
                    "A95: female single",
                ],
            },
            "Attribute10": {
                "name": "Other debtors/guarantors",
                "type": "categorical",
                "sensitive": True,
                "sensitive_reason": "Used as age proxy in some analyses",
                "values": [
                    "A101: none", "A102: co-applicant", "A103: guarantor",
                ],
            },
            "Attribute11": {
                "name": "Present residence since",
                "type": "numeric",
                "range": (1, 4),
                "unit": "years",
            },
            "Attribute12": {
                "name": "Property",
                "type": "categorical",
                "values": [
                    "A121: real estate", "A122: building society savings/life insurance",
                    "A123: car or other", "A124: unknown/no property",
                ],
            },
            "Attribute13": {
                "name": "Age in years",
                "type": "numeric",
                "range": (19, 75),
                "unit": "years",
                "sensitive": True,
                "sensitive_reason": "Age is a protected attribute",
            },
            "Attribute14": {
                "name": "Other installment plans",
                "type": "categorical",
                "values": [
                    "A141: bank", "A142: stores", "A143: none",
                ],
            },
            "Attribute15": {
                "name": "Housing",
                "type": "categorical",
                "values": [
                    "A151: rent", "A152: own", "A153: for free",
                ],
            },
            "Attribute16": {
                "name": "Number of existing credits at this bank",
                "type": "numeric",
                "range": (1, 4),
            },
            "Attribute17": {
                "name": "Job",
                "type": "categorical",
                "values": [
                    "A171: unemployed/unskilled non-resident",
                    "A172: unskilled resident", "A173: skilled employee/official",
                    "A174: management/self-employed/highly qualified",
                ],
            },
            "Attribute18": {
                "name": "Number of people being liable to provide maintenance for",
                "type": "numeric",
                "range": (1, 2),
            },
            "Attribute19": {
                "name": "Telephone",
                "type": "categorical",
                "values": ["A191: none", "A192: yes, registered under customer name"],
            },
            "Attribute20": {
                "name": "Foreign worker",
                "type": "categorical",
                "sensitive": True,
                "sensitive_reason": "Nationality/immigration status",
                "values": ["A201: yes", "A202: no"],
            },
            "class": {
                "name": "Credit risk",
                "type": "binary_target",
                "values": {1: "good", 2: "bad"},
                "description": "1 = good credit risk, 2 = bad credit risk",
            },
        }

        for col in self.COLUMNS:
            if col in descriptions:
                feature_info[col] = descriptions[col]
            else:
                if col in self.NUMERIC_COLUMNS:
                    feature_info[col] = {"name": col, "type": "numeric"}
                else:
                    feature_info[col] = {"name": col, "type": "categorical"}

            # Add schema classification
            if col in self.NUMERIC_COLUMNS:
                feature_info[col]["schema_type"] = "numeric"
            elif col in self.CATEGORICAL_COLUMNS:
                feature_info[col]["schema_type"] = "categorical"
            elif col == self.TARGET_COLUMN:
                feature_info[col]["schema_type"] = "target"

            if col in self.SENSITIVE_COLUMNS:
                feature_info[col]["is_sensitive"] = True

        return feature_info

    def get_sensitive_feature_indices(self, df: Any) -> Dict[str, int]:
        """
        Get column indices of sensitive features in a DataFrame.

        Args:
            df: pandas DataFrame.

        Returns:
            Dict mapping sensitive column name to its column index in the feature matrix.
        """
        indices = {}
        feature_cols = [c for c in self.COLUMNS if c != self.TARGET_COLUMN]
        for col in self.SENSITIVE_COLUMNS:
            if col in df.columns:
                idx = list(df.columns).index(col)
                indices[col] = idx
            elif col in feature_cols:
                # Use position relative to schema
                idx = feature_cols.index(col)
                indices[col] = idx
        return indices

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get recommended preprocessing configuration for the dataset.

        Returns:
            Dict with preprocessing parameters.
        """
        return {
            "dataset_name": "german_credit",
            "n_features": self.N_FEATURES,
            "n_samples_expected": self.N_SAMPLES,
            "numeric_columns": self.NUMERIC_COLUMNS,
            "categorical_columns": self.CATEGORICAL_COLUMNS,
            "target_column": self.TARGET_COLUMN,
            "sensitive_columns": self.SENSITIVE_COLUMNS,
            "target_mapping": {"good": 1, "bad": 2},
            "recommended_scaling": "standard",  # StandardScaler for numeric features
            "recommended_encoding": "onehot",   # OneHotEncoder for categorical features
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
        }

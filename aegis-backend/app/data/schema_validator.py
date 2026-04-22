"""AEGIS Schema Validator - Validate CSV datasets against expected schemas."""
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validate datasets against expected column names and types."""

    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}

    def register_schema(
        self, name: str, required_columns: List[str], optional_columns: Optional[List[str]] = None
    ) -> None:
        """Register a schema with required and optional columns."""
        self.schemas[name] = {
            "required": required_columns,
            "optional": optional_columns or [],
        }

    def validate(self, df: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
        """Validate a DataFrame against a registered schema.

        Args:
            df: DataFrame to validate.
            schema_name: Name of registered schema.

        Returns:
            Validation result dict with 'valid', 'missing', 'extra' keys.

        Raises:
            ValueError: If schema name is not registered.
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {schema_name}")

        schema = self.schemas[schema_name]
        required = set(schema["required"])
        optional = set(schema["optional"])
        present = set(df.columns)

        missing = required - present
        extra = present - required - optional

        result = {
            "valid": len(missing) == 0,
            "missing_columns": sorted(missing),
            "extra_columns": sorted(extra),
            "warnings": [],
        }

        if extra:
            result["warnings"].append(
                f"Extra columns found (will be ignored): {sorted(extra)}"
            )

        # Check for null values in required columns
        null_counts = df[list(required)].isnull().sum() if required else pd.Series()
        null_cols = null_counts[null_counts > 0].to_dict()
        if null_cols:
            result["warnings"].append(f"Null values found: {null_cols}")

        logger.info(f"Schema validation for '{schema_name}': valid={result['valid']}")
        return result


# Pre-register known schemas
_default_validator: Optional[SchemaValidator] = None


def get_schema_validator() -> SchemaValidator:
    """Get default schema validator with pre-registered schemas."""
    global _default_validator
    if _default_validator is None:
        _default_validator = SchemaValidator()
        _default_validator.register_schema(
            "adult_census",
            required_columns=["age", "workclass", "education", "income", "sex"],
            optional_columns=["race", "native_country", "occupation", "marital_status",
                              "fnlwgt", "education_num", "capital_gain", "capital_loss",
                              "hours_per_week"],
        )
        _default_validator.register_schema(
            "compas",
            required_columns=["age", "sex", "race", "is_recid"],
            optional_columns=["priors_count", "juv_fel_count", "juv_misd_count",
                              "c_charge_degree", "days_b_screening_arrest"],
        )
        _default_validator.register_schema(
            "german_credit",
            required_columns=["age", "sex", "credit_risk"],
            optional_columns=["credit_amount", "duration", "purpose", "housing",
                              "employment", "savings_account", "checking_account"],
        )
    return _default_validator

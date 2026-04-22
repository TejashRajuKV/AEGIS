"""Input validation utilities for AEGIS."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_dataframe(
    df: Any, name: str = "dataframe", min_rows: int = 1
) -> pd.DataFrame:
    """Validate that input is a non-empty DataFrame."""
    if not isinstance(df, (pd.DataFrame,)):
        raise ValidationError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
    if len(df) < min_rows:
        raise ValidationError(
            f"{name} must have at least {min_rows} rows, got {len(df)}"
        )
    return df


def validate_model_input(
    X: Any, y: Any, name: str = "model_input"
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate model input features and labels."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif not isinstance(X, np.ndarray):
        raise ValidationError(
            f"{name}: X must be numpy array or DataFrame, got {type(X).__name__}"
        )

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.ravel()
    elif not isinstance(y, np.ndarray):
        raise ValidationError(
            f"{name}: y must be numpy array or Series, got {type(y).__name__}"
        )

    if len(X) != len(y):
        raise ValidationError(
            f"{name}: X and y must have same length. X={len(X)}, y={len(y)}"
        )

    return X.astype(np.float64), y


def validate_numeric_range(
    value: float,
    param_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    strict: bool = False,
) -> float:
    """Validate a numeric parameter is within bounds."""
    if min_val is not None:
        if strict and value <= min_val:
            raise ValidationError(
                f"{param_name} must be > {min_val}, got {value}"
            )
        elif not strict and value < min_val:
            raise ValidationError(
                f"{param_name} must be >= {min_val}, got {value}"
            )
    if max_val is not None:
        if strict and value >= max_val:
            raise ValidationError(
                f"{param_name} must be < {max_val}, got {value}"
            )
        elif not strict and value > max_val:
            raise ValidationError(
                f"{param_name} must be <= {max_val}, got {value}"
            )
    return value


def validate_categorical_values(
    values: Sequence, valid_options: List, param_name: str = "parameter"
) -> Sequence:
    """Validate that all values are from a set of valid options."""
    invalid = set(values) - set(valid_options)
    if invalid:
        raise ValidationError(
            f"{param_name}: invalid values {invalid}. "
            f"Valid options: {valid_options}"
        )
    return values


def validate_array_shape(
    arr: np.ndarray,
    expected_ndim: Optional[int] = None,
    expected_shape: Optional[Tuple] = None,
    name: str = "array",
) -> np.ndarray:
    """Validate numpy array dimensions and shape."""
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValidationError(
            f"{name}: expected ndim={expected_ndim}, got {arr.ndim}"
        )
    if expected_shape is not None:
        for i, (exp, got) in enumerate(zip(expected_shape, arr.shape)):
            if exp is not None and exp != got:
                raise ValidationError(
                    f"{name}: expected shape dimension {i}={exp}, got {got}"
                )
    return arr


def validate_probability(value: float, param_name: str = "probability") -> float:
    """Validate value is in [0, 1] range."""
    return validate_numeric_range(
        value, param_name, min_val=0.0, max_val=1.0
    )


def validate_column_exists(
    df: pd.DataFrame,
    columns: List[str],
    context: str = "",
) -> List[str]:
    """Validate that columns exist in a DataFrame."""
    missing = set(columns) - set(df.columns)
    if missing:
        ctx = f" for {context}" if context else ""
        raise ValidationError(
            f"Missing columns{ctx}: {missing}. "
            f"Available: {list(df.columns)}"
        )
    return columns


def validate_not_empty(value: Any, param_name: str = "value") -> Any:
    """Validate value is not None or empty."""
    if value is None:
        raise ValidationError(f"{param_name} must not be None")
    if isinstance(value, (str, list, dict, np.ndarray)) and len(value) == 0:
        raise ValidationError(f"{param_name} must not be empty")
    return value

"""AEGIS Feature Engineering - Create derived features."""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create interaction and polynomial features from datasets."""

    def __init__(self):
        self._fitted = False
        self._interaction_columns: List[str] = []

    def create_interactions(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create pairwise interaction features (multiplication).

        Args:
            df: Input DataFrame.
            columns: Specific columns for interactions. If None, use all numeric.

        Returns:
            DataFrame with additional interaction columns.
        """
        result = df.copy()
        num_cols = columns or list(
            df.select_dtypes(include=[np.number]).columns
        )
        self._interaction_columns = []

        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col_a = num_cols[i]
                col_b = num_cols[j]
                new_col = f"{col_a}_x_{col_b}"
                result[new_col] = result[col_a] * result[col_b]
                self._interaction_columns.append(new_col)

        logger.info(f"Created {len(self._interaction_columns)} interaction features")
        return result

    def bin_continuous(
        self,
        df: pd.DataFrame,
        column: str,
        num_bins: int = 5,
        strategy: str = "quantile",
    ) -> pd.DataFrame:
        """Bin a continuous column into discrete bins.

        Args:
            df: Input DataFrame.
            column: Column name to bin.
            num_bins: Number of bins.
            strategy: 'quantile' or 'uniform'.

        Returns:
            DataFrame with binned column replacing original.
        """
        result = df.copy()
        if strategy == "quantile":
            result[column] = pd.qcut(
                result[column], q=num_bins, labels=False, duplicates="drop"
            )
        else:
            result[column] = pd.cut(
                result[column], bins=num_bins, labels=False
            )
        return result

    def create_polynomial_features(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None, degree: int = 2
    ) -> pd.DataFrame:
        """Create polynomial features (squared terms).

        Args:
            df: Input DataFrame.
            columns: Columns to create polynomials for.
            degree: Maximum polynomial degree.

        Returns:
            DataFrame with polynomial features appended.
        """
        result = df.copy()
        num_cols = columns or list(
            df.select_dtypes(include=[np.number]).columns
        )

        for col in num_cols:
            for d in range(2, degree + 1):
                new_col = f"{col}^pow{d}"
                result[new_col] = result[col] ** d

        logger.info(f"Created polynomial features up to degree {degree}")
        return result

    def create_age_groups(self, df: pd.DataFrame, age_column: str = "age") -> pd.DataFrame:
        """Create age group feature from age column."""
        result = df.copy()
        if age_column in result.columns:
            result["age_group"] = pd.cut(
                result[age_column],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            )
        return result

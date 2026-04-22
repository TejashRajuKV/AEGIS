"""AEGIS Data Splitter - Train/test splitting with stratification."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split datasets into train and test sets with optional stratification."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split features and target into train/test sets.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            stratify: Array to stratify by. If None, uses y.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        strat = y if stratify is None else stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat,
        )
        logger.info(
            f"Data split: train={len(X_train)}, test={len(X_test)} "
            f"(ratio={1 - self.test_size:.1%}/{self.test_size:.1%})"
        )
        return X_train, X_test, y_train, y_test

    def split_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a single DataFrame by target column.

        Args:
            df: Full DataFrame.
            target_column: Name of the target column.

        Returns:
            Tuple of (train_df, test_df).
        """
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[target_column],
        )
        logger.info(
            f"DataFrame split: train={len(train_df)}, test={len(test_df)}"
        )
        return train_df, test_df

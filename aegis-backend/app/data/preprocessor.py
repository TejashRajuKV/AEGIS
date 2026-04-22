"""AEGIS Data Preprocessor - Clean, normalize, and encode data."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Tuple
import logging
import pickle

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocess datasets: handle missing values, encode categoricals, scale numerics."""

    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.numerical_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit the preprocessor on a DataFrame.

        Args:
            df: Raw DataFrame to fit on.

        Returns:
            Self for chaining.
        """
        self.numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(df.select_dtypes(exclude=[np.number]).columns)

        if self.numerical_columns:
            self.numerical_imputer.fit(df[self.numerical_columns])
            self.scaler.fit(df[self.numerical_columns])

        if self.categorical_columns:
            self.categorical_imputer.fit(df[self.categorical_columns])

        self._fitted = True
        logger.info(
            f"Preprocessor fitted: {len(self.numerical_columns)} numerical, "
            f"{len(self.categorical_columns)} categorical columns"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame using fitted preprocessor.

        Args:
            df: Raw DataFrame to transform.

        Returns:
            Preprocessed DataFrame with all numeric columns.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        result = df.copy()

        # Impute and scale numerical columns
        if self.numerical_columns:
            num_data = result[self.numerical_columns].values
            num_data = self.numerical_imputer.transform(num_data)
            num_data = self.scaler.transform(num_data)
            result[self.numerical_columns] = num_data

        # Encode categorical columns using label encoding
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                le = LabelEncoder()
                vals = self.categorical_imputer.transform(
                    result[[col]].values
                ).ravel()
                le.fit(vals)
                self.label_encoders[col] = le
            le = self.label_encoders[col]
            vals = self.categorical_imputer.transform(
                result[[col]].values
            ).ravel()
            result[col] = le.transform(vals)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """Get all feature column names after preprocessing."""
        return self.numerical_columns + self.categorical_columns

    def save(self, path: str) -> None:
        """Save fitted preprocessor to file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        """Load preprocessor from file."""
        with open(path, "rb") as f:
            return pickle.load(f)

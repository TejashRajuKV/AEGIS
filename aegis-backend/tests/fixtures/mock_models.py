"""Test fixtures for mock models."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pytest


class MockClassifier(BaseEstimator, ClassifierMixin):
    """Simple mock classifier for testing."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return (np.random.rand(len(X)) > self.threshold).astype(int)

    def predict_proba(self, X):
        probs = np.random.rand(len(X), 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


@pytest.fixture
def mock_model():
    """Provide a mock classifier."""
    return MockClassifier()

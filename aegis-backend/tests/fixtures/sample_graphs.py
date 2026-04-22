"""Test fixtures for sample graphs."""

import numpy as np
import pytest


@pytest.fixture
def sample_adjacency():
    """Generate a sample DAG adjacency matrix."""
    adj = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ], dtype=float)
    return adj


@pytest.fixture
def sample_feature_names():
    """Sample feature names."""
    return ["age", "income", "education", "employment", "credit_score"]

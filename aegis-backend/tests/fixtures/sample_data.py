"""Test fixtures for AEGIS tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_data():
    """Generate sample classification data."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def sample_protected_attr():
    """Generate sample protected attribute."""
    np.random.seed(42)
    return (np.random.rand(200) > 0.5).astype(int)


@pytest.fixture
def sample_reference_data():
    """Generate reference distribution for drift detection."""
    np.random.seed(42)
    return np.random.randn(500)


@pytest.fixture
def sample_test_data():
    """Generate test distribution with drift."""
    np.random.seed(43)
    data = np.random.randn(200)
    data[:100] += 0.5  # Add mean shift
    return data

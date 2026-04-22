"""Math utility functions for AEGIS ML modules.

Pure numpy implementations of mathematical operations used across
causal discovery, drift detection, and neural network modules.
"""

from typing import Optional

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max, axis=axis) + np.log(
        np.sum(np.exp(x - x_max), axis=axis)
    )


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 20))))


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute KL(p || q) for discrete distributions."""
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return float(np.sum(p * np.log(p / q)))


def entropy(p: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute Shannon entropy of discrete distribution."""
    p = np.clip(p, epsilon, 1.0)
    return float(-np.sum(p * np.log(p)))


def wasserstein_1d(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Compute Wasserstein-1 (Earth Mover's Distance) for 1D distributions.

    Uses the closed-form solution for 1D: difference of sorted CDFs.
    """
    a_sorted = np.sort(samples_a.ravel())
    b_sorted = np.sort(samples_b.ravel())
    return float(np.mean(np.abs(a_sorted - b_sorted)))


def trace_expm(
    matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-7
) -> float:
    """Compute trace of matrix exponential via scaled Taylor expansion.

    Used for the DAG constraint: h(W) = tr(e^{W o W}) - d = 0
    where o is the Hadamard product.

    Args:
        matrix: Square numpy array.
        max_iter: Maximum Taylor series iterations.
        tol: Convergence tolerance.

    Returns:
        Trace of matrix exponential.
    """
    n = matrix.shape[0]
    # Scale to prevent overflow
    norm = np.linalg.norm(matrix, ord=1)
    if norm == 0:
        return float(n)

    scale = max(int(np.ceil(norm)), 1)
    scaled = matrix / scale

    result = np.eye(n)
    term = np.eye(n)
    for i in range(1, max_iter + 1):
        term = term @ scaled / i
        result += term
        if np.linalg.norm(term) < tol:
            break

    return float(np.trace(result))


def hutchinson_trace(
    matrix: np.ndarray, num_samples: int = 50, seed: Optional[int] = None
) -> float:
    """Stochastic trace estimator using Hutchinson's estimator.

    Uses random +/-1 vectors to estimate tr(matrix) without
    computing the full matrix exponential diagonal.

    Args:
        matrix: Square numpy array.
        num_samples: Number of random probe vectors.
        seed: Random seed for reproducibility.

    Returns:
        Estimated trace value.
    """
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    traces = []

    for _ in range(num_samples):
        # Random +/-1 vector
        v = rng.choice([-1.0, 1.0], size=n)
        result = matrix @ v
        traces.append(float(np.dot(v, result)))

    return float(np.mean(traces))


def matrix_hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise (Hadamard) product of two matrices."""
    return a * b


def is_dag(adj_matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if adjacency matrix represents a valid DAG.

    Uses the topological sorting approach: a graph is a DAG iff
    it has no cycles.
    """
    n = adj_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    rec_stack = np.zeros(n, dtype=bool)

    def _dfs(node: int) -> bool:
        visited[node] = True
        rec_stack[node] = True
        for child in range(n):
            if adj_matrix[node, child] > tol:
                if not visited[child]:
                    if _dfs(child):
                        return True
                elif rec_stack[child]:
                    return True
        rec_stack[node] = False
        return False

    for i in range(n):
        if not visited[i]:
            if _dfs(i):
                return False
    return True


def compute_dag_constraint(adj_matrix: np.ndarray) -> float:
    """Compute DAG constraint: h(W) = tr(e^{W o W}) - d.

    Value <= 0 means the matrix is a valid DAG.
    """
    n = adj_matrix.shape[0]
    w_pow = matrix_hadamard(adj_matrix, adj_matrix)
    trace = trace_expm(w_pow)
    return trace - n


def threshold_adjacency(
    adj_matrix: np.ndarray, threshold: float = 0.3
) -> np.ndarray:
    """Threshold adjacency matrix to produce a sparse DAG."""
    result = np.zeros_like(adj_matrix)
    result[adj_matrix > threshold] = 1.0
    return result

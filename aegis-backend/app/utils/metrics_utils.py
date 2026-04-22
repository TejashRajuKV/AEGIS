"""Metric utility functions for fairness evaluation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None
) -> Dict[str, Any]:
    """Compute confusion matrix and derived metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Compute ROC AUC and PR AUC."""
    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision) if len(recall) > 1 else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score."""
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """Compute precision and recall."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    return float(p), float(r)


def compute_demographic_parity_difference(
    y_pred: np.ndarray, sensitive_attr: np.ndarray
) -> float:
    """Compute demographic parity difference.

    DP = |P(Y=1|A=0) - P(Y=1|A=1)|
    Lower is more fair.
    """
    groups = np.unique(sensitive_attr)
    if len(groups) < 2:
        return 0.0
    group_probs = {}
    for g in groups:
        mask = sensitive_attr == g
        if mask.sum() == 0:
            group_probs[g] = 0.0
        else:
            group_probs[g] = float(np.mean(y_pred[mask]))
    probs = list(group_probs.values())
    return float(np.max(probs) - np.min(probs))


def compute_equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Compute equalized odds difference.

    EO = max(|FPR_a - FPR_b|, |FNR_a - FNR_b|)
    """
    groups = np.unique(sensitive_attr)
    if len(groups) < 2:
        return 0.0

    fprs = []
    fnrs = []
    for g in groups:
        mask = sensitive_attr == g
        if mask.sum() == 0:
            continue
        y_t = y_true[mask]
        y_p = y_pred[mask]
        fp = np.sum((y_p == 1) & (y_t == 0))
        tn = np.sum((y_p == 0) & (y_t == 0))
        fn = np.sum((y_p == 0) & (y_t == 1))
        tp = np.sum((y_p == 1) & (y_t == 1))
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        fprs.append(fpr)
        fnrs.append(fnr)

    if len(fprs) < 2:
        return 0.0
    return float(max(max(fprs) - min(fprs), max(fnrs) - min(fnrs)))


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = np.mean(y_prob[mask])
        avg_accuracy = np.mean(y_true[mask])
        ece += (count / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def format_metric_value(value: float, as_percentage: bool = False) -> str:
    """Format metric value for display."""
    if as_percentage:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"

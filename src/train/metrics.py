"""
Evaluation metrics for seizure forecasting.
"""

from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix as sklearn_confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True binary labels [N]
        y_prob: Predicted probabilities [N]
        threshold: Classification threshold
        
    Returns:
        Dict of metric names to values
    """
    # Ensure 1D and correct types
    y_true = np.asarray(y_true).ravel().astype(int)  # Labels must be integers
    y_prob = np.asarray(y_prob).ravel().astype(float)  # Probabilities must be float
    
    # Check for valid data
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "ppv": 0.0,
            "npv": 0.0,
            "f1": 0.0,
        }
    
    # Binary predictions
    y_pred = (y_prob >= threshold).astype(int)
    
    # ROC-AUC
    auroc = roc_auc_score(y_true, y_prob)
    
    # Precision-Recall AUC
    auprc = average_precision_score(y_true, y_prob)
    
    # Confusion matrix elements
    tn, fp, fn, tp = sklearn_confusion_matrix(y_true, y_pred).ravel()
    
    # Derived metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / max(tp + fn, 1)  # Recall
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)  # Precision
    npv = tn / max(tn + fn, 1)
    
    f1 = 2 * ppv * sensitivity / max(ppv + sensitivity, 1e-10)
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels [N]
        y_prob: Predicted probabilities [N]
        threshold: Classification threshold
        
    Returns:
        Confusion matrix [[TN, FP], [FN, TP]]
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return sklearn_confusion_matrix(y_true, y_pred)


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "youden",
) -> float:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        criterion: 'youden' (maximize J = sensitivity + specificity - 1)
                   or 'f1' (maximize F1 score)
        
    Returns:
        Optimal threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if criterion == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
    
    elif criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])  # Last element is 0
        return thresholds[best_idx]
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def compute_metrics_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
) -> Dict[float, Dict[str, float]]:
    """
    Compute metrics at multiple thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dict mapping threshold to metrics dict
    """
    results = {}
    for thresh in thresholds:
        results[thresh] = compute_metrics(y_true, y_prob, threshold=thresh)
    return results


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Tuple of (bin_centers, true_fractions, bin_counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    true_fractions = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            true_fractions[i] = y_true[mask].mean()
            bin_counts[i] = mask.sum()
    
    return bin_centers, true_fractions, bin_counts

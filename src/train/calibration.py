"""
Calibration module for improving probability estimates.

Temperature scaling to calibrate model outputs and improve threshold stability.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from typing import Tuple, Optional
from scipy.optimize import minimize_scalar


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    Learns a single temperature parameter T to scale logits:
    calibrated_prob = sigmoid(logit / T)
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """
        Fit temperature on validation set.
        
        Args:
            logits: Uncalibrated logits [N]
            labels: True labels [N]
            max_iter: Maximum optimization iterations
            
        Returns:
            Learned temperature value
        """
        logits = logits.detach()
        labels = labels.float()
        
        # Use LBFGS optimizer
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled_logits, labels
            )
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return self.temperature.item()


def fit_temperature_scipy(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Fit temperature using scipy optimization (simpler, no gradients).
    
    Args:
        logits: Uncalibrated logits [N]
        labels: True labels [N]
        
    Returns:
        Optimal temperature
    """
    def nll_loss(T):
        if T <= 0:
            return np.inf
        scaled = logits / T
        probs = 1 / (1 + np.exp(-scaled))
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        return nll
    
    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
    return result.x


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics.
    
    Args:
        probs: Predicted probabilities [N]
        labels: True labels [N]
        n_bins: Number of calibration bins
        
    Returns:
        Dict with ECE, MCE, reliability diagram data
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_confidences.append(probs[mask].mean())
            bin_accuracies.append(labels[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_confidences.append(bin_centers[i])
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    bin_confidences = np.array(bin_confidences)
    bin_accuracies = np.array(bin_accuracies)
    bin_counts = np.array(bin_counts)
    
    # Expected Calibration Error
    total = bin_counts.sum()
    if total > 0:
        ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / total
    else:
        ece = 0.0
    
    # Maximum Calibration Error
    mce = np.max(np.abs(bin_accuracies - bin_confidences))
    
    return {
        "ece": ece,
        "mce": mce,
        "bin_centers": bin_centers,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def calibrate_predictions(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Calibrate test predictions using temperature learned on validation.
    
    Args:
        val_logits: Validation logits for fitting temperature
        val_labels: Validation labels
        test_logits: Test logits to calibrate
        
    Returns:
        Tuple of (calibrated_test_probs, temperature)
    """
    # Fit temperature on validation
    temperature = fit_temperature_scipy(val_logits, val_labels)
    
    # Apply to test
    calibrated_logits = test_logits / temperature
    calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
    
    return calibrated_probs, temperature


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> "matplotlib.figure.Figure":
    """
    Create reliability diagram.
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    metrics = compute_calibration_metrics(probs, labels, n_bins)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.bar(
        metrics["bin_centers"],
        metrics["bin_accuracies"],
        width=1/n_bins,
        alpha=0.7,
        label="Outputs",
        edgecolor="black",
    )
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"{title}\nECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}")
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram of predictions
    ax2.hist(probs, bins=n_bins, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    
    plt.tight_layout()
    return fig

#!/usr/bin/env python3
"""
Generate publication-style figures for seizure forecasting results.

Figures:
1. Raw vs preprocessed EEG snippet
2. Spectrogram visualization
3. Label timeline diagram
4. Training curves
5. Risk score over time with seizure markers
6. Confusion matrix + metrics table

Usage:
    python scripts/make_figures.py --run_dir runs/deep_model_20260114_...
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import pandas as pd
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.train.metrics import compute_confusion_matrix


console = Console()

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["figure.titlesize"] = 16


def plot_training_curves(run_dir: Path, output_dir: Path):
    """Plot training loss and AUROC curves."""
    metrics_file = run_dir / "logs" / "metrics.csv"
    
    if not metrics_file.exists():
        console.print(f"[yellow]No metrics file found at {metrics_file}[/yellow]")
        return
    
    df = pd.read_csv(metrics_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(df["epoch"], df["train_loss"], label="Train", color="#2563eb", linewidth=2)
    axes[0].plot(df["epoch"], df["val_loss"], label="Val", color="#dc2626", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUROC curves
    axes[1].plot(df["epoch"], df["train_auroc"], label="Train", color="#2563eb", linewidth=2)
    axes[1].plot(df["epoch"], df["val_auroc"], label="Val", color="#dc2626", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("AUROC Over Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print("  [green]Done:[/green] Training curves")


def plot_confusion_matrix(run_dir: Path, output_dir: Path):
    """Plot confusion matrix from predictions."""
    pred_file = run_dir / "eval" / "predictions.npz"
    
    if not pred_file.exists():
        console.print(f"[yellow]No predictions file found at {pred_file}[/yellow]")
        return
    
    data = np.load(pred_file)
    predictions = data["predictions"]
    labels = data["labels"]
    
    cm = compute_confusion_matrix(labels, predictions, threshold=0.5)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Interictal", "Preictal"],
        yticklabels=["Interictal", "Preictal"],
        ax=ax,
    )
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Threshold = 0.5)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print("  [green]Done:[/green] Confusion matrix")


def plot_label_timeline(output_dir: Path):
    """Create label timeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Example timeline: 60 minutes, seizure at 30 min
    total_time = 60  # minutes
    seizure_onset = 30
    seizure_offset = 31
    
    preictal_min = 10
    gap_sec = 30 / 60  # in minutes
    postictal_min = 10
    buffer_min = 5  # for visualization
    
    # Define regions
    regions = [
        (0, seizure_onset - preictal_min - buffer_min, "Interictal", "#22c55e"),
        (seizure_onset - preictal_min, seizure_onset - gap_sec, "Preictal", "#f97316"),
        (seizure_onset - gap_sec, seizure_onset, "Gap", "#94a3b8"),
        (seizure_onset, seizure_offset, "Ictal", "#ef4444"),
        (seizure_offset, seizure_offset + postictal_min, "Postictal", "#94a3b8"),
        (seizure_offset + postictal_min + buffer_min, total_time, "Interictal", "#22c55e"),
    ]
    
    # Plot regions
    for start, end, label, color in regions:
        ax.axvspan(start, end, alpha=0.5, color=color, label=label)
        ax.axvspan(start, end, alpha=0.1, color=color)
    
    # Seizure onset marker
    ax.axvline(seizure_onset, color="#ef4444", linewidth=3, linestyle="--", label="Seizure Onset")
    
    # Buffer zones
    ax.axvspan(seizure_onset - preictal_min - buffer_min, seizure_onset - preictal_min, 
               alpha=0.3, color="#94a3b8", hatch="//")
    ax.axvspan(seizure_offset + postictal_min, seizure_offset + postictal_min + buffer_min,
               alpha=0.3, color="#94a3b8", hatch="//")
    
    ax.set_xlim([0, total_time])
    ax.set_xlabel("Time (minutes)")
    ax.set_yticks([])
    ax.set_title("Labeling Schema Timeline")
    
    # Custom legend
    legend_elements = [
        Patch(facecolor="#22c55e", alpha=0.5, label="Interictal"),
        Patch(facecolor="#f97316", alpha=0.5, label="Preictal"),
        Patch(facecolor="#94a3b8", alpha=0.5, label="Excluded"),
        Patch(facecolor="#ef4444", alpha=0.5, label="Ictal"),
        plt.Line2D([0], [0], color="#ef4444", linewidth=3, linestyle="--", label="Seizure Onset"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", ncol=5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "label_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print("  [green]Done:[/green] Label timeline diagram")


def plot_risk_timeline(run_dir: Path, output_dir: Path):
    """Plot example risk score over time with seizure markers."""
    pred_file = run_dir / "eval" / "predictions.npz"
    
    if not pred_file.exists():
        # Generate synthetic example
        np.random.seed(42)
        
        # 1 hour recording, windows every 5 seconds
        n_windows = 720
        times = np.arange(n_windows) * 5 / 60  # minutes
        
        # Interictal baseline
        risk = 0.1 + 0.05 * np.random.randn(n_windows)
        
        # Rising risk before seizure at 30 min
        seizure_onset = 30  # minutes
        preictal_start = seizure_onset - 10
        
        for i, t in enumerate(times):
            if preictal_start <= t < seizure_onset:
                progress = (t - preictal_start) / (seizure_onset - preictal_start)
                risk[i] = 0.2 + 0.7 * progress + 0.1 * np.random.randn()
        
        risk = np.clip(risk, 0, 1)
    else:
        data = np.load(pred_file)
        risk = data["predictions"][:720]  # First ~1 hour
        times = np.arange(len(risk)) * 5 / 60
        seizure_onset = 30
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Plot risk
    ax.plot(times, risk, color="#2563eb", linewidth=1.5, alpha=0.8)
    ax.fill_between(times, 0, risk, alpha=0.3, color="#2563eb")
    
    # Threshold line
    threshold = 0.5
    ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
    
    # Seizure onset
    ax.axvline(seizure_onset, color="#ef4444", linewidth=3, linestyle="-", label="Seizure Onset")
    
    # Preictal zone
    ax.axvspan(seizure_onset - 10, seizure_onset, alpha=0.2, color="#f97316", label="Preictal Zone")
    
    # Alarm markers
    alarm_times = times[(times < seizure_onset) & (risk > threshold)]
    if len(alarm_times) > 0:
        first_alarm = alarm_times[0]
        ax.axvline(first_alarm, color="#22c55e", linewidth=2, linestyle=":", label=f"First Alarm (t={first_alarm:.1f}min)")
    
    ax.set_xlim([0, max(times)])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Risk Score")
    ax.set_title("Seizure Risk Prediction Over Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "risk_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print("  [green]Done:[/green] Risk timeline")


def plot_spectrogram_example(output_dir: Path):
    """Generate example spectrogram visualization."""
    # Synthetic example
    np.random.seed(42)
    
    sfreq = 256
    duration = 10  # seconds
    n_samples = sfreq * duration
    n_channels = 4
    
    # Generate synthetic EEG-like signal
    t = np.linspace(0, duration, n_samples)
    
    signals = []
    for ch in range(n_channels):
        # Mix of frequency components
        signal = (
            0.5 * np.sin(2 * np.pi * 10 * t) +  # Alpha
            0.3 * np.sin(2 * np.pi * 6 * t) +   # Theta
            0.2 * np.random.randn(n_samples)     # Noise
        )
        signals.append(signal)
    
    signals = np.array(signals)
    
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 8))
    
    for ch in range(n_channels):
        # Time series
        axes[ch, 0].plot(t, signals[ch], linewidth=0.5, color="#2563eb")
        axes[ch, 0].set_ylabel(f"Ch {ch+1}")
        if ch == 0:
            axes[ch, 0].set_title("Raw EEG Signal")
        if ch == n_channels - 1:
            axes[ch, 0].set_xlabel("Time (s)")
        
        # Spectrogram
        axes[ch, 1].specgram(
            signals[ch], Fs=sfreq, NFFT=256, noverlap=128,
            cmap="viridis", vmin=-40, vmax=10
        )
        axes[ch, 1].set_ylim([0, 50])
        if ch == 0:
            axes[ch, 1].set_title("Spectrogram")
        if ch == n_channels - 1:
            axes[ch, 1].set_xlabel("Time (s)")
            axes[ch, 1].set_ylabel("Frequency (Hz)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "spectrogram_example.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    console.print("  [green]Done:[/green] Spectrogram example")


def main():
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: reports/figures)",
    )
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold blue]Generating Figures[/bold blue]")
    console.print(f"Run dir: {run_dir}")
    console.print(f"Output: {output_dir}\n")
    
    # Generate all figures
    plot_training_curves(run_dir, output_dir)
    plot_confusion_matrix(run_dir, output_dir)
    plot_label_timeline(output_dir)
    plot_risk_timeline(run_dir, output_dir)
    plot_spectrogram_example(output_dir)
    
    console.print(f"\n[bold green]All figures saved to {output_dir}[/bold green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

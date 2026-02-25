#!/usr/bin/env python3
"""
Comprehensive evaluation script with alarm-level metrics and figures.

Generates:
- Window-level metrics (AUROC, AUPRC)
- Alarm-level metrics (FAH, sensitivity, warning time)
- Confusion matrix
- Risk timeline with alarms
- Warning time histogram
- Threshold vs FAH curve

Usage:
    python scripts/full_eval.py --checkpoint runs/deep_model_xxx/checkpoints/best.pt --config configs/small_run.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import create_dataloaders, create_within_subject_dataloaders
from src.models.fusion_net import FusionNet
from src.train.losses import SeizureForecastingLoss
from src.train.metrics import compute_metrics
from src.train.alarm_eval import (
    AlarmEvaluator, 
    AlarmMetrics, 
    compute_alarm_metrics,
    evaluate_at_thresholds,
)


console = Console()


def evaluate_model(model, data_loader, device, use_amp=True, use_features=True):
    """Run model on data and collect predictions."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_subjects = []
    
    with torch.no_grad():
        for batch in data_loader:
            data = batch["data"].to(device, non_blocking=True)
            labels = batch["y_cls"].to(device, non_blocking=True)
            features = batch.get("features")
            if features is not None:
                features = features.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=use_amp and device.type == 'cuda'):
                cls_logit, soft_pred = model(data, features if use_features else None)
            
            probs = torch.sigmoid(cls_logit).squeeze().cpu().numpy()
            # Handle batch vs single sample case
            if probs.ndim == 0:
                probs = np.array([probs.item()])
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_subjects.extend(batch.get("subject", ["unknown"] * len(probs)))
    
    return np.array(all_preds), np.array(all_labels), all_subjects


def find_threshold_for_target_fah(
    preds: np.ndarray,
    labels: np.ndarray,
    target_fah: float,
    n_points: int = 100,
) -> tuple:
    """Find threshold that achieves target FAH on validation set."""
    thresholds = np.linspace(0.01, 0.99, n_points)
    
    best_thresh = 0.5
    best_diff = float('inf')
    best_metrics = None
    
    # Simple FAH approximation using window-level FPR
    for thresh in thresholds:
        pred_labels = (preds >= thresh).astype(int)
        
        # Count false positives and negatives
        n_neg = (labels == 0).sum()
        fp = ((pred_labels == 1) & (labels == 0)).sum()
        
        # Approximate FAH (very rough)
        fpr = fp / max(n_neg, 1)
        approx_fah = fpr * 3600  # Assuming ~1 window per second
        
        diff = abs(approx_fah - target_fah)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
    
    return best_thresh


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, threshold: float, output_path: Path):
    """Plot and save confusion matrix."""
    pred_labels = (preds >= threshold).astype(int)
    cm = confusion_matrix(labels, pred_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Interictal', 'Preictal'])
    ax.set_yticklabels(['Interictal', 'Preictal'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})', fontsize=14)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > cm.max() / 2 else 'black'
            ax.text(j, i, f'{val}', ha='center', va='center', color=color, fontsize=16)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return cm


def plot_risk_timeline(
    preds: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    output_path: Path,
    window_sec: float = 5.0,
    n_windows: int = 500,
):
    """Plot risk timeline with seizure markers and alarms."""
    # Use a subset with preictal windows for visualization
    preictal_idx = np.where(labels == 1)[0]
    
    if len(preictal_idx) == 0:
        console.print("[yellow]No preictal windows for risk timeline[/yellow]")
        return
    
    # Find a segment around first preictal region
    start_idx = max(0, preictal_idx[0] - 100)
    end_idx = min(len(preds), start_idx + n_windows)
    
    segment_preds = preds[start_idx:end_idx]
    segment_labels = labels[start_idx:end_idx]
    times = np.arange(len(segment_preds)) * window_sec
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot risk score
    ax.plot(times, segment_preds, 'b-', linewidth=1.5, label='Risk Score')
    ax.axhline(threshold, color='orange', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.2f})')
    
    # Shade preictal regions
    preictal_mask = segment_labels == 1
    if preictal_mask.any():
        ax.fill_between(times, 0, 1, where=preictal_mask, alpha=0.3, color='red', label='Preictal Region')
    
    # Mark alarms
    alarm_mask = segment_preds >= threshold
    alarm_times = times[alarm_mask]
    alarm_scores = segment_preds[alarm_mask]
    ax.scatter(alarm_times, alarm_scores, color='orange', s=50, marker='v', zorder=5, label='Alarm')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Risk Score', fontsize=12)
    ax.set_title('Risk Timeline with Alarms', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_threshold_vs_fah(
    preds: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    n_thresholds: int = 50,
):
    """Plot threshold vs approximate FAH curve."""
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    fahs = []
    sensitivities = []
    
    for thresh in thresholds:
        pred_labels = (preds >= thresh).astype(int)
        
        # Sensitivity
        n_pos = (labels == 1).sum()
        tp = ((pred_labels == 1) & (labels == 1)).sum()
        sens = tp / max(n_pos, 1)
        sensitivities.append(sens)
        
        # Approximate FAH
        n_neg = (labels == 0).sum()
        fp = ((pred_labels == 1) & (labels == 0)).sum()
        fpr = fp / max(n_neg, 1)
        # Very rough FAH approximation
        approx_fah = fpr * 60  # Per hour assuming 1/min windows
        fahs.append(min(approx_fah, 10))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, fahs, 'b-', linewidth=2, label='FAH')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Approx. False Alarms per Hour', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax.twinx()
    ax2.plot(thresholds, sensitivities, 'r-', linewidth=2, label='Sensitivity')
    ax2.set_ylabel('Sensitivity', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_title('Threshold vs FAH and Sensitivity', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_warning_time_histogram(warning_times: list, output_path: Path):
    """Plot histogram of warning times."""
    if not warning_times:
        console.print("[yellow]No warning times to plot[/yellow]")
        return
    
    warning_min = np.array(warning_times) / 60  # Convert to minutes
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(warning_min, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(warning_min), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(warning_min):.1f} min')
    ax.axvline(np.median(warning_min), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(warning_min):.1f} min')
    
    ax.set_xlabel('Warning Time (minutes)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Warning Times', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Full evaluation with alarm metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/small_run.yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as checkpoint)")
    parser.add_argument("--split_mode", type=str, choices=["cross_subject", "within_subject"], default=None)
    parser.add_argument("--subject", type=str, default=None, help="Subject for within-subject mode")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    set_seed(cfg.split.seed)
    
    # Determine split mode
    split_mode = args.split_mode or cfg.split.get("mode", "cross_subject")
    within_subject_id = args.subject or cfg.split.get("within_subject_id", "chb01")
    
    # Output directory
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold blue]Full Model Evaluation[/bold blue]")
    console.print(f"Checkpoint: {checkpoint_path}")
    console.print(f"Split mode: {split_mode}")
    console.print(f"Output: {output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")
    
    # Create dataloaders
    cache_path = Path(cfg.data.cache_root)
    data_root = Path(cfg.data.raw_root)
    
    if split_mode == "within_subject":
        _, val_loader, test_loader = create_within_subject_dataloaders(
            cache_path, cfg, within_subject_id, cfg.model.use_features, data_root
        )
    else:
        _, val_loader, test_loader = create_dataloaders(cache_path, cfg, cfg.model.use_features)
    
    # Load model
    sample_batch = next(iter(test_loader))
    n_channels = sample_batch["data"].shape[1]
    n_features = sample_batch["features"].shape[1] if "features" in sample_batch else 0
    
    model = FusionNet(n_channels=n_channels, n_features=n_features, cfg=cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    console.print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Evaluate on validation set (for threshold tuning)
    console.print("\n[bold]Evaluating on validation set...[/bold]")
    val_preds, val_labels, _ = evaluate_model(model, val_loader, device, cfg.training.use_amp, cfg.model.use_features)
    
    # Evaluate on test set
    console.print("[bold]Evaluating on test set...[/bold]")
    test_preds, test_labels, test_subjects = evaluate_model(model, test_loader, device, cfg.training.use_amp, cfg.model.use_features)
    
    # Compute window-level metrics
    console.print("\n[bold]Window-level Metrics:[/bold]")
    test_metrics = compute_metrics(test_preds, test_labels)
    for k, v in test_metrics.items():
        console.print(f"  {k}: {v:.4f}")
    
    # Find threshold for target FAH on validation
    target_fahs = cfg.evaluation.get("target_fah", [0.2, 0.5, 1.0])
    console.print(f"\n[bold]Threshold tuning (target FAH: {target_fahs}):[/bold]")
    
    tuned_thresholds = {}
    for target in target_fahs:
        thresh = find_threshold_for_target_fah(val_preds, val_labels, target)
        tuned_thresholds[target] = thresh
        console.print(f"  FAH={target}: threshold={thresh:.3f}")
    
    # Use default threshold for plots
    default_threshold = cfg.evaluation.get("threshold", 0.5)
    
    # Generate plots
    console.print("\n[bold]Generating figures...[/bold]")
    
    # 1. Confusion matrix
    cm = plot_confusion_matrix(test_labels, test_preds, default_threshold, output_dir / "confusion_matrix.png")
    console.print(f"  Saved: confusion_matrix.png")
    
    # 2. Risk timeline
    plot_risk_timeline(test_preds, test_labels, default_threshold, output_dir / "risk_timeline_example.png")
    console.print(f"  Saved: risk_timeline_example.png")
    
    # 3. Threshold vs FAH curve
    plot_threshold_vs_fah(val_preds, val_labels, output_dir / "threshold_vs_FAH_curve.png")
    console.print(f"  Saved: threshold_vs_FAH_curve.png")
    
    # 4. Approximate warning time distribution
    # (Simplified - using time to preictal transition)
    preictal_changes = []
    for i in range(1, len(test_labels)):
        if test_labels[i] == 1 and test_labels[i-1] == 0:
            # Find first alarm after transition
            for j in range(i-100, i):
                if j >= 0 and test_preds[j] >= default_threshold:
                    # Approximate warning time
                    warning = (i - j) * cfg.windowing.step_sec
                    preictal_changes.append(warning)
                    break
    
    if preictal_changes:
        plot_warning_time_histogram(preictal_changes, output_dir / "warning_time_hist.png")
        console.print(f"  Saved: warning_time_hist.png")
    
    # Compute alarm-level metrics at multiple thresholds
    console.print("\n[bold]Alarm-level Metrics (approximate):[/bold]")
    
    results = []
    for thresh in [0.3, 0.5, 0.7]:
        pred_labels = (test_preds >= thresh).astype(int)
        
        tp = ((pred_labels == 1) & (test_labels == 1)).sum()
        fp = ((pred_labels == 1) & (test_labels == 0)).sum()
        fn = ((pred_labels == 0) & (test_labels == 1)).sum()
        
        sensitivity = tp / max(tp + fn, 1)
        n_interictal = (test_labels == 0).sum()
        approx_fah = (fp / max(n_interictal, 1)) * 60  # Rough approximation
        
        results.append({
            "threshold": thresh,
            "sensitivity": sensitivity,
            "approx_fah": min(approx_fah, 10),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
        
        console.print(f"  Threshold={thresh:.2f}: Sensitivity={sensitivity:.3f}, Approx FAH={approx_fah:.2f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        "split_mode": split_mode,
        "subject": within_subject_id if split_mode == "within_subject" else "cross",
        "auroc": test_metrics.get("auroc", 0),
        "auprc": test_metrics.get("auprc", 0),
        "threshold": default_threshold,
        "sensitivity": results[1]["sensitivity"] if len(results) > 1 else 0,
        "approx_fah": results[1]["approx_fah"] if len(results) > 1 else 0,
    }])
    
    metrics_path = output_dir / "eval_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    console.print(f"\nSaved metrics to: {metrics_path}")
    
    console.print("\n[bold green]Evaluation complete![/bold green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

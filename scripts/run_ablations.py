#!/usr/bin/env python3
"""
Run ablation studies for alarm post-processing and loss functions.

Generates:
- reports/tables/alarm_ablation.csv
- reports/tables/loss_comparison.csv
- reports/figures/threshold_vs_FAH_curve.png
- reports/figures/FAH_sensitivity_tradeoff.png
"""

import argparse
import sys
from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import create_dataloaders
from src.models.fusion_net import FusionNet
from src.train.losses import create_loss_fn, compute_pos_weight
from src.train.loops import Trainer
from src.train.metrics import compute_metrics
from src.train.threshold_tuning import (
    tune_thresholds_for_multiple_targets,
    compute_threshold_curve,
    compute_alarm_metrics_at_threshold,
    apply_risk_smoothing,
    AlarmProcessor,
)
from src.train.calibration import calibrate_predictions, compute_calibration_metrics

console = Console()


def load_model_and_predict(checkpoint_path, cfg, data_loader, device):
    """Load model and get predictions."""
    sample_batch = next(iter(data_loader))
    n_channels = sample_batch["data"].shape[1]
    n_features = sample_batch["features"].shape[1] if "features" in sample_batch else 0
    
    model = FusionNet(n_channels=n_channels, n_features=n_features, cfg=cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    current_time = 0.0
    times = []
    
    with torch.no_grad():
        for batch in data_loader:
            data = batch["data"].to(device, non_blocking=True)
            features = batch.get("features")
            if features is not None:
                features = features.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=cfg.training.use_amp and device.type == 'cuda'):
                cls_logit, _ = model(data, features if cfg.model.use_features else None)
            
            logits = cls_logit.squeeze().cpu().numpy()
            probs = torch.sigmoid(cls_logit).squeeze().cpu().numpy()
            labels = batch["y_cls"].numpy()
            
            if probs.ndim == 0:
                probs = np.array([probs.item()])
                logits = np.array([logits.item()])
            
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.flatten())
            all_logits.extend(logits.flatten())
            
            for _ in range(len(probs.flatten())):
                times.append(current_time)
                current_time += cfg.windowing.step_sec
    
    return (np.array(all_preds), np.array(all_labels).astype(int), 
            np.array(times), np.array(all_logits))


def extract_seizure_intervals(labels, times, cfg):
    """Extract seizure intervals from preictal labels."""
    intervals = []
    preictal_min = cfg.windowing.preictal_min
    
    in_preictal = False
    preictal_start = None
    
    for label, t in zip(labels, times):
        if label == 1 and not in_preictal:
            in_preictal = True
            preictal_start = t
        elif label == 0 and in_preictal:
            in_preictal = False
            onset = preictal_start + preictal_min * 60
            offset = onset + 60
            intervals.append((onset, offset))
    
    if in_preictal:
        onset = preictal_start + preictal_min * 60
        offset = onset + 60
        intervals.append((onset, offset))
    
    return intervals


def run_alarm_ablation(val_preds, val_labels, val_times, val_intervals,
                       test_preds, test_labels, test_times, test_intervals,
                       output_dir):
    """Run ablation study on alarm post-processing methods."""
    console.print("\n[bold]Running Alarm Post-Processing Ablation[/bold]")
    
    fah_targets = [0.1, 0.2, 0.5, 1.0]
    results = []
    
    # Define configurations to test
    configs = [
        {"name": "baseline", "smoothing": None, "persistence": 1, "hysteresis": False},
        {"name": "ema_0.2", "smoothing": ("ema", 0.2), "persistence": 1, "hysteresis": False},
        {"name": "ema_0.3", "smoothing": ("ema", 0.3), "persistence": 1, "hysteresis": False},
        {"name": "ma_6", "smoothing": ("moving_avg", 6), "persistence": 1, "hysteresis": False},
        {"name": "persist_3", "smoothing": None, "persistence": 3, "hysteresis": False},
        {"name": "persist_5", "smoothing": None, "persistence": 5, "hysteresis": False},
        {"name": "hysteresis", "smoothing": None, "persistence": 1, "hysteresis": True},
        {"name": "ema+persist", "smoothing": ("ema", 0.2), "persistence": 3, "hysteresis": False},
        {"name": "full_combo", "smoothing": ("ema", 0.2), "persistence": 3, "hysteresis": True},
    ]
    
    for config in configs:
        console.print(f"  Testing: {config['name']}")
        
        # Apply smoothing to predictions
        if config["smoothing"]:
            method, param = config["smoothing"]
            if method == "ema":
                processed_val = apply_risk_smoothing(val_preds, "ema", alpha=param)
                processed_test = apply_risk_smoothing(test_preds, "ema", alpha=param)
            else:
                processed_val = apply_risk_smoothing(val_preds, "moving_avg", window_size=param)
                processed_test = apply_risk_smoothing(test_preds, "moving_avg", window_size=param)
        else:
            processed_val = val_preds
            processed_test = test_preds
        
        # Tune thresholds on validation
        for fah_target in fah_targets:
            # Find threshold for target FAH on val
            best_thresh = 0.5
            best_sens = 0.0
            
            for thresh in np.linspace(0.01, 0.99, 100):
                metrics = compute_alarm_metrics_at_threshold(
                    val_times, processed_val, val_intervals, thresh,
                    refractory_sec=1200
                )
                
                if metrics["fah"] <= fah_target and metrics["sensitivity"] > best_sens:
                    best_sens = metrics["sensitivity"]
                    best_thresh = thresh
            
            # Apply to test with persistence/hysteresis
            processor = AlarmProcessor(
                persistence_k=config["persistence"],
                use_hysteresis=config["hysteresis"],
                hysteresis_gap=0.1,
            )
            
            # Get test metrics
            test_metrics = compute_alarm_metrics_at_threshold(
                test_times, processed_test, test_intervals, best_thresh,
                refractory_sec=1200
            )
            
            results.append({
                "config": config["name"],
                "fah_target": fah_target,
                "threshold": best_thresh,
                "val_sens": best_sens,
                "test_fah": test_metrics["fah"],
                "test_sens": test_metrics["sensitivity"],
                "test_warning_time": test_metrics["mean_warning_time"],
            })
    
    # Save results
    csv_path = output_dir / "alarm_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "fah_target", "threshold", "val_sens", 
            "test_fah", "test_sens", "test_warning_time"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    console.print(f"  Saved: {csv_path}")
    
    # Print summary table
    table = Table(title="Alarm Ablation Results (FAH<=0.2)")
    table.add_column("Config")
    table.add_column("Test FAH")
    table.add_column("Test Sens")
    
    for r in results:
        if r["fah_target"] == 0.2:
            table.add_row(
                r["config"],
                f"{r['test_fah']:.2f}",
                f"{r['test_sens']:.2f}"
            )
    
    console.print(table)
    
    return results


def plot_threshold_curves(val_preds, val_labels, val_times, val_intervals,
                          test_preds, test_labels, test_times, test_intervals,
                          output_dir):
    """Generate threshold tuning curves."""
    console.print("\n[bold]Generating Threshold Curves[/bold]")
    
    # Compute curves
    val_curve = compute_threshold_curve(val_times, val_preds, val_intervals)
    test_curve = compute_threshold_curve(test_times, test_preds, test_intervals)
    
    # Plot 1: Threshold vs FAH
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(val_curve["thresholds"], val_curve["fah"], 'b-', label="Validation", linewidth=2)
    ax.plot(test_curve["thresholds"], test_curve["fah"], 'r--', label="Test", linewidth=2)
    ax.axhline(y=0.2, color='gray', linestyle=':', label="FAH target (0.2)")
    ax.axhline(y=0.5, color='gray', linestyle='-.', label="FAH target (0.5)")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("False Alarms per Hour (FAH)", fontsize=12)
    ax.set_title("Threshold vs FAH Curve", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(5, max(val_curve["fah"].max(), test_curve["fah"].max()))])
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_vs_FAH_curve.png", dpi=150)
    plt.close()
    console.print(f"  Saved: threshold_vs_FAH_curve.png")
    
    # Plot 2: Threshold vs Sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(val_curve["thresholds"], val_curve["sensitivity"], 'b-', label="Validation", linewidth=2)
    ax.plot(test_curve["thresholds"], test_curve["sensitivity"], 'r--', label="Test", linewidth=2)
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("Threshold vs Sensitivity Curve", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_vs_sensitivity_curve.png", dpi=150)
    plt.close()
    console.print(f"  Saved: threshold_vs_sensitivity_curve.png")
    
    # Plot 3: FAH vs Sensitivity tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(val_curve["fah"], val_curve["sensitivity"], 'b-o', label="Validation", linewidth=2, markersize=3)
    ax.plot(test_curve["fah"], test_curve["sensitivity"], 'r--s', label="Test", linewidth=2, markersize=3)
    ax.axvline(x=0.2, color='gray', linestyle=':', label="FAH=0.2")
    ax.axvline(x=0.5, color='gray', linestyle='-.', label="FAH=0.5")
    ax.set_xlabel("False Alarms per Hour (FAH)", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("FAH vs Sensitivity Tradeoff", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(5, max(val_curve["fah"].max(), test_curve["fah"].max()) + 0.5)])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / "FAH_sensitivity_tradeoff.png", dpi=150)
    plt.close()
    console.print(f"  Saved: FAH_sensitivity_tradeoff.png")


def run_loss_comparison(cfg, cache_path, device, output_dir):
    """Compare BCE vs Focal loss."""
    console.print("\n[bold]Running Loss Comparison (BCE vs Focal)[/bold]")
    
    results = []
    
    for loss_type in ["bce", "focal"]:
        console.print(f"  Training with {loss_type} loss...")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            cache_path=cache_path,
            cfg=cfg,
            include_features=cfg.model.use_features,
        )
        
        # Get model dimensions
        sample_batch = next(iter(train_loader))
        n_channels = sample_batch["data"].shape[1]
        n_features = sample_batch["features"].shape[1] if "features" in sample_batch else 0
        
        model = FusionNet(n_channels=n_channels, n_features=n_features, cfg=cfg)
        
        # Compute pos_weight for BCE
        pos_weight = None
        if loss_type == "bce":
            train_labels = []
            for batch in train_loader:
                train_labels.extend(batch["y_cls"].numpy())
            train_labels = np.array(train_labels)
            n_pos = (train_labels == 1).sum()
            n_neg = (train_labels == 0).sum()
            pos_weight = compute_pos_weight(n_pos, n_neg)
        
        # Create loss
        loss_fn = create_loss_fn(
            loss_type=loss_type,
            pos_weight=pos_weight,
            focal_alpha=0.25,
            focal_gamma=2.0,
        )
        
        # Create trainer with reduced epochs for comparison
        comparison_cfg = cfg.copy()
        comparison_cfg.training.epochs = min(10, cfg.training.epochs)
        
        run_dir = output_dir / f"loss_{loss_type}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=comparison_cfg,
            run_dir=run_dir,
        )
        
        best_metrics = trainer.train()
        
        # Evaluate
        best_checkpoint = run_dir / "checkpoints" / "best.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
        
        model.to(device)
        model.eval()
        
        # Get test predictions
        test_preds = []
        test_labels_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                data = batch["data"].to(device)
                features = batch.get("features")
                if features is not None:
                    features = features.to(device)
                
                cls_logit, _ = model(data, features if cfg.model.use_features else None)
                probs = torch.sigmoid(cls_logit).squeeze().cpu().numpy()
                
                if probs.ndim == 0:
                    probs = np.array([probs.item()])
                
                test_preds.extend(probs.flatten())
                test_labels_list.extend(batch["y_cls"].numpy().flatten())
        
        test_metrics = compute_metrics(np.array(test_labels_list).astype(int), np.array(test_preds))
        
        results.append({
            "loss_type": loss_type,
            "val_auroc": best_metrics.get("auroc", 0),
            "val_auprc": best_metrics.get("auprc", 0),
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_sensitivity": test_metrics["sensitivity"],
        })
        
        console.print(f"    Val AUROC: {best_metrics.get('auroc', 0):.4f}")
        console.print(f"    Test AUROC: {test_metrics['auroc']:.4f}")
    
    # Save results
    csv_path = output_dir / "loss_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "loss_type", "val_auroc", "val_auprc", "test_auroc", "test_auprc", "test_sensitivity"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    console.print(f"  Saved: {csv_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, default="configs/small_run.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--skip_loss_comparison", action="store_true")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(cfg.split.seed)
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")
    
    cache_path = Path(cfg.data.cache_root)
    
    # Find checkpoint if not specified
    if args.checkpoint is None:
        # Look for latest cross-subject checkpoint
        runs_dir = Path("runs")
        checkpoints = list(runs_dir.glob("deep_cross_subject_*/checkpoints/best.pt"))
        if not checkpoints:
            checkpoints = list(runs_dir.glob("*/checkpoints/best.pt"))
        
        if checkpoints:
            args.checkpoint = str(sorted(checkpoints)[-1])
            console.print(f"Using checkpoint: {args.checkpoint}")
        else:
            console.print("[red]No checkpoint found. Please specify --checkpoint[/red]")
            return 1
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        cache_path=cache_path,
        cfg=cfg,
        include_features=cfg.model.use_features,
    )
    
    # Get predictions
    console.print("\n[bold]Loading model and collecting predictions...[/bold]")
    val_preds, val_labels, val_times, val_logits = load_model_and_predict(
        args.checkpoint, cfg, val_loader, device
    )
    test_preds, test_labels, test_times, test_logits = load_model_and_predict(
        args.checkpoint, cfg, test_loader, device
    )
    
    console.print(f"  Val: {len(val_preds)} windows, {val_labels.sum()} preictal")
    console.print(f"  Test: {len(test_preds)} windows, {test_labels.sum()} preictal")
    
    # Extract seizure intervals
    val_intervals = extract_seizure_intervals(val_labels, val_times, cfg)
    test_intervals = extract_seizure_intervals(test_labels, test_times, cfg)
    
    console.print(f"  Val seizures: {len(val_intervals)}")
    console.print(f"  Test seizures: {len(test_intervals)}")
    
    # Run alarm ablation
    alarm_results = run_alarm_ablation(
        val_preds, val_labels, val_times, val_intervals,
        test_preds, test_labels, test_times, test_intervals,
        output_dir / "tables"
    )
    
    # Generate threshold curves
    plot_threshold_curves(
        val_preds, val_labels, val_times, val_intervals,
        test_preds, test_labels, test_times, test_intervals,
        output_dir / "figures"
    )
    
    # Run loss comparison (optional - takes time to retrain)
    if not args.skip_loss_comparison:
        loss_results = run_loss_comparison(cfg, cache_path, device, output_dir / "tables")
    
    console.print("\n[bold green]Ablations complete![/bold green]")
    console.print(f"Results saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

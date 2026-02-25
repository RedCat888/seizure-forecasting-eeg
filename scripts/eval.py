#!/usr/bin/env python3
"""
Evaluate seizure forecasting model with window-level and alarm-level metrics.

Usage:
    python scripts/eval.py --config configs/small_run.yaml --checkpoint runs/deep_model/checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import SeizureDataset
from src.models.fusion_net import FusionNet
from src.train.metrics import compute_metrics, compute_metrics_at_thresholds
from src.train.alarm_eval import compute_alarm_metrics, evaluate_at_thresholds
from src.chbmit import get_seizure_times


console = Console()


def get_recording_predictions(
    model,
    cache_path: Path,
    subject: str,
    device: torch.device,
    cfg,
):
    """
    Get predictions for all windows in a subject's recordings.
    
    Returns predictions grouped by original EDF file.
    """
    cache_file = cache_path / f"{subject}.h5"
    
    if not cache_file.exists():
        return {}
    
    model.eval()
    
    with h5py.File(cache_file, "r") as f:
        data = f["data"][:]
        features = f["features"][:] if "features" in f else None
        y_cls = f["y_cls"][:]
        y_tte = f["y_tte"][:]
        
    # Get predictions
    predictions = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = torch.from_numpy(data[i:i+batch_size]).float().to(device)
            batch_features = None
            if features is not None:
                batch_features = torch.from_numpy(features[i:i+batch_size]).float().to(device)
            
            probs = model.predict_proba(batch_data, batch_features)
            predictions.extend(probs.cpu().numpy())
    
    return np.array(predictions), y_cls, y_tte


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/small_run.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as checkpoint)",
    )
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    set_seed(cfg.split.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Output directory
    checkpoint_path = Path(args.checkpoint)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold blue]Model Evaluation[/bold blue]")
    console.print(f"Checkpoint: {checkpoint_path}")
    console.print(f"Output: {output_dir}")
    
    # Get cache path
    cache_path = Path(cfg.data.cache_root)
    
    # Get model dimensions from cache
    sample_file = cache_path / f"{cfg.split.test_subjects[0]}.h5"
    with h5py.File(sample_file, "r") as f:
        n_channels = f["data"].shape[1]
        n_features = f["features"].shape[1] if "features" in f else 0
    
    # Create model
    model = FusionNet(
        n_channels=n_channels,
        n_features=n_features,
        cfg=cfg,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    console.print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate on test subjects
    test_subjects = cfg.split.test_subjects
    console.print(f"\nEvaluating on {len(test_subjects)} test subjects...")
    
    all_preds = []
    all_labels = []
    all_tte = []
    
    for subject in test_subjects:
        predictions, y_cls, y_tte = get_recording_predictions(
            model, cache_path, subject, device, cfg
        )
        
        # Filter valid windows
        valid_mask = y_cls >= 0
        predictions = predictions[valid_mask]
        y_cls = y_cls[valid_mask]
        y_tte = y_tte[valid_mask]
        
        all_preds.extend(predictions)
        all_labels.extend(y_cls)
        all_tte.extend(y_tte)
        
        console.print(f"  {subject}: {len(predictions)} windows")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_tte = np.array(all_tte)
    
    # Window-level metrics
    console.print("\n[bold]Window-Level Metrics:[/bold]")
    
    thresholds = cfg.evaluation.threshold_sweep
    metrics_by_thresh = compute_metrics_at_thresholds(all_labels, all_preds, thresholds)
    
    table = Table()
    table.add_column("Threshold", style="cyan")
    table.add_column("AUROC", justify="right")
    table.add_column("AUPRC", justify="right")
    table.add_column("Sens", justify="right")
    table.add_column("Spec", justify="right")
    table.add_column("F1", justify="right")
    
    for thresh in thresholds:
        m = metrics_by_thresh[thresh]
        table.add_row(
            f"{thresh:.1f}",
            f"{m['auroc']:.4f}",
            f"{m['auprc']:.4f}",
            f"{m['sensitivity']:.4f}",
            f"{m['specificity']:.4f}",
            f"{m['f1']:.4f}",
        )
    
    console.print(table)
    
    # Overall metrics (threshold-independent)
    overall_metrics = compute_metrics(all_labels, all_preds, threshold=0.5)
    console.print(f"\nAUROC: {overall_metrics['auroc']:.4f}")
    console.print(f"AUPRC: {overall_metrics['auprc']:.4f}")
    
    # Save window metrics
    with open(output_dir / "window_metrics.csv", "w") as f:
        f.write("threshold,auroc,auprc,sensitivity,specificity,f1\n")
        for thresh in thresholds:
            m = metrics_by_thresh[thresh]
            f.write(f"{thresh},{m['auroc']},{m['auprc']},{m['sensitivity']},{m['specificity']},{m['f1']}\n")
    
    # Alarm-level metrics (simplified - would need recording-level data for full impl)
    console.print("\n[bold]Alarm-Level Metrics (per threshold):[/bold]")
    
    table = Table()
    table.add_column("Threshold", style="cyan")
    table.add_column("FAH", justify="right")
    table.add_column("Sensitivity", justify="right")
    table.add_column("Avg Warning (s)", justify="right")
    
    # Estimate FAH from window-level data
    # (Approximate - actual alarm eval needs temporal continuity)
    for thresh in thresholds:
        fp = np.sum((all_preds >= thresh) & (all_labels == 0))
        total_interictal = np.sum(all_labels == 0)
        
        # Approximate hours of interictal data
        window_hours = total_interictal * cfg.windowing.step_sec / 3600
        fah = fp / max(window_hours, 0.1)
        
        # Sensitivity
        tp = np.sum((all_preds >= thresh) & (all_labels == 1))
        total_preictal = np.sum(all_labels == 1)
        sens = tp / max(total_preictal, 1)
        
        # Average warning time for true positives
        true_pos_mask = (all_preds >= thresh) & (all_labels == 1)
        if np.any(true_pos_mask):
            avg_warning = np.mean(all_tte[true_pos_mask])
        else:
            avg_warning = 0
        
        table.add_row(
            f"{thresh:.1f}",
            f"{fah:.2f}",
            f"{sens:.4f}",
            f"{avg_warning:.1f}",
        )
    
    console.print(table)
    
    # Save predictions
    np.savez(
        output_dir / "predictions.npz",
        predictions=all_preds,
        labels=all_labels,
        tte=all_tte,
    )
    
    console.print(f"\nResults saved to {output_dir}")
    console.print("\n[bold green]Evaluation complete![/bold green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

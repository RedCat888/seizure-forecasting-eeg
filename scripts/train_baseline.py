#!/usr/bin/env python3
"""
Train baseline classifier using handcrafted EEG features.

Usage:
    python scripts/train_baseline.py --config configs/small_run.yaml
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.paths import get_run_dir
from src.utils.logging import setup_logging, get_logger
from src.data.dataset import load_cached_data, get_dataset_stats
from src.models.baseline import BaselineClassifier, train_baseline_model, compute_sample_weights


console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train baseline classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/small_run.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Set seed
    set_seed(cfg.split.seed)
    
    # Setup run directory
    run_dir = get_run_dir(cfg, create=True)
    run_dir = Path(str(run_dir).replace("full_run", "baseline").replace("small_run", "baseline"))
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(run_dir / "logs")
    logger = get_logger(__name__)
    
    console.print("[bold blue]Training Baseline Classifier[/bold blue]")
    console.print(f"Config: {args.config}")
    console.print(f"Run dir: {run_dir}")
    
    # Save config
    save_config(cfg, run_dir / "config.yaml")
    
    # Get cache path
    cache_path = Path(cfg.data.cache_root)
    
    # Load data
    console.print("\nLoading cached data...")
    
    X_train, y_train_cls, y_train_tte, y_train_soft = load_cached_data(
        cache_path, cfg.split.train_subjects
    )
    X_val, y_val_cls, y_val_tte, y_val_soft = load_cached_data(
        cache_path, cfg.split.val_subjects
    )
    X_test, y_test_cls, y_test_tte, y_test_soft = load_cached_data(
        cache_path, cfg.split.test_subjects
    )
    
    console.print(f"Train: {len(X_train)} samples")
    console.print(f"Val:   {len(X_val)} samples")
    console.print(f"Test:  {len(X_test)} samples")
    
    # Check data
    if len(X_train) == 0:
        console.print("[red]Error: No training data found![/red]")
        console.print("Run build_cache.py first.")
        return 1
    
    # Compute sample weights
    sample_weight = None
    if cfg.training.weight_by_proximity:
        sample_weight = compute_sample_weights(
            y_train_tte, y_train_cls, cfg.windowing.preictal_min
        )
    
    # Train model
    console.print(f"\nTraining {cfg.baseline.model_type} classifier...")
    
    model, train_metrics, val_metrics = train_baseline_model(
        X_train, y_train_cls,
        X_val, y_val_cls,
        cfg=cfg,
        sample_weight=sample_weight,
    )
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test_cls)
    
    # Display results
    console.print("\n[bold]Results:[/bold]")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("Test", justify="right")
    
    for metric in ["auroc", "auprc", "sensitivity", "specificity", "accuracy"]:
        table.add_row(
            metric.upper(),
            f"{train_metrics[metric]:.4f}",
            f"{val_metrics[metric]:.4f}",
            f"{test_metrics[metric]:.4f}",
        )
    
    console.print(table)
    
    # Save model
    model_path = run_dir / "baseline_model.pkl"
    model.save(model_path)
    console.print(f"\nModel saved to {model_path}")
    
    # Save metrics
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("split,auroc,auprc,sensitivity,specificity,accuracy\n")
        f.write(f"train,{train_metrics['auroc']},{train_metrics['auprc']},{train_metrics['sensitivity']},{train_metrics['specificity']},{train_metrics['accuracy']}\n")
        f.write(f"val,{val_metrics['auroc']},{val_metrics['auprc']},{val_metrics['sensitivity']},{val_metrics['specificity']},{val_metrics['accuracy']}\n")
        f.write(f"test,{test_metrics['auroc']},{test_metrics['auprc']},{test_metrics['sensitivity']},{test_metrics['specificity']},{test_metrics['accuracy']}\n")
    
    console.print(f"Metrics saved to {metrics_path}")
    console.print("\n[bold green]Training complete![/bold green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

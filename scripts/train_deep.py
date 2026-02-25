#!/usr/bin/env python3
"""
Train deep learning model (FusionNet) for seizure forecasting.

Uses:
- Spectrogram CNN for raw EEG
- Feature MLP for handcrafted features
- Fusion head for combined prediction

Supports:
- Cross-subject (patient-wise) splits
- Within-subject (patient-specific) splits
- Data augmentation

Usage:
    python scripts/train_deep.py --config configs/small_run.yaml
    python scripts/train_deep.py --config configs/small_run.yaml --split_mode within_subject --subject chb01
"""

import argparse
import sys
from pathlib import Path
import torch
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.paths import get_run_dir
from src.utils.logging import setup_logging, get_logger
from src.data.dataset import (
    SeizureDataset, 
    create_dataloaders, 
    create_dataloaders_from_config,
    create_within_subject_dataloaders,
)
from src.data.augmentation import create_augmenter
from src.models.fusion_net import FusionNet
from src.train.loops import Trainer
from src.features.handcrafted import get_feature_names


console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train deep model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/small_run.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["cross_subject", "within_subject"],
        default=None,
        help="Override split mode from config",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Subject for within-subject mode (overrides config)",
    )
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Override split mode if specified
    split_mode = args.split_mode or cfg.split.get("mode", "cross_subject")
    within_subject_id = args.subject or cfg.split.get("within_subject_id", "chb01")
    
    # Set seed
    set_seed(cfg.split.seed)
    
    # Setup run directory with split mode in name
    run_dir = get_run_dir(cfg, create=True)
    run_dir = Path(str(run_dir).replace("full_run", f"deep_{split_mode}").replace("small_run", f"deep_{split_mode}"))
    if split_mode == "within_subject":
        run_dir = Path(str(run_dir) + f"_{within_subject_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging(run_dir / "logs")
    logger = get_logger(__name__)
    
    console.print("[bold blue]Training Deep Model (FusionNet)[/bold blue]")
    console.print(f"Config: {args.config}")
    console.print(f"Split mode: [bold]{split_mode}[/bold]")
    if split_mode == "within_subject":
        console.print(f"Subject: [bold]{within_subject_id}[/bold]")
    console.print(f"Run dir: {run_dir}")
    
    # Save config
    save_config(cfg, run_dir / "config.yaml")
    
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: {device}")
    
    if device.type == "cuda":
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        console.print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get cache path
    cache_path = Path(cfg.data.cache_root)
    data_root = Path(cfg.data.raw_root)
    
    # Create dataloaders based on split mode
    console.print("\nCreating dataloaders...")
    
    if split_mode == "within_subject":
        train_loader, val_loader, test_loader = create_within_subject_dataloaders(
            cache_path=cache_path,
            cfg=cfg,
            subject=within_subject_id,
            include_features=cfg.model.use_features,
            data_root=data_root,
        )
        console.print(f"[dim]Within-subject split: {within_subject_id}[/dim]")
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            cache_path=cache_path,
            cfg=cfg,
            include_features=cfg.model.use_features,
        )
        console.print(f"[dim]Cross-subject: train={cfg.split.train_subjects}, val={cfg.split.val_subjects}, test={cfg.split.test_subjects}[/dim]")
    
    console.print(f"Train batches: {len(train_loader)}")
    console.print(f"Val batches: {len(val_loader)}")
    console.print(f"Test batches: {len(test_loader)}")
    
    # Get model dimensions from data
    sample_batch = next(iter(train_loader))
    n_channels = sample_batch["data"].shape[1]
    n_features = sample_batch["features"].shape[1] if "features" in sample_batch else 0
    
    console.print(f"Channels: {n_channels}")
    console.print(f"Features: {n_features}")
    
    # Create model
    model = FusionNet(
        n_channels=n_channels,
        n_features=n_features,
        cfg=cfg,
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Model parameters: {n_params:,} (trainable: {n_trainable:,})")
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        console.print(f"Resumed from {args.resume}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=run_dir,
    )
    
    # Train
    console.print("\n[bold]Starting training...[/bold]\n")
    
    best_metrics = trainer.train()
    
    # Display final results
    console.print("\n[bold]Best Validation Results:[/bold]")
    for k, v in best_metrics.items():
        console.print(f"  {k}: {v:.4f}")
    
    # Final evaluation on test set
    console.print("\nEvaluating on test set...")
    
    # Load best model
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    if best_checkpoint.exists():
        # weights_only=False needed for OmegaConf DictConfig in checkpoint
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)
    model.eval()
    
    from src.train.loops import validate_epoch
    from src.train.losses import SeizureForecastingLoss
    
    loss_fn = SeizureForecastingLoss()
    test_metrics, _, _ = validate_epoch(
        model, test_loader, loss_fn, device,
        use_amp=cfg.training.use_amp,
        use_features=cfg.model.use_features,
    )
    
    console.print("\n[bold]Test Results:[/bold]")
    for k, v in test_metrics.items():
        console.print(f"  {k}: {v:.4f}")
    
    # Save final metrics
    metrics_path = run_dir / "final_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Validation Metrics:\n")
        for k, v in best_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nTest Metrics:\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
    
    console.print(f"\nResults saved to {run_dir}")
    console.print("\n[bold green]Training complete![/bold green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

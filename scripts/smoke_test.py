#!/usr/bin/env python3
"""Smoke test for training pipeline - run 2 epochs to verify everything works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
import torch

from src.utils.config import load_config
from src.data.cache_v2 import create_dataloaders_v2
from src.models.fusion_net import FusionNet
from src.train.loops import Trainer

def main():
    print("=" * 60)
    print("SMOKE TEST - 2 epochs on mini split")
    print("=" * 60)
    
    cfg = load_config('configs/full_run.yaml')
    cfg.training.epochs = 2  # Just 2 epochs for smoke test

    # Create small dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_v2(
        cache_dir=Path('data/chbmit_cache_v2'),
        train_subjects=['chb02', 'chb03', 'chb04', 'chb05'],
        val_subjects=['chb23'],
        test_subjects=['chb01'],
        batch_size=128,
        num_workers=0,
    )

    # Create model
    model = FusionNet(n_channels=18, n_features=30, cfg=cfg)

    # Create trainer with run_dir
    run_dir = Path('runs/smoke_test')
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, train_loader, val_loader, cfg, run_dir)
    print(f'Trainer created successfully!')
    print(f'  save_every_n_epochs: {trainer.save_every_n_epochs}')
    print(f'  epochs: {trainer.epochs}')

    # Train
    best_metrics = trainer.train()
    
    auroc = best_metrics.get('auroc', 0)
    print(f'\nSmoke test PASSED! Best AUROC: {auroc:.4f}')
    return 0

if __name__ == "__main__":
    sys.exit(main())

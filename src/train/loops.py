"""
Training loops for seizure forecasting models.

Supports:
- Mixed precision (AMP) training
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Checkpointing

PERFORMANCE OPTIMIZATIONS:
- Uses torch.amp (new API) for mixed precision
- Device verification logging
- Non-blocking GPU transfers
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # Updated API
from tqdm import tqdm
from omegaconf import DictConfig

from .losses import SeizureForecastingLoss, compute_pos_weight, create_loss_fn
from .metrics import compute_metrics
from ..utils.logging import get_logger, MetricsLogger

logger = get_logger(__name__)

# Flag to print device info once
_device_verified = False


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: SeizureForecastingLoss,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    use_features: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    PERFORMANCE: Uses non-blocking GPU transfers and AMP for throughput.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to use
        scaler: GradScaler for AMP
        use_amp: Whether to use mixed precision
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_features: Whether to use handcrafted features
        
    Returns:
        Dict of training metrics
    """
    global _device_verified
    
    model.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_soft_loss = 0.0
    n_batches = 0
    
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for step, batch in enumerate(pbar):
        # PERFORMANCE: Non-blocking transfers with pin_memory
        data = batch["data"].to(device, non_blocking=True)
        y_cls = batch["y_cls"].to(device, non_blocking=True)
        y_tte = batch["y_tte"].to(device, non_blocking=True)
        y_soft = batch["y_soft"].to(device, non_blocking=True)
        
        features = None
        if use_features and "features" in batch:
            features = batch["features"].to(device, non_blocking=True)
        
        # PERFORMANCE: Verify CUDA placement once
        if not _device_verified and device.type == 'cuda':
            logger.info(f"[DEVICE CHECK] data.device={data.device}, model on cuda={next(model.parameters()).device}")
            _device_verified = True
        
        # Forward pass with AMP (uses new torch.amp API)
        with autocast(device_type=device.type, enabled=use_amp):
            cls_logit, soft_pred = model(data, features)
            loss, loss_dict = loss_fn(cls_logit, soft_pred, y_cls, y_soft, y_tte)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss_dict["loss_total"]
        total_cls_loss += loss_dict["loss_cls"]
        total_soft_loss += loss_dict["loss_soft"]
        n_batches += 1
        
        # Collect predictions for epoch metrics
        with torch.no_grad():
            probs = torch.sigmoid(cls_logit).cpu().numpy()
            all_preds.extend(probs.ravel())
            all_labels.extend(y_cls.cpu().numpy().ravel())
        
        pbar.set_postfix({"loss": loss_dict["loss_total"]})
    
    # Compute epoch metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    epoch_metrics = compute_metrics(all_labels, all_preds)
    epoch_metrics["loss"] = total_loss / max(n_batches, 1)
    epoch_metrics["loss_cls"] = total_cls_loss / max(n_batches, 1)
    epoch_metrics["loss_soft"] = total_soft_loss / max(n_batches, 1)
    
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: SeizureForecastingLoss,
    device: torch.device,
    use_amp: bool = True,
    use_features: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Validate for one epoch.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to use
        use_amp: Whether to use mixed precision
        use_features: Whether to use handcrafted features
        
    Returns:
        Tuple of (metrics dict, all predictions, all labels)
    """
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # PERFORMANCE: Non-blocking transfers
            data = batch["data"].to(device, non_blocking=True)
            y_cls = batch["y_cls"].to(device, non_blocking=True)
            y_tte = batch["y_tte"].to(device, non_blocking=True)
            y_soft = batch["y_soft"].to(device, non_blocking=True)
            
            features = None
            if use_features and "features" in batch:
                features = batch["features"].to(device, non_blocking=True)
            
            # Use new torch.amp API
            with autocast(device_type=device.type, enabled=use_amp):
                cls_logit, soft_pred = model(data, features)
                loss, loss_dict = loss_fn(cls_logit, soft_pred, y_cls, y_soft, y_tte)
            
            total_loss += loss_dict["loss_total"]
            n_batches += 1
            
            probs = torch.sigmoid(cls_logit).cpu().numpy()
            all_preds.extend(probs.ravel())
            all_labels.extend(y_cls.cpu().numpy().ravel())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    epoch_metrics = compute_metrics(all_labels, all_preds)
    epoch_metrics["loss"] = total_loss / max(n_batches, 1)
    
    return epoch_metrics, all_preds, all_labels


class Trainer:
    """
    Complete training pipeline with logging, checkpointing, and early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        run_dir: Path,
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            cfg: Configuration
            run_dir: Directory for checkpoints and logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Training settings
        train_cfg = cfg.training
        self.epochs = train_cfg.epochs
        self.use_amp = train_cfg.use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = train_cfg.gradient_accumulation_steps
        self.use_features = cfg.model.get("use_features", True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        
        # Learning rate scheduler
        if train_cfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=train_cfg.epochs,
                eta_min=train_cfg.learning_rate * 0.01,
            )
        elif train_cfg.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_cfg.epochs // 3,
                gamma=0.1,
            )
        else:
            self.scheduler = None
        
        # Loss function - use config to determine type
        _, pos_weight = train_loader.dataset.get_class_weights()
        loss_cfg = cfg.get("loss", {})
        loss_type = loss_cfg.get("type", "bce")
        self.loss_fn = create_loss_fn(
            loss_type=loss_type,
            pos_weight=pos_weight if loss_type == "bce" else None,
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            lambda_soft=train_cfg.lambda_soft,
            weight_by_proximity=train_cfg.weight_by_proximity,
            preictal_sec=cfg.windowing.preictal_min * 60,
        )
        
        # AMP scaler (use new API to avoid deprecation warning)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self.best_auroc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stopping_patience = train_cfg.early_stopping_patience
        
        # Logging config with defensive defaults
        logging_cfg = cfg.get("logging", {})
        self.save_every_n_epochs = logging_cfg.get("save_every_n_epochs", 5)
        self.log_every_n_steps = logging_cfg.get("log_every_n_steps", 50)
        
        # Logging
        self.metrics_logger = MetricsLogger(self.run_dir / "logs" / "metrics.csv")
    
    def train(self) -> Dict[str, float]:
        """
        Run complete training loop.
        
        Returns:
            Best validation metrics
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Using AMP: {self.use_amp}")
        
        best_metrics = {}
        
        for epoch in range(1, self.epochs + 1):
            logger.info(f"Epoch {epoch}/{self.epochs}")
            
            # Train
            train_metrics = train_epoch(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                device=self.device,
                scaler=self.scaler,
                use_amp=self.use_amp,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                use_features=self.use_features,
            )
            
            # Validate
            val_metrics, val_preds, val_labels = validate_epoch(
                model=self.model,
                val_loader=self.val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                use_amp=self.use_amp,
                use_features=self.use_features,
            )
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            lr = self.optimizer.param_groups[0]["lr"]
            log_metrics = {
                "epoch": epoch,
                "lr": lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.metrics_logger.log(log_metrics)
            
            logger.info(
                f"Train: loss={train_metrics['loss']:.4f}, auroc={train_metrics['auroc']:.4f} | "
                f"Val: loss={val_metrics['loss']:.4f}, auroc={val_metrics['auroc']:.4f}"
            )
            
            # Check for best model
            if val_metrics["auroc"] > self.best_auroc:
                self.best_auroc = val_metrics["auroc"]
                self.best_epoch = epoch
                self.patience_counter = 0
                best_metrics = val_metrics.copy()
                
                # Save best checkpoint
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model! AUROC: {self.best_auroc:.4f}")
            else:
                self.patience_counter += 1
            
            # Periodic checkpoint
            if epoch % self.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info(f"Training complete. Best AUROC: {self.best_auroc:.4f} at epoch {self.best_epoch}")
        
        return best_metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_auroc": self.best_auroc,
            "config": dict(self.cfg),
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        else:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch}.pt")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model from checkpoint."""
        # weights_only=False needed for OmegaConf DictConfig in checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")


def load_model_for_eval(
    model: nn.Module,
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load model for evaluation.
    
    Args:
        model: Model architecture (uninitialized weights)
        checkpoint_path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Model with loaded weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # weights_only=False needed for OmegaConf DictConfig in checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model

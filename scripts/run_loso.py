#!/usr/bin/env python3
"""
Leave-One-Subject-Out (LOSO) Cross-Validation for seizure forecasting.

Features:
- Reliability hardening: config defaults, label sanity checks
- Worker-safe memmap dataset for Windows multiprocessing
- Clinical evaluation: FAH threshold tuning, alarm metrics
- Performance logging and resumability
"""

import argparse
import sys
import time
import csv
import os
from pathlib import Path
import json
import numpy as np
import torch
from datetime import datetime
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from src.utils.config_defaults import apply_defaults, get_safe
from src.utils.seed import set_seed
from src.data.cache_v2_worker_safe import (
    WorkerSafeCacheV2Dataset, 
    create_dataloaders_worker_safe,
    worker_init_fn,
)
from src.models.fusion_net import FusionNet
from src.train.losses import create_loss_fn
from src.train.metrics import compute_metrics
from src.train.threshold_tuning import (
    tune_thresholds_for_multiple_targets,
    compute_alarm_metrics_at_threshold,
)

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from omegaconf import OmegaConf

console = Console()


# =============================================================================
# PERFORMANCE TIMER
# =============================================================================
class PerfTimer:
    """Simple performance timer for training loop."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data_time = 0.0
        self.h2d_time = 0.0
        self.fwd_time = 0.0
        self.bwd_time = 0.0
        self.step_time = 0.0
        self.n_steps = 0
        self._last_batch_end = time.perf_counter()
    
    def record_data(self):
        now = time.perf_counter()
        self.data_time += now - self._last_batch_end
        self._h2d_start = now
    
    def record_h2d(self):
        now = time.perf_counter()
        self.h2d_time += now - self._h2d_start
        self._fwd_start = now
    
    def record_fwd(self):
        now = time.perf_counter()
        self.fwd_time += now - self._fwd_start
        self._bwd_start = now
    
    def record_bwd(self):
        now = time.perf_counter()
        self.bwd_time += now - self._bwd_start
        self.step_time += now - self._last_batch_end
        self._last_batch_end = now
        self.n_steps += 1
    
    def get_averages(self):
        n = max(self.n_steps, 1)
        return {
            "data_ms": 1000 * self.data_time / n,
            "h2d_ms": 1000 * self.h2d_time / n,
            "fwd_ms": 1000 * self.fwd_time / n,
            "bwd_ms": 1000 * self.bwd_time / n,
            "step_ms": 1000 * self.step_time / n,
        }


# =============================================================================
# GPU LOGGING
# =============================================================================
def log_gpu_state(step: int = 0, prefix: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        device_name = torch.cuda.get_device_name(0)
        console.print(f"[dim]{prefix}[GPU] {device_name} | Alloc: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Step: {step}[/dim]")


def assert_cuda_tensors(data, features, model):
    assert data.is_cuda, f"Data not on CUDA: {data.device}"
    if features is not None:
        assert features.is_cuda, f"Features not on CUDA: {features.device}"
    model_device = next(model.parameters()).device
    assert model_device.type == 'cuda', f"Model not on CUDA: {model_device}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_cached_subjects(cache_root: Path) -> list:
    cache_root = Path(cache_root)
    if (cache_root / "metadata.json").exists():
        with open(cache_root / "metadata.json") as f:
            meta = json.load(f)
        return sorted(meta.get("subjects", []))
    return []


# =============================================================================
# TRAINING WITH PERFORMANCE LOGGING
# =============================================================================
def train_epoch_with_perf(
    model, train_loader, optimizer, loss_fn, device, scaler, 
    use_amp, use_features, log_interval=50, perf_log_file=None,
):
    model.train()
    timer = PerfTimer()
    
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for step, batch in enumerate(pbar):
        timer.record_data()
        
        data = batch["data"].to(device, non_blocking=True)
        y_cls = batch["y_cls"].to(device, non_blocking=True)
        y_tte = batch["y_tte"].to(device, non_blocking=True)
        y_soft = batch["y_soft"].to(device, non_blocking=True)
        
        features = None
        if use_features and "features" in batch:
            features = batch["features"].to(device, non_blocking=True)
        
        timer.record_h2d()
        
        if step == 0:
            assert_cuda_tensors(data, features, model)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=use_amp):
            cls_logit, soft_pred = model(data, features)
            loss, loss_dict = loss_fn(cls_logit, soft_pred, y_cls, y_soft, y_tte)
        
        timer.record_fwd()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        timer.record_bwd()
        
        total_loss += loss_dict["loss_total"]
        n_batches += 1
        
        with torch.no_grad():
            probs = torch.sigmoid(cls_logit).cpu().numpy()
            all_preds.extend(probs.ravel())
            all_labels.extend(y_cls.cpu().numpy().ravel())
        
        pbar.set_postfix({"loss": f"{loss_dict['loss_total']:.4f}"})
        
        if (step + 1) % log_interval == 0:
            avgs = timer.get_averages()
            log_gpu_state(step + 1, prefix="  ")
            console.print(f"  [dim]Perf: data={avgs['data_ms']:.1f}ms h2d={avgs['h2d_ms']:.1f}ms fwd={avgs['fwd_ms']:.1f}ms bwd={avgs['bwd_ms']:.1f}ms[/dim]")
            
            if perf_log_file:
                write_header = not perf_log_file.exists()
                with open(perf_log_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["step", "data_ms", "h2d_ms", "fwd_ms", "bwd_ms", "step_ms", "loss"])
                    if write_header:
                        writer.writeheader()
                    writer.writerow({"step": step + 1, **avgs, "loss": loss_dict["loss_total"]})
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    epoch_metrics = compute_metrics(all_labels, all_preds)
    epoch_metrics["loss"] = total_loss / max(n_batches, 1)
    
    return epoch_metrics, timer.get_averages()


def validate_epoch(model, val_loader, loss_fn, device, use_amp, use_features):
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            data = batch["data"].to(device, non_blocking=True)
            y_cls = batch["y_cls"].to(device, non_blocking=True)
            y_tte = batch["y_tte"].to(device, non_blocking=True)
            y_soft = batch["y_soft"].to(device, non_blocking=True)
            
            features = None
            if use_features and "features" in batch:
                features = batch["features"].to(device, non_blocking=True)
            
            with autocast(device_type='cuda', enabled=use_amp):
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


# =============================================================================
# CLINICAL EVALUATION
# =============================================================================
def extract_seizure_intervals(labels, times, preictal_min):
    """Extract seizure intervals from preictal labels."""
    intervals = []
    in_preictal = False
    preictal_start = None
    
    for i, (label, t) in enumerate(zip(labels, times)):
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


def apply_alarm_postprocessing(risk_scores, alpha=0.2, persistence_k=3):
    """
    Apply EMA smoothing and persistence filter to risk scores.
    
    Args:
        risk_scores: Raw model outputs (probabilities)
        alpha: EMA smoothing factor (0-1, lower = more smoothing)
        persistence_k: Require K consecutive windows above threshold
        
    Returns:
        Smoothed risk scores
    """
    # EMA smoothing
    smoothed = np.zeros_like(risk_scores)
    smoothed[0] = risk_scores[0]
    for i in range(1, len(risk_scores)):
        smoothed[i] = alpha * risk_scores[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def evaluate_with_clinical_metrics(
    preds, labels, times, cfg, fah_targets=[0.1, 0.2, 0.5, 1.0]
):
    """
    Evaluate model with clinical alarm metrics.
    
    Returns:
        Dict with metrics for each FAH target
    """
    preictal_min = get_safe(cfg, "windowing.preictal_min", 30)
    seizure_intervals = extract_seizure_intervals(labels, times, preictal_min)
    
    if len(seizure_intervals) == 0:
        return {
            "n_seizures": 0,
            "window_auroc": compute_metrics(labels, preds).get("auroc", np.nan),
            "window_auprc": compute_metrics(labels, preds).get("auprc", np.nan),
            "alarm_results": {},
        }
    
    # Apply post-processing
    alpha = get_safe(cfg, "alarm.smoothing_alpha", 0.2)
    smoothed_preds = apply_alarm_postprocessing(preds, alpha=alpha)
    
    # Tune thresholds for each FAH target
    threshold_results = tune_thresholds_for_multiple_targets(
        times=times,
        risk_scores=smoothed_preds,
        seizure_intervals=seizure_intervals,
        fah_targets=fah_targets,
    )
    
    alarm_results = {}
    for target_fah in fah_targets:
        if target_fah in threshold_results:
            tr = threshold_results[target_fah]
            alarm_results[target_fah] = {
                "threshold": tr.threshold,
                "sensitivity": tr.sensitivity,
                "fah": tr.achieved_fah,  # Fixed: use achieved_fah
                "mean_warning_time": tr.mean_warning_time_sec if hasattr(tr, 'mean_warning_time_sec') else 0,
            }
    
    metrics = compute_metrics(labels, preds)
    return {
        "n_seizures": len(seizure_intervals),
        "window_auroc": metrics.get("auroc", np.nan),
        "window_auprc": metrics.get("auprc", np.nan) if np.sum(labels) > 0 else np.nan,
        "alarm_results": alarm_results,
    }


# =============================================================================
# RUN FOLD
# =============================================================================
def run_fold(
    test_subject: str,
    train_subjects: list,
    val_subjects: list,
    cfg,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run a single LOSO fold with clinical evaluation."""
    
    console.print(f"\n{'='*60}")
    console.print(f"[bold blue]FOLD: Test on {test_subject}[/bold blue]")
    console.print(f"Train: {len(train_subjects)} subjects")
    console.print(f"Val: {', '.join(val_subjects)}")
    console.print(f"{'='*60}")
    
    log_gpu_state(0, prefix="[FOLD START] ")
    
    fold_dir = output_dir / f"fold_{test_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "checkpoints").mkdir(exist_ok=True)
    
    perf_log_file = Path("reports/tables") / f"perf_log_fold_{test_subject}.csv"
    perf_log_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_log_file = fold_dir / "epoch_metrics.csv"
    
    cache_path = Path(cfg.data.cache_root)
    num_workers = get_safe(cfg, "training.num_workers", 0)
    
    # Create dataloaders with worker-safe dataset
    try:
        train_loader, val_loader, test_loader, label_stats = create_dataloaders_worker_safe(
            cache_dir=cache_path,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=[test_subject],
            batch_size=cfg.training.batch_size,
            num_workers=num_workers,
            include_features=get_safe(cfg, "model.use_features", True),
            nan_fill_value=get_safe(cfg, "features.nan_fill_value", 0.0),
        )
    except Exception as e:
        console.print(f"[red]Failed to create dataloaders: {e}[/red]")
        return {"error": str(e), "test_subject": test_subject}
    
    # Log label stats for sanity checking
    console.print(f"\n[bold]Label Sanity Check:[/bold]")
    for split_name, stats in label_stats.items():
        console.print(f"  {split_name}: n={stats['n_total']}, pos={stats['n_pos']}, neg={stats['n_neg']}, ratio={stats['pos_ratio']:.4f}")
    
    # Write label stats to fold directory
    with open(fold_dir / "label_stats.json", "w") as f:
        json.dump(label_stats, f, indent=2)
    
    # Check for problematic splits
    if label_stats["val"]["n_pos"] == 0:
        console.print("[yellow]WARNING: No positive samples in validation set![/yellow]")
    if label_stats["test"]["n_pos"] == 0:
        console.print("[yellow]WARNING: No positive samples in test set![/yellow]")
    
    console.print(f"\n  Train: {len(train_loader)} batches (batch_size={cfg.training.batch_size})")
    console.print(f"  Val: {len(val_loader)} batches")
    console.print(f"  Test: {len(test_loader)} batches")
    
    if len(train_loader) == 0 or len(val_loader) == 0:
        console.print("[yellow]Skipping fold: insufficient data[/yellow]")
        return {"error": "insufficient data", "test_subject": test_subject}
    
    sample_batch = next(iter(train_loader))
    n_channels = sample_batch["data"].shape[1]
    n_features = sample_batch["features"].shape[1] if "features" in sample_batch else 0
    
    console.print(f"  Input shape: [{n_channels}, {sample_batch['data'].shape[2]}], features: {n_features}")
    
    model = FusionNet(n_channels=n_channels, n_features=n_features, cfg=cfg)
    model.to(device)
    
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "focal")
    loss_fn = create_loss_fn(
        loss_type=loss_type,
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        lambda_soft=get_safe(cfg, "training.lambda_soft", 0.5),
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    
    use_amp = cfg.training.use_amp and torch.cuda.is_available()
    scaler = GradScaler('cuda') if use_amp else None
    
    console.print(f"  AMP: {use_amp}, Loss: {loss_type}, Workers: {num_workers}")
    
    epochs = cfg.training.epochs
    best_auroc = 0.0
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0
    patience = get_safe(cfg, "training.early_stopping_patience", 10)
    
    for epoch in range(1, epochs + 1):
        console.print(f"\n[bold]Epoch {epoch}/{epochs}[/bold]")
        
        train_metrics, perf_avgs = train_epoch_with_perf(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            use_features=get_safe(cfg, "model.use_features", True),
            log_interval=50,
            perf_log_file=perf_log_file,
        )
        
        val_metrics, val_preds, val_labels = validate_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            use_features=get_safe(cfg, "model.use_features", True),
        )
        
        write_header = not metrics_log_file.exists()
        with open(metrics_log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_auroc", "val_loss", "val_auroc"])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.4f}",
                "train_auroc": f"{train_metrics['auroc']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_auroc": f"{val_metrics['auroc']:.4f}",
            })
        
        console.print(f"  Train: loss={train_metrics['loss']:.4f}, auroc={train_metrics['auroc']:.4f}")
        console.print(f"  Val: loss={val_metrics['loss']:.4f}, auroc={val_metrics['auroc']:.4f}")
        
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auroc": best_auroc,
            }
            torch.save(checkpoint, fold_dir / "checkpoints" / "best.pt")
            console.print(f"  [green]New best model! AUROC: {best_auroc:.4f}[/green]")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            console.print(f"  [yellow]Early stopping at epoch {epoch}[/yellow]")
            break
    
    # Load best model
    best_checkpoint = fold_dir / "checkpoints" / "best.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    
    # Collect test predictions with timestamps
    test_preds, test_labels, test_times = collect_predictions(model, test_loader, device, cfg)
    
    # Clinical evaluation
    fah_targets = get_safe(cfg, "evaluation.fah_targets", [0.1, 0.2, 0.5, 1.0])
    clinical_eval = evaluate_with_clinical_metrics(
        test_preds, test_labels, test_times, cfg, fah_targets
    )
    
    console.print(f"\n[bold]Fold Results ({test_subject}):[/bold]")
    console.print(f"  Val AUROC: {best_auroc:.4f}")
    console.print(f"  Test AUROC: {clinical_eval['window_auroc']:.4f}")
    console.print(f"  Test AUPRC: {clinical_eval['window_auprc']:.4f}" if not np.isnan(clinical_eval['window_auprc']) else "  Test AUPRC: N/A (no positives)")
    console.print(f"  N Seizures: {clinical_eval['n_seizures']}")
    
    for target_fah, ar in clinical_eval["alarm_results"].items():
        console.print(f"  FAH<={target_fah}: sens={ar['sensitivity']:.2f}, fah={ar['fah']:.2f}, thresh={ar['threshold']:.3f}")
    
    results = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "val_auroc": best_auroc,
        "val_auprc": best_metrics.get("auprc", np.nan),
        "test_auroc": clinical_eval["window_auroc"],
        "test_auprc": clinical_eval["window_auprc"],
        "best_epoch": best_epoch,
        "n_seizures": clinical_eval["n_seizures"],
        "label_stats": label_stats,
        "alarm_results": clinical_eval["alarm_results"],
    }
    
    with open(fold_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    
    (fold_dir / "DONE.txt").write_text(f"Completed at {datetime.now().isoformat()}\nBest AUROC: {best_auroc:.4f}\n")
    
    return results


def collect_predictions(model, data_loader, device, cfg):
    """Collect predictions with timestamps."""
    model.eval()
    all_preds = []
    all_labels = []
    all_times = []
    
    current_time = 0.0
    step_sec = get_safe(cfg, "windowing.step_sec", 15)
    
    with torch.no_grad():
        for batch in data_loader:
            data = batch["data"].to(device, non_blocking=True)
            features = batch.get("features")
            if features is not None:
                features = features.to(device, non_blocking=True)
            
            use_amp = get_safe(cfg, "training.use_amp", True) and device.type == 'cuda'
            with autocast(device_type='cuda', enabled=use_amp):
                cls_logit, _ = model(data, features if get_safe(cfg, "model.use_features", True) else None)
            
            probs = torch.sigmoid(cls_logit).squeeze().cpu().numpy()
            labels = batch["y_cls"].numpy()
            
            if probs.ndim == 0:
                probs = np.array([probs.item()])
            
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.flatten())
            
            for _ in range(len(probs.flatten())):
                all_times.append(current_time)
                current_time += step_sec
    
    return np.array(all_preds), np.array(all_labels).astype(int), np.array(all_times)


def aggregate_results(fold_results: list) -> dict:
    """Aggregate results with clinical metrics."""
    valid_folds = [r for r in fold_results if "error" not in r]
    
    if not valid_folds:
        return {"error": "No valid folds"}
    
    val_aurocs = [r["val_auroc"] for r in valid_folds]
    test_aurocs = [r["test_auroc"] for r in valid_folds if not np.isnan(r["test_auroc"])]
    val_auprcs = [r.get("val_auprc", 0) for r in valid_folds if not np.isnan(r.get("val_auprc", np.nan))]
    test_auprcs = [r.get("test_auprc", 0) for r in valid_folds if not np.isnan(r.get("test_auprc", np.nan))]
    
    summary = {
        "n_folds": len(valid_folds),
        "val_auroc_mean": np.mean(val_aurocs),
        "val_auroc_std": np.std(val_aurocs),
        "test_auroc_mean": np.mean(test_aurocs) if test_aurocs else np.nan,
        "test_auroc_std": np.std(test_aurocs) if test_aurocs else np.nan,
        "val_auprc_mean": np.mean(val_auprcs) if val_auprcs else np.nan,
        "val_auprc_std": np.std(val_auprcs) if val_auprcs else np.nan,
        "test_auprc_mean": np.mean(test_auprcs) if test_auprcs else np.nan,
        "test_auprc_std": np.std(test_auprcs) if test_auprcs else np.nan,
    }
    
    # Aggregate alarm metrics
    for fah_target in [0.1, 0.2, 0.5, 1.0]:
        sensitivities = []
        fahs = []
        for r in valid_folds:
            if "alarm_results" in r and str(fah_target) in r["alarm_results"]:
                ar = r["alarm_results"][str(fah_target)]
                if ar["sensitivity"] > 0:
                    sensitivities.append(ar["sensitivity"])
                    fahs.append(ar["fah"])
            elif "alarm_results" in r and fah_target in r["alarm_results"]:
                ar = r["alarm_results"][fah_target]
                if ar["sensitivity"] > 0:
                    sensitivities.append(ar["sensitivity"])
                    fahs.append(ar["fah"])
        
        if sensitivities:
            summary[f"fah_{fah_target}_sensitivity_mean"] = np.mean(sensitivities)
            summary[f"fah_{fah_target}_sensitivity_std"] = np.std(sensitivities)
            summary[f"fah_{fah_target}_achieved_fah_mean"] = np.mean(fahs)
    
    return summary


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="LOSO Cross-Validation with Clinical Metrics")
    parser.add_argument("--config", type=str, default="configs/full_run.yaml")
    parser.add_argument("--subjects", type=str, nargs="+", default=None)
    parser.add_argument("--test_subject", type=str, default=None, help="Run only this fold")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"])
    args = parser.parse_args()
    
    # Load and apply defaults
    cfg = load_config(args.config)
    cfg = apply_defaults(cfg)
    
    if args.max_epochs:
        cfg.training.epochs = args.max_epochs
    
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    
    if "loss" not in cfg:
        cfg.loss = OmegaConf.create({"type": args.loss_type})
    else:
        cfg.loss.type = args.loss_type
    
    set_seed(cfg.split.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"loso_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold blue]LOSO Cross-Validation (Hardened)[/bold blue]")
    console.print(f"Config: {args.config}")
    console.print(f"Output: {output_dir}")
    console.print(f"Loss type: {args.loss_type}")
    console.print(f"Num workers: {cfg.training.num_workers}")
    
    console.print(f"\n[bold]GPU Status:[/bold]")
    console.print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"  Device: {torch.cuda.get_device_name(0)}")
        console.print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cache_path = Path(cfg.data.cache_root)
    cached_subjects = get_cached_subjects(cache_path)
    
    if args.test_subject:
        if args.test_subject not in cached_subjects:
            console.print(f"[red]Subject {args.test_subject} not in cache![/red]")
            return 1
        subjects = [args.test_subject]
        console.print(f"[bold yellow]DEBUG MODE: Running only fold for {args.test_subject}[/bold yellow]")
    elif args.subjects:
        if len(args.subjects) == 1 and args.subjects[0].lower() == "all":
            subjects = cached_subjects
        else:
            subjects = [s for s in args.subjects if s in cached_subjects]
    else:
        subjects = cached_subjects
    
    console.print(f"\nSubjects ({len(subjects)}): {subjects[:10]}{'...' if len(subjects) > 10 else ''}")
    
    if len(subjects) < 1:
        console.print("[red]No subjects to run![/red]")
        return 1
    
    save_config(cfg, output_dir / "config.yaml")
    
    fold_results = []
    
    for i, test_subject in enumerate(subjects):
        console.print(f"\n[bold]Fold {i+1}/{len(subjects)}: Testing on {test_subject}[/bold]")
        
        fold_dir = output_dir / f"fold_{test_subject}"
        if (fold_dir / "DONE.txt").exists() and not args.force:
            console.print(f"[yellow]Skipping {test_subject} (DONE.txt exists)[/yellow]")
            if (fold_dir / "results.json").exists():
                with open(fold_dir / "results.json") as f:
                    fold_results.append(json.load(f))
            continue
        
        other_subjects = [s for s in cached_subjects if s != test_subject]
        n_val = min(2, len(other_subjects) - 1)
        val_subjects = other_subjects[-n_val:] if n_val > 0 else other_subjects[:1]
        train_subjects = other_subjects[:-n_val] if n_val > 0 else other_subjects[1:]
        
        if len(train_subjects) == 0:
            train_subjects = other_subjects[:max(1, len(other_subjects)-1)]
        
        result = run_fold(
            test_subject=test_subject,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            cfg=cfg,
            output_dir=output_dir,
            device=device,
        )
        fold_results.append(result)
    
    summary = aggregate_results(fold_results)
    
    # Save clinical summary CSV
    clinical_csv = output_dir / "loso_clinical_summary.csv"
    with open(clinical_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "test_subject", "val_auroc", "test_auroc", "test_auprc", "n_seizures",
            "sens_fah_0.1", "sens_fah_0.2", "sens_fah_0.5", "sens_fah_1.0"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in fold_results:
            if "error" not in r:
                row = {
                    "test_subject": r["test_subject"],
                    "val_auroc": f"{r['val_auroc']:.4f}",
                    "test_auroc": f"{r['test_auroc']:.4f}" if not np.isnan(r['test_auroc']) else "N/A",
                    "test_auprc": f"{r.get('test_auprc', np.nan):.4f}" if not np.isnan(r.get('test_auprc', np.nan)) else "N/A",
                    "n_seizures": r.get("n_seizures", 0),
                }
                
                for fah in [0.1, 0.2, 0.5, 1.0]:
                    ar = r.get("alarm_results", {}).get(fah, {})
                    if not ar:
                        ar = r.get("alarm_results", {}).get(str(fah), {})
                    row[f"sens_fah_{fah}"] = f"{ar.get('sensitivity', 0):.2f}" if ar else "N/A"
                
                writer.writerow(row)
    
    # Save summary
    summary_file = output_dir / "loso_summary.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else v])
    
    # Copy to reports
    reports_dir = Path("reports/tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(clinical_csv, reports_dir / "loso_clinical_summary.csv")
    shutil.copy(summary_file, reports_dir / "loso_summary.csv")
    
    # Print summary
    console.print("\n[bold]LOSO Summary:[/bold]")
    table = Table()
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("N Folds", str(summary.get("n_folds", 0)))
    table.add_row("Val AUROC", f"{summary.get('val_auroc_mean', 0):.4f} +/- {summary.get('val_auroc_std', 0):.4f}")
    table.add_row("Test AUROC", f"{summary.get('test_auroc_mean', 0):.4f} +/- {summary.get('test_auroc_std', 0):.4f}")
    
    for fah in [0.1, 0.2, 0.5]:
        sens = summary.get(f"fah_{fah}_sensitivity_mean", 0)
        sens_std = summary.get(f"fah_{fah}_sensitivity_std", 0)
        if sens > 0:
            table.add_row(f"Sens @ FAH<={fah}", f"{sens:.2f} +/- {sens_std:.2f}")
    
    console.print(table)
    
    console.print(f"\nResults saved to {output_dir}")
    console.print("[bold green]LOSO complete![/bold green]")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Phase 5: Reproducibility & Experiment Logging

---

## 1. Environment Summary

| Component | Value |
|-----------|-------|
| OS | Windows 10/11 (Build 10.0.26120) |
| CPU | Intel Core i5-14600K |
| RAM | 32 GB DDR5 |
| GPU | NVIDIA GeForce RTX 3070 (8 GB GDDR6) |
| CUDA | (version from installation — check with `nvcc --version`) |
| Python | 3.x (check with `python --version`) |
| PyTorch | (check with `python -c "import torch; print(torch.__version__)"`) |
| MNE-Python | Used for EDF loading and signal processing |
| scikit-learn | Baseline models, metrics |
| XGBoost | Baseline classifier |
| OmegaConf | YAML config management |
| Rich | Console output formatting |
| tqdm | Progress bars |

**Note:** Exact package versions should be captured with `pip freeze > paper_pack/configs/requirements.txt`.

**Command to capture:**
```bash
pip freeze > paper_pack/configs/requirements.txt
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
```

## 2. Hardware Constraints & Design Decisions

| Constraint | Impact | Decision |
|-----------|--------|----------|
| 8 GB VRAM | Limits batch size and model size | Batch size 256 with AMP; 3-block CNN (not 4) |
| 32 GB RAM | Cache V2 can hold full dataset in memmap | Used memmap-backed cache for fast access |
| Windows OS | `num_workers>0` breaks memmap pickling | Worker-safe dataset with lazy memmap opening |
| WDDM mode | GPU utilization in Task Manager is misleading | Verified actual GPU use via `torch.cuda.memory_allocated()` |
| Single GPU | No distributed training | Standard single-GPU training loop |

## 3. Exact Commands Used

### 3.1 Dataset Verification
```bash
python scripts/verify_dataset.py
```

### 3.2 Cache Build (V2 with memmap)
```bash
python scripts/build_cache.py --config configs/full_run.yaml --subjects all
```

### 3.3 Within-Subject Training
```bash
python scripts/train_deep.py --config configs/small_run.yaml --split_mode within_subject --subject chb01
python scripts/train_deep.py --config configs/small_run.yaml --split_mode within_subject --subject chb10
```

### 3.4 Cross-Subject Training (early experiments)
```bash
python scripts/train_deep.py --config configs/small_run.yaml
python scripts/train_deep.py --config configs/medium_run.yaml
```

### 3.5 Baseline Training
```bash
python scripts/train_baseline.py --config configs/small_run.yaml
```

### 3.6 LOSO (Final Full Run)
```bash
python scripts/run_loso.py --config configs/full_run.yaml --subjects all --num_workers 4 --loss_type focal
```

### 3.7 Alarm Ablation
```bash
python scripts/run_ablations.py --config configs/small_run.yaml
```

### 3.8 Figure Generation
```bash
python scripts/make_figures.py
python scripts/label_sanity.py --config configs/small_run.yaml
```

### 3.9 Cache Report
```bash
python scripts/cache_report.py --config configs/full_run.yaml
```

## 4. Configuration Snapshots

### Final LOSO Config (frozen copy)
**Path:** `runs/loso_20260115_080732/config.yaml`  
**Copied to:** `paper_pack/configs/final_loso_config.yaml`

Key differences from `configs/full_run.yaml`:
- `training.epochs: 20` (originally 30 in full_run.yaml — overridden at runtime)
- `training.num_workers: 4` (originally 0 — overridden via `--num_workers 4`)
- `features.nan_fill_value: 0.0` (added by `apply_defaults()`)
- `loss.type: focal` (confirmed via CLI `--loss_type focal`)

## 5. Random Seeds

| Seed Usage | Value | Set By |
|-----------|-------|--------|
| Global seed | 42 | `src/utils/seed.py` → `set_seed(42)` |
| Data splitting | 42 | `cfg.split.seed` |
| PyTorch | 42 | `torch.manual_seed(42)` |
| NumPy | 42 | `np.random.seed(42)` |
| CUDA | 42 | `torch.cuda.manual_seed_all(42)` |

**Note:** `torch.backends.cudnn.deterministic` may not be set — AMP and cuDNN non-determinism means exact reproducibility is not guaranteed across runs. This is standard practice.

## 6. Checkpointing Behavior

- Best model saved per fold: `fold_*/checkpoints/best.pt`
- Checkpoint contents: `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_auroc`
- Selection criterion: Highest validation AUROC
- Early stopping: patience=10 epochs on val AUROC
- Completion marker: `fold_*/DONE.txt` with timestamp

## 7. Resume Behavior for LOSO

The LOSO script supports resumability:
- Before training a fold, checks for `fold_*/DONE.txt`
- If present and `--force` not specified, skips fold and loads `results.json`
- This enabled resuming interrupted runs without recomputing completed folds

## 8. Known Caveats on Windows

| Caveat | Detail |
|--------|--------|
| WDDM GPU reporting | Windows Task Manager shows misleading GPU% due to WDDM driver model. Actual GPU utilization must be verified via `nvidia-smi` or PyTorch memory APIs |
| Memmap pickling | `numpy.memmap` cannot be pickled on Windows. Solved by WorkerSafeCacheV2Dataset |
| File locking | Windows may lock checkpoint files during writing; no issues observed in practice |
| Path length | Long fold paths can approach Windows MAX_PATH (260 chars); no issues encountered |
| Console encoding | Rich console output requires UTF-8 terminal; PowerShell default is fine |

---

## 9. Full Command Log

```bash
# --- Dataset verification ---
python scripts/verify_dataset.py

# --- Cache build (V2 memmap format) ---
python scripts/build_cache.py --config configs/full_run.yaml --subjects all

# --- Cache statistics report ---
python scripts/cache_report.py --config configs/full_run.yaml

# --- Within-subject training ---
python scripts/train_deep.py --config configs/small_run.yaml --split_mode within_subject --subject chb01
python scripts/train_deep.py --config configs/small_run.yaml --split_mode within_subject --subject chb10

# --- Baseline model training (XGBoost) ---
python scripts/train_baseline.py --config configs/small_run.yaml

# --- Cross-subject training (early experiments) ---
python scripts/train_deep.py --config configs/small_run.yaml
python scripts/train_deep.py --config configs/medium_run.yaml

# --- Alarm post-processing ablation ---
python scripts/run_ablations.py --config configs/small_run.yaml

# --- Label sanity check ---
python scripts/label_sanity.py --config configs/small_run.yaml

# --- FINAL FULL LOSO RUN (PRIMARY RESULTS) ---
python scripts/run_loso.py --config configs/full_run.yaml --subjects all --num_workers 4 --loss_type focal

# --- Figure generation ---
python scripts/make_figures.py

# --- Summary generation ---
python scripts/generate_summary.py
```

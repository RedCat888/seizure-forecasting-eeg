# LOSO Pipeline Performance & Reliability Report

## Overview

This report documents the improvements made to the LOSO (Leave-One-Subject-Out) cross-validation pipeline for seizure forecasting, focusing on:
1. **Reliability hardening** - preventing config crashes and handling edge cases
2. **Windows multiprocessing** - enabling num_workers > 0 with memmap datasets
3. **Clinical evaluation** - adding FAH-targeted threshold tuning and alarm metrics

---

## Phase 1: Reliability Hardening

### Problem
- Missing config keys (e.g., `cfg.logging.save_every_n_epochs`) caused crashes
- NaN values in features led to training instability
- No label sanity checking (silent failures when val/test had no positives)

### Solution

**1. Config Defaults Module** (`src/utils/config_defaults.py`)
- Created comprehensive config schema with sensible defaults
- `apply_defaults(cfg)` function fills missing keys automatically
- `get_safe(cfg, "key.path", default)` for defensive access

**2. Label Sanity Logging**
- Per-fold logging of `n_pos`, `n_neg` for train/val/test splits
- Warnings when validation or test sets have no positive samples
- Saved to `fold_*/label_stats.json` for auditing

**3. NaN Handling**
- NaN values in features replaced with configurable `nan_fill_value` (default: 0.0)
- Applied at dataset `__getitem__` level for consistency
- `np.nan_to_num()` with posinf/neginf handling

---

## Phase 2: Windows Multiprocessing Fix

### Problem
- `numpy.memmap` objects cannot be pickled on Windows
- `num_workers > 0` with memmap dataset caused:
  ```
  OSError: [Errno 22] Invalid argument when serializing numpy.memmap object
  ```
- Training was CPU-bound with `num_workers=0` (~2.2 it/s)

### Solution

**Worker-Safe Memmap Dataset** (`src/data/cache_v2_worker_safe.py`)
- Dataset stores only **paths + shapes** (no open memmaps in `__init__`)
- Memmaps opened **lazily** in `__getitem__` via `_ensure_memmaps()`
- Custom `__getstate__` / `__setstate__` exclude memmaps during pickling
- Each DataLoader worker opens its own memmaps independently

**DataLoader Configuration:**
```python
DataLoader(
    dataset,
    num_workers=4,           # Now works on Windows!
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2,
    worker_init_fn=worker_init_fn,
)
```

### Performance Results

| Metric | Before (num_workers=0) | After (num_workers=4) | Improvement |
|--------|------------------------|----------------------|-------------|
| data_ms | 390-520ms | 24-44ms | **12x faster** |
| step_ms | 480-640ms | 112-130ms | **4-5x faster** |
| Throughput | ~2.2 it/s | ~8-9 it/s | **4x faster** |

---

## Phase 3: Clinical Evaluation

### Problem
- Window-level metrics (AUROC/AUPRC) don't reflect clinical utility
- Fixed threshold (0.5) leads to uncontrolled false alarm rates
- No alarm-level evaluation (FAH, sensitivity, warning time)

### Solution

**Threshold Tuning for FAH Targets**
- Tune threshold on validation set to achieve target FAH
- FAH targets: 0.1, 0.2, 0.5, 1.0 false alarms per hour
- Maximize sensitivity while respecting FAH constraint

**Alarm Post-Processing**
- EMA smoothing (configurable alpha)
- Persistence filter (K consecutive windows)
- Hysteresis (trigger/reset thresholds)
- Refractory period (default 20 minutes)

**Clinical Metrics Output**
- `loso_clinical_summary.csv` with per-fold:
  - test_subject, AUROC, AUPRC, n_seizures
  - sensitivity @ FAH targets (0.1, 0.2, 0.5, 1.0)
  - Achieved FAH, warning time

### Example Output
```
Fold Results (chb01):
  Val AUROC: 0.7897
  Test AUROC: 0.6626
  Test AUPRC: 0.1600
  N Seizures: 4
  FAH<=0.1: sens=0.00, fah=0.09, thresh=0.762
  FAH<=0.2: sens=0.25, fah=0.19, thresh=0.703
  FAH<=0.5: sens=0.25, fah=0.47, thresh=0.634
  FAH<=1.0: sens=0.25, fah=0.94, thresh=0.554
```

---

## Files Changed

### New Files
- `src/utils/config_defaults.py` - Config schema with defaults
- `src/data/cache_v2_worker_safe.py` - Worker-safe memmap dataset

### Modified Files
- `scripts/run_loso.py` - Complete rewrite with:
  - Performance timers
  - GPU logging
  - Label sanity checks
  - Clinical evaluation
  - Resumability (DONE.txt markers)

---

## Usage

### One-Fold Debug Mode
```bash
python scripts/run_loso.py --test_subject chb01 --max_epochs 2 --num_workers 4
```

### Full LOSO
```bash
python scripts/run_loso.py --subjects all --num_workers 4 --loss_type focal
```

### Performance Logging
Performance metrics logged to `reports/tables/perf_log_fold_*.csv`:
- `data_ms` - Time waiting for next batch
- `h2d_ms` - Host-to-device transfer time
- `fwd_ms` - Forward pass time
- `bwd_ms` - Backward pass + optimizer step time

---

## Baseline Results (Previous LOSO)

From the previous full LOSO run (24 folds):

| Metric | Value |
|--------|-------|
| Val AUROC | 0.8163 ± 0.0126 |
| Test AUROC | 0.5684 ± 0.1835 |
| Val AUPRC | 0.3591 ± 0.0688 |
| Test AUPRC | 0.1079 ± 0.1160 |

---

## Next Steps

1. Run full LOSO with improved pipeline (`num_workers=4`)
2. Compare clinical metrics across subjects
3. Identify subjects with poor performance for targeted analysis
4. Tune alarm post-processing parameters

---

*Report generated: 2026-01-15*

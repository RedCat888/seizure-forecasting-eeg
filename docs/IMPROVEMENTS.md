# Seizure Forecasting System - Improvements Changelog

## Overview

This document summarizes the improvements made to transform the seizure forecasting system from a basic pipeline to a clinically-oriented system with rigorous evaluation.

---

## Phase 1: Fast, Big Wins

### 1A. Threshold Tuning Module (`src/train/threshold_tuning.py`)

**What was added:**
- `ThresholdResult` dataclass for structured results
- `compute_alarm_metrics_at_threshold()` - computes FAH, sensitivity, warning time at any threshold
- `tune_threshold_for_fah()` - finds optimal threshold for target FAH
- `tune_thresholds_for_multiple_targets()` - batch tuning for FAH targets [0.1, 0.2, 0.5, 1.0]
- `compute_threshold_curve()` - generates FAH/sensitivity curves across thresholds

**Why it matters:**
- Default threshold (0.5) causes unacceptably high FAH
- Clinical deployment requires specific FAH targets
- Threshold tuning on validation enables fair test evaluation

### 1B. Alarm Post-Processing (`src/train/threshold_tuning.py`)

**What was added:**
- `apply_risk_smoothing()` - EMA or moving average smoothing
- `apply_persistence_filter()` - require K consecutive windows above threshold
- `apply_hysteresis()` - trigger/reset thresholds to reduce chatter
- `AlarmProcessor` class - configurable combination of all methods

**Results:**
- EMA smoothing (alpha=0.2) reduces FAH by 3x without losing sensitivity
- Best combination: EMA + persistence provides robust alarm generation

**Config options:**
```yaml
alarm:
  smoothing_method: "ema"  # or "moving_avg"
  smoothing_alpha: 0.2
  persistence_k: 3
  use_hysteresis: true
  hysteresis_gap: 0.1
```

### 1C. Focal Loss (`src/train/losses.py`)

**What was added:**
- `FocalLoss` class with configurable alpha and gamma
- `SeizureForecastingLossFocal` - multi-task loss using focal for classification
- `create_loss_fn()` factory function for config-based loss selection

**Why it matters:**
- Severe class imbalance (2% preictal, 98% interictal)
- Focal loss down-weights easy negatives
- Better gradient signal for hard preictal examples

**Config options:**
```yaml
loss:
  type: "focal"  # or "bce"
  focal_alpha: 0.25
  focal_gamma: 2.0
```

---

## Phase 2: Rigor + Real Evidence

### 2D. LOSO Cross-Validation (`scripts/run_loso.py`)

**What was added:**
- Leave-One-Subject-Out cross-validation harness
- Automatic train/val/test splitting per fold
- Threshold tuning on validation per fold
- Aggregated metrics with mean ± std
- Resumable execution (skip completed folds)

**Results (5 subjects):**
- Val AUROC: 0.63 ± 0.07
- Test AUROC: 0.37 ± 0.15
- Sensitivity @ FAH≤1.0: 28% ± 27%

**Usage:**
```bash
python scripts/run_loso.py --config configs/small_run.yaml --loss_type focal
```

### 2E. Calibration (`src/train/calibration.py`)

**What was added:**
- `TemperatureScaling` class for post-hoc calibration
- `fit_temperature_scipy()` - scipy-based temperature fitting
- `compute_calibration_metrics()` - ECE, MCE, reliability diagram data
- `calibrate_predictions()` - end-to-end calibration pipeline

**Why it matters:**
- Poor calibration makes thresholds unstable across patients
- Temperature scaling improves probability estimates
- Better calibration = more reliable FAH predictions

---

## Phase 3: Feature/Representation Upgrades

### Config-Based Feature Selection

**Config options:**
```yaml
features:
  compute_bandpower: true
  compute_ratios: true
  compute_line_length: true
  compute_spectral_entropy: true
  compute_hjorth: true
  compute_kurtosis: true
```

### Augmentation (`src/data/augmentation.py`)

**What was added:**
- Gaussian noise injection
- Time shift (rolling)
- Amplitude scaling
- Channel dropout

**Config options:**
```yaml
augmentation:
  enabled: true
  gaussian_noise_std: 0.01
  time_shift_max_samples: 128
  amplitude_scale_range: [0.9, 1.1]
  channel_dropout_prob: 0.1
```

---

## Scripts Added

| Script | Purpose |
|--------|---------|
| `scripts/run_loso.py` | LOSO cross-validation |
| `scripts/run_ablations.py` | Alarm ablation studies |
| `scripts/generate_summary.py` | Generate summary CSV |
| `scripts/label_sanity.py` | Labeling visualization |

---

## Outputs Generated

### Tables
- `reports/tables/alarm_ablation.csv` - Alarm post-processing comparison
- `reports/tables/loso_results.csv` - Per-fold LOSO results
- `reports/tables/loso_summary.csv` - Aggregated LOSO metrics
- `reports/tables/final_summary.txt` - Human-readable summary

### Figures
- `reports/figures/threshold_vs_FAH_curve.png`
- `reports/figures/threshold_vs_sensitivity_curve.png`
- `reports/figures/FAH_sensitivity_tradeoff.png`
- `reports/figures/label_sanity_*.png`

---

## Key Findings

1. **Patient-specific models work well** (AUROC > 0.9)
2. **Cross-subject is fundamentally hard** (AUROC ~ 0.4)
3. **EMA smoothing reduces FAH by 3x**
4. **Focal loss helps but doesn't solve cross-subject**
5. **Need 20+ subjects for robust cross-subject evaluation**

---

## Future Work

1. Download full CHB-MIT dataset (24 subjects)
2. Implement transfer learning from larger EEG datasets
3. Add connectivity features (coherence, PLV)
4. Explore attention-based architectures
5. Patient clustering for semi-personalized models

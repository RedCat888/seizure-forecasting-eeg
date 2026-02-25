# Phase 2: Methods — Technical Architecture & Pipeline

**Provenance:** All details extracted from source code, config files, and documentation in this repository. File paths cited inline.

---

## 1. Problem Formulation

**Binary seizure forecasting / preictal risk prediction.** Given a 30-second EEG window, predict whether a seizure will occur within the next 30 minutes (preictal state) vs. normal baseline (interictal state). The model additionally outputs a continuous soft risk score via a regression head, trained with an exponential proximity-weighted target.

## 2. Model Inputs

Each sample consists of:
- **Raw EEG:** [18 channels × 7,680 samples] (30s at 256 Hz) → transformed to spectrogram inside the model
- **Handcrafted features:** 30-dimensional vector (bandpower, ratios, line length, spectral entropy, Hjorth, statistics)

## 3. Network Architecture: FusionNet

**Source:** `src/models/fusion_net.py`

### 3.1 Overview

FusionNet is a dual-branch architecture:
1. **CNN Encoder** — processes spectrogram (computed on GPU)
2. **Feature MLP** — processes handcrafted features
3. **Fusion Head** — concatenates embeddings → classification + regression

### 3.2 CNN Encoder (Spectrogram Branch)

| Component | Detail |
|-----------|--------|
| Input | Spectrogram [B, 18, 129, 117] |
| Conv blocks | 3 blocks: {32, 64, 128} channels |
| Each block | Conv2d(k=3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2) → Dropout2d(0.15) |
| Adaptive pool | AdaptiveAvgPool2d(4, 4) |
| FC projection | Flatten → Linear(128×4×4, 128) → ReLU → Dropout(0.3) |
| Output | Embedding [B, 128] |

**Note:** The frozen config uses `cnn_channels: [32, 64, 128]` (3 blocks, not 4 as in default config).

### 3.3 Feature MLP Branch

| Component | Detail |
|-----------|--------|
| Input | Features [B, 30] |
| Hidden layers | [256, 128] (from config `mlp_hidden`) |
| Each layer | Linear → BatchNorm1d → ReLU → Dropout(0.3) |
| Output | Embedding [B, 64] (default `feature_embed_dim`) |

### 3.4 Fusion Head

| Component | Detail |
|-----------|--------|
| Input | Concatenation of CNN embed [128] + Feature embed [64] = [192] |
| Shared MLP | [128, 64] hidden dims with BN + ReLU + Dropout(0.3) |
| Classification head | Linear(64, 1) → output logit |
| Regression head | Linear(64, 1) → Sigmoid → soft risk ∈ [0,1] |

### 3.5 Parameter Count

**MISSING:** Exact parameter count was not logged. Estimate based on architecture:
- CNN Encoder: ~600K parameters
- Feature MLP: ~45K parameters
- Fusion Head: ~35K parameters
- **Total: ~680K parameters** (estimated)

To verify: `sum(p.numel() for p in model.parameters())`

## 4. Baseline Models

**Source:** `src/models/baseline.py`

| Model | Implementation | Key Hyperparameters |
|-------|---------------|---------------------|
| XGBoost | `xgboost.XGBClassifier` | n_estimators=200, max_depth=6, lr=0.1 |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | max_iter=1000, class_weight="balanced" |
| MLP | `sklearn.neural_network.MLPClassifier` | hidden=(128,64), early_stopping=True |

All baselines use StandardScaler preprocessing and operate on 30-dimensional handcrafted features only.

## 5. Loss Functions

**Source:** `src/train/losses.py`

### 5.1 Multi-Task Loss (Final Run)

\[
\mathcal{L} = \mathcal{L}_{\text{focal}}(\hat{y}, y) + \lambda \cdot \text{MSE}(\hat{r}, r_{\text{soft}})
\]

- **Classification:** Focal Loss with α=0.25, γ=2.0
- **Regression:** MSE on soft risk score, λ_soft=0.5
- **Proximity weighting:** Preictal samples weighted by \( w = 1 + (1 - t/t_{\max}) \), range [1, 2]

### 5.2 Focal Loss

\[
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
\]

Down-weights easy negatives (vast interictal majority), amplifies gradient for hard preictal examples.

## 6. Optimizer & Training Configuration

| Parameter | Value (Final Run) |
|-----------|-------------------|
| Optimizer | AdamW |
| Learning rate | 5×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| Scheduler | Cosine annealing (warmup: 2 epochs) |
| Epochs | 20 (with early stopping) |
| Early stopping patience | 10 epochs on val AUROC |
| Batch size | 256 |
| Gradient clipping | 1.0 |
| Mixed precision (AMP) | Enabled (torch.amp) |
| Gradient accumulation | 1 (no accumulation) |
| Random seed | 42 |

**Source:** `runs/loso_20260115_080732/config.yaml`

## 7. Data Augmentation (Training Only)

| Augmentation | Parameter | Value |
|-------------|-----------|-------|
| Gaussian noise | std | 0.01 |
| Time shift | max_samples | 64 (~0.25s at 256 Hz) |
| Amplitude scaling | range | [0.95, 1.05] |
| Channel dropout | probability | 0.05 |

**Source:** `configs/full_run.yaml`, augmentation section.

## 8. Threshold Tuning & Alarm Post-Processing

**Source:** `src/train/threshold_tuning.py`

### 8.1 Threshold Tuning

- Sweep 100 thresholds linearly from 0.01 to 0.99
- For each FAH target (0.1, 0.2, 0.5, 1.0), find threshold maximizing sensitivity subject to FAH ≤ target
- Threshold tuned on **validation set** predictions per fold; applied to test set

### 8.2 EMA Smoothing

\[
s_t = \alpha \cdot r_t + (1 - \alpha) \cdot s_{t-1}
\]

With α=0.2 (lower = more smoothing).

### 8.3 Persistence Filter

Require K=3 consecutive windows above threshold before raising alarm.

### 8.4 Hysteresis

- Trigger threshold: θ
- Reset threshold: θ − 0.1
- Prevents alarm chatter near decision boundary

### 8.5 Refractory Period

Minimum 1200 seconds (20 minutes) between consecutive alarms.

## 9. Calibration

**Source:** `src/train/calibration.py`

- Temperature scaling implemented (TemperatureScaling class)
- Fits single scalar T on validation logits via LBFGS or scipy
- ECE/MCE computation implemented
- **Status:** Module exists but was NOT applied in the final LOSO run. Calibration metrics are not in the final results.

## 10. Evaluation Modes

| Mode | Splitting Strategy | Subjects |
|------|-------------------|----------|
| Within-subject | Chronological 70/15/15 per patient | chb01, chb10 tested |
| Cross-subject | Fixed train/val/test split | small_run, medium_run configs |
| **LOSO** | Train on N−1, val on 2 held-out, test on 1 | All 24 subjects (final run) |

### LOSO Details

For each fold testing on subject S:
- **Train:** 22 subjects (all except S and 2 val subjects)
- **Val:** 2 subjects (last 2 alphabetically from remaining)
- **Test:** Subject S

## 11. Hardware Optimization & Engineering

**Source:** `docs/PERFORMANCE_RELIABILITY_REPORT.md`

### 11.1 Data Pipeline Bottleneck

| Metric | Before (workers=0) | After (workers=4) |
|--------|--------------------|--------------------|
| data_ms per step | 390–520 ms | 24–44 ms |
| step_ms total | 480–640 ms | 112–130 ms |
| Throughput | ~2.2 it/s | ~8–9 it/s |
| Improvement | — | **4–5× faster** |

### 11.2 Key Engineering Decisions

| Issue | Solution | File |
|-------|----------|------|
| Memmap can't pickle on Windows | Worker-safe dataset with lazy memmap opening | `src/data/cache_v2_worker_safe.py` |
| NaN in handcrafted features | `nan_fill_value=0.0` at `__getitem__` level | `src/data/cache_v2_worker_safe.py` |
| Config key crashes | `apply_defaults()` + `get_safe()` defensive accessors | `src/utils/config_defaults.py` |
| HDF5 I/O bottleneck (V1) | Replaced with memmap Cache V2 | `scripts/build_cache.py` |
| AMP suspected NaN source | Root cause was NaN features, not AMP | Documented in config comment |
| WDDM GPU reporting | Task Manager GPU% misleading under WDDM mode | Known Windows caveat |

### 11.3 Engineering Failure Analysis

1. **NaN Features:** Welch PSD computation on zero-padded missing channels produced NaN values. Fixed by `np.nan_to_num()` at dataset load time.
2. **Config Contract Mismatch:** The original `default.yaml` used different key names (`bandpass_low`) vs. `full_run.yaml` (`lowcut`). The `config_defaults.py` module unifies both schemas.
3. **Windows Memmap Multiprocessing:** `numpy.memmap` objects cannot be serialized via `pickle` on Windows. The `WorkerSafeCacheV2Dataset` stores only paths/shapes and opens memmaps lazily per worker.

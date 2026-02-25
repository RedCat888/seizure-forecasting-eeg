# Phase 1: Dataset & Cache Facts

**Provenance:** Extracted from `reports/tables/window_counts_full.csv`, `configs/full_run.yaml`, `runs/loso_20260115_080732/config.yaml`, `src/signal/preprocess.py`, `src/features/handcrafted.py`, `runs/loso_20260115_080732/fold_*/results.json`.

---

## 1. Dataset Identity

| Field | Value |
|-------|-------|
| Dataset | CHB-MIT Scalp EEG Database |
| Source | PhysioNet (physionet.org/content/chbmit/1.0.0/) |
| Version | 1.0.0 |
| Format | European Data Format (EDF) |
| Modality | Scalp EEG, bipolar montage |

## 2. Subject & Recording Summary

| Metric | Value |
|--------|-------|
| Number of subjects | 24 |
| Subject IDs | chb01–chb24 (note: chb12 is the same patient as chb01 at different age) |
| Recording type | Continuous ambulatory EEG |
| Total recordings | ~686 EDF files across all subjects (varies per subject) |

### Per-Subject Seizure Counts (from final LOSO `results.json`)

| Subject | Test Seizures | Subject | Test Seizures |
|---------|---------------|---------|---------------|
| chb01 | 4 | chb13 | 1 |
| chb02 | 2 | chb14 | 3 |
| chb03 | 2 | chb15 | 10 |
| chb04 | 2 | chb16 | 2 |
| chb05 | 2 | chb17 | 2 |
| chb06 | 0 | chb18 | 2 |
| chb07 | 3 | chb19 | 1 |
| chb08 | 1 | chb20 | 2 |
| chb09 | 3 | chb21 | 1 |
| chb10 | 1 | chb22 | 2 |
| chb11 | 1 | chb23 | 2 |
| chb12 | 6 | chb24 | 4 |

**Note:** chb06 had 0 seizure intervals detected in its test partition, likely due to labeling/exclusion windows.

## 3. Preprocessing Pipeline

| Step | Implementation | Detail |
|------|---------------|--------|
| Loading | `mne.io.read_raw_edf()` | MNE-Python |
| Channel selection | `src/signal/preprocess.py` | 18 canonical bipolar channels enforced |
| Resampling | MNE `.resample()` | Target: 256 Hz |
| Bandpass filter | FIR (firwin) | 0.5–50.0 Hz (full_run) / 0.5–45.0 Hz (default) |
| Notch filter | FIR (firwin) | 60 Hz (US line noise) |
| Re-referencing | Common average | `raw.set_eeg_reference("average")` |
| Artifact rejection | Amplitude threshold | 500 µV peak-to-peak per channel |
| Channel normalization | Canonical ordering | Zero-fill missing channels |

### Canonical 18-Channel Montage

```
FP1-F7, F7-T7, T7-P7, P7-O1,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2,
FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ, CZ-PZ
```

Source: `src/signal/preprocess.py`, lines 40–46 (`CANONICAL_CHANNELS`).

## 4. Windowing & Labeling

| Parameter | Value (full_run.yaml) |
|-----------|-----------------------|
| Window length | 30 seconds |
| Window stride | 15 seconds (50% overlap) |
| Preictal horizon | 30 minutes before seizure onset |
| Exclusion gap | 300 seconds (5 min) before onset |
| Postictal exclusion | 30 minutes after seizure offset |
| Interictal buffer | 60 minutes from any seizure |
| Soft risk τ | 1800 seconds (30 min decay) |

### Label Definitions

- **Preictal (y=1):** Window ends within [onset − 30min, onset − 5min]
- **Interictal (y=0):** Window center >60 min from any seizure onset/offset
- **Excluded:** Ictal, postictal, gap, or buffer-zone windows (not used in training)

Source: `docs/LABELING_SCHEMA.md`, `configs/full_run.yaml`.

## 5. Cache Format & Statistics

| Metric | Value |
|--------|-------|
| Cache version | V2 (memmap-backed) |
| Cache root | `data/chbmit_cache_v2/` |
| Storage format | NumPy memmap (`.npy` files) |
| Data dtype | float32 |
| Total windows (full dataset) | 66,695 |
| Preictal windows | 3,107 (4.66%) |
| Interictal windows | 63,588 (95.34%) |
| Excluded windows | 27,847 (not included in above counts) |
| Number of subjects cached | 24 |
| Class imbalance ratio | ~1:20.5 (preictal:interictal) |

Source: `reports/tables/window_counts_full.csv`.

### Window Shape per Sample

| Tensor | Shape | Description |
|--------|-------|-------------|
| Raw EEG data | [18, 7680] | 18 channels × (30s × 256Hz) |
| Handcrafted features | [30] | 30-dimensional feature vector |
| y_cls | scalar | Binary label (0 or 1) |
| y_tte | scalar | Time-to-event in seconds (−1 for interictal) |
| y_soft | scalar | Continuous soft risk ∈ [0, 1] |

## 6. Handcrafted Feature Set (30 features)

Features are computed per channel, then aggregated as (mean, std) across 18 channels:

| Feature Group | Per-Channel Features | Aggregated Count |
|---------------|---------------------|-----------------|
| Bandpower (5 bands: δ, θ, α, β, γ) | 5 | 10 (mean+std) |
| Band ratios (θ/α, β/α, slow/fast) | 3 | 6 (mean+std) |
| Line length | 1 | 2 (mean+std) |
| Spectral entropy | 1 | 2 (mean+std) |
| Hjorth mobility | 1 | 2 (mean+std) |
| Hjorth complexity | 1 | 2 (mean+std) |
| Kurtosis | 1 | 2 (mean+std) |
| Skewness | 1 | 2 (mean+std) |
| Variance | 1 | 2 (mean+std) |
| **Total** | | **30** |

Source: `src/features/handcrafted.py` (`extract_features()`, `get_feature_names()`).

## 7. Spectrogram Parameters

| Parameter | Value |
|-----------|-------|
| n_fft | 256 |
| hop_length | 64 |
| win_length | 256 |
| Window function | Hann |
| Transform | log(1.0 + magnitude) |
| Computed by | PyTorch STFT (on GPU, inside model forward pass) |
| Output shape per window | [18, 129, T'] where T' = (7680 − 256) / 64 + 1 = 117 |

Source: `src/signal/spectrograms.py` (`SpectrogramTransform`).

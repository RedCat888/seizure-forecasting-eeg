# Seizure Forecasting System

A complete, reproducible end-to-end seizure **forecasting** (prediction) system using scalp EEG from the CHB-MIT dataset. This system predicts whether a seizure will occur soon (within the next 10 minutes) by detecting the **preictal** state.

## Key Features

- **True Forecasting**: Predicts seizures *before* they occur, not just detection during ictal phase
- **Multi-scale Features**: Combines handcrafted EEG features with deep learning on spectrograms
- **Clinically Meaningful Metrics**: False Alarm Rate, Seizure Sensitivity, Time-to-Warning
- **Patient-wise Splitting**: No data leakage between train/val/test sets
- **GPU Optimized**: Mixed precision (AMP) training for RTX 3070 (8GB VRAM)
- **Reproducible**: Deterministic seeds, cached preprocessing, version-pinned dependencies

## Hardware Requirements

- **CPU**: Intel i5-14600K or equivalent
- **RAM**: 32GB recommended
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) or better
- **Storage**: ~50GB for full dataset + cache

## Quickstart

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -e .
```

### 2. Download Dataset

See [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md) for detailed instructions.

```bash
# Option: AWS S3 (fastest)
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ data/chbmit_raw
```

### 3. Verify Dataset

```bash
python scripts/verify_dataset.py --data_root data/chbmit_raw
```

### 4. Build Cache (Preprocess + Window + Labels + Features)

```bash
# Small run (3 subjects for testing)
python scripts/build_cache.py --data_root data/chbmit_raw --out_root data/chbmit_cache --subjects chb01 chb02 chb03

# Full run (all subjects)
python scripts/build_cache.py --data_root data/chbmit_raw --out_root data/chbmit_cache
```

### 5. Train Baseline Model (Handcrafted Features)

```bash
python scripts/train_baseline.py --config configs/small_run.yaml
```

### 6. Train Deep Model (CNN + Feature Fusion)

```bash
python scripts/train_deep.py --config configs/small_run.yaml
```

### 7. Evaluate Model

```bash
python scripts/eval.py --config configs/small_run.yaml --checkpoint runs/deep_model/best.pt
python scripts/make_figures.py --run_dir runs/deep_model/
```

### 8. Launch Demo App

```bash
streamlit run app/app.py
```

## Project Structure

```
├── README.md
├── pyproject.toml          # Dependencies with pinned versions
├── configs/
│   ├── default.yaml        # Full training config
│   └── small_run.yaml      # Quick iteration config (3-5 subjects)
├── docs/
│   ├── DATA_DOWNLOAD.md    # Dataset download instructions
│   ├── LABELING_SCHEMA.md  # Preictal/interictal labeling logic
│   └── EXPERIMENT_SUMMARY.md
├── reports/
│   ├── figures/            # Publication-style visualizations
│   └── tables/             # Metrics CSV files
├── runs/                   # Checkpoints and training logs (gitignored)
├── scripts/
│   ├── verify_dataset.py   # Validate dataset integrity
│   ├── build_cache.py      # Preprocess and cache windows
│   ├── train_baseline.py   # Train handcrafted feature model
│   ├── train_deep.py       # Train CNN fusion model
│   ├── eval.py             # Evaluate with alarm metrics
│   └── make_figures.py     # Generate publication figures
├── app/
│   └── app.py              # Streamlit demo application
└── src/
    ├── chbmit/             # CHB-MIT dataset parsing
    ├── signal/             # Preprocessing and spectrograms
    ├── features/           # Handcrafted EEG features
    ├── models/             # Neural network architectures
    ├── train/              # Training loops, losses, metrics
    └── utils/              # Config, logging, utilities
```

## Model Architecture

The deep model (`FusionNet`) combines:
1. **Spectrogram CNN**: 4 conv blocks processing log-magnitude STFT
2. **Feature MLP**: 2-layer network for handcrafted features
3. **Fusion Head**: Concatenated embeddings → classification + soft risk regression

## Labeling Schema

See [docs/LABELING_SCHEMA.md](docs/LABELING_SCHEMA.md) for complete details.

**Key Definitions:**
- **Preictal**: Windows ending 30s to 10min before seizure onset
- **Interictal**: Windows at least 30min away from any seizure
- **Excluded**: Ictal periods, postictal recovery (10min), and gap periods

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUROC | Area under ROC curve (window-level) |
| AUPRC | Area under precision-recall curve |
| FAH | False alarms per hour |
| Sensitivity | Fraction of seizures with timely warning |
| Time-to-Warning | Lead time before seizure onset |

## Results

Leave-one-subject-out (LOSO) cross-validation across 24 CHB-MIT subjects. See [docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md) for full details.

## License

MIT License — see [LICENSE](LICENSE) for details.

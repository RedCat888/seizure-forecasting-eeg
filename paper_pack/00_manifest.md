# Paper Pack Manifest — Sources of Truth

**Generated:** 2026-02-25  
**Repository root:** `c:\Users\ansar\Downloads\isp`

---

## 1. Repository Structure (Key Directories & Files)

```
isp/
├── configs/
│   ├── default.yaml            # Original default config
│   ├── small_run.yaml          # 5-subject debug config
│   ├── medium_run.yaml         # Medium experiment config
│   └── full_run.yaml           # *** FINAL full LOSO config ***
├── src/
│   ├── chbmit/                 # CHB-MIT parsing (indexing.py, parse_summary.py)
│   ├── features/               # Handcrafted feature extraction (handcrafted.py)
│   ├── models/                 # FusionNet (fusion_net.py), baselines (baseline.py)
│   ├── signal/                 # Preprocessing (preprocess.py), spectrograms (spectrograms.py)
│   ├── train/                  # Training loops, losses, metrics, calibration, threshold_tuning, alarm_eval
│   └── utils/                  # Config, seed, paths, logging, config_defaults
├── scripts/
│   ├── run_loso.py             # *** LOSO cross-validation harness ***
│   ├── run_ablations.py        # Alarm post-processing ablation
│   ├── train_deep.py           # Deep model training
│   ├── train_baseline.py       # Baseline model training
│   ├── build_cache.py          # Cache V2 builder
│   ├── cache_report.py         # Cache statistics
│   ├── make_figures.py         # Figure generation
│   ├── generate_summary.py     # Summary CSV generation
│   ├── label_sanity.py         # Label verification plots
│   ├── verify_dataset.py       # Dataset download verification
│   ├── eval.py / full_eval.py  # Evaluation scripts
│   └── smoke_test.py           # Quick pipeline test
├── reports/
│   ├── tables/                 # All result CSVs (33 files)
│   └── figures/                # Figure outputs (currently empty .gitkeep only)
├── runs/                       # 37 run directories (see below)
├── docs/                       # Documentation (6 markdown files)
├── paper/                      # seizure_forecasting_paper.md (draft)
├── app/                        # Demo Streamlit app
├── data/                       # Raw EDF + Cache V2 (not tracked)
└── setup.py
```

## 2. Run Directories

### Final Full LOSO Run (PRIMARY)

| Field | Value |
|-------|-------|
| **Directory** | `runs/loso_20260115_080732/` |
| **Started** | 2026-01-15 08:07:32 |
| **Completed** | 2026-01-15 10:56:02 |
| **Duration** | ~2 hours 49 minutes |
| **N Folds** | 24 (all completed — DONE.txt in every fold) |
| **Config** | `configs/full_run.yaml` (copied to `runs/loso_20260115_080732/config.yaml`) |
| **Loss** | Focal (alpha=0.25, gamma=2.0) |
| **Clinical Metrics** | Yes — `loso_clinical_summary.csv` with FAH-targeted evaluation |
| **Epochs/fold** | 30 (with early stopping, patience=10) |
| **Batch size** | 256 |
| **AMP** | Enabled |
| **num_workers** | 0 (Windows memmap constraint) |

**Justification as final run:** This is the latest LOSO run using `full_run.yaml` on all 24 CHB-MIT subjects with clinical evaluation (FAH-targeted threshold tuning per fold). All 24 folds have `DONE.txt` markers confirming completion. The run includes alarm post-processing and per-fold sensitivity at FAH targets.

### Earlier Full LOSO Run (SECONDARY — no clinical metrics)

| Field | Value |
|-------|-------|
| **Directory** | `runs/loso_20260114_232245/` |
| **Started** | 2026-01-14 23:22:45 |
| **Completed** | 2026-01-15 07:01:30 |
| **Duration** | ~7 hours 39 minutes |
| **N Folds** | 24 (all completed) |
| **Clinical Metrics** | No — only window-level AUROC/AUPRC |

**Note:** This earlier run took ~3x longer (no num_workers optimization) and lacks clinical alarm metrics. Its results are in `runs/loso_20260114_232245/loso_results.csv` and `loso_summary.csv`. It serves as a secondary validation of the LOSO results.

### Within-Subject Runs

| Run Directory | Subject | Notes |
|---------------|---------|-------|
| `deep_within_subject_20260114_034117_chb01` | chb01 | Val AUROC 0.91, Test AUROC 0.95 |
| `deep_within_subject_20260114_073245_chb10` | chb10 | Val AUROC 0.98, Test AUROC 0.50 (no test seizures) |

### Cross-Subject Runs (early experiments)

| Run | Train | Val | Test | Notes |
|-----|-------|-----|------|-------|
| `small_run_*` | chb01,02,03 | chb04 | chb05 | 5-subject debug |
| `medium_run_*` | chb01,02,03 | chb05 | chb10 | Medium scale |

### Ablation Runs

Alarm ablation results are in `reports/tables/alarm_ablation.csv` (generated from `scripts/run_ablations.py`).

## 3. Config Files Used for Final Run

- **Primary:** `configs/full_run.yaml`
- **Frozen copy:** `runs/loso_20260115_080732/config.yaml`

## 4. Scripts Used

| Script | Purpose |
|--------|---------|
| `scripts/run_loso.py` | Full LOSO cross-validation with clinical evaluation |
| `scripts/build_cache.py` | Build Cache V2 from raw EDFs |
| `scripts/run_ablations.py` | Alarm post-processing ablation study |
| `scripts/verify_dataset.py` | Dataset integrity verification |
| `scripts/make_figures.py` | Figure generation |
| `scripts/label_sanity.py` | Label verification |
| `scripts/cache_report.py` | Cache statistics |

## 5. All Available Tables (CSV)

| File | Description | Source |
|------|-------------|--------|
| `reports/tables/loso_clinical_summary.csv` | Per-fold LOSO with FAH metrics (24 folds) | Final LOSO run |
| `reports/tables/loso_summary.csv` | Aggregated LOSO metrics (mean±std) | Final LOSO run |
| `reports/tables/loso_results.csv` | Per-fold AUROC/AUPRC (24 folds) | Earlier LOSO run |
| `reports/tables/alarm_ablation.csv` | Alarm post-processing comparison (8 configs × 4 FAH targets) | Ablation script |
| `reports/tables/baseline_metrics.csv` | XGBoost baseline (train/val/test) | Baseline training |
| `reports/tables/summary_results.csv` | Within-subject + cross-subject results | Summary script |
| `reports/tables/window_counts_full.csv` | Full dataset window statistics | Cache report |
| `reports/tables/window_counts.csv` | 5-subject subset window counts | Cache report |
| `reports/tables/channel_log.csv` | Per-file channel information | Preprocessing |
| `reports/tables/perf_log_fold_*.csv` | Per-fold performance profiling (24 files) | LOSO run |

## 6. Available Figures

**Current state:** `reports/figures/` contains only `.gitkeep`. No PNG figures are currently present in the repository. Previous figures were likely generated but not committed/retained.

**MISSING figures that need regeneration:**
- Dataset composition (subjects/seizures/class distribution)
- Labeling timeline schematic
- Pipeline diagram
- ROC/PR curves
- Training curves
- Alarm tradeoff plots
- LOSO per-fold performance bar plot
- Calibration reliability diagram
- Performance profiling plot

## 7. Missing Artifacts Needed for Paper

| Artifact | Status | How to Generate |
|----------|--------|-----------------|
| All figures | MISSING | `python scripts/make_figures.py` or custom scripts |
| Per-fold epoch metrics consolidated | Available in fold dirs | Parse `runs/loso_20260115_080732/fold_*/epoch_metrics.csv` |
| Model parameter count | Not logged | Add `sum(p.numel() for p in model.parameters())` to training script |
| Dataset subject-level summary | Partially available | Parse from `window_counts_full.csv` + cache metadata |
| Calibration results (ECE/MCE) | Not run on final LOSO | Run calibration module on saved predictions |
| requirements.txt / environment snapshot | MISSING | `pip freeze > requirements.txt` |

---

## Appendix: Machine-Readable Manifest (JSON)

```json
{
  "project": "seizure_forecasting",
  "dataset": "CHB-MIT",
  "generated": "2026-02-25",
  "repo_root": "c:\\Users\\ansar\\Downloads\\isp",
  "dataset_summary_file": "reports/tables/window_counts_full.csv",
  "cache_report_file": "reports/tables/window_counts_full.csv",
  "channel_log_file": "reports/tables/channel_log.csv",
  "final_loso_run_dir": "runs/loso_20260115_080732",
  "final_loso_results_csv": "reports/tables/loso_clinical_summary.csv",
  "final_loso_summary_csv": "reports/tables/loso_summary.csv",
  "earlier_loso_results_csv": "reports/tables/loso_results.csv",
  "config_file": "configs/full_run.yaml",
  "frozen_config": "runs/loso_20260115_080732/config.yaml",
  "ablation_tables": ["reports/tables/alarm_ablation.csv"],
  "within_subject_runs": [
    "runs/deep_within_subject_20260114_034117_chb01",
    "runs/deep_within_subject_20260114_073245_chb10"
  ],
  "baseline_metrics": "reports/tables/baseline_metrics.csv",
  "summary_results": "reports/tables/summary_results.csv",
  "perf_logs": "reports/tables/perf_log_fold_chb01.csv ... perf_log_fold_chb24.csv (24 files)",
  "checkpoint_paths": "runs/loso_20260115_080732/fold_chb{01..24}/checkpoints/best.pt",
  "figures_paths": "MISSING — need regeneration",
  "source_code": {
    "model": "src/models/fusion_net.py",
    "baseline": "src/models/baseline.py",
    "preprocessing": "src/signal/preprocess.py",
    "spectrograms": "src/signal/spectrograms.py",
    "features": "src/features/handcrafted.py",
    "training_loop": "src/train/loops.py",
    "losses": "src/train/losses.py",
    "metrics": "src/train/metrics.py",
    "threshold_tuning": "src/train/threshold_tuning.py",
    "alarm_eval": "src/train/alarm_eval.py",
    "calibration": "src/train/calibration.py",
    "loso_script": "scripts/run_loso.py",
    "ablation_script": "scripts/run_ablations.py"
  }
}
```

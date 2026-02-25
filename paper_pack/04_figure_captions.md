# Phase 4: Figure Descriptions & Captions

**Status:** All figures are MISSING from `reports/figures/`. They must be regenerated. Below are the specifications and draft captions for each required figure, along with exact data sources and generation commands.

---

## Figure 1: Dataset Composition

**Filename:** `paper_pack/figures/dataset_composition.png`  
**Data source:** `paper_pack/tables/dataset_subject_summary.csv`, `reports/tables/window_counts_full.csv`

**Description:** Three-panel figure:
- (a) Bar chart of seizure count per subject (24 bars)
- (b) Pie chart of window class distribution (preictal 4.66% vs interictal 95.34%)
- (c) Bar chart of test partition size per subject

**Caption draft:** "Dataset composition of the CHB-MIT Scalp EEG Database. (a) Number of seizures per subject in the test partition, ranging from 0 (chb06) to 10 (chb15). (b) Class distribution across all 66,695 windows: 3,107 preictal (4.66%) vs 63,588 interictal (95.34%), reflecting severe class imbalance. (c) Number of test windows per subject, showing heterogeneous recording durations."

**Generation:** MISSING — requires custom plotting script.

---

## Figure 2: Labeling Timeline Schematic

**Filename:** `paper_pack/figures/labeling_timeline.png`  
**Data source:** `docs/LABELING_SCHEMA.md`

**Description:** Annotated horizontal timeline showing interictal, preictal (30 min), gap (5 min), ictal, postictal (30 min), and interictal buffer (60 min) zones relative to seizure onset.

**Caption draft:** "Labeling schema for seizure forecasting. Each EEG window is classified as preictal (up to 30 minutes before seizure onset, excluding a 5-minute gap), interictal (at least 60 minutes from any seizure), or excluded (ictal, postictal recovery, or buffer zone). The soft risk target y_soft increases exponentially approaching onset with τ=1800s."

**Generation:** MISSING — requires custom schematic script.

---

## Figure 3: Preprocessing / Pipeline Diagram

**Filename:** `paper_pack/figures/pipeline_diagram.png`  
**Data source:** Architectural description

**Description:** Flow diagram: Raw EDF → Channel Selection (18 canonical) → Resample (256 Hz) → Bandpass (0.5–50 Hz) → Notch (60 Hz) → CAR → Windowing (30s/15s stride) → [Branch A: STFT → CNN Encoder] + [Branch B: Handcrafted Features → MLP] → Fusion → Classification + Regression → Post-processing (EMA + Persistence + Hysteresis) → Alarm

**Caption draft:** "End-to-end seizure forecasting pipeline. Raw EDF recordings undergo preprocessing (channel standardization, resampling, filtering, re-referencing) and windowing. Each 30-second window is processed through a dual-branch FusionNet: a CNN branch operating on GPU-computed spectrograms and an MLP branch encoding 30 handcrafted features. The fusion head produces both a binary classification logit and a continuous risk score, which undergoes alarm post-processing (EMA smoothing, persistence filter, hysteresis, refractory period) before clinical alarm generation."

**Generation:** MISSING — requires diagram tool (draw.io, matplotlib, or similar).

---

## Figure 4: Model Architecture Diagram

**Filename:** `paper_pack/figures/model_architecture.png`  
**Data source:** `src/models/fusion_net.py`

**Description:** Block diagram of FusionNet showing CNN Encoder (Conv→BN→ReLU→Pool ×3 → AdaptivePool → FC), Feature MLP (Linear→BN→ReLU ×2 → Linear), Fusion Head (Concat → MLP → Classification Head + Regression Head).

**Caption draft:** "FusionNet architecture for seizure forecasting. The CNN encoder processes 18-channel log-magnitude spectrograms through three convolutional blocks with batch normalization, producing a 128-dimensional embedding. The feature MLP encodes 30 handcrafted EEG features into a 64-dimensional embedding. The fusion head concatenates both embeddings and produces a binary classification logit and a continuous soft risk score via separate output heads."

**Generation:** MISSING.

---

## Figure 5: Training Curves

**Filename:** `paper_pack/figures/training_curves_representative.png`  
**Data source:** `runs/loso_20260115_080732/fold_chb01/epoch_metrics.csv` (LOSO), `runs/deep_within_subject_20260114_034117_chb01/` (within-subject)

**Description:** Two-panel figure:
- (a) Within-subject chb01: train loss + val AUROC across epochs
- (b) LOSO fold chb01: train loss + val AUROC across epochs

**Caption draft:** "Representative training curves. (a) Within-subject training on chb01 showing rapid convergence and high validation AUROC. (b) LOSO fold with chb01 as test subject, showing training loss decrease with stable validation AUROC around 0.82."

**Generation:** MISSING — parse epoch_metrics.csv files and plot.

---

## Figure 6: ROC and PR Curves

**Filename:** `paper_pack/figures/roc_pr_curves.png`  
**Data source:** Would require saved predictions (not stored)

**Description:** Four-panel figure:
- (a) ROC curve for best LOSO fold (e.g., chb20, AUROC=0.94)
- (b) ROC curve for worst LOSO fold (e.g., chb24, AUROC=0.30)
- (c) PR curve for best fold
- (d) PR curve for worst fold

**Caption draft:** "ROC and precision-recall curves for representative LOSO folds. (a,c) Best-performing fold (chb20, AUROC=0.94). (b,d) Worst-performing fold (chb24, AUROC=0.30). The disparity reflects substantial inter-patient variability in cross-subject seizure forecasting."

**Generation:** MISSING — requires re-running inference with saved checkpoints to obtain raw predictions. Commands:

```bash
python -c "
import torch
from src.models.fusion_net import FusionNet
# Load best.pt for chb20, run inference on test loader, save predictions
"
```

---

## Figure 7: Alarm Tradeoff Plots

**Filename:** `paper_pack/figures/alarm_tradeoff.png`  
**Data source:** `reports/tables/loso_summary.csv` (FAH-sensitivity means)

**Description:** Three-panel figure:
- (a) FAH target vs achieved sensitivity (from summary)
- (b) Threshold vs FAH (from ablation)
- (c) FAH vs sensitivity tradeoff curve

**Caption draft:** "Alarm performance tradeoffs across FAH targets in LOSO evaluation. (a) Mean sensitivity increases from 43.8% to 51.4% as the FAH budget increases from 0.1 to 1.0 per hour. (b) Higher thresholds reduce false alarms but sacrifice seizure detection. (c) The FAH-sensitivity operating characteristic shows the fundamental tradeoff in cross-subject forecasting."

**Generation:** MISSING — plot from summary CSV data.

---

## Figure 8: LOSO Per-Fold Performance

**Filename:** `paper_pack/figures/loso_per_fold_barplot.png`  
**Data source:** `paper_pack/tables/loso_per_fold_results.csv`

**Description:** Grouped bar/point plot with 24 subjects on x-axis, test AUROC (blue) and val AUROC (orange) on y-axis, with horizontal mean±std bands.

**Caption draft:** "Per-fold test AUROC across 24 LOSO folds. Validation AUROC (orange) is consistently high (0.81±0.01), while test AUROC (blue) shows substantial inter-patient variability (0.60±0.20). Dashed lines indicate mean ± one standard deviation."

**Generation:** MISSING — plot from CSV.

---

## Figure 9: Calibration Plot

**Filename:** `paper_pack/figures/calibration_reliability.png`  
**Data source:** Not available (calibration not run on final LOSO)

**Description:** Reliability diagram (predicted probability vs observed frequency) with ECE annotation.

**Caption draft:** Would document calibration quality of the model's probability estimates.

**Generation:** MISSING — requires running calibration module on saved LOSO predictions.

---

## Figure 10: Performance Profiling

**Filename:** `paper_pack/figures/perf_profiling.png`  
**Data source:** `reports/tables/perf_log_fold_chb01.csv`, `reports/tables/perf_log_fold_chb24.csv`

**Description:** Stacked area or line plot showing data_ms, fwd_ms, bwd_ms over training steps, demonstrating the data loading bottleneck and its resolution.

**Caption draft:** "Training step timing breakdown for a representative LOSO fold. Data loading (blue) dominates early steps (~520ms) before OS-level caching reduces it to ~30ms. GPU forward (orange, ~40ms) and backward (green, ~40ms) passes are stable throughout, confirming that the data pipeline — not GPU compute — was the primary bottleneck."

**Generation:** MISSING — plot from perf_log CSVs.

---

## Generation Commands

To regenerate all figures that can be produced from existing data:

```bash
# Ensure matplotlib is installed
pip install matplotlib seaborn

# Run the figure generation script (if it supports paper_pack output)
python scripts/make_figures.py --output paper_pack/figures/

# Or generate individual figures with a custom script (to be written)
python paper_pack/generate_paper_figures.py
```

**Note:** Figures 6 and 9 require re-running model inference to obtain raw prediction arrays. All other figures can be generated from existing CSV data.

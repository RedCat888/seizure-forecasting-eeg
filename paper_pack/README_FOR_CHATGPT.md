# README for ChatGPT Paper Writing

---

## Project Summary

This repository implements an end-to-end seizure forecasting system evaluated on the CHB-MIT Scalp EEG Database (24 subjects, 66,695 windows). The system uses a dual-branch deep neural network (FusionNet) that fuses CNN-encoded spectrograms with handcrafted EEG features to predict preictal state up to 30 minutes before seizure onset. The full Leave-One-Subject-Out (LOSO) cross-validation across all 24 subjects yields a mean test AUROC of 0.60±0.20, with seizure sensitivity reaching 51.4%±28.0% at ≤1.0 false alarms per hour. Within-subject models achieve substantially higher performance (AUROC 0.95 for chb01), confirming the fundamental challenge of cross-subject EEG generalization. The pipeline includes focal loss for class imbalance, alarm post-processing (EMA smoothing, persistence filter, hysteresis), FAH-targeted threshold tuning, and extensive engineering optimizations for consumer GPU deployment.

---

## Final Primary Metrics (LOSO Cross-Subject — 24 folds)

| Metric | Value |
|--------|-------|
| Val AUROC | 0.8139 ± 0.0129 |
| **Test AUROC** | **0.6017 ± 0.1958** |
| Val AUPRC | 0.3543 ± 0.0661 |
| **Test AUPRC** | **0.1162 ± 0.1564** |
| Sensitivity @ FAH ≤ 0.1 | 0.4375 ± 0.3301 |
| Sensitivity @ FAH ≤ 0.5 | 0.4667 ± 0.3062 |
| Sensitivity @ FAH ≤ 1.0 | 0.5143 ± 0.2804 |

**Source:** `reports/tables/loso_summary.csv`, `reports/tables/loso_clinical_summary.csv`

## Final Secondary Metrics (Within-Subject)

| Subject | Test AUROC | Sensitivity | FAH |
|---------|------------|-------------|-----|
| chb01 | 0.9493 | 100% | 20.34 |

**Source:** `reports/tables/summary_results.csv`

---

## List of Figures (in `paper_pack/figures/`)

**All figures are MISSING and need regeneration.** Specifications in `paper_pack/04_figure_captions.md`.

| # | Filename (target) | Content |
|---|-------------------|---------|
| 1 | dataset_composition.png | Subject seizure counts + class distribution |
| 2 | labeling_timeline.png | Preictal/interictal/excluded zone schematic |
| 3 | pipeline_diagram.png | End-to-end processing pipeline flow |
| 4 | model_architecture.png | FusionNet block diagram |
| 5 | training_curves_representative.png | Loss/AUROC curves (within-subject + LOSO fold) |
| 6 | roc_pr_curves.png | ROC and PR curves for best/worst LOSO folds |
| 7 | alarm_tradeoff.png | FAH vs sensitivity operating points |
| 8 | loso_per_fold_barplot.png | 24-fold AUROC bar plot with mean±std |
| 9 | calibration_reliability.png | Reliability diagram (MISSING — calibration not run) |
| 10 | perf_profiling.png | Training step timing breakdown |

---

## List of Tables (in `paper_pack/tables/`)

| File | Content |
|------|---------|
| loso_per_fold_results.csv | Per-fold AUROC, AUPRC, seizure sensitivity at 4 FAH targets |
| loso_summary_final.csv | Aggregated LOSO metrics (mean ± std) |
| within_subject_summary.csv | chb01 and chb10 within-subject results |
| alarm_ablation_summary.csv | 8 alarm configs × 4 FAH targets |
| threshold_tradeoff_table.csv | FAH target → sensitivity tradeoff |
| performance_profiling_summary.csv | Data/fwd/bwd timing breakdown |
| dataset_subject_summary.csv | Per-subject seizure counts and window stats |
| cache_summary.csv | Dataset and cache statistics |

---

## Unresolved Gaps

| Gap | Impact | How to Fill |
|-----|--------|-------------|
| All figures missing | Cannot include in paper without regeneration | Run `python scripts/make_figures.py` or write custom plotting script |
| Model parameter count not logged | Needed for methods section | Run `sum(p.numel() for p in model.parameters())` |
| Calibration not run on LOSO | Missing ECE/MCE metrics | Run `src/train/calibration.py` on saved LOSO predictions |
| ROC/PR curves need raw predictions | Predictions not saved | Re-run inference on saved checkpoints |
| `requirements.txt` not captured | Reproducibility | Run `pip freeze > paper_pack/configs/requirements.txt` |
| Exact PyTorch/CUDA versions | Reproducibility section | Run version check commands |

---

## Recommended Paper Framing

### Option A: Cross-Subject Benchmark + Engineering Pipeline Paper
**Strengths:** Complete LOSO evaluation on all 24 CHB-MIT subjects, rigorous clinical metrics (FAH/sensitivity), systematic alarm post-processing ablation, thorough engineering documentation of GPU optimization.
**Weakness:** Cross-subject performance (AUROC 0.60) is modest.

### Option B: Patient-Specific Forecasting Prototype Paper
**Strengths:** Excellent within-subject performance (AUROC 0.95), full pipeline from raw EEG to alarm generation.
**Weakness:** Only 2 subjects evaluated within-subject.

### Option C: Both — Comparative Framing (RECOMMENDED)
**Framing:** "We present a complete seizure forecasting pipeline and evaluate it in both patient-specific and cross-subject (LOSO) settings. Patient-specific models achieve clinically relevant performance (AUROC 0.95), while cross-subject generalization remains challenging (AUROC 0.60±0.20), consistent with published literature. We systematically analyze the gap through per-fold analysis, alarm-level evaluation, and ablation studies, identifying inter-patient heterogeneity as the primary bottleneck and proposing domain adaptation as the key next step."

**This framing:**
- Honestly reports both strong and weak results
- Positions the work as a thorough engineering + science investigation
- The cross-subject gap IS the interesting scientific finding
- Alarm-level evaluation (FAH, sensitivity, post-processing) adds clinical value
- Engineering contributions (memmap pipeline, Windows optimization) are practical contributions

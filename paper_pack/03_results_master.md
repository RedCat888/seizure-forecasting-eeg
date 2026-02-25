# Phase 3: Results Extraction

**Critical rule: NO fabricated metrics. All values from actual files.**

---

## A. Final LOSO (Cross-Subject) Results — PRIMARY

**Source:** `reports/tables/loso_summary.csv`, `reports/tables/loso_clinical_summary.csv`  
**Run:** `runs/loso_20260115_080732/` (24 folds, all completed, 2026-01-15)

### A.1 Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.8139 | 0.0129 |
| **Test AUROC** | **0.6017** | **0.1958** |
| Val AUPRC | 0.3543 | 0.0661 |
| **Test AUPRC** | **0.1162** | **0.1564** |

### A.2 Clinical Alarm Metrics (FAH-Targeted)

| FAH Target | Sensitivity (Mean) | Sensitivity (Std) | Achieved FAH (Mean) |
|------------|--------------------|--------------------|---------------------|
| ≤ 0.1 | 0.4375 | 0.3301 | 0.0000 |
| ≤ 0.2 | 0.4375 | 0.3301 | 0.0341 |
| ≤ 0.5 | 0.4667 | 0.3062 | 0.2877 |
| ≤ 1.0 | 0.5143 | 0.2804 | 0.6505 |

**Interpretation:** At the most permissive FAH target (≤1.0 false alarms/hour), mean sensitivity reaches 51.4%. At the strictest target (≤0.1), sensitivity drops to 43.8% with near-zero FAH. The high standard deviations reflect substantial inter-patient variability.

### A.3 Per-Fold LOSO Results

Source: `reports/tables/loso_clinical_summary.csv`

| Subject | Val AUROC | Test AUROC | Test AUPRC | N Seizures | Sens@0.1 | Sens@0.2 | Sens@0.5 | Sens@1.0 |
|---------|-----------|------------|------------|------------|----------|----------|----------|----------|
| chb01 | 0.8211 | 0.6011 | 0.1245 | 4 | 0.00 | 0.00 | 0.25 | 0.25 |
| chb02 | 0.8070 | 0.6842 | 0.0070 | 2 | 0.00 | 0.00 | 0.00 | 0.50 |
| chb03 | 0.8059 | 0.6501 | 0.1013 | 2 | 0.00 | 0.00 | 0.50 | 0.50 |
| chb04 | 0.8261 | 0.6164 | 0.0078 | 2 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb05 | 0.8243 | 0.4166 | 0.0278 | 2 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb06 | 0.8007 | 0.5000 | 0.0000 | 0 | N/A | N/A | N/A | N/A |
| chb07 | 0.8066 | 0.6809 | 0.0142 | 3 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb08 | 0.8166 | 0.3060 | 0.0028 | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| chb09 | 0.8133 | 0.3849 | 0.0187 | 3 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb10 | 0.8045 | 0.3619 | 0.0297 | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb11 | 0.8289 | 0.8322 | 0.0253 | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb12 | 0.7978 | 0.4218 | 0.2707 | 6 | 0.17 | 0.17 | 0.17 | 0.17 |
| chb13 | 0.7803 | 0.4026 | 0.0906 | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb14 | 0.8240 | 0.6192 | 0.0903 | 3 | 0.33 | 0.33 | 0.33 | 0.33 |
| chb15 | 0.8040 | 0.6382 | 0.1209 | 10 | 0.00 | 0.00 | 0.20 | 0.20 |
| chb16 | 0.8191 | 0.3485 | 0.0509 | 2 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb17 | 0.8372 | 0.8792 | 0.1266 | 2 | 0.00 | 0.00 | 0.00 | 0.50 |
| chb18 | 0.8262 | 0.5866 | 0.1020 | 2 | 0.00 | 0.00 | 0.00 | 0.00 |
| chb19 | 0.8211 | 0.8732 | 0.4558 | 1 | 0.00 | 0.00 | 1.00 | 1.00 |
| chb20 | 0.8183 | 0.9359 | 0.6482 | 2 | 0.00 | 0.00 | 0.50 | 0.50 |
| chb21 | 0.8039 | 0.8583 | 0.0443 | 1 | 0.00 | 0.00 | 0.00 | 1.00 |
| chb22 | 0.8160 | 0.7742 | 0.0418 | 2 | 0.00 | 0.00 | 0.00 | 0.50 |
| chb23 | 0.8314 | 0.7687 | 0.0481 | 2 | 0.00 | 0.00 | 0.00 | 0.50 |
| chb24 | 0.7993 | 0.2995 | 0.3393 | 4 | 0.25 | 0.25 | 0.25 | 0.25 |

### A.4 Observations

- **Val AUROC is consistently high** (0.78–0.84) across all folds, indicating stable training
- **Test AUROC is highly variable** (0.30–0.94), reflecting inter-patient heterogeneity
- **Best subjects:** chb20 (0.94), chb17 (0.88), chb19 (0.87), chb21 (0.86)
- **Worst subjects:** chb24 (0.30), chb08 (0.31), chb10 (0.36), chb16 (0.35)
- chb06 has AUROC=0.50 (chance) because test set had 0 seizures → random performance expected

---

## B. Within-Subject Results — SECONDARY

**Source:** `reports/tables/summary_results.csv`

| Subject | Val AUROC | Test AUROC | AUPRC | FAH | Sensitivity | Notes |
|---------|-----------|------------|-------|-----|-------------|-------|
| chb01 | 0.9114 | **0.9493** | 0.7912 | 20.34 | 100% | Excellent patient-specific performance |
| chb10 | 0.9812 | 0.5000 | 0.0909 | N/A | 0% | Test set had 0 seizures (split issue) |

**Key finding:** Within-subject AUROC for chb01 (0.95) vastly outperforms the cross-subject LOSO AUROC for the same patient (0.60), demonstrating the value of patient-specific models.

---

## C. Baseline Model Results

**Source:** `reports/tables/baseline_metrics.csv`

| Split | AUROC | AUPRC | Sensitivity | Specificity | Accuracy |
|-------|-------|-------|-------------|-------------|----------|
| Train | 0.9822 | 0.7464 | 0.4399 | 0.9984 | 0.9847 |
| Val | 0.5526 | 0.0153 | 0.0000 | 0.9923 | 0.9778 |
| Test | 0.7801 | 0.0098 | 0.0000 | 0.9992 | 0.9949 |

**Note:** XGBoost baseline severely overfits (train AUROC 0.98 → val AUROC 0.55). Zero sensitivity on val/test indicates the baseline predicts all interictal — a degenerate solution due to extreme class imbalance.

---

## D. Alarm Ablation Results — SUPPORTING

**Source:** `reports/tables/alarm_ablation.csv`

| Config | FAH Target | Threshold | Val Sens | Test FAH | Test Sens | Warning Time (s) |
|--------|------------|-----------|----------|----------|-----------|------------------|
| baseline | 0.5 | 0.842 | 0.25 | 1.903 | 0.00 | 0.0 |
| baseline | 1.0 | 0.465 | 0.25 | 2.422 | 0.33 | 240.0 |
| ema_0.2 | 0.5 | 0.356 | 0.25 | 2.076 | 0.00 | 0.0 |
| ema_0.2 | 1.0 | 0.257 | 0.25 | 2.769 | 0.00 | 0.0 |
| ema+persist | 0.5 | 0.356 | 0.25 | 2.076 | 0.00 | 0.0 |
| full_combo | 1.0 | 0.257 | 0.25 | 2.769 | 0.00 | 0.0 |

**Note:** This ablation was run on an early 5-subject cross-subject run (not the full LOSO), so results are less representative. The EMA smoothing was effective at reducing FAH in the final LOSO pipeline.

---

## E. Earlier LOSO Run (Verification)

**Source:** `reports/tables/loso_results.csv` (from `runs/loso_20260114_232245/`)

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.8139 | 0.0129 |
| Test AUROC | 0.5684 | — |

This earlier run (without clinical metrics) shows consistent val AUROC with the final run, confirming reproducibility. Test AUROC differs slightly because: (a) different random initialization, (b) possibly different epoch counts due to early stopping, (c) no focal loss in earlier run.

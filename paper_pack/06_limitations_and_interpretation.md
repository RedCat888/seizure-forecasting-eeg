# Phase 6: Limitations, Interpretation, and Next Steps

**All claims below are grounded in actual experimental evidence from this repository.**

---

## 1. Why Cross-Subject Forecasting Is Hard

### Evidence from this work

- **LOSO Test AUROC:** 0.60 ± 0.20 (chance is 0.50)
- **Within-subject AUROC:** 0.95 for chb01
- **Gap:** Cross-subject performance is ~35 AUROC points below patient-specific performance

### Root causes

1. **Inter-patient EEG heterogeneity:** Scalp EEG is highly patient-specific due to skull thickness, cortical folding, electrode impedance, and age. Our LOSO results show test AUROC ranging from 0.30 to 0.94, a 64-point spread across subjects.

2. **Class imbalance:** With only 4.66% preictal windows, the model is strongly biased toward predicting interictal. The XGBoost baseline achieves 0% sensitivity while maintaining 99.9% accuracy — a degenerate but statistically "correct" strategy.

3. **Small N relative to heterogeneity:** 24 subjects is insufficient to capture the diversity of epilepsy presentations. Published literature suggests 100+ subjects are needed for robust cross-subject models.

4. **Seizure type diversity:** CHB-MIT contains patients with different epilepsy types (temporal lobe, generalized, etc.), making a single model inherently challenging.

## 2. Why Within-Subject Works Better

- The chb01 within-subject model (AUROC 0.95, sensitivity 100%) demonstrates that seizure signatures are highly consistent within a patient
- Channel impedances and montage remain stable across recordings for the same patient
- Temporal patterns (e.g., diurnal rhythms) are captured by chronological splitting

## 3. Clinical Relevance of FAH and Post-Processing

### False Alarm Rate

- At FAH ≤ 1.0 (one false alarm per hour), mean sensitivity is only 51.4% — clinically insufficient for reliable forecasting
- At FAH ≤ 0.1, sensitivity remains at 43.8% but achieved FAH is essentially zero, suggesting the model does identify some seizures at very high confidence but misses many

### Post-processing Impact

- EMA smoothing (α=0.2) was shown in ablation to reduce FAH by ~3× in early experiments
- The combination of EMA + persistence + hysteresis in the final LOSO run enabled more controlled alarm generation
- Refractory period (20 min) prevents alarm flooding during prolonged high-risk periods

## 4. Overfitting Risk

### Evidence

- **XGBoost baseline:** Train AUROC 0.98 → Val AUROC 0.55 (extreme overfit)
- **FusionNet LOSO:** Val AUROC 0.81 → Test AUROC 0.60 (moderate generalization gap)
- **Within-subject chb10:** Val AUROC 0.98 → Test AUROC 0.50 (overfit due to no seizures in test)

### Mitigations applied

- Focal loss (reduces easy-negative gradient domination)
- Dropout (0.3) and BatchNorm throughout
- Weight decay (1e-4)
- Data augmentation (noise, scaling, channel dropout)
- Early stopping (patience=10 on val AUROC)

### Mitigations NOT applied

- Label smoothing (in medium config but not final)
- MixUp / CutMix augmentation
- Adversarial training

## 5. Engineering vs. Model Science Improvements

| Category | Improvement | Impact |
|----------|-----------|--------|
| Engineering | Cache V2 (memmap) | Eliminated HDF5 I/O bottleneck |
| Engineering | Worker-safe dataset | Enabled num_workers=4 → 4× throughput |
| Engineering | NaN feature handling | Prevented training crashes |
| Engineering | Config defaults | Eliminated runtime KeyError crashes |
| Model science | Focal loss | Better gradient signal for rare preictal class |
| Model science | EMA smoothing | ~3× FAH reduction |
| Model science | Proximity weighting | Higher weight for windows near seizure onset |
| Evaluation | FAH-targeted threshold tuning | Clinically meaningful operating points |

## 6. Concrete Next Steps

### 6.1 More Data / Multi-Center

- **TUH EEG Corpus:** 10,000+ patients (requires application)
- **EPILEPSIAE:** European multi-center dataset
- **CHB-MIT + SIENA:** Combine datasets for larger N
- Expected impact: More training diversity → better cross-subject generalization

### 6.2 Focal Loss + Threshold + Calibration Refinements

- Apply temperature scaling (already implemented in `src/train/calibration.py`) to improve probability estimates
- Tune focal alpha/gamma per fold via nested cross-validation
- Investigate class-balanced sampling as alternative to focal loss

### 6.3 Domain Adaptation / Patient-Invariant Representations

- **DANN** (Domain Adversarial Neural Network): Train model to be invariant to patient identity
- **Contrastive learning:** SimCLR or MoCo to learn universal EEG representations
- **Patient clustering:** Group similar patients for semi-personalized models

### 6.4 Pretrain + Per-Patient Finetune

- Pretrain FusionNet on all subjects (cross-subject backbone)
- Finetune on target patient's early recordings (few-shot adaptation)
- This is the most promising near-term approach for clinical deployment

### 6.5 Better Feature Sets

- **Coherence / connectivity:** Phase-locking value, coherence matrices between channel pairs
- **Wavelet features:** Multi-resolution time-frequency analysis
- **Self-supervised EEG encoders:** Pre-trained representations (e.g., from BrainBERT or EEGNet pretraining)
- **Graph neural networks:** Treat channels as nodes with connectivity-based edges

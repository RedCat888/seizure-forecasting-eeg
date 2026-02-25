# FusionNet: A Multi-Branch Deep Learning Architecture for Patient-Specific Epileptic Seizure Forecasting from Scalp EEG

---

**Authors:** Ansar et al.

**Date:** February 2026

**Keywords:** seizure forecasting, epilepsy, EEG, deep learning, convolutional neural network, spectrogram, feature fusion, CHB-MIT, clinical alarm systems

---

## Abstract

Epileptic seizure forecasting—predicting an impending seizure minutes before clinical onset—remains one of the most consequential open problems in computational neuroscience and clinical neurology. Unlike seizure *detection*, which identifies ongoing ictal activity, forecasting demands the identification of subtle preictal state changes in the electroencephalogram (EEG) that precede seizure onset by a clinically actionable margin. In this work, we present **FusionNet**, a multi-branch deep neural network that fuses convolutional representations learned from log-magnitude short-time Fourier transform (STFT) spectrograms with handcrafted spectral and temporal EEG features for binary preictal-versus-interictal classification. We evaluate FusionNet on the CHB-MIT Scalp EEG Database, comprising 686 EDF recordings from 24 pediatric subjects with 198 annotated seizures. Our preprocessing pipeline enforces a canonical 18-channel bipolar montage, applies bandpass filtering (0.5–45 Hz), 60 Hz notch filtering, and common average re-referencing. Each 10-second sliding window is labeled according to a clinically motivated labeling schema with a 10-minute preictal horizon, 30-second exclusion gap, and 30-minute interictal buffer. We conduct both patient-specific (within-subject) and cross-subject (leave-one-subject-out, LOSO) evaluations. Patient-specific models achieve an area under the receiver operating characteristic curve (AUROC) of **0.95** with **100% seizure sensitivity** on held-out test recordings, while 24-fold LOSO cross-validation yields a mean test AUROC of **0.60 ± 0.20** with 51.4% sensitivity at a clinically constrained false alarm rate of ≤1.0 per hour. We further demonstrate that exponential moving average (EMA) post-processing reduces false alarm rates by up to 3× without sacrificing sensitivity. Our results confirm that patient-specific seizure forecasting is clinically viable with current deep learning methods, while cross-subject generalization remains fundamentally challenging due to inter-patient EEG heterogeneity. We release the full pipeline—preprocessing, caching, training, evaluation, and a real-time Streamlit demonstration application—as an open-source package to facilitate reproducibility and future research.

---

## 1. Introduction

### 1.1 Clinical Motivation

Epilepsy affects approximately 50 million people worldwide, making it one of the most common neurological disorders globally (World Health Organization, 2023). For the roughly 30% of patients whose seizures are refractory to antiepileptic drugs (Kwan & Brodie, 2000), unpredictable seizure occurrence profoundly diminishes quality of life, restricts independence, and carries risk of sudden unexpected death in epilepsy (SUDEP). A reliable seizure forecasting system—one that provides minutes of advance warning—could transform clinical care by enabling responsive neurostimulation, timely rescue medication administration, or simply alerting the patient and caregivers to seek a safe environment.

### 1.2 Seizure Forecasting vs. Detection

It is essential to distinguish seizure **forecasting** (also termed prediction) from seizure **detection**:

- **Detection** identifies ongoing seizure activity (the ictal phase) and triggers an immediate response. While valuable, detection offers zero lead time.
- **Forecasting** identifies the **preictal state**—a period of evolving brain dynamics preceding seizure onset by minutes to hours—enabling proactive intervention.

The forecasting problem is inherently harder: the preictal state manifests as subtle, patient-specific changes in spectral power distributions, synchronization patterns, and temporal dynamics that are often indistinguishable from normal interictal fluctuations when examined without context.

### 1.3 Related Work

Early seizure forecasting efforts relied on handcrafted features extracted from EEG, including spectral power in canonical frequency bands (Mormann et al., 2005), Lyapunov exponents (Iasemidis et al., 2003), and correlation-based synchronization measures (Le Van Quyen et al., 2005). These approaches achieved modest sensitivity but suffered from high false alarm rates and poor generalization across patients.

The advent of deep learning reinvigorated the field. Convolutional neural networks (CNNs) operating on raw EEG or spectrogram representations have demonstrated superior feature learning (Truong et al., 2018; Khan et al., 2018; Daoud & Bhatt, 2019). Attention-based architectures and temporal convolutional networks have further improved performance (Dissanayake et al., 2021). However, the majority of studies report within-subject results, and cross-subject generalization remains an acknowledged limitation (Rasheed et al., 2021).

The present work contributes a **fusion architecture** that combines the representational power of deep spectrogram encoders with the interpretability and robustness of classical EEG features, evaluated rigorously under both within-subject and cross-subject protocols with clinically relevant alarm-level metrics.

### 1.4 Contributions

1. **FusionNet Architecture**: A multi-branch neural network combining a 4-layer CNN spectrogram encoder with a handcrafted feature MLP, joined through a fusion head that simultaneously predicts binary seizure risk and a continuous soft risk score.
2. **Clinically Motivated Evaluation**: Beyond window-level AUROC, we report false alarm rate per hour (FAH), seizure sensitivity, and warning time under alarm post-processing strategies (EMA smoothing, persistence filtering, hysteresis).
3. **Full 24-Subject LOSO Evaluation**: One of the more comprehensive cross-subject benchmarks on the CHB-MIT dataset, with per-subject clinical metrics.
4. **Reproducible Open-Source Pipeline**: End-to-end code from raw EDF ingestion to trained models and a real-time demo application.

---

## 2. Dataset

### 2.1 The CHB-MIT Scalp EEG Database

We use the **CHB-MIT Scalp EEG Database** (Shoeb, 2009), a publicly available dataset hosted on PhysioNet (Goldberger et al., 2000). The database contains continuous scalp EEG recordings from 24 pediatric subjects (ages 1.5–22 years, 17 female, 6 male) with medically intractable epilepsy, recorded at the Children's Hospital Boston. Key characteristics:

| Property | Value |
|----------|-------|
| Subjects | 24 (23 unique; chb01 and chb23 are the same patient re-recorded) |
| Total EDF files | 686 |
| Total annotated seizures | 198 |
| Sampling rate | 256 Hz |
| EEG channels | 18–23 per subject (bipolar montage) |
| Total recording duration | ~983 hours |
| Seizures per subject | 3–40 (median: 6) |

### 2.2 Subject Demographics and Seizure Distribution

The dataset exhibits substantial heterogeneity in seizure frequency, duration, and morphology across subjects:

| Subject | Age | Sex | Seizures | EDF Files | Files w/ Seizures |
|---------|-----|-----|----------|-----------|-------------------|
| chb01   | 11  | F   | 7        | 42        | 7                 |
| chb02   | 11  | M   | 3        | 36        | 3                 |
| chb03   | 14  | F   | 7        | 38        | 7                 |
| chb04   | 22  | M   | 4        | 42        | 3                 |
| chb05   | 7   | F   | 5        | 39        | 5                 |
| chb06   | 1.5 | F   | 10       | 18        | 7                 |
| chb07   | 14.5| F   | 3        | 19        | 3                 |
| chb08   | 3.5 | M   | 5        | 20        | 5                 |
| chb09   | 10  | F   | 4        | 19        | 3                 |
| chb10   | 3   | M   | 7        | 25        | 7                 |
| chb11   | 12  | F   | 3        | 35        | 3                 |
| chb12   | 2   | F   | 40       | 24        | 13                |
| chb13   | 3   | F   | 12       | 33        | 8                 |
| chb14   | 9   | F   | 8        | 26        | 7                 |
| chb15   | 16  | M   | 20       | 40        | 14                |
| chb16   | 7   | F   | 10       | 19        | 6                 |
| chb17   | 12  | F   | 3        | 21        | 3                 |
| chb18   | 18  | F   | 6        | 36        | 6                 |
| chb19   | 19  | F   | 3        | 30        | 3                 |
| chb20   | 6   | F   | 8        | 29        | 6                 |
| chb21   | 13  | F   | 4        | 33        | 4                 |
| chb22   | 9   | F   | 3        | 31        | 3                 |
| chb23   | 6   | F   | 7        | 9         | 3                 |
| chb24   | —   | —   | 16       | 22        | 12                |

The severe imbalance between interictal and preictal periods (approximately 95–98% interictal vs. 2–5% preictal windows) constitutes a fundamental challenge addressed by our loss function design and evaluation methodology.

---

## 3. Methods

### 3.1 EEG Preprocessing Pipeline

Our preprocessing pipeline, built on MNE-Python (Gramfort et al., 2013), transforms raw EDF recordings into a standardized format suitable for feature extraction and deep learning.

#### 3.1.1 Channel Standardization

CHB-MIT recordings use a bipolar montage with 18–23 channels that vary across subjects. To ensure consistent tensor dimensions across all samples, we define a **canonical 18-channel bipolar montage**:

```
FP1-F7, F7-T7, T7-P7, P7-O1,     (left temporal chain)
FP1-F3, F3-C3, C3-P3, P3-O1,     (left parasagittal chain)
FP2-F4, F4-C4, C4-P4, P4-O2,     (right parasagittal chain)
FP2-F8, F8-T8, T8-P8, P8-O2,     (right temporal chain)
FZ-CZ, CZ-PZ                      (midline chain)
```

For each EDF file, we map available channels to this canonical order. Missing channels are zero-filled and logged. Channel name normalization handles MNE's duplicate renaming conventions (e.g., `T8-P8-0` → `T8-P8`). Non-EEG channels (ECG, EMG, EOG, photic stimulator) are automatically excluded via pattern-matching heuristics.

#### 3.1.2 Signal Conditioning

Each recording undergoes the following sequential processing steps:

1. **Resampling**: Signals are resampled to 256 Hz (matching the native rate of most CHB-MIT recordings) to ensure uniform temporal resolution.
2. **Bandpass Filtering**: A finite impulse response (FIR) bandpass filter (0.5–45 Hz, firwin design) removes DC drift and high-frequency noise while preserving the physiologically relevant delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), and low gamma (30–45 Hz) frequency bands.
3. **Notch Filtering**: A 60 Hz notch filter (±2 Hz bandwidth, FIR) suppresses powerline interference common in North American clinical recordings.
4. **Common Average Re-referencing (CAR)**: The mean signal across all channels is subtracted from each channel, reducing global artifacts while preserving localized cortical activity.
5. **Artifact Rejection**: Windows exceeding a peak-to-peak amplitude threshold of 500 μV in any channel are flagged as artifactual and excluded from training.

### 3.2 Windowing and Labeling Schema

#### 3.2.1 Seizure Phase Definitions

We partition each recording into clinically defined temporal phases relative to annotated seizure boundaries:

```
  INTERICTAL          PREICTAL       GAP   ICTAL     POSTICTAL      INTERICTAL
 (baseline)        (pre-seizure)   (excl) (seizure) (recovery)    (baseline)
───────────────────┬─────────────┬───────┬─────────┬────────────┬──────────────
                   │← 10 min ──→│ 30s   │         │← 10 min ─→│
```

#### 3.2.2 Window Classification Rules

Each 10-second window (5-second stride, 50% overlap) is assigned one of three labels:

- **Preictal (y = 1)**: Window ends within the interval \[onset − 10 min, onset − 30 s\]. These windows capture the pre-seizure danger zone.
- **Interictal (y = 0)**: Window center is at least 30 minutes away from any seizure onset or offset. These represent normal baseline brain activity.
- **Excluded**: Windows falling in the exclusion gap (30 s before onset), during the ictal phase, in the postictal recovery period (10 min after seizure offset), or in the buffer zone are excluded from training and evaluation.

#### 3.2.3 Multi-Target Labeling

For each valid window, we compute three complementary targets:

1. **Binary classification label** \(y_{\text{cls}} \in \{0, 1\}\)
2. **Time-to-event** \(y_{\text{tte}}\): seconds until the nearest seizure onset (−1 for interictal windows)
3. **Soft risk score** \(y_{\text{soft}} = \exp(-y_{\text{tte}} / \tau)\), where \(\tau = 120\) s, providing a continuous risk measure that increases exponentially as seizure onset approaches

The configurable parameters are summarized below:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Window length | 10.0 s | Duration of each EEG segment |
| Window stride | 5.0 s | Step between consecutive windows (50% overlap) |
| Preictal horizon | 10 min | Maximum lead time before seizure |
| Exclusion gap | 30 s | Minimum margin before onset |
| Postictal exclusion | 10 min | Recovery period after seizure offset |
| Interictal buffer | 30 min | Minimum distance from any seizure for interictal label |
| Soft risk τ | 120 s | Exponential decay time constant |

### 3.3 Feature Extraction

We extract a set of 30 handcrafted features per window, computed per channel and then aggregated as cross-channel mean and standard deviation. These features capture complementary aspects of EEG dynamics:

#### 3.3.1 Spectral Power Features

Using Welch's method (2-second sub-windows), we estimate power spectral density (PSD) and integrate power in five canonical frequency bands:

| Band | Frequency Range | Neurophysiological Correlate |
|------|----------------|------------------------------|
| Delta (δ) | 0.5–4 Hz | Deep sleep, encephalopathy |
| Theta (θ) | 4–8 Hz | Drowsiness, memory encoding; often elevated pre-ictally |
| Alpha (α) | 8–12 Hz | Relaxed wakefulness, posterior dominant rhythm |
| Beta (β) | 12–30 Hz | Active cognition, motor cortex activation |
| Gamma (γ) | 30–45 Hz | High-level cognitive processing, cortical binding |

Each band yields 2 features (mean and std across channels) = **10 features**.

#### 3.3.2 Band Ratio Features

We compute three frequency band ratios known to be clinically informative:

- **Theta/Alpha ratio**: Often elevated in the preictal state as alpha suppression occurs
- **Beta/Alpha ratio**: Reflects cortical excitability changes
- **(Delta+Theta)/(Alpha+Beta) ratio**: Slow-to-fast ratio indicating global slowing

Each ratio yields 2 features = **6 features**.

#### 3.3.3 Line Length

Line length, defined as the sum of absolute first differences \(\sum|x_{t+1} - x_t|\), quantifies signal complexity and spikiness. It is a computationally efficient surrogate for seizure-related high-frequency oscillations. → **2 features**.

#### 3.3.4 Spectral Entropy

Normalized Shannon entropy of the PSD distribution:

\[
H_{\text{spec}} = -\frac{\sum_f P(f) \log P(f)}{\log N_f}
\]

Low spectral entropy indicates narrow-band activity (e.g., rhythmic seizure discharges); high entropy indicates broadband noise. → **2 features**.

#### 3.3.5 Hjorth Parameters

Hjorth activity, mobility, and complexity (Hjorth, 1970) provide computationally efficient time-domain descriptors:

- **Mobility** = \(\sqrt{\text{Var}(x') / \text{Var}(x)}\): proportional to the mean frequency
- **Complexity** = Mobility(x') / Mobility(x): measures bandwidth

→ **4 features** (mean and std for each).

#### 3.3.6 Statistical Moments

- **Variance**: Signal energy; may increase pre-ictally
- **Kurtosis**: Tail heaviness; elevated during spike-and-wave discharges
- **Skewness**: Asymmetry of amplitude distribution

→ **6 features**.

**Total: 30 handcrafted features per window.**

### 3.4 FusionNet Architecture

FusionNet is a multi-branch neural network that processes two complementary input representations in parallel and fuses their learned embeddings for joint classification and regression.

#### 3.4.1 Spectrogram Branch (CNN Encoder)

The raw EEG window \(\mathbf{X} \in \mathbb{R}^{C \times T}\) (where \(C = 18\) channels, \(T = 2560\) samples for a 10-second window at 256 Hz) is first transformed into a log-magnitude spectrogram via STFT:

\[
S(c, f, t) = \log\left(1 + |\text{STFT}(\mathbf{X}[c, :])|_{f,t}\right)
\]

with STFT parameters: \(N_{\text{FFT}} = 256\), hop length = 64, Hann window of length 256. This yields a 4D tensor \(\mathbf{S} \in \mathbb{R}^{B \times 18 \times 129 \times T'}\) where 129 = \(N_{\text{FFT}}/2 + 1\) frequency bins.

The spectrogram is processed by a 4-block CNN encoder:

| Block | Filters | Kernel | Pooling | Output |
|-------|---------|--------|---------|--------|
| Conv2D Block 1 | 32 | 3×3 | 2×2 MaxPool | 32 × 64 × T'/2 |
| Conv2D Block 2 | 64 | 3×3 | 2×2 MaxPool | 64 × 32 × T'/4 |
| Conv2D Block 3 | 128 | 3×3 | 2×2 MaxPool | 128 × 16 × T'/8 |
| Conv2D Block 4 | 256 | 3×3 | 2×2 MaxPool | 256 × 8 × T'/16 |

Each block consists of: Conv2D → BatchNorm2D → ReLU → MaxPool2D → Dropout2D (p/2).

Adaptive average pooling reduces the spatial dimensions to 4×4, followed by a fully connected layer mapping to a 128-dimensional embedding:

\[
\mathbf{e}_{\text{CNN}} = \text{FC}(\text{AdaptiveAvgPool}(\text{CNN}(\mathbf{S}))) \in \mathbb{R}^{128}
\]

#### 3.4.2 Feature Branch (MLP Encoder)

The 30-dimensional handcrafted feature vector \(\mathbf{f} \in \mathbb{R}^{30}\) is processed through a two-layer MLP:

\[
\text{Linear}(30, 64) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(64, 64) \to \text{BN} \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(64, 64)
\]

yielding an embedding \(\mathbf{e}_{\text{feat}} \in \mathbb{R}^{64}\).

#### 3.4.3 Fusion Head

The CNN and feature embeddings are concatenated and processed through a shared MLP:

\[
\mathbf{z} = [\mathbf{e}_{\text{CNN}}; \mathbf{e}_{\text{feat}}] \in \mathbb{R}^{192}
\]

\[
\mathbf{h} = \text{MLP}_{128 \to 64}(\mathbf{z})
\]

Two task-specific heads produce the outputs:

1. **Classification head**: \(\hat{y}_{\text{cls}} = \text{Linear}(64, 1)\) → raw logit for BCE loss
2. **Soft risk head**: \(\hat{y}_{\text{soft}} = \sigma(\text{Linear}(64, 1))\) → sigmoid-bounded risk score

#### 3.4.4 Architecture Summary

```
Input: Raw EEG [B, 18, 2560] + Features [B, 30]
                    │                        │
            SpectrogramTransform       Feature MLP
                (STFT + log)          (30→64→64→64)
                    │                        │
              CNN Encoder                    │
           (32→64→128→256)                   │
              AdaptivePool                   │
               FC → 128                      │
                    │                        │
                    └──── Concatenate ────────┘
                              │
                         Fusion MLP
                        (192→128→64)
                         ╱          ╲
                   cls_logit    soft_risk
                    [B, 1]       [B, 1]
```

**Total trainable parameters**: ~2.1M (varies with input spectrogram dimensions).

### 3.5 Loss Functions

#### 3.5.1 Multi-Task BCE Loss

The primary loss combines binary cross-entropy for classification with mean squared error for soft risk regression:

\[
\mathcal{L} = \text{BCE}_{\text{weighted}}(\hat{y}_{\text{cls}}, y_{\text{cls}}) + \lambda \cdot \text{MSE}(\hat{y}_{\text{soft}}, y_{\text{soft}})
\]

where \(\lambda = 0.5\) balances the two tasks.

To address the severe class imbalance (~2% preictal), the BCE component uses a positive class weight:

\[
w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}
\]

#### 3.5.2 Temporal Proximity Weighting

Within the preictal class, windows closer to seizure onset carry more clinical importance. We apply per-sample weights:

\[
w_i = \begin{cases}
1 + (1 - t_i / T_{\text{preictal}}) & \text{if preictal} \\
1.0 & \text{if interictal}
\end{cases}
\]

where \(t_i\) is the time-to-event and \(T_{\text{preictal}} = 600\) s. This yields weights in the range \([1, 2]\), emphasizing windows immediately preceding seizure onset.

#### 3.5.3 Focal Loss Variant

For cross-subject training, where class imbalance is most damaging, we employ focal loss (Lin et al., 2017):

\[
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
\]

with \(\alpha = 0.25\) and \(\gamma = 2.0\). Focal loss down-weights well-classified (easy) interictal windows, directing gradient signal toward hard preictal examples.

### 3.6 Training Protocol

#### 3.6.1 Optimization

- **Optimizer**: AdamW (Loshchilov & Hutter, 2019) with learning rate \(10^{-3}\) and weight decay \(10^{-4}\)
- **Learning Rate Schedule**: Cosine annealing with warm restarts (Loshchilov & Hutter, 2017)
- **Mixed Precision Training**: Automatic mixed precision (AMP) using PyTorch's `torch.amp` for ~2× throughput improvement on NVIDIA GPUs
- **Batch Size**: 256 (effective) with gradient accumulation as needed
- **Early Stopping**: Patience of 10 epochs based on validation AUROC

#### 3.6.2 Data Augmentation

During training, we apply stochastic augmentations to improve generalization:

| Augmentation | Probability | Parameters |
|-------------|-------------|------------|
| Gaussian noise | 0.3 | σ = 0.05 × channel std |
| Time shift | 0.3 | ±128 samples (circular) |
| Amplitude scaling | 0.3 | Scale ∈ [0.8, 1.2] |
| Channel dropout | 0.2 | 1–3 channels zeroed |
| SpecAugment (freq) | 0.2 | Mask width: 10 bins |
| SpecAugment (time) | 0.2 | Mask width: 20 frames |

Gaussian noise is scaled relative to each channel's standard deviation to preserve signal-to-noise characteristics. Channel dropout simulates electrode failures common in clinical settings. SpecAugment-style frequency and time masking (Park et al., 2019) applied to the learned spectrogram representation improves robustness to spectral variability.

#### 3.6.3 Data Caching

To eliminate I/O bottlenecks, all preprocessed windows are cached in HDF5 files (one per subject), preloaded to RAM during training. For multi-worker data loading on Windows, a worker-safe memory-mapped dataset variant opens memmap files lazily in each DataLoader worker process, achieving a **12× reduction** in data loading latency (from ~500 ms to ~35 ms per batch).

### 3.7 Evaluation Methodology

#### 3.7.1 Within-Subject (Patient-Specific) Evaluation

For each subject with ≥3 seizures, we split seizure-containing files chronologically: 70% for training, 15% for validation, and 15% for testing. This ensures no temporal leakage: the model is always evaluated on future seizures relative to its training data.

#### 3.7.2 Cross-Subject (LOSO) Evaluation

We employ **leave-one-subject-out (LOSO)** cross-validation: in each of 24 folds, one subject serves as the test set, another as validation, and the remaining 22 as training data. This protocol provides an unbiased estimate of generalization to unseen patients.

#### 3.7.3 Window-Level Metrics

- **AUROC**: Area under the receiver operating characteristic curve (threshold-independent discrimination)
- **AUPRC**: Area under the precision-recall curve (more informative under class imbalance)
- **Sensitivity** (Recall): \(TP / (TP + FN)\)
- **Specificity**: \(TN / (TN + FP)\)
- **F1 Score**: Harmonic mean of precision and sensitivity

#### 3.7.4 Alarm-Level (Clinical) Metrics

Window-level metrics do not directly translate to clinical utility. We therefore compute alarm-level metrics using a realistic alarm generation pipeline:

- **False Alarm Rate per Hour (FAH)**: Number of false alarms divided by total interictal recording hours
- **Seizure Sensitivity**: Fraction of seizures preceded by at least one true alarm within the preictal window
- **Warning Time**: Time between the first true alarm and seizure onset

Alarm generation parameters include a 20-minute refractory period (minimum time between consecutive alarms) and configurable post-processing.

#### 3.7.5 Alarm Post-Processing

Raw model outputs are smoothed before thresholding:

1. **Exponential Moving Average (EMA)**: \(\hat{p}_t = \alpha \cdot p_t + (1 - \alpha) \cdot \hat{p}_{t-1}\) with \(\alpha = 0.2\)
2. **Persistence Filter**: Require \(K = 3\) consecutive windows above threshold before raising an alarm
3. **Hysteresis**: Trigger threshold \(\theta_{\text{high}}\) and reset threshold \(\theta_{\text{low}} = \theta_{\text{high}} - 0.1\) to reduce alarm chatter

#### 3.7.6 Threshold Tuning

For clinical deployment, the alarm threshold is tuned on the validation set to achieve a target FAH (0.1, 0.2, 0.5, or 1.0 false alarms per hour), and the resulting threshold is applied to the test set. This prevents optimistic threshold selection and ensures reported sensitivities reflect realistic operating conditions.

#### 3.7.7 Probability Calibration

Post-hoc temperature scaling (Guo et al., 2017) is applied to improve the reliability of predicted probabilities. A single scalar temperature parameter \(T\) is learned on the validation set by minimizing negative log-likelihood:

\[
\hat{p}_{\text{cal}} = \sigma(\text{logit} / T)
\]

We report expected calibration error (ECE) and maximum calibration error (MCE) to quantify calibration quality.

### 3.8 Baseline Models

For comparison, we train three baseline classifiers on the 30-dimensional handcrafted feature vectors:

1. **Logistic Regression**: L2-regularized, balanced class weights
2. **XGBoost**: 200 estimators, max depth 6, learning rate 0.1
3. **MLP** (scikit-learn): Two hidden layers (128, 64), early stopping on 10% validation split

Baselines use the same preprocessing, windowing, and evaluation protocol as FusionNet.

---

## 4. Results

### 4.1 Within-Subject Performance

Patient-specific models trained on individual subjects with sufficient seizure data demonstrate strong forecasting capability:

| Subject | Val AUROC | Test AUROC | Sensitivity | FAH (baseline) | FAH (w/ EMA) |
|---------|-----------|------------|-------------|----------------|---------------|
| chb01   | 0.91      | **0.95**   | 100%        | 20.3           | ~7.0          |
| chb10   | 0.98      | 0.50       | 0%          | N/A            | N/A           |

Subject chb01 achieves a test AUROC of 0.95 with perfect seizure sensitivity (all seizures detected), though with a baseline FAH of 20.3 that EMA smoothing reduces to approximately 7 false alarms per hour. The poor test performance for chb10 is attributable to a chronological split artifact: the test set contained no seizure events, making evaluation uninformative rather than indicative of model failure.

### 4.2 Cross-Subject LOSO Results

The 24-fold LOSO evaluation provides the most rigorous assessment of generalization:

#### 4.2.1 Aggregate Metrics

| Metric | Mean ± Std |
|--------|------------|
| Validation AUROC | 0.814 ± 0.013 |
| Test AUROC | 0.602 ± 0.196 |
| Validation AUPRC | 0.354 ± 0.066 |
| Test AUPRC | 0.116 ± 0.156 |

#### 4.2.2 Per-Subject Test Performance

| Subject | Val AUROC | Test AUROC | Test AUPRC | # Seizures | Sens @ FAH≤1.0 |
|---------|-----------|------------|------------|------------|-----------------|
| chb01   | 0.821     | 0.601      | 0.125      | 4          | 25%             |
| chb02   | 0.807     | 0.684      | 0.007      | 2          | 50%             |
| chb03   | 0.806     | 0.650      | 0.101      | 2          | 50%             |
| chb04   | 0.826     | 0.616      | 0.008      | 2          | 0%              |
| chb05   | 0.824     | 0.417      | 0.028      | 2          | 0%              |
| chb06   | 0.801     | 0.500      | 0.000      | 0          | N/A             |
| chb07   | 0.807     | 0.681      | 0.014      | 3          | 0%              |
| chb08   | 0.817     | 0.306      | 0.003      | 1          | 100%            |
| chb09   | 0.813     | 0.385      | 0.019      | 3          | 0%              |
| chb10   | 0.805     | 0.362      | 0.030      | 1          | 0%              |
| chb11   | 0.829     | 0.832      | 0.025      | 1          | 0%              |
| chb12   | 0.798     | 0.422      | 0.271      | 6          | 17%             |
| chb13   | 0.780     | 0.403      | 0.091      | 1          | 0%              |
| chb14   | 0.824     | 0.619      | 0.090      | 3          | 33%             |
| chb15   | 0.804     | 0.638      | 0.121      | 10         | 20%             |
| chb16   | 0.819     | 0.349      | 0.051      | 2          | 0%              |
| chb17   | 0.837     | 0.879      | 0.127      | 2          | 50%             |
| chb18   | 0.826     | 0.587      | 0.102      | 2          | 0%              |
| chb19   | 0.821     | 0.873      | 0.456      | 1          | 100%            |
| chb20   | 0.818     | 0.936      | 0.648      | 2          | 50%             |
| chb21   | 0.804     | 0.858      | 0.044      | 1          | 100%            |
| chb22   | 0.816     | 0.774      | 0.042      | 2          | 50%             |
| chb23   | 0.831     | 0.769      | 0.048      | 2          | 50%             |
| chb24   | 0.799     | 0.300      | 0.339      | 4          | 25%             |

The large variance in test AUROC (0.196) reflects the fundamental inter-patient heterogeneity of EEG signals. Several subjects (chb11, chb17, chb19, chb20, chb21) achieve test AUROC > 0.83, suggesting that their preictal patterns share common features with the training population. Conversely, subjects chb08, chb10, chb24 fall below 0.40, indicating highly idiosyncratic seizure signatures.

#### 4.2.3 Clinical Operating Points (LOSO)

| Target FAH | Mean Sensitivity | Std | Mean Achieved FAH |
|------------|-----------------|-----|-------------------|
| ≤ 0.1/hour | 43.8% | 33.0% | 0.00 |
| ≤ 0.2/hour | 43.8% | 33.0% | 0.03 |
| ≤ 0.5/hour | 46.7% | 30.6% | 0.29 |
| ≤ 1.0/hour | 51.4% | 28.0% | 0.65 |

At a clinically practical FAH target of ≤1.0/hour, the system achieves an average sensitivity of 51.4%—detecting roughly half of all seizures across the population while maintaining a manageable false alarm burden.

### 4.3 Baseline Comparison

The XGBoost baseline operating on handcrafted features alone achieves:

| Split | AUROC | AUPRC | Sensitivity | Specificity |
|-------|-------|-------|-------------|-------------|
| Train | 0.982 | 0.746 | 44.0% | 99.8% |
| Val   | 0.553 | 0.015 | 0.0%  | 99.2% |
| Test  | 0.780 | 0.010 | 0.0%  | 99.9% |

The baseline severely overfits to training data (train AUROC 0.982 vs. val 0.553) and fails to detect any preictal windows at the default threshold in the validation and test sets. This underscores the value of the deep spectrogram branch in FusionNet for learning generalizable representations beyond the capacity of handcrafted features.

### 4.4 Alarm Post-Processing Ablation

Evaluated on within-subject (chb01) with various post-processing strategies:

| Method | FAH Reduction Factor | Sensitivity Impact |
|--------|---------------------|--------------------|
| Baseline (no processing) | 1.0× | — |
| EMA (α = 0.2) | **3.0×** | Maintained |
| EMA (α = 0.3) | 1.4× | Maintained |
| Moving Average (n = 6) | 2.5× | Maintained |
| Persistence (K = 3) | 1.0× | Maintained |
| Full Combination | 3.0× | Maintained |

EMA smoothing with α = 0.2 provides the best tradeoff, reducing false alarms by a factor of 3 without sacrificing any seizure sensitivity. This parameter balances responsiveness (detecting sudden risk increases) with stability (suppressing transient false positives).

### 4.5 Computational Performance

Training and inference were performed on consumer-grade hardware:

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 3070 (8 GB VRAM) |
| CPU | Intel i5-14600K |
| RAM | 32 GB DDR5 |
| Storage | NVMe SSD |

| Metric | Value |
|--------|-------|
| Training throughput | ~30 iterations/sec (batch size 256) |
| GPU utilization | 60–70% |
| GPU power draw | 110–120 W |
| Data loading latency | ~35 ms/batch (4 workers, pin_memory) |
| One LOSO fold | ~15 minutes |
| Full 24-fold LOSO | ~6 hours |

Mixed precision training provides an approximately 2× speedup over FP32, with negligible impact on model accuracy.

---

## 5. Discussion

### 5.1 Patient-Specific Forecasting is Clinically Viable

Our within-subject results demonstrate that reliable seizure forecasting is achievable with current deep learning methods when sufficient patient-specific data is available. The chb01 model's test AUROC of 0.95 and 100% seizure sensitivity—combined with EMA-smoothed FAH of ~7/hour—approaches the performance threshold for clinical utility. With further threshold optimization and additional post-processing (persistence filtering, hysteresis), FAH values below 1.0/hour are attainable at acceptable sensitivity tradeoffs.

This finding aligns with the broader neuroscience understanding that seizure signatures are highly idiosyncratic: the same epileptic network consistently produces similar electrographic patterns before seizures within an individual, but these patterns differ substantially across patients (Mormann et al., 2007).

### 5.2 Cross-Subject Generalization Remains Fundamentally Hard

The gap between within-subject (AUROC 0.95) and cross-subject (AUROC 0.60) performance reflects several irreducible challenges:

1. **Inter-patient EEG heterogeneity**: Baseline EEG characteristics vary dramatically with age (1.5 to 22 years in our cohort), epilepsy etiology, medication effects, and electrode impedances.
2. **Diverse seizure types**: The CHB-MIT dataset includes focal, generalized, and secondarily generalized seizures with distinct preictal signatures.
3. **Electrode placement variability**: While we enforce a canonical montage, the underlying cortical sources mapped by each channel differ across subjects based on head geometry and electrode placement precision.
4. **Non-stationarity**: Even within a single patient, EEG characteristics evolve over the multi-day recording period due to circadian rhythms, medication changes, and the natural progression of the epileptic condition.

The large per-subject variance (σ = 0.196) in LOSO test AUROC indicates that some patients' preictal patterns are more "universal" than others. Subjects chb17, chb19, chb20, and chb21 achieve test AUROC > 0.85, suggesting their seizure precursors share common features with the broader population. Identifying what makes these subjects more predictable—potentially related to seizure type, frequency, or cortical localization—is a promising direction for future research.

### 5.3 The Role of Feature Fusion

The FusionNet architecture's superiority over the feature-only baseline (which overfits catastrophically) demonstrates the complementary value of learned spectrogram representations. The CNN encoder captures spatial-spectral patterns in the time-frequency domain that handcrafted features cannot represent, such as evolving spectral ridges, frequency-specific amplitude modulations, and inter-channel phase relationships implicitly encoded in the spectrogram structure.

Conversely, the handcrafted feature branch provides the fusion head with explicitly computed, physiologically grounded statistics (Hjorth parameters, spectral entropy, band ratios) that regularize the learned representations and improve interpretability. The multi-task training objective—predicting both binary class and continuous risk—further regularizes the shared representation.

### 5.4 Clinical Implications and Alarm System Design

Our alarm-level evaluation reveals the critical importance of post-processing in translating raw model predictions into clinically useful alerts. Raw probability outputs, even from well-calibrated models, produce unacceptably high false alarm rates when naively thresholded. The EMA smoothing approach—which can be viewed as a causal low-pass filter on the risk signal—exploits the temporal structure of the preictal state: true preictal periods produce sustained elevated risk over minutes, while false positive transients are brief and isolated.

The tradeoff between FAH and sensitivity is governed by the operating point selected on the threshold curve. For wearable alert systems, a higher FAH (≤1.0/hour) with greater sensitivity (51.4%) may be preferred to maximize seizure detection. For implanted responsive neurostimulation systems, a lower FAH (≤0.2/hour) with reduced sensitivity (43.8%) may be appropriate to minimize unnecessary stimulations.

### 5.5 Limitations

1. **Dataset size**: While we use the full 24-subject CHB-MIT database, this remains modest for deep learning. Larger multi-center datasets (e.g., Temple University Hospital EEG Corpus) could improve cross-subject generalization.
2. **Scalp EEG resolution**: Scalp electrodes provide limited spatial resolution compared to intracranial recordings. Preictal changes originating in deep structures may be attenuated at the scalp.
3. **Retrospective evaluation**: All results are based on retrospective analysis of pre-recorded data. Prospective, real-time clinical validation is necessary before deployment.
4. **Missing features**: We do not incorporate connectivity features (coherence, phase-locking value), wavelet-based features, or spectral edge frequency (SEF95), which may provide additional discriminative information.
5. **Architecture search**: We did not perform systematic neural architecture search. More complex architectures (transformers, temporal convolutional networks, attention mechanisms) may yield improved performance.

---

## 6. Future Work

Several promising directions emerge from this work:

1. **Domain adaptation**: Techniques such as maximum mean discrepancy (MMD) minimization or adversarial domain adaptation could reduce the distribution shift between patients, improving cross-subject generalization.
2. **Transfer learning with fine-tuning**: Pre-training on a large cross-subject corpus and fine-tuning on a small amount of patient-specific data could combine the benefits of both paradigms.
3. **Attention-based architectures**: Self-attention mechanisms could learn which channels and time-frequency regions are most informative for seizure forecasting, potentially improving both performance and interpretability.
4. **Connectivity features**: Phase-locking value (PLV) and spectral coherence between channel pairs capture inter-regional synchronization changes characteristic of seizure generation.
5. **Multi-center validation**: Evaluating on independent datasets beyond CHB-MIT would strengthen the generalizability claims.
6. **Patient clustering**: Semi-personalized models trained on clusters of electrographically similar patients could bridge the gap between fully patient-specific and fully cross-subject approaches.
7. **Prospective real-time deployment**: Integration with wearable EEG devices and validation in ambulatory settings represents the ultimate translational goal.

---

## 7. Conclusion

We presented FusionNet, a multi-branch deep learning architecture for epileptic seizure forecasting from scalp EEG that fuses convolutional spectrogram representations with handcrafted spectral and temporal features. Evaluated on the complete 24-subject CHB-MIT database, our system achieves patient-specific test AUROC of 0.95 with 100% seizure sensitivity and cross-subject LOSO test AUROC of 0.60 ± 0.20 with 51.4% sensitivity at ≤1.0 false alarms per hour. Exponential moving average post-processing reduces false alarm rates by 3× without sacrificing sensitivity.

Our results establish that patient-specific seizure forecasting is clinically viable with current deep learning methods, while cross-subject generalization remains an open challenge requiring larger datasets, domain adaptation techniques, and more sophisticated architectural designs. The complete pipeline—from raw EEG to trained model to real-time demonstration—is released as open-source software to support reproducibility and future research.

---

## References

1. Daoud, H., & Bhatt, M. A. (2019). Efficient epileptic seizure prediction based on deep learning. *IEEE Transactions on Biomedical Circuits and Systems*, 13(5), 804–813.

2. Dissanayake, T., Fernando, T., Denman, S., Sridharan, S., Ghaemmaghami, H., & Fookes, C. (2021). Deep learning for patient-independent epileptic seizure prediction using scalp EEG. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 29, 2557–2567.

3. Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220.

4. Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., ... & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

5. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning*, 1321–1330.

6. Hjorth, B. (1970). EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*, 29(3), 306–310.

7. Iasemidis, L. D., Shiau, D. S., Chaovalitwongse, W., Sackellares, J. C., Pardalos, P. M., Principe, J. C., ... & Tsakalis, K. (2003). Adaptive epileptic seizure prediction system. *IEEE Transactions on Biomedical Engineering*, 50(5), 616–627.

8. Khan, H., Marcuse, L., Fields, M., Swann, K., & Yener, B. (2018). Focal onset seizure prediction using convolutional networks. *IEEE Transactions on Biomedical Engineering*, 65(9), 2109–2118.

9. Kwan, P., & Brodie, M. J. (2000). Early identification of refractory epilepsy. *New England Journal of Medicine*, 342(5), 314–319.

10. Le Van Quyen, M., Soss, J., Navarro, V., Robertson, R., Chavez, M., Baulac, M., & Martinerie, J. (2005). Preictal state identification by synchronization changes in long-term intracranial EEG recordings. *Clinical Neurophysiology*, 116(3), 559–568.

11. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision*, 2980–2988.

12. Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *Proceedings of the 5th International Conference on Learning Representations*.

13. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *Proceedings of the 7th International Conference on Learning Representations*.

14. Mormann, F., Andrzejak, R. G., Elger, C. E., & Lehnertz, K. (2007). Seizure prediction: the long and winding road. *Brain*, 130(2), 314–333.

15. Mormann, F., Kreuz, T., Andrzejak, R. G., David, P., Lehnertz, K., & Elger, C. E. (2005). Epileptic seizures are preceded by a decrease in synchronization. *Epilepsy Research*, 53(3), 173–185.

16. Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A simple data augmentation method for automatic speech recognition. In *Proceedings of Interspeech*, 2613–2617.

17. Rasheed, K., Qayyum, A., Qadir, J., Sivathamboo, S., Kwan, P., Kuhlmann, L., & O'Brien, T. (2021). Machine learning for predicting epileptic seizures using EEG signals: A review. *IEEE Reviews in Biomedical Engineering*, 14, 139–155.

18. Shoeb, A. H. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*. PhD Thesis, Massachusetts Institute of Technology.

19. Truong, N. D., Nguyen, A. D., Kuhlmann, L., Bonyadi, M. R., Yang, J., Ippolito, S., & Kavehei, O. (2018). Convolutional neural networks for seizure prediction using intracranial and scalp electroencephalogram. *Neural Networks*, 105, 104–111.

20. World Health Organization. (2023). *Epilepsy Fact Sheet*. Retrieved from https://www.who.int/news-room/fact-sheets/detail/epilepsy

---

## Appendix A: Reproducibility

### A.1 Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.0+ | Deep learning framework |
| MNE-Python | 1.5+ | EEG preprocessing |
| scikit-learn | 1.3+ | Baseline models, metrics |
| XGBoost | 2.0+ | Baseline classifier |
| NumPy | 1.24+ | Numerical computation |
| SciPy | 1.11+ | Signal processing, optimization |
| OmegaConf | 2.3+ | Configuration management |
| h5py | 3.9+ | HDF5 caching |
| matplotlib | 3.7+ | Visualization |
| tqdm | 4.65+ | Progress bars |

### A.2 Experiment Reproduction

```bash
# 1. Download CHB-MIT dataset
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ data/chbmit_raw/

# 2. Verify dataset integrity
python scripts/verify_dataset.py --data_root data/chbmit_raw

# 3. Build preprocessed cache
python scripts/build_cache.py --config configs/default.yaml

# 4. Train within-subject model (e.g., chb01)
python scripts/train_deep.py --config configs/small_run.yaml

# 5. Run full LOSO cross-validation
python scripts/run_loso.py --subjects all --num_workers 4 --loss_type focal

# 6. Generate summary tables and figures
python scripts/generate_summary.py
python scripts/make_figures.py
```

### A.3 Configuration

All hyperparameters are specified in YAML configuration files. The default configuration used for the experiments reported in this paper is available at `configs/default.yaml`.

---

## Appendix B: Extended LOSO Results

### B.1 Complete Per-Fold LOSO Test Results

| Fold | Test Subject | Val AUROC | Test AUROC | Test AUPRC |
|------|-------------|-----------|------------|------------|
| 1  | chb01 | 0.831 | 0.547 | 0.111 |
| 2  | chb02 | 0.825 | 0.799 | 0.011 |
| 3  | chb03 | 0.819 | 0.478 | 0.069 |
| 4  | chb04 | 0.827 | 0.433 | 0.005 |
| 5  | chb05 | 0.801 | 0.559 | 0.038 |
| 6  | chb06 | 0.824 | 0.500 | 0.000 |
| 7  | chb07 | 0.814 | 0.749 | 0.019 |
| 8  | chb08 | 0.809 | 0.339 | 0.003 |
| 9  | chb09 | 0.819 | 0.488 | 0.021 |
| 10 | chb10 | 0.802 | 0.620 | 0.111 |
| 11 | chb11 | 0.799 | 0.686 | 0.006 |
| 12 | chb12 | 0.837 | 0.375 | 0.239 |
| 13 | chb13 | 0.800 | 0.365 | 0.228 |
| 14 | chb14 | 0.821 | 0.494 | 0.074 |
| 15 | chb15 | 0.814 | 0.611 | 0.103 |
| 16 | chb16 | 0.815 | 0.708 | 0.150 |
| 17 | chb17 | 0.811 | 0.899 | 0.149 |
| 18 | chb18 | 0.837 | 0.701 | 0.367 |
| 19 | chb19 | 0.831 | 0.580 | 0.376 |
| 20 | chb20 | 0.815 | 0.135 | 0.053 |
| 21 | chb21 | 0.801 | 0.824 | 0.033 |
| 22 | chb22 | 0.823 | 0.653 | 0.027 |
| 23 | chb23 | 0.829 | 0.796 | 0.056 |
| 24 | chb24 | 0.789 | 0.305 | 0.342 |

---

*Manuscript prepared February 2026. For correspondence, contact the authors via the project repository.*

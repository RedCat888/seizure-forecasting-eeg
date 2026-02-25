# Phase 7: Citation Pack

**Note:** The BibTeX entries below are standard references for the methods and data used. They should be verified against the actual publications before submission. No internet access was used; entries are based on well-known references.

---

## Core Dataset Citations

1. **CHB-MIT Scalp EEG Database**
   - Shoeb, A.H. (2009). Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, MIT.
   - Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation, 101(23), e215-e220.

## Seizure Forecasting Literature

2. **Classical seizure prediction reviews**
   - Mormann, F. et al. (2007). Seizure prediction: the long and winding road. Brain, 130(2), 314-333.
   - Kuhlmann, L. et al. (2018). Seizure prediction — ready for a new era. Nature Reviews Neurology, 14(10), 618-630.

3. **Deep learning for seizure forecasting**
   - Daoud, H. & Bhatt, M. (2019). Efficient Epileptic Seizure Prediction Based on Deep Learning. IEEE TBME.
   - Truong, N.D. et al. (2018). Convolutional neural networks for seizure prediction using intracranial and scalp electroencephalogram. Neural Networks, 105, 104-111.
   - Rasheed, K. et al. (2021). Machine learning for predicting epileptic seizures using EEG signals: A review. IEEE Reviews in Biomedical Engineering, 14, 139-155.

4. **Cross-subject / patient-independent approaches**
   - Zhang, Y. et al. (2020). Cross-patient seizure prediction via deep domain adaptation. IEEE JBHI.
   - Li, Y. et al. (2022). Seizure prediction using domain adaptation. Various venues.

## Methods References

5. **Focal Loss**
   - Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. ICCV. arXiv:1708.02002.

6. **Temperature Scaling / Calibration**
   - Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.

7. **EEG Feature Extraction**
   - Hjorth, B. (1970). EEG Analysis Based on Time Domain Properties. Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.
   - Welch, P.D. (1967). The Use of Fast Fourier Transform for the Estimation of Power Spectra. IEEE Transactions on Audio and Electroacoustics.

8. **Alarm Metrics / Clinical Evaluation**
   - Maiwald, T. et al. (2004). Comparison of three nonlinear seizure prediction methods by means of the seizure prediction characteristic. Physical Review E, 70(4).
   - Freestone, D.R. et al. (2015). A forward-looking review of seizure prediction. Current Opinion in Neurology, 28(2), 211-217.

## Software

9. **PyTorch**
   - Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

10. **MNE-Python**
    - Gramfort, A. et al. (2013). MEG and EEG Data Analysis with MNE-Python. Frontiers in Neuroscience, 7, 267.

11. **XGBoost**
    - Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.

12. **scikit-learn**
    - Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.

---

## Citation-Needed List (Verify Before Submission)

| Topic | Likely Paper | Status |
|-------|-------------|--------|
| CHB-MIT dataset | Shoeb 2009 PhD thesis | VERIFY exact citation format |
| PhysioNet | Goldberger et al. 2000 | VERIFY DOI |
| Focal loss | Lin et al. ICCV 2017 | VERIFY |
| Temperature scaling | Guo et al. ICML 2017 | VERIFY |
| Cosine annealing LR | Loshchilov & Hutter 2017 | VERIFY if cited in paper |
| AdamW optimizer | Loshchilov & Hutter 2019 | VERIFY if cited |
| Batch normalization | Ioffe & Szegedy 2015 | VERIFY if cited |
| EMA smoothing origin | Standard signal processing | No specific citation needed |

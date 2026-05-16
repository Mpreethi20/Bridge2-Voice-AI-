
# 🎙️ Voice Disorder Screening using Acoustic & Psychosocial Features

> An interpretable machine learning screening model for voice disorders using acoustic biomarkers and Voice Handicap Index (VHI) psychosocial features with XGBoost and SHAP explainability.

# 📌 Project Overview

This project explored the development of an interpretable AI-based screening system for voice disorders using both:

- Objective acoustic voice biomarkers
- Subjective psychosocial Voice Handicap Index (VHI-10) measures

The study was built on the **biopsychosocial framework**, recognizing that voice disorders affect not only vocal acoustics but also emotional well-being, physical effort, and daily functioning.

An XGBoost-based binary classifier was developed using acoustic features extracted from the NIH Bridge2AI Voice Dataset, followed by detailed error analysis of misclassified patients using VHI psychosocial subscales.

---

# 🎯 Objectives

- Build an interpretable machine learning screening model for voice disorders
- Predict voice disorder status using acoustic voice biomarkers
- Analyze model errors using psychosocial VHI-10 scores
- Investigate the disconnect between acoustic abnormalities and patient-reported voice impact
- Preserve clinical interpretability by excluding MFCC-based black-box features

---

# 📂 Dataset

- **Dataset:** NIH Bridge2AI Voice Dataset
- **Hospitals:** 9 clinical sites
- **Sample Size:** 179 patients
- **Train/Test Split:** 80% / 20%

### Class Distribution

| Group | Count |
|-------|------|
| Voice Disorder | 79 |
| Control | 97 |

---

# 👥 Patient Demographics

| Demographic | Value |
|-------------|------|
| Mean Age | 56.1 ± 18.3 years |
| Female | 63.1% |
| Male | 34.1% |
| Non-binary | 2.8% |

> Dataset statistics were derived from project analysis materials and should be validated against final source data before publication.

---

# ⚙️ Feature Engineering

## 🎤 Acoustic Features

Acoustic biomarkers were extracted using **openSMILE**.

### Key Acoustic Features
- `logRelF0-H1-H2_sma3nz_stddevNorm`
- `shimmerLocaldB_sma3nz_stddevNorm`
- `slopeUV0-500_sma3nz_amean`
- `alphaRatioUV_sma3nz_amean`
- `F2bandwidth_sma3nz_amean`
- `CPP (Cepstral Peak Prominence)`

These features captured:
- Frequency variation
- Spectral characteristics
- Voice stability
- Loudness dynamics
- Harmonic voice quality

---

# 🧠 Psychosocial Features

Patient-reported Voice Handicap Index (VHI-10) subscales included:

| Subscale | Description |
|----------|-------------|
| Emotional (E_score) | Emotional impact of voice symptoms |
| Physical (P_score) | Physical discomfort and strain |
| Functional (F_score) | Impact on daily communication |

---

# 🔍 Clinical Interpretability Design

### Why MFCCs Were Excluded

Mel-Frequency Cepstral Coefficients (MFCCs) were intentionally excluded from the final feature set to improve:

- Clinical transparency
- Feature interpretability
- Physician trust
- Explainable AI capability

Instead, acoustically meaningful voice biomarkers such as CPP and shimmer were prioritized.

---

# 🤖 Machine Learning Model

## XGBoost Binary Classifier

### Validation Strategy
- 3-Fold Stratified Cross-Validation
- Grid Search Hyperparameter Optimization

### Performance Metrics

| Metric | Value |
|--------|------|
| Accuracy | 83.33% |
| Precision | 0.854 |
| Recall | 0.819 |
| F1 Score | 0.786 |

---

# 📊 Explainability with SHAP

SHAP (SHapley Additive Explanations) was used to interpret feature importance and model decisions.

### Top Interpretable Feature
```python
logRelF0-H1-H2_sma3nz_stddevNorm
```

### Additional Important Predictors
- CPP
- Shimmer variability
- Spectral slope
- Formant bandwidth features

SHAP analysis helped identify which acoustic biomarkers most strongly influenced predictions.

---

# 🔬 Error Analysis — Biopsychosocial Framework

A detailed analysis was conducted on model misclassifications.

## Misclassified Patients
- False Positives (FP): 18
- False Negatives (FN): 10

---

# 📈 VHI-10 Error Analysis Results

| VHI Subscale | False Positives | False Negatives | p-value |
|--------------|----------------|----------------|---------|
| Total Score | 15.05 ± 10.23 | 7.5 ± 5.01 | 0.01 |
| Emotional | 2.27 ± 2.27 | 1.3 ± 1.63 | 0.12 |
| Physical | 5.44 ± 3.46 | 2.1 ± 2.02 | 0.002 |
| Functional | 7.33 ± 5.23 | 4.1 ± 3.84 | 0.003 |

---

# 🩺 Clinical Interpretation

## False Positives
Patients acoustically resembled voice disorder cases despite lacking formal diagnosis.

These patients showed:
- Higher physical VHI scores
- Higher functional impairment
- Potential early-stage or undiagnosed voice disorders

This suggests the model may identify clinically relevant voice impairment before formal diagnosis.

---

## False Negatives
Patients diagnosed with voice disorders but acoustically appearing normal.

These patients demonstrated:
- Lower psychosocial impact
- Mild self-reported symptoms
- Possible disconnect between acoustic abnormality and perceived impairment

---

# 📊 CPP (Cepstral Peak Prominence) Analysis

CPP distributions were analyzed across prediction groups.

| Group | Mean CPP |
|-------|----------|
| True Negatives | 14.92 |
| False Positives | 15.77 |
| False Negatives | 15.84 |
| True Positives | 15.12 |

CPP emerged as an important clinically interpretable biomarker for voice disorder screening.

---

# 🔭 Key Findings

- Acoustic biomarkers alone may not fully represent patient-perceived voice impact
- False positives may reflect clinically meaningful but undiagnosed dysfunction
- Explainable acoustic features improved clinical interpretability
- Combining objective acoustics with psychosocial context improves screening insight
- AI models can support earlier identification of patients needing clinical evaluation

---

# 🚀 Future Directions

- Validate findings on larger multi-site datasets
- Integrate VHI directly into model training
- Compare Random Forest, SVM, and deep learning approaches
- Explore multimodal AI using speech + text + psychosocial data
- Investigate treatment-seeking behavior as a predictive feature

---

# 🛠️ Tech Stack

- Python
- XGBoost
- openSMILE
- SHAP
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# 📁 Project Structure

```bash
voice-disorder-screening/
│
├── data/
│   └── bridge2ai_voice_dataset/
│
├── notebooks/
│   ├── 01_EDA_Demographics.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_XGBoost_Model.ipynb
│   └── 04_Error_Analysis_VHI.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── shap_summary_plot.png
│   ├── vhi_scores_by_group.png
│   └── cpp_histograms.png
│
├── requirements.txt
└── README.md
```

---

# 📄 Publication

### AMIA 2025 Annual Symposium

**Paper Title:**  
*"Mixed Feelings About My Voice: Analyzing Errors in a Voice Disorders Screening Model using the Biopsychosocial Framework"*


# 📚 References

1. Stemple JC, Roy N, Klaben BG. *Clinical Voice Pathology: Theory and Management.*
2. Jacobson BH et al. *The Voice Handicap Index (VHI).* AJSLP, 1997.
3. Timmons Sund L et al. *VHI-10 Scores in Treatment-Seeking Dysphonia.* Journal of Voice, 2023.

---

# 📄 License & Data Access

This repository is intended for academic and research purposes only.

The NIH Bridge2AI Voice Dataset is not publicly included in this repository.

Please refer to the Bridge2AI initiative for dataset access:
https://bridge2ai.org/

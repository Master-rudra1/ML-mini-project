# Automated Identification of Gait Abnormalities

[[https://img.shields.io/badge/python-3.8+-blue.svg]](https://www.python.org/downloads/)
[[https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg]](https://pytorch.org/)
[[https://img.shields.io/badge/License-MIT-yellow.svg]](https://opensource.org/licenses/MIT)
[[https://img.shields.io/badge/Course-UE24CS352A%20Machine%20Learning-brightgreen.svg]]()

An end-to-end Machine Learning and Deep Learning pipeline for the automated classification of neurodegenerative gait pathologies (**Healthy Control**, **Parkinson's Disease**, **Huntington's Disease**, and **Amyotrophic Lateral Sclerosis [ALS]**) using clinical in-shoe ground reaction force sensor recordings.

---

## 1. Project Overview & Clinical Problem Statement

Human locomotion is regulated by complex neuromuscular pathways. Neurological disorders alter gait dynamics in distinctive patterns long before gross macroscopic symptoms become pronounced:
- **Parkinson's Disease (PD)**: Characterized by shuffling steps, shortened stride length, elevated cadence (festination), and reduced foot clearance.
- **Huntington's Disease (HD)**: Involuntary choreic movements cause severe stride-to-stride timing instability and high rhythm variability.
- **Amyotrophic Lateral Sclerosis (ALS)**: Progressive unilateral and bilateral motor neuron loss leads to foot drop, prolonged stance phases to avoid falling, and severe bilateral asymmetry.
- **Healthy Controls (CO)**: High rhythmic consistency (CV < 2.5%) and balanced bilateral symmetry.

### Clinical Objective
Traditional clinical gait evaluation relies on subjective visual rating scales (e.g., UPDRS) or expensive multi-camera motion capture laboratories (e.g., optical Vicon systems). This project develops an objective, computational diagnostic pipeline that processes sensor time series, extracts domain-specific biomechanical biomarkers, and performs automated 4-class differential diagnosis.

---

## 2. Dataset Architecture

This project utilizes the **PhysioNet Gait in Neurodegenerative Disease Database (`gaitndd`)** (Dr. Jeffrey Hausdorff et al., Harvard Medical School / MIT / Beth Israel Deaconess Medical Center):
* **Cohort Size**: 64 human subjects
  * 16 Healthy Controls (`control1` – `control16`)
  * 15 Parkinson's Disease patients (`park1` – `park15`)
  * 20 Huntington's Disease patients (`hunt1` – `hunt20`)
  * 13 ALS patients (`als1` – `als13`)
* **Signal Type**: In-shoe force-sensitive resistor (FSR) time-series (`.ts` files) recording consecutive footfall contact intervals over 5-minute continuous walking trials.

---

## 3. Methodology & Feature Engineering

### A. Preprocessing & Artifact Removal
In the clinical trials, patients walked back and forth along a 77-meter hospital corridor. Turning pauses at the end of the hallway introduce artificial 3–4 second delays. We apply:
1. **Physiological Gait Filtering**: Excludes turning stalls and non-physiological sensor outliers (0.60 s <= Stride <= 1.80 s).
2. **Temporal Sliding-Window Segmentation**: Segments walks into overlapping 25-stride windows (10-stride step), capturing dynamic motor changes (freezing, fatigue, limping) across 334+ evaluation windows.

### B. Extracted Biomechanical Biomarkers (8 Features)
1. **`mean_stride`**: Average duration of a full gait cycle (seconds).
2. **`stride_cv`**: Stride Coefficient of Variation (CV% = σ / μ * 100%) measuring step timing regularity.
3. **`cadence`**: Steps per minute (60 / mean_stride).
4. **`stride_asymmetry`**: Bilateral difference between left and right stride times.
5. **`swing_asymmetry`**: Airborne duration difference between limbs (direct measure of limping/foot drop).
6. **`stance_swing_ratio`**: Weight-bearing vs. airborne ratio (reflects balance compensation).
7. **`double_support_mean`**: Duration when both feet touch the floor simultaneously.
8. **`double_support_cv`**: Unsteadiness in the two-foot balance phase.

---

## 4. Machine Learning & Deep Learning Architectures

### A. Random Forest Classifier (Ensemble Baseline)
* **Configuration**: 100 bagged decision trees (`n_estimators=100`, `max_depth=8`).
* Employs bootstrap sampling with feature subset randomization (sqrt(n_features)) and balanced class weighting.

### B. Calibrated Deep Neural Network (`CalibratedGaitNet` in PyTorch)
* **Architecture**:
  * Input Layer: 8 clinical biomarkers
  * Hidden Layer 1: Linear(8 -> 48) -> Batch Normalization -> ReLU -> Dropout (p=0.15)
  * Hidden Layer 2: Linear(48 -> 24) -> Batch Normalization -> ReLU -> Dropout (p=0.10)
  * Output Layer: Linear(24 -> 4 classes: Control, Parkinson's, Huntington's, ALS)
* **Loss Function**: `nn.CrossEntropyLoss` with inverse-frequency class weighting to prevent majority-class bias on rare cohorts (ALS).
* **Optimizer**: Adam (lr=0.005, weight decay=1e-4) on GPU (`cuda`).

### C. Sequential Sequence Models (`src/models_torch.py`)
* **1D-CNN**: 3-block 1D convolutional feature extractor for raw multi-channel temporal sensor streams.
* **BiLSTM**: 2-layer Bidirectional LSTM capturing forward cadence buildup and backward balance recovery context.

---

## 5. Experimental Results

Evaluated on held-out test patients across **334 test segments**:

### Quantitative Metrics Summary
| Metric | Score | Clinical Meaning |
| :--- | :---: | :--- |
| **Test Accuracy** | **78.44%** | Overall diagnostic agreement on unseen patient segments. |
| **Macro Precision** | **78.95%** | High reliability with minimal false positive alarms. |
| **Macro Recall** | **80.19%** | High sensitivity in catching actual neurodegenerative cases. |
| **Macro F1-Score** | **79.05%** | Balanced harmonic mean across common and rare classes. |

### Per-Class Diagnostic Performance
| Condition | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **ALS** | **73.91%** | **94.44%** | **82.93%** | 54 |
| **Healthy Control** | **78.57%** | **81.91%** | **80.21%** | 94 |
| **Huntington's (HD)** | **73.47%** | **70.59%** | **72.00%** | 102 |
| **Parkinson's (PD)** | **89.86%** | **73.81%** | **81.05%** | 84 |
| **Overall / Macro** | **78.95%** | **80.19%** | **79.05%** | **334** |

### Key Clinical Observations
* **Highest Sensitivity on ALS (94.44% Recall)**: Caught 51 out of 54 segments due to distinct unilateral foot-drop asymmetry.
* **High Precision on Parkinson's (89.86%)**: When the model flags Parkinson's, it is nearly 90% reliable.
* **Elderly vs. Chorea Overlap**: Normal age-related stride slowing and mild Huntington's chorea share natural overlap in timing, consistent with published medical literature.

---

## 6. Repository Structure

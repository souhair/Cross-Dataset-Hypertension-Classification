# Cross-Dataset Evaluation of Interpretable Photoplethysmography Features for Blood Pressure Category Classification

**Official implementation** of:

> **Cross-Dataset Evaluation of Interpretable Photoplethysmography Features for Blood Pressure Category Classification**  
> Souhair Moskar and Roman Davydov  
> Higher School of Cyber-Physical Systems & Control, Peter the Great St. Petersburg Polytechnic University

[![Release](https://img.shields.io/badge/release-v2.1--submission-blue)](../../releases/tag/v2.1-submission)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of contents

- [Overview](#overview)
- [Headline results](#headline-results)
- [Datasets](#datasets)
- [Installation](#installation)
- [Running the pipeline](#running-the-pipeline)
  - [Phase 0: Verify installation](#phase-0-verify-installation)
  - [Phase 1: Full extraction](#phase-1-full-extraction)
  - [Phase 2: Analysis](#phase-2-analysis)
  - [Phase 3: Figures](#phase-3-figures)
  - [Phase 4: Audits and diagnostics](#phase-4-audits-and-diagnostics)
- [Configuration](#configuration)
- [Output structure](#output-structure)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

## Overview

This repository contains the full pipeline for three-class JNC7 blood pressure category classification (Normal / PreHTN / HTN) from short, single-channel photoplethysmography (PPG) segments. The experiments evaluate four independent cohorts: Aurora-BP, PPG-BP, PulseDB-Vital, and SWCS.

The pipeline is:

- **Interpretable** — every feature has a defined physiological meaning, and SHAP values map each feature's contribution to the model's prediction.
- **Sampling-rate canonical** — the extractor resamples every input to 125 Hz at its entry point. A paired 500 Hz vs. 125 Hz audit on 460 Aurora-BP subjects found agreement across all 32 audited features and 14,720 feature–subject pairs, with a ratio of **1.0000000000 ± 0.0000000000**.
- **Cross-dataset validated** — trained on Aurora-BP and evaluated on PPG-BP, PulseDB-Vital, and SWCS using subject-grouped splits.
- **Leakage-controlled** — uses subject-grouped splitting, fold-aware preprocessing, subject-cluster bootstrap confidence intervals, and leakage diagnostics for subject overlap, random-label control, and single-feature scans.

## Headline results

### Aurora-BP: three-class JNC7 classification

*Subject-grouped 5-fold cross-validation.*

| Model | Accuracy | Macro F1 | Macro OvR AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.577 | 0.576 | 0.765 ± 0.014 |
| Random Forest | 0.583 | 0.583 | 0.763 ± 0.012 |
| **XGBoost** | **0.595** | **0.599** | **0.778 ± 0.011** |
| Best NN (ResidualMLP) | 0.591 | 0.586 | 0.777 ± 0.013 |

The 95% subject-cluster bootstrap confidence interval for XGBoost AUC is **[0.762, 0.794]**. Per-class AUC: Normal **0.837**, PreHTN **0.644**, HTN **0.852**.

### Binary HTN classification

*Aurora-BP, ACC/AHA 2017.*

| Protocol | AUC |
|---|---:|
| Segment-level | 0.820 |
| **Subject-aggregated** | **0.841** |
| Subject-mean-feature | 0.816 |

### Feature-family ablation

Demographics is the only family whose removal materially degrades performance (**ΔAUC = −0.068**). The PPG-only model, without demographics, achieves **AUC 0.704 [0.688, 0.719]**.

### SHAP analysis

Demographics dominates by 6× (mean absolute SHAP value **0.182**, compared with **0.029** for the next-closest family). The top individual PPG feature, `period_median__compliance_index` (**0.232**), ranks above every demographic variable except weight.

### External validation

*5-fold `StratifiedGroupKFold`.*

| Dataset | Best strategy | AUC |
|---|---|---:|
| PPG-BP | Warm-start | 0.735 ± 0.070 |
| PulseDB-Vital | From-scratch | 0.690 ± 0.025 |
| SWCS | Warm-start | 0.736 ± 0.032 |

**Few-shot transfer:** on SWCS with 8 training subjects, warm-start pretraining improves AUC by **+0.311**.

---

## Datasets

The pipeline uses four public datasets. **The datasets are not redistributed in this repository.** Download them from their original sources and place them in the expected local paths.

| Dataset | Subjects | Segments | Native sampling rate | Source |
|---|---:|---:|---:|---|
| **Aurora-BP** | 656 | 7,329 | 500 Hz | [PhysioNet](https://physionet.org/content/aurora-bp/) |
| **PPG-BP** | 218 | 218 | 1000 Hz | [Figshare](https://doi.org/10.6084/m9.figshare.c.4040847) |
| **PulseDB-Vital** | 250 | 7,673 | 125 Hz | [PulseDB](https://pulsedb.readthedocs.io/) |
| **SWCS** | 115 | 2,554 | 50 Hz | [IEEE Dataport](https://dx.doi.org/10.21227/3yte-wz05) |

**Dataset naming note:** in the codebase, the SWCS dataset is named `DS4`. Throughout the paper and this README, **SWCS** and **DS4** refer to the same dataset.

### Aurora-BP directory layout

Adjust the `ROOT` constant in `runners/run_aurora_full_v2.py` if your local path differs.

```text
<Aurora root>/
├── participants.tsv
├── measurements_auscultatory.tsv
└── measurements_auscultatory/
    ├── a000/
    │   └── *.tsv
    └── ...
```

For the other datasets, update the corresponding entries in the `CSV` dictionary in `analysis/v2_utils.py`.

---

## Installation

### Requirements

- Python **3.10 or newer**
- Optional: CUDA-capable GPU for `v2_nn_sweep.py`. CPU execution is supported but may be approximately 3× slower.

### Setup

```bash
git clone https://github.com/souhair/Cross-Dataset-Hypertension-Classification.git
cd Cross-Dataset-Hypertension-Classification

python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Key dependencies

The project uses the following main packages (the complete environment is defined in `requirements.txt`):

```text
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.1
xgboost>=2.0
neurokit2>=0.2.7
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
h5py>=3.8
openpyxl>=3.1
torch>=2.0        # optional; required only for v2_nn_sweep.py
antropy>=0.1.5    # complexity features
```

PyTorch is only required for the neural-network sweep. The other scripts can run on CPU without it.

---

## Running the pipeline

Run commands from the repository root with the project virtual environment activated. The durations below are approximate and depend on the machine and dataset location.

### Phase 0: Verify installation

```bash
python -c "from config import *; print('config OK')"

python -c "from extractors.ppg_feature_extractor import SEGMENT_FEATURE_ORDER; assert len(SEGMENT_FEATURE_ORDER) == len(set(SEGMENT_FEATURE_ORDER)); print(f'Feature list clean: {len(SEGMENT_FEATURE_ORDER)} unique names')"

python -c "from extractors.ppg_feature_extractor import test_join_mask_filtering; test_join_mask_filtering()"

python -c "from extractors.ppg_feature_extractor import process_single_segment_unified; import inspect; src = inspect.getsource(process_single_segment_unified); assert 'CANONICAL_FS' in src; print('canonicalization present')"
```

Expected output:

```text
config OK
Feature list clean: 164 unique names
[join_mask_v2] test passed: 6 retained of 7 unmasked
canonicalization present
```

### Phase 1: Full extraction

*Approximate duration: 2–3 hours.*

```bash
python runners/run_all_extraction.py
```

The extraction runner executes the following steps sequentially:

| # | Script | Approx. time | Main output / purpose |
|---:|---|---:|---|
| 1 | `run_aurora_full_v2.py` | ~2 h | `out_v2/aurora_unified_v2_fixed.csv` |
| 2 | `run_ppgbp_full_v2.py` | ~2 min | `out_v2/ppgbp_unified_v2_fixed.csv` |
| 3 | `run_pulsedb_vital_full_v2.py` | ~30 min | `out_v2/pulsedb_vital_unified_v2_fixed.csv` |
| 4 | `run_ds4_full_v2.py` | ~30 min | `out_v2/ds4_unified_v2_fixed.csv` |
| 5 | `v2_fix_subject_ids.py` | Instant | Subject ID recovery (`*_fixed.csv`) |
| 6 | `verify_ids.py` | Instant | Integrity check |

Physio-PPG is **not** part of the paper's four cohorts and is not included by default. To opt in to its auxiliary extraction, set the environment variable before running the extraction pipeline:

```powershell
# Windows PowerShell
$env:INCLUDE_PHYSIO_PPG = "1"
```

```bash
# Linux / macOS
export INCLUDE_PHYSIO_PPG=1
```

Expected final counts:

| Dataset | Rows / segments | Subjects | Zero-ID rows |
|---|---:|---:|---:|
| Aurora-BP | 7,329 | 656 | 0 |
| PPG-BP | 218 | 218 | 0 |
| PulseDB-Vital | 7,673 | 250 | 0 |
| SWCS / DS4 | 2,554 | 115 | 0 |


### Phase 2: Analysis

*Approximate duration: 1–2 hours.*

```bash
python analysis/run_all_analysis.py
python analysis/postprocess_fraction_csv.py
```

The analysis pipeline runs 11 scripts. Approximate total time is 1 hour 30 minutes on CPU.

| # | Script | Approx. time | Main output |
|---:|---|---:|---|
| 1 | `v2_3class.py` | ~9 min | `v2_3class_summary.csv` |
| 2 | `v2_ablation.py` | ~17 min | `v2_ablation_corrected.csv` |
| 3 | `v2_no_demographics.py` | ~8 min | `v2_no_demographics_corrected.csv` |
| 4 | `v2_shap_analysis.py` | ~30 sec | `v2_shap_family_summary.csv`, `v2_shap_per_class.npy` |
| 5 | `v2_figures.py` | ~2 min | Figures 1, 2, 3, and 5 |
| 6 | `v2_nn_sweep.py` | ~16 min | `v2_nn_sweep_summary.csv` |
| 7 | `v2_transfer_strategies.py` | ~5 min | `v2_transfer_strategies_corrected.csv` |
| 8 | `v2_external_cv_stratified.py` | ~3 min | `v2_external_cv_stratified_corrected.csv` |
| 9 | `v2_fraction_experiment.py` | ~1 min | `v2_fraction_experiment_corrected.csv` |
| 10 | `v2_optional_analyses.py` | ~3 min | `v2_calibration_corrected.csv` |
| 11 | `v2_threshold_tuning.py` | ~30 min | `v2_threshold_tuning_corrected.csv` |

### Phase 3: Figures

*Approximate duration: 2 minutes.*

```bash
python figures/build_missing_figures.py       # Figures 8, 12, 13
python figures/build_fig4_and_fig6.py         # Figures 4, 6
python figures/build_fig7_from_saved_shap.py  # Figure 7
python figures/build_fig9_only.py             # Figure 9
python figures/rebuild_fig6_and_fig11.py      # Figure 11
python figures/v2_two_path_fixed.py           # Table 9 data
```

The output directory is `out_v2/results/figures/`. The figure workflow is intended to produce 14 paper figures.

### Phase 4: Audits and diagnostics

*Approximate duration: 15 minutes.*

```bash
# Reproducibility checks
python verification/print_family_counts.py
python verification/verify_ids.py

# Diagnostics
python diagnostics/verify_bp_labels.py
python diagnostics/diagnose_pulsedb_discrepancy.py
python diagnostics/check_complexity_trigger.py

# Independent audits
python audits/sampling_rate_audit_fresh.py
python audits/binary_auc_protocols.py
python audits/v2_leakage.py
python audits/polarity_full_test.py
python audits/audit_design_choices.py
python audits/audit_scaler_comparison.py
```

The scripts check, respectively, feature-family counts, extraction ID integrity, BP label integrity, the expected PulseDB protocol difference, the complexity-feature decimation branch, sampling-rate consistency, binary AUC protocols, leakage diagnostics, Aurora polarity, feature-extraction design choices, and scaler sensitivity.

---

## Configuration

Paths, thresholds, and constants are defined in `config.py`.

```python
CANONICAL_FS = 125        # Target sampling rate for all extraction
SEED = 42                 # Global random seed
PERIOD_P_DEFAULT = 2.0    # SMD period parameter
N_FOLDS = 5               # Number of cross-validation folds
NAN_THRESH = 0.5          # Maximum fraction of NaN values per feature
CORR_THRESH = 0.95        # Correlation threshold for de-duplication
```

If your local dataset paths differ, adjust `ROOT` in `runners/run_aurora_full_v2.py` and the `CSV` dictionary in `analysis/v2_utils.py`.

---

## Output structure

After a successful run, the output directory should contain files similar to the following. Some files are created only by particular analysis steps.

```text
out_v2/
├── aurora_unified_v2_fixed.csv
├── aurora_unified_v2_periods.csv
├── aurora_unified_v2_sequences.h5
├── ppgbp_unified_v2_fixed.csv
├── pulsedb_vital_unified_v2_fixed.csv
├── ds4_unified_v2_fixed.csv
└── results/
    ├── v2_3class_summary.csv
    ├── v2_ablation_corrected.csv
    ├── v2_no_demographics_corrected.csv
    ├── v2_shap_family_summary.csv
    ├── v2_shap_per_class.npy
    ├── v2_feature_names.txt
    ├── v2_nn_sweep_summary.csv
    ├── v2_nn_sweep_predictions.npz
    ├── v2_transfer_strategies_corrected.csv
    ├── v2_external_cv_stratified_corrected.csv
    ├── v2_fraction_experiment_corrected.csv
    ├── v2_calibration_corrected.csv
    ├── v2_threshold_tuning_corrected.csv
    ├── v2_binary_auc_protocols.csv
    ├── v2_sampling_rate_audit_fresh.csv
    ├── v2_leakage_single_feature.csv
    ├── v2_scaler_comparison.csv
    ├── v2_two_path_corrected.csv
    ├── v2_oof_3class_*.csv
    └── figures/
        ├── fig1_confusion_aurora.png
        ├── fig2_per_class_aurora.png
        ├── fig3_roc_aurora.png
        ├── fig5_shap_top25.png
        ├── fig6_feature_family_shap.png
        ├── fig7_per_class_shap.png
        ├── fig8_nn_comparison.png
        ├── fig9_per_class_roc_nn.png
        ├── fig9_calibration.png
        ├── fig10_decision_curve.png
        ├── fig12_transfer_heatmap.png
        └── fig13_fraction_curves.png
```
The out2 file of the project: https://drive.google.com/drive/folders/1LGbCB3e6CmVCc6V-DRj_teWX39F7QbUc?usp=drive_link
---

## Reproducibility notes

### Seeds

All analysis scripts set the NumPy, `random`, and `PYTHONHASHSEED` seeds to 42, as well as the PyTorch seed when PyTorch is available. Re-running an analysis script with the same seed, on the same machine and NumPy version, produces identical output. Cross-machine reproducibility is not guaranteed because of hardware-dependent floating-point behavior and library version differences.

### Split protocols

Two evaluation protocols are used deliberately:

- **5-fold `StratifiedGroupKFold`:** used for model comparison, ablation, SHAP, calibration, decision-curve analysis (DCA), transfer strategies, and threshold tuning. Subject ID is the grouping variable.
- **Single 70/30 stratified subject-level split:** used only for the fraction sweep, because fair comparison of pretrained and scratch models requires a fixed held-out test set.

These protocols produce slightly different absolute AUC values for the same nominal strategy. Both are documented in Section 3.9 of the paper.

### Extractor canonicalization

`process_single_segment_unified` resamples all inputs to 125 Hz at its entry point via polyphase FIR filtering. This is a no-op for the production pipeline because all adapters pass `fs=125`. The behavior is verified by `audits/sampling_rate_audit_fresh.py`: 14,720 feature–subject pairs, with a ratio of **1.0000000000 ± 0.0000000000**.

### Feature extraction

- **437** candidate features are extracted per segment.
- **213** features remain after correlation filtering (`|r| > 0.95`).
- Features are assigned to **9 mutually exclusive families** by `family_of()` in `feature_families.py`.

### PPG-BP extraction

Three 2.1-second clips per subject are concatenated into a single recording of approximately 6.3 seconds. Pulse periods whose foot-to-foot interval strictly contains a concatenation position are discarded. After the fix, **218 of 219 subjects** produce valid segments.

### Warm-start fine-tuning

Warm-start uses the source scaler to transform target features, preserving the coordinate system of the source booster's split thresholds. The from-scratch and target-scaler branches use a target-refit scaler. All warm-start folds succeeded in the reported runs (`continuation_ok_rate = 1.0` for every dataset).

---
## Known limitations

- **HRV frequency-domain handling:** for short segments (below the duration recommended by NeuroKit2 for reliable frequency-domain estimation), the current implementation returns zero-valued HRV features rather than missing values. HRV contributes only 0.095 to the total family SHAP attribution, so this convention does not materially affect the reported AUCs. However, HRV features should not be interpreted as informative for short recordings. A code-level fix is deferred to future work.
- **Transformer order-insensitivity:** the model labelled `attention-pooling` in Section 3.8 contains no positional encoding and is invariant to the temporal order of input periods. Only the BiLSTM in the flat-versus-sequence comparison encodes order.
- **Sequence normalization:** imputed and padded entries are set to exactly zero in standardized coordinates. Only finite, valid entries contribute to normalization statistics.
- **Sampling-rate audit scope:** the audit covers 32 features spanning five of the nine families (SMD, period morphology, spectral + nonlinear, morphology, and HRV) for 460 Aurora-BP subjects at 500 Hz versus 125 Hz. It does not establish invariance for arbitrary sampling rates or for feature families outside the audited set.

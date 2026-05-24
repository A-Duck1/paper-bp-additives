# Machine Learning-Guided Screening of B/P Bifunctional Additives for SEI/CEI Regulation in Li-Ion Batteries

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This repository contains the code, data, and manuscript for the project:

**"Machine Learning-Guided Screening of Boron/Phosphorus-Containing Bifunctional Additives for Synergistic SEI/CEI Regulation in Lithium-Ion Batteries"**

Target journal: **Digital Discovery** (RSC)

### Core Innovation
B+P dual-functional electrolyte additives — Boron stabilizes SEI (anode), Phosphorus stabilizes CEI (cathode) — screened via ML + DFT high-throughput workflow.

## 🧪 Workflow

```
Data Curation → Feature Engineering → ML Training → Virtual Screening → SHAP Analysis → DFT Verification
```

## 📁 Repository Structure

```
.
├── data/                      # Training & screening data
│   ├── ml_results/            # ML outputs (model metrics, screening results, SHAP)
│   ├── training_additives_v2.csv
│   └── candidates_v2.csv
├── scripts/                   # All Python scripts
│   ├── train_v4.py            # RF training (n=500 trees)
│   ├── screen_v5.py           # RF+XGBoost virtual screening
│   ├── shap_analysis.py       # SHAP feature importance
│   ├── xgb_loocv.py           # XGBoost LOOCV validation
│   ├── data_expansion_v3.py   # Candidate pool expansion
│   ├── generate_figures_v2.py # Figure generation
│   └── generate_paper_v3.py   # Manuscript generation
├── figures/                   # All figures (PDF+PNG)
│   ├── fig1_homo_lumo_v2.*
│   ├── fig2_feature_importance_v2.*
│   ├── fig3_model_performance_v2.*
│   ├── fig4_feature_comparison_v2.*
│   ├── fig5_performance_summary_v2.*
│   └── fig_shap_*.*          # SHAP beeswarm + bar plots
├── manuscript/                # Manuscript DOCX
│   ├── manuscript.docx        # v1 (initial)
│   ├── manuscript_v2.docx     # v2 (data corrected)
│   ├── manuscript_v3.docx     # v3 (LOOCV added)
│   └── manuscript_v4.docx     # v4 (SHAP chapter added)
├── papers/                    # Literature database
├── README.md                  # This file
└── .gitignore
```

## 🚀 Key Results

### Model Performance (LOOCV)
| Target | RF R² (train) | RF MAE | RF LOOCV R² (Morgan) |
|--------|-------------|--------|---------------------|
| HOMO | 0.797 | 0.219 eV | 0.567 |
| LUMO | 0.783 | 0.240 eV | 0.435 |
| Gap | 0.806 | 0.281 eV | 0.622 |

### Candidate Pool
- **14,425** B/P-containing candidates (MW < 600 Da)
- 152+ chemical scaffolds, 48 functional groups
- B: 5,093 | P: 6,579 | B+P: 2,497

### Virtual Screening (RF-only, Top 5)
| Rank | SMILES | Score | Type |
|------|--------|-------|------|
| 1 | O=P(O)(O)CP(=O)(O)O | 0.821 | P (pyrophosphate) |
| 2 | O=P(O)(O)OP(=O)(O)O | 0.780 | P (diphosphate) |
| 3 | CP(C)C.O=P(O)(O)OP(=O)(O)O | 0.768 | P |
| 4 | CP(C)(C)=O.O=P(O)(O)OP(=O)(O)O | 0.763 | P |
| 5 | C[Si]1(C)O[Si](C)(C)O1.O=P(O)(O)OP(=O)(O)O | 0.760 | P |

## 🛠 Dependencies

- Python 3.13+
- RDKit 2024+
- scikit-learn 1.8+
- xgboost 3.2+
- matplotlib 3.10+
- pandas 2.3+
- seaborn 0.13+
- python-docx (for manuscript generation)
- shap (for feature importance analysis)

## 📄 License

MIT

## 📬 Contact

For questions about this repository, please open an issue.

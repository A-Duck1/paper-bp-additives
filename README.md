# Machine Learning-Guided Screening of B/P Bifunctional Additives for SEI/CEI Regulation in Li-Ion Batteries

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)

> **Latest version:** v5 — Final data version (100 training samples, 14,425 candidates) (2026-05-24)

## 📋 Overview

This repository contains the code, data, and manuscript for the project:

**"Machine Learning-Guided Screening of Boron/Phosphorus-Containing Bifunctional Additives for Synergistic SEI/CEI Regulation in Lithium-Ion Batteries"**

Target journal: **Digital Discovery** (RSC)

### Core Innovation
B+P dual-functional electrolyte additives — Boron stabilizes SEI (anode), Phosphorus stabilizes CEI (cathode) — screened via ML + DFT high-throughput workflow.

## 🧪 Workflow

```
Data Curation → Feature Engineering → ML Training → Virtual Screening → SHAP Analysis → DFT Verification → GNN Comparison
```

## 📁 Repository Structure

```
.
├── data/                          # Training & screening data
│   ├── ml_results/                # ML outputs (screening, SHAP, GNN, LOOCV metrics)
│   │   ├── gnn_v2_results.json    # GNN benchmark results (GIN, GCN, GAT)
│   │   ├── screening_rf_v7_full.csv
│   │   ├── screening_rf_v7_top200.csv
│   │   ├── screening_v5_full.csv
│   │   ├── screening_v5_top200.csv
│   │   ├── shap_analysis.json
│   │   ├── train_metrics_v5.json
│   │   └── xgb_loocv_results.json
│   ├── candidate_features_v2.csv  # Feature vectors for 14K+ candidates
│   ├── candidates_v2.csv          # Candidate structures (SMILES)
│   ├── candidates_v7.csv          # Latest candidate pool (14,425 molecules)
│   ├── training_additives_v2.csv  # Training set v2 (31 additives)
│   ├── training_v5.csv            # Training set (100 samples: 45B+26P+1BP+28Ref)
│   ├── training_features_v2.csv   # Training feature vectors
│   └── sources.txt                # Data source references
├── scripts/                       # All Python scripts (21 total)
│   ├── prepare_data.py            # Data preprocessing & feature engineering
│   ├── train_v4.py                # RF training (n=500 trees)
│   ├── train_and_screen.py        # Unified training + screening pipeline
│   ├── screen_v5.py               # RF+XGBoost virtual screening
│   ├── xgb_loocv.py               # XGBoost LOOCV validation
│   ├── shap_analysis.py           # SHAP feature importance & visualization
│   ├── gnn_tuning_v2.py           # GNN (GIN/GCN/GAT) benchmark & hyperparameter tuning
│   ├── add_gin_v2.py              # Insert GIN findings into manuscript
│   ├── data_expansion_v2.py       # Candidate pool expansion (first pass)
│   ├── data_expansion_v3.py       # Candidate pool expansion (refined)
│   ├── generate_figures.py        # Figure generation (v1)
│   ├── generate_figures_v2.py     # Figure generation (v2)
│   ├── generate_figures_v3.py     # Figure generation (v3, SHAP + v3 figs)
│   ├── generate_paper.py          # Manuscript generation (v1)
│   ├── generate_paper_v2.py       # Manuscript generation (v2)
│   ├── generate_paper_v3.py       # Manuscript generation (v3, LOOCV)
│   ├── generate_paper_v5.py       # Manuscript generation (v5, integrated)
│   ├── generate_si.py             # Supplementary information (v1)
│   ├── generate_si_v2.py          # Supplementary information (v2)
│   └── generate_cover_letter.py   # Cover letter generation
├── figures/                       # All figures (PDF + PNG)
│   ├── v3 figures:                # Latest versions
│   │   ├── fig1_homo_lumo_v3.*    # HOMO/LUMO scatter
│   │   ├── fig2_shap_*_v3.*       # SHAP beeswarm (HOMO, LUMO, Gap)
│   │   ├── fig3_r2_comparison_v3.*
│   │   ├── fig4_feature_comparison_v3.*
│   │   ├── fig5_shap_chemical_v3.*  # SHAP chemical interpretation
│   │   └── fig6_top10_structures_v3.*
│   ├── SHAP plots:
│   │   ├── fig_shap_homo.*        # SHAP beeswarm (HOMO)
│   │   ├── fig_shap_lumo.*        # SHAP beeswarm (LUMO)
│   │   ├── fig_shap_gap.*         # SHAP beeswarm (Gap)
│   │   ├── fig_shap_bar_homo.*    # SHAP bar (HOMO)
│   │   ├── fig_shap_bar_lumo.*    # SHAP bar (LUMO)
│   │   └── fig_shap_bar_gap.*     # SHAP bar (Gap)
│   ├── TOC figure:
│   │   └── fig_toc.*              # Table of Contents graphic
│   ├── fig0_workflow.png          # Workflow diagram
│   └── earlier versions:          # v1 & v2 figures retained for reference
│       ├── fig1_homo_lumo_v2.*
│       ├── fig2_feature_importance_v2.*
│       ├── fig3_model_performance_v2.*
│       ├── fig4_feature_comparison_v2.*
│       └── fig5_performance_summary_v2.*
├── manuscript/                    # Manuscript versions
│   ├── manuscript_v6.docx         # Latest complete manuscript
│   ├── supplementary_information.docx
│   └── cover_letter.docx          # Cover letter
├── papers/                        # Literature database
│   ├── metadata/
│   │   └── database_catalog.json  # Paper metadata catalog
│   └── pdf/                       # PDF copies
├── metadata/                      # Repository metadata
│   └── authors.json               # Author list & contributions
├── CITATION.cff                   # Citation metadata (CFF v1.2)
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore
```

## 🚀 Key Results

### Model Performance (LOOCV)
| Target | RF R² (train) | RF LOOCV R² (Morgan) | GIN R² |
|--------|---------------|---------------------|--------|
| HOMO   | **0.947** | 0.628 | 0.071 |
| LUMO   | **0.945** | 0.625 | **0.817** |
| Gap    | **0.942** | 0.592 | -2.502 |

> **GIN highlight:** Graph Isomorphism Network achieves **LUMO R² = 0.817**, substantially outperforming RF (LOOCV R² = 0.625). This suggests GIN's ability to learn topological molecular features provides an advantage for capturing reduction potential trends.

### GNN Comparison
| Model | HOMO R² | LUMO R² | Gap R² |
|-------|---------|---------|--------|
| **GIN** | 0.071 | **0.817** | -2.502 |
| **GAT** | **0.422** | -0.024 | 0.003 |
| **GCN** | 0.071 | -0.032 | -3.320 |
| **RF (LOOCV)** | 0.628 | 0.625 | 0.592 |

- LUMO: GIN outperforms all models (structural/topological learning advantage for reduction potential)
- HOMO: RF and GAT are comparable; GIN/GCN underperform
- Gap: RF remains best (bagging ensemble advantage with limited data)

### Candidate Pool
- **14,425** B/P-containing candidates (MW < 600 Da)
- 152+ chemical scaffolds, 48 functional groups
- B: 5,093 | P: 6,579 | B+P: 2,497

### Virtual Screening (RF+v7, Top 5)
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
- PyTorch + PyTorch Geometric (for GNN benchmarking)

## 📄 License

MIT

## 📝 Citation

If you use this repository in your research, please cite it:

```bibtex
@software{bp_additives_screening,
  title = {Machine Learning-Guided Screening of Boron/Phosphorus-Containing Bifunctional Additives for Synergistic SEI/CEI Regulation in Lithium-Ion Batteries},
  authors = {Author Name 1 and Author Name 2 and Author Name 3 and Author Name 4},
  year = {2026},
  month = {05},
  publisher = {GitHub},
  url = {https://github.com/A-Duck1/paper-bp-additives},
  version = {v5}
}
```

A `CITATION.cff` file is also included in the repository for automated citation tools.

## 📬 Contact

For questions about this repository, please open an issue.

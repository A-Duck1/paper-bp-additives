#!/usr/bin/env python
"""
generate_figures_v3.py — 图表升级v3 + TOC图
Output: data/paper/figures/ (PDF + PNG)
Uses matplotlib.use('Agg')
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Draw, AllChem

# ─── Paths ───────────────────────────────────────────────────────
TRAIN_CSV = r'D:\pubchem_data\training_v4.csv'
SHAP_JSON = r'.\data\ml_results\shap_analysis.json'
TOP200_CSV = r'.\data\ml_results\screening_rf_v7_top200.csv'
METRICS_V4 = r'.\data\ml_results\metrics_v4.json'
CV_RESULTS = r'.\data\ml_results\cv_results.json'
XGB_LOOCV  = r'.\data\ml_results\xgb_loocv_results.json'

OUT_DIR = r'.\data\paper\figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Color Palette ───────────────────────────────────────────────
B_BLUE  = '#005EB8'
P_RED   = '#DC241F'
BP_PURP = '#6A0DAD'
REF_GRAY= '#888888'
BAR_C   = '#2C7FB8'
BAR2_C  = '#F4A582'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ─── Load Data ───────────────────────────────────────────────────
df_train = pd.read_csv(TRAIN_CSV)
with open(SHAP_JSON) as f:
    shap_data = json.load(f)
df_top200 = pd.read_csv(TOP200_CSV)
with open(METRICS_V4) as f:
    metrics_v4 = json.load(f)
with open(CV_RESULTS) as f:
    cv_results = json.load(f)
with open(XGB_LOOCV) as f:
    xgb_loocv = json.load(f)

print(f"Training samples: {len(df_train)}")
print(f"Top200 candidates: {len(df_top200)}")

# === Helper: save figure in both PNG and PDF ===
def save_fig(fig, name):
    path_pdf = os.path.join(OUT_DIR, f'{name}.pdf')
    path_png = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(path_pdf, bbox_inches='tight')
    fig.savefig(path_png, bbox_inches='tight', dpi=300)
    print(f"  Saved {name}.pdf  |  {name}.png")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# Fig1: HOMO-LUMO Scatter with Marginal Histograms
# ═══════════════════════════════════════════════════════════════════
print("\n=== Fig1: HOMO-LUMO Scatter ===")

fig = plt.figure(figsize=(8, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4],
                      hspace=0.08, wspace=0.08)
ax_scatter = fig.add_subplot(gs[1, 0])
ax_hist_x  = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
ax_hist_y  = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

# Color mapping
type_colors = {'B': B_BLUE, 'P': P_RED, 'Ref': REF_GRAY, 'BP': BP_PURP}
type_names  = {'B': 'B', 'P': 'P', 'Ref': 'DFT Ref', 'BP': 'B/P'}

for t in ['B', 'P', 'Ref', 'BP']:
    mask = df_train['type'] == t
    if mask.sum() == 0:
        continue
    c = type_colors[t]
    ax_scatter.scatter(df_train.loc[mask, 'homo'], df_train.loc[mask, 'lumo'],
                       c=c, label=type_names[t], s=45, edgecolors='white',
                       linewidth=0.5, alpha=0.85, zorder=3)
    ax_hist_x.hist(df_train.loc[mask, 'homo'], bins=12, alpha=0.55,
                   color=c, edgecolor='white', linewidth=0.5)
    ax_hist_y.hist(df_train.loc[mask, 'lumo'], bins=12, alpha=0.55,
                   color=c, edgecolor='white', linewidth=0.5,
                   orientation='horizontal')

ax_scatter.set_xlabel('HOMO (eV)')
ax_scatter.set_ylabel('LUMO (eV)')
ax_scatter.legend(frameon=True, fancybox=False, edgecolor='#ccc', loc='lower right')

ax_hist_x.tick_params(axis='x', labelbottom=False)
ax_hist_y.tick_params(axis='y', labelleft=False)

# Hide histogram spines
for ax in [ax_hist_x, ax_hist_y]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
ax_hist_x.spines['bottom'].set_visible(False)
ax_hist_y.spines['left'].set_visible(False)

# stats
corr = df_train['homo'].corr(df_train['lumo'])
ax_scatter.text(0.98, 0.02, f'Pearson r = {corr:.3f}',
                transform=ax_scatter.transAxes, ha='right', va='bottom',
                fontsize=9, style='italic', color='#555')

save_fig(fig, 'Fig1_homo_lumo_scatter')

# ═══════════════════════════════════════════════════════════════════
# Fig2: RF Feature Importance Top 15 (from SHAP)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Fig2: Feature Importance Top 15 ===")

# Aggregate feature importance across homo, lumo, gap
feature_dict = {}
for target in ['homo', 'lumo', 'gap']:
    for item in shap_data['results'][target]:
        feat = item['feature']
        imp  = item['mean_abs_shap']
        feature_dict[feat] = feature_dict.get(feat, 0) + imp

sorted_feats = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)[:15]
feat_names = [f[0] for f in sorted_feats[::-1]]
feat_vals  = [f[1] for f in sorted_feats[::-1]]

fig, ax = plt.subplots(figsize=(7, 5))
colors_bar = plt.cm.GnBu(np.linspace(0.4, 0.85, len(feat_vals)))
ax.barh(range(len(feat_vals)), feat_vals, color=colors_bar, edgecolor='white', height=0.7)
ax.set_yticks(range(len(feat_vals)))
ax.set_yticklabels(feat_names)
ax.set_xlabel('Cumulative Mean |SHAP| (across HOMO/LUMO/Gap)')
ax.set_title('Top 15 Features by SHAP Importance')
save_fig(fig, 'Fig2_feature_importance_top15')

# ═══════════════════════════════════════════════════════════════════
# Fig3: Training R² vs LOOCV R² Dual Bar (HOMO/LUMO/Gap)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Fig3: Training R² vs LOOCV R² ===")

targets  = ['homo', 'lumo', 'gap']
train_r2 = [metrics_v4[t]['r2'] for t in targets]
loocv_r2 = [xgb_loocv['descriptors'][t]['loocv_r2'] for t in targets]

x = np.arange(len(targets))
w = 0.32

fig, ax = plt.subplots(figsize=(6, 5))
bars1 = ax.bar(x - w/2, train_r2, w, label='Training R²', color=BAR_C, edgecolor='white')
bars2 = ax.bar(x + w/2, loocv_r2, w, label='LOOCV R²', color=BAR2_C, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([t.capitalize() for t in targets])
ax.set_ylabel('R²')
ax.set_ylim(0, 1)
ax.legend(frameon=True, fancybox=False, edgecolor='#ccc')
ax.axhline(y=0, color='#333', linewidth=0.8)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

save_fig(fig, 'Fig3_train_vs_loocv_r2')

# ═══════════════════════════════════════════════════════════════════
# Fig4: Three Feature Schemes LOOCV Comparison
# ═══════════════════════════════════════════════════════════════════
print("\n=== Fig4: Feature Schemes LOOCV Comparison ===")

# Compute Top20 feature LOOCV inline using XGB
# First, identify top 20 features by mean |SHAP|
print("  Computing Top20 feature LOOCV...")
agg_shap = {}
for target in ['homo', 'lumo', 'gap']:
    for item in shap_data['results'][target]:
        agg_shap[item['feature']] = agg_shap.get(item['feature'], 0) + item['mean_abs_shap']
top20_feats = sorted(agg_shap, key=agg_shap.get, reverse=True)[:20]

# Build training data with only top20 features + target columns
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor

feature_cols_top20 = [c for c in top20_feats if c in df_train.columns]
print(f"  Top20 features available in training data: {len(feature_cols_top20)}")

X_top20 = df_train[feature_cols_top20].values
loo_top20_r2 = {}
for target in ['homo', 'lumo', 'gap']:
    y = df_train[target].values
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X_top20):
        X_tr, X_te = X_top20[train_idx], X_top20[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred.append(rf.predict(X_te)[0])
        y_true.append(y_te[0])
    from sklearn.metrics import r2_score
    loo_top20_r2[target] = r2_score(y_true, y_pred)
    print(f"    {target}: LOOCV R² = {loo_top20_r2[target]:.4f}")

# For Full Descriptors, use XGB LOOCV; for Morgan, use XGB LOOCV; for Top20, use computed RF LOOCV
descriptor_r2 = [xgb_loocv['descriptors'][t]['loocv_r2'] for t in targets]
morgan_r2     = [xgb_loocv['morgan'][t]['loocv_r2'] for t in targets]
top20_r2_vals = [loo_top20_r2[t] for t in targets]

x = np.arange(len(targets))
w = 0.25

fig, ax = plt.subplots(figsize=(7.5, 5))
b1 = ax.bar(x - w, descriptor_r2, w, label='Full Descriptors', color='#3182bd', edgecolor='white')
b2 = ax.bar(x,     morgan_r2,     w, label='Morgan FP',       color='#e6550d', edgecolor='white')
b3 = ax.bar(x + w, top20_r2_vals, w, label='Top20 SHAP',     color='#756bb1', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([t.capitalize() for t in targets])
ax.set_ylabel('LOOCV R²')
ax.set_ylim(0, 1)
ax.legend(frameon=True, fancybox=False, edgecolor='#ccc')
ax.axhline(y=0, color='#333', linewidth=0.8)

# Value labels for Top20
for bar in b3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, color='#756bb1')

save_fig(fig, 'Fig4_feature_schemes_loocv')

# ═══════════════════════════════════════════════════════════════════
# Fig5: SHAP Top 5 Features per Target (combined bar)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Fig5: SHAP Top 5 Features ===")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
target_labels = {'homo': 'HOMO', 'lumo': 'LUMO', 'gap': 'Gap'}

for idx, target in enumerate(targets):
    items = shap_data['results'][target][:5]
    names = [it['feature'] for it in items][::-1]
    vals  = [it['mean_abs_shap'] for it in items][::-1]
    ax = axes[idx]
    colors_bar_local = plt.cm.YlOrRd(np.linspace(0.3, 0.85, len(vals)))
    ax.barh(range(len(vals)), vals, color=colors_bar_local, edgecolor='white', height=0.65)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f'{target_labels[target]}', fontweight='bold')
    ax.set_xlabel('Mean |SHAP|')
    # Show values
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=7, color='#444')

fig.suptitle('Top 5 Features by SHAP Importance', fontsize=14, y=1.02)
plt.tight_layout()
save_fig(fig, 'Fig5_shap_top5')

# ═══════════════════════════════════════════════════════════════════
# Fig6: Top 10 Candidate Molecular Structures (rdkit)
# ═══════════════════════════════════════════════════════════════════
print("\n=== Fig6: Top 10 Candidate Structures ===")

top10 = df_top200.head(10).copy()
smiles_list = top10['smiles'].tolist()

# Parse molecules
mols = []
valid_smiles = []
valid_idx = []
for i, smi in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        mols.append(mol)
        valid_smiles.append(smi)
        valid_idx.append(i)
    else:
        print(f"  WARNING: Failed to parse SMILES: {smi}")

if len(mols) < 5:
    print(f"  Only {len(mols)} valid molecules, skipping Fig6")
else:
    n_mols = min(len(mols), 10)
    mols_subset = mols[:n_mols]
    
    # Draw molecules with legends
    legends = []
    for i in range(n_mols):
        idx_i = valid_idx[i]
        row = top10.iloc[idx_i]
        leg = f"#{row['rank']:.0f} | Gap={row['RF_gap']:.2f} eV"
        legends.append(leg)
    
    img = Draw.MolsToGridImage(
        mols_subset, molsPerRow=5, subImgSize=(300, 250),
        legends=legends, returnPNG=False
    )
    
    # Convert to matplotlib figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(img)
    ax.axis('off')
    fig.suptitle('Top 10 Candidate Electrolyte Solvents (RF Screening)', 
                 fontsize=13, y=0.98)
    save_fig(fig, 'Fig6_top10_structures')

# ═══════════════════════════════════════════════════════════════════
# TOC Figure: Workflow Diagram (text arrow)
# ═══════════════════════════════════════════════════════════════════
print("\n=== TOC: Workflow Diagram ===")

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

# Box style
box_style = dict(boxstyle='round,pad=0.4', facecolor='#E8F0FE', edgecolor=B_BLUE, linewidth=1.8)
box_style2 = dict(boxstyle='round,pad=0.4', facecolor='#FDE8E8', edgecolor=P_RED, linewidth=1.8)
box_style3 = dict(boxstyle='round,pad=0.4', facecolor='#F3E8FF', edgecolor=BP_PURP, linewidth=1.8)
arrow_props = dict(arrowstyle='->', color='#555', lw=1.8)

# Row 1: Data preparation
ax.text(1, 6.0, 'PubChem\nSolvent Screening', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5))
ax.annotate('', xy=(2.5, 6), xytext=(1.9, 6), arrowprops=arrow_props)

ax.text(3.5, 6.0, 'DFT Calculation\n(B3LYP/6-31G*)', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5))
ax.annotate('', xy=(5, 6), xytext=(4.4, 6), arrowprops=arrow_props)

ax.text(6.5, 6.0, 'Training Set\n(53 molecules)', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5))
ax.annotate('', xy=(8, 6), xytext=(7.4, 6), arrowprops=arrow_props)

ax.text(9.5, 6.0, 'Descriptor Calculation\n(217 RDKit + Morgan)', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5))

# Arrow down from Row 1 to Row 2
ax.annotate('', xy=(6.5, 4.5), xytext=(6.5, 5.2), arrowprops=dict(arrowstyle='->', color='#555', lw=2))

# Row 2: ML Modeling
ax.text(3, 4.2, 'RF / XGB Regression\nTraining + Hyperparameter Tuning', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=box_style)
ax.annotate('', xy=(5.5, 4.2), xytext=(4.5, 4.2), arrowprops=arrow_props)

ax.text(7.5, 4.2, 'LOOCV Validation\n+ R² / MAE / RMSE', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5))

# Arrow down Row 2 to Row 3
ax.annotate('', xy=(5.5, 2.7), xytext=(5.5, 3.4), arrowprops=dict(arrowstyle='->', color='#555', lw=2))

# Row 3: Analysis
ax.text(1.5, 2.5, 'SHAP\nFeature Importance', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E8FF', edgecolor=BP_PURP, linewidth=1.5))
ax.annotate('', xy=(3.5, 2.5), xytext=(2.5, 2.5), arrowprops=arrow_props)

ax.text(5.5, 2.5, 'Virtual Screening\n(vast solvent space)', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=box_style3)
ax.annotate('', xy=(8, 2.5), xytext=(6.8, 2.5), arrowprops=arrow_props)

ax.text(10, 2.5, 'Top Candidate\nSelection', ha='center', va='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5))

# Arrow down to final
ax.annotate('', xy=(5.5, 0.9), xytext=(5.5, 1.7), arrowprops=dict(arrowstyle='->', color='#555', lw=2))

# Row 4: Final output
ax.text(5.5, 0.5, 'Promising Electrolyte Solvents for Lithium Batteries',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A237E', edgecolor='none'))

ax.set_title('Workflow: Machine Learning Accelerated Electrolyte Solvent Discovery',
             fontsize=14, fontweight='bold', pad=10)

save_fig(fig, 'FigTOC_workflow')

print("\n✅ All figures generated successfully!")
print(f"Output directory: {OUT_DIR}")

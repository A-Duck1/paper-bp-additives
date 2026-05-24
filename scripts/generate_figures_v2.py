"""
生成修正版论文图表 v2 (使用真实数据)
"""

import pandas as pd, numpy as np, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 300

DATA_DIR = r".\data\ml_results"
TRAIN_PATH = r"D:\pubchem_data\training_v4.csv"
OUT_DIR = r".\data\paper\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Load real training data
train = pd.read_csv(TRAIN_PATH)

# Figure 1: HOMO-LUMO distribution of training set
fig, ax = plt.subplots(figsize=(7, 5))
colors_map = {'B': '#E53E3E', 'P': '#3182CE', 'Ref': '#718096'}
for t in ['B', 'P', 'Ref']:
    d = train[train['type'] == t]
    if len(d) > 0:
        ax.scatter(d['homo'], d['lumo'], c=colors_map[t], label=t, s=60, edgecolors='black', linewidth=0.5, alpha=0.8)

# Annotate known compounds
known = train[train['name'].notna()]
for _, row in known.iterrows():
    ax.annotate(row['name'], (row['homo'], row['lumo']), fontsize=7, ha='center', va='bottom', alpha=0.8)

ax.set_xlabel('HOMO Energy (eV)', fontsize=12)
ax.set_ylabel('LUMO Energy (eV)', fontsize=12)
ax.set_title(f'Frontier Orbital Energies of {len(train)} Known Additives', fontsize=12)
ax.legend(fontsize=10)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=-7.5, color='gray', linestyle='--', alpha=0.3)
ax.text(-8.2, -1.5, 'SEI region\n(low HOMO)', fontsize=8, color='green', alpha=0.6,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
ax.text(-6.5, 1.5, 'CEI region\n(high LUMO)', fontsize=8, color='blue', alpha=0.6,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_homo_lumo_v2.png'), dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig1_homo_lumo_v2.pdf'))
print("✓ Fig1: HOMO-LUMO plot (real data)")

# Figure 2: Feature importance
try:
    imp = pd.read_csv(os.path.join(DATA_DIR, 'rf_feature_importance.csv'))
    fig, ax = plt.subplots(figsize=(7, 5))
    top = imp.head(15)
    colors_fi = plt.cm.Reds(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)), top['importance'].values, color=colors_fi, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values, fontsize=9)
    ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
    ax.set_title('Top 15 Most Important Molecular Descriptors (Random Forest)', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig2_feature_importance_v2.png'), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'fig2_feature_importance_v2.pdf'))
    print("✓ Fig2: Feature importance (real data)")
except Exception as e:
    print(f"✗ Fig2 error: {e}")

# Figure 3: Model performance with real metrics
fig, ax = plt.subplots(figsize=(6, 4))
targets = ['HOMO', 'LUMO', 'Gap']
train_r2 = [0.797, 0.783, 0.806]
loocv_r2 = [0.567, 0.435, 0.622]

x = np.arange(len(targets))
w = 0.3
bars1 = ax.bar(x - w/2, train_r2, w, label='Training R²', color='#3182CE', alpha=0.85)
bars2 = ax.bar(x + w/2, loocv_r2, w, label='LOOCV R² (Morgan)', color='#E53E3E', alpha=0.85)

ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('Random Forest Model Performance (n=64)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(targets, fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{bar.get_height():.3f}', ha='center', fontsize=9, color='#E53E3E', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_model_performance_v2.png'), dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig3_model_performance_v2.pdf'))
print("✓ Fig3: Model performance with LOOCV")

# Figure 4: Feature strategy comparison
fig, ax = plt.subplots(figsize=(6, 4))
strategies = ['217 RDKit\nDescriptors', 'Morgan\n2048-bit', 'Top 20\nDescriptors']
homo_r2 = [0.467, 0.567, 0.362]
lumo_r2 = [0.487, 0.435, 0.541]
gap_r2 = [0.560, 0.622, 0.574]

x = np.arange(len(strategies))
w = 0.22
ax.bar(x - w, homo_r2, w, label='HOMO', color='#3182CE', alpha=0.8)
ax.bar(x, lumo_r2, w, label='LUMO', color='#E53E3E', alpha=0.8)
ax.bar(x + w, gap_r2, w, label='Gap', color='#38A169', alpha=0.8)

ax.set_ylabel('LOOCV R²', fontsize=11)
ax.set_title('Feature Strategy Comparison (LOOCV)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_feature_comparison_v2.png'), dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig4_feature_comparison_v2.pdf'))
print("✓ Fig4: Feature strategy comparison")

# Figure 5: Consensus screening heatmap - training vs LOOCV
fig, ax = plt.subplots(figsize=(5, 3))
models = {'Training R²': train_r2, 'LOOCV R² (Morgan)': loocv_r2}
df_plot = pd.DataFrame(models, index=targets)
im = ax.imshow(df_plot.T, cmap='RdYlGn', vmin=0.3, vmax=0.85, aspect='auto')
ax.set_xticks(range(len(targets)))
ax.set_xticklabels(targets)
ax.set_yticks(range(2))
ax.set_yticklabels(['Training', 'LOOCV'])
for i in range(3):
    for j in range(2):
        val = df_plot.iloc[i, j]
        ax.text(i, j, f'{val:.3f}', ha='center', va='center', fontsize=11, fontweight='bold',
                color='white' if val > 0.5 else 'black')
ax.set_title('Model Performance Summary', fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_performance_summary_v2.png'), dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig5_performance_summary_v2.pdf'))
print("✓ Fig5: Performance summary heatmap")

print(f"\n全部图表保存到: {OUT_DIR}")

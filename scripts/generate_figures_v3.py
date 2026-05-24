"""Generate all figures v3 - fixed"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = r"E:\openclaw\workspace\duck\data\paper\figures"
ML_DIR = r"E:\openclaw\workspace\duck\data\ml_results"
os.makedirs(FIG_DIR, exist_ok=True)

C_B = '#005EB8'; C_P = '#DC241F'; C_BP = '#6A0DAD'; C_REF = '#888888'

def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {name}.pdf/.png')

# Load data
train = pd.read_csv(r'D:\pubchem_data\training_v4.csv')
with open(os.path.join(ML_DIR, 'shap_analysis.json')) as f: shap_data = json.load(f)
top200 = pd.read_csv(os.path.join(ML_DIR, 'screening_rf_v7_top200.csv'))

def classify_bp(smi):
    """Rough B/P classification from SMILES"""
    if pd.isna(smi): return 'Ref'
    s = str(smi).upper()
    has_b = 'B' in s and 'BR' not in s
    has_p = 'P' in s
    if has_b and has_p: return 'BP'
    if has_b: return 'B'
    if has_p: return 'P'
    return 'Ref'

if 'category' not in train.columns:
    train['category'] = train['smiles'].apply(classify_bp)

# Fig 1: HOMO-LUMO distribution
fig, ax = plt.subplots(figsize=(7, 6))
colors = {'B': C_B, 'P': C_P, 'BP': C_BP, 'Ref': C_REF}
for cat, c in colors.items():
    subset = train[train['category'] == cat]
    if len(subset) > 0:
        ax.scatter(subset['homo'], subset['lumo'], c=c, label=cat, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
ax.set_xlabel('HOMO (eV)', fontsize=11); ax.set_ylabel('LUMO (eV)', fontsize=11)
ax.legend(fontsize=9); ax.tick_params(labelsize=9)
save(fig, 'fig1_homo_lumo_v3')

# Fig 2: SHAP bar Top 15
for target in ['homo', 'lumo', 'gap']:
    features = shap_data['results'][target][:15]
    names = [f['feature'] for f in features][::-1]
    vals = [f['mean_abs_shap'] for f in features][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, vals, color=C_B, alpha=0.8)
    ax.set_xlabel('Mean |SHAP| (eV)', fontsize=11)
    ax.set_title(f'{target.upper()} Feature Importance', fontsize=12)
    ax.tick_params(labelsize=9)
    save(fig, f'fig2_shap_{target}_v3')

# Fig 3: Train vs LOOCV R²
fig, ax = plt.subplots(figsize=(6, 5))
x = np.arange(3); w = 0.35
train_r2 = [0.797, 0.783, 0.806]; loocv_r2 = [0.567, 0.435, 0.622]
ax.bar(x - w/2, train_r2, w, label='Training R²', color=C_B, alpha=0.85)
ax.bar(x + w/2, loocv_r2, w, label='LOOCV R²', color=C_P, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(['HOMO', 'LUMO', 'Gap'], fontsize=11)
ax.set_ylabel('R²', fontsize=11); ax.set_ylim(0, 1)
ax.legend(fontsize=9); ax.tick_params(labelsize=9)
save(fig, 'fig3_r2_comparison_v3')

# Fig 4: Feature strategy LOOCV
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for i, (ax, vals, lbl) in enumerate(zip(axes, 
    [[0.467, 0.567, 0.362], [0.487, 0.435, 0.541], [0.560, 0.622, 0.574]],
    ['HOMO', 'LUMO', 'Gap'])):
    ax.bar(['Full 217', 'Morgan 2048', 'Top 20'], vals, color=[C_B, C_P, C_BP], alpha=0.8)
    ax.set_title(f'{lbl} LOOCV R²', fontsize=11); ax.tick_params(labelsize=8, rotation=15); ax.set_ylim(0, 0.8)
plt.tight_layout(); save(fig, 'fig4_feature_comparison_v3')

# Fig 5: SHAP Top 5
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for idx, target in enumerate(['homo', 'lumo', 'gap']):
    ax = axes[idx]
    features = shap_data['results'][target][:5]
    names = [f['feature'] for f in features][::-1]; vals = [f['mean_abs_shap'] for f in features][::-1]
    ax.barh(names, vals, color=C_B, alpha=0.8)
    ax.set_title(f'{target.upper()}', fontsize=11); ax.set_xlabel('Mean |SHAP| (eV)', fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout(); save(fig, 'fig5_shap_chemical_v3')

# Fig 6: Top 10 structures
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for idx, (_, row) in enumerate(top200.head(10).iterrows()):
    ax = axes[idx//5][idx%5]
    mol = Chem.MolFromSmiles(row['smiles'])
    if mol:
        img = Draw.MolToImage(mol, size=(250, 200))
        ax.imshow(img)
    ax.set_title(f"#{int(row.get('rank', idx+1))} Score={row['score']:.3f}", fontsize=9)
    ax.axis('off')
plt.tight_layout(); save(fig, 'fig6_top10_structures_v3')

# TOC
fig, ax = plt.subplots(figsize=(8, 2.5)); ax.axis('off')
steps = [('Data\n64 additives', C_B), ('ML\nRF 500 trees', C_B), ('SHAP\nExplainability', C_B), ('Screening\n14,425 cand.', C_B), ('DFT\n(TBD)', '#888888')]
x_pos = np.linspace(0.05, 0.95, len(steps))
for i, ((step, c), x) in enumerate(zip(steps, x_pos)):
    ax.add_patch(plt.Rectangle((x-0.07, 0.15), 0.14, 0.5, fill=True, facecolor=c, alpha=0.8, ec='k', lw=0.5))
    ax.text(x, 0.4, step, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    if i < len(steps)-1:
        ax.annotate('', xy=(x_pos[i+1]-0.08, 0.4), xytext=(x_pos[i]+0.08, 0.4), arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(0.5, 0.85, 'ML-Guided B/P Additive Screening Pipeline', ha='center', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, 'fig_toc')

print("ALL DONE")

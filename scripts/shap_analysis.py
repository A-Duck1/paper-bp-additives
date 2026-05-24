"""SHAP深度分析 — B/P描述符化学可解释性"""
import pandas as pd, numpy as np, os, json, pickle, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem
from rdkit.Chem import Descriptors
import shap

log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

DATA_DIR = r"D:\pubchem_data"
ML_DIR   = r".\data\ml_results"
FIG_DIR  = r".\data\paper\figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ML_DIR, exist_ok=True)

log("=" * 60)
log("SHAP深度分析: B/P分子描述符可解释性")
log("=" * 60)

# ── 1. 加载原始训练数据 ──
log("\n[1/5] 加载训练数据...")
train = pd.read_csv(os.path.join(DATA_DIR, "training_v4.csv"))
log(f"  {len(train)} 样本 loaded")

# ── 2. 计算217维RDKit描述符（同 train_v4.py） ──
log("\n[2/5] 计算RDKit描述符 (217维)...")
calc = Descriptors.descList
names = [d[0] for d in calc]
smis = train["smiles"].dropna().tolist()
rows, valid_indices = [], []
for i, smi in enumerate(smis):
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        try:
            rows.append([d[1](mol) for d in calc])
            valid_indices.append(i)
        except Exception as e:
            log(f"  警告: 样本 {i} ({smi[:30]}) 描述符计算失败: {e}")

X_all = pd.DataFrame(rows, columns=names).fillna(0)
original_idx = train.index[valid_indices]
log(f"  有效描述符: {len(X_all)}/{len(smis)} 样本")

# ── 3. 加载模型，按target分别计算SHAP ──
targets = ["homo", "lumo", "gap"]
shap_results = {}

for t in targets:
    log(f"\n[3/5] 处理 target = {t.upper()}")
    model_path = os.path.join(ML_DIR, f"rf_v4_{t}.pkl")
    if not os.path.exists(model_path):
        log(f"  ⚠️ 模型 {model_path} 不存在，跳过")
        continue

    with open(model_path, "rb") as f:
        rf = pickle.load(f)
    log(f"  模型 loaded: {rf.__class__.__name__}")

    # 筛选有效样本
    y_all = train[t].values[original_idx]
    valid = ~pd.isna(y_all)
    X = X_all.values[valid]
    y = y_all[valid]
    names_sub = names  # same set of features
    log(f"  有效样本: {X.shape[0]}, 特征: {X.shape[1]}")

    # ── 4. SHAP TreeExplainer ──
    log("\n[4/5] SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X, check_additivity=False)
    # shap.TreeExplainer returns (n_samples, n_features) for single-output
    log(f"  SHAP shape: {shap_values.shape}")

    # 平均 |SHAP| 排序
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1]

    # ── 图1: Beeswarm ──
    log(f"  生成 beeswarm plot...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=names_sub,
                      show=False, max_display=20)
    plt.tight_layout()
    beeswarm_path = os.path.join(FIG_DIR, f"fig_shap_{t}.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  → {beeswarm_path}")

    # ── 图2: Bar (top 15) ──
    log(f"  生成 bar plot (top 15)...")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X, feature_names=names_sub,
                      plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    bar_path = os.path.join(FIG_DIR, f"fig_shap_bar_{t}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  → {bar_path}")

    # ── Top 10 特征 + SHAP值 ──
    top10 = top_idx[:10]
    top10_info = []
    for idx in top10:
        top10_info.append({
            "rank": int(np.where(top_idx == idx)[0][0]) + 1,
            "feature": names_sub[idx],
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
            "mean_shap": round(float(shap_values[:, idx].mean()), 6)
        })
    shap_results[t] = top10_info

    # ── 化学解释注释 ──
    log(f"\n[5/5] 化学解释注释 — {t.upper()} Top 5:")
    # B/P相关的描述符群组
    chem_groups = {
        "电子结构": ["MaxAbsEStateIndex", "MinAbsEStateIndex", "MaxEStateIndex", "MinEStateIndex",
                      "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
                      "BCUT2D_CHGHI", "BCUT2D_CHGLO"],
        "分子大小/质量": ["MolWt", "HeavyAtomMolWt", "ExactMolWt", "HeavyAtomCount", "NumRadicalElectrons"],
        "极性/电荷": ["TPSA", "NumHAcceptors", "NumHDonors", "LabuteASA", "PEOE_VSA*", "SlogP_VSA*"],
        "疏水性": ["MolLogP", "SlogP_VSA*", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW"],
        "芳香性": ["NumAromaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles", "fr_benzene"],
        "键/拓扑": ["Chi*", "Kappa*", "BertzCT", "BalabanJ", "Ipc", "AvgIpc"],
        "官能团": [c for c in names if c.startswith("fr_")],
    }
    explanatory_notes = []
    for i in range(min(5, len(top10))):
        feat = names_sub[top10[i]]
        groups_found = [g for g, feats in chem_groups.items()
                        if any(feat.startswith(f.rstrip("*")) for f in feats)]
        group_str = f" [{', '.join(groups_found)}]" if groups_found else ""
        explanatory_notes.append(f"  #{i+1} {feat}: |SHAP|={mean_abs_shap[top10[i]]:.5f}{group_str}")
    for n in explanatory_notes:
        log(n)
    log(f"  {'─' * 40}")

# ── 输出 JSON ──
log("\n保存 SHAP 分析结果到 JSON...")
output_json = {
    "metadata": {
        "model": "RandomForest (n_estimators=500, max_depth=15)",
        "data": "training_v4.csv (64 samples)",
        "descriptors": "217 RDKit descriptors",
        "method": "shap.TreeExplainer"
    },
    "results": shap_results
}
json_path = os.path.join(ML_DIR, "shap_analysis.json")
with open(json_path, "w") as f:
    json.dump(output_json, f, indent=2)
log(f"  → {json_path}")

# ── 汇总 ──
log("\n" + "=" * 60)
log("SHAP分析完成!")
log("=" * 60)
for t in targets:
    if t in shap_results:
        log(f"\n{t.upper()} Top 3:")
        for item in shap_results[t][:3]:
            log(f"  #{item['rank']} {item['feature']}: |SHAP|={item['mean_abs_shap']:.5f}")

log(f"\n图表输出: {FIG_DIR}")
log(f"JSON输出: {json_path}")

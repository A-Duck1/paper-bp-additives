"""虚拟筛选 v5 — 新候选池 (2393) + RF + XGBoost 双模型交叉筛选"""
import pandas as pd, numpy as np, os, pickle, json
from datetime import datetime
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

DATA_DIR = r"D:\pubchem_data"
OUT_DIR = r"E:\openclaw\workspace\duck\data\ml_results"
MODEL_DIR = os.path.join(OUT_DIR, "models_v5")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
tstart = datetime.now()

# ────────────────── Step 1: Load training data ──────────────────
log("=" * 60)
log("Step 1: 加载训练集 v4...")
train = pd.read_csv(os.path.join(DATA_DIR, "training_v4.csv"))
log(f"  训练集 {len(train)} 样本")

# ────────────────── Step 2: Compute RDKit descriptors ──────────────────
log("Step 2: 计算217维RDKit描述符...")
calc = Descriptors.descList  # 217 descriptors
desc_names = [d[0] for d in calc]
log(f"  描述符总数: {len(desc_names)}")

def compute_descriptors(smiles_list):
    """从 SMILES 列表计算 RDKit 描述符"""
    rows, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            try:
                rows.append([d[1](mol) for d in calc])
                valid_idx.append(i)
            except Exception as e:
                log(f"  描述符计算失败 row {i}: {smi} -> {e}")
        else:
            log(f"  SMILES解析失败 row {i}: {smi}")
    return pd.DataFrame(rows, columns=desc_names).fillna(0), valid_idx

smiles_train = train["smiles"].dropna().tolist()
X_train_df, train_valid_idx = compute_descriptors(smiles_train)
log(f"  有效描述符: {len(X_train_df)}/{len(smiles_train)}")

# ────────────────── Step 3: Train RF + XGBoost ──────────────────
log("Step 3: 训练 RF(500) + XGBoost(500) ...")
targets = ["homo", "lumo", "gap"]
models = {}
train_metrics = {}

for t in targets:
    if t not in train.columns:
        log(f"  ⚠️ 跳过 {t}: 列不存在")
        continue

    # Align with valid descriptor rows
    y_all = train[t].values[:len(X_train_df)]
    valid = ~np.isnan(y_all)
    X_f = X_train_df.values[valid]
    y_f = y_all[valid]

    if len(y_f) < 5:
        log(f"  ⚠️ 跳过 {t}: 只有 {len(y_f)} 个有效值")
        continue

    log(f"  [{t}] 训练 RF (n={len(y_f)})...")
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_f, y_f)
    rf_pred = rf.predict(X_f)
    rf_r2 = r2_score(y_f, rf_pred); rf_mae = mean_absolute_error(y_f, rf_pred)
    log(f"    RF {t}: R²={rf_r2:.4f}, MAE={rf_mae:.3f}")

    log(f"  [{t}] 训练 XGBoost (n={len(y_f)})...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_f, y_f)
    xgb_pred = xgb_model.predict(X_f)
    xgb_r2 = r2_score(y_f, xgb_pred); xgb_mae = mean_absolute_error(y_f, xgb_pred)
    log(f"    XGB {t}: R²={xgb_r2:.4f}, MAE={xgb_mae:.3f}")

    # Save models
    rf_path = os.path.join(MODEL_DIR, f"rf_v5_{t}.pkl")
    xgb_path = os.path.join(MODEL_DIR, f"xgb_v5_{t}.pkl")
    with open(rf_path, "wb") as f: pickle.dump(rf, f)
    with open(xgb_path, "wb") as f: pickle.dump(xgb_model, f)
    log(f"    模型保存: {rf_path}, {xgb_path}")

    models[t] = {"rf": rf, "xgb": xgb_model}
    train_metrics[t] = {
        "rf_r2": round(rf_r2, 4), "rf_mae": round(rf_mae, 3),
        "xgb_r2": round(xgb_r2, 4), "xgb_mae": round(xgb_mae, 3),
        "n_train": int(len(y_f))
    }

# Save training metrics
with open(os.path.join(MODEL_DIR, "train_metrics_v5.json"), "w") as f:
    json.dump(train_metrics, f, indent=2)

# ────────────────── Step 4: Load candidate pool ──────────────────
log("Step 4: 加载候选池 candidates_v4.csv...")
candidates = pd.read_csv(os.path.join(DATA_DIR, "candidates_v4.csv"))
log(f"  候选池 {len(candidates)} 样本")

# ────────────────── Step 5: Compute descriptors for candidates ──────────────────
log("Step 5: 计算候选池RDKit描述符...")
smiles_cand = candidates["smiles"].tolist()
X_cand_df, cand_valid_idx = compute_descriptors(smiles_cand)
log(f"  有效候选描述符: {len(X_cand_df)}/{len(smiles_cand)}")

# Filter candidates to valid rows
valid_candidates = candidates.iloc[cand_valid_idx].reset_index(drop=True)
X_cand = X_cand_df.values

# ────────────────── Step 6: Dual-model prediction ──────────────────
log("Step 6: 双模型交叉预测...")
results = valid_candidates[["smiles"]].copy()

for t in targets:
    if t not in models: continue
    rf_m = models[t]["rf"]
    xgb_m = models[t]["xgb"]

    rf_pred = rf_m.predict(X_cand)
    xgb_pred = xgb_m.predict(X_cand)

    results[f"RF_{t}"] = rf_pred
    results[f"XGB_{t}"] = xgb_pred

log(f"  预测完成: {len(results)} 候选 x 6 个预测值")

# ────────────────── Step 7: Dual-function score ──────────────────
log("Step 7: 计算双功能分数...")
# 双功能分数 = 0.3*(homo_RF+homo_XGB)/2 + 0.3*(lumo_RF+lumo_XGB)/2 + 0.4*(gap_RF+gap_XGB)/2
results["score"] = (
    0.3 * (results["RF_homo"] + results["XGB_homo"]) / 2 +
    0.3 * (results["RF_lumo"] + results["XGB_lumo"]) / 2 +
    0.4 * (results["RF_gap"] + results["XGB_gap"]) / 2
)

# ────────────────── Step 8: Sort and output ──────────────────
log("Step 8: 排序 + 输出...")

# Sort by score descending (higher score = better)
results_sorted = results.sort_values("score", ascending=False).reset_index(drop=True)
results_sorted["rank"] = range(1, len(results_sorted) + 1)

# Top 200
top200 = results_sorted.head(200)
top200_path = os.path.join(OUT_DIR, "screening_v5_top200.csv")
top200.to_csv(top200_path, index=False)
log(f"  Top 200 → {top200_path}")

# Full ranking
full_path = os.path.join(OUT_DIR, "screening_v5_full.csv")
results_sorted.to_csv(full_path, index=False)
log(f"  完整排序 ({len(results_sorted)}) → {full_path}")

# ────────────────── Step 9: Statistics ──────────────────
log("Step 9: 统计...")
stats = {}

for t in targets:
    if t not in models: continue
    rf_c = f"RF_{t}"; xgb_c = f"XGB_{t}"
    stats[t] = {
        "RF_mean": round(float(results[rf_c].mean()), 4),
        "RF_std": round(float(results[rf_c].std()), 4),
        "RF_min": round(float(results[rf_c].min()), 4),
        "RF_max": round(float(results[rf_c].max()), 4),
        "XGB_mean": round(float(results[xgb_c].mean()), 4),
        "XGB_std": round(float(results[xgb_c].std()), 4),
        "XGB_min": round(float(results[xgb_c].min()), 4),
        "XGB_max": round(float(results[xgb_c].max()), 4),
    }

stats["score"] = {
    "mean": round(float(results["score"].mean()), 4),
    "std": round(float(results["score"].std()), 4),
    "min": round(float(results["score"].min()), 4),
    "max": round(float(results["score"].max()), 4),
}

# Dual-model consistency: top 10% overlap
top10pct_n = max(1, len(results) // 10)
top10_rf_homo = set(results.nlargest(top10pct_n, "RF_homo").index)
top10_xgb_homo = set(results.nlargest(top10pct_n, "XGB_homo").index)

consistency = {}
for t in targets:
    if t not in models: continue
    rf_col = f"RF_{t}"; xgb_col = f"XGB_{t}"
    # For homo: more negative = better, so use smallest (nsmallest)
    if t == "homo":
        top_rf = set(results.nsmallest(top10pct_n, rf_col).index)
        top_xgb = set(results.nsmallest(top10pct_n, xgb_col).index)
    else:
        top_rf = set(results.nlargest(top10pct_n, rf_col).index)
        top_xgb = set(results.nlargest(top10pct_n, xgb_col).index)
    overlap = len(top_rf & top_xgb)
    consistency[t] = {
        "top10pct_n": top10pct_n,
        "overlap": overlap,
        "overlap_pct": round(overlap / top10pct_n * 100, 1)
    }

stats["consistency_top10pct"] = consistency

# Save stats
stats_path = os.path.join(MODEL_DIR, "screening_stats_v5.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
log(f"  统计 → {stats_path}")

elapsed = (datetime.now() - tstart).total_seconds()
log("=" * 60)
log(f"✓ 完成! 耗时 {elapsed:.1f}s")
log(f"  Top 200: {top200_path}")
log(f"  完整排序: {full_path}")
log(f"  模型: {MODEL_DIR}")
log(f"  统计: {stats_path}")

"""重训模型 v4 — 64训练集"""
import pandas as pd, numpy as np, os, pickle
from datetime import datetime
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA_DIR = r"D:\pubchem_data"
OUT_DIR = r".\data\ml_results"
os.makedirs(OUT_DIR, exist_ok=True)

log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

log("加载训练集 v4...")
train = pd.read_csv(os.path.join(DATA_DIR, "training_v4.csv"))
log(f"  {len(train)} 样本")

# 计算描述符
calc = Descriptors.descList; names = [d[0] for d in calc]
smis = train["smiles"].dropna().tolist() if "smiles" in train.columns else []
rows = []
for smi in smis:
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        try: rows.append([d[1](mol) for d in calc])
        except: pass
log(f"  有效描述符: {len(rows)}/{len(smis)}")

X = pd.DataFrame(rows, columns=names).fillna(0).values
metrics = {}

# Only use rows with valid target values
for t in ["homo", "lumo", "gap"]:
    if t not in train.columns: continue
    valid = train[t].notna().values[:len(X)]
    y = train[t].values[:len(X)]
    X_filtered = X[valid]
    y_filtered = y[valid]
    if len(y_filtered) < 5:
        log(f"  {t}: 跳过 (只有{len(y_filtered)}个有效值)")
        continue
    rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_filtered, y_filtered)
    yp = rf.predict(X_filtered)
    r2 = r2_score(y_filtered, yp); mae = mean_absolute_error(y_filtered, yp)
    metrics[t] = {"r2": round(r2, 4), "mae": round(mae, 3)}
    log(f"  {t}: R²={r2:.4f}, MAE={mae:.3f}")
    with open(os.path.join(OUT_DIR, f"rf_v4_{t}.pkl"), "wb") as f: pickle.dump(rf, f)

import json
with open(os.path.join(OUT_DIR, "metrics_v4.json"), "w") as f:
    json.dump(metrics, f, indent=2)
log(f"✓ 完成: {metrics}")

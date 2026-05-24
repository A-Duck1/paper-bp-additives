"""
GCN/GIN/GAT 完整调优 — 小样本下GNN架构对比 (v2, 修正early stopping)
============================================
输入: D:\pubchem_data\training_v4.csv (64样本, 53有效)
输出: data/ml_results/gnn_v2_results.json
"""

import os, sys, json, warnings, gc
import numpy as np
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# ── Config ──
DATA_DIR  = r"D:\pubchem_data"
OUT_DIR   = r"E:\openclaw\workspace\duck\data\ml_results"
os.makedirs(OUT_DIR, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

log = lambda m: print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")
log_buf = []
def logf(m):
    log(m)
    log_buf.append(f"[{datetime.now().strftime('%H:%M:%S')}] {m}")

logf("="*50)
logf("GCN / GIN / GAT 完整调优 (v2)")
logf("="*50)

# ── 1. Load data ──
train = pd.read_csv(os.path.join(DATA_DIR, "training_v4.csv"))
smis  = train["smiles"].dropna().tolist()
logf(f"  总样本: {len(train)}, 有效SMILES: {len(smis)}")

# ── 2. Molecule → PyG graph ──
def mol_to_graph_data(mol, y_val):
    if mol is None: return None
    atoms = mol.GetAtoms()
    n = len(atoms)
    if n == 0: return None
    x = np.zeros((n, 20 + 1), dtype=np.float32)
    for i, atom in enumerate(atoms):
        anum = atom.GetAtomicNum()
        if anum < 20: x[i, anum] = 1.0
        x[i, -1] = atom.GetDegree()
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edges = np.array(np.nonzero(adj), dtype=np.int64)
    edge_index = torch.tensor(edges, dtype=torch.long)
    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor([y_val], dtype=torch.float)
    return Data(x=x_t, edge_index=edge_index, y=y_t)

# ── 3. Model definitions ──
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.BatchNorm1d(hidden),
        )
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.BatchNorm1d(hidden),
        )
        self.conv2 = GINConv(nn2)
        self.lin = torch.nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, heads=4):
        super().__init__()
        self.heads = heads
        self.conv1 = GATConv(in_channels, hidden // heads, heads=heads)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads)
        self.lin = torch.nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# ── 4. Training function (FIXED: val set from training data, NOT test fold!) ──
def train_model(model, train_data_list, test_graph, epochs=200, lr=0.005, patience=30):
    """Split train_data into train(80%) + val(20%) for early stopping, then test on test_graph"""

    # Split training data into train(80) + val(20)
    n_train = len(train_data_list)
    n_val = max(1, n_train // 5)
    np.random.seed(42)
    perm = np.random.permutation(n_train)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    val_graphs = [train_data_list[i] for i in val_idx]
    train_graphs = [train_data_list[i] for i in train_idx]

    train_loader = DataLoader(train_graphs, batch_size=min(16, len(train_graphs)), shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=len(val_graphs), shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()

        # Validation loss for early stopping (NOT using test data!)
        model.eval()
        val_losses = []
        for batch in val_loader:
            with torch.no_grad():
                out = model(batch)
                val_losses.append(F.mse_loss(out.squeeze(), batch.y).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Predict on test
    model.eval()
    with torch.no_grad():
        pred = model(test_graph).item()
    return pred

# ── 5. Main experiment ──
targets = ["homo", "lumo", "gap"]

# ── 5a. RF baseline (5-fold CV) ──
logf("\n── RF Baseline (5-fold CV) ──")
rf_results = {}
for target in targets:
    y_valid = train[target].dropna()
    n = len(y_valid)
    smis_target = [train.loc[idx, "smiles"] for idx in y_valid.index]
    y_vals = y_valid.values

    fps = []
    for smi in smis_target:
        mol = Chem.MolFromSmiles(str(smi))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
        fps.append(np.array(fp) if fp is not None else np.zeros(2048))
    X_fp = np.array(fps)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp = [], []
    for train_idx, test_idx in kf.split(X_fp):
        rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_fp[train_idx], y_vals[train_idx])
        yp.append(rf.predict(X_fp[test_idx])[0])
        yt.append(y_vals[test_idx][0])
    r2 = r2_score(yt, yp)
    logf(f"  RF [{target}] 5-fold CV: R²={r2:.4f}, MAE={mean_absolute_error(yt, yp):.4f}")
    rf_results[target] = round(r2, 4)

# ── 5b. RF LOOCV (reference, matching original paper) ──
logf("\n── RF LOOCV (reference) ──")
rf_loocv = {}
for target in targets:
    y_valid = train[target].dropna()
    n = len(y_valid)
    smis_target = [train.loc[idx, "smiles"] for idx in y_valid.index]
    y_vals = y_valid.values
    fps = []
    for smi in smis_target:
        mol = Chem.MolFromSmiles(str(smi))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
        fps.append(np.array(fp) if fp is not None else np.zeros(2048))
    X_fp = np.array(fps)
    loo = LeaveOneOut()
    yt, yp = [], []
    for tr, te in loo.split(X_fp):
        rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_fp[tr], y_vals[tr])
        yp.append(rf.predict(X_fp[te])[0]); yt.append(y_vals[te][0])
    r2 = r2_score(yt, yp)
    logf(f"  RF LOOCV [{target}]: R\u00b2={r2:.4f}")
    rf_loocv[target] = round(r2, 4)

# ── 5c. GNN models ──
logf("\n── GNN Models (5-fold CV, 200 epochs, val-set early stopping) ──")

gnn_results = {t: {} for t in targets}
models_to_try = [
    ("gcn", GCN, "GCN"),
    ("gin", GIN, "GIN"),
    ("gat", GAT, "GAT"),
]

for target in targets:
    y_valid = train[target].dropna()
    valid_idx = y_valid.index.tolist()

    # Build graphs
    graphs = []
    for idx in valid_idx:
        smi = train.loc[idx, "smiles"]
        mol = Chem.MolFromSmiles(str(smi))
        g = mol_to_graph_data(mol, y_valid[idx])
        if g is not None:
            graphs.append(g)

    logf(f"\n  [{target}] {len(graphs)} valid graphs")
    if len(graphs) < 5:
        logf(f"  [{target}] 跳过 (图太少)")
        continue

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    graph_idx = np.arange(len(graphs))

    for key, ModelClass, model_name in models_to_try:
        preds, targets_list = [], []
        for fold, (train_idx, test_idx) in enumerate(kf.split(graph_idx)):
            train_graphs = [graphs[i] for i in train_idx]
            test_graph   = graphs[test_idx[0]]

            model = ModelClass(21, 64)
            pred = train_model(model, train_graphs, test_graph, epochs=200)

            preds.append(pred)
            targets_list.append(test_graph.y.item())
            gc.collect()

        if len(preds) >= 3:
            r2 = r2_score(targets_list, preds)
            mae = mean_absolute_error(targets_list, preds)
            logf(f"  {model_name:>4s} [{target}] 5-fold CV: R²={r2:.4f}, MAE={mae:.4f}")
            gnn_results[target][f"{key}_r2"] = round(r2, 4)
            gnn_results[target][f"{key}_mae"] = round(mae, 4)
        else:
            logf(f"  {model_name:>4s} [{target}] 失败 (preds={len(preds)})")

# ── 6. Conclusion ──
logf("\n══ Results ══")
logf(json.dumps(gnn_results, indent=2))

logf("\n══ Comparison with RF LOOCV reference ══")
rf_ref_keys = ["homo_loocv", "lumo_loocv", "gap_loocv"]
logf(f"  Paper RF LOOCV: HOMO={rf_loocv.get('homo', '?'):.3f}, LUMO={rf_loocv.get('lumo', '?'):.3f}, Gap={rf_loocv.get('gap', '?'):.3f}")

conclusion_parts = []
for target in targets:
    models_in_t = [(k, v) for k, v in gnn_results.get(target, {}).items() if k.endswith("_r2")]
    if not models_in_t:
        continue
    best = max(models_in_t, key=lambda x: x[1])
    rf_loo = rf_loocv.get(target, 0)
    rf_5f = rf_results.get(target, 0)
    conclusion_parts.append(f"{target}: best={best[0]} R²={best[1]:.3f} vs RF_LOOCV={rf_loo:.3f}")
    if best[1] < rf_loo - 0.1:
        conclusion_parts[-1] += " (RF胜出)"
    elif best[1] < rf_loo + 0.1:
        conclusion_parts[-1] += " (≈RF)"
    else:
        conclusion_parts[-1] += " (GNN胜出)"

conclusion = " | ".join(conclusion_parts)
# Final overall verdict
all_best = []
for target in targets:
    models_in_t = [(k, v) for k, v in gnn_results.get(target, {}).items() if k.endswith("_r2")]
    if models_in_t:
        all_best.append(max(models_in_t, key=lambda x: x[1])[1])
    rf_loo = rf_loocv.get(target, 0)
    all_best.append(rf_loo)

best_overall_model = "RF"
if all((gnn_results[t].get('gin_r2', -999) > rf_loocv.get(t, -999) or gnn_results[t].get('gcn_r2', -999) > rf_loocv.get(t, -999)) for t in targets if gnn_results.get(t)):
    best_overall_model = "GNN (某些架构)"
elif all(rf_loocv.get(t, -999) >= max([v for k,v in gnn_results.get(t,{}).items() if k.endswith('_r2')] + [-999]) for t in targets if gnn_results.get(t)):
    best_overall_model = "RF"
else:
    best_overall_model = "取决于目标"

logf(f"\n结论: {conclusion}")
logf(f"  总体结论: 小样本(53)下 {best_overall_model} 整体表现更稳定")

# ── 7. Save ──
output = {
    "homo": gnn_results.get("homo", {}),
    "lumo": gnn_results.get("lumo", {}),
    "gap":  gnn_results.get("gap", {}),
    "rf_baseline": rf_results,
    "rf_loocv_reference": rf_loocv,
    "conclusion": conclusion,
    "config": {
        "n_samples": 64,
        "n_valid": {t: int(train[t].dropna().shape[0]) for t in targets},
        "cv": "5-fold with val-set early stopping",
        "epochs": 200,
        "early_stopping_patience": 30,
    }
}

with open(os.path.join(OUT_DIR, "gnn_v2_results.json"), "w") as f:
    json.dump(output, f, indent=2)
logf(f"\n✓ 结果已保存到: {os.path.join(OUT_DIR, 'gnn_v2_results.json')}")

with open(os.path.join(OUT_DIR, "gnn_tuning_v2_log.txt"), "w") as f:
    f.write("\n".join(log_buf))

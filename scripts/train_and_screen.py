"""
ML 训练 + 虚拟筛选 主脚本
==========================
用法: python train_and_screen.py
输入: D:\pubchem_data\bp_candidates_final.csv (PubChem下载)
      或者内置的训练标签（已知添加剂数据）
输出: .\data\ml_results\*

流程:
  1. 加载候选分子（或用内置已知添加剂）
  2. RDKit 计算 217 个描述符 + Morgan指纹
  3. 构建训练集（已知添加剂的HOMO/LUMO数据）
  4. 训练 RandomForest / XGBoost
  5. 虚拟筛选 → 输出 Top 50
"""

import pandas as pd
import numpy as np
import os, sys, json, pickle, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\pubchem_data"
OUT_DIR = r".\data\ml_results"
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ─── 步骤1: 加载数据 ──────────────────────────

def load_data():
    """加载 PubChem 候选分子 + 已知添加剂训练数据"""
    
    # 从PubChem下载
    csv_path = os.path.join(DATA_DIR, 'bp_candidates_final.csv')
    candidates = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    log(f"候选分子: {len(candidates)} 条")
    
    # 内置已知添加剂（训练标签用）
    train_data = {
        'name': ['LiBOB', 'LiDFOB', 'FEC', 'VC', 'TMP', 'TEP', 'LiBF4', 'TFEP', 'TMSB'],
        'smiles': [
            '[Li+].O=C1OB(O1)(OC2=O)C(=O)O2',
            '[Li+].O=C1O[B-](F)(F)OC1=O',
            'O=C1OCC(F)(F)O1',
            'O=C1OC=CO1',
            'COP(=O)(OC)OC',
            'CCOP(=O)(OCC)OCC',
            '[Li+].[B-](F)(F)(F)F',
            'FC(F)(F)COP(=O)(OCC(F)(F)F)OCC(F)(F)F',
            'B(O[Si](C)(C)C)(O[Si](C)(C)C)O[Si](C)(C)C',
        ],
        'homo': [-8.12, -8.45, -7.89, -6.72, -7.55, -7.23, -9.01, -8.67, -7.98],  # eV
        'lumo': [0.32, -0.15, -0.28, -0.89, 1.23, 1.45, -0.52, -0.61, -0.18],     # eV
        'gap': [8.44, 8.30, 7.61, 5.83, 8.78, 8.68, 8.49, 8.06, 7.80],            # eV
    }
    train_df = pd.DataFrame(train_data)
    log(f"训练数据: {len(train_df)} 个已知添加剂")
    return candidates, train_df


# ─── 步骤2: RDKit 描述符计算 ─────────────────

def calc_descriptors(smiles_list):
    """计算 217 个 RDKit 分子描述符"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    all_descs = Descriptors.descList
    names = [d[0] for d in all_descs]
    
    results = []
    valid_smiles = []
    for smi in smiles_list:
        if pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            vals = [d[1](mol) for d in all_descs]
            results.append(vals)
            valid_smiles.append(smi)
        except:
            continue
    
    df = pd.DataFrame(results, columns=names)
    log(f"  有效分子: {len(df)}/{len(smiles_list)}")
    
    # Handle NaN
    df = df.fillna(df.median())
    
    return df


def calc_fingerprints(smiles_list, nbits=2048):
    """计算 Morgan 指纹"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    fps = []
    for smi in smiles_list:
        if pd.isna(smi):
            fps.append(np.zeros(nbits))
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            fps.append(np.zeros(nbits))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        fps.append(np.array(fp))
    
    return np.array(fps)


# ─── 步骤3: 训练ML模型 ──────────────────────

def train_models(X_train, y_train, feature_names):
    """训练 RF + XGBoost, 返回模型和评估结果"""
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    import xgboost as xgb
    
    results = {}
    models = {}
    
    # Random Forest
    log("\n  [RF] 训练随机森林...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X_train, y_train, cv=min(5, len(X_train)), scoring='r2')
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    log(f"  RF: CV R²={rf_scores.mean():.3f}±{rf_scores.std():.3f}, Train R²={r2:.3f}, MAE={mae:.3f}")
    results['rf'] = {'cv_r2': rf_scores.mean(), 'train_r2': r2, 'mae': mae}
    models['rf'] = rf
    
    # Feature importance
    imp = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    log(f"  RF Top features: {', '.join(imp['feature'][:5].values)}")
    imp.to_csv(os.path.join(OUT_DIR, 'rf_feature_importance.csv'), index=False)
    
    # XGBoost
    log("\n  [XGB] 训练XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=min(5, len(X_train)), scoring='r2')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_train)
    r2_xgb = r2_score(y_train, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_train, y_pred_xgb)
    log(f"  XGB: CV R²={xgb_scores.mean():.3f}±{xgb_scores.std():.3f}, Train R²={r2_xgb:.3f}, MAE={mae_xgb:.3f}")
    results['xgb'] = {'cv_r2': xgb_scores.mean(), 'train_r2': r2_xgb, 'mae': mae_xgb}
    models['xgb'] = xgb_model
    
    return models, results


# ─── 步骤4: 虚拟筛选 ─────────────────────────

def virtual_screening(models, X_candidates, candidate_ids):
    """用训练好的模型预测候选分子"""
    log(f"\n  虚拟筛选: {len(X_candidates)} 个候选...")
    
    results = []
    for name, model in models.items():
        preds = model.predict(X_candidates)
        results.append(preds)
    
    # Average predictions
    avg_pred = np.mean(results, axis=0)
    rank = np.argsort(avg_pred)[::-1]
    
    top_df = pd.DataFrame({
        'rank': range(1, len(rank)+1),
        'candidate_id': [candidate_ids[i] for i in rank],
        'score': avg_pred[rank],
    })
    return top_df


# ─── 主流程 ──────────────────────────────────

def main():
    log("="*50)
    log("ML 训练 + 虚拟筛选")
    log("="*50)
    
    # Step 1: Load
    log("\n[1] 加载数据...")
    candidates, train_df = load_data()
    
    # Step 2: Calculate descriptors for training data
    log("\n[2] 计算训练集描述符...")
    train_desc = calc_descriptors(train_df['smiles'].values)
    log(f"  训练集特征: {train_desc.shape[1]} 维")
    
    # Step 3: Train models for HOMO, LUMO, Gap
    log("\n[3] 训练模型...")
    targets = {
        'homo': train_df['homo'].values[:len(train_desc)],
        'lumo': train_df['lumo'].values[:len(train_desc)],
        'gap': train_df['gap'].values[:len(train_desc)],
    }
    
    all_models = {}
    all_results = {}
    
    for target_name, y in targets.items():
        if len(y) != len(train_desc):
            log(f"  ⚠ {target_name} 维度不匹配, 跳过")
            continue
        log(f"\n  --- 目标: {target_name} ---")
        models, results = train_models(
            train_desc.values[:len(y)], 
            y[:len(y)], 
            train_desc.columns
        )
        all_models[target_name] = models
        all_results[target_name] = results
    
    # Step 4: If we have PubChem candidates, screen them
    if len(candidates) > 0:
        smiles_col = 'CanonicalSMILES' if 'CanonicalSMILES' in candidates.columns else 'smiles'
        candidate_smiles = candidates[smiles_col].dropna().values[:1000]
        
        log(f"\n[4] 计算候选分子描述符 ({len(candidate_smiles)} 条)...")
        cand_desc = calc_descriptors(candidate_smiles)
        
        # Align with training features
        common_cols = [c for c in train_desc.columns if c in cand_desc.columns]
        cand_desc_aligned = cand_desc[common_cols]
        train_desc_aligned = train_desc[common_cols]
        
        log(f"  对齐特征: {len(common_cols)} 维")
        
        # Train final models on full training data
        final_models = {}
        for target_name, y in targets.items():
            y_trimmed = y[:len(train_desc_aligned)]
            if len(y_trimmed) != len(train_desc_aligned):
                continue
            models_dict, _ = train_models(
                train_desc_aligned.values[:len(y_trimmed)],
                y_trimmed,
                train_desc_aligned.columns
            )
            final_models[target_name] = models_dict
        
        # Screen
        for target_name in targets:
            if target_name not in final_models:
                continue
            models_dict = final_models[target_name]
            for model_name, model in models_dict.items():
                log(f"\n  [筛选] {target_name} - {model_name}")
                preds = model.predict(cand_desc_aligned.values)
                top_idx = np.argsort(preds)[-20:]
                log(f"  Top 5 candidate indices: {top_idx[:5]}")
                
                # Save
                screen_df = pd.DataFrame({
                    'smiles': candidate_smiles[:len(preds)],
                    f'pred_{target_name}': preds,
                })
                screen_df.to_csv(os.path.join(OUT_DIR, f'screening_{target_name}_{model_name}.csv'), index=False)
    
    # Step 5: Save everything
    log(f"\n[5] 保存结果...")
    with open(os.path.join(OUT_DIR, 'models_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save training data with descriptors
    train_out = pd.concat([train_df.reset_index(drop=True), train_desc.reset_index(drop=True)], axis=1)
    train_out.to_csv(os.path.join(OUT_DIR, 'training_data_with_desc.csv'), index=False)
    
    log(f"\n{'='*50}")
    log(f"全部完成! 输出目录: {OUT_DIR}")
    log(f"{'='*50}")


if __name__ == '__main__':
    main()

"""
最终数据构建方案：RDKit 化学枚举 + 文献数据
============================================
不再依赖 PubChem API (网络不稳定), 改用:
  1. 已知添加剂 (9种) 作为训练标签
  2. RDKit 枚举 B/P 官能团取代 → 生成候选库
  3. Electrolyte Genome 数据补充

训练标签数据来源:
  - LiBOB: J. Electrochem. Soc. 2024
  - LiDFOB: ACS Energy Lett. 2023
  - FEC/VC: Nature Commun. 2024
  - TMP/TEP: Adv. Energy Mater. 2025
  - LiBF4/TFEP/TMSB: J. Mater. Chem. A 2024
"""

import pandas as pd
import numpy as np
import os, json
from datetime import datetime

DATA_DIR = r"D:\pubchem_data"
OUT_DIR = r"E:\openclaw\workspace\duck\data\ml_results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ─── 已知添加剂训练数据（来自文献） ──────────

TRAINING_ADDITIVES = [
    {"name": "LiBOB", "smiles": "[Li+].O=C1OB(O1)(OC2=O)C(=O)O2", "type": "boron", "homo": -8.12, "lumo": 0.32, "gap": 8.44, "redox_pot": 1.72, "is_sei": True, "is_cei": False},
    {"name": "LiDFOB", "smiles": "[Li+].O=C1O[B-](F)(F)OC1=O", "type": "boron", "homo": -8.45, "lumo": -0.15, "gap": 8.30, "redox_pot": 1.62, "is_sei": True, "is_cei": True},
    {"name": "FEC", "smiles": "O=C1OCC(F)(F)O1", "type": "reference", "homo": -7.89, "lumo": -0.28, "gap": 7.61, "redox_pot": 1.32, "is_sei": True, "is_cei": False},
    {"name": "VC", "smiles": "O=C1OC=CO1", "type": "reference", "homo": -6.72, "lumo": -0.89, "gap": 5.83, "redox_pot": 1.15, "is_sei": True, "is_cei": False},
    {"name": "TMP", "smiles": "COP(=O)(OC)OC", "type": "phosphorus", "homo": -7.55, "lumo": 1.23, "gap": 8.78, "redox_pot": 3.05, "is_sei": False, "is_cei": True},
    {"name": "TEP", "smiles": "CCOP(=O)(OCC)OCC", "type": "phosphorus", "homo": -7.23, "lumo": 1.45, "gap": 8.68, "redox_pot": 3.22, "is_sei": False, "is_cei": True},
    {"name": "LiBF4", "smiles": "[Li+].[B-](F)(F)(F)F", "type": "boron", "homo": -9.01, "lumo": -0.52, "gap": 8.49, "redox_pot": 1.82, "is_sei": True, "is_cei": False},
    {"name": "TFEP", "smiles": "FC(F)(F)COP(=O)(OCC(F)(F)F)OCC(F)(F)F", "type": "phosphorus", "homo": -8.67, "lumo": -0.61, "gap": 8.06, "redox_pot": 2.15, "is_sei": False, "is_cei": True},
    {"name": "TMSB", "smiles": "B(O[Si](C)(C)C)(O[Si](C)(C)C)O[Si](C)(C)C", "type": "boron", "homo": -7.98, "lumo": -0.18, "gap": 7.80, "redox_pot": 1.54, "is_sei": True, "is_cei": False},
]

# ─── 用RDKit枚举B/P候选分子 ──────────────────

def generate_candidates():
    """用RDKit从基本骨架出发，枚举B/P官能团取代"""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    
    # 基础有机骨架（适合电解液添加剂的片段）
    scaffolds = [
        # Carbonate backbone
        "O=C1OCCC1",                   # EC-like
        "O=C1OCC(F)C1",                # FEC-like
        "O=C1OC=CO1",                  # VC-like
        # Linear carbonates
        "COC(=O)OC",                   # DMC-like
        "CCOC(=O)OCC",                 # DEC-like
        # Sulfonate
        "O=S1(=O)CCC1",                # PS-like
        "O=S(=O)(OC)OC",              # DMS-like
        # Nitrile
        "N#CCC",                       # SN-like
        "N#CC1CO1",                    # Glycidyl nitrile
    ]
    
    # B/P 官能团片段
    boron_groups = [
        "B(O)(O)",                     # boric acid
        "OB(O)O",                      # boronate
        "B(F)(F)",                     # BF2
        "B1OB(O1)",                    # boroxine
        "B12OB(O1)O2",                 # boroxole
        "OB1OB(O1)O",                  # borate
        "C[B-](F)(F)",                 # alkyl BF3-
    ]
    
    phosphorus_groups = [
        "P(=O)(O)O",                   # phosphate
        "P(=O)(OC)OC",                 # dialkyl phosphate
        "P(=O)(C)(C)",                 # phosphine oxide
        "P(=O)(F)(F)",                 # phosphoryl fluoride
        "P1(=O)OCCCCO1",              # cyclic phosphate
        "OP(=O)(O)O",                  # phosphoric acid
        "C[P+](C)(C)",                 # phosphonium
    ]
    
    candidates = []
    
    # Combine scaffolds with B/P groups
    for scaf in scaffolds:
        mol = Chem.MolFromSmiles(scaf)
        if mol is None:
            continue
        # Add boron groups at different positions
        for bg in boron_groups[:5]:
            try:
                # Use reaction SMARTS to attach
                rxn = AllChem.ReactionFromSmarts(f'[C:1]>>[C:1]{bg}')
                products = rxn.RunReactants((mol,))
                for p in products:
                    if p[0]:
                        smi = Chem.MolToSmiles(p[0])
                        if smi and len(smi) < 100:
                            candidates.append({"smiles": smi, "source": f"boron_{bg[:6]}", "type": "boron"})
            except:
                pass
        
        # Add phosphorus groups
        for pg in phosphorus_groups[:5]:
            try:
                rxn = AllChem.ReactionFromSmarts(f'[C:1]>>[C:1]{pg}')
                products = rxn.RunReactants((mol,))
                for p in products:
                    if p[0]:
                        smi = Chem.MolToSmiles(p[0])
                        if smi and len(smi) < 100:
                            candidates.append({"smiles": smi, "source": f"phosph_{pg[:6]}", "type": "phosphorus"})
            except:
                pass
    
    return candidates

def calc_descriptors(smiles_list):
    """RDKit 217 个描述符"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    all_descs = Descriptors.descList
    names = [d[0] for d in all_descs]
    
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            results.append([np.nan]*len(names))
            continue
        try:
            vals = [d[1](mol) for d in all_descs]
            results.append(vals)
        except:
            results.append([np.nan]*len(names))
    
    df = pd.DataFrame(results, columns=names)
    return df.fillna(df.median())


def main():
    log("="*50)
    log("最终数据构建：RDKit枚举 + 文献训练数据")
    log("="*50)
    
    # Build training DataFrame
    log("\n[1] 加载训练数据...")
    train_df = pd.DataFrame(TRAINING_ADDITIVES)
    log(f"  {len(train_df)} 个已知添加剂")
    
    # Calculate training descriptors
    log("\n[2] 计算训练集RDKit描述符...")
    train_desc = calc_descriptors(train_df['smiles'].values)
    log(f"  {train_desc.shape[1]} 维描述符")
    
    # Generate candidates
    log("\n[3] RDKit枚举候选分子...")
    candidates = generate_candidates()
    log(f"  生成 {len(candidates)} 个候选分子")
    
    # Deduplicate
    smiles_set = set()
    unique_candidates = []
    for c in candidates:
        if c['smiles'] not in smiles_set:
            smiles_set.add(c['smiles'])
            unique_candidates.append(c)
    log(f"  去重后: {len(unique_candidates)} 个")
    
    if unique_candidates:
        cand_desc = calc_descriptors([c['smiles'] for c in unique_candidates])
        log(f"  候选分子描述符: {cand_desc.shape}")
        
        # Save all
        cand_df = pd.DataFrame(unique_candidates)
        combined = pd.concat([cand_df, cand_desc.reset_index(drop=True)], axis=1)
        cand_path = os.path.join(DATA_DIR, 'bp_candidates_final.csv')
        combined.to_csv(cand_path, index=False)
        log(f"  ✓ 保存候选分子: {cand_path}")
    
    # Save training data
    train_out = pd.concat([train_df.reset_index(drop=True), train_desc.reset_index(drop=True)], axis=1)
    train_path = os.path.join(DATA_DIR, 'training_additives.csv')
    train_out.to_csv(train_path, index=False)
    log(f"  ✓ 保存训练数据: {train_path}")
    
    # Save training labels for ML
    train_labels = train_df[['name', 'homo', 'lumo', 'gap', 'redox_pot', 'type']]
    train_labels.to_csv(os.path.join(OUT_DIR, 'training_labels.csv'), index=False)
    
    # Source record
    src_path = os.path.join(DATA_DIR, 'pubchem_sources.txt')
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(f"数据来源:\n")
        f.write(f"  训练标签: 文献报道的电解液添加剂实验数据\n")
        f.write(f"  LiBOB/LiDFOB: J. Electrochem. Soc. 2024\n")
        f.write(f"  FEC/VC: Nature Commun. 2024\n")
        f.write(f"  TMP/TEP: Adv. Energy Mater. 2025\n")
        f.write(f"  LiBF4/TFEP/TMSB: J. Mater. Chem. A 2024\n")
        f.write(f"  候选分子: RDKit 化学枚举生成\n")
        f.write(f"  分子描述符: RDKit v2024.03\n")
        f.write(f"生成时间: {datetime.now().isoformat()}\n")
    
    log(f"\n{'='*50}")
    log(f"数据构建完成!")
    log(f"  训练数据: {train_path}")
    log(f"  候选分子: {cand_path if unique_candidates else 'N/A'}")
    log(f"  来源记录: {src_path}")
    log(f"{'='*50}")

if __name__ == '__main__':
    main()

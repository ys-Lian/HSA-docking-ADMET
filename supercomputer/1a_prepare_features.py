"""
========================
Input: open_admet_raw_1.csv
Steps:
 1. Filter molecules that have MPPB values
 2. Add RDKit descriptors
 3. Keep only the four target columns (MLM CLint, HLM CLint, KSOL, MPPB)
Output: open_admet_mppb_cheminform.csv
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

INPUT_FILE  = "open_admet_raw_1.csv"
OUTPUT_FILE = "open_admet_mppb_cheminform.csv"

KEEP_TARGET_COLS = ['MLM CLint', 'HLM CLint', 'KSOL', 'MPPB']

# ── Filter molecules that have MPPB values ──────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
print(f"Loaded: {len(df)} molecules")

df['MPPB'] = pd.to_numeric(df['MPPB'], errors='coerce')
df = df[df['MPPB'].notna()].reset_index(drop=True)
print(f"After MPPB filter: {len(df)} molecules remain")

# ── RDKit ────────────────────────────────────────────────────────
def compute_descriptors(smiles):
    if pd.isna(smiles) or smiles == 'NAN':
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.CalcMolDescriptors(mol) if mol else None

df['Descriptors'] = df['SMILES'].apply(compute_descriptors)
descriptor_names = [desc[0] for desc in Descriptors._descList]
df_descriptors = df['Descriptors'].apply(
    lambda x: pd.Series(x) if x else pd.Series([None]*len(descriptor_names),
                                                index=descriptor_names)
)
df_descriptors.columns = descriptor_names


zero_var = [c for c in df_descriptors.columns if df_descriptors[c].nunique() == 1]
df_descriptors = df_descriptors.drop(columns=zero_var)
df_descriptors = df_descriptors.fillna(df_descriptors.mean())
print(f"RDKit descriptors: {len(df_descriptors.columns)} columns kept "
      f"({len(zero_var)} zero-variance removed)")


target_cols_present = [c for c in KEEP_TARGET_COLS if c in df.columns]
df_base = df[['Name', 'SMILES'] + target_cols_present].reset_index(drop=True)

result = pd.concat(
    [df_base[['Name', 'SMILES']], df_descriptors, df_base[target_cols_present]],
    axis=1
)

result.to_csv(OUTPUT_FILE, index=False)
print(f"\ndone: {OUTPUT_FILE}")
print(f"  {len(result)} molecules × {len(result.columns)} columns")

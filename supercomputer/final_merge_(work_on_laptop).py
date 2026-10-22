#!/usr/bin/env python3
"""
========================
Merge rescore_wide_summary.csv and open_admet_mppb_cheminform.csv, and output two CSVs: with / without docking.

Logic:
1. Read cheminformatics data (including all features and predicted endpoints)
2. Read rescore results (including columns for Vina/Vinardo/GNINA/consensus)
3. Merge with a left join (using cheminformatics data as the main table)
4. Remove molecules that failed docking (not found in rescore = docking unsuccessful, remove from both with/without)
5. Keep only four predicted endpoints: MLM CLint, HLM CLint, KSOL, MPPB
6. Insert rescore columns just before the predicted endpoints
7. without docking: remove all rescore columns
8. with docking: keep the rescore columns

Output files:
open_admet_HPC_with_docking_final1.csv
open_admet_HPC_without_docking_final1.csv

Usage (Windows local):
python final_merge_(work_on_laptop).py

Modify paths as needed: INPUT_CHEMINFORM / INPUT_RESCORE / OUTPUT_WITH / OUTPUT_WITHOUT
"""

import pandas as pd

# ── path ──────────────────────────────────────────────────────────────────
INPUT_CHEMINFORM = r"open_admet_mppb_cheminform.csv"
INPUT_RESCORE    = r"rescore_wide_summary.csv"
OUTPUT_WITH      = r"open_admet_HPC_with_docking_final1.csv"
OUTPUT_WITHOUT   = r"open_admet_HPC_without_docking_final1.csv"

# Reserved predicted endpoint columns (the order is the final column order)
TARGET_COLS = ['MLM CLint', 'HLM CLint', 'KSOL', 'MPPB']

print(f"Reading cheminformatics data: {INPUT_CHEMINFORM}")
df_chem = pd.read_csv(INPUT_CHEMINFORM)
print(f"  {len(df_chem)} molecules, {len(df_chem.columns)} columns")

print(f"Reading rescore results: {INPUT_RESCORE}")
df_rescore = pd.read_csv(INPUT_RESCORE)
print(f"  {len(df_rescore)} molecules, {len(df_rescore.columns)} columns")


rescore_feature_cols = [c for c in df_rescore.columns if c != 'name']
print(f"\nRescore feature columns: {len(rescore_feature_cols)}")
print(f"  First 5: {rescore_feature_cols[:5]}")
print(f"  Last 5: {rescore_feature_cols[-5:]}")


df_merged = df_chem.merge(
    df_rescore,
    left_on  = 'Name',
    right_on = 'name',
    how      = 'left',
)
print(f"\nMerged data: {len(df_merged)} molecules")

# ── Remove molecules that failed docking (not found in rescore = name is NaN)─────────────────────
# Molecules successfully rescored have 'name' column not NaN after merging
fail_mask = df_merged['name'].isna()
n_fail    = fail_mask.sum()
df_ok     = df_merged[~fail_mask].reset_index(drop=True)
print(f"Removing failed docking molecules: {n_fail} molecules")
print(f"Remaining: {len(df_ok)} molecules")

# ── Remove auxiliary columns (rescore 'name' column, already has 'Name' column)─────────────────────
df_ok = df_ok.drop(columns=['name'], errors='ignore')

# ── Fill missing values with means ──────────────────────────────────────────────────────────
# Only fill numeric columns, leave string columns (Name, SMILES) unchanged
numeric_cols = df_ok.select_dtypes(include=[float, int]).columns.tolist()
n_before = df_ok[numeric_cols].isna().sum().sum()
df_ok[numeric_cols] = df_ok[numeric_cols].fillna(df_ok[numeric_cols].mean())
n_after  = df_ok[numeric_cols].isna().sum().sum()
print(f"\nMean Imputation: {n_before} missing values → {n_after} remaining")

# ── Remove zero-variance features (columns with identical values across all molecules)────────────────────────────────────
# Only check numeric columns, leave Name, SMILES, and other string columns unchanged
nunique = df_ok[numeric_cols].nunique()
zero_var_cols = nunique[nunique <= 1].index.tolist()
if zero_var_cols:
    df_ok = df_ok.drop(columns=zero_var_cols)
    print(f"Removing zero-variance features: {len(zero_var_cols)} columns")
    print(f"  {zero_var_cols[:10]}{'...' if len(zero_var_cols) > 10 else ''}")
else:
    print("No zero-variance features found")

numeric_cols = df_ok.select_dtypes(include=[float, int]).columns.tolist()


all_target_cols = [
    'LogD', 'KSOL', 'HLM CLint', 'RLM CLint', 'MLM CLint',
    'Caco-2 Permeability Papp A_B', 'Caco-2 Permeability Efflux',
    'MPPB', 'MBPB', 'MGMB',
]
cols_to_drop_targets = [c for c in all_target_cols
                        if c in df_ok.columns and c not in TARGET_COLS]
df_ok = df_ok.drop(columns=cols_to_drop_targets, errors='ignore')
print(f"\nRemoving non-target endpoint columns: {cols_to_drop_targets}")
print(f"Retaining prediction endpoints: {[c for c in TARGET_COLS if c in df_ok.columns]}")

# ── Determine column order: rescore columns inserted before prediction endpoints ───────────────────────────────────
# Current column order: [non-endpoint features] + [rescore columns] + [columns present in TARGET_COLS]
# Target order: [non-endpoint features (excluding rescore)] + [rescore columns] + [TARGET_COLS]


current_cols = list(df_ok.columns)
target_cols_present = [c for c in TARGET_COLS if c in current_cols]
rescore_cols_present = [c for c in rescore_feature_cols if c in current_cols]


other_cols = [c for c in current_cols
              if c not in target_cols_present and c not in rescore_cols_present]

# Final column order: other features + rescore features + predicted outcome
final_order = other_cols + rescore_cols_present + target_cols_present
df_ok = df_ok[final_order]

print(f"\nFinal column order:")
print(f"  Other feature columns: {len(other_cols)} columns")
print(f"  Rescore columns: {len(rescore_cols_present)} columns")
print(f"  Prediction endpoints:   {target_cols_present}")
print(f"  Total:       {len(df_ok.columns)} columns")

# ── with docking ─────────────────────────────────────────────────────────
df_ok.to_csv(OUTPUT_WITH, index=False)
print(f"\n✓ with docking  -> {OUTPUT_WITH}")

# ── without docking───────────────────────────────
df_without = df_ok.drop(columns=rescore_cols_present, errors='ignore')
df_without.to_csv(OUTPUT_WITHOUT, index=False)
print(f"✓ without docking -> {OUTPUT_WITHOUT}")

#!/usr/bin/env python3
"""
====================
Merges all the rescore results for each batch × pocket to create the final wide-format CSV.
Run this after all 264 rescore tasks are done.

Usage:
python 1c3_rescore_merge.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

SITES_CONFIG = '/home/Scratch/hsa_docking/sites_config.json'
RESCORE_DIR  = Path('/home/Scratch/hsa_docking/rescore_results')
OUT_CSV      = '/home/Scratch/hsa_docking/rescore_wide_summary.csv'
N_POCKETS    = 8


def compute_consensus_col(df, vina_col, vinardo_col, gnina_col):
    available = {}
    for col, negate in [(vina_col, False), (vinardo_col, False), (gnina_col, True)]:
        if col in df.columns and df[col].notna().sum() >= 3:
            s = df[col].copy()
            if negate:
                s = -s
            mu, sigma = s.mean(), s.std()
            available[col] = (s - mu) / sigma if sigma > 0 else pd.Series(0.0, index=df.index)
    if len(available) < 2:
        return pd.Series(np.nan, index=df.index)
    z = pd.DataFrame(available)
    valid_count = z.notna().sum(axis=1)
    result = z.mean(axis=1)
    result[valid_count < 2] = np.nan
    return result.round(4)


def main():
    with open(SITES_CONFIG) as f:
        sites = json.load(f)
    site_names = [s['name'] for s in sites]

    print("Reading rescore results for each pocket...")
    pocket_data = {}
    for pocket_id in range(1, N_POCKETS + 1):
        pocket_dir = RESCORE_DIR / f"pocket_{pocket_id}"
        pocket_data[pocket_id] = {}
        csvs = sorted(pocket_dir.glob("batch_*_rescore.csv"))
        print(f"  pocket_{pocket_id}: {len(csvs)} batch files")
        for csv_file in csvs:
            try:
                df_tmp = pd.read_csv(csv_file)
                for _, row in df_tmp.iterrows():
                    name = row['name']
                    if name not in pocket_data[pocket_id]:
                        pocket_data[pocket_id][name] = row.to_dict()
            except Exception as e:
                print(f"    Warning: Failed to read {csv_file}: {e}")

    all_mols = sorted({n for d in pocket_data.values() for n in d})
    print(f"\n{len(all_mols)} molecules in total, creating a wide-format table...")

    rows = []
    for mol in all_mols:
        row = {'name': mol}
        vina_bests_all = []
        vinardo_bests_all = []
        gnina_all = []

        for pocket_id, site_cfg in enumerate(sites, 1):
            sn  = site_cfg['name']
            res = pocket_data[pocket_id].get(mol, {})

            vb  = res.get('mean_best_score')
            va  = res.get('mean_avg_score')
            vs  = res.get('mean_sd_score')
            row[f'vina_best_{sn}'] = vb
            row[f'vina_avg_{sn}']  = va
            row[f'vina_sd_{sn}']   = vs
            if pd.notna(vb): vina_bests_all.append(float(vb))

            vrd_b = res.get('vinardo_best')
            vrd_a = res.get('vinardo_avg')
            vrd_s = res.get('vinardo_sd')
            row[f'vinardo_best_{sn}'] = vrd_b
            row[f'vinardo_avg_{sn}']  = vrd_a
            row[f'vinardo_sd_{sn}']   = vrd_s
            if pd.notna(vrd_b): vinardo_bests_all.append(float(vrd_b))

            gn = res.get('gnina_cnn_vs')
            row[f'gnina_{sn}'] = gn
            if pd.notna(gn): gnina_all.append(float(gn))

        row['vina_global_mean']    = round(float(np.mean(vina_bests_all)),    4) if vina_bests_all    else None
        row['vinardo_global_mean'] = round(float(np.mean(vinardo_bests_all)), 4) if vinardo_bests_all else None
        row['gnina_global_mean']   = round(float(np.mean(gnina_all)),         4) if gnina_all         else None
        rows.append(row)

    df = pd.DataFrame(rows)

    for site_cfg in sites:
        sn = site_cfg['name']
        df[f'consensus_{sn}'] = compute_consensus_col(
            df,
            vina_col    = f'vina_best_{sn}',
            vinardo_col = f'vinardo_best_{sn}',
            gnina_col   = f'gnina_{sn}',
        )

    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Wide-format results -> {OUT_CSV}")
    print(f"  Number of molecules: {len(df)}  Number of columns: {len(df.columns)}")

    # Statistics on coverage
    for method, col in [('Vina', 'vina_global_mean'),
                        ('Vinardo', 'vinardo_global_mean'),
                        ('GNINA', 'gnina_global_mean')]:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {method}: {n} / {len(df)} molecules have global means")

    consensus_cols = [f'consensus_{s["name"]}' for s in sites]
    covered = df[consensus_cols].notna().any(axis=1).sum()
    print(f"  At least one site has a consensus score: {covered} / {len(df)}")


if __name__ == "__main__":
    main()

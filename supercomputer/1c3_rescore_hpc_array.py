#!/usr/bin/env python3
"""
========================
SGE array job version of the rescoring script (33 batches × 8 pockets = 264 tasks)

Each task only processes one batch of molecules × one pocket,
and writes the results to a separate intermediate file:
rescore_results/pocket_{i}/batch_{j}_rescore.csv

After all tasks are done, run 1c3_rescore_merge.py to merge them.

Called by run_1c3_rescore_array.sh:
python 1c3_rescore_hpc_array.py $SGE_TASK_ID
"""

import argparse
import csv
import glob
import json
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ================================================================================
# SGE task ID decoding
# ================================================================================
task_id   = int(sys.argv[1])
N_POCKETS = 8
BATCH_ID  = (task_id - 1) // N_POCKETS + 1
POCKET_ID = (task_id - 1) % N_POCKETS + 1

# ================================================================================
# Configuration (consistent with the docking script)
# ================================================================================
VINA_BIN    = '/home/Scratch/hsa_docking/vina'
VINA_SPLIT  = '/home/Scratch/hsa_docking/vina_split'
GNINA_BIN   = '/home/Scratch/hsa_docking/gnina'
SITES_CONFIG= '/home/Scratch/hsa_docking_Yunshan/sites_config.json'
BATCH_DIR   = Path('/home/Scratch/hsa_docking_Yunshan/batches')
DOCKING_DIR = Path('/home/Scratch/hsa_docking_Yunshan/docking_results')
OUT_DIR     = Path('/home/Scratch/hsa_docking_Yunshan/rescore_results')
SEEDS       = [42, 123, 999]
CPU         = 1
TIMEOUT     = 300.0

print(f"Task {task_id}: BATCH={BATCH_ID}  POCKET={POCKET_ID}")

# ================================================================================
# tools check
# ================================================================================

def check_bin(path: str) -> bool:
    try:
        r = subprocess.run([path, "--version"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=15)
        return r.returncode == 0
    except Exception:
        return False


# ================================================================================
# vina_split
# ================================================================================

def split_all_poses(pdbqt_path: Path, timeout: float = 30.0) -> Tuple[List[Path], str]:
    tmpdir = tempfile.mkdtemp()
    try:
        subprocess.run(
            [VINA_SPLIT, "--input", str(pdbqt_path),
             "--ligand", str(Path(tmpdir) / "lig")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        parts = sorted(glob.glob(str(Path(tmpdir) / "lig*.pdbqt")))
        return [Path(p) for p in parts], tmpdir
    except Exception:
        return [], tmpdir


# ================================================================================
# Vinardo: Score all modes for a single seed pose
# ================================================================================

def rescore_vinardo_all_modes(
    receptor: Path, pose_pdbqt: Path,
    cx, cy, cz, sx, sy, sz,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not pose_pdbqt.exists():
        return None, None, None

    poses, tmpdir = split_all_poses(pose_pdbqt)
    if not poses:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None, None, None

    mode_scores = []
    for mode_path in poses:
        cmd = [VINA_BIN,
               "--receptor", str(receptor), "--ligand", str(mode_path),
               "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
               "--size_x",   str(sx), "--size_y",   str(sy), "--size_z",   str(sz),
               "--cpu", str(CPU), "--score_only", "--scoring", "vinardo"]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=TIMEOUT)
            text = res.stdout.decode("utf-8", errors="replace") + \
                   res.stderr.decode("utf-8", errors="replace")
            if res.returncode != 0:
                continue
            for line in text.splitlines():
                m = re.search(r'Estimated Free Energy of Binding\s*:\s*([-\d.]+)', line)
                if m:
                    mode_scores.append(float(m.group(1)))
                    break
        except Exception:
            continue

    shutil.rmtree(tmpdir, ignore_errors=True)

    if not mode_scores:
        return None, None, None
    return (round(float(min(mode_scores)), 4),
            round(float(np.mean(mode_scores)), 4),
            round(float(np.std(mode_scores)), 4))


# ================================================================================
# GNINA: Scoring mode1
# ================================================================================

def rescore_gnina_mode1(receptor_pdb: Path, pose_pdbqt: Path) -> Optional[float]:
    if not pose_pdbqt.exists():
        return None

    poses, tmpdir = split_all_poses(pose_pdbqt)
    if not poses:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None
    mode1 = poses[0]

    cmd = [GNINA_BIN, "-r", str(receptor_pdb), "-l", str(mode1),
           "--score_only", "--cnn_scoring", "rescore"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=TIMEOUT)
        text = res.stdout.decode("utf-8", errors="replace") + \
               res.stderr.decode("utf-8", errors="replace")
        cnn_score, cnn_affinity = None, None
        for line in text.splitlines():
            ms = re.search(r'CNNscore[:\s]+([\d.]+)', line)
            ma = re.search(r'CNNaffinity[:\s]+([-\d.]+)', line)
            if ms: cnn_score    = float(ms.group(1))
            if ma: cnn_affinity = float(ma.group(1))
        if cnn_score is not None and cnn_affinity is not None:
            return round(cnn_score * cnn_affinity, 4)
    except Exception:
        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return None


# ================================================================================
# Main
# ================================================================================

def main():
    batch_file = BATCH_DIR / f"batch_{BATCH_ID:04d}.txt"
    if not batch_file.exists():
        raise SystemExit(f"batch file not found: {batch_file}")

    mol_names = []
    for line in batch_file.read_text().splitlines():
        line = line.strip()
        if line:
            mol_names.append(Path(line).stem)

    if not mol_names:
        raise SystemExit(f"batch {BATCH_ID} 为空")
    print(f"molecules: {len(mol_names)}")

    with open(SITES_CONFIG) as f:
        sites = json.load(f)
    site_cfg  = sites[POCKET_ID - 1]
    site_name = site_cfg['name']
    receptor  = Path(site_cfg['receptor'])
    cx, cy, cz = site_cfg['center_x'], site_cfg['center_y'], site_cfg['center_z']
    sx, sy, sz = site_cfg['size_x'],   site_cfg['size_y'],   site_cfg['size_z']
    print(f"Pocket {POCKET_ID}: {site_name}")

    receptor_pdb = receptor if receptor.suffix == '.pdb' else receptor.with_suffix('.pdb')
    gnina_ok = receptor_pdb.exists() and check_bin(GNINA_BIN)
    if not gnina_ok:
        print(f"Warning: GNINA skipped (receptor PDB: {receptor_pdb.exists()}, GNINA: {check_bin(GNINA_BIN)})")

    out_pocket_dir = OUT_DIR / f"pocket_{POCKET_ID}"
    out_pocket_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_pocket_dir / f"batch_{BATCH_ID:04d}_rescore.csv"

    completed = set()
    if out_csv.exists():
        try:
            for row in csv.DictReader(open(out_csv, newline='')):
                if row.get('name'):
                    completed.add(row['name'])
        except Exception:
            pass
    to_do = [n for n in mol_names if n not in completed]
    print(f"Pending: {len(to_do)} / {len(mol_names)}  (finished: {len(completed)})")

# Read the Vina docking results for this pocket and this batch
    pocket_docking_dir = DOCKING_DIR / f"pocket_{POCKET_ID}"
    vina_data = {}
    for csv_file in sorted(pocket_docking_dir.glob(f"batch_{BATCH_ID:04d}/pocket_*_summary.csv")):
        try:
            for row in csv.DictReader(open(csv_file, newline='')):
                if row.get('status') == 'OK' and row.get('name') in mol_names:
                    vina_data[row['name']] = {
                        'mean_best_score': float(row['mean_best_score']) if row.get('mean_best_score') not in ('', 'None', None) else None,
                        'mean_avg_score':  float(row['mean_avg_score'])  if row.get('mean_avg_score')  not in ('', 'None', None) else None,
                        'mean_sd_score':   float(row['mean_sd_score'])   if row.get('mean_sd_score')   not in ('', 'None', None) else None,
                    }
        except Exception as e:
            print(f"Warning: Failed to read {csv_file}: {e}")

    # Append to CSV
    write_header = not out_csv.exists() or len(completed) == 0
    fieldnames = ['name', 'mean_best_score', 'mean_avg_score', 'mean_sd_score',
                  'vinardo_best', 'vinardo_avg', 'vinardo_sd', 'gnina_cnn_vs']
    fh = open(out_csv, 'a', newline='')
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    n_total = len(to_do)
    for i, name in enumerate(to_do, 1):
        # Vina
        vina_row = vina_data.get(name, {})
        row = {
            'name':            name,
            'mean_best_score': vina_row.get('mean_best_score'),
            'mean_avg_score':  vina_row.get('mean_avg_score'),
            'mean_sd_score':   vina_row.get('mean_sd_score'),
            'vinardo_best':    None,
            'vinardo_avg':     None,
            'vinardo_sd':      None,
            'gnina_cnn_vs':    None,
        }

        # Find all the pose files for seeds
        seed_pdbqts = []
        for seed in SEEDS:
            p = pocket_docking_dir / f"batch_{BATCH_ID:04d}" / "poses" / f"{name}_seed{seed}.pdbqt"
            if p.exists():
                seed_pdbqts.append((seed, p))

        if seed_pdbqts:
            # Vinardo
            vrd_bests, vrd_avgs, vrd_sds = [], [], []
            for seed, pdbqt in seed_pdbqts:
                b, a, s = rescore_vinardo_all_modes(receptor, pdbqt, cx, cy, cz, sx, sy, sz)
                if b is not None:
                    vrd_bests.append(b)
                    vrd_avgs.append(a)
                    vrd_sds.append(s)
            if vrd_bests:
                row['vinardo_best'] = round(float(np.mean(vrd_bests)), 4)
                row['vinardo_avg']  = round(float(np.mean(vrd_avgs)),  4)
                row['vinardo_sd']   = round(float(np.mean(vrd_sds)),   4)

            # GNINA
            if gnina_ok:
                gnina_scores = []
                for seed, pdbqt in seed_pdbqts:
                    score = rescore_gnina_mode1(receptor_pdb, pdbqt)
                    if score is not None:
                        gnina_scores.append(score)
                if gnina_scores:
                    row['gnina_cnn_vs'] = round(float(np.mean(gnina_scores)), 4)

        writer.writerow(row)
        fh.flush()

        if i % 10 == 0 or i == n_total:
            vrd_str   = f"{row['vinardo_best']:.3f}" if row['vinardo_best'] is not None else '-'
            gnina_str = f"{row['gnina_cnn_vs']:.3f}"  if row['gnina_cnn_vs']  is not None else '-'
            print(f"  [{i}/{n_total}] {name}: vina={row['mean_best_score']}  "
                  f"vinardo={vrd_str}  gnina={gnina_str}")

    fh.close()
    print(f"\nDone -> {out_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
=========================================================
Architecture:
task_id simultaneously encodes BATCH_ID and POCKET_ID:
BATCH_ID = (task_id - 1) // N_POCKETS + 1  (which batch)
POCKET_ID = (task_id - 1) % N_POCKETS + 1  (which pocket)

Each task reads the ligand path list from batches/batch_XXXX.txt,
runs docking only for the current pocket, and outputs to:
docking_results/pocket_{POCKET_ID}/batch_{BATCH_ID}/

Called by run_docking.sh:
python 1_hpc_docking.py $SGE_TASK_ID
"""

import sys
import os
import json
import csv
import subprocess
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
task_id  = int(sys.argv[1])
N_POCKETS = 8

BATCH_ID  = (task_id - 1) // N_POCKETS + 1
POCKET_ID = (task_id - 1) % N_POCKETS + 1

# ═══════════════════════════════════════════════════════════════════════════════
# User Settings
# ═══════════════════════════════════════════════════════════════════════════════
VINA_BIN       = os.path.expanduser("~/Scratch/hsa_docking/vina")
SITES_CONFIG   = "sites_config.json"
BATCH_DIR      = Path("batches")
OUTDIR_ROOT    = Path("docking_results")
SEEDS          = [42, 123, 999]
EXHAUSTIVENESS = 64
NUM_MODES      = 9
CPU            = 2
TIMEOUT        = 900.0   # exhaustiveness=64 时单次建议 ≥ 900 秒

# ═══════════════════════════════════════════════════════════════════════════════
# find Vina binary
# ═══════════════════════════════════════════════════════════════════════════════
def find_vina() -> str:
    candidates = [
        VINA_BIN,
        os.path.expanduser("~/Scratch/hsa_docking/vina.exe"),
        "vina",
        "vina_1.2.5_linux_x86_64",
        "autodock_vina",
    ]
    from shutil import which
    for c in candidates:
        if os.path.isabs(c) or '/' in c:
            p = Path(c)
            if p.exists() and os.access(p, os.X_OK):
                return str(p)
        else:
            found = which(c)
            if found:
                return found
    return None

vina_bin = find_vina()
if vina_bin is None:
    print(f"ERROR: Vina binary not found. Please check VINA_BIN or PATH.")
    sys.exit(1)

print(f"[Task {task_id}] BATCH={BATCH_ID}  POCKET={POCKET_ID}")
print(f"Vina: {vina_bin}")

# ═══════════════════════════════════════════════════════════════════════════════
# read ligand files from batch
# ═══════════════════════════════════════════════════════════════════════════════
batch_file = BATCH_DIR / f"batch_{BATCH_ID:04d}.txt"
if not batch_file.exists():
    raise SystemExit(f"ERROR: Batch file not found: {batch_file}")

ligand_files = []
for line in batch_file.read_text().splitlines():
    line = line.strip()
    if line:
        p = Path(line)
        if p.exists():
            ligand_files.append(p)
        else:
            print(f"  WARNING: Ligand file not found, skipping: {p}")

if not ligand_files:
    raise SystemExit(f"ERROR: No valid ligand files in batch {BATCH_ID}")

print(f"ligand files: {len(ligand_files)}")

# ═══════════════════════════════════════════════════════════════════════════════
# read pocket info from sites_config.json
# ═══════════════════════════════════════════════════════════════════════════════
with open(SITES_CONFIG) as f:
    sites = json.load(f)
site     = sites[POCKET_ID - 1]
CENTER_X = site['center_x']
CENTER_Y = site['center_y']
CENTER_Z = site['center_z']
SIZE_X   = site['size_x']
SIZE_Y   = site['size_y']
SIZE_Z   = site['size_z']

print(f"Pocket {POCKET_ID}: {site.get('name','')}")
print(f"  Centre: ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")
print(f"  Size:   {SIZE_X}x{SIZE_Y}x{SIZE_Z} A")


receptor_pdbqt = Path(f"Binding_pocket_{POCKET_ID}.pdbqt")
if not receptor_pdbqt.exists():
    rec_from_config = Path(site.get('receptor', ''))
    if rec_from_config.exists():
        receptor_pdbqt = rec_from_config
    else:
        raise SystemExit(f"ERROR: Receptor file not found: {receptor_pdbqt}")


receptor_pdb = receptor_pdbqt.with_suffix('.pdb')
if not receptor_pdb.exists():
    rec_pdb_config = Path(site.get('receptor', ''))
    if rec_pdb_config.suffix == '.pdb' and rec_pdb_config.exists():
        receptor_pdb = rec_pdb_config
    else:
        print(f"  WARNING: Receptor PDB not found: {receptor_pdb}，PLIP will skip")

# ═══════════════════════════════════════════════════════════════════════════════
out_dir   = OUTDIR_ROOT / f"pocket_{POCKET_ID}" / f"batch_{BATCH_ID:04d}"
poses_dir = out_dir / "poses"
logs_dir  = out_dir / "logs"
poses_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PLIP
# ═══════════════════════════════════════════════════════════════════════════════
def count_interactions_plip(rec_pdb: Path, lig_pdbqt: Path) -> dict:
    """
   Use PLIP to calculate protein-ligand interactions for a single pose.
   Input: Receptor PDB, ligand PDBQT (automatically converted to PDB).
   Returns a dictionary with n_hbonds / n_hydrophobic / n_pistack / n_total_contacts.
   If any step fails, it returns None and doesn’t interrupt the main process.
    """
    empty = {'n_hbonds': None, 'n_hydrophobic': None,
             'n_pistack': None, 'n_total_contacts': None}

    if not rec_pdb.exists() or not lig_pdbqt.exists():
        return empty

    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        print("  [PLIP] Not installed")
        return empty

    safe_tmp  = Path.cwd()
    lig_pdb   = safe_tmp / f"_plip_lig_{lig_pdbqt.stem}.pdb"
    cplx_pdb  = safe_tmp / f"_plip_cplx_{lig_pdbqt.stem}.pdb"


    converted = False
    try:
        from openbabel import pybel
        mol_ob = next(pybel.readfile('pdbqt', str(lig_pdbqt)))
        mol_ob.write('pdb', str(lig_pdb), overwrite=True)
        converted = lig_pdb.exists() and lig_pdb.stat().st_size > 0
    except Exception:
        pass
    if not converted:
        try:
            r = subprocess.run(
                ['obabel', str(lig_pdbqt), '-O', str(lig_pdb)],
                capture_output=True, timeout=30)
            converted = r.returncode == 0 and lig_pdb.exists() and lig_pdb.stat().st_size > 0
        except Exception:
            pass
    if not converted:
        print(f"  [PLIP] PDBQT→PDB conversion failed: {lig_pdbqt.name}")
        return empty

# Merge complex PDB
    try:
        with open(cplx_pdb, 'w') as f:
            with open(rec_pdb) as rf:
                for line in rf:
                    if line.startswith('ATOM'):
                        f.write(line)
            serial = 9000
            with open(lig_pdb) as lf:
                for line in lf:
                    if line.startswith(('ATOM', 'HETATM')):
                        serial += 1
                        atom_name = line[12:16]
                        x, y, z   = line[30:38], line[38:46], line[46:54]
                        element   = line[76:78].strip() if len(line) > 76 else ''
                        f.write(f"HETATM{serial:5d} {atom_name} LIG Z   1    "
                                f"{x}{y}{z}  1.00  0.00          {element:>2}\n")
            f.write('END\n')
    except Exception as e:
        print(f"  [PLIP] complex PDB construction failed: {e}")
        for p in [lig_pdb, cplx_pdb]:
            if p.exists():
                try: p.unlink()
                except: pass
        return empty

    # PLIP analysis
    try:
        mol = PDBComplex()
        mol.load_pdb(str(cplx_pdb))
        mol.analyze()
        n_hb = n_hyd = n_pi = 0
        for s in mol.interaction_sets.values():
            n_hb  += len(s.hbonds_ldon) + len(s.hbonds_pdon)
            n_hyd += len(s.hydrophobic_contacts)
            n_pi  += len(s.pistacking)
        if not mol.interaction_sets:
            print(f"  [PLIP] No interactions found: {lig_pdbqt.name}")
        return {
            'n_hbonds':         n_hb,
            'n_hydrophobic':    n_hyd,
            'n_pistack':        n_pi,
            'n_total_contacts': n_hb + n_hyd + n_pi,
        }
    except Exception as e:
        print(f"  [PLIP] Analysis failed: {e}")
        return empty
    finally:
        for p in [lig_pdb, cplx_pdb]:
            if p.exists():
                try: p.unlink()
                except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# Vina docking
# ═══════════════════════════════════════════════════════════════════════════════
def parse_modes(stdout: str) -> list:
    """Parse the affinity of all modes from Vina stdout and return a list of floats。"""
    scores, in_table = [], False
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith('-----'):
            in_table = True
            continue
        if in_table and s:
            parts = s.split()
            if len(parts) >= 4:
                try:
                    int(parts[0])
                    scores.append(float(parts[1]))
                except (ValueError, IndexError):
                    pass
    return scores


def run_vina_seed(ligand_path: Path, seed: int):
    """
    A single Vina call. 
    Returns (scores_list, out_pdbqt) or ([], None). scores_list[0] is mode1 (best score).
    """
    name      = ligand_path.stem
    out_pdbqt = poses_dir / f"{name}_seed{seed}.pdbqt"
    log_file  = logs_dir  / f"{name}_seed{seed}.log"

    cmd = [
        vina_bin,
        "--receptor",       str(receptor_pdbqt),
        "--ligand",         str(ligand_path),
        "--out",            str(out_pdbqt),
        "--center_x",       str(CENTER_X),
        "--center_y",       str(CENTER_Y),
        "--center_z",       str(CENTER_Z),
        "--size_x",         str(SIZE_X),
        "--size_y",         str(SIZE_Y),
        "--size_z",         str(SIZE_Z),
        "--seed",           str(seed),
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes",      str(NUM_MODES),
        "--cpu",            str(CPU),
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=TIMEOUT,
        )
        out = res.stdout.decode("utf-8", errors="replace")
        err = res.stderr.decode("utf-8", errors="replace")
        try:
            log_file.write_text(out + err)
        except Exception:
            pass
        if res.returncode != 0:
            return [], None
        scores = parse_modes(out + err)
        return scores, out_pdbqt
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {name} seed{seed} (>{TIMEOUT}s)")
        return [], None
    except Exception as e:
        print(f"  [ERROR]   {name} seed{seed}: {e}")
        return [], None


# ═══════════════════════════════════════════════════════════════════════════════
# Single ligand docking (3 seeds, take mean_best)
# ═══════════════════════════════════════════════════════════════════════════════
def dock_one(ligand_path: Path) -> dict:
    """
    Run docking for a single ligand with 3 seeds, taking mode1 (best) for each seed,
    and compute the mean of the best scores → mean_best_score.
    At the same time, retain mean_avg_score and mean_sd_score for subsequent analysis.
    """
    name = ligand_path.stem
    seed_bests = []  
    seed_avgs  = []   
    seed_sds   = []  
    best_pose  = None  
    best_score = None

    for seed in SEEDS:
        scores, out_pdbqt = run_vina_seed(ligand_path, seed)
        if not scores:
            continue
        mode1 = scores[0] 
        avg   = float(np.mean(scores))
        sd    = float(np.std(scores))
        seed_bests.append(mode1)
        seed_avgs.append(avg)
        seed_sds.append(sd)

        if best_score is None or mode1 < best_score:
            best_score = mode1
            best_pose  = str(out_pdbqt) if out_pdbqt else None

    if not seed_bests:
        return {
            'name': name, 'status': 'FAIL',
            'mean_best_score': None, 'mean_avg_score': None, 'mean_sd_score': None,
            'best_pose': None,
            'n_hbonds': None, 'n_hydrophobic': None,
            'n_pistack': None, 'n_total_contacts': None,
        }


    plip_result = count_interactions_plip(receptor_pdb, Path(best_pose)) if best_pose else \
        {'n_hbonds': None, 'n_hydrophobic': None, 'n_pistack': None, 'n_total_contacts': None}

    return {
        'name':            name,
        'status':          'OK',
        'mean_best_score': round(float(np.mean(seed_bests)), 4),
        'mean_avg_score':  round(float(np.mean(seed_avgs)),  4),
        'mean_sd_score':   round(float(np.mean(seed_sds)),   4),
        'best_pose':       best_pose,
        **plip_result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main docking loop
# ═══════════════════════════════════════════════════════════════════════════════
out_csv    = out_dir / f"pocket_{POCKET_ID}_batch_{BATCH_ID:04d}_summary.csv"
fieldnames = ['name', 'status',
              'mean_best_score', 'mean_avg_score', 'mean_sd_score',
              'best_pose',
              'n_hbonds', 'n_hydrophobic', 'n_pistack', 'n_total_contacts']


completed = set()
if out_csv.exists():
    try:
        with open(out_csv, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('status') == 'OK':
                    completed.add(row['name'])
        print(f"[Resume] {len(completed)} molecules have been completed, skipping...")
    except Exception:
        pass

to_dock      = [p for p in ligand_files if p.stem not in completed]
write_header = not out_csv.exists() or len(completed) == 0

print(f"Pending connection: {len(to_dock)} / {len(ligand_files)}")
print(f"exhaustiveness={EXHAUSTIVENESS}  num_modes={NUM_MODES}  "
      f"seeds={SEEDS}  cpu={CPU}")
print(f"output → {out_csv}\n")

with open(out_csv, 'a', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    for i, lp in enumerate(to_dock, 1):
        row = dock_one(lp)
        writer.writerow(row)
        fh.flush()
        score_str = f"{row['mean_best_score']:.3f}" if row['mean_best_score'] is not None else '-'
        print(f"  [{i}/{len(to_dock)}] [{row['status']}] {lp.stem}  "
              f"mean_best={score_str} kcal/mol")

ok_count = sum(1 for p in ligand_files
               if p.stem in completed or
               any(p.stem == lp.stem and dock_one.__doc__ for lp in to_dock))

print(f"\nDone → {out_csv}")
print(f"Pocket {POCKET_ID} / Batch {BATCH_ID} finished")

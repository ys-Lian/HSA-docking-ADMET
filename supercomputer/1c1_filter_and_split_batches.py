#!/usr/bin/env python3
"""
=====================
Pre-docking preprocessing: Ro5 filtering + batching

Steps:
1. Scan all .sdf files in SDF_DIR and calculate full Ro5 descriptors using RDKit (MW / LogP / HBD / HBA / RotBonds)
2. Filter out molecules that don't meet Ro5 criteria
3. Map the passing molecules to PDBQT files and batch them according to BATCH_SIZE
batches/batch_001.txt, batch_002.txt, ... (one absolute path to a pdbqt per line)
4. Print statistics and suggested SGE array job range

Note:
SDF and PDBQT filenames must match (same stem),
for example ligprep_out/sdf/aspirin.sdf ↔ ligprep_out/pdbqt/aspirin.pdbqt.
If SDF doesn’t exist, fall back to rough screening by heavy atom count from PDBQT (HA ≤ 38).

Usage:
  python 0_filter_and_split.py \
      --sdf_dir    ligprep_out/sdf \
      --ligand_dir ligprep_out/pdbqt \
      --batch_size 50 \
      --outdir     batches
"""

import argparse
from pathlib import Path

# ── Ro5 ────────────────────────────────────────────────────────────────
RO5_MW_MAX       = 500.0
RO5_LOGP_MAX     = 5.0
RO5_HBD_MAX      = 5
RO5_HBA_MAX      = 10
RO5_ROTBONDS_MAX = 10
RO5_HA_MAX       = 38


# ================================================================================
# Perform complete Ro5 screening from SDF (main path)
# ================================================================================

def check_ro5_from_sdf(sdf_path: Path):
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
        mol = None
        for m in suppl:
            if m is not None:
                mol = m
                break
        if mol is None:
            return None, {}

        mw       = Descriptors.MolWt(mol)
        logp     = Descriptors.MolLogP(mol)
        hbd      = rdMolDescriptors.CalcNumHBD(mol)
        hba      = rdMolDescriptors.CalcNumHBA(mol)
        rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        ha       = mol.GetNumHeavyAtoms()

        passed = (mw       <= RO5_MW_MAX       and
                  logp     <= RO5_LOGP_MAX     and
                  hbd      <= RO5_HBD_MAX      and
                  hba      <= RO5_HBA_MAX      and
                  rotbonds <= RO5_ROTBONDS_MAX)

        return passed, {
            'MW':       round(mw, 1),
            'LogP':     round(logp, 2),
            'HBD':      hbd,
            'HBA':      hba,
            'RotBonds': rotbonds,
            'HA':       ha,
        }
    except Exception:
        return None, {}


# ================================================================================
# Rough screening by heavy atom count from PDBQT (downgrade path)
# ================================================================================

def count_heavy_atoms_pdbqt(pdbqt_path: Path) -> int:
    count = 0
    try:
        for line in pdbqt_path.read_text(errors='replace').splitlines():
            if line.startswith(('ATOM', 'HETATM')):
                element  = line[76:78].strip() if len(line) > 76 else ''
                atomname = line[12:16].strip()
                symbol   = element if element else (atomname[0] if atomname else '')
                if symbol.upper() not in ('H', 'HD', ''):
                    count += 1
    except Exception:
        pass
    return count


def check_ro5_simple(pdbqt_path: Path):
    ha = count_heavy_atoms_pdbqt(pdbqt_path)
    return ha <= RO5_HA_MAX, {'HA': ha}


# ================================================================================
# Main
# ================================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Ro5 filtering (from SDF) + batching, for HPC array job use",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sdf_dir",     default="ligprep_out/sdf",
                    help="Ligand SDF directory (for complete Ro5 filtering)")
    ap.add_argument("--ligand_dir",  default="ligprep_out/pdbqt",
                    help="Ligand PDBQT directory (for actual docking use)")
    ap.add_argument("--batch_size",  type=int, default=50,
                    help="Number of molecules per batch")
    ap.add_argument("--outdir",      default="batches",
                    help="Batch list output directory")
    ap.add_argument("--n_pockets",   type=int, default=8,
                    help="Number of pockets (for printing SGE range)")
    args = ap.parse_args()

    sdf_dir    = Path(args.sdf_dir)
    ligand_dir = Path(args.ligand_dir)
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_pdbqt = sorted(ligand_dir.glob("*.pdbqt"))
    if not all_pdbqt:
        raise SystemExit(f"PDBQT file not found：{ligand_dir}")
    print(f"Total ligands (PDBQT): {len(all_pdbqt)}")


    try:
        from rdkit import Chem  # noqa
        use_rdkit = True
        print(f"RDKit available, performing complete Ro5 filtering from SDF")
        print(f"SDF directory: {sdf_dir}")
    except ImportError:
        use_rdkit = False
        print("Warning: RDKit not available, downgrading to rough screening by heavy atom count (HA ≤ 38)")


    passed     = []
    failed_ro5 = []
    fallback   = [] 

    for i, pdbqt in enumerate(all_pdbqt, 1):
        if i % 500 == 0:
            print(f"  Filtering progress: {i}/{len(all_pdbqt)}")

        if use_rdkit:
            sdf_path = sdf_dir / f"{pdbqt.stem}.sdf"
            if sdf_path.exists():
                ok, info = check_ro5_from_sdf(sdf_path)
                if ok is None:
                    ok, info = check_ro5_simple(pdbqt)
                    fallback.append(pdbqt.stem)
            else:
                ok, info = check_ro5_simple(pdbqt)
                fallback.append(pdbqt.stem)
        else:
            ok, info = check_ro5_simple(pdbqt)

        if ok:
            passed.append(pdbqt)
        else:
            failed_ro5.append((pdbqt.stem, info))

    print(f"\nFiltering results:")
    print(f"  Passed Ro5:   {len(passed)}")
    print(f"  Failed Ro5: {len(failed_ro5)}")
    if fallback:
        print(f"  Downgraded to rough screening (no SDF): {len(fallback)}")

    if failed_ro5:
        print(f"\n  Failed Ro5 molecules (first 10):")
        for stem, info in failed_ro5[:10]:
            print(f"    {stem}: {info}")

    # batches
    n_batches = (len(passed) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        chunk      = passed[b * args.batch_size : (b + 1) * args.batch_size]
        batch_file = outdir / f"batch_{b+1:04d}.txt"
        batch_file.write_text('\n'.join(str(p.resolve()) for p in chunk) + '\n')

    total_tasks = n_batches * args.n_pockets
    print(f"\nBatching results:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of batches:   {n_batches}")
    print(f"  Batch files: {outdir}/batch_0001.txt ~ batch_{n_batches:04d}.txt")
    print(f"\nSGE parameter suggestions:")
    print(f"  #$ -t 1-{total_tasks}")
    print(f"  Task ID : BATCH_ID  = (task_id - 1) // {args.n_pockets} + 1")
    print(f"                POCKET_ID = (task_id - 1) % {args.n_pockets} + 1")


    summary = outdir / 'filter_summary.txt'
    summary.write_text(
        f"total_input={len(all_pdbqt)}\n"
        f"passed_ro5={len(passed)}\n"
        f"failed_ro5={len(failed_ro5)}\n"
        f"fallback_simple={len(fallback)}\n"
        f"n_batches={n_batches}\n"
        f"batch_size={args.batch_size}\n"
        f"n_pockets={args.n_pockets}\n"
        f"total_tasks={total_tasks}\n"
    )
    print(f"\n✓ Filtering summary -> {summary}")
    print(f"✓ Batching completed, submit with: qsub run_docking.sh")
    print(f"  （Change TOTAL_TASKS in run_docking.sh to {total_tasks}）")


if __name__ == "__main__":
    main()

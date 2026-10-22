#!/usr/bin/env python3
"""
Ligand preparation pipeline: SMILES → standardised states → 3D → MMFF94 → PDBQT

Steps per compound:
  1. Dimorphite-DL protonation at target pH
  2. RDKit tautomer enumeration
  3. Stereoisomer enumeration (up to 2 undefined centres)
  4. ETKDGv3 3D embedding + MMFF94s minimisation
  5. Best conformer written to SDF → PDBQT via Meeko or Open Babel

Usage:
  python 0_prepare_ligands.py --in_csv molecules.csv --outdir ligprep_out --write_pdbqt

Input CSV must have columns "name" and "smiles" (or "id"/"chembl_id" + "smiles"/"canonical_smiles").
"""

import argparse, os, sys, csv, json, subprocess, shlex
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import concurrent.futures as futures

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize as Standardize
except Exception:
    try:
        from rdkit.Chem import rdMolStandardize as Standardize
    except Exception:
        Standardize = None
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def has_binary(name: str) -> bool:
    from shutil import which
    return which(name) is not None


def get_dimorphite_cmds(custom_cmd: Optional[str] = None) -> List[List[str]]:
    bases: List[List[str]] = []
    if custom_cmd:
        try:
            bases.append(shlex.split(custom_cmd))
        except Exception:
            bases.append([custom_cmd])
    try:
        import dimorphite_dl  # noqa: F401
        bases.append(["python", "-m", "dimorphite_dl"])
    except Exception:
        pass
    for name in ["dimorphite_dl.py", "dimorphite_dl", "dimorphite"]:
        if has_binary(name):
            bases.append([name])
    return bases


def standardize_parent(m: Chem.Mol) -> Chem.Mol:
    if Standardize is None:
        Chem.SanitizeMol(m)
        return m
    if hasattr(Standardize, "CleanupParameters"):
        m = Standardize.Cleanup(m, Standardize.CleanupParameters())
    else:
        m = Standardize.Cleanup(m)
    try:
        md = Standardize.MetalDisconnector()
        m = md.Disconnect(m)
    except Exception:
        pass
    if hasattr(Standardize, "FragmentParentParameters"):
        m = Standardize.FragmentParent(m, Standardize.FragmentParentParameters())
    else:
        m = Standardize.FragmentParent(m)
    reion = getattr(Standardize, "Reionize", None)
    if reion is not None:
        try:
            m = reion(m)
        except Exception:
            pass
    uncharger = getattr(Standardize, "Uncharger", None)
    if uncharger is not None:
        try:
            m = uncharger(m)
        except Exception:
            pass
    Chem.SanitizeMol(m)
    return m


def enumerate_protomers(smiles: str, ph: float, ph_tol: float,
                        timeout: float = 20.0, verbose: bool = False,
                        dimorphite_cmd: Optional[str] = None) -> List[str]:
    bases = get_dimorphite_cmds(dimorphite_cmd)
    if not bases:
        return [smiles]

    def _parse(stdout: str) -> List[str]:
        out, seen = [], set()
        for ln in stdout.splitlines():
            toks = ln.strip().split()
            if not toks:
                continue
            cand = toks[0]
            if Chem.MolFromSmiles(cand) is not None and cand not in seen:
                out.append(cand); seen.add(cand)
        return out

    min_ph = f"{ph - ph_tol:.2f}"
    max_ph = f"{ph + ph_tol:.2f}"
    variants = [
        [smiles, "--ph_min", min_ph, "--ph_max", max_ph],
        ["--smiles", smiles, "--min_ph", min_ph, "--max_ph", max_ph],
        [smiles],
    ]
    for base in bases:
        for args_list in variants:
            try:
                res = subprocess.run(base + args_list, capture_output=True, text=True, timeout=timeout)
                if res.returncode == 0:
                    outs = _parse(res.stdout)
                    if outs:
                        return outs
            except Exception:
                continue
    if verbose:
        sys.stderr.write(f"[warn] dimorphite failed for {smiles[:40]}; using input SMILES\n")
    return [smiles]


def enumerate_tautomers(m: Chem.Mol, max_tautomers: int = 8) -> List[Chem.Mol]:
    if Standardize is None:
        return [m]
    te_ctor = getattr(Standardize, "TautomerEnumerator", None)
    if te_ctor is None:
        return [m]
    te = te_ctor()
    enum = getattr(te, "Enumerate", None) or getattr(te, "enumerate", None)
    if enum is None:
        return [m]
    tauto = list(enum(m))
    uniq, seen = [], set()
    for t in tauto:
        k = Chem.MolToSmiles(t, isomericSmiles=True)
        if k not in seen:
            uniq.append(t); seen.add(k)
        if len(uniq) >= max_tautomers:
            break
    return uniq or [m]


def enumerate_stereo(m: Chem.Mol, max_centers: int = 2, max_enum: int = 8) -> List[Chem.Mol]:
    ch = Chem.FindMolChiralCenters(m, includeUnassigned=True)
    undef = [c for c in ch if c[1] == '?']
    if not undef or len(undef) > max_centers:
        return [m]
    try:
        opts = StereoEnumerationOptions(maxIsomers=max_enum, onlyUnassigned=True)
    except TypeError:
        opts = StereoEnumerationOptions()
    enums = list(EnumerateStereoisomers(m, options=opts))
    return [Chem.Mol(x) for x in enums] or [m]


def smiles_to_states(smiles: str, ph: float, ph_tol: float,
                     max_tautomers: int, stereo_max_centers: int,
                     verbose: bool = False,
                     dimorphite_cmd: Optional[str] = None) -> List[Tuple[Chem.Mol, Dict]]:
    out, seen = [], set()
    for pi, smi in enumerate(enumerate_protomers(smiles, ph, ph_tol, verbose=verbose, dimorphite_cmd=dimorphite_cmd)):
        m0 = Chem.MolFromSmiles(smi)
        if m0 is None:
            continue
        try:
            m_std = standardize_parent(m0)
        except Exception:
            continue
        for ti, mt in enumerate(enumerate_tautomers(m_std, max_tautomers)):
            for si, ms in enumerate(enumerate_stereo(mt, stereo_max_centers)):
                mH = Chem.AddHs(ms)
                key = Chem.MolToSmiles(Chem.RemoveHs(mH), isomericSmiles=True)
                if key in seen:
                    continue
                seen.add(key)
                out.append((mH, {"protomer_idx": pi, "tautomer_idx": ti, "stereo_idx": si,
                                  "canonical_smiles": Chem.MolToSmiles(ms, isomericSmiles=True)}))
    return out[:32]


# ---------------------------------------------------------------------------
# 3D embedding and minimisation
# ---------------------------------------------------------------------------

def embed_and_minimize(mH: Chem.Mol, n_confs: int, prune_rms: float,
                       max_iters: int, seed: int) -> Tuple:
    m = Chem.Mol(mH)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.pruneRmsThresh = prune_rms
    ps.useSmallRingTorsions = True
    try:
        ps.maxAttempts = 1000
    except Exception:
        pass
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=ps))
    if not cids:
        ps.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=ps))
    if not cids:
        return None, None, None, "Embed failed"

    props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
    energies = []
    if props is not None:
        for cid in cids:
            try:
                AllChem.MMFFOptimizeMolecule(m, confId=cid, mmffVariant="MMFF94s", maxIters=max_iters)
                ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=cid)
                energies.append(ff.CalcEnergy())
            except Exception:
                energies.append(float("inf"))
    if not energies or all(not np.isfinite(e) for e in energies):
        for cid in cids:
            try:
                AllChem.UFFOptimizeMolecule(m, confId=cid, maxIters=max_iters)
                ff = AllChem.UFFGetMoleculeForceField(m, confId=cid)
                energies.append(ff.CalcEnergy())
            except Exception:
                energies.append(float("inf"))
    if all(not np.isfinite(e) for e in energies):
        return None, None, None, "All energies inf"

    best_idx = int(np.nanargmin(energies))
    return m, int(cids[best_idx]), float(energies[best_idx]), "OK"


def write_sdf(m: Chem.Mol, conf_id: int, name: str, sdf_dir: Path) -> Path:
    sdf = sdf_dir / f"{name}.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(m, confId=conf_id)
    w.close()
    return sdf


# ---------------------------------------------------------------------------
# PDBQT conversion
# ---------------------------------------------------------------------------

def to_pdbqt_meeko(m: Chem.Mol, conf_id: int, out_path: Path) -> Tuple[bool, str]:
    try:
        from meeko import MoleculePreparation
    except Exception as e:
        return False, f"Meeko not installed: {e}"
    try:
        single = Chem.Mol(m)
        for other_id in [c.GetId() for c in single.GetConformers() if c.GetId() != conf_id]:
            single.RemoveConformer(other_id)
        prep = MoleculePreparation()
        # Try multiple Meeko API versions
        prepped = None
        for attr in ["MolFromRDKit", "from_rdkit"]:
            try:
                mod = __import__("meeko", fromlist=["RDKitMolCreate"])
                ctor = getattr(mod, "RDKitMolCreate", None)
                if ctor:
                    mk = getattr(ctor, attr, lambda x: None)(single)
                    prepped = prep.prepare(mk)
                    break
            except Exception:
                pass
        if prepped is None:
            prepped = prep.prepare(single)
        # Write PDBQT
        for method in ["write_pdbqt_string", "writePDBQTString", "to_pdbqt_string"]:
            f = getattr(prepped, method, None)
            if callable(f):
                txt = f()
                if isinstance(txt, str) and txt.strip():
                    out_path.write_text(txt)
                    return True, "OK"
        for cname in ("PDBQTWriter", "PDBQTWriterLegacy"):
            try:
                cls = getattr(__import__("meeko", fromlist=[cname]), cname, None)
                if cls:
                    w = cls()
                    for method in ("write_string", "write", "to_string"):
                        fn = getattr(w, method, None)
                        if callable(fn):
                            s = fn(prepped)
                            if isinstance(s, str) and s.strip():
                                out_path.write_text(s)
                                return True, "OK"
            except Exception:
                pass
        return False, "Meeko: could not write PDBQT"
    except Exception as e:
        return False, str(e)


def to_pdbqt_meeko_cli(sdf_path: Path, out_path: Path, timeout: float = 60.0) -> Tuple[bool, str]:
    for name in ["mk_prepare_ligand.py", "mk_prepare_ligand"]:
        if has_binary(name):
            res = subprocess.run([name, "-i", str(sdf_path), "-o", str(out_path)],
                                 capture_output=True, text=True, timeout=timeout)
            if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return True, "OK (meeko CLI)"
    return False, "Meeko CLI not found or failed"


def to_pdbqt_obabel(sdf_path: Path, out_path: Path, ph: float, timeout: float = 60.0) -> Tuple[bool, str]:
    if not has_binary("obabel"):
        return False, "obabel not found"
    res = subprocess.run(
        ["obabel", "-isdf", str(sdf_path), "-opdbqt", "-O", str(out_path), "-xh", "-p", f"{ph:.2f}", "-xp"],
        capture_output=True, text=True, timeout=timeout
    )
    if res.returncode != 0:
        return False, f"obabel failed: {res.stderr.strip()}"
    return True, "OK"


def obabel_smiles_to_pdbqt(smiles: str, out_path: Path, ph: float, timeout: float = 60.0) -> Tuple[bool, str]:
    if not has_binary("obabel"):
        return False, "obabel not found"
    res = subprocess.run(
        ["obabel", "-:", smiles, "-opdbqt", "-O", str(out_path), "-xh", "-xp", "-p", f"{ph:.2f}"],
        capture_output=True, text=True, timeout=timeout
    )
    if res.returncode != 0 or not (out_path.exists() and out_path.stat().st_size > 0):
        return False, res.stderr.strip() or "no output"
    return True, "OK (obabel direct)"


# ---------------------------------------------------------------------------
# Per-ligand worker
# ---------------------------------------------------------------------------

def process_one(record: Dict, args) -> Dict:
    name = str(record["name"]).strip()
    smiles = str(record["smiles"]).strip()
    sdf_dir = Path(args.outdir) / "sdf"
    pdbqt_dir = Path(args.outdir) / "pdbqt" if args.write_pdbqt else None

    out = {"name": name, "smiles": smiles, "status": "FAIL", "message": "",
           "n_states": 0, "best_energy": None, "pdbqt_path": None}

    m0 = Chem.MolFromSmiles(smiles)
    if m0 is None:
        out["message"] = "SMILES parse failed"
        return out

    states = smiles_to_states(
        smiles, ph=args.ph, ph_tol=args.ph_tol,
        max_tautomers=args.max_tautomers,
        stereo_max_centers=args.stereo_max_centers,
        verbose=args.verbose,
        dimorphite_cmd=getattr(args, "dimorphite_cmd", None),
    )

    results = []
    for (mH, meta) in states:
        m3d, cid, E, msg = embed_and_minimize(
            mH, n_confs=args.n_confs, prune_rms=args.prune_rms,
            max_iters=args.max_iters, seed=args.seed
        )
        if m3d is not None:
            results.append((E, m3d, cid, meta))

    if not results:
        # Salvage: try Open Babel directly
        if pdbqt_dir and has_binary("obabel"):
            ok, msg = obabel_smiles_to_pdbqt(smiles, pdbqt_dir / f"{name}.pdbqt", ph=args.ph)
            if ok:
                out["status"] = "OK"; out["message"] = "salvaged via obabel"
                out["pdbqt_path"] = str(pdbqt_dir / f"{name}.pdbqt")
                return out
        out["message"] = "3D embedding failed for all states"
        return out

    results.sort(key=lambda x: x[0])
    best_E, best_m, best_cid, best_meta = results[0]
    out["n_states"] = len(results)
    out["best_energy"] = float(best_E)

    sdf_path = write_sdf(best_m, best_cid, name, sdf_dir)

    if pdbqt_dir:
        pdbqt_path = pdbqt_dir / f"{name}.pdbqt"
        ok, msg = to_pdbqt_meeko(best_m, best_cid, pdbqt_path)
        if not ok:
            ok, msg = to_pdbqt_meeko_cli(sdf_path, pdbqt_path)
        if not ok:
            ok, msg = to_pdbqt_obabel(sdf_path, pdbqt_path, ph=args.ph)
        if ok:
            out["pdbqt_path"] = str(pdbqt_path)
            out["status"] = "OK"; out["message"] = msg
        else:
            out["status"] = "PARTIAL"; out["message"] = f"PDBQT failed: {msg}"
    else:
        out["status"] = "OK"; out["message"] = "OK (SDF only)"

    if args.verbose:
        print(f"  [{out['status']}] {name} — {out['message']}")
        sys.stdout.flush()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Prepare ligands from SMILES for AutoDock Vina docking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--in_csv", required=True,
                    help="Input CSV with 'name' and 'smiles' columns")
    ap.add_argument("--outdir", default="ligprep_out",
                    help="Output directory (sdf/ and pdbqt/ subdirs created automatically)")
    ap.add_argument("--write_pdbqt", action="store_true",
                    help="Convert SDF → PDBQT for Vina (requires Meeko or Open Babel)")
    ap.add_argument("--ph", type=float, default=7.4,
                    help="Target pH for protonation")
    ap.add_argument("--ph_tol", type=float, default=1.0,
                    help="±pH tolerance for Dimorphite-DL")
    ap.add_argument("--max_tautomers", type=int, default=6)
    ap.add_argument("--stereo_max_centers", type=int, default=2,
                    help="Enumerate stereo only if ≤ this many undefined chiral centres")
    ap.add_argument("--n_confs", type=int, default=100,
                    help="Number of conformers to generate per state")
    ap.add_argument("--max_iters", type=int, default=500,
                    help="Max MMFF minimisation iterations")
    ap.add_argument("--prune_rms", type=float, default=0.5,
                    help="RMSD threshold for conformer pruning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = use all CPU cores)")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Only read the first N rows from the input CSV (default: all)")
    ap.add_argument("--dimorphite_cmd", type=str, default=None,
                    help="Explicit path/command for Dimorphite-DL if not on PATH")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip ligands whose PDBQT already exists")
    ap.add_argument("--no_resume", action="store_true",
                    help="Reprocess all ligands even if PDBQT exists")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_base = Path(args.outdir)
    sdf_dir = out_base / "sdf"; sdf_dir.mkdir(parents=True, exist_ok=True)
    if args.write_pdbqt:
        (out_base / "pdbqt").mkdir(parents=True, exist_ok=True)

    # Load CSV; auto-detect column names
    if args.max_rows is not None and args.max_rows > 0:
        df = pd.read_csv(args.in_csv, nrows=args.max_rows)
    else:
        df = pd.read_csv(args.in_csv)
    cols_lower = {c.lower(): c for c in df.columns}
    smiles_col = next((cols_lower[k] for k in cols_lower if "smiles" in k), None)
    name_col = next((cols_lower[k] for k in cols_lower
                     if k in ("name", "id", "chembl_id") or k.endswith("_id")), None)
    if smiles_col is None:
        raise SystemExit("No SMILES column found in input CSV")
    if name_col is None:
        df["_name"] = [f"ligand_{i+1:05d}" for i in range(len(df))]
        name_col = "_name"

    records = df[[name_col, smiles_col]].rename(
        columns={name_col: "name", smiles_col: "smiles"}
    ).astype(str).to_dict(orient="records")

    # Resume: skip ligands with existing PDBQT
    if args.write_pdbqt and args.resume and not args.no_resume:
        pdbqt_dir = out_base / "pdbqt"
        before = len(records)
        records = [r for r in records
                   if not (pdbqt_dir / f"{r['name']}.pdbqt").exists()]
        skipped = before - len(records)
        if skipped:
            print(f"[resume] Skipped {skipped} already-prepared ligands; {len(records)} remaining")

    print(f"Preparing {len(records)} ligands → {out_base}/")
    n_workers = os.cpu_count() if args.workers == 0 else args.workers
    results = []

    if n_workers and n_workers > 1:
        with futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            jobs = {ex.submit(process_one, r, args): r for r in records}
            for i, j in enumerate(futures.as_completed(jobs), 1):
                try:
                    res = j.result()
                except Exception as e:
                    res = {"name": jobs[j]["name"], "status": "FAIL", "message": str(e)}
                results.append(res)
                print(f"[{i}/{len(records)}] [{res['status']}] {res['name']} — {res.get('message','')}")
                sys.stdout.flush()
    else:
        for i, r in enumerate(records, 1):
            res = process_one(r, args)
            results.append(res)
            print(f"[{i}/{len(records)}] [{res['status']}] {res['name']} — {res.get('message','')}")
            sys.stdout.flush()

    summary = pd.DataFrame(results)
    summary.to_csv(out_base / "ligprep_summary.csv", index=False)
    n_ok = (summary["status"] == "OK").sum()
    print(f"\nDone: {n_ok}/{len(records)} succeeded. Summary → {out_base}/ligprep_summary.csv")


if __name__ == "__main__":
    main()

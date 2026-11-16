"""Fix a 26-molecule, molecule-disjoint, timsTOF natural-product panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()

    mol = pd.read_parquet(args.cache / "molecules.parquet")
    spec = pd.read_parquet(args.cache / "spectra.parquet")
    categories = json.loads((args.metadata_dir / "categories.json").read_text())
    source_id = categories["ingest_lib"].index("enveda-np-examples")
    instrument_id = categories["instrument_type"].index("timsTOF")
    ingest = np.load(args.metadata_dir / "ingest_id.npy", mmap_mode="r")
    instrument = np.load(args.metadata_dir / "instrument_id.npy", mmap_mode="r")
    if len(ingest) != len(spec) or len(instrument) != len(spec):
        raise ValueError("Source metadata is not aligned with the spectrum cache")
    target_rows = np.flatnonzero(ingest == source_id)
    if len(target_rows) != 1184 or not np.all(instrument[target_rows] == instrument_id):
        raise ValueError("The expected 1,184 timsTOF example spectra were not found")
    mol_idx = spec.mol_idx.to_numpy(dtype=np.int64)
    split = mol.split.to_numpy()
    query_rows = target_rows[split[mol_idx[target_rows]] == "validation"]
    targets = np.unique(mol_idx[query_rows])
    if len(targets) != 26:
        raise ValueError(f"Expected 26 held-out example molecules, got {len(targets)}")
    if len(set(mol.key.iloc[targets]) & set(mol.key.loc[mol.split.eq("train")])):
        raise AssertionError("Molecule key leaks into training")
    if not np.all(spec.split.iloc[query_rows].eq("validation")):
        raise AssertionError("Training spectrum included among validation queries")

    formula_groups = {
        str(formula): group.index.to_numpy(dtype=np.int64)
        for formula, group in mol.groupby("formula", sort=False)
    }
    rng = np.random.default_rng(args.seed)
    rows = []
    for target in targets:
        formula = str(mol.formula.iloc[target])
        decoys = formula_groups[formula]
        decoys = decoys[decoys != target]
        if len(decoys) > 24:
            decoys = rng.choice(decoys, size=24, replace=False)
        candidates = np.r_[target, decoys]
        rng.shuffle(candidates)
        target_index = int(np.flatnonzero(candidates == target)[0])
        spectra = query_rows[mol_idx[query_rows] == target]
        if not len(spectra) or not np.all(instrument[spectra] == instrument_id):
            raise AssertionError("Missing timsTOF validation spectrum")
        rows.append({
            "molecule_key": str(mol.key.iloc[target]),
            "mol_idx": int(target),
            "formula": formula,
            "query_cache_rows": spectra.tolist(),
            "candidate_mol_indices": candidates.tolist(),
            "candidate_keys": mol.key.iloc[candidates].tolist(),
            "candidate_smiles": mol.smiles.iloc[candidates].tolist(),
            "target_index": target_index,
            "candidate_count": len(candidates),
            "query_spectra_count": len(spectra),
        })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    check = pd.read_parquet(args.output)
    if len(check) != 26 or check.candidate_count.tolist() != frame.candidate_count.tolist():
        raise AssertionError("Panel Parquet round-trip failed")
    report = {
        "panel_sha256": sha(args.output),
        "cache_molecules_sha256": sha(args.cache / "molecules.parquet"),
        "cache_spectra_sha256": sha(args.cache / "spectra.parquet"),
        "metadata_manifest_sha256": sha(args.metadata_dir / "manifest.json"),
        "seed": args.seed,
        "query_molecules": len(rows),
        "query_spectra": int(sum(row["query_spectra_count"] for row in rows)),
        "candidate_count_histogram": dict(sorted(Counter(frame.candidate_count).items())),
        "single_candidate_molecules": int((frame.candidate_count == 1).sum()),
        "candidate_rule": (
            "all same-full-formula cache structures if <=25; otherwise target plus "
            "24 random decoys with fixed seed; shuffle all candidate orders"
        ),
        "query_rule": "all accepted enveda-np-examples spectra of the 26 validation molecules",
        "instrument": "timsTOF",
        "molecule_key_train_overlap": 0,
        "comparison_note": (
            "Supplementary diagnostic selected after inspecting the 1,000-molecule results. "
            "Synthetic cache candidate pool, not the hidden competition candidate set."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

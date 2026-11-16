"""Build a fixed same-formula retrieval panel shared by three CASMI model arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(
    cache: Path,
    out: Path,
    seed: int = 20261107,
    molecules: int = 1000,
    candidates: int = 25,
    spectra: int = 4,
):
    mol = pd.read_parquet(cache / "molecules.parquet")
    spec = pd.read_parquet(cache / "spectra.parquet")
    audit = json.loads((cache / "fixed-split-audit.json").read_text())
    if (
        audit["overlap"] != 0
        or len(mol) != audit["molecule_counts"]["train"] + audit["molecule_counts"]["validation"]
    ):
        raise ValueError("Molecule-disjoint fixed split audit failed")
    formula_groups = {
        str(formula): group.index.to_numpy(dtype=np.int64)
        for formula, group in mol.groupby("formula", sort=False)
    }
    spectrum_groups = {
        int(index): group.index.to_numpy(dtype=np.int64)
        for index, group in spec.loc[spec.source.eq(0)].groupby("mol_idx", sort=False)
    }
    eligible = [
        int(index)
        for index in np.flatnonzero(mol.split.eq("validation").to_numpy())
        if index in spectrum_groups
        and len(formula_groups[str(mol.formula.iloc[index])]) >= candidates
    ]
    if len(eligible) < molecules:
        raise ValueError(f"Only {len(eligible)} eligible validation molecules")
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.asarray(eligible), molecules, replace=False)
    rows = []
    for target in selected:
        formula = str(mol.formula.iloc[target])
        decoys = formula_groups[formula]
        decoys = decoys[decoys != target]
        decoys = rng.choice(decoys, candidates - 1, replace=False)
        ids = np.r_[target, decoys]
        rng.shuffle(ids)
        target_index = int(np.flatnonzero(ids == target)[0])
        query_rows = spectrum_groups[target][:spectra]
        rows.append(
            {
                "molecule_key": mol.key.iloc[target],
                "mol_idx": int(target),
                "formula": formula,
                "query_cache_rows": query_rows.tolist(),
                "candidate_mol_indices": ids.tolist(),
                "candidate_keys": mol.key.iloc[ids].tolist(),
                "candidate_smiles": mol.smiles.iloc[ids].tolist(),
                "target_index": target_index,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    report = {
        "panel_sha256": sha256(out),
        "cache_fixed_split_sha256": audit["fixed_split_sha256"],
        "seed": seed,
        "molecules": len(rows),
        "candidates_per_molecule": candidates,
        "query_spectra_max": spectra,
        "eligible_validation_molecules": len(eligible),
        "candidate_rule": "target plus random same-formula molecules from full cache",
        "selection_note": (
            "Only formulas with at least 25 known structures; synthetic retrieval "
            "diagnostic, not competition candidate pool"
        ),
        "train_validation_key_overlap": 0,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.cache, args.out), indent=2))


if __name__ == "__main__":
    main()

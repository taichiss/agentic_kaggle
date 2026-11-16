"""Apply the previous source-stratified molecular split to a Saner cache.

Fails rather than silently assigning unknown, conflicting, or empty identities.
Run once after the upstream prepare step and before either training arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def apply(cache: Path, split_path: Path):
    split = pd.read_parquet(split_path, columns=["normalized_molecule_key", "split"])
    if split.isna().any().any() or not set(split["split"]).issubset({"train", "validation"}):
        raise ValueError("Missing or unknown split or molecule keys")
    assignment = split.drop_duplicates()
    if assignment.normalized_molecule_key.duplicated().any():
        raise ValueError("A molecule is assigned to multiple splits")
    mapping = assignment.set_index("normalized_molecule_key")["split"]
    molecules_path, spectra_path = cache / "molecules.parquet", cache / "spectra.parquet"
    mol = pd.read_parquet(molecules_path)
    spec = pd.read_parquet(spectra_path)
    if mol.key.isna().any() or mol.key.duplicated().any() or spec.key.isna().any():
        raise ValueError("Malformed preprocessed molecule keys")
    new_mol = mol.key.map(mapping)
    new_spec = spec.key.map(mapping)
    if new_mol.isna().any() or new_spec.isna().any():
        raise ValueError("Prepared identity absent from fixed split; abort")
    train = set(mol.loc[new_mol.eq("train"), "key"])
    validation = set(mol.loc[new_mol.eq("validation"), "key"])
    if not train or not validation or train & validation:
        raise ValueError("Invalid or overlapping split")
    if not spec.key.isin(mol.key).all():
        raise ValueError("Spectrum identity absent from molecule file")
    mol["split"], spec["split"] = new_mol, new_spec
    # All checks before writing; cache remains disposable and outside git.
    mol.to_parquet(molecules_path, index=False)
    spec.to_parquet(spectra_path, index=False)
    h = hashlib.sha256()
    with split_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    result = {
        "fixed_split_sha256": h.hexdigest(),
        "molecule_counts": mol["split"].value_counts().to_dict(),
        "spectrum_counts": spec["split"].value_counts().to_dict(),
        "overlap": len(train & validation),
    }
    (cache / "fixed-split-audit.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("split", type=Path)
    args = parser.parse_args()
    print(json.dumps(apply(args.cache, args.split), indent=2))


if __name__ == "__main__":
    main()

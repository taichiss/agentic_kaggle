"""Count a full four-source validation panel before deciding whether to evaluate it."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ("gnps", "riken", "massbank", "enveda-np-examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--comparison-panel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mol = pd.read_parquet(args.cache / "molecules.parquet")
    spec = pd.read_parquet(args.cache / "spectra.parquet")
    categories = json.loads((args.metadata_dir / "categories.json").read_text())
    ingest = np.load(args.metadata_dir / "ingest_id.npy", mmap_mode="r")
    mol_idx = spec.mol_idx.to_numpy(dtype=np.int64)
    split = mol.split.to_numpy()
    formula_sizes = mol.formula.value_counts()
    train_keys = set(mol.key[split == "train"])
    result = {"sources": {}, "total_query_molecule_source_pairs": 0,
              "total_query_spectra": 0, "total_multi_candidate_pairs": 0}
    for source in SOURCES:
        source_id = categories["ingest_lib"].index(source)
        rows = np.flatnonzero((ingest == source_id) & (split[mol_idx] == "validation"))
        targets, counts = np.unique(mol_idx[rows], return_counts=True)
        formula_count = formula_sizes.reindex(mol.formula.iloc[targets]).to_numpy(dtype=int)
        item = {
            "molecules": len(targets),
            "spectra": len(rows),
            "single_candidate_molecules": int(np.sum(formula_count == 1)),
            "multi_candidate_molecules": int(np.sum(formula_count > 1)),
            "mean_query_spectra_per_molecule": float(np.mean(counts)) if len(counts) else 0,
            "max_query_spectra_per_molecule": int(counts.max()) if len(counts) else 0,
            "mean_capped_candidate_count": float(np.minimum(formula_count, 25).mean())
            if len(targets) else 0,
            "train_structure_key_overlap": len(set(mol.key.iloc[targets]) & train_keys),
        }
        result["sources"][source] = item
        result["total_query_molecule_source_pairs"] += len(targets)
        result["total_query_spectra"] += len(rows)
        result["total_multi_candidate_pairs"] += item["multi_candidate_molecules"]
    result["estimated_model_query_encodes_for_5_models"] = (
        result["total_query_molecule_source_pairs"] * 5
    )
    if args.comparison_panel is not None:
        comparison = pd.read_parquet(args.comparison_panel)
        result["comparison_panel"] = {
            "queries": len(comparison),
            "spectra": int(sum(len(rows) for rows in comparison.query_cache_rows)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

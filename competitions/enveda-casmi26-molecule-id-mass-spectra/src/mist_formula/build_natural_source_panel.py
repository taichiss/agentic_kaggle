"""Build an exhaustive held-out four-source panel with fixed same-formula candidates."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = ("gnps", "riken", "massbank", "enveda-np-examples")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
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
    source_categories = categories["ingest_lib"]
    ingest = np.load(args.metadata_dir / "ingest_id.npy", mmap_mode="r")
    instrument = np.load(args.metadata_dir / "instrument_id.npy", mmap_mode="r")
    timstof_id = categories["instrument_type"].index("timsTOF")
    if len(ingest) != len(spec) or len(instrument) != len(spec):
        raise ValueError("source metadata does not align with spectrum cache")
    mol_idx = spec.mol_idx.to_numpy(dtype=np.int64)
    mol_split = mol.split.to_numpy()
    spec_split = spec.split.to_numpy()
    train_keys = set(mol.key[mol_split == "train"])
    formula_groups = {
        str(formula): group.index.to_numpy(dtype=np.int64)
        for formula, group in mol.groupby("formula", sort=False)
    }

    # Candidate selection depends on the target only, so a molecule measured in
    # two libraries receives identical candidates in both source-specific rows.
    candidate_cache = {}
    records = []
    by_source = {}
    for source in SOURCES:
        source_id = source_categories.index(source)
        query_rows = np.flatnonzero(
            (ingest == source_id) & (mol_split[mol_idx] == "validation")
        )
        if not np.all(spec_split[query_rows] == "validation"):
            raise AssertionError(f"non-validation spectrum in {source}")
        targets = np.unique(mol_idx[query_rows])
        if set(mol.key.iloc[targets]) & train_keys:
            raise AssertionError(f"training structure leaked into {source}")
        source_records = []
        for target in targets:
            target = int(target)
            formula = str(mol.formula.iloc[target])
            if target not in candidate_cache:
                pool = formula_groups[formula]
                decoys = pool[pool != target]
                rng = np.random.default_rng(args.seed + target)
                if len(decoys) > 24:
                    decoys = rng.choice(decoys, size=24, replace=False)
                candidates = np.r_[target, decoys]
                rng.shuffle(candidates)
                candidate_cache[target] = candidates
            candidates = candidate_cache[target]
            if (len(candidates) > 25 or target not in candidates
                    or len(set(candidates)) != len(candidates)):
                raise AssertionError("invalid candidate list")
            if not np.all(mol.formula.iloc[candidates].eq(formula)):
                raise AssertionError("cross-formula candidate")
            spectra = query_rows[mol_idx[query_rows] == target]
            if not len(spectra):
                raise AssertionError("query has no source-specific spectrum")
            timstof_count = int(np.sum(instrument[spectra] == timstof_id))
            source_records.append({
                "source": source,
                "molecule_key": str(mol.key.iloc[target]),
                "mol_idx": target,
                "formula": formula,
                "query_cache_rows": spectra.tolist(),
                "candidate_mol_indices": candidates.tolist(),
                "target_index": int(np.flatnonzero(candidates == target)[0]),
                "candidate_count": len(candidates),
                "query_spectra_count": len(spectra),
                "timsTOF_spectra_count": timstof_count,
                "query_all_timsTOF": timstof_count == len(spectra),
                "query_any_timsTOF": timstof_count > 0,
                "query_instrument_count": len(np.unique(instrument[spectra])),
            })
        records.extend(source_records)
        by_source[source] = {
            "molecules": len(source_records),
            "spectra": int(sum(row["query_spectra_count"] for row in source_records)),
            "single_candidate_molecules": int(sum(row["candidate_count"] == 1
                                                  for row in source_records)),
            "all_timsTOF_molecules": int(sum(row["query_all_timsTOF"]
                                            for row in source_records)),
            "any_timsTOF_molecules": int(sum(row["query_any_timsTOF"]
                                            for row in source_records)),
            "timsTOF_spectra": int(sum(row["timsTOF_spectra_count"]
                                        for row in source_records)),
        }

    frame = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    check = pd.read_parquet(args.output)
    if len(check) != len(records):
        raise AssertionError("Parquet round-trip length mismatch")
    candidate_content = "\n".join(
        f"{row.mol_idx}\t{','.join(str(i) for i in row.candidate_mol_indices)}"
        for row in frame.itertuples(index=False)
    )
    report = {
        "panel_sha256": sha(args.output),
        "candidate_lists_sha256": hashlib.sha256(candidate_content.encode()).hexdigest(),
        "cache_molecules_sha256": sha(args.cache / "molecules.parquet"),
        "cache_spectra_sha256": sha(args.cache / "spectra.parquet"),
        "source_metadata_sha256": sha(args.metadata_dir / "manifest.json"),
        "seed": args.seed,
        "sources": by_source,
        "molecule_source_pairs": len(frame),
        "unique_molecule_keys": int(frame.molecule_key.nunique()),
        "query_spectra": int(frame.query_spectra_count.sum()),
        "candidate_count_histogram": dict(sorted(Counter(frame.candidate_count).items())),
        "single_candidate_pairs": int((frame.candidate_count == 1).sum()),
        "train_structure_key_overlap": 0,
        "candidate_rule": (
            "All full-formula cache structures when <=25; otherwise target plus 24 decoys "
            "from a target-seeded random draw. Candidate order is shuffled with the same seed."
        ),
        "query_rule": "All accepted validation spectra measured in each named source",
        "comparison_note": (
            "Post-hoc supplementary panel. Candidate counts differ by formula and must be "
            "stratified; absolute MRR is not comparable with earlier fixed-25 panels."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

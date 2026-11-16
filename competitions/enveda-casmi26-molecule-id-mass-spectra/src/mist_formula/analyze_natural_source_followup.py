"""Add paired M1/weak and candidate-band results plus a split-leakage audit."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_natural_source_panel import paired_metrics


def paired_delta(rows: list[dict], first: str, second: str, *, seed: int) -> dict:
    keys = pd.factorize(
        np.asarray([row["molecule_key"] for row in rows], dtype=object), sort=False
    )[0]
    group_count = int(keys.max()) + 1
    first_rank = np.asarray([row[f"{first}_rank"] for row in rows])
    second_rank = np.asarray([row[f"{second}_rank"] for row in rows])
    delta = 1 / first_rank - 1 / second_rank
    sums = np.bincount(keys, weights=delta, minlength=group_count)
    counts = np.bincount(keys, minlength=group_count)
    rng = np.random.default_rng(seed)
    draws = rng.integers(group_count, size=(2000, group_count))
    bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "molecule_source_pairs": len(rows),
        "unique_structure_keys": group_count,
        "first": first,
        "second": second,
        "first_mrr25": float(np.mean(1 / first_rank)),
        "second_mrr25": float(np.mean(1 / second_rank)),
        "delta_mrr25_first_minus_second": float(np.mean(delta)),
        "delta_cluster_bootstrap_95pct": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "first_better": int(np.sum(first_rank < second_rank)),
        "second_better": int(np.sum(first_rank > second_rank)),
        "same_rank": int(np.sum(first_rank == second_rank)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ranks", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--A-manifest", type=Path, required=True)
    parser.add_argument("--M1-manifest", type=Path, required=True)
    parser.add_argument("--M2-manifest", type=Path, required=True)
    parser.add_argument("--M3-manifest", type=Path, required=True)
    parser.add_argument("--WEAK-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.ranks.read_text().splitlines()]
    panel = pd.read_parquet(args.panel)
    mol = pd.read_parquet(args.cache / "molecules.parquet")
    spec = pd.read_parquet(args.cache / "spectra.parquet")
    ingest = np.load(args.source_metadata / "ingest_id.npy", mmap_mode="r")
    sources = json.loads((args.source_metadata / "categories.json").read_text())[
        "ingest_lib"
    ]
    if len(rows) != len(panel):
        raise AssertionError("panel and rank rows differ")
    train_keys = set(mol.key[mol.split.eq("train")])
    query_keys = set(panel.molecule_key)
    overlap = len(query_keys & train_keys)
    if overlap:
        raise AssertionError("held-out query structures overlap training")
    train_decoy_slots = validation_decoy_slots = 0
    for item, row in zip(panel.itertuples(index=False), rows, strict=True):
        if item.source != row["source"] or item.mol_idx != row["mol_idx"]:
            raise AssertionError("panel and rank order differs")
        target = int(item.mol_idx)
        candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
        query = np.asarray(item.query_cache_rows, dtype=np.int64)
        if (mol.split.iloc[target] != "validation"
                or item.molecule_key != mol.key.iloc[target]
                or candidates[int(item.target_index)] != target
                or not np.all(spec.split.iloc[query].eq("validation"))
                or not np.all(spec.mol_idx.iloc[query] == target)
                or not np.all(ingest[query] == sources.index(item.source))
                or not np.all(mol.formula.iloc[candidates].eq(item.formula))):
            raise AssertionError("label, source, formula or split audit failed")
        decoys = candidates[candidates != target]
        train_decoy_slots += int(np.sum(mol.split.iloc[decoys].eq("train")))
        validation_decoy_slots += int(np.sum(mol.split.iloc[decoys].eq("validation")))
    manifests = {
        name: json.loads(getattr(args, f"{name}_manifest").read_text())
        for name in ("A", "M1", "M2", "M3", "WEAK")
    }
    cache_molecule_hashes = {x["cache_molecules_sha256"] for x in manifests.values()}
    cache_spectra_hashes = {x["cache_spectra_sha256"] for x in manifests.values()}
    splits = {x["split"] for x in manifests.values()}
    seeds = {x["seed"] for x in manifests.values()}
    if any(len(group) != 1 for group in (
        cache_molecule_hashes, cache_spectra_hashes, splits, seeds
    )):
        raise AssertionError("the five models do not share the same training split/cache")

    multi = [row for row in rows if row["candidate_count"] > 1]
    bands = {
        "2-5": [row for row in multi if 2 <= row["candidate_count"] <= 5],
        "6-24": [row for row in multi if 6 <= row["candidate_count"] <= 24],
        "25": [row for row in multi if row["candidate_count"] == 25],
    }
    followup = {
        "M1_minus_WEAK_multi": paired_delta(multi, "M1", "WEAK", seed=args.seed),
        "M1_minus_WEAK_by_source": {
            source: paired_delta(
                [row for row in multi if row["source"] == source],
                "M1", "WEAK", seed=args.seed + index + 1,
            )
            for index, source in enumerate(sources)
            if source in {row["source"] for row in multi}
        },
        "candidate_count_bands": {
            band: paired_metrics(group, seed=args.seed + 100 + index)
            for index, (band, group) in enumerate(bands.items())
        },
        "GNPS_candidate_count_histogram": dict(sorted(Counter(
            row["candidate_count"] for row in rows if row["source"] == "gnps"
        ).items())),
        "audit": {
            "query_molecule_source_pairs": len(panel),
            "unique_query_structure_keys": len(query_keys),
            "train_query_structure_key_overlap": overlap,
            "all_query_spectra_validation": True,
            "all_true_labels_validation": True,
            "all_source_assignments_verified": True,
            "all_candidates_same_full_formula": True,
            "train_decoy_candidate_slots": train_decoy_slots,
            "validation_decoy_candidate_slots": validation_decoy_slots,
            "same_training_cache_molecules_sha256": next(iter(cache_molecule_hashes)),
            "same_training_cache_spectra_sha256": next(iter(cache_spectra_hashes)),
            "same_training_split": next(iter(splits)),
            "same_training_seed": next(iter(seeds)),
            "note": (
                "Training split molecules appear only as decoy candidates. "
                "No validation query spectrum, true-label molecule, or query structure key "
                "appears in the training split."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(followup, indent=2) + "\n")
    print(json.dumps({"M1_minus_WEAK_multi": followup["M1_minus_WEAK_multi"],
                      "bands": {name: len(group) for name, group in bands.items()},
                      "audit": followup["audit"]}))


if __name__ == "__main__":
    main()

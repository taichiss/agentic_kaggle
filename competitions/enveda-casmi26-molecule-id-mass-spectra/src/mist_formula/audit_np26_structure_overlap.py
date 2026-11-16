"""Audit NP26 query structures against the molecule-disjoint training cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def structure_keys(
    smiles: str, enumerator: rdMolStandardize.TautomerEnumerator
) -> tuple[str, str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    raw_inchikey14 = Chem.MolToInchiKey(molecule)[:14]
    Chem.RemoveStereochemistry(molecule)
    tautomer = enumerator.Canonicalize(molecule)
    return raw_inchikey14, Chem.MolToSmiles(tautomer), Chem.MolToInchiKey(tautomer)[:14]


def subset_metrics(rows: list[dict], bootstrap_seed: int) -> dict:
    result = {"molecules": len(rows)}
    if not rows:
        return result
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.integers(len(rows), size=(10_000, len(rows)))
    baseline = np.array([row["A_rank"] for row in rows])
    for name in ("A", "M1", "M2", "M3"):
        ranks = np.array([row[f"{name}_rank"] for row in rows])
        entry = {
            "mrr25": float((1 / ranks).mean()),
            "top1": float((ranks == 1).mean()),
        }
        if name != "A":
            delta = 1 / ranks - 1 / baseline
            entry.update({
                "delta_mrr25_vs_A": float(delta.mean()),
                "paired_bootstrap_95pct": [
                    float(value)
                    for value in np.quantile(delta[sampled].mean(axis=1), [0.025, 0.975])
                ],
                "improved": int((ranks < baseline).sum()),
                "worsened": int((ranks > baseline).sum()),
                "unchanged": int((ranks == baseline).sum()),
            })
        result[name] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-molecules", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--per-molecule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261117)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")

    mol = pd.read_parquet(args.cache_molecules, columns=["key", "smiles", "formula", "split"])
    panel = pd.read_parquet(args.panel, columns=["mol_idx", "molecule_key", "formula"])
    rank_rows = [json.loads(line) for line in args.per_molecule.read_text().splitlines()]
    if len(panel) != 26 or len(rank_rows) != 26:
        raise ValueError("Expected a fixed 26-molecule panel and 26 rank records")
    if panel.mol_idx.tolist() != [row["mol_idx"] for row in rank_rows]:
        raise ValueError("Panel and rank rows are not aligned")
    train = mol.index[mol.split.eq("train")]
    if not mol.split.iloc[panel.mol_idx].eq("validation").all():
        raise ValueError("A panel molecule belongs to the training split")

    train_by_cache_key: dict[str, list[int]] = defaultdict(list)
    for index in train:
        train_by_cache_key[str(mol.key.iat[index])].append(int(index))
    train_formulas = set(panel.formula)
    relevant_train = [int(index) for index in train if mol.formula.iat[index] in train_formulas]
    enumerator = rdMolStandardize.TautomerEnumerator()
    target_indices = [int(index) for index in panel.mol_idx]
    keys = {
        index: structure_keys(str(mol.smiles.iat[index]), enumerator)
        for index in target_indices + relevant_train
    }
    train_by_raw_key: dict[str, list[int]] = defaultdict(list)
    train_by_tautomer_smiles: dict[tuple[str, str], list[int]] = defaultdict(list)
    train_by_tautomer_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in relevant_train:
        formula = str(mol.formula.iat[index])
        raw, tautomer_smiles, tautomer_key = keys[index]
        train_by_raw_key[raw].append(index)
        train_by_tautomer_smiles[formula, tautomer_smiles].append(index)
        train_by_tautomer_key[formula, tautomer_key].append(index)

    detailed = []
    for index, row in zip(target_indices, rank_rows, strict=True):
        raw, tautomer_smiles, tautomer_key = keys[index]
        formula = str(mol.formula.iat[index])
        cache_matches = train_by_cache_key[str(mol.key.iat[index])]
        raw_matches = train_by_raw_key[raw]
        tautomer_smiles_matches = train_by_tautomer_smiles[formula, tautomer_smiles]
        tautomer_key_matches = train_by_tautomer_key[formula, tautomer_key]
        union = sorted(set(
            cache_matches + raw_matches + tautomer_smiles_matches + tautomer_key_matches
        ))
        detailed.append({
            **row,
            "query_smiles": str(mol.smiles.iat[index]),
            "cache_key_matches_raw_inchikey14": str(mol.key.iat[index]) == raw,
            "raw_inchikey14": raw,
            "tautomer_canonical_smiles": tautomer_smiles,
            "tautomer_canonical_inchikey14": tautomer_key,
            "train_cache_key_match_indices": cache_matches,
            "train_raw_inchikey14_match_indices": raw_matches,
            "train_tautomer_smiles_match_indices": tautomer_smiles_matches,
            "train_tautomer_inchikey14_match_indices": tautomer_key_matches,
            "train_any_structure_match_indices": union,
            "train_any_structure_match_smiles": [str(mol.smiles.iat[value]) for value in union],
        })
    overlapping = [row for row in detailed if row["train_any_structure_match_indices"]]
    clean = [row for row in detailed if not row["train_any_structure_match_indices"]]
    report = {
        "panel_sha256": sha(args.panel),
        "cache_molecules_sha256": sha(args.cache_molecules),
        "per_molecule_ranks_sha256": sha(args.per_molecule),
        "train_molecules": len(train),
        "train_molecules_in_26_query_formula_groups": len(relevant_train),
        "query_molecules": len(detailed),
        "query_cache_key_equal_rdkit_raw_inchikey14": sum(
            row["cache_key_matches_raw_inchikey14"] for row in detailed
        ),
        "train_same_formula_cache_key_equal_rdkit_raw_inchikey14": sum(
            str(mol.key.iat[index]) == keys[index][0] for index in relevant_train
        ),
        "overlap_counts": {
            "cache_key_all_train": sum(
                bool(row["train_cache_key_match_indices"]) for row in detailed
            ),
            "raw_inchikey14_same_formula_train": sum(
                bool(row["train_raw_inchikey14_match_indices"]) for row in detailed
            ),
            "canonical_tautomer_smiles_same_formula_train": sum(
                bool(row["train_tautomer_smiles_match_indices"]) for row in detailed
            ),
            "canonical_tautomer_inchikey14_same_formula_train": sum(
                bool(row["train_tautomer_inchikey14_match_indices"]) for row in detailed
            ),
            "any_structure_key": len(overlapping),
        },
        "clean_query_molecule_keys": [row["molecule_key"] for row in clean],
        "overlapping_query_molecule_keys": [row["molecule_key"] for row in overlapping],
        "all_26": subset_metrics(detailed, args.bootstrap_seed),
        "clean_multi_candidate": subset_metrics(
            [row for row in clean if row["candidate_count"] > 1], args.bootstrap_seed
        ),
        "overlap_multi_candidate": subset_metrics(
            [row for row in overlapping if row["candidate_count"] > 1], args.bootstrap_seed
        ),
        "single_candidate_excluded_from_paired_interpretation": sum(
            row["candidate_count"] == 1 for row in clean
        ),
        "method": (
            "Compare cache keys to all train molecules; compute RDKit raw InChIKey14 and the "
            "stereo-removed canonical tautomer SMILES/InChIKey14 with the frozen A "
            "TautomerEnumerator definition for all queries and train molecules sharing a "
            "query molecular formula. Tautomerization preserves molecular formula."
        ),
        "interpretation_limit": (
            "This is a post hoc, 26-molecule supplementary panel. Structure-key screening does "
            "not establish absence of related scaffold or analog leakage, and confidence "
            "intervals are wide at this sample size."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "structure-overlap-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "per-molecule-structure-overlap.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in detailed)
    )
    print(json.dumps({
        "overlap_counts": report["overlap_counts"],
        "train_formula_pool": len(relevant_train),
    }))


if __name__ == "__main__":
    main()

"""Evaluate a fixed-generation CE analog rank-feature ablation on external80.

All prediction CSVs must exist before labels are read.  The initial Engine1
candidate parity check compares merged-generated keys with an independently
saved G0 diagnostic; the parent may still select different final top-25 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/derived/candidate-rescue-corrected-v2"
CE = ROOT / "data/derived/ce-views-20261117"
BASELINE = Path("/kaggle/working/submission_0477_pre_ensemble.csv")
G1 = CE / "g1-local-mount-v3/working/submission.csv"
RANK_ONLY = CE / "rank-only-local-mount-v1/working/submission.csv"
AUDIT = CE / "rank-only-local-mount-v1/working/ce-rank-only-audit.json"
RANK_POOL_META = CE / "rank-only-local-mount-v1/working/work/pool_meta.parquet"
G0_GENERATION = CE / "generator-external80/per-molecule.jsonl"
OUTPUT = CE / "rank-only-local-mount-v1/evaluation.json"
BASELINE_SHA256 = "2492e1ae70335c67a87c1d756964d52bfc3c299e86aeeb6f6a7d351b4909db47"
POOL_META_SHA256 = "82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_key(smiles: str, canon: rdMolStandardize.TautomerEnumerator) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToInchiKey(canon.Canonicalize(mol))[:14]


def candidate_list(smiles: str) -> list[str]:
    result = smiles.split(";")
    if len(result) != 25 or any(not candidate for candidate in result):
        raise ValueError("Expected exactly 25 nonempty candidate SMILES")
    return result


def paired_interval(delta: np.ndarray, seed: int = 20261117) -> list[float]:
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(delta), size=(5000, len(delta)))
    return [float(x) for x in np.quantile(delta[picks].mean(axis=1), [0.025, 0.975])]


def evaluate(
    *, baseline_path: Path, g1_path: Path, rank_only_path: Path,
    audit_path: Path, rank_pool_meta_path: Path, g0_generation_path: Path, labels_path: Path,
    test_path: Path, output_path: Path,
) -> dict:
    paths = [baseline_path, g1_path, rank_only_path, audit_path,
             rank_pool_meta_path, g0_generation_path, labels_path, test_path]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    if sha(baseline_path) != BASELINE_SHA256:
        raise ValueError("External80 original 0.477 control changed")
    if sha(rank_pool_meta_path) != POOL_META_SHA256:
        raise ValueError("Candidate pool differs from G0's frozen 0.477 pool")
    predictions = {
        "G0": pd.read_csv(baseline_path, dtype=str),
        "G1": pd.read_csv(g1_path, dtype=str),
        "rank_only": pd.read_csv(rank_only_path, dtype=str),
    }
    mids = predictions["G0"].molecule_id.tolist()
    if len(mids) != 80 or len(set(mids)) != 80:
        raise ValueError("Expected 80 external molecules")
    for name, frame in predictions.items():
        if frame.molecule_id.tolist() != mids:
            raise ValueError(f"{name} query order differs")
        if frame.columns.tolist() != ["molecule_id", "smiles"]:
            raise ValueError(f"{name} submission schema differs")

    history = json.loads(audit_path.read_text())
    g0_records = [json.loads(line) for line in g0_generation_path.read_text().splitlines()]
    if len(history) != 80 or len(g0_records) != 80:
        raise ValueError("Expected 80 generator calls and 80 G0 diagnostics")
    if [row["molecule_id"] for row in g0_records] != mids:
        raise ValueError("Independent G0 generator query order differs")
    candidate_generation = []
    for mid, record, saved in zip(mids, g0_records, history, strict=True):
        expected_keys = set(record["merged"]["generated_keys"])
        actual_keys = set(saved["generated_keys"])
        candidate_generation.append({
            "molecule_id": mid,
            "target_da_difference": abs(float(record["neutral_mass"]) - float(saved["target"])),
            "generated_set_identical": expected_keys == actual_keys,
            "generated_keys_unique": len(saved["generated_keys"]) == len(actual_keys),
            "g0_merged_parent_top6_identical": (
                record["merged"]["analog_sid_top6"] == saved["generator_analog_sids"]
            ),
            "g0_generated_count": len(expected_keys),
            "rank_only_generated_count": len(actual_keys),
            "rank_analog_sids_changed": saved["rank_analog_sids"] != saved["generator_analog_sids"],
        })

    # Load labels after all three predictions and their candidate-generation
    # audit have been read and validated.
    labels = pd.read_csv(labels_path, dtype=str).set_index("molecule_id")
    if set(labels.index) != set(mids):
        raise ValueError("Labels and predictions differ")
    canon = rdMolStandardize.TautomerEnumerator()
    rows = []
    for index, mid in enumerate(mids):
        truth = labels.loc[mid, "metric_key"]
        arms = {}
        for name, frame in predictions.items():
            candidates = candidate_list(frame.iloc[index].smiles)
            rank = next(
                (i for i, smiles in enumerate(candidates, 1)
                 if metric_key(smiles, canon) == truth), 0
            )
            arms[name] = {
                "rank": rank,
                "rr25": 1.0 / rank if rank else 0.0,
                "top1": candidates[0],
                "candidate_set": set(candidates),
            }
        rows.append({
            "molecule_id": mid,
            "G0_rank": arms["G0"]["rank"],
            "G1_rank": arms["G1"]["rank"],
            "rank_only_rank": arms["rank_only"]["rank"],
            "G0_rr25": arms["G0"]["rr25"],
            "G1_rr25": arms["G1"]["rr25"],
            "rank_only_rr25": arms["rank_only"]["rr25"],
            "G1_final_candidate_set_changed": arms["G1"]["candidate_set"] != arms["G0"]["candidate_set"],
            "rank_only_final_candidate_set_changed": arms["rank_only"]["candidate_set"] != arms["G0"]["candidate_set"],
            "G1_top1_changed": arms["G1"]["top1"] != arms["G0"]["top1"],
            "rank_only_top1_changed": arms["rank_only"]["top1"] != arms["G0"]["top1"],
            "G1_vs_rank_only_final_candidate_set_changed": (
                arms["G1"]["candidate_set"] != arms["rank_only"]["candidate_set"]
            ),
            "G1_vs_rank_only_top1_changed": arms["G1"]["top1"] != arms["rank_only"]["top1"],
            "G1_only_candidates": len(arms["G1"]["candidate_set"] - arms["G0"]["candidate_set"]),
            "rank_only_only_candidates": len(arms["rank_only"]["candidate_set"] - arms["G0"]["candidate_set"]),
        })
    report = {
        "scope": "external80 full frozen 0.477 parent; local only, no Kaggle LB",
        "matching": "RDKit 2026.03.3, remove stereo, canonical tautomer, InChIKey14",
        "n_molecules": len(rows),
        "initial_engine1_candidate_parity": {
            "generated_set_identical": sum(r["generated_set_identical"] for r in candidate_generation),
            "generated_keys_unique": sum(r["generated_keys_unique"] for r in candidate_generation),
            "g0_merged_parent_top6_identical": sum(
                r["g0_merged_parent_top6_identical"] for r in candidate_generation
            ),
            "max_target_da_difference": max(r["target_da_difference"] for r in candidate_generation),
            "pool_meta_sha256": sha(rank_pool_meta_path),
            "g0_generated_count": sum(r["g0_generated_count"] for r in candidate_generation),
            "rank_only_generated_count": sum(r["rank_only_generated_count"] for r in candidate_generation),
            "rank_analog_top6_changed": sum(r["rank_analog_sids_changed"] for r in candidate_generation),
            "note": "Same pool and merged-generated candidates at Engine1 input; final top25 may differ due ranking/truncation.",
        },
        "arms": {},
        "comparisons": {},
        "input_sha256": {str(path): sha(path) for path in paths},
        "limitations": [
            "External80 has 80 selected structures; it is not hidden Public LB.",
            "Upstream ranker/model training overlap with external80 has not been independently audited.",
            "Fixed generation isolates rank features at Engine1 input; downstream top25 can change through ranking and truncation.",
        ],
        "per_molecule": rows,
        "candidate_generation_per_molecule": candidate_generation,
    }
    for name in predictions:
        rr = np.asarray([row[f"{name}_rr25"] for row in rows])
        report["arms"][name] = {
            "mrr25": float(rr.mean()),
            "top1": int(np.count_nonzero(rr == 1.0)),
            "recall25": int(np.count_nonzero(rr)),
        }
    for name in ("G1", "rank_only"):
        delta = np.asarray([row[f"{name}_rr25"] - row["G0_rr25"] for row in rows])
        same_final = np.asarray([not row[f"{name}_final_candidate_set_changed"] for row in rows])
        report["comparisons"][f"{name}_minus_G0"] = {
            "delta_mrr25": float(delta.mean()),
            "paired_bootstrap_95pct": paired_interval(delta),
            "improved": int(np.count_nonzero(delta > 0)),
            "worsened": int(np.count_nonzero(delta < 0)),
            "unchanged": int(np.count_nonzero(delta == 0)),
            "final_candidate_set_changed": sum(row[f"{name}_final_candidate_set_changed"] for row in rows),
            "top1_changed": sum(row[f"{name}_top1_changed"] for row in rows),
            "new_final_candidates_total": sum(row[f"{name}_only_candidates"] for row in rows),
            "same_final_set_n": int(same_final.sum()),
            "same_final_set_delta_mrr25": float(delta[same_final].mean()) if same_final.any() else None,
            "same_final_set_rank_changed": int(np.count_nonzero(delta[same_final])),
            "changed_final_set_delta_mrr25": float(delta[~same_final].mean()) if (~same_final).any() else None,
        }
    g1_minus_rank = np.asarray([row["G1_rr25"] - row["rank_only_rr25"] for row in rows])
    report["comparisons"]["G1_minus_rank_only"] = {
        "delta_mrr25": float(g1_minus_rank.mean()),
        "paired_bootstrap_95pct": paired_interval(g1_minus_rank),
        "improved": int(np.count_nonzero(g1_minus_rank > 0)),
        "worsened": int(np.count_nonzero(g1_minus_rank < 0)),
        "unchanged": int(np.count_nonzero(g1_minus_rank == 0)),
        "final_candidate_set_changed": sum(
            row["G1_vs_rank_only_final_candidate_set_changed"] for row in rows
        ),
        "top1_changed": sum(row["G1_vs_rank_only_top1_changed"] for row in rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return {k: v for k, v in report.items() if k not in {"per_molecule", "candidate_generation_per_molecule"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--g1", type=Path, default=G1)
    parser.add_argument("--rank-only", type=Path, default=RANK_ONLY)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--rank-pool-meta", type=Path, default=RANK_POOL_META)
    parser.add_argument("--g0-generation", type=Path, default=G0_GENERATION)
    parser.add_argument("--labels", type=Path, default=PANEL / "labels-local-only.csv")
    parser.add_argument("--test", type=Path, default=PANEL / "test.parquet")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        baseline_path=args.baseline, g1_path=args.g1, rank_only_path=args.rank_only,
        audit_path=args.audit, rank_pool_meta_path=args.rank_pool_meta,
        g0_generation_path=args.g0_generation,
        labels_path=args.labels, test_path=args.test, output_path=args.output,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

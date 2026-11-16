"""Probe same-formula rank-1 swaps on the frozen labeled 80-parent proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bcf_rerank import formula, rerank
from evaluate_mixup_parent import rank
from rdkit import RDLogger

PARENT_WEIGHT = 0.25
GATES = (("any_positive", 0.0), ("margin_0_02", 0.02))
EXPECTED_CONTROL_SHA = "2492e1ae70335c67a87c1d756964d52bfc3c299e86aeeb6f6a7d351b4909db47"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def top1_swap(names: list[str], scores: np.ndarray, gate: float) -> tuple[list[str], dict]:
    """Run frozen RRF once, then swap rank 1 within its formula if the gate passes."""
    order, rrf_diag = rerank(names, scores, weight=PARENT_WEIGHT)
    parent_formula = formula(names[0])
    eligible = [
        index
        for index in range(1, len(names))
        if parent_formula is not None
        and formula(names[index]) == parent_formula
        and np.isfinite(scores[index])
    ]
    margin = None
    best = None
    if eligible and np.isfinite(scores[0]):
        best = max(eligible, key=lambda index: (scores[index], -index))
        margin = float(scores[best] - scores[0])
        if margin > 0 and margin >= gate:
            slot = order.index(best)
            order[0], order[slot] = order[slot], order[0]
    ranked = [names[index] for index in order]
    if sorted(ranked) != sorted(names) or any(
        formula(before) != formula(after) for before, after in zip(names, ranked, strict=True)
    ):
        raise AssertionError("Rank-1 swap crossed formula slots or changed candidate set")
    return ranked, {
        "eligible_same_formula_challengers": len(eligible),
        "best_original_rank": None if best is None else best + 1,
        "best_score_margin": margin,
        "promoted": ranked[0] != names[0],
        "rrf_missing_groups": rrf_diag["missing_groups"],
    }


def reciprocal(ranks: list[int | None]) -> np.ndarray:
    return np.asarray([1 / value if value else 0 for value in ranks], dtype=float)


def metrics(ranks: list[int | None]) -> dict:
    arr = reciprocal(ranks)
    return {
        "mrr25": float(arr.mean()),
        "top1": int(sum(value == 1 for value in ranks)),
        "truth_present": int(sum(value is not None for value in ranks)),
    }


def paired(tested: list[int | None], baseline: list[int | None], seed: int) -> dict:
    delta = reciprocal(tested) - reciprocal(baseline)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(delta), size=(5000, len(delta)))
    ci = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
    return {
        "delta_mrr25": float(delta.mean()),
        "bootstrap_95pct": [float(value) for value in ci],
        "rescued": int(np.count_nonzero(delta > 0)),
        "worsened": int(np.count_nonzero(delta < 0)),
        "unchanged": int(np.count_nonzero(delta == 0)),
        "top1_rescued": int(
            sum(before != 1 and after == 1 for before, after in zip(baseline, tested, strict=True))
        ),
        "top1_lost": int(
            sum(before == 1 and after != 1 for before, after in zip(baseline, tested, strict=True))
        ),
    }


def run(args: argparse.Namespace) -> dict:
    RDLogger.DisableLog("rdApp.*")
    if sha(args.control) != EXPECTED_CONTROL_SHA:
        raise ValueError("Parent 80-row control changed")
    parent = pd.read_csv(args.control, dtype=str, keep_default_na=False)
    truth = pd.read_csv(args.labels, dtype=str).set_index("molecule_id").metric_key.to_dict()
    saved = [json.loads(line) for line in args.scores.read_text().splitlines()]
    if (
        len(parent) != 80
        or len(saved) != 80
        or set(parent.molecule_id) != set(truth)
        or set(parent.molecule_id) != {row["molecule_id"] for row in saved}
    ):
        raise ValueError("Parent, labels, and saved scores do not share 80 IDs")
    saved_by_id = {row["molecule_id"]: row for row in saved}
    records = []
    for item in parent.itertuples(index=False):
        mid = item.molecule_id
        names = item.smiles.split(";")
        row = saved_by_id[mid]
        actual = truth[mid]
        parent_rank = rank(names, actual)
        if parent_rank != row["parent_rank"]:
            raise ValueError(f"Saved parent rank differs at {mid}")
        record = {"molecule_id": mid, "candidate_count": len(names), "parent_rank": parent_rank}
        for model in ("A", "M1"):
            scores = np.asarray(row[model + "_scores"], dtype=float)
            if scores.shape != (len(names),) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid {model} score array at {mid}")
            fixed_order, _ = rerank(names, scores, weight=PARENT_WEIGHT)
            fixed = [names[index] for index in fixed_order]
            if fixed[0] != names[0] or sorted(fixed) != sorted(names):
                raise AssertionError("Frozen RRF changed protected rank 1 or candidates")
            fixed_rank = rank(fixed, actual)
            if fixed_rank != row[model + "_rank"]:
                raise ValueError(f"Saved {model} fixed rank differs at {mid}")
            record[model + "_fixed_rank"] = fixed_rank
            for gate_name, gate in GATES:
                ranked, diag = top1_swap(names, scores, gate)
                record[model + "_" + gate_name] = {
                    "rank": rank(ranked, actual),
                    **diag,
                }
        records.append(record)

    baseline_parent = [row["parent_rank"] for row in records]
    report = {
        "scope": "Frozen 80-molecule 0.477-parent proxy; no Kaggle test truth or LB submission",
        "parent_rank_fusion_weight": PARENT_WEIGHT,
        "top1_policy": (
            "After one frozen RRF pass, swap original rank 1 with the highest-scoring "
            "same-formula challenger only when its raw soft-Tanimoto margin is positive "
            "and passes the fixed gate. Other ranks retain the RRF order."
        ),
        "gates": {name: gate for name, gate in GATES},
        "rows": 80,
        "parent": metrics(baseline_parent),
        "models": {},
        "evidence_sha256": {
            "control": sha(args.control),
            "labels": sha(args.labels),
            "saved_scores": sha(args.scores),
        },
        "limitations": (
            "The saved 80-row proxy has only 54 truth-present queries and was selected locally. "
            "These are two fixed diagnostic gates, not a threshold search. Results do not "
            "estimate hidden test accuracy or justify a Kaggle submission by themselves."
        ),
    }
    for model in ("A", "M1"):
        fixed_ranks = [row[model + "_fixed_rank"] for row in records]
        result = {"fixed_top1": metrics(fixed_ranks), "unlock": {}}
        result["fixed_top1"]["paired_vs_parent"] = paired(fixed_ranks, baseline_parent, args.seed)
        for gate_name, _gate in GATES:
            entries = [row[model + "_" + gate_name] for row in records]
            tested = [entry["rank"] for entry in entries]
            result["unlock"][gate_name] = {
                **metrics(tested),
                "higher_score_eligible": sum(
                    entry["best_score_margin"] is not None and entry["best_score_margin"] > 0
                    for entry in entries
                ),
                "top1_swaps": sum(entry["promoted"] for entry in entries),
                "missing_formula_groups": sum(entry["rrf_missing_groups"] for entry in entries),
                "paired_vs_fixed": paired(tested, fixed_ranks, args.seed),
                "paired_vs_parent": paired(tested, baseline_parent, args.seed),
            }
        report["models"][model] = result
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"parent": report["parent"], "models": report["models"]}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

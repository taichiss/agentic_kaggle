"""Compare frozen Engine1 rankings for all 200 CE-matched Enveda groups.

This reuses the validated six-case evaluator with an unselected full-panel
group loader. The ranking excludes all truth reference SIDs and the truth pool
PID; frozen model training overlap remains unaudited. It is not the complete
competition pipeline (Engine2, PubChem, final fusion are omitted).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import evaluate_ce_union_rank_cases as base


def all_groups():
    g4_rows = base.read_jsonl(base.G4_RECORDS)
    g5_rows = base.read_jsonl(base.G5_RECORDS)
    if len(g4_rows) != 200 or len(g5_rows) != 200:
        raise ValueError("Expected the original 200-group CE-matched panel")
    g4 = {row["truth"]: row for row in g4_rows}
    g5 = {row["truth"]: row for row in g5_rows}
    if len(g4) != 200 or set(g4) != set(g5):
        raise ValueError("G4 and G5 validation truth sets differ")
    groups, selection = base.holdout.select_groups(200, 20261117)
    if selection["selected_groups"] != 200 or len(groups) != 200:
        raise ValueError("Held-out group selection changed")
    for item in groups:
        key = item["key"]
        if key not in g4 or item["mol_idx"] != g4[key]["mol_idx"] or item["cache_rows"] != g4[key]["cache_rows"]:
            raise ValueError(f"G4 query mismatch for {key}")
        if item["mol_idx"] != g5[key]["mol_idx"] or item["cache_rows"] != g5[key]["cache_rows"]:
            raise ValueError(f"G5 query mismatch for {key}")
    return groups, g4, g5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=base.DERIVED / "enveda200-engine1-rank",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--postprocess-only", action="store_true",
        help="Recompute summary statistics from existing per-group output.",
    )
    args = parser.parse_args()
    if args.postprocess_only:
        report = json.loads((args.output / "report.json").read_text())
    else:
        base.selected_groups = all_groups
        report = base.run(args.output, args.device)
    rows = base.read_jsonl(args.output / "per-group.jsonl")
    if len(rows) != 200 or report["n_groups"] != 200:
        raise ValueError("Expected results for all 200 CE-matched groups")
    rng = np.random.default_rng(20261117)
    indices = rng.integers(0, len(rows), size=(5000, len(rows)))
    ci = {}
    for mode in ("g4", "g5"):
        paired = np.asarray([
            row[mode]["reciprocal_rank_25"] - row["g0"]["reciprocal_rank_25"]
            for row in rows
        ])
        ci[f"{mode}_minus_g0"] = {
            "mean_difference": float(paired.mean()),
            "ci95": np.quantile(paired[indices].mean(axis=1), [0.025, 0.975]).tolist(),
        }
    report["paired_bootstrap"] = {
        "seed": 20261117,
        "replicates": 5000,
        "sampling_unit": "molecule",
        "quantile_method": "linear",
        **ci,
    }
    report["scope"] = (
        "200 molecule-disjoint CE-matched Enveda-180 groups; frozen 0.477 "
        "Engine1 full features, ranker and pool popularity; no Engine2, "
        "PubChem or final fusion"
    )
    report["meaning"] = (
        "selected local Engine1 diagnostic; not full-pipeline MRR@25, "
        "unseen-model CV or Kaggle Public LB"
    )
    report["input_sha256"]["full_panel_evaluator"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["per_group_sha256"] = hashlib.sha256(
        (args.output / "per-group.jsonl").read_bytes()
    ).hexdigest()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

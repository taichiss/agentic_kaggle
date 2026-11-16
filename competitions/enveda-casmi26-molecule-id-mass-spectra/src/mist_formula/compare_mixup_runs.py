"""Paired molecule-level comparison of a mixup run with the fixed A panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--treatment-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()

    baseline = {}
    for line in args.baseline_jsonl.read_text().splitlines():
        row = json.loads(line)
        key = int(row["mol_idx"])
        if key in baseline:
            raise ValueError(f"duplicate baseline molecule {key}")
        baseline[key] = int(row["A_rank"])
    report = json.loads(args.treatment_json.read_text())
    result = report.get("validation", report)
    queries = [int(x) for x in result["query_mols"]]
    ranks = np.asarray(result["ranks"], dtype=int)
    if len(queries) != len(ranks) or len(queries) != len(set(queries)):
        raise ValueError("treatment queries are missing or repeated")
    if set(queries) != set(baseline):
        raise ValueError("treatment and A query molecules differ")
    if np.any(ranks < 1) or np.any(ranks > 25):
        raise ValueError("invalid treatment rank")
    a = np.asarray([baseline[key] for key in queries], dtype=int)
    delta = 1 / ranks - 1 / a
    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, len(delta), size=(5000, len(delta)))
    ci = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
    summary = {
        "rows": len(delta),
        "baseline_mrr25": float(np.mean(1 / a)),
        "treatment_mrr25": float(np.mean(1 / ranks)),
        "delta_mrr25": float(np.mean(delta)),
        "paired_bootstrap_95pct": [float(v) for v in ci],
        "baseline_top1": float(np.mean(a == 1)),
        "treatment_top1": float(np.mean(ranks == 1)),
        "improved": int(np.sum(ranks < a)),
        "worsened": int(np.sum(ranks > a)),
        "unchanged": int(np.sum(ranks == a)),
        "bootstrap_seed": args.seed,
        "baseline_jsonl": str(args.baseline_jsonl),
        "treatment_json": str(args.treatment_json),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

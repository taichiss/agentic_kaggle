"""Evaluate predeclared two-model score fusions on the fixed 80-parent proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bcf_rerank import rerank_fused
from evaluate_mixup_parent import rank
from rdkit import RDLogger

PAIRS = (("A", "M1"), ("A", "M2"), ("M1", "M2"), ("A", "M3"), ("M1", "M3"))


def run(proxy: Path, control: Path, labels: Path, output: Path) -> dict:
    RDLogger.DisableLog("rdApp.*")
    records = [json.loads(line) for line in proxy.read_text().splitlines()]
    parent = pd.read_csv(control, dtype=str, keep_default_na=False).set_index("molecule_id")
    truth = pd.read_csv(labels, dtype=str).set_index("molecule_id").metric_key.to_dict()
    if len(records) != len(parent) or set(parent.index) != {r["molecule_id"] for r in records}:
        raise ValueError("Proxy, parent and labels do not cover the same 80 molecules")
    report = {"rows": len(records), "parent_control": str(control), "pairs": {}}
    for left, right in PAIRS:
        baseline, fused = [], []
        changed = missing_groups = 0
        for rec in records:
            mid = rec["molecule_id"]
            names = parent.loc[mid, "smiles"].split(";")
            if mid not in truth or len(rec[left + "_scores"]) != len(names):
                raise ValueError("Missing label or score length mismatch")
            order, diag = rerank_fused(
                names,
                np.asarray(rec[left + "_scores"], dtype=float),
                np.asarray(rec[right + "_scores"], dtype=float),
                weight=0.25,
            )
            ranked = [names[i] for i in order]
            if ranked[0] != names[0] or sorted(ranked) != sorted(names):
                raise AssertionError("Fusion changed candidate set or top1")
            baseline.append(rec["A_rank"])
            fused.append(rank(ranked, truth[mid]))
            changed += ranked != names
            missing_groups += diag["missing_groups"]
        base_rr = np.asarray([1 / i if i else 0 for i in baseline])
        fused_rr = np.asarray([1 / i if i else 0 for i in fused])
        report["pairs"][f"{left}+{right}"] = {
            "mrr25": float(fused_rr.mean()),
            "delta_mrr25_vs_A": float((fused_rr - base_rr).mean()),
            "top1": float(np.mean(np.asarray(fused) == 1)),
            "improved_vs_A": int(np.sum(fused_rr > base_rr)),
            "worsened_vs_A": int(np.sum(fused_rr < base_rr)),
            "unchanged_vs_A": int(np.sum(fused_rr == base_rr)),
            "changed_parent_rows": changed,
            "missing_formula_groups": missing_groups,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.proxy, args.control, args.labels, args.output)


if __name__ == "__main__":
    main()

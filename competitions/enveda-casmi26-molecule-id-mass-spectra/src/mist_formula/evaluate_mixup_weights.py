"""Audit rank-fusion strength on the frozen labeled parent proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bcf_rerank import rerank
from evaluate_mixup_parent import rank
from rdkit import RDLogger


def metrics(ranks: list[int | None]) -> dict:
    return {
        "mrr25": sum(1 / value for value in ranks if value is not None) / len(ranks),
        "top1": sum(value == 1 for value in ranks) / len(ranks),
        "present": sum(value is not None for value in ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-jsonl", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.25, 0.4, 0.6, 0.75])
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    rows = {
        row["molecule_id"]: row
        for line in args.parent_jsonl.read_text().splitlines()
        if (row := json.loads(line))
    }
    control = pd.read_csv(args.control, dtype=str, keep_default_na=False)
    truth = pd.read_csv(args.labels, dtype=str).set_index("molecule_id").metric_key.to_dict()
    if set(rows) != set(control.molecule_id) or set(rows) != set(truth):
        raise ValueError("fixed proxy IDs differ")
    conditions = [
        key.removesuffix("_scores") for key in next(iter(rows.values())) if key.endswith("_scores")
    ]
    results = {}
    for model in conditions:
        for weight in args.weights:
            ranks = []
            for item in control.itertuples(index=False):
                names = item.smiles.split(";")
                row = rows[item.molecule_id]
                scores = np.asarray(row[model + "_scores"], float)
                order, _audit = rerank(names, scores, weight=weight)
                result = [names[index] for index in order]
                actual = rank(result, truth[item.molecule_id])
                if weight == 0.25 and actual != row[model + "_rank"]:
                    raise ValueError(f"saved {model} rank differs for {item.molecule_id}")
                ranks.append(actual)
            results[f"{model}@{weight:g}"] = metrics(ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()

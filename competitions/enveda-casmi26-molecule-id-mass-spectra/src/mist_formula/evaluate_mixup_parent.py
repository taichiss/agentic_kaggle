"""Compare Mixup checkpoints on the same labeled 80-row 0.477 parent proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bcf_rerank import rerank
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(mol))[:14]


def rank(smiles: list[str], truth: str) -> int | None:
    return next((i for i, smiles_i in enumerate(smiles[:25], 1) if key(smiles_i) == truth), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--prior-a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=0.25)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    sys.path.insert(0, str(args.model_source.parent))
    from model import _load_checkpoint, score_candidates  # noqa: PLC0415

    parsed = [item.split("=", 1) for item in args.model]
    if any(len(item) != 2 or not Path(item[1]).is_file() for item in parsed):
        parser.error("each --model must be NAME=existing checkpoint")
    models = {name: Path(path).resolve() for name, path in parsed}
    if len(models) != len(parsed) or "A" not in models:
        parser.error("model names must be unique and include A")
    if sha(args.control) != "2492e1ae70335c67a87c1d756964d52bfc3c299e86aeeb6f6a7d351b4909db47":
        raise ValueError("80-molecule parent control changed")
    if sha(args.test) != "40be4cbdbb532a72e2886dc1b12ba86edcca7a79808c2cdaf6c6d05e9e3d00a1":
        raise ValueError("80-molecule spectra changed")
    control = pd.read_csv(args.control, dtype=str, keep_default_na=False)
    test = pd.read_parquet(args.test)
    groups = {str(m): g.to_dict("records") for m, g in test.groupby("molecule_id", sort=False)}
    truth = pd.read_csv(args.labels, dtype=str).set_index("molecule_id").metric_key.to_dict()
    prior = {
        row["molecule_id"]: row
        for line in args.prior_a.read_text().splitlines()
        if (row := json.loads(line))
    }
    if set(control.molecule_id) != set(groups) or set(control.molecule_id) != set(truth):
        raise ValueError("parent, labels, spectra IDs differ")

    import torch  # noqa: PLC0415

    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for path in models.values():
        _load_checkpoint(str(path), device)
    records = []
    for n, row in enumerate(control.itertuples(index=False), 1):
        names = row.smiles.split(";")
        rec = {"molecule_id": row.molecule_id, "parent_rank": rank(names, truth[row.molecule_id])}
        for name, path in models.items():
            scores = np.asarray(
                score_candidates(groups[row.molecule_id], names, path, device), float
            )
            order, audit = rerank(names, scores, weight=args.weight)
            result = [names[index] for index in order]
            if sorted(result) != sorted(names) or result[0] != names[0]:
                raise ValueError(f"candidate set or top1 changed for {name}")
            rec[name + "_rank"] = rank(result, truth[row.molecule_id])
            rec[name + "_missing_groups"] = audit["missing_groups"]
            rec[name + "_scores"] = scores.tolist()
        if rec["A_rank"] != prior[row.molecule_id]["A_rank"]:
            raise ValueError(f"A rank differs from previous evaluation at {row.molecule_id}")
        records.append(rec)
        if n % 20 == 0:
            print(f"evaluated {n}/{len(control)}", flush=True)
    report = {
        "rows": len(records),
        "control_sha256": sha(args.control),
        "spectra_sha256": sha(args.test),
        "labels_sha256": sha(args.labels),
        "model_source_sha256": sha(args.model_source),
        "checkpoints_sha256": {name: sha(path) for name, path in models.items()},
        "weight": args.weight,
        "conditions": {},
    }
    for name in ("parent", *models):
        ranks = [row[name + "_rank"] for row in records]
        report["conditions"][name] = {
            "mrr25": sum(1 / value for value in ranks if value is not None) / len(ranks),
            "top1": sum(value == 1 for value in ranks) / len(ranks),
            "present": sum(value is not None for value in ranks),
        }
        if name != "parent":
            report["conditions"][name]["missing_groups"] = sum(
                row[name + "_missing_groups"] for row in records
            )
    baseline = np.asarray([0.0 if row["A_rank"] is None else 1 / row["A_rank"] for row in records])
    rng = np.random.default_rng(20261116)
    samples = rng.integers(0, len(records), size=(5000, len(records)))
    for name in models:
        values = np.asarray(
            [0.0 if row[name + "_rank"] is None else 1 / row[name + "_rank"] for row in records]
        )
        delta = values - baseline
        ci = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
        report["conditions"][name]["paired_vs_A"] = {
            "delta_mrr25": float(delta.mean()),
            "bootstrap_95pct": [float(x) for x in ci],
            "improved": int(np.count_nonzero(delta > 0)),
            "worsened": int(np.count_nonzero(delta < 0)),
            "unchanged": int(np.count_nonzero(delta == 0)),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

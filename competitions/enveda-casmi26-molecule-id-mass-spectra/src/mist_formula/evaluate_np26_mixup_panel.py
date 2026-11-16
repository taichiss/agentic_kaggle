"""Paired checkpoint evaluation on the fixed natural timsTOF panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import FormulaCatalog, FormulaTransformer
from train_mixup_weak import Cache, evaluate


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(rows: list[dict], name: str, seed: int) -> dict:
    if not rows:
        return {"molecules": 0}
    ranks = np.asarray([row[f"{name}_rank"] for row in rows], np.int64)
    report = {
        "molecules": len(rows),
        "mrr25": float(np.mean(1 / ranks)),
        "top1": float(np.mean(ranks == 1)),
    }
    if name != "A":
        reference = np.asarray([row["A_rank"] for row in rows], np.int64)
        difference = 1 / ranks - 1 / reference
        rng = np.random.default_rng(seed)
        sampled = rng.integers(len(rows), size=(10_000, len(rows)))
        report.update({
            "delta_mrr25_vs_A": float(difference.mean()),
            "paired_bootstrap_95pct": [
                float(value)
                for value in np.quantile(difference[sampled].mean(axis=1), [.025, .975])
            ],
            "improved": int(np.sum(ranks < reference)),
            "worsened": int(np.sum(ranks > reference)),
            "unchanged": int(np.sum(ranks == reference)),
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=FILE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap-seed", type=int, default=20261117)
    args = parser.parse_args()
    parsed = [item.split("=", 1) for item in args.checkpoint]
    names = [item[0] for item in parsed]
    if len(parsed) < 2 or len(set(names)) != len(names) or "A" not in names:
        parser.error("provide A and at least one unique comparison checkpoint")
    checkpoints = {name: Path(path) for name, path in parsed}
    if not all(path.is_file() for path in checkpoints.values()):
        parser.error("checkpoint file missing")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(4)
    panel = pd.read_parquet(args.panel)
    data = Cache(args.cache, args.precise_mzs)
    if len(panel) != 26 or not panel.mol_idx.isin(data.val_mols).all():
        raise ValueError("Expected 26 held-out validation molecules")
    if panel.mol_idx.duplicated().any():
        raise ValueError("Panel repeats a validation molecule")
    counts = panel.candidate_count.to_numpy(dtype=int)
    if np.any(counts < 1) or np.any(counts > 25):
        raise ValueError("Candidate counts must lie in 1..25")
    if any(
        len(item.candidate_mol_indices) != item.candidate_count
        or len(item.query_cache_rows) != item.query_spectra_count
        for item in panel.itertuples(index=False)
    ):
        raise ValueError("Panel candidate/query count mismatch")

    results = {}
    catalog_counts = None
    model_config = None
    for name in checkpoints:
        checkpoint = torch.load(checkpoints[name], map_location="cpu", weights_only=False)
        if checkpoint.get("step") != 12_000:
            raise ValueError(f"{name} checkpoint does not have 12,000 updates")
        current_counts = np.asarray(checkpoint["formula_counts"])
        current_config = checkpoint["model_config"]
        if catalog_counts is None:
            catalog_counts, model_config = current_counts, current_config
        elif not np.array_equal(catalog_counts, current_counts) or model_config != current_config:
            raise ValueError("Model catalog or architecture differs between conditions")
        model = FormulaTransformer(**current_config)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(args.device).eval()
        catalog = FormulaCatalog(
            current_counts,
            max_peaks=checkpoint["max_peaks"],
            ppm=checkpoint["ppm"],
            k=checkpoint["candidate_formulas"],
        )
        result = evaluate(model, data, catalog, panel, args.device, len(panel))
        if result["query_mols"] != panel.mol_idx.tolist():
            raise AssertionError("Evaluation reordered query molecules")
        results[name] = result
        print(
            json.dumps({"model": name, "queries": result["queries"],
                        "mrr25_all": result["mrr25"], "top1_all": result["top1"]}),
            flush=True,
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    records = []
    for i, item in enumerate(panel.itertuples(index=False)):
        record = {
            "mol_idx": int(item.mol_idx),
            "molecule_key": str(item.molecule_key),
            "formula": str(item.formula),
            "candidate_count": int(item.candidate_count),
            "query_spectra_count": int(item.query_spectra_count),
            "target_index": int(item.target_index),
            "query_cache_rows": item.query_cache_rows.tolist(),
            "candidate_mol_indices": item.candidate_mol_indices.tolist(),
        }
        for name in results:
            rank = int(results[name]["ranks"][i])
            if rank > item.candidate_count:
                raise AssertionError("Rank exceeds candidate count")
            record[f"{name}_rank"] = rank
        records.append(record)
    multi = [row for row in records if row["candidate_count"] > 1]
    singleton = [row for row in records if row["candidate_count"] == 1]
    bands = {
        "2-5": [row for row in multi if 2 <= row["candidate_count"] <= 5],
        "6-24": [row for row in multi if 6 <= row["candidate_count"] <= 24],
        "25": [row for row in multi if row["candidate_count"] == 25],
    }
    report = {
        "panel_sha256": sha(args.panel),
        "checkpoints_sha256": {name: sha(path) for name, path in checkpoints.items()},
        "formula_compositions": len(catalog_counts),
        "models": tuple(results),
        "query_molecules": len(records),
        "multi_candidate_molecules": len(multi),
        "single_candidate_molecules": len(singleton),
        "candidate_counts": [row["candidate_count"] for row in records],
        "all_molecules": {
            name: metrics(records, name, args.bootstrap_seed) for name in results
        },
        "multi_candidate_only": {
            name: metrics(multi, name, args.bootstrap_seed) for name in results
        },
        "candidate_count_bands": {
            band: {name: metrics(rows, name, args.bootstrap_seed) for name in results}
            for band, rows in bands.items()
        },
        "single_candidate_note": (
            "Singleton queries have forced rank 1 for all models and are excluded from "
            "the main paired difference."
        ),
        "interpretation_limit": (
            "Post hoc supplementary panel built after the 1,000-query results were examined; "
            "24 informative molecules, with varying 1-25 candidate counts. Synthetic cache "
            "candidate sets are not hidden competition candidates. Do not tune on this panel "
            "and claim independent validation, or numerically compare its MRR to other panels."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Evaluate score-level Mixup checkpoint fusions on the fixed 1000-query panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import FormulaCatalog, FormulaTransformer
from train_mix_soft import Cache, tanimoto

PAIRS = (("A", "M1"), ("A", "M2"), ("M1", "M2"), ("A", "M3"), ("M1", "M3"))


def zscore(values: np.ndarray) -> np.ndarray:
    sd = values.std()
    return (values - values.mean()) / sd if sd > 1e-9 else np.zeros_like(values)


def rank(scores: np.ndarray, target: int) -> int:
    return int(np.count_nonzero(scores >= scores[target]))


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(4)
    data = Cache(args.cache, args.precise_mzs)
    panel = pd.read_parquet(args.panel)
    baseline = [json.loads(line) for line in args.a_scores.read_text().splitlines()]
    if len(panel) < 1000 or len(baseline) != 1000:
        raise ValueError("Fixed panel or A baseline is incomplete")
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in (
            ("A", args.a_checkpoint),
            ("M1", args.m1_checkpoint),
            ("M2", args.m2_checkpoint),
            ("M3", args.m3_checkpoint),
        )
    }
    reference = checkpoints["A"]
    for name, checkpoint in checkpoints.items():
        if checkpoint["model_config"] != reference["model_config"] or not np.array_equal(
            checkpoint["formula_counts"], reference["formula_counts"]
        ):
            raise ValueError(f"{name} differs from frozen A architecture/catalog")
    catalog = FormulaCatalog(
        reference["formula_counts"],
        max_peaks=reference["max_peaks"],
        ppm=reference["ppm"],
        k=reference["candidate_formulas"],
    )
    models = {}
    for name, checkpoint in checkpoints.items():
        model = FormulaTransformer(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        models[name] = model.to(args.device).eval()
    results = []
    for index, item in enumerate(panel.itertuples(index=False)):
        if index == 1000:
            break
        base = baseline[index]
        candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
        target = int(item.target_index)
        if (
            int(item.mol_idx) != int(base["mol_idx"])
            or candidates.tolist() != base["candidate_mol_indices"]
            or candidates[target] != item.mol_idx
        ):
            raise ValueError("A and treatment candidate panel differ")
        rows = np.asarray(item.query_cache_rows, dtype=np.int64)
        batch = data.batch(rows, catalog, args.device)
        fps = torch.as_tensor(np.asarray(data.fp[candidates], dtype=np.float32), device=args.device)
        scores = {}
        for name, model in models.items():
            probabilities = model(*batch).mean(0).sigmoid()
            scores[name] = tanimoto(probabilities[None], fps).cpu().numpy()
        if not np.allclose(scores["A"], base["A_scores"], atol=1e-6, rtol=1e-5):
            raise AssertionError("Frozen A panel candidate scores changed")
        rec = {"mol_idx": int(item.mol_idx), "A_rank": rank(scores["A"], target)}
        if rec["A_rank"] != base["A_rank"]:
            raise AssertionError("Frozen A panel rank changed")
        for name in ("M1", "M2", "M3"):
            rec[name + "_rank"] = rank(scores[name], target)
        rec["target_index"] = target
        for name in ("A", "M1", "M2", "M3"):
            rec[name + "_scores"] = scores[name].astype(float).tolist()
        for left, right in PAIRS:
            fused = 0.5 * zscore(scores[left]) + 0.5 * zscore(scores[right])
            rec[left + "+" + right + "_rank"] = rank(fused, target)
        results.append(rec)
        if (index + 1) % 250 == 0:
            print(f"panel fused {index + 1}/1000", flush=True)
    names = ["A", "M1", "M2", "M3", *(f"{left}+{right}" for left, right in PAIRS)]
    baseline_rr = np.asarray([1 / row["A_rank"] for row in results])
    rng = np.random.default_rng(20261116)
    samples = rng.integers(0, len(results), size=(5000, len(results)))
    report = {"queries": len(results), "conditions": {}}
    for name in names:
        ranks = np.asarray([row[name + "_rank"] for row in results])
        rr = 1 / ranks
        delta = rr - baseline_rr
        ci = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
        report["conditions"][name] = {
            "mrr25": float(rr.mean()),
            "top1": float(np.mean(ranks == 1)),
            "delta_mrr25_vs_A": float(delta.mean()),
            "bootstrap_95pct": [float(ci[0]), float(ci[1])],
            "improved": int(np.count_nonzero(delta > 0)),
            "worsened": int(np.count_nonzero(delta < 0)),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    print(json.dumps(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cache",
        "precise-mzs",
        "panel",
        "a-scores",
        "a-checkpoint",
        "m1-checkpoint",
        "m2-checkpoint",
        "m3-checkpoint",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

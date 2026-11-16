"""Audit a small no-training M4/M5 grid on fixed 1000- and 80-molecule panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from bcf_rerank import formula, rerank
from evaluate_mixup_parent import rank as parent_rank
from rdkit import RDLogger

BLENDS = (("A", "M1", 0.75), ("A", "M1", 0.90), ("A", "M2", 0.75), ("A", "M2", 0.90))
WEIGHTS = (("A", 0.15), ("A", 0.35), ("M1", 0.15), ("M1", 0.35), ("M2", 0.15), ("M2", 0.35))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def z(values: np.ndarray) -> np.ndarray:
    sigma = float(values.std())
    return (values - values.mean()) / sigma if sigma > 1e-9 else np.zeros_like(values)


def blend(scores_a: np.ndarray, scores_b: np.ndarray, alpha_a: float) -> np.ndarray:
    if not np.isfinite(scores_a).all() or not np.isfinite(scores_b).all():
        raise ValueError("Blend requires finite model scores")
    return alpha_a * z(scores_a) + (1 - alpha_a) * z(scores_b)


def parent_blend(names: list[str], a: np.ndarray, b: np.ndarray, alpha_a: float) -> np.ndarray:
    """Standardize each score array within a formula group, including rank 1."""
    fused = np.full(len(names), -np.inf)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, smiles in enumerate(names):
        key = formula(smiles)
        if key is not None:
            groups[key].append(index)
    for indices in groups.values():
        if sum(index > 0 for index in indices) < 2:
            continue
        values_a, values_b = a[indices], b[indices]
        if not np.isfinite(values_a).all() or not np.isfinite(values_b).all():
            continue
        values = blend(values_a, values_b, alpha_a)
        for index, value in zip(indices, values, strict=True):
            if index > 0:
                fused[index] = value
    return fused


def summary(ranks: list[int | None], baseline: list[int | None], seed: int) -> dict:
    rr = np.asarray([1 / value if value else 0 for value in ranks])
    base = np.asarray([1 / value if value else 0 for value in baseline])
    delta = rr - base
    rng = np.random.default_rng(seed)
    resampled = rng.integers(0, len(rr), size=(5000, len(rr)))
    ci = np.quantile(delta[resampled].mean(axis=1), [0.025, 0.975])
    return {
        "mrr25": float(rr.mean()),
        "top1": float(np.mean(np.asarray(ranks) == 1)),
        "delta_mrr25_vs_A": float(delta.mean()),
        "molecule_bootstrap_95pct_delta": [float(ci[0]), float(ci[1])],
        "improved": int(np.count_nonzero(delta > 0)),
        "worsened": int(np.count_nonzero(delta < 0)),
        "unchanged": int(np.count_nonzero(delta == 0)),
    }


def run(args: argparse.Namespace) -> dict:
    RDLogger.DisableLog("rdApp.*")
    panel = [json.loads(line) for line in args.panel_scores.read_text().splitlines()]
    proxy = [json.loads(line) for line in args.proxy_scores.read_text().splitlines()]
    control = pd.read_csv(args.control, dtype=str, keep_default_na=False).set_index("molecule_id")
    truth = pd.read_csv(args.labels, dtype=str).set_index("molecule_id").metric_key.to_dict()
    if len(panel) != 1000 or len(proxy) != 80 or len(control) != 80:
        raise ValueError("Fixed panel or proxy size changed")
    baseline_panel = [int(row["A_rank"]) for row in panel]
    baseline_proxy = [row["A_rank"] for row in proxy]
    panel_out, proxy_out = {}, {}
    for left, right, alpha in BLENDS:
        name = f"{left}{int(alpha * 100)}+{right}{int((1 - alpha) * 100)}"
        panel_ranks, proxy_ranks = [], []
        for row in panel:
            values = blend(
                np.asarray(row[left + "_scores"], dtype=float),
                np.asarray(row[right + "_scores"], dtype=float),
                alpha,
            )
            panel_ranks.append(int(np.count_nonzero(values >= values[row["target_index"]])))
        for row in proxy:
            mid = row["molecule_id"]
            names = control.loc[mid, "smiles"].split(";")
            values = parent_blend(
                names,
                np.asarray(row[left + "_scores"], dtype=float),
                np.asarray(row[right + "_scores"], dtype=float),
                alpha,
            )
            order, _ = rerank(names, values, weight=0.25)
            ranked = [names[index] for index in order]
            if ranked[0] != names[0] or sorted(ranked) != sorted(names):
                raise AssertionError("Blend changed protected top1 or candidate set")
            proxy_ranks.append(parent_rank(ranked, truth[mid]))
        panel_out[name] = summary(panel_ranks, baseline_panel, 20261116)
        proxy_out[name] = summary(proxy_ranks, baseline_proxy, 20261116)
    for model, weight in WEIGHTS:
        name = f"{model}@{weight:.2f}"
        ranks = []
        for row in proxy:
            mid = row["molecule_id"]
            names = control.loc[mid, "smiles"].split(";")
            scores = np.asarray(row[model + "_scores"], dtype=float)
            order, _ = rerank(names, scores, weight=weight)
            ranked = [names[index] for index in order]
            if ranked[0] != names[0] or sorted(ranked) != sorted(names):
                raise AssertionError("Rank weight changed protected top1 or candidate set")
            ranks.append(parent_rank(ranked, truth[mid]))
        proxy_out[name] = summary(ranks, baseline_proxy, 20261116)
    report = {
        "scope": "exploratory small grid, selected 80 proxy and fixed 1000 synthetic panel; no LB",
        "panel_queries": 1000,
        "proxy_queries": 80,
        "proxy_truth_present": 54,
        "baseline_A_mrr25_panel": summary(baseline_panel, baseline_panel, 20261116)["mrr25"],
        "baseline_A_mrr25_proxy": summary(baseline_proxy, baseline_proxy, 20261116)["mrr25"],
        "score_blends": {
            name: {"panel": panel_out[name], "proxy": proxy_out[name]} for name in panel_out
        },
        "parent_rank_weights": {name: proxy_out[name] for name in proxy_out if "@" in name},
        "evidence_sha256": {
            "panel_scores": sha(args.panel_scores),
            "proxy_scores": sha(args.proxy_scores),
            "control": sha(args.control),
            "labels": sha(args.labels),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel-scores", "proxy-scores", "control", "labels", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

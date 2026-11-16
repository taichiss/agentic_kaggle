"""Explore A75:weak-M1 25 beside the saved A75:strict-M1 25 blend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bcf_rerank import rerank
from evaluate_mixup_parent import rank as parent_rank
from explore_mixup_adaptive import blend, parent_blend
from rdkit import RDLogger

ALPHA_A = 0.75
PARENT_WEIGHT = 0.25
CONDITIONS = ("A", "M1", "WEAK", "A75+M125", "A75+WEAK25")
BASELINES = ("A", "WEAK", "A75+M125")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def score_rank(scores: np.ndarray, target: int) -> int:
    return int(np.count_nonzero(scores >= scores[target]))


def score_triplet(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(record["A_scores"], dtype=float)
    m1 = np.asarray(record["M1_scores"], dtype=float)
    weak = np.asarray(record["WEAK_scores"], dtype=float)
    if a.shape != m1.shape or a.shape != weak.shape or not np.isfinite(a).all():
        raise ValueError("Inconsistent A/M1/WEAK candidate-score arrays")
    if not np.isfinite(m1).all() or not np.isfinite(weak).all():
        raise ValueError("Nonfinite candidate scores")
    return a, m1, weak


def summarize(rows: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(rows), size=(5000, len(rows)))
    rr = {
        name: np.asarray([1 / row[name + "_rank"] if row[name + "_rank"] else 0 for row in rows])
        for name in CONDITIONS
    }
    report = {"molecules": len(rows), "conditions": {}}
    for name in CONDITIONS:
        ranks = [row[name + "_rank"] for row in rows]
        metrics = {
            "mrr25": float(rr[name].mean()),
            "top1": float(np.mean(np.asarray(ranks, dtype=object) == 1)),
            "present": sum(value is not None for value in ranks),
        }
        for baseline in BASELINES:
            if baseline == name:
                continue
            delta = rr[name] - rr[baseline]
            ci = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
            metrics["paired_vs_" + baseline] = {
                "delta_mrr25": float(delta.mean()),
                "bootstrap_95pct": [float(value) for value in ci],
                "improved": int(np.count_nonzero(delta > 0)),
                "worsened": int(np.count_nonzero(delta < 0)),
                "unchanged": int(np.count_nonzero(delta == 0)),
            }
        report["conditions"][name] = metrics
    return report


def panel1000(base: list[dict], weak_scores: list[dict], definition: pd.DataFrame) -> list[dict]:
    if len(base) != 1000 or len(weak_scores) != 1000 or len(definition) != 1000:
        raise ValueError("Fixed panel size changed")
    result = []
    for saved, weak, item in zip(
        base, weak_scores, definition.itertuples(index=False), strict=True
    ):
        if (
            saved["mol_idx"] != weak["mol_idx"]
            or saved["mol_idx"] != item.mol_idx
            or weak["candidate_mol_indices"] != item.candidate_mol_indices.tolist()
            or saved["target_index"] != weak["target_index"]
            or saved["target_index"] != item.target_index
        ):
            raise ValueError("Weak scores use a different 1,000-query candidate panel")
        target = int(saved["target_index"])
        combined = {
            "A_scores": saved["A_scores"],
            "M1_scores": saved["M1_scores"],
            "WEAK_scores": weak["WEAK_scores"],
        }
        a, m1, weak_values = score_triplet(combined)
        if a.size != 25:
            raise ValueError("Fixed 1,000 panel does not hold 25 candidates")
        for name, values, expected in (
            ("A", a, saved["A_rank"]),
            ("M1", m1, saved["M1_rank"]),
            ("WEAK", weak_values, weak["WEAK_rank"]),
        ):
            if score_rank(values, target) != expected:
                raise ValueError(f"Saved {name} panel rank differs at {saved['mol_idx']}")
        result.append(
            {
                "mol_idx": int(saved["mol_idx"]),
                "A_rank": int(saved["A_rank"]),
                "M1_rank": int(saved["M1_rank"]),
                "WEAK_rank": int(weak["WEAK_rank"]),
                "A75+M125_rank": score_rank(blend(a, m1, ALPHA_A), target),
                "A75+WEAK25_rank": score_rank(blend(a, weak_values, ALPHA_A), target),
            }
        )
    return result


def parent80(rows: list[dict], control_path: Path, label_path: Path) -> list[dict]:
    control = pd.read_csv(control_path, dtype=str, keep_default_na=False).set_index("molecule_id")
    truth = pd.read_csv(label_path, dtype=str).set_index("molecule_id").metric_key.to_dict()
    if (
        len(rows) != 80
        or len(control) != 80
        or set(control.index) != set(truth)
        or set(control.index) != {row["molecule_id"] for row in rows}
    ):
        raise ValueError("Saved 80-row parent proxy changed")
    result = []
    for row in rows:
        mid = row["molecule_id"]
        names = control.loc[mid, "smiles"].split(";")
        a, m1, weak = score_triplet(row)
        if len(a) != len(names):
            raise ValueError("Parent score and candidate lengths disagree")
        for name, scores in (("A", a), ("M1", m1), ("WEAK", weak)):
            order, _ = rerank(names, scores, weight=PARENT_WEIGHT)
            found = parent_rank([names[index] for index in order], truth[mid])
            if found != row[name + "_rank"]:
                raise ValueError(f"Saved {name} proxy rank differs at {mid}")
        ranks = {}
        for name, scores in (
            ("A75+M125", parent_blend(names, a, m1, ALPHA_A)),
            ("A75+WEAK25", parent_blend(names, a, weak, ALPHA_A)),
        ):
            order, _ = rerank(names, scores, weight=PARENT_WEIGHT)
            ranked = [names[index] for index in order]
            if ranked[0] != names[0] or sorted(ranked) != sorted(names):
                raise AssertionError("Fusion changed candidate set or protected top1")
            ranks[name + "_rank"] = parent_rank(ranked, truth[mid])
        result.append(
            {
                "molecule_id": mid,
                "A_rank": row["A_rank"],
                "M1_rank": row["M1_rank"],
                "WEAK_rank": row["WEAK_rank"],
                **ranks,
            }
        )
    return result


def np26(scores: list[dict], prior: list[dict]) -> list[dict]:
    if len(scores) != 26 or len(prior) != 26:
        raise ValueError("Expected 26 fixed NP queries")
    result = []
    for row, saved in zip(scores, prior, strict=True):
        if (
            row["mol_idx"] != saved["mol_idx"]
            or row["candidate_mol_indices"] != saved["candidate_mol_indices"]
            or row["target_index"] != saved["target_index"]
        ):
            raise ValueError("NP26 candidate panel differs")
        target = int(saved["target_index"])
        a, m1, weak = score_triplet(row)
        if len(a) != saved["candidate_count"]:
            raise ValueError("NP26 candidate count differs")
        for name, values in (("A", a), ("M1", m1), ("WEAK", weak)):
            if score_rank(values, target) != saved[name + "_rank"]:
                raise ValueError(f"NP26 {name} rank differs at {saved['mol_idx']}")
        result.append(
            {
                "mol_idx": int(saved["mol_idx"]),
                "candidate_count": len(a),
                "A_rank": int(saved["A_rank"]),
                "M1_rank": int(saved["M1_rank"]),
                "WEAK_rank": int(saved["WEAK_rank"]),
                "A75+M125_rank": score_rank(blend(a, m1, ALPHA_A), target),
                "A75+WEAK25_rank": score_rank(blend(a, weak, ALPHA_A), target),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict:
    RDLogger.DisableLog("rdApp.*")
    p1000 = panel1000(
        read_jsonl(args.panel_base),
        read_jsonl(args.panel_weak),
        pd.read_parquet(args.panel_definition),
    )
    p80 = parent80(read_jsonl(args.proxy_weak), args.control, args.labels)
    p26 = np26(read_jsonl(args.np26_scores), read_jsonl(args.np26_prior))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("same_formula_1000", p1000),
        ("saved_parent_80", p80),
        ("np26", p26),
    ):
        (args.output / (name + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    report = {
        "condition": "A75:weak-M1 25 versus A75:strict-M1 25; formula z scores; RRF 0.25 once",
        "selection": "Post hoc local blend exploration after A/M1/M2/M3 and weak-M1 results",
        "same_formula_1000": summarize(p1000, args.seed),
        "saved_parent_80": summarize(p80, args.seed),
        "np26_all": summarize(p26, args.seed),
        "np26_multi_candidate": summarize(
            [row for row in p26 if row["candidate_count"] > 1], args.seed
        ),
        "evidence_sha256": {
            name: sha(getattr(args, name))
            for name in (
                "panel_base",
                "panel_definition",
                "panel_weak",
                "proxy_weak",
                "control",
                "labels",
                "np26_scores",
                "np26_prior",
                "a_checkpoint",
                "m1_checkpoint",
                "weak_checkpoint",
            )
        },
        "limitations": (
            "The 1,000-query panel uses synthetic same-formula candidate sets, the saved 80-row "
            "parent proxy has only 54 truth-present queries, and NP26 was assembled post hoc with "
            "24 informative multi-candidate queries. Absolute MRR across panels is not comparable. "
            "This blend was chosen after inspecting several local variants; confidence intervals "
            "do not account for that selection and cannot establish leaderboard improvement."
        ),
    }
    if not np.isclose(
        report["same_formula_1000"]["conditions"]["A75+M125"]["mrr25"],
        0.39225763256317875,
        atol=1e-12,
    ) or not np.isclose(
        report["saved_parent_80"]["conditions"]["A75+M125"]["mrr25"],
        0.3554964826839827,
        atol=1e-12,
    ):
        raise ValueError("Strict M1 blend does not reproduce the saved local exploration")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {name: report[name] for name in ("same_formula_1000", "saved_parent_80", "np26_all")}
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "panel-base",
        "panel-definition",
        "panel-weak",
        "proxy-weak",
        "control",
        "labels",
        "np26-scores",
        "np26-prior",
        "a-checkpoint",
        "m1-checkpoint",
        "weak-checkpoint",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

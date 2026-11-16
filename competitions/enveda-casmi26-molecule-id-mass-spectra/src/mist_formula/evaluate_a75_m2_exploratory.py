"""Explore the fixed A75:M2 25 candidate-score blend on three saved panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bcf_rerank import rerank
from evaluate_mixup_parent import rank as parent_rank
from explore_mixup_adaptive import blend, parent_blend
from model import FormulaCatalog, FormulaTransformer
from rdkit import RDLogger
from train_mix_soft import Cache, tanimoto

ALPHA_A = 0.75
PARENT_WEIGHT = 0.25
CONDITIONS = ("A", "M2", "A75+M225")


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


def summarize(rows: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(rows), size=(5000, len(rows)))
    reciprocal = {
        name: np.asarray([1 / row[name + "_rank"] if row[name + "_rank"] else 0 for row in rows])
        for name in CONDITIONS
    }
    result = {"molecules": len(rows), "conditions": {}}
    for name in CONDITIONS:
        ranks = [row[name + "_rank"] for row in rows]
        current = reciprocal[name]
        entry = {
            "mrr25": float(current.mean()),
            "top1": float(np.mean(np.asarray(ranks, dtype=object) == 1)),
            "present": sum(value is not None for value in ranks),
        }
        for baseline in ("A", "M2"):
            if name == baseline:
                continue
            delta = current - reciprocal[baseline]
            ci = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
            entry["paired_vs_" + baseline] = {
                "delta_mrr25": float(delta.mean()),
                "bootstrap_95pct": [float(value) for value in ci],
                "improved": int(np.count_nonzero(delta > 0)),
                "worsened": int(np.count_nonzero(delta < 0)),
                "unchanged": int(np.count_nonzero(delta == 0)),
            }
        result["conditions"][name] = entry
    return result


def panel1000(rows: list[dict]) -> list[dict]:
    if len(rows) != 1000 or len({row["mol_idx"] for row in rows}) != 1000:
        raise ValueError("Expected exactly 1,000 fixed distinct panel molecules")
    result = []
    for row in rows:
        a = np.asarray(row["A_scores"], dtype=float)
        m2 = np.asarray(row["M2_scores"], dtype=float)
        target = int(row["target_index"])
        if a.shape != m2.shape or a.size != 25 or not 0 <= target < a.size:
            raise ValueError("Fixed panel candidate scores changed")
        if score_rank(a, target) != row["A_rank"] or score_rank(m2, target) != row["M2_rank"]:
            raise ValueError("Saved panel rank and scores disagree")
        fused = blend(a, m2, ALPHA_A)
        result.append(
            {
                "mol_idx": int(row["mol_idx"]),
                "A_rank": int(row["A_rank"]),
                "M2_rank": int(row["M2_rank"]),
                "A75+M225_rank": score_rank(fused, target),
            }
        )
    return result


def parent80(rows: list[dict], control_path: Path, label_path: Path) -> list[dict]:
    control = pd.read_csv(control_path, dtype=str, keep_default_na=False).set_index("molecule_id")
    truth = pd.read_csv(label_path, dtype=str).set_index("molecule_id").metric_key.to_dict()
    if (
        len(rows) != 80
        or len(control) != 80
        or set(control.index) != {row["molecule_id"] for row in rows}
        or set(control.index) != set(truth)
    ):
        raise ValueError("Fixed 80-row parent proxy changed")
    result = []
    for row in rows:
        mid = row["molecule_id"]
        names = control.loc[mid, "smiles"].split(";")
        a = np.asarray(row["A_scores"], dtype=float)
        m2 = np.asarray(row["M2_scores"], dtype=float)
        if len(a) != len(m2) or len(a) != len(names):
            raise ValueError("Parent candidate score length changed")
        for name, scores in (("A", a), ("M2", m2)):
            prior_order, _ = rerank(names, scores, weight=PARENT_WEIGHT)
            prior_rank = parent_rank([names[index] for index in prior_order], truth[mid])
            if prior_rank != row[name + "_rank"]:
                raise ValueError(f"Saved {name} proxy rank and candidate scores disagree at {mid}")
        fused = parent_blend(names, a, m2, ALPHA_A)
        order, diag = rerank(names, fused, weight=PARENT_WEIGHT)
        ranked = [names[index] for index in order]
        if ranked[0] != names[0] or sorted(ranked) != sorted(names):
            raise AssertionError("Blend changed parent top1 or candidate set")
        target_rank = parent_rank(ranked, truth[mid])
        result.append(
            {
                "molecule_id": mid,
                "A_rank": row["A_rank"],
                "M2_rank": row["M2_rank"],
                "A75+M225_rank": target_rank,
                "changed_parent_row": ranked != names,
                "missing_formula_groups": diag["missing_groups"],
            }
        )
    return result


@torch.inference_mode()
def np26(args: argparse.Namespace) -> list[dict]:
    data = Cache(args.cache, args.precise_mzs)
    panel = pd.read_parquet(args.np26_panel)
    prior = read_jsonl(args.np26_prior)
    if len(panel) != 26 or len(prior) != 26 or not panel.mol_idx.isin(data.val_mols).all():
        raise ValueError("NP26 held-out panel changed")
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in (("A", args.a_checkpoint), ("M2", args.m2_checkpoint))
    }
    reference = checkpoints["A"]
    if reference["model_config"] != checkpoints["M2"]["model_config"] or not np.array_equal(
        reference["formula_counts"], checkpoints["M2"]["formula_counts"]
    ):
        raise ValueError("A and M2 architecture or formula catalog differ")
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
    result = []
    for item, saved in zip(panel.itertuples(index=False), prior, strict=True):
        candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
        target = int(item.target_index)
        if (
            int(item.mol_idx) != saved["mol_idx"]
            or candidates.tolist() != saved["candidate_mol_indices"]
            or candidates[target] != item.mol_idx
            or len({data.mol.formula.iloc[int(index)] for index in candidates}) != 1
        ):
            raise ValueError("NP26 candidate set or formula changed")
        batch = data.batch(np.asarray(item.query_cache_rows, dtype=np.int64), catalog, args.device)
        fps = torch.as_tensor(np.asarray(data.fp[candidates], dtype=np.float32), device=args.device)
        scores = {}
        for name, model in models.items():
            probs = model(*batch).mean(0).sigmoid()
            scores[name] = tanimoto(probs[None], fps).cpu().numpy()
            if score_rank(scores[name], target) != saved[name + "_rank"]:
                raise ValueError(f"NP26 {name} rank differs at {item.mol_idx}")
        fused = blend(scores["A"].astype(float), scores["M2"].astype(float), ALPHA_A)
        result.append(
            {
                "mol_idx": int(item.mol_idx),
                "candidate_count": len(candidates),
                "A_rank": int(saved["A_rank"]),
                "M2_rank": int(saved["M2_rank"]),
                "A75+M225_rank": score_rank(fused, target),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict:
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(4)
    p1000 = panel1000(read_jsonl(args.panel_scores))
    p80 = parent80(read_jsonl(args.proxy_scores), args.control, args.labels)
    p26 = np26(args)
    args.output.mkdir(parents=True, exist_ok=True)
    records = {"same_formula_1000": p1000, "saved_parent_80": p80, "np26": p26}
    for name, rows in records.items():
        (args.output / (name + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    report = {
        "condition": "A75:M2 25, within-formula candidate-score z fusion, parent RRF once at 0.25",
        "selection": "Post hoc local exploration after A/M1/M2/M3 results; not unbiased validation",
        "same_formula_1000": summarize(p1000, args.seed),
        "saved_parent_80": summarize(p80, args.seed),
        "np26_all": summarize(p26, args.seed),
        "np26_multi_candidate": summarize(
            [row for row in p26 if row["candidate_count"] > 1], args.seed
        ),
        "parent_80_changed_rows": sum(row["changed_parent_row"] for row in p80),
        "parent_80_missing_formula_groups": sum(row["missing_formula_groups"] for row in p80),
        "evidence_sha256": {
            name: sha(getattr(args, name))
            for name in (
                "panel_scores",
                "proxy_scores",
                "control",
                "labels",
                "np26_panel",
                "np26_prior",
                "np26_overlap_audit",
                "a_checkpoint",
                "m2_checkpoint",
            )
        },
        "limitations": (
            "The 1,000 panel is a synthetic same-formula task. The saved 80 parent proxy "
            "contains only 54 truth-present queries. NP26 was assembled after prior local results "
            "with only 24 informative multi-candidate queries. Their absolute MRR values are not "
            "directly comparable. A75:M2 25 was selected after local inspection and must not be "
            "interpreted as an independent estimate or used to claim LB improvement."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {name: report[name] for name in ("same_formula_1000", "saved_parent_80", "np26_all")}
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "panel-scores",
        "proxy-scores",
        "control",
        "labels",
        "np26-panel",
        "np26-prior",
        "np26-overlap-audit",
        "cache",
        "precise-mzs",
        "a-checkpoint",
        "m2-checkpoint",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261117)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

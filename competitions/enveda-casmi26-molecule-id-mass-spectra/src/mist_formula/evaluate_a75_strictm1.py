"""Compare A75:strict-M1 25 with the frozen A, strict M1 and old M4.

The fixed-1000 and natural panels have no 0.477 parent ranking. The saved
80-query proxy applies the unchanged 0.477 parent RRF exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bcf_rerank import rerank
from evaluate_mixup_parent import rank as parent_rank
from explore_mixup_adaptive import blend, parent_blend
from rdkit import RDLogger
from train_mix_same import Cache, tanimoto


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data/derived"
MIXUP = DERIVED / "mist-mixup-20261116"
INPUTS = {
    "fixed_panel": DERIVED / "model-comparison-20261107/common-panel.parquet",
    "fixed_scores": MIXUP / "panel-fusions-1000/per-molecule.jsonl",
    "strict_manifest": MIXUP / "M1-ce-strict-sharedmerge/manifest.json",
    "strict_checkpoint": MIXUP / "M1-ce-strict-sharedmerge/final.pt",
    "natural_panel": MIXUP / "natural-source-full/panel.parquet",
    "natural_old_m4": MIXUP / "natural-source-full/m4-evaluation/per-molecule.jsonl",
    "natural_strict": MIXUP / "M1-ce-strict-sharedmerge/natural-source-eval/per-molecule.jsonl",
    "proxy_scores": MIXUP / "M1-ce-strict-sharedmerge/parent-proxy/per-molecule.jsonl",
    "control": DERIVED / "exp0015-cv/submission-control.csv",
    "labels": DERIVED / "candidate-rescue-corrected-v2/labels-local-only.csv",
    "old_m4_grid": MIXUP / "adaptive-grid-local.json",
    "cache_molecules": DERIVED / "saner-multipositive/cache-parallel/molecules.parquet",
    "cache_spectra": DERIVED / "saner-multipositive/cache-parallel/spectra.parquet",
    "precise_mzs": DERIVED / "mist-formula-20261107/precise-mzs.bin",
    "source_metadata": MIXUP / "source-metadata/manifest.json",
    "frozen_model": ROOT / "src/mist_formula/frozen/a_0479_model.py",
}


def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def rank_from_scores(scores: np.ndarray, target: int) -> int:
    return int(np.count_nonzero(scores >= scores[target]))


def fixed_strict_scores(panel: pd.DataFrame, cached: list[dict]) -> list[np.ndarray]:
    """Reproduce trainer validation ranks while retaining per-candidate scores."""
    payload = torch.load(INPUTS["strict_checkpoint"], map_location="cpu", weights_only=False)
    spec = importlib.util.spec_from_file_location("frozen_mist_model", INPUTS["frozen_model"])
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    catalog = frozen.FormulaCatalog(
        payload["formula_counts"], max_peaks=payload["max_peaks"],
        ppm=payload["ppm"], k=payload["candidate_formulas"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = frozen.FormulaTransformer(**payload["model_config"]).to(device).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    data = Cache(
        INPUTS["cache_molecules"].parent,
        INPUTS["precise_mzs"], INPUTS["source_metadata"].parent,
    )
    manifest = json.loads(INPUTS["strict_manifest"].read_text())
    validation = manifest["validation"]
    if len(panel) != len(cached) or len(panel) != 1000:
        raise AssertionError("Fixed panel/cache does not have 1000 queries")
    if len(validation["query_mols"]) != len(panel):
        raise AssertionError("Strict validation query count changed")
    train_keys = set(data.mol.key[data.mol.split.eq("train")])
    if set(panel.molecule_key) & train_keys:
        raise AssertionError("Query structure overlaps training")
    scores_all = []
    torch.set_num_threads(4)
    for index, (item, previous) in enumerate(zip(panel.itertuples(index=False), cached, strict=True)):
        candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
        target = int(item.target_index)
        query = np.asarray(item.query_cache_rows, dtype=np.int64)
        if (int(item.mol_idx) != previous["mol_idx"]
                or target != previous["target_index"]
                or int(item.mol_idx) != validation["query_mols"][index]
                or candidates[target] != int(item.mol_idx)
                or data.mol.split.iloc[item.mol_idx] != "validation"
                or not np.all(data.spec.split.iloc[query].eq("validation"))
                or not np.all(data.mol_idx[query] == item.mol_idx)
                or not np.all(data.mol.formula.iloc[candidates].eq(item.formula))):
            raise AssertionError(f"Fixed panel/validation mismatch at {index}")
        with torch.inference_mode():
            probs = model(*data.batch(query, catalog, device)).mean(0).sigmoid()
            fp = torch.as_tensor(np.asarray(data.fp[candidates], dtype=np.float32), device=device)
            scores = tanimoto(probs[None], fp).cpu().numpy().astype(float)
        if rank_from_scores(scores, target) != validation["ranks"][index]:
            raise AssertionError(f"Strict score/rank parity failed at query {index}")
        scores_all.append(scores)
        if (index + 1) % 250 == 0:
            print(f"strict fixed scores {index + 1}/{len(panel)}", flush=True)
    return scores_all


def paired(rows: list[dict], first: str, second: str, *, seed: int,
           cluster: bool = False) -> dict:
    if not rows:
        return {"rows": 0}
    a = np.asarray([0 if row[first + "_rank"] is None else 1 / row[first + "_rank"] for row in rows])
    b = np.asarray([0 if row[second + "_rank"] is None else 1 / row[second + "_rank"] for row in rows])
    delta = a - b
    rng = np.random.default_rng(seed)
    if cluster:
        keys = pd.factorize(np.asarray([row["molecule_key"] for row in rows], dtype=object), sort=False)[0]
        n = int(keys.max()) + 1
        sums = np.bincount(keys, weights=delta, minlength=n)
        counts = np.bincount(keys, minlength=n)
        draws = rng.integers(n, size=(2000, n))
        draws_delta = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        unit = "structure key, retaining all source measurements"
        resamples = 2000
    else:
        draws = rng.integers(len(rows), size=(5000, len(rows)))
        draws_delta = delta[draws].mean(axis=1)
        unit = "molecule query"
        resamples = 5000
    return {
        "rows": len(rows),
        "unique_structure_keys": len({row["molecule_key"] for row in rows}) if cluster else None,
        "first": first, "second": second,
        "first_mrr25": float(a.mean()), "second_mrr25": float(b.mean()),
        "first_top1": float(np.mean([row[first + "_rank"] == 1 for row in rows])),
        "second_top1": float(np.mean([row[second + "_rank"] == 1 for row in rows])),
        "delta_mrr25_first_minus_second": float(delta.mean()),
        "delta_bootstrap_95pct": [float(x) for x in np.quantile(draws_delta, [0.025, 0.975])],
        "better": int(np.count_nonzero(delta > 0)),
        "worse": int(np.count_nonzero(delta < 0)),
        "unchanged": int(np.count_nonzero(delta == 0)),
        "bootstrap_unit": unit, "bootstrap_resamples": resamples,
    }


def evaluate_fixed(output: Path) -> dict:
    panel = pd.read_parquet(INPUTS["fixed_panel"])
    old = load_jsonl(INPUTS["fixed_scores"])
    strict = fixed_strict_scores(panel, old)
    rows = []
    for item, row, strict_scores in zip(panel.itertuples(index=False), old, strict, strict=True):
        target = int(item.target_index)
        a = np.asarray(row["A_scores"], dtype=float)
        m1 = np.asarray(row["M1_scores"], dtype=float)
        if len(a) != len(strict_scores) or len(a) != 25:
            raise AssertionError("Fixed 25 candidate set changed")
        if rank_from_scores(a, target) != row["A_rank"] or rank_from_scores(m1, target) != row["M1_rank"]:
            raise AssertionError("A or original M1 score/rank cache mismatch")
        rows.append({
            "mol_idx": int(item.mol_idx), "target_index": target,
            "A_rank": row["A_rank"], "STRICT_rank": rank_from_scores(strict_scores, target),
            "OLD_M4_rank": rank_from_scores(blend(a, m1, .75), target),
            "M5_rank": rank_from_scores(blend(a, strict_scores, .75), target),
            "STRICT_scores": strict_scores.tolist(),
        })
    write_jsonl(output / "fixed1000-per-molecule.jsonl", rows)
    old_m4 = json.loads(INPUTS["old_m4_grid"].read_text())["score_blends"]["A75+M125"]["panel"]["mrr25"]
    if abs(np.mean([1 / row["OLD_M4_rank"] for row in rows]) - old_m4) > 1e-12:
        raise AssertionError("Old M4 fixed-panel MRR parity failed")
    return compare(rows, cluster=False, seed=20261117)


def evaluate_proxy(output: Path) -> dict:
    scores = load_jsonl(INPUTS["proxy_scores"])
    control = pd.read_csv(INPUTS["control"], dtype=str, keep_default_na=False).set_index("molecule_id")
    truth = pd.read_csv(INPUTS["labels"], dtype=str).set_index("molecule_id").metric_key.to_dict()
    if len(scores) != len(control) or len(scores) != 80 or set(control.index) != set(truth):
        raise AssertionError("Saved parent80 inputs differ")
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for row in scores:
        mid = row["molecule_id"]
        names = control.loc[mid, "smiles"].split(";")
        a = np.asarray(row["A_scores"], dtype=float)
        old_m1 = np.asarray(row["M1_original_scores"], dtype=float)
        strict = np.asarray(row["M1_ce_strict_scores"], dtype=float)
        if len(names) != len(a) or len(names) != len(old_m1) or len(names) != len(strict):
            raise AssertionError("Saved parent candidate set differs")
        ranks = {}
        for label, second in (("OLD_M4", old_m1), ("M5", strict)):
            fused = parent_blend(names, a, second, .75)
            order, _ = rerank(names, fused, weight=.25)
            reranked = [names[index] for index in order]
            if reranked[0] != names[0] or sorted(reranked) != sorted(names):
                raise AssertionError("Parent fusion changed top1 or candidate set")
            ranks[label] = parent_rank(reranked, truth[mid])
        rows.append({
            "molecule_id": mid, "parent_rank": row["parent_rank"],
            "A_rank": row["A_rank"], "STRICT_rank": row["M1_ce_strict_rank"],
            "OLD_M4_rank": ranks["OLD_M4"], "M5_rank": ranks["M5"],
        })
    write_jsonl(output / "parent80-per-molecule.jsonl", rows)
    old_m4 = json.loads(INPUTS["old_m4_grid"].read_text())["score_blends"]["A75+M125"]["proxy"]["mrr25"]
    if abs(np.mean([1 / row["OLD_M4_rank"] if row["OLD_M4_rank"] else 0 for row in rows]) - old_m4) > 1e-12:
        raise AssertionError("Old M4 parent80 MRR parity failed")
    return compare(rows, cluster=False, seed=20261118)


def evaluate_natural(output: Path) -> dict:
    panel = pd.read_parquet(INPUTS["natural_panel"])
    old = load_jsonl(INPUTS["natural_old_m4"])
    strict = load_jsonl(INPUTS["natural_strict"])
    if len(panel) != len(old) or len(panel) != len(strict) or len(panel) != 7112:
        raise AssertionError("Natural source panel/caches differ")
    rows = []
    for item, previous, latest in zip(panel.itertuples(index=False), old, strict, strict=True):
        identity = (item.source, item.molecule_key, int(item.mol_idx), len(item.candidate_mol_indices))
        for cached in (previous, latest):
            if identity != (cached["source"], cached["molecule_key"], cached["mol_idx"], cached["candidate_count"]):
                raise AssertionError("Natural candidate panel differs from score cache")
        if (previous["A_rank"] != latest["A_rank"]
                or previous["M1_rank"] != latest["M1_rank"]
                or item.candidate_mol_indices[int(item.target_index)] != int(item.mol_idx)):
            raise AssertionError("Natural query or label mismatch")
        if len(item.candidate_mol_indices) == 1:
            new_rank = 1
        else:
            a = np.asarray(previous["A_scores"], dtype=float)
            s = np.asarray(latest["STRICT_scores"], dtype=float)
            if len(a) != len(item.candidate_mol_indices) or len(s) != len(a):
                raise AssertionError("Natural candidate scores are incomplete")
            if rank_from_scores(a, item.target_index) != previous["A_rank"] or rank_from_scores(s, item.target_index) != latest["STRICT_rank"]:
                raise AssertionError("Natural score/rank parity failed")
            new_rank = rank_from_scores(blend(a, s, .75), item.target_index)
        rows.append({
            "source": item.source, "molecule_key": item.molecule_key,
            "mol_idx": int(item.mol_idx), "candidate_count": len(item.candidate_mol_indices),
            "query_all_timsTOF": bool(item.query_all_timsTOF),
            "A_rank": previous["A_rank"], "STRICT_rank": latest["STRICT_rank"],
            "OLD_M4_rank": previous["M4_rank"], "M5_rank": new_rank,
        })
    write_jsonl(output / "natural-per-molecule.jsonl", rows)
    multi = [row for row in rows if row["candidate_count"] > 1]
    result = {
        "multi_candidate_primary": compare(multi, cluster=True, seed=20261119),
        "single_candidate_trivial": compare([row for row in rows if row["candidate_count"] == 1], cluster=True, seed=20261120),
        "sources": {}, "candidate_count_bands": {},
    }
    for index, source in enumerate(("gnps", "riken", "massbank", "enveda-np-examples")):
        result["sources"][source] = compare([row for row in multi if row["source"] == source], cluster=True, seed=20261130 + index)
    for index, (label, test) in enumerate({
        "2-5": lambda n: 2 <= n <= 5,
        "6-24": lambda n: 6 <= n <= 24,
        "25": lambda n: n == 25,
    }.items()):
        result["candidate_count_bands"][label] = compare([row for row in multi if test(row["candidate_count"])], cluster=True, seed=20261140 + index)
    old_m4 = json.loads((MIXUP / "natural-source-full/m4-evaluation/report.json").read_text())["metrics"]["multi_candidate_primary"]["M4_mrr25"]
    if abs(result["multi_candidate_primary"]["M5_vs_OLD_M4"]["second_mrr25"] - old_m4) > 1e-12:
        raise AssertionError("Old M4 natural panel MRR parity failed")
    return result


def compare(rows: list[dict], *, cluster: bool, seed: int) -> dict:
    if not rows:
        return {"rows": 0}
    result = {"rows": len(rows)}
    for index, baseline in enumerate(("A", "STRICT", "OLD_M4")):
        result["M5_vs_" + baseline] = paired(rows, "M5", baseline, seed=seed + index, cluster=cluster)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MIXUP / "M5-a75-strictm1-local")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "condition": "M5 = 0.75*z(A candidate scores) + 0.25*z(CE-strict M1 candidate scores)",
        "formula_scope": "same full formula candidate group only; no candidate addition or removal",
        "fixed1000": evaluate_fixed(args.output),
        "parent80": evaluate_proxy(args.output),
        "natural_sources": evaluate_natural(args.output),
        "parent_RRF": "weight 0.25 applied exactly once on saved 80 proxy, with top1 protected",
        "no_parent_RRF_on_fixed_or_natural": "no genuine 0.477 parent rankings for these validation queries",
        "natural_panel_note": "Post-hoc supplementary panel, variable candidate count; compare only paired differences within its strata, not absolute MRR against fixed25, parent80 or LB.",
        "bootstrap": "5000 molecule resamples on fixed1000/parent80; 2000 structure-key resamples on natural panel",
        "input_sha256": {key: sha(path) for key, path in INPUTS.items()},
    }
    output_path = args.output / "report.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "fixed1000": report["fixed1000"],
        "parent80": report["parent80"],
        "natural_multi": report["natural_sources"]["multi_candidate_primary"],
        "report": str(output_path),
    }), flush=True)


if __name__ == "__main__":
    main()

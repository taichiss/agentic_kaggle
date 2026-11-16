"""Evaluate the M4 A75:M1 score blend on the fixed four-source validation panel."""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_natural_source_followup import paired_delta
from explore_mixup_adaptive import blend
from train_mix_same import Cache, tanimoto

ALPHA_A = 0.75
MODEL = "M4"
BASELINES = ("A", "M1", "WEAK")
SOURCES = ("gnps", "riken", "massbank", "enveda-np-examples")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(rows: list[dict], *, seed: int) -> dict:
    if not rows:
        return {"molecule_source_pairs": 0}
    ranks = np.asarray([row["M4_rank"] for row in rows])
    result = {
        "molecule_source_pairs": len(rows),
        "unique_structure_keys": len({row["molecule_key"] for row in rows}),
        "M4_mrr25": float(np.mean(1 / ranks)),
        "M4_top1": float(np.mean(ranks == 1)),
    }
    for index, baseline in enumerate(BASELINES):
        result["M4_vs_" + baseline] = paired_delta(
            rows, MODEL, baseline, seed=seed + index
        )
    return result


def summarize(rows: list[dict], *, seed: int) -> dict:
    multi = [row for row in rows if row["candidate_count"] > 1]
    result = {
        "multi_candidate_primary": metrics(multi, seed=seed),
        "single_candidate_trivial": metrics(
            [row for row in rows if row["candidate_count"] == 1], seed=seed + 10
        ),
        "source": {},
        "candidate_count_bands": {},
        "source_x_timsTOF_only": {},
    }
    for index, source in enumerate(SOURCES):
        source_rows = [row for row in multi if row["source"] == source]
        result["source"][source] = metrics(source_rows, seed=seed + 20 + index)
        result["source_x_timsTOF_only"][source] = metrics(
            [row for row in source_rows if row["query_all_timsTOF"]],
            seed=seed + 30 + index,
        )
    bands = {
        "2-5": lambda count: 2 <= count <= 5,
        "6-24": lambda count: 6 <= count <= 24,
        "25": lambda count: count == 25,
    }
    for index, (name, predicate) in enumerate(bands.items()):
        result["candidate_count_bands"][name] = metrics(
            [row for row in multi if predicate(row["candidate_count"])],
            seed=seed + 40 + index,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--prior-ranks", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path, required=True)
    parser.add_argument("--A-checkpoint", type=Path, required=True)
    parser.add_argument("--M1-checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--parent80-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full panel evaluation")
    torch.set_num_threads(4)
    frozen_spec = importlib.util.spec_from_file_location("frozen_mist_model", args.frozen_model)
    frozen = importlib.util.module_from_spec(frozen_spec)
    frozen_spec.loader.exec_module(frozen)
    payload_a = torch.load(args.A_checkpoint, map_location="cpu", weights_only=False)
    payload_m1 = torch.load(args.M1_checkpoint, map_location="cpu", weights_only=False)
    if (payload_a["model_config"] != payload_m1["model_config"]
            or not np.array_equal(payload_a["formula_counts"], payload_m1["formula_counts"])
            or (payload_a["max_peaks"], payload_a["ppm"], payload_a["candidate_formulas"])
            != (payload_m1["max_peaks"], payload_m1["ppm"],
                payload_m1["candidate_formulas"])):
        raise AssertionError("A/M1 model or CHNO catalog differs")
    catalog = frozen.FormulaCatalog(
        payload_a["formula_counts"], max_peaks=payload_a["max_peaks"],
        ppm=payload_a["ppm"], k=payload_a["candidate_formulas"],
    )
    models = {}
    for name, payload in (("A", payload_a), ("M1", payload_m1)):
        model = frozen.FormulaTransformer(**payload["model_config"]).cuda().eval()
        model.load_state_dict(payload["state_dict"], strict=True)
        models[name] = model
    data = Cache(args.cache, args.precise_mzs, args.source_metadata)
    panel = pd.read_parquet(args.panel)
    train_keys = set(data.mol.key[data.mol.split.eq("train")])
    query_training_overlap = len(set(panel.molecule_key) & train_keys)
    if query_training_overlap:
        raise AssertionError("natural-panel query structure overlaps training")
    saved = [json.loads(line) for line in args.prior_ranks.read_text().splitlines()]
    if args.limit is not None:
        panel = panel.iloc[:args.limit]
        saved = saved[:args.limit]
    if len(panel) != len(saved):
        raise AssertionError("fixed panel and prior ranks differ")
    prior_report = json.loads(args.prior_report.read_text())
    if prior_report["panel_sha256"] != sha(args.panel):
        raise AssertionError("prior evaluation used a different panel")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.monotonic()
    with (args.output / "per-molecule.jsonl").open("w") as stream:
        for index, (item, previous) in enumerate(
            zip(panel.itertuples(index=False), saved, strict=True), 1
        ):
            candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
            target_index = int(item.target_index)
            query_rows = np.asarray(item.query_cache_rows, dtype=np.int64)
            if (item.source != previous["source"]
                    or item.mol_idx != previous["mol_idx"]
                    or len(candidates) != previous["candidate_count"]
                    or candidates[target_index] != item.mol_idx
                    or data.mol.split.iloc[item.mol_idx] != "validation"
                    or not np.all(data.spec.split.iloc[query_rows].eq("validation"))
                    or not np.all(data.mol_idx[query_rows] == item.mol_idx)
                    or not np.all(data.mol.formula.iloc[candidates].eq(item.formula))):
                raise AssertionError("candidate panel differs from five-model evaluation")
            record = dict(previous)
            if len(candidates) == 1:
                record["M4_rank"] = 1
            else:
                batch = data.batch(
                    query_rows, catalog, "cuda"
                )
                fp = torch.as_tensor(
                    np.asarray(data.fp[candidates], dtype=np.float32), device="cuda"
                )
                scores = {}
                with torch.inference_mode():
                    for name, model in models.items():
                        probs = model(*batch).mean(0).sigmoid()
                        scores[name] = tanimoto(probs[None], fp).cpu().numpy().astype(float)
                for name in ("A", "M1"):
                    check_rank = int(np.count_nonzero(scores[name] >= scores[name][target_index]))
                    if check_rank != previous[name + "_rank"]:
                        raise AssertionError(f"{name} score/rank parity failed at {item.mol_idx}")
                fused = blend(scores["A"], scores["M1"], ALPHA_A)
                record["M4_rank"] = int(np.count_nonzero(fused >= fused[target_index]))
                record["A_scores"] = scores["A"].tolist()
                record["M1_scores"] = scores["M1"].tolist()
                record["M4_fused_scores"] = fused.tolist()
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            if index % 500 == 0 or index == len(panel):
                stream.flush()
                print(json.dumps({"evaluated": index, "total": len(panel),
                                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
    parent80 = json.loads(args.parent80_reference.read_text())["score_blends"]["A75+M125"][
        "proxy"
    ]
    report = {
        "condition": "A75:M1 25 score z fusion within the one full-formula candidate group",
        "parent_RRF_applied_to_natural_panel": False,
        "parent_RRF_note": (
            "No 0.477 parent ranks exist for these validation queries. Candidate order was "
            "randomly shuffled when the panel was fixed, so treating it as a parent rank "
            "would introduce random prior information. The separate saved 80-row parent "
            "proxy applies RRF exactly once at weight 0.25."
        ),
        "parent80_RRF_once_reference": parent80,
        "panel_sha256": sha(args.panel),
        "prior_ranks_sha256": sha(args.prior_ranks),
        "A_checkpoint_sha256": sha(args.A_checkpoint),
        "M1_checkpoint_sha256": sha(args.M1_checkpoint),
        "frozen_model_sha256": sha(args.frozen_model),
        "parent80_reference_sha256": sha(args.parent80_reference),
        "score_rank_parity": "A and M1 reproduced every saved natural-panel rank",
        "query_training_structure_overlap": query_training_overlap,
        "query_and_true_label_validation_only": True,
        "scored_molecule_source_pairs": len(records),
        "seconds": round(time.monotonic() - start, 1),
        "bootstrap": {
            "unit": "structure key; all source measurements move together",
            "resamples": 2000,
            "seed": args.seed,
        },
        "metrics": summarize(records, seed=args.seed),
        "comparison_note": (
            "Post-hoc supplementary panel with variable candidate counts. Compare paired "
            "differences within candidate-count bands; do not compare absolute MRR with the "
            "fixed-25 panel, saved 80 parent proxy, or Public LB."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"primary": report["metrics"]["multi_candidate_primary"],
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

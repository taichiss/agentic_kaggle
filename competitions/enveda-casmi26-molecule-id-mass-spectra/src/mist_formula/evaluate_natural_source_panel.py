"""Evaluate fixed natural-source candidates with five frozen 2048-bit checkpoints."""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from train_mix_same import Cache, tanimoto

CONDITIONS = ("A", "M1", "M2", "M3", "WEAK")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_metrics(rows: list[dict], *, seed: int, bootstrap: int = 2000) -> dict:
    if not rows:
        return {"molecule_source_pairs": 0}
    baseline = np.asarray([row["A_rank"] for row in rows], dtype=np.int16)
    keys = pd.factorize(
        np.asarray([row["molecule_key"] for row in rows], dtype=object), sort=False
    )[0]
    groups = int(keys.max()) + 1
    group_counts = np.bincount(keys, minlength=groups)
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(groups, size=(bootstrap, groups))
    result = {
        "molecule_source_pairs": len(rows),
        "unique_structure_keys": groups,
        "A_mrr25": float(np.mean(1 / baseline)),
        "A_top1": float(np.mean(baseline == 1)),
    }
    for condition in CONDITIONS[1:]:
        ranks = np.asarray([row[f"{condition}_rank"] for row in rows], dtype=np.int16)
        delta = 1 / ranks - 1 / baseline
        group_deltas = np.bincount(keys, weights=delta, minlength=groups)
        bootstrap_delta = (
            group_deltas[bootstrap_indices].sum(axis=1)
            / group_counts[bootstrap_indices].sum(axis=1)
        )
        result[condition] = {
            "mrr25": float(np.mean(1 / ranks)),
            "top1": float(np.mean(ranks == 1)),
            "delta_mrr25_vs_A": float(np.mean(delta)),
            "delta_cluster_bootstrap_95pct": [
                float(value) for value in np.quantile(bootstrap_delta, [0.025, 0.975])
            ],
            "improved": int(np.sum(ranks < baseline)),
            "worsened": int(np.sum(ranks > baseline)),
            "unchanged": int(np.sum(ranks == baseline)),
        }
    return result


def summarize(records: list[dict], *, seed: int) -> dict:
    multi = [row for row in records if row["candidate_count"] > 1]
    single = [row for row in records if row["candidate_count"] == 1]
    result = {
        "all_pairs": paired_metrics(records, seed=seed),
        "multi_candidate_primary": paired_metrics(multi, seed=seed + 1),
        "single_candidate_trivial": paired_metrics(single, seed=seed + 2),
        "source": {},
        "candidate_count_bands": {},
        "source_x_timsTOF_only": {},
    }
    for index, source in enumerate(("gnps", "riken", "massbank", "enveda-np-examples")):
        group = [row for row in records if row["source"] == source]
        group_multi = [row for row in group if row["candidate_count"] > 1]
        result["source"][source] = {
            "all_pairs": paired_metrics(group, seed=seed + 10 + index),
            "multi_candidate_primary": paired_metrics(
                group_multi, seed=seed + 20 + index
            ),
            "single_candidate_count": len(group) - len(group_multi),
        }
        timstof = [row for row in group_multi if row["query_all_timsTOF"]]
        result["source_x_timsTOF_only"][source] = paired_metrics(
            timstof, seed=seed + 30 + index
        )
    bands = {
        "2-4": lambda count: 2 <= count <= 4,
        "5-9": lambda count: 5 <= count <= 9,
        "10-24": lambda count: 10 <= count <= 24,
        "25": lambda count: count == 25,
    }
    for index, (name, predicate) in enumerate(bands.items()):
        group = [row for row in records if predicate(row["candidate_count"])]
        result["candidate_count_bands"][name] = paired_metrics(
            group, seed=seed + 40 + index
        )
    result["bootstrap"] = {
        "resamples": 2000,
        "unit": "structure key, retaining all source measurements of each sampled key",
        "point_estimate_unit": "molecule-source pair",
        "seed": seed,
    }
    result["interpretation_note"] = (
        "This is a post-hoc supplementary panel with varying candidate counts. "
        "Compare paired model differences within this panel and candidate-count bands; "
        "do not compare absolute MRR with the earlier fixed-25 panel."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    provided = dict(entry.split("=", 1) for entry in args.model)
    if set(provided) != set(CONDITIONS) or len(provided) != len(args.model):
        parser.error(f"--model must provide each of {CONDITIONS} exactly once")
    checkpoints = {name: Path(provided[name]) for name in CONDITIONS}
    if not all(path.is_file() for path in checkpoints.values()):
        parser.error("checkpoint is missing")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full panel evaluation")
    device = "cuda"
    frozen_spec = importlib.util.spec_from_file_location("frozen_mist_model", args.frozen_model)
    frozen = importlib.util.module_from_spec(frozen_spec)
    frozen_spec.loader.exec_module(frozen)
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in checkpoints.items()
    }
    reference = payloads["A"]
    for name, payload in payloads.items():
        if (payload["model_config"] != reference["model_config"]
                or not np.array_equal(payload["formula_counts"], reference["formula_counts"])
                or (payload["max_peaks"], payload["ppm"], payload["candidate_formulas"])
                != (reference["max_peaks"], reference["ppm"],
                    reference["candidate_formulas"])):
            raise AssertionError(f"{name} model or CHNO catalog differs from A")
    catalog = frozen.FormulaCatalog(
        reference["formula_counts"], max_peaks=reference["max_peaks"],
        ppm=reference["ppm"], k=reference["candidate_formulas"],
    )
    models = {}
    for name, payload in payloads.items():
        model = frozen.FormulaTransformer(**payload["model_config"]).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        models[name] = model
    del payloads
    data = Cache(args.cache, args.precise_mzs, args.source_metadata)
    panel = pd.read_parquet(args.panel)
    if args.limit is not None:
        panel = panel.iloc[:args.limit]
    train_keys = set(data.mol.key[data.mol.split.eq("train")])
    if set(panel.molecule_key) & train_keys:
        raise AssertionError("query structure overlaps training")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.monotonic()
    with (args.output / "per-molecule.jsonl").open("w") as stream:
        for index, item in enumerate(panel.itertuples(index=False), 1):
            target = int(item.mol_idx)
            candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
            target_index = int(item.target_index)
            rows = np.asarray(item.query_cache_rows, dtype=np.int64)
            if (data.mol.split.iloc[target] != "validation" or candidates[target_index] != target
                    or not np.all(data.mol.formula.iloc[candidates].eq(item.formula))
                    or not np.all(data.mol_idx[rows] == target)):
                raise AssertionError("panel query/candidates do not match target")
            record = {
                "source": item.source,
                "molecule_key": item.molecule_key,
                "mol_idx": target,
                "candidate_count": len(candidates),
                "query_spectra_count": len(rows),
                "query_all_timsTOF": bool(item.query_all_timsTOF),
            }
            if len(candidates) == 1:
                for name in CONDITIONS:
                    record[f"{name}_rank"] = 1
            else:
                batch = data.batch(rows, catalog, device)
                fp = torch.as_tensor(
                    np.asarray(data.fp[candidates], dtype=np.float32), device=device
                )
                with torch.inference_mode():
                    for name, model in models.items():
                        probabilities = model(*batch).mean(0).sigmoid()
                        scores = tanimoto(probabilities[None], fp).cpu().numpy()
                        record[f"{name}_rank"] = int(
                            np.count_nonzero(scores >= scores[target_index])
                        )
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            if index % 500 == 0 or index == len(panel):
                stream.flush()
                print(json.dumps({"evaluated": index, "total": len(panel),
                                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
    report = {
        "panel_sha256": sha(args.panel),
        "frozen_inference_sha256": sha(args.frozen_model),
        "checkpoints_sha256": {name: sha(path) for name, path in checkpoints.items()},
        "cache_spectra_sha256": sha(args.cache / "spectra.parquet"),
        "precise_mzs_sha256": sha(args.precise_mzs),
        "source_metadata_sha256": sha(args.source_metadata / "manifest.json"),
        "query_training_structure_overlap": 0,
        "scored_pairs": len(records),
        "scored_spectra": int(sum(row["query_spectra_count"] for row in records)),
        "seconds": round(time.monotonic() - start, 1),
        "metrics": summarize(records, seed=args.seed),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scored_pairs": len(records), "seconds": report["seconds"],
                      "primary": report["metrics"]["multi_candidate_primary"]}), flush=True)


if __name__ == "__main__":
    main()

"""Score CE-strict M1 on the existing four-source held-out candidate panel."""

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
from train_mix_same import Cache, tanimoto


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(rows: list[dict], *, seed: int) -> dict:
    if not rows:
        return {"molecule_source_pairs": 0}
    ranks = np.asarray([row["STRICT_rank"] for row in rows], dtype=int)
    result = {
        "molecule_source_pairs": len(rows),
        "unique_structure_keys": len({row["molecule_key"] for row in rows}),
        "STRICT_mrr25": float(np.mean(1 / ranks)),
        "STRICT_top1": float(np.mean(ranks == 1)),
    }
    for index, baseline in enumerate(("A", "M1", "WEAK")):
        result["STRICT_vs_" + baseline] = paired_delta(
            rows, "STRICT", baseline, seed=seed + index
        )
    return result


def summarize(rows: list[dict], *, seed: int) -> dict:
    multi = [row for row in rows if row["candidate_count"] > 1]
    singles = [row for row in rows if row["candidate_count"] == 1]
    result = {
        "multi_candidate_primary": metrics(multi, seed=seed),
        "single_candidate_trivial": metrics(singles, seed=seed + 10),
        "source": {},
        "candidate_count_bands": {},
        "source_x_timsTOF_only": {},
    }
    for index, source in enumerate(("gnps", "riken", "massbank", "enveda-np-examples")):
        selected = [row for row in multi if row["source"] == source]
        result["source"][source] = metrics(selected, seed=seed + 20 + index)
        result["source_x_timsTOF_only"][source] = metrics(
            [row for row in selected if row["query_all_timsTOF"]],
            seed=seed + 30 + index,
        )
    bands = {
        "2-5": lambda n: 2 <= n <= 5,
        "6-24": lambda n: 6 <= n <= 24,
        "25": lambda n: n == 25,
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    model_spec = importlib.util.spec_from_file_location("frozen_mist_model", args.frozen_model)
    frozen = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(frozen)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    catalog = frozen.FormulaCatalog(
        payload["formula_counts"], max_peaks=payload["max_peaks"],
        ppm=payload["ppm"], k=payload["candidate_formulas"],
    )
    model = frozen.FormulaTransformer(**payload["model_config"]).cuda().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    manifest = json.loads(args.manifest.read_text())
    if (manifest["checkpoint_sha256"] != sha(args.checkpoint)
            or manifest["formula_rows"] != len(payload["formula_counts"])):
        raise AssertionError("checkpoint or CHNO catalog differs from training manifest")
    data = Cache(args.cache, args.precise_mzs, args.source_metadata)
    panel = pd.read_parquet(args.panel)
    prior = [json.loads(line) for line in args.prior_ranks.open()]
    prior_report = json.loads(args.prior_report.read_text())
    if prior_report["panel_sha256"] != sha(args.panel) or len(panel) != len(prior):
        raise AssertionError("fixed natural-source panel differs")
    if set(panel.molecule_key) & set(data.mol.key[data.mol.split.eq("train")]):
        raise AssertionError("query structure overlaps training")
    if args.limit is not None:
        panel = panel.iloc[:args.limit]
        prior = prior[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.monotonic()
    with (args.output / "per-molecule.jsonl").open("w") as stream:
        for index, (item, previous) in enumerate(
            zip(panel.itertuples(index=False), prior, strict=True), 1
        ):
            candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
            query = np.asarray(item.query_cache_rows, dtype=np.int64)
            target = int(item.mol_idx)
            target_index = int(item.target_index)
            if (item.source != previous["source"]
                    or target != previous["mol_idx"]
                    or len(candidates) != previous["candidate_count"]
                    or candidates[target_index] != target
                    or data.mol.split.iloc[target] != "validation"
                    or not np.all(data.spec.split.iloc[query].eq("validation"))
                    or not np.all(data.mol_idx[query] == target)
                    or not np.all(data.mol.formula.iloc[candidates].eq(item.formula))):
                raise AssertionError("query, labels, or candidates differ from fixed panel")
            record = dict(previous)
            if len(candidates) == 1:
                record["STRICT_rank"] = 1
            else:
                batch = data.batch(query, catalog, "cuda")
                fp = torch.as_tensor(
                    np.asarray(data.fp[candidates], dtype=np.float32), device="cuda"
                )
                with torch.inference_mode():
                    probs = model(*batch).mean(0).sigmoid()
                    scores = tanimoto(probs[None], fp).cpu().numpy()
                record["STRICT_rank"] = int(
                    np.count_nonzero(scores >= scores[target_index])
                )
                record["STRICT_scores"] = scores.tolist()
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            if index % 500 == 0 or index == len(panel):
                stream.flush()
                print(json.dumps({"evaluated": index, "total": len(panel),
                                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
    report = {
        "checkpoint_sha256": sha(args.checkpoint),
        "manifest_sha256": sha(args.manifest),
        "frozen_model_sha256": sha(args.frozen_model),
        "panel_sha256": sha(args.panel),
        "prior_ranks_sha256": sha(args.prior_ranks),
        "prior_report_sha256": sha(args.prior_report),
        "query_training_structure_overlap": 0,
        "scored_molecule_source_pairs": len(records),
        "seconds": round(time.monotonic() - start, 1),
        "bootstrap": {"unit": "structure key; repeated source measurements stay together",
                      "resamples": 2000, "seed": args.seed},
        "parent_RRF_applied": False,
        "comparison_note": "Post-hoc variable-candidate natural-source panel. Parent rank is unavailable; compare paired model-only ranks within the same candidate list, not absolute MRR against Public LB or parent80.",
        "metrics": summarize(records, seed=args.seed),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"]["multi_candidate_primary"]), flush=True)


if __name__ == "__main__":
    main()

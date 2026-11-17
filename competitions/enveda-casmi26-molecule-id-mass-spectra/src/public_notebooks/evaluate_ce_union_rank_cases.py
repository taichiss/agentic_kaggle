"""Inspect six G4 rescue/loss structures with the frozen 0.477 Engine1 ranker.

This is a local diagnostic. Truth spectra/SIDs and pool PID are excluded per
query. The parent ranker weights may still have seen these structures, so the
result is not an unseen-molecule CV or a complete competition submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/public_notebooks"))
import evaluate_ce_generator_holdout as holdout  # noqa: E402
from ce_diverse_generator import install as install_g4  # noqa: E402
from ce_union_generator import install as install_g5  # noqa: E402

MODEL = Path("/kaggle/input/casmi26-attributed-v4g-reproduction-assets/casmi26-v4b-models")
CODE = MODEL / "code"
FE_CODE = CODE / "fe_v4"
sys.path[:0] = [str(CODE), str(FE_CODE)]
import ctxlib  # noqa: E402
import hooks_v1  # noqa: E402
import v1engine  # noqa: E402
from casmi import fpnet  # noqa: E402
from casmi.engine import FEATURES, Engine, EngineCfg  # noqa: E402
from casmi.library import Library  # noqa: E402
from casmi.pool import Pool  # noqa: E402

DERIVED = ROOT / "data/derived/ce-views-20261117"
G4_RECORDS = DERIVED / "enveda-ce-diverse-holdout/per-group.jsonl"
G5_RECORDS = DERIVED / "enveda-ce-union-holdout/per-group.jsonl"
PARENT_MOUNT = DERIVED / "g5-local-mount-v2/working"
POOL_DIR = Path("/kaggle/working/work")
POPULARITY_DIR = Path("/kaggle/input/casmi26-pubchem-popularity-prior")
OUTPUT = DERIVED / "g5-six-full-rank"
TOPN = 60
POOLPOP_MU = 0.15


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def selected_groups() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    g4_rows = read_jsonl(G4_RECORDS)
    g5_rows = read_jsonl(G5_RECORDS)
    if len(g4_rows) != len(g5_rows) or len(g4_rows) != 200:
        raise ValueError("G4/G5 held-out groups differ")
    g4 = {row["truth"]: row for row in g4_rows}
    g5 = {row["truth"]: row for row in g5_rows}
    keys = [
        row["truth"] for row in g4_rows
        if row["baseline_truth_generated"] != row["diverse"]["truth_in_generated_candidates"]
    ]
    if len(keys) != 6:
        raise ValueError("Expected exactly three G4 rescues and three G4 losses")
    groups, selection = holdout.select_groups(200, 20261117)
    if selection["selected_groups"] != 200:
        raise ValueError("Held-out group selection changed")
    chosen = [item for item in groups if item["key"] in keys]
    if len(chosen) != 6:
        raise ValueError("Selected query structure keys differ")
    for item in chosen:
        key = item["key"]
        if item["mol_idx"] != g4[key]["mol_idx"] or item["cache_rows"] != g4[key]["cache_rows"]:
            raise ValueError("Frozen G4 query rows differ")
        if item["mol_idx"] != g5[key]["mol_idx"] or item["cache_rows"] != g5[key]["cache_rows"]:
            raise ValueError("Frozen G5 query rows differ")
    return chosen, g4, g5


def load_engine_resources(device: str):
    manifest = json.loads((MODEL / "MANIFEST.json").read_text())
    external = [
        p for p in PARENT_MOUNT.glob("external_reference_*")
        if p.is_dir() and (p / "spec_meta.parquet").is_file()
    ]
    if len(external) != 1:
        raise ValueError("Expected one frozen extended reference library")
    library = Library(str(external[0]), verbose=False)
    pool = Pool(str(POOL_DIR), verbose=False)
    if len(pool.key2pid) != len(pool.key):
        raise ValueError("Frozen pool has duplicate structure keys")
    train_fp = np.load(POOL_DIR / "train_fp_sel.npy")
    if len(train_fp) != len(library.struct_mass):
        raise ValueError("Reference SID and fingerprint rows differ")
    full_fp = Path(
        "/kaggle/input/casmi26-attributed-v4g-reproduction-assets/"
        "casmi26-fpnet-full1/fpnet_full1.pt"
    )
    fp_path = full_fp if full_fp.is_file() else MODEL / "fpnet_0.pt"
    bank = fpnet.ModelBank([str(fp_path)], device=device)
    ranker_path = MODEL / manifest["ranker"]["file"]
    ranker = pickle.loads(ranker_path.read_bytes())
    features = list(ranker["features"])
    boosters = [lgb.Booster(model_str=value) for value in ranker["boosters"]]
    return library, pool, train_fp, bank, manifest, features, boosters, {
        "reference_spec_meta": sha(external[0] / "spec_meta.parquet"),
        "pool_meta": sha(POOL_DIR / "pool_meta.parquet"),
        "fp_bank": sha(fp_path),
        "model_manifest": sha(MODEL / "MANIFEST.json"),
        "ranker": sha(ranker_path),
    }


def score_candidates(x: np.ndarray, names: list[str], features: list[str],
                     boosters: list[lgb.Booster]) -> np.ndarray:
    pos = {name: index for index, name in enumerate(names)}
    missing = [name for name in features if name not in pos]
    if missing:
        raise ValueError(f"Frozen ranker features unavailable: {missing[:5]}")
    selected = np.ascontiguousarray(x[:, [pos[name] for name in features]], dtype=np.float32)
    return np.mean([model.predict(selected, num_threads=4) for model in boosters], axis=0)


def ranking(candidates, scores: np.ndarray, truth: str, pool_pop: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    seen = set()
    keys, pids = [], []
    for index in order:
        key = str(candidates.key[index])
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
        pids.append(int(candidates.pid[index]))
    raw_rank = keys.index(truth) + 1 if truth in seen else 0
    shortlist_keys = keys[:TOPN]
    shortlist_pids = np.asarray(pids[:TOPN], dtype=np.int64)
    shortlist_scores = np.asarray([
        float(scores[next(index for index in order if str(candidates.key[index]) == key)])
        for key in shortlist_keys
    ], dtype=np.float64)
    if len(shortlist_keys) > 1:
        z = (shortlist_scores - shortlist_scores.mean()) / (shortlist_scores.std() + 1e-9)
        is_generated = shortlist_pids < 0
        fused = z + POOLPOP_MU * np.where(
            is_generated, 0.0, pool_pop[np.maximum(shortlist_pids, 0)])
        pool_slots = np.where(~is_generated)[0]
        permutation = np.arange(len(shortlist_keys))
        permutation[pool_slots] = pool_slots[np.argsort(-fused[pool_slots], kind="stable")]
        shortlist_keys = [shortlist_keys[index] for index in permutation]
    rank60 = shortlist_keys.index(truth) + 1 if truth in shortlist_keys else 0
    return {
        "all_candidates": len(candidates.pid),
        "unique_candidates": len(keys),
        "truth_in_generated": any(
            str(key) == truth and int(pid) < 0
            for key, pid in zip(candidates.key, candidates.pid, strict=True)
        ),
        "truth_rank_all": raw_rank,
        "truth_rank_top60": rank60,
        "truth_rank_top25": rank60 if 0 < rank60 <= 25 else 0,
        "reciprocal_rank_25": 1.0 / rank60 if 0 < rank60 <= 25 else 0.0,
    }


def run(output: Path, device: str) -> dict:
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    selected, g4, g5 = selected_groups()
    raw = holdout.load_raw_rows([row for item in selected for row in item["cache_rows"]])
    library, pool, train_fp, bank, manifest, ranker_features, boosters, hashes = (
        load_engine_resources(device)
    )
    key_to_sids: dict[str, list[int]] = defaultdict(list)
    for sid, key in enumerate(library.struct_key):
        key_to_sids[str(key)].append(sid)
    sorted_order = np.argsort(library.sid, kind="stable")
    sorted_sid = library.sid[sorted_order]
    pool_pop = (
        np.load(POPULARITY_DIR / "pool_lsid.npy").astype(np.float64)
        + np.load(POPULARITY_DIR / "pool_lpmid.npy").astype(np.float64)
    )
    if len(pool_pop) != len(pool.key):
        raise ValueError("Frozen pool popularity rows differ")
    engines = {
        mode: Engine(
            library, pool, train_fp,
            cfg=EngineCfg(generate=True), bank=bank, verbose=False,
        )
        for mode in ("g0", "g4", "g5")
    }
    viewer = v1engine.V1FE(
        engines["g0"],
        [str(MODEL / model["file"]) for model in manifest["fe_models"]],
        manifest["families"],
        device=device,
    )
    hooks_v1.install(engines["g4"])
    hooks_v1.install(engines["g5"])
    install_g4(engines["g4"])
    install_g5(engines["g5"])

    records = {item["key"]: {
        "truth": item["key"],
        "mol_idx": item["mol_idx"],
        "cache_rows": item["cache_rows"],
        "g4_candidate_only_rescue": (
            not g4[item["key"]]["baseline_truth_generated"]
            and g4[item["key"]]["diverse"]["truth_in_generated_candidates"]
        ),
        "g4_candidate_only_loss": (
            g4[item["key"]]["baseline_truth_generated"]
            and not g4[item["key"]]["diverse"]["truth_in_generated_candidates"]
        ),
        "g5_candidate_only_truth": g5[item["key"]]["union"]["truth_in_generated_candidates"],
    } for item in selected}
    for mode in ("g0", "g4", "g5"):
        engine = engines[mode]
        if viewer.E is not engine:
            viewer.E = engine
            viewer.res = ctxlib.resources_from_engine(engine)
            for module in viewer.mods.values():
                if hasattr(module, "prepare"):
                    module.prepare(viewer.res)
        for item in selected:
            truth = item["key"]
            sids = key_to_sids.get(truth, [])
            pid = int(pool.key2pid.get(truth, -1))
            if not sids or pid < 0:
                raise ValueError("Held-out truth absent from reference or pool")
            exclusion = np.zeros(len(library.sid), dtype=bool)
            indices = []
            for sid in sids:
                left = int(np.searchsorted(sorted_sid, sid, side="left"))
                right = int(np.searchsorted(sorted_sid, sid, side="right"))
                indices.extend(sorted_order[left:right].tolist())
            exclusion[indices] = True
            if set(library.sid[exclusion]) != set(sids):
                raise AssertionError("Truth reference SID exclusion is incomplete")
            spectra, target = holdout.make_query(item, raw)
            candidates, x, f, names, _ = viewer.run(
                spectra, target, exclude=exclusion, drop_pid=pid,
            )
            if np.any(candidates.pid == pid):
                raise AssertionError("Truth pool PID survived the holdout mask")
            if any(
                str(key) == truth and candidate_pid >= 0
                for key, candidate_pid in zip(candidates.key, candidates.pid, strict=True)
            ):
                raise AssertionError("Truth entered through the candidate pool")
            if len(candidates.pid) == 0:
                result = {
                    "all_candidates": 0, "unique_candidates": 0,
                    "truth_in_generated": False,
                    "truth_rank_all": 0, "truth_rank_top60": 0,
                    "truth_rank_top25": 0, "reciprocal_rank_25": 0.0,
                }
            else:
                scores = score_candidates(
                    np.concatenate([x, f], axis=1), list(FEATURES) + list(names),
                    ranker_features, boosters,
                )
                result = ranking(candidates, scores, truth, pool_pop)
            result["excluded_truth_sids"] = sids
            result["excluded_truth_reference_spectra"] = len(indices)
            result["excluded_truth_pool_pid"] = pid
            records[truth][mode] = result
            print(mode, truth, result["truth_rank_top25"], result["truth_rank_all"],
                  f"{time.monotonic() - started:.0f}s", flush=True)
    rows = list(records.values())
    summary = {
        "scope": (
            "six Enveda-180 groups, frozen 0.477 Engine1 full features and ranker; "
            "no Engine2/PubChem/final blend"
        ),
        "meaning": "targeted post-hoc diagnostic; not unseen-molecule CV or Public LB",
        "n_groups": len(rows),
        "n_g4_rescue": sum(row["g4_candidate_only_rescue"] for row in rows),
        "n_g4_loss": sum(row["g4_candidate_only_loss"] for row in rows),
        "top25_truth": {
            mode: sum(row[mode]["truth_rank_top25"] > 0 for row in rows)
            for mode in ("g0", "g4", "g5")
        },
        "mean_reciprocal_rank25": {
            mode: float(np.mean([row[mode]["reciprocal_rank_25"] for row in rows]))
            for mode in ("g0", "g4", "g5")
        },
        "model_device": device,
        "input_sha256": {
            **hashes,
            "g4_records": sha(G4_RECORDS),
            "g5_records": sha(G5_RECORDS),
            "adapter_g4": sha(ROOT / "src/public_notebooks/ce_diverse_generator.py"),
            "adapter_g5": sha(ROOT / "src/public_notebooks/ce_union_generator.py"),
            "evaluator": sha(Path(__file__)),
        },
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(summary, indent=2))
    (output / "per-group.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    run(args.output, args.device)


if __name__ == "__main__":
    main()

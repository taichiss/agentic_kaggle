"""Label-free G0/G4/G5 frozen 0.477 Engine1 ranking on full 400 test groups.

The output is diagnostic: Engine1 through the top-60 popularity reorder, not
Engine2, PubChem, final fusion, a competition submission, or an accuracy score.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "src/public_notebooks"
sys.path.insert(0, str(LOCAL))

import audit_ce_generator_test as generation  # noqa: E402
import evaluate_ce_union_rank_cases as frozen  # noqa: E402
from ce_diverse_generator import choose_ce_views  # noqa: E402
from ce_diverse_generator import install as install_g4  # noqa: E402
from ce_union_generator import install as install_g5  # noqa: E402

DERIVED = ROOT / "data/derived/ce-views-20261117"
PREVIOUS = DERIVED / "g5-test-generator-audit/per-molecule.jsonl"
OUTPUT = DERIVED / "g5-full-test-engine1-rank"
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


def rank_keys(candidates, scores: np.ndarray, pool_pop: np.ndarray) -> list[str]:
    """Mirror frozen 0.477 cell 39: dedupe, top 60, reorder pool slots."""
    order = np.argsort(-scores, kind="stable")
    seen = set()
    keys, pids, values = [], [], []
    for index in order:
        key = str(candidates.key[index])
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
        pids.append(int(candidates.pid[index]))
        values.append(float(scores[index]))
        if len(keys) >= TOPN:
            break
    if len(keys) > 1:
        values_array = np.asarray(values, np.float64)
        z = (values_array - values_array.mean()) / (values_array.std() + 1e-9)
        pid_array = np.asarray(pids, np.int64)
        is_generated = pid_array < 0
        fused = z + POOLPOP_MU * np.where(
            is_generated, 0.0, pool_pop[np.maximum(pid_array, 0)]
        )
        pool_slots = np.where(~is_generated)[0]
        permutation = np.arange(len(keys))
        permutation[pool_slots] = pool_slots[np.argsort(
            -fused[pool_slots], kind="stable"
        )]
        keys = [keys[index] for index in permutation]
    return keys


def prepare_viewer(viewer, engine) -> None:
    if viewer.E is engine:
        return
    viewer.E = engine
    viewer.res = frozen.ctxlib.resources_from_engine(engine)
    for module in viewer.mods.values():
        if hasattr(module, "prepare"):
            module.prepare(viewer.res)


def run_mode(mode: str, engine, viewer, groups, previous, ranker_features,
             boosters, pool_pop, *, eligible_only: bool) -> tuple[dict[str, dict], dict[str, list]]:
    prepare_viewer(viewer, engine)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    analogs = {}
    started = time.monotonic()
    path = OUTPUT / f"{mode}.jsonl"
    with path.open("w") as output:
        for index, (molecule_id, spectra, target) in enumerate(groups, 1):
            expected = previous[molecule_id]
            selected = [float(ce) for ce, _ in choose_ce_views(
                spectra[:engine.cfg.max_query_spectra]
            )]
            if selected != expected["selected_ce_views"]:
                raise ValueError(f"Frozen CE selector diverged: {molecule_id}")
            if eligible_only and not selected:
                continue
            candidates, x, f, names, _ = viewer.run(spectra, target)
            generated = [
                str(key) for key, pid in zip(candidates.key, candidates.pid, strict=True)
                if pid < 0
            ]
            if generated != expected["generated_keys"][mode]:
                raise AssertionError(f"Generator audit mismatch: {mode} {molecule_id}")
            if len(candidates.pid):
                scores = frozen.score_candidates(
                    np.concatenate([x, f], axis=1),
                    list(frozen.FEATURES) + list(names), ranker_features, boosters,
                )
                keys = rank_keys(candidates, scores, pool_pop)
            else:
                keys = []
            row = {
                "molecule_id": molecule_id,
                "eligible": bool(selected),
                "candidate_set_changed_vs_g0": bool(
                    expected[f"{mode}_all_set_changed"] if mode != "g0" else False
                ),
                "top60": keys,
                "top25": keys[:25],
                "top1": keys[0] if keys else None,
                "n_candidates": len(candidates.pid),
                "n_generated": len(generated),
            }
            results[molecule_id] = row
            analogs[molecule_id] = list(engine._last_an or [])
            output.write(json.dumps(row) + "\n")
            if index == 1 or index % 25 == 0:
                print(f"{mode}: {index}/400 in {time.monotonic()-started:.0f}s", flush=True)
    return results, analogs


def compare(base: dict[str, dict], other: dict[str, dict], ids: list[str],
            candidate_changed: dict[str, bool]) -> dict:
    rows = []
    for molecule_id in ids:
        a, b = base[molecule_id], other[molecule_id]
        same_set = set(a["top25"]) == set(b["top25"])
        rows.append({
            "molecule_id": molecule_id,
            "eligible": a["eligible"],
            "candidate_set_changed": candidate_changed[molecule_id],
            "top1_changed": a["top1"] != b["top1"],
            "top25_set_changed": not same_set,
            "top25_order_changed": a["top25"] != b["top25"],
            "top25_rank_only_changed": same_set and a["top25"] != b["top25"],
            "top25_new_count": len(set(b["top25"]) - set(a["top25"])),
        })

    def count(rows: list[dict]) -> dict:
        return {
            "groups": len(rows),
            "candidate_set_changed": sum(r["candidate_set_changed"] for r in rows),
            "top1_changed": sum(r["top1_changed"] for r in rows),
            "top25_set_changed": sum(r["top25_set_changed"] for r in rows),
            "top25_order_changed": sum(r["top25_order_changed"] for r in rows),
            "top25_rank_only_changed": sum(r["top25_rank_only_changed"] for r in rows),
            "top25_new_candidates_total": sum(r["top25_new_count"] for r in rows),
            "top1_changed_with_candidate_set_changed": sum(
                r["top1_changed"] and r["candidate_set_changed"] for r in rows
            ),
            "top25_set_changed_with_candidate_set_changed": sum(
                r["top25_set_changed"] and r["candidate_set_changed"] for r in rows
            ),
        }
    return {
        "all400": count(rows),
        "ce_eligible": count([r for r in rows if r["eligible"]]),
        "ce_ineligible": count([r for r in rows if not r["eligible"]]),
        "per_molecule": rows,
    }


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    previous_rows = read_jsonl(PREVIOUS)
    previous = {row["molecule_id"]: row for row in previous_rows}
    groups = generation.query_groups()
    if len(groups) != len(previous) or len(previous) != 400:
        raise ValueError("Generator audit and full test groups differ")
    ids = [molecule_id for molecule_id, _, _ in groups]
    if set(ids) != set(previous):
        raise ValueError("Generator audit molecule IDs differ")
    library, pool, train_fp, bank, manifest, features, boosters, hashes = (
        frozen.load_engine_resources(device)
    )
    pool_pop = (
        np.load(frozen.POPULARITY_DIR / "pool_lsid.npy").astype(np.float64)
        + np.load(frozen.POPULARITY_DIR / "pool_lpmid.npy").astype(np.float64)
    )
    if len(pool_pop) != len(pool.key):
        raise ValueError("Pool popularity array differs from frozen pool")
    engines = {
        mode: frozen.Engine(
            library, pool, train_fp, cfg=frozen.EngineCfg(generate=True),
            bank=bank, verbose=False,
        )
        for mode in ("g0", "g4", "g5")
    }
    viewer = frozen.v1engine.V1FE(
        engines["g0"], [str(frozen.MODEL / row["file"]) for row in manifest["fe_models"]],
        manifest["families"], device=device,
    )
    frozen.hooks_v1.install(engines["g4"])
    frozen.hooks_v1.install(engines["g5"])
    install_g4(engines["g4"])
    install_g5(engines["g5"])
    runs, analogs = {}, {}
    for mode in ("g0", "g4", "g5"):
        runs[mode], analogs[mode] = run_mode(
            mode, engines[mode], viewer, groups, previous, features, boosters,
            pool_pop, eligible_only=mode != "g0",
        )
        if mode != "g0":
            for molecule_id in runs[mode]:
                if analogs[mode][molecule_id] != analogs["g0"][molecule_id]:
                    raise AssertionError(f"Merged analogs changed: {mode} {molecule_id}")
    ineligible = [
        molecule_id for molecule_id in ids
        if not previous[molecule_id]["selected_ce_views"]
    ]
    # For no eligible CE views, both adapters execute exactly the G0 generator
    # once; verify three examples through V1FE before imputing the other 118.
    parity_ids = ineligible[:3]
    for mode in ("g4", "g5"):
        prepare_viewer(viewer, engines[mode])
        for molecule_id in parity_ids:
            _, spectra, target = next(row for row in groups if row[0] == molecule_id)
            candidates, x, f, names, _ = viewer.run(spectra, target)
            scores = frozen.score_candidates(
                np.concatenate([x, f], axis=1),
                list(frozen.FEATURES) + list(names), features, boosters,
            )
            keys = rank_keys(candidates, scores, pool_pop)
            if keys != runs["g0"][molecule_id]["top60"]:
                raise AssertionError(f"No-CE parity failed: {mode} {molecule_id}")
        for molecule_id in ineligible:
            runs[mode][molecule_id] = {
                **runs["g0"][molecule_id], "candidate_set_changed_vs_g0": False,
            }
    candidate_changed = {"g4_vs_g0": {}, "g5_vs_g0": {}, "g5_vs_g4": {}}
    for molecule_id, _, target in groups:
        pooled = generation.pool_keys(pool, target, engines["g0"].cfg)[1]
        candidate_sets = {
            mode: pooled | set(previous[molecule_id]["generated_keys"][mode])
            for mode in ("g0", "g4", "g5")
        }
        candidate_changed["g4_vs_g0"][molecule_id] = (
            candidate_sets["g4"] != candidate_sets["g0"]
        )
        candidate_changed["g5_vs_g0"][molecule_id] = (
            candidate_sets["g5"] != candidate_sets["g0"]
        )
        candidate_changed["g5_vs_g4"][molecule_id] = (
            candidate_sets["g5"] != candidate_sets["g4"]
        )
    g4 = compare(runs["g0"], runs["g4"], ids, candidate_changed["g4_vs_g0"])
    g5 = compare(runs["g0"], runs["g5"], ids, candidate_changed["g5_vs_g0"])
    g5_vs_g4 = compare(runs["g4"], runs["g5"], ids, candidate_changed["g5_vs_g4"])
    reversions = [
        molecule_id for molecule_id in ids
        if runs["g4"][molecule_id]["top1"] != runs["g0"][molecule_id]["top1"]
        and runs["g5"][molecule_id]["top1"] == runs["g0"][molecule_id]["top1"]
    ]
    hashes.update({
        "test": sha(generation.TEST),
        "candidate_audit": sha(PREVIOUS),
        "ce_selector_g4": sha(LOCAL / "ce_diverse_generator.py"),
        "ce_union_g5": sha(LOCAL / "ce_union_generator.py"),
        "pool_lsid": sha(frozen.POPULARITY_DIR / "pool_lsid.npy"),
        "pool_lpmid": sha(frozen.POPULARITY_DIR / "pool_lpmid.npy"),
        "evaluator": sha(Path(__file__)),
    })
    report = {
        "scope": "label-free frozen 0.477 Engine1 through top60 pool popularity; no final fusion",
        "n_test_molecules": len(ids),
        "n_ce_eligible": 400 - len(ineligible),
        "n_ce_ineligible_imputed_after_three_parity_checks": len(ineligible) - len(parity_ids),
        "no_ce_parity_ids": parity_ids,
        "g4_vs_g0": {key: value for key, value in g4.items() if key != "per_molecule"},
        "g5_vs_g0": {key: value for key, value in g5.items() if key != "per_molecule"},
        "g5_vs_g4": {key: value for key, value in g5_vs_g4.items() if key != "per_molecule"},
        "g5_reverts_g4_top1_to_g0_count": len(reversions),
        "g5_reverts_g4_top1_to_g0_ids": reversions,
        "input_sha256": hashes,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "comparison.jsonl").write_text(
        "".join(json.dumps({
            "molecule_id": molecule_id,
            "g0": runs["g0"][molecule_id],
            "g4": runs["g4"][molecule_id],
            "g5": runs["g5"][molecule_id],
        }) + "\n" for molecule_id in ids)
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

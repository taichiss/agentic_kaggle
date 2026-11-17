"""Compare CE-union generator candidates on the same Enveda validation sample."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import evaluate_ce_generator_holdout as base
import numpy as np
from ce_union_generator import choose_ce_views, install, union_generated

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "data/derived/ce-views-20261117/enveda-generator-holdout"
OUTPUT = ROOT / "data/derived/ce-views-20261117/enveda-ce-union-holdout"


def load_baseline(path: Path) -> tuple[dict, list[dict]]:
    report = json.loads((path / "report.json").read_text())
    records = [json.loads(line) for line in (path / "per-group.jsonl").read_text().splitlines()]
    if report["n_groups"] != len(records):
        raise ValueError("Baseline report and per-group rows disagree")
    return report, records


def evaluate(groups: list[dict], raw: dict[int, dict], baseline: list[dict]) -> list[dict]:
    started = time.monotonic()
    library = base.Library(str(base.WORK), verbose=False)
    pool = base.Pool(str(base.WORK), verbose=False)
    train_fp = np.load(base.WORK / "train_fp_sel.npy")
    engine = base.Engine(
        library,
        pool,
        train_fp,
        cfg=base.EngineCfg(generate=True, use_frag=False, use_subformula=False),
        bank=None,
        verbose=False,
    )
    base.install_hooks(engine)
    key_to_sids: dict[str, list[int]] = defaultdict(list)
    for sid, key in enumerate(library.struct_key):
        key_to_sids[str(key)].append(sid)
    order = np.argsort(library.sid, kind="stable")
    sorted_sid = library.sid[order]
    exclusion = np.zeros(len(library.sid), dtype=bool)
    records = []
    for number, (item, prior) in enumerate(zip(groups, baseline, strict=True), 1):
        if item["mol_idx"] != prior["mol_idx"] or item["cache_rows"] != prior["cache_rows"]:
            raise ValueError("Baseline and G5 validation groups differ")
        base.reset_hooks(engine)
        truth = item["key"]
        sids = key_to_sids.get(truth, [])
        pid = int(pool.key2pid.get(truth, -1))
        if not sids or pid < 0:
            raise ValueError("Held-out truth missing from frozen library or pool")
        if sids != prior["library_truth_sids_excluded"] or pid != prior["pool_truth_pid_excluded"]:
            raise ValueError("Self-reference exclusion differs from baseline")
        excluded_indices = []
        for sid in sids:
            lo = int(np.searchsorted(sorted_sid, sid, side="left"))
            hi = int(np.searchsorted(sorted_sid, sid, side="right"))
            excluded_indices.extend(order[lo:hi].tolist())
        exclusion[excluded_indices] = True
        spectra, target = base.make_query(item, raw)
        baseline_analogs = engine.analogs(spectra, target, exclusion)
        selected = choose_ce_views(spectra)
        per_ce = []
        try:
            engine.cfg.analog_query = "each"
            for ce, spectrum in selected:
                per_ce.append((ce, engine.analogs([spectrum], target, exclusion)))
        finally:
            engine.cfg.analog_query = "merged"
        window = pool.window(target, engine.cfg.ppm_win)
        window = window[window != pid]
        exclude_keys = set(pool.key[index] for index in window)
        if baseline_analogs:
            generated, provenance, n_base = union_generated(
                engine.generate, baseline_analogs, per_ce, target, exclude_keys,
                limit=engine.cfg.gen_n_analog,
                min_similarity=engine.cfg.gen_min_sim,
                max_total=engine.cfg.gen_max_total,
            )
        else:
            generated, provenance, n_base = None, [], 0
        candidates = engine.candidates(target, drop_pid=pid, generated=generated)
        result = base.summarize_mode(candidates, truth)
        baseline_keys = set(prior["merged"]["generated_keys"])
        union_keys = set(result["generated_keys"])
        records.append(
            {
                "mol_idx": item["mol_idx"],
                "adduct_id": item["adduct_id"],
                "cache_rows": item["cache_rows"],
                "truth": truth,
                "eligible_ce_views": [ce for ce, _ in selected],
                "baseline_parent_sids": [
                    int(row[0]) for row in baseline_analogs if row[1] >= engine.cfg.gen_min_sim
                ][: engine.cfg.gen_n_analog],
                "union_parents": provenance,
                "baseline_products": n_base,
                "baseline_truth_generated": prior["merged"]["truth_in_generated_candidates"],
                "baseline_generated_candidate_count": prior["merged"]["generated_candidate_count"],
                "baseline_generated_keys": prior["merged"]["generated_keys"],
                "separate_truth_generated": prior["separate"]["truth_in_generated_candidates"],
                "separate_generated_keys": prior["separate"]["generated_keys"],
                "mild_truth_generated": prior["mild"]["truth_in_generated_candidates"],
                "mild_generated_keys": prior["mild"]["generated_keys"],
                "union": result,
                "added_generated_keys": sorted(union_keys - baseline_keys),
                "removed_generated_keys": sorted(baseline_keys - union_keys),
            }
        )
        exclusion[excluded_indices] = False
        if number == 1 or number % 25 == 0:
            print(
                f"evaluated {number}/{len(groups)} in {time.monotonic() - started:.0f}s", flush=True
            )
    return records


def parent_crosscheck(
    groups: list[dict], raw: dict[int, dict], records: list[dict], limit: int
) -> None:
    if limit <= 0:
        return
    library = base.Library(str(base.WORK), verbose=False)
    pool = base.Pool(str(base.WORK), verbose=False)
    train_fp = np.load(base.WORK / "train_fp_sel.npy")
    engine = base.Engine(
        library,
        pool,
        train_fp,
        cfg=base.EngineCfg(generate=True, use_frag=False, use_subformula=False),
        bank=None,
        verbose=False,
    )
    base.install_hooks(engine)
    install(engine)
    key_to_sids: dict[str, list[int]] = defaultdict(list)
    for sid, key in enumerate(library.struct_key):
        key_to_sids[str(key)].append(sid)
    exclusion = np.zeros(len(library.sid), dtype=bool)
    for item, direct in zip(groups[:limit], records[:limit], strict=True):
        base.reset_hooks(engine)
        truth = item["key"]
        sids = key_to_sids[truth]
        pid = int(pool.key2pid[truth])
        exclusion[np.isin(library.sid, sids)] = True
        spectra, target = base.make_query(item, raw)
        expected_analogs = engine.analogs(spectra, target, exclusion)
        candidates, _, _ = engine.run(spectra, target, exclude=exclusion, drop_pid=pid)
        actual_generated = [
            key for key, candidate_pid in zip(candidates.key, candidates.pid, strict=True)
            if candidate_pid < 0
        ]
        if actual_generated != [row["key"] for row in engine._last_gen]:
            raise AssertionError("G5 hooked _last_gen lost generated candidate context")
        if engine._last_an != expected_analogs:
            raise AssertionError("G5 changed the merged analog rank-feature source")
        baseline_keys = set(direct["baseline_generated_keys"])
        if any(
            row[3] == "pool" and row[4] == "key" and row[5] in baseline_keys
            for row in engine._derlog
        ):
            raise AssertionError("G5 attributed baseline generated products to pool")
        full = base.summarize_mode(candidates, truth)
        if full["generated_keys"] != direct["union"]["generated_keys"]:
            raise AssertionError("G5 direct candidate path disagrees with full Engine.run")
        exclusion[np.isin(library.sid, sids)] = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=PREVIOUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--parent-crosscheck", type=int, default=2)
    args = parser.parse_args()
    started = time.monotonic()
    baseline_report, baseline = load_baseline(args.baseline)
    groups, selection = base.select_groups(baseline_report["limit"], baseline_report["seed"])
    if selection != baseline_report["selection"]:
        raise ValueError("Baseline validation selection changed")
    raw = base.load_raw_rows([row for group in groups for row in group["cache_rows"]])
    records = evaluate(groups, raw, baseline)
    if not 0 <= args.parent_crosscheck <= len(groups):
        parser.error("parent-crosscheck must be within selected group count")
    parent_crosscheck(groups, raw, records, args.parent_crosscheck)
    comparisons = {}
    for label, truth_field, keys_field in (
        ("merged", "baseline_truth_generated", "baseline_generated_keys"),
        ("separate", "separate_truth_generated", "separate_generated_keys"),
        ("mild", "mild_truth_generated", "mild_generated_keys"),
    ):
        comparisons[f"{label}_to_union"] = {
            "truth_rescued": sum(
                not r[truth_field] and r["union"]["truth_in_generated_candidates"]
                for r in records
            ),
            "truth_lost": sum(
                r[truth_field] and not r["union"]["truth_in_generated_candidates"]
                for r in records
            ),
            "generated_candidate_set_changed": sum(
                set(r[keys_field]) != set(r["union"]["generated_keys"]) for r in records
            ),
            "generated_candidates_added_total": sum(
                len(set(r["union"]["generated_keys"]) - set(r[keys_field])) for r in records
            ),
            "generated_candidates_removed_total": sum(
                len(set(r[keys_field]) - set(r["union"]["generated_keys"])) for r in records
            ),
        }
    summary = {
        "metric": "generator candidate inclusion; not MRR@25 or Public LB",
        "n_groups": len(records),
        "seed": baseline_report["seed"],
        "selection": selection,
        "parent_crosschecked_groups": args.parent_crosscheck,
        "baseline_truth_generated": sum(r["baseline_truth_generated"] for r in records),
        "union_truth_generated": sum(
            r["union"]["truth_in_generated_candidates"] for r in records
        ),
        "baseline_truth_any_candidate": baseline_report["mode"]["merged"]["truth_any_candidate"],
        "union_truth_any_candidate": sum(
            r["union"]["truth_in_all_candidates"] for r in records
        ),
        "truth_rescued": sum(
            not r["baseline_truth_generated"] and r["union"]["truth_in_generated_candidates"]
            for r in records
        ),
        "truth_lost": sum(
            r["baseline_truth_generated"] and not r["union"]["truth_in_generated_candidates"]
            for r in records
        ),
        "eligible_ce_view_groups": sum(len(r["eligible_ce_views"]) >= 2 for r in records),
        "generator_parent_set_changed": sum(
            [p["sid"] for p in r["union_parents"]] != r["baseline_parent_sids"] for r in records
        ),
        "generated_candidate_set_changed": sum(
            bool(r["added_generated_keys"] or r["removed_generated_keys"]) for r in records
        ),
        "generated_candidates_added_total": sum(len(r["added_generated_keys"]) for r in records),
        "generated_candidates_removed_total": sum(
            len(r["removed_generated_keys"]) for r in records
        ),
        "comparisons": comparisons,
        "mean_union_generated_candidates": float(
            np.mean([r["union"]["generated_candidate_count"] for r in records])
        ),
        "input_sha256": {
            "baseline_report": base.sha(args.baseline / "report.json"),
            "baseline_per_group": base.sha(args.baseline / "per-group.jsonl"),
            "adapter": base.sha(Path(__file__).with_name("ce_union_generator.py")),
            "evaluator": base.sha(Path(__file__)),
        },
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "per-group.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}))


if __name__ == "__main__":
    main()

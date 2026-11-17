"""Paired generator-only CE-view diagnostic on held-out Enveda molecules.

The formula-Transformer split's validation structures remain in the frozen
0.477 reference library and candidate pool. This script removes each query's
own reference structure and pool entry before comparing the frozen merged
generator with separate measured views and mild same-structure CE Mixup.
The result measures candidate inclusion before the rankers, not MRR@25.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MODEL_CODE = (
    ROOT
    / "data/derived/hypothesis1-assets/casmi26-attributed-v4g-reproduction-assets"
    / "casmi26-v4b-models/code"
)
FE_CODE = MODEL_CODE / "fe_v4"
sys.path[:0] = [str(MODEL_CODE), str(FE_CODE)]

from casmi import chem  # noqa: E402
from casmi.engine import Engine, EngineCfg  # noqa: E402
from casmi.library import Library  # noqa: E402
from casmi.pool import Pool  # noqa: E402
from ce_mild_mixup import augmented_views  # noqa: E402
from ce_mild_mixup import install as install_ce_adapter  # noqa: E402
from hooks_v1 import install as install_hooks  # noqa: E402
from hooks_v1 import reset as reset_hooks  # noqa: E402

CACHE = ROOT / "data/derived/saner-multipositive/cache-parallel"
META = ROOT / "data/derived/mist-mixup-20261116/source-metadata"
PATTERNS = ROOT / "data/derived/mist-mixup-20261116/ce-pattern-signatures"
RAW = ROOT / "data/input/train.parquet"
TEST = ROOT / "data/input/test.parquet"
WORK = Path("/kaggle/working/work")
OUTPUT = ROOT / "data/derived/ce-views-20261117/enveda-generator-holdout"
CE_LEVELS = (20.0, 40.0, 60.0)
ADDUCTS = {1: "[M+H]+", 2: "[M-H]-"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_groups(limit: int, seed: int) -> tuple[list[dict], dict]:
    molecules = pd.read_parquet(CACHE / "molecules.parquet", columns=["key", "split"])
    spectra = pd.read_parquet(
        CACHE / "spectra.parquet", columns=["mol_idx", "adduct_id", "split", "length"]
    )
    categories = json.loads((META / "categories.json").read_text())
    ingest_id = categories["ingest_lib"].index("enveda-180")
    instrument_id = categories["instrument_type"].index("timsTOF")
    ingest = np.load(META / "ingest_id.npy", mmap_mode="r")
    instrument = np.load(META / "instrument_id.npy", mmap_mode="r")
    ce_kind = np.load(META / "ce_kind.npy", mmap_mode="r")
    ce_ev = np.load(META / "ce_ev.npy", mmap_mode="r")
    pattern = np.load(PATTERNS / "pattern_id.npy", mmap_mode="r")
    if any(len(a) != len(spectra) for a in (ingest, instrument, ce_kind, ce_ev, pattern)):
        raise ValueError("Source metadata no longer aligns with the spectrum cache")
    exact_patterns = json.loads((PATTERNS / "patterns.json").read_text())
    complete_ids = {
        int(item["id"])
        for item in exact_patterns
        if tuple(item["original_values_sorted"]) in {("20", "40", "60"), ("-60", "-40", "-20")}
        and item["unit"] == "ev"
    }
    if len(complete_ids) != 2:
        raise ValueError("Exact 20/40/60 composite CE patterns changed")
    adduct_id = spectra.adduct_id.to_numpy()
    eligible = np.flatnonzero(
        (ingest == ingest_id)
        & (instrument == instrument_id)
        & spectra.split.eq("validation").to_numpy()
        & np.isin(adduct_id, tuple(ADDUCTS))
    )
    rows = pd.DataFrame(
        {
            "cache_row": eligible,
            "mol_idx": spectra.mol_idx.to_numpy()[eligible],
            "adduct_id": adduct_id[eligible],
            "length": spectra.length.to_numpy()[eligible],
            "kind": ce_kind[eligible],
            "ce_ev": ce_ev[eligible],
            "pattern_id": pattern[eligible],
        }
    )
    rows["single_ce"] = np.where((rows.kind == 1) & rows.ce_ev.isin(CE_LEVELS), rows.ce_ev, np.nan)
    grouped = rows.groupby(["mol_idx", "adduct_id"], sort=True)
    candidates: list[dict] = []
    for (mol_idx, adduct), group in grouped:
        composite = group[group.pattern_id.isin(complete_ids)]
        if composite.empty:
            continue
        per_energy = []
        for energy in CE_LEVELS:
            match = group[group.single_ce == energy]
            if match.empty:
                break
            selected = match.sort_values(["length", "cache_row"], ascending=[False, True]).iloc[0]
            per_energy.append(int(selected.cache_row))
        if len(per_energy) == 3:
            combined = composite.sort_values(["length", "cache_row"], ascending=[False, True]).iloc[
                0
            ]
            candidates.append(
                {
                    "mol_idx": int(mol_idx),
                    "adduct_id": int(adduct),
                    "cache_rows": per_energy + [int(combined.cache_row)],
                }
            )
    by_mol: dict[int, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_mol[candidate["mol_idx"]].append(candidate)
    rng = np.random.default_rng(seed)
    chosen_mols = rng.choice(sorted(by_mol), size=min(limit, len(by_mol)), replace=False)
    selected = [by_mol[int(mol)][0] for mol in chosen_mols]
    for item in selected:
        if molecules.split.iloc[item["mol_idx"]] != "validation":
            raise ValueError("Query structure entered the training split")
        item["key"] = str(molecules.key.iloc[item["mol_idx"]])
    return selected, {
        "eligible_groups": len(candidates),
        "eligible_unique_molecules": len(by_mol),
        "selected_groups": len(selected),
        "selected_unique_molecules": len(set(int(i) for i in chosen_mols)),
    }


def load_raw_rows(cache_rows: list[int]) -> dict[int, dict]:
    mapping = np.load(PATTERNS / "raw_row_index.npy", mmap_mode="r")
    raw_rows = {int(mapping[row]): row for row in cache_rows}
    if len(raw_rows) != len(cache_rows):
        raise ValueError("Cache rows mapped to duplicate raw training rows")
    parquet = pq.ParquetFile(RAW)
    offsets = np.cumsum(
        [0] + [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
    )
    by_group: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for raw_row, cache_row in raw_rows.items():
        group = int(np.searchsorted(offsets, raw_row, side="right") - 1)
        by_group[group].append((int(raw_row - offsets[group]), cache_row))
    columns = [
        "ingest_lib",
        "adduct",
        "ionization_mode",
        "instrument_type",
        "precursor_mz",
        "collision_energy_ev",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]
    records = {}
    for group, pairs in sorted(by_group.items()):
        table = parquet.read_row_group(group, columns=columns)
        picked = table.take(pa.array([position for position, _ in pairs])).to_pylist()
        records.update(
            {cache_row: record for (_, cache_row), record in zip(pairs, picked, strict=True)}
        )
        del table
    if len(records) != len(cache_rows):
        raise ValueError("Some selected raw training spectra were not loaded")
    return records


def make_query(item: dict, raw_rows: dict[int, dict]) -> tuple[list[dict], float]:
    spectra = []
    expected_ce = (*CE_LEVELS, None)
    for energy, cache_row in zip(expected_ce, item["cache_rows"], strict=True):
        row = raw_rows[cache_row]
        ce = np.asarray(row["collision_energy_ev"], dtype=float).reshape(-1)
        ce_matches = (
            len(ce) == 1 and abs(float(ce[0]) - energy) <= 0.01
            if energy is not None
            else len(ce) == 3 and np.allclose(np.sort(ce), CE_LEVELS, atol=0.01)
        )
        if (
            row["ingest_lib"] != "enveda-180"
            or row["instrument_type"] != "timsTOF"
            or row["adduct"] != ADDUCTS[item["adduct_id"]]
            or not ce_matches
        ):
            raise ValueError("Raw CE-view row disagrees with the aligned cache metadata")
        mz = np.asarray(row["ms2_mzs"], dtype=np.float64)
        intensity = np.asarray(row["ms2_normalized_intensities"], dtype=np.float64)
        spectra.append(
            {
                "mz": mz,
                "it": intensity,
                "mode": 1 if row["ionization_mode"] == "positive" else -1,
                "adduct": row["adduct"],
                "prec": float(row["precursor_mz"]),
                "ce": float(np.median(ce)),
                "ce_n": len(ce),
                "instrument": row["instrument_type"],
            }
        )
    precursors = np.asarray([row["prec"] for row in spectra], dtype=float)
    adducts = np.asarray([row["adduct"] for row in spectra], dtype=str)
    mass = chem.neutral_mass(precursors, adducts)
    mass = mass[np.isfinite(mass)]
    if len(mass) != len(spectra):
        raise ValueError("Unknown adduct or invalid precursor neutral mass")
    return spectra, float(np.median(mass))


def summarize_mode(candidates, truth_key: str) -> dict:
    generated = [
        str(key) for key, pid in zip(candidates.key, candidates.pid, strict=True) if int(pid) < 0
    ]
    all_keys = set(map(str, candidates.key))
    return {
        "all_candidate_count": len(all_keys),
        "generated_candidate_count": len(generated),
        "truth_in_all_candidates": truth_key in all_keys,
        "truth_in_generated_candidates": truth_key in generated,
        "generated_keys": generated,
    }


def evaluate_parent(groups: list[dict], raw_rows: dict[int, dict], *, limit: int) -> list[dict]:
    started = time.monotonic()
    if not WORK.is_dir():
        raise FileNotFoundError("Frozen 0.477 Library/Pool cache is missing")
    library = Library(str(WORK), verbose=False)
    pool = Pool(str(WORK), verbose=False)
    train_fp = np.load(WORK / "train_fp_sel.npy")
    cfg = EngineCfg(generate=True, use_frag=False, use_subformula=False)
    engines = {
        name: Engine(library, pool, train_fp, cfg=EngineCfg(**vars(cfg)), bank=None, verbose=False)
        for name in ("merged", "separate", "mild")
    }
    for engine in engines.values():
        install_hooks(engine)
    install_ce_adapter(engines["separate"], include_mixup=False)
    install_ce_adapter(engines["mild"], include_mixup=True)
    key_to_sids: dict[str, list[int]] = defaultdict(list)
    for sid, key in enumerate(library.struct_key):
        key_to_sids[str(key)].append(sid)
    order = np.argsort(library.sid, kind="stable")
    sorted_sid = library.sid[order]
    exclusion = np.zeros(len(library.sid), dtype=bool)
    records: list[dict] = []
    for number, item in enumerate(groups[:limit], 1):
        truth = item["key"]
        sids = key_to_sids.get(truth, [])
        pid = int(pool.key2pid.get(truth, -1))
        if not sids or pid < 0:
            raise ValueError("Held-out query truth missing from frozen library or pool")
        if len(pool.key2pid) != len(pool.key):
            raise ValueError("Frozen pool has duplicate metric keys; one PID cannot hide truth")
        excluded_indices = []
        for sid in sids:
            lo = int(np.searchsorted(sorted_sid, sid, side="left"))
            hi = int(np.searchsorted(sorted_sid, sid, side="right"))
            excluded_indices.extend(order[lo:hi].tolist())
        exclusion[excluded_indices] = True
        spectra, target = make_query(item, raw_rows)
        rec = {
            "mol_idx": item["mol_idx"],
            "adduct_id": item["adduct_id"],
            "cache_rows": item["cache_rows"],
            "library_truth_sids_excluded": sids,
            "library_truth_spectra_excluded": len(excluded_indices),
            "pool_truth_pid_excluded": pid,
        }
        for name, engine in engines.items():
            reset_hooks(engine)
            candidate, _, _ = engine.run(spectra, target, exclude=exclusion, drop_pid=pid)
            rec[name] = summarize_mode(candidate, truth)
            if name != "merged":
                rec[name]["mixed_view_count"] = engine._ce_view_history[-1]["mixed_views"]
                rec[name]["generator_parent_changed"] = (
                    engine._ce_view_history[-1]["baseline_analog_sids"]
                    != engine._ce_view_history[-1]["generator_analog_sids"]
                )
        exclusion[excluded_indices] = False
        records.append(rec)
        if number == 1 or number % 25 == 0:
            print(
                f"evaluated {number}/{min(limit, len(groups))} groups "
                f"in {time.monotonic() - started:.0f}s",
                flush=True,
            )
    return records


def evaluate(groups: list[dict], raw_rows: dict[int, dict], *, limit: int) -> list[dict]:
    """Execute exactly the frozen Engine.run candidate-generation prefix.

    The full rank feature path is omitted. Derivation outputs are reused inside
    one query when the selected generator parent SIDs are identical, since the
    candidate *keys* depend on those SIDs and target mass, not analog scores.
    """
    started = time.monotonic()
    library = Library(str(WORK), verbose=False)
    pool = Pool(str(WORK), verbose=False)
    train_fp = np.load(WORK / "train_fp_sel.npy")
    engine = Engine(
        library,
        pool,
        train_fp,
        cfg=EngineCfg(generate=True, use_frag=False, use_subformula=False),
        bank=None,
        verbose=False,
    )
    install_hooks(engine)
    key_to_sids: dict[str, list[int]] = defaultdict(list)
    for sid, key in enumerate(library.struct_key):
        key_to_sids[str(key)].append(sid)
    order = np.argsort(library.sid, kind="stable")
    sorted_sid = library.sid[order]
    exclusion = np.zeros(len(library.sid), dtype=bool)
    records: list[dict] = []
    for number, item in enumerate(groups[:limit], 1):
        reset_hooks(engine)
        truth = item["key"]
        sids = key_to_sids.get(truth, [])
        pid = int(pool.key2pid.get(truth, -1))
        if not sids or pid < 0:
            raise ValueError("Held-out query truth missing from frozen library or pool")
        if len(pool.key2pid) != len(pool.key):
            raise ValueError("Frozen pool has duplicate metric keys; one PID cannot hide truth")
        excluded_indices = []
        for sid in sids:
            lo = int(np.searchsorted(sorted_sid, sid, side="left"))
            hi = int(np.searchsorted(sorted_sid, sid, side="right"))
            excluded_indices.extend(order[lo:hi].tolist())
        exclusion[excluded_indices] = True
        spectra, target = make_query(item, raw_rows)
        rec = {
            "mol_idx": item["mol_idx"],
            "adduct_id": item["adduct_id"],
            "cache_rows": item["cache_rows"],
            "library_truth_sids_excluded": sids,
            "library_truth_spectra_excluded": len(excluded_indices),
            "pool_truth_pid_excluded": pid,
        }
        engine.cfg.analog_query = "merged"
        baseline = engine.analogs(spectra, target, exclusion)
        window = pool.window(target, engine.cfg.ppm_win)
        window = window[window != pid]
        excluded_pool_keys = set(pool.key[index] for index in window)
        generate_cache: dict[tuple[int, ...], list[dict] | None] = {}
        for name in ("merged", "separate", "mild"):
            mixed_count = 0
            analogs = baseline
            if name != "merged" and baseline:
                views, mixed_count = augmented_views(
                    spectra,
                    include_mixup=name == "mild",
                    max_views=engine.cfg.max_query_spectra,
                )
                try:
                    engine.cfg.analog_query = "each"
                    analogs = engine.analogs(views, target, exclusion)
                finally:
                    engine.cfg.analog_query = "merged"
                if not analogs:
                    analogs = baseline
            parents = tuple(
                int(a[0])
                for a in [a for a in analogs if a[1] >= engine.cfg.gen_min_sim][
                    : engine.cfg.gen_n_analog
                ]
            )
            if not baseline:
                generated = None
            elif parents in generate_cache:
                generated = generate_cache[parents]
            else:
                generated = engine.generate(analogs, target, excluded_pool_keys)
                generate_cache[parents] = generated
            candidate = engine.candidates(target, drop_pid=pid, generated=generated)
            rec[name] = summarize_mode(candidate, truth)
            if name != "merged":
                rec[name]["mixed_view_count"] = mixed_count
                rec[name]["generator_parent_changed"] = [
                    int(a[0]) for a in baseline[: engine.cfg.gen_n_analog]
                ] != [int(a[0]) for a in analogs[: engine.cfg.gen_n_analog]]
        exclusion[excluded_indices] = False
        records.append(rec)
        if number == 1 or number % 25 == 0:
            print(
                f"evaluated {number}/{min(limit, len(groups))} groups "
                f"in {time.monotonic() - started:.0f}s",
                flush=True,
            )
    return records


def report(records: list[dict], selection: dict, args: argparse.Namespace) -> dict:
    names = ("merged", "separate", "mild")
    totals = {
        name: {
            "truth_generated": sum(row[name]["truth_in_generated_candidates"] for row in records),
            "truth_any_candidate": sum(row[name]["truth_in_all_candidates"] for row in records),
            "mean_generated_candidates": float(
                np.mean([row[name]["generated_candidate_count"] for row in records])
            ),
        }
        for name in names
    }
    paired = {}
    for left, right in (("merged", "separate"), ("separate", "mild")):
        paired[f"{left}_to_{right}"] = {
            "truth_rescued": sum(
                not row[left]["truth_in_all_candidates"] and row[right]["truth_in_all_candidates"]
                for row in records
            ),
            "truth_lost": sum(
                row[left]["truth_in_all_candidates"] and not row[right]["truth_in_all_candidates"]
                for row in records
            ),
            "generated_candidate_set_changed": sum(
                set(row[left]["generated_keys"]) != set(row[right]["generated_keys"])
                for row in records
            ),
            "generated_candidates_added_total": sum(
                len(set(row[right]["generated_keys"]) - set(row[left]["generated_keys"]))
                for row in records
            ),
            "generated_candidates_removed_total": sum(
                len(set(row[left]["generated_keys"]) - set(row[right]["generated_keys"]))
                for row in records
            ),
            "groups_with_added_generated_candidate": sum(
                bool(set(row[right]["generated_keys"]) - set(row[left]["generated_keys"]))
                for row in records
            ),
            "groups_with_removed_generated_candidate": sum(
                bool(set(row[left]["generated_keys"]) - set(row[right]["generated_keys"]))
                for row in records
            ),
        }
    test = pd.read_parquet(
        TEST, columns=["molecule_id", "adduct", "instrument_type", "collision_energy_ev"]
    )
    test["single_ce"] = test.collision_energy_ev.map(
        lambda values: (
            float(values[0]) if len(values) == 1 and float(values[0]) in CE_LEVELS else np.nan
        )
    )
    test_groups = test.groupby(["molecule_id", "adduct", "instrument_type"]).single_ce.agg(
        lambda values: set(values.dropna())
    )
    pair_groups = [key for key, energies in test_groups.items() if len(energies) >= 2]
    triple_groups = [key for key, energies in test_groups.items() if len(energies) == 3]
    return {
        "metric": "generator candidate inclusion; not MRR@25 or Public LB",
        "split": (
            "existing molecule-disjoint validation, self library structure and pool PID excluded"
        ),
        "source": (
            "enveda-180 timsTOF same molecule/adduct measured 20/40/60 eV "
            "singles and their exact composite"
        ),
        "n_groups": len(records),
        "selection": selection,
        "mode": totals,
        "paired": paired,
        "mild_queries_with_mixed_view": sum(row["mild"]["mixed_view_count"] > 0 for row in records),
        "mild_mixed_views_total": sum(row["mild"]["mixed_view_count"] for row in records),
        "label_free_test_ce_eligibility": {
            "all_molecule_adduct_instrument_groups": len(test_groups),
            "all_molecules": test.molecule_id.nunique(),
            "at_least_two_single_ce_groups": len(pair_groups),
            "at_least_two_single_ce_molecules": len({key[0] for key in pair_groups}),
            "complete_three_single_ce_groups": len(triple_groups),
            "complete_three_single_ce_molecules": len({key[0] for key in triple_groups}),
        },
        "input_sha256": {
            "cache_molecules": sha(CACHE / "molecules.parquet"),
            "cache_spectra": sha(CACHE / "spectra.parquet"),
            "raw_train": sha(RAW),
            "test": sha(TEST),
            "raw_row_index": sha(PATTERNS / "raw_row_index.npy"),
            "adapter": sha(Path(__file__).with_name("ce_mild_mixup.py")),
            "parent_engine": sha(MODEL_CODE / "casmi/engine.py"),
            "parent_hooks": sha(FE_CODE / "hooks_v1.py"),
            "pool_meta": sha(WORK / "pool_meta.parquet"),
            "train_structs": sha(WORK / "train_structs.parquet"),
            "evaluator": sha(Path(__file__)),
        },
        "seed": args.seed,
        "limit": args.limit,
    }


def write_deltas(records: list[dict], output: Path) -> None:
    molecule_keys = pd.read_parquet(CACHE / "molecules.parquet", columns=["key"]).key
    with (output / "per-molecule-deltas.jsonl").open("w") as stream:
        for row in records:
            delta = {
                "mol_idx": row["mol_idx"],
                "molecule_key": str(molecule_keys.iloc[row["mol_idx"]]),
                "adduct_id": row["adduct_id"],
                "cache_rows": row["cache_rows"],
                "library_truth_sids_excluded": row["library_truth_sids_excluded"],
                "pool_truth_pid_excluded": row["pool_truth_pid_excluded"],
            }
            for name in ("merged", "separate", "mild"):
                delta[f"{name}_truth_generated"] = row[name]["truth_in_generated_candidates"]
                delta[f"{name}_truth_any_candidate"] = row[name]["truth_in_all_candidates"]
                delta[f"{name}_generated_candidates"] = row[name]["generated_candidate_count"]
            for left, right in (("merged", "separate"), ("separate", "mild")):
                before = set(row[left]["generated_keys"])
                after = set(row[right]["generated_keys"])
                delta[f"{left}_to_{right}_added"] = sorted(after - before)
                delta[f"{left}_to_{right}_removed"] = sorted(before - after)
            stream.write(json.dumps(delta) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--parent-crosscheck", type=int, default=2)
    parser.add_argument("--rebuild-report", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        parser.error("limit must be between 1 and 500")
    if args.rebuild_report:
        prior = json.loads((args.output / "report.json").read_text())
        records = [
            json.loads(line) for line in (args.output / "per-group.jsonl").read_text().splitlines()
        ]
        args.seed = prior["seed"]
        args.limit = prior["limit"]
        summary = report(records, prior["selection"], args)
        summary["elapsed_seconds"] = prior["elapsed_seconds"]
        summary["parent_crosschecked_groups"] = prior["parent_crosschecked_groups"]
        (args.output / "report.json").write_text(json.dumps(summary, indent=2))
        write_deltas(records, args.output)
        print(json.dumps({"output": str(args.output), "report_rebuilt": True}))
        return
    t0 = time.monotonic()
    groups, selection = select_groups(args.limit, args.seed)
    raw = load_raw_rows([row for group in groups for row in group["cache_rows"]])
    records = evaluate(groups, raw, limit=args.limit)
    if args.parent_crosscheck:
        if not 0 <= args.parent_crosscheck <= len(groups):
            parser.error("parent-crosscheck must be within selected group count")
        parent = evaluate_parent(groups, raw, limit=args.parent_crosscheck)
        for direct, full in zip(records[: len(parent)], parent, strict=True):
            for name in ("merged", "separate", "mild"):
                if direct[name]["generated_keys"] != full[name]["generated_keys"]:
                    raise AssertionError(
                        f"Candidate-only path disagrees with full Engine.run for {name}"
                    )
    summary = report(records, selection, args)
    summary["parent_crosschecked_groups"] = args.parent_crosscheck
    summary["elapsed_seconds"] = round(time.monotonic() - t0, 1)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "per-group.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    write_deltas(records, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "summary": summary["mode"], "paired": summary["paired"]}
        )
    )


if __name__ == "__main__":
    main()

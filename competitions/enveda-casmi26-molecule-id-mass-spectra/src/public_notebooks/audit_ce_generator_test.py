"""Compare frozen G0/G4/G5 generated candidate sets on all 400 test groups.

This diagnostic reads no structure labels and does not run rank models. It
replays Engine.run's analog/generator/candidate prefix with the same extended
reference library and pool used in the G5 v2 notebook. Selected queries are
crosschecked against the actual hooked Engine.run path.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "src/public_notebooks"
MODEL = Path("/kaggle/input/casmi26-attributed-v4g-reproduction-assets/casmi26-v4b-models")
sys.path[:0] = [str(LOCAL), str(MODEL / "code"), str(MODEL / "code/fe_v4")]

from audit_ce_selector_test import spectra_for_group  # noqa: E402
from casmi import chem  # noqa: E402
from casmi.engine import Engine, EngineCfg  # noqa: E402
from casmi.library import Library  # noqa: E402
from casmi.pool import Pool  # noqa: E402
from ce_diverse_generator import choose_ce_views, round_robin_parents  # noqa: E402
from ce_diverse_generator import install as install_g4  # noqa: E402
from ce_union_generator import install as install_g5  # noqa: E402
from ce_union_generator import union_generated  # noqa: E402
from hooks_v1 import install as install_hooks  # noqa: E402
from hooks_v1 import reset as reset_hooks  # noqa: E402

DERIVED = ROOT / "data/derived/ce-views-20261117"
PARENT_MOUNT = DERIVED / "g5-local-mount-v2/working"
WORK = Path("/kaggle/working/work")
TEST = ROOT / "data/input/test.parquet"
OUTPUT = DERIVED / "g5-test-generator-audit"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load() -> tuple[Library, Pool, np.ndarray, Path]:
    external = [
        path for path in PARENT_MOUNT.glob("external_reference_*")
        if path.is_dir() and (path / "spec_meta.parquet").is_file()
    ]
    if len(external) != 1:
        raise ValueError("Expected exactly one frozen extended reference library")
    library = Library(str(external[0]), verbose=False)
    pool = Pool(str(WORK), verbose=False)
    train_fp = np.load(WORK / "train_fp_sel.npy")
    if len(train_fp) != len(library.struct_mass):
        raise ValueError("Reference structures and selected fingerprints differ")
    if len(pool.key2pid) != len(pool.key):
        raise ValueError("Frozen pool structure keys are not unique")
    return library, pool, train_fp, external[0]


def query_groups() -> list[tuple[str, list[dict], float]]:
    frame = pd.read_parquet(TEST)
    frame["_neutral_mass"] = chem.neutral_mass(
        frame.precursor_mz.to_numpy(dtype=np.float64), frame.adduct.values
    )
    groups = []
    for molecule_id, group in frame.groupby("molecule_id", sort=False):
        mass = group._neutral_mass.to_numpy(dtype=np.float64)
        mass = mass[np.isfinite(mass)]
        if not len(mass):
            raise ValueError(f"No neutral mass for {molecule_id}")
        groups.append((
            str(molecule_id), spectra_for_group(group), float(np.median(mass))
        ))
    if len(groups) != 400:
        raise ValueError("Full test does not contain 400 molecule groups")
    return groups


def pool_keys(pool: Pool, target: float, cfg: EngineCfg) -> tuple[set[str], set[str]]:
    window = pool.window(target, cfg.ppm_win)
    excluded = {str(pool.key[pid]) for pid in window}
    if len(window) == 0:
        window = pool.window(target, cfg.ppm_fallback)
    candidates = {str(pool.key[pid]) for pid in window}
    return excluded, candidates


def generated_keys(rows: list[dict]) -> list[str]:
    keys = [str(row["key"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise AssertionError("Generator emitted duplicate structure keys")
    return keys


def direct_group(engine: Engine, molecule_id: str, spectra: list[dict], target: float) -> dict:
    reset_hooks(engine)
    spectra = spectra[:engine.cfg.max_query_spectra]
    selected = choose_ce_views(spectra)
    baseline = engine.analogs(spectra, target)
    base_exclude, pooled = pool_keys(engine.pool, target, engine.cfg)
    per_ce = []
    if selected and baseline:
        saved = engine._last_an
        try:
            engine.cfg.analog_query = "each"
            per_ce = [
                (ce, engine.analogs([spectrum], target))
                for ce, spectrum in selected
            ]
        finally:
            engine.cfg.analog_query = "merged"
            engine._last_an = saved
    if baseline:
        reset_hooks(engine)
        g0 = engine.generate(baseline, target, base_exclude)
        if selected:
            chosen, _ = round_robin_parents(
                baseline, per_ce, limit=engine.cfg.gen_n_analog,
                min_similarity=engine.cfg.gen_min_sim,
            )
            if not chosen:
                chosen, _ = round_robin_parents(
                    baseline, [], limit=engine.cfg.gen_n_analog,
                    min_similarity=engine.cfg.gen_min_sim,
                )
            reset_hooks(engine)
            g4 = engine.generate(chosen, target, base_exclude)
            reset_hooks(engine)
            g5, _, n_base = union_generated(
                engine.generate, baseline, per_ce, target, base_exclude,
                limit=engine.cfg.gen_n_analog,
                min_similarity=engine.cfg.gen_min_sim,
                max_total=engine.cfg.gen_max_total, hook_state=engine,
            )
            if n_base != len(g0):
                raise AssertionError("G5 baseline generation differs from G0")
        else:
            g4 = g5 = g0
    else:
        g0 = g4 = g5 = []
    generated = {name: generated_keys(rows) for name, rows in (("g0", g0), ("g4", g4), ("g5", g5))}
    sets = {name: set(keys) for name, keys in generated.items()}
    all_sets = {name: pooled | keys for name, keys in sets.items()}
    if not sets["g0"] <= sets["g5"] or not all_sets["g0"] <= all_sets["g5"]:
        raise AssertionError("G5 lost a G0 candidate")
    if not selected and not (sets["g0"] == sets["g4"] == sets["g5"]):
        raise AssertionError("CE-ineligible query changed generated candidates")
    return {
        "molecule_id": molecule_id,
        "target_mass": target,
        "n_query_spectra": len(spectra),
        "selected_ce_views": [float(ce) for ce, _ in selected],
        "n_baseline_analogs": len(baseline),
        "n_ce_analogs": {str(int(ce)): len(rows) for ce, rows in per_ce},
        "pool_candidate_count": len(pooled),
        "generated_keys": generated,
        "generated_count": {name: len(keys) for name, keys in sets.items()},
        "all_candidate_count": {name: len(keys) for name, keys in all_sets.items()},
        "g4_generated_added": sorted(sets["g4"] - sets["g0"]),
        "g4_generated_removed": sorted(sets["g0"] - sets["g4"]),
        "g5_generated_added": sorted(sets["g5"] - sets["g0"]),
        "g5_generated_removed": sorted(sets["g0"] - sets["g5"]),
        "g4_all_set_changed": all_sets["g4"] != all_sets["g0"],
        "g5_all_set_changed": all_sets["g5"] != all_sets["g0"],
        "g4_generated_set_changed": sets["g4"] != sets["g0"],
        "g5_generated_set_changed": sets["g5"] != sets["g0"],
    }


def crosscheck(library: Library, pool: Pool, train_fp: np.ndarray,
               groups: list[tuple[str, list[dict], float]], records: list[dict]) -> list[str]:
    checks = []
    selected = []
    for eligible, changed in ((True, True), (True, False), (False, False)):
        match = next((
            index for index, row in enumerate(records)
            if bool(row["selected_ce_views"]) == eligible
            and row["g5_generated_set_changed"] == changed
        ), None)
        if match is not None and match not in selected:
            selected.append(match)
    engines = {}
    for mode in ("g0", "g4", "g5"):
        engine = Engine(
            library, pool, train_fp,
            cfg=EngineCfg(generate=True, use_frag=False, use_subformula=False),
            bank=None, verbose=False,
        )
        install_hooks(engine)
        if mode == "g4":
            install_g4(engine)
        elif mode == "g5":
            install_g5(engine)
        engines[mode] = engine
    for index in selected:
        molecule_id, spectra, target = groups[index]
        expected = records[index]
        analogs = {}
        for mode, engine in engines.items():
            reset_hooks(engine)
            candidates, _, _ = engine.run(spectra, target)
            keys = [
                str(key) for key, pid in zip(candidates.key, candidates.pid, strict=True)
                if pid < 0
            ]
            if keys != expected["generated_keys"][mode]:
                raise AssertionError(f"Direct/{mode} Engine.run mismatch: {molecule_id}")
            analogs[mode] = engine._last_an
            if mode == "g5" and keys != [row["key"] for row in engine._last_gen]:
                raise AssertionError("G5 hooked generation context differs from candidates")
        if analogs["g0"] != analogs["g4"] or analogs["g0"] != analogs["g5"]:
            raise AssertionError("CE adapter changed merged analog rank-feature input")
        checks.append(molecule_id)
    return checks


def summary(records: list[dict], crosschecked: list[str], hashes: dict[str, str],
            elapsed: float) -> dict:
    eligible = [row for row in records if row["selected_ce_views"]]
    ineligible = [row for row in records if not row["selected_ce_views"]]

    def counts(rows: list[dict]) -> dict:
        return {
            "queries": len(rows),
            "g4_generated_set_changed": sum(row["g4_generated_set_changed"] for row in rows),
            "g5_generated_set_changed": sum(row["g5_generated_set_changed"] for row in rows),
            "g4_all_candidate_set_changed": sum(row["g4_all_set_changed"] for row in rows),
            "g5_all_candidate_set_changed": sum(row["g5_all_set_changed"] for row in rows),
            "g4_generated_added": sum(len(row["g4_generated_added"]) for row in rows),
            "g4_generated_removed": sum(len(row["g4_generated_removed"]) for row in rows),
            "g5_generated_added": sum(len(row["g5_generated_added"]) for row in rows),
            "g5_generated_removed": sum(len(row["g5_generated_removed"]) for row in rows),
            "generated_total": {
                mode: sum(row["generated_count"][mode] for row in rows)
                for mode in ("g0", "g4", "g5")
            },
            "no_merged_analogs": sum(row["n_baseline_analogs"] == 0 for row in rows),
        }

    return {
        "scope": (
            "label-free frozen 0.477 Engine candidate-generation prefix; "
            "no ranker/Engine2/final LB"
        ),
        "all400": counts(records),
        "ce_metadata_eligible": counts(eligible),
        "ce_metadata_ineligible": counts(ineligible),
        "full_engine_crosschecked_molecules": crosschecked,
        "input_sha256": hashes,
        "elapsed_seconds": round(elapsed, 1),
    }


def main() -> None:
    started = time.monotonic()
    library, pool, train_fp, external = load()
    engine = Engine(
        library, pool, train_fp,
        cfg=EngineCfg(generate=True, use_frag=False, use_subformula=False),
        bank=None, verbose=False,
    )
    install_hooks(engine)
    groups = query_groups()
    records = []
    for index, (molecule_id, spectra, target) in enumerate(groups, 1):
        records.append(direct_group(engine, molecule_id, spectra, target))
        if index == 1 or index % 25 == 0:
            print(f"{index}/400 groups in {time.monotonic() - started:.0f}s", flush=True)
    checks = crosscheck(library, pool, train_fp, groups, records)
    hashes = {
        "test": sha(TEST),
        "reference_spec_meta": sha(external / "spec_meta.parquet"),
        "pool_meta": sha(WORK / "pool_meta.parquet"),
        "train_fp_sel": sha(WORK / "train_fp_sel.npy"),
        "ce_selector_g4": sha(LOCAL / "ce_diverse_generator.py"),
        "ce_union_g5": sha(LOCAL / "ce_union_generator.py"),
        "hook": sha(MODEL / "code/fe_v4/hooks_v1.py"),
        "engine": sha(MODEL / "code/casmi/engine.py"),
        "evaluator": sha(Path(__file__)),
    }
    report = summary(records, checks, hashes, time.monotonic() - started)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

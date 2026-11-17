"""Compare the frozen 0.477 analog generator's merged and separate CE views.

This is an engine-only candidate-coverage diagnostic, not the 0.477 full ranker
or a Kaggle submission. It reuses the saved 0.477 library and pool caches, and
it excludes each labelled query's own structure from reference analog search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectra_for_molecule(frame: pd.DataFrame) -> list[dict]:
    spectra = []
    for row in frame.itertuples():
        ce = row.collision_energy_ev
        values = np.asarray(ce, dtype=float) if ce is not None else np.empty(0)
        values = values[np.isfinite(values)]
        spectra.append({
            "mz": np.asarray(row.ms2_mzs, dtype=np.float64),
            "it": np.asarray(row.ms2_normalized_intensities, dtype=np.float64),
            "mode": 1 if row.ionization_mode == "positive" else -1,
            "adduct": row.adduct,
            "prec": float(row.precursor_mz),
            "ce": float(values.mean()) if len(values) else np.nan,
            "ce_n": int(len(values)) if len(values) else 1,
        })
    return spectra


def evaluate(
    *,
    code_dir: Path,
    library_dir: Path,
    pool_dir: Path,
    test_path: Path,
    labels_path: Path,
    output_dir: Path,
    limit: int,
) -> dict:
    sys.path.insert(0, str(code_dir.resolve()))
    from casmi import chem  # noqa: PLC0415
    from casmi.engine import Engine, EngineCfg  # noqa: PLC0415
    from casmi.library import Library  # noqa: PLC0415
    from casmi.pool import Pool  # noqa: PLC0415

    library = Library(str(library_dir), verbose=False)
    pool = Pool(str(pool_dir), load_frag=False, verbose=False)
    train_fp = np.load(pool_dir / "train_fp_sel.npy")
    engine = Engine(library, pool, train_fp, EngineCfg(generate=True), bank=None, verbose=False)
    if engine.cfg.analog_query != "merged":
        raise ValueError("Frozen 0.477 analog-query control changed")

    test = pd.read_parquet(test_path)
    labels = pd.read_csv(labels_path, dtype=str).set_index("molecule_id")
    keys_to_sid = {str(key): sid for sid, key in enumerate(library.struct_key)}
    records = []
    start = time.monotonic()
    for mid, frame in test.groupby("molecule_id", sort=False):
        if limit and len(records) >= limit:
            break
        truth = str(labels.loc[mid, "metric_key"])
        neutral = chem.neutral_mass(
            frame.precursor_mz.to_numpy(np.float64), frame.adduct.to_numpy()
        )
        neutral = neutral[np.isfinite(neutral)]
        if not len(neutral):
            records.append({"molecule_id": mid, "error": "no_neutral_mass"})
            continue
        target = float(np.median(neutral))
        spectra = spectra_for_molecule(frame)
        sid = keys_to_sid.get(truth, -1)
        pool_ids = pool.window(target, engine.cfg.ppm_win)
        if not len(pool_ids):
            pool_ids = pool.window(target, engine.cfg.ppm_fallback)
        pool_keys = {str(pool.key[i]) for i in pool_ids}
        modes = {}
        for mode in ("merged", "each"):
            engine.cfg.analog_query = mode
            analogs = engine.analogs(spectra, target, exclude_sid=sid)
            generated = engine.generate(analogs, target, pool_keys)
            generated_keys = {str(item["key"]) for item in generated}
            modes[mode] = {
                "analog_sid_top6": [int(item[0]) for item in analogs[:6]],
                "analog_score_top6": [float(item[1]) for item in analogs[:6]],
                "generated_keys": sorted(generated_keys),
                "generated_count": len(generated_keys),
                "truth_generated": truth in generated_keys,
            }
        merged = modes["merged"]
        each = modes["each"]
        record = {
            "molecule_id": str(mid),
            "truth_key": truth,
            "query_spectra": len(spectra),
            "query_ce_n": [int(s["ce_n"]) for s in spectra],
            "truth_in_reference_library": sid >= 0,
            "truth_in_pool": truth in pool_keys,
            "pool_candidates": len(pool_keys),
            "neutral_mass": target,
            "merged": merged,
            "each": each,
            "top6_analog_parent_changed": merged["analog_sid_top6"] != each["analog_sid_top6"],
            "generated_set_changed": merged["generated_keys"] != each["generated_keys"],
        }
        records.append(record)
        print(
            f"{len(records)}/{test.molecule_id.nunique()} {mid} "
            f"pool={record['truth_in_pool']} "
            f"generated={merged['truth_generated']}/{each['truth_generated']} "
            f"time={time.monotonic() - start:.1f}s",
            flush=True,
        )
    rows = [row for row in records if "error" not in row]
    summary = {
        "scope": "engine-only pre-ranker candidate generation; not Recall@25 or MRR",
        "comparison": "0.477 EngineCfg(analog_query='merged' vs 'each')",
        "n_molecules": len(rows),
        "n_input_spectra": int(sum(row["query_spectra"] for row in rows)),
        "n_truth_in_reference_library_excluded": sum(
            row["truth_in_reference_library"] for row in rows
        ),
        "n_truth_in_pool": sum(row["truth_in_pool"] for row in rows),
        "n_out_of_pool": sum(not row["truth_in_pool"] for row in rows),
        "pool_candidates_total": sum(row["pool_candidates"] for row in rows),
        "n_analog_top6_changed": sum(row["top6_analog_parent_changed"] for row in rows),
        "n_generated_set_changed": sum(row["generated_set_changed"] for row in rows),
        "merged_generated_candidates_total": sum(row["merged"]["generated_count"] for row in rows),
        "each_generated_candidates_total": sum(row["each"]["generated_count"] for row in rows),
        "merged_generated_truth": sum(row["merged"]["truth_generated"] for row in rows),
        "each_generated_truth": sum(row["each"]["truth_generated"] for row in rows),
        "merged_union_truth": sum(
            row["truth_in_pool"] or row["merged"]["truth_generated"] for row in rows
        ),
        "each_union_truth": sum(
            row["truth_in_pool"] or row["each"]["truth_generated"] for row in rows
        ),
        "elapsed_seconds": time.monotonic() - start,
        "inputs": {
            "test_sha256": sha256(test_path),
            "labels_sha256": sha256(labels_path),
            "library_spec_meta_sha256": sha256(library_dir / "spec_meta.parquet"),
            "pool_meta_sha256": sha256(pool_dir / "pool_meta.parquet"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per-molecule.jsonl").open("w") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=root
        / "data/derived/hypothesis1-assets/casmi26-attributed-v4g-reproduction-assets/"
        "casmi26-v4b-models/code",
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=root / "data/derived/hypothesis1-local/attempt1-kaggle/working/"
        "external_reference_42zwn464",
    )
    parser.add_argument("--pool-dir", type=Path, default=Path("/kaggle/working/work"))
    parser.add_argument(
        "--test",
        type=Path,
        default=root / "data/derived/candidate-rescue-corrected-v2/test.parquet",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=root / "data/derived/candidate-rescue-corrected-v2/labels-local-only.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/derived/ce-views-20261117/generator-external80",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    report = evaluate(
        code_dir=args.code_dir,
        library_dir=args.library_dir,
        pool_dir=args.pool_dir,
        test_path=args.test,
        labels_path=args.labels,
        output_dir=args.output,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

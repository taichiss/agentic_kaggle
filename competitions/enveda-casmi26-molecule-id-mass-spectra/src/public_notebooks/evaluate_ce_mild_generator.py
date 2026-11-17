"""Compare generator-only separate CE views and mild Mixup on external 80.

This is a pre-ranker analog/derivative diagnostic. It is not the full 0.477
pipeline, Recall@25, or Public LB. Query truth is excluded from the reference
library analog lookup, and labels are loaded only for this local diagnostic.
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
from ce_mild_mixup import augmented_views
from evaluate_ce_analog_generator import sha256, spectra_for_molecule


def evaluate(
    *, code_dir: Path, library_dir: Path, pool_dir: Path, test_path: Path,
    labels_path: Path, output_dir: Path, limit: int = 0,
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
    test = pd.read_parquet(test_path)
    labels = pd.read_csv(labels_path, dtype=str).set_index("molecule_id")
    if not test.molecule_id.isin(labels.index).all():
        raise ValueError("External panel has queries without local labels")
    sid_by_key = {str(key): sid for sid, key in enumerate(library.struct_key)}
    records = []
    started = time.monotonic()
    for mid, frame in test.groupby("molecule_id", sort=False):
        if limit and len(records) >= limit:
            break
        truth = str(labels.loc[mid, "metric_key"])
        neutral = chem.neutral_mass(
            frame.precursor_mz.to_numpy(np.float64), frame.adduct.to_numpy()
        )
        neutral = neutral[np.isfinite(neutral)]
        if not len(neutral):
            records.append({"molecule_id": str(mid), "error": "no_neutral_mass"})
            continue
        target = float(np.median(neutral))
        spectra = spectra_for_molecule(frame)
        for spectrum, instrument in zip(spectra, frame.instrument_type, strict=True):
            spectrum["instrument"] = str(instrument)
        views, mixed_count = augmented_views(
            spectra, include_mixup=True, max_views=engine.cfg.max_query_spectra
        )
        sid = sid_by_key.get(truth, -1)
        pool_ids = pool.window(target, engine.cfg.ppm_win)
        if not len(pool_ids):
            pool_ids = pool.window(target, engine.cfg.ppm_fallback)
        pool_keys = {str(pool.key[i]) for i in pool_ids}
        arms = {}
        for arm, mode, query in (
            ("merged", "merged", spectra),
            ("separate", "each", spectra),
            ("mild", "each", views),
        ):
            engine.cfg.analog_query = mode
            analogs = engine.analogs(query, target, exclude_sid=sid)
            generated = engine.generate(analogs, target, pool_keys)
            generated_keys = {str(item["key"]) for item in generated}
            arms[arm] = {
                "top6_parent_sids": [int(item[0]) for item in analogs[:6]],
                "generated_candidates": len(generated_keys),
                "truth_generated": truth in generated_keys,
                "truth_in_pool_or_generated": truth in pool_keys or truth in generated_keys,
                "generated_key_sha256": hashlib.sha256(
                    "\n".join(sorted(generated_keys)).encode()
                ).hexdigest(),
            }
        records.append({
            "molecule_id": str(mid),
            "truth_in_reference_excluded": sid >= 0,
            "truth_in_pool": truth in pool_keys,
            "measured_views": len(spectra),
            "mixed_views": mixed_count,
            "arms": arms,
            "separate_parent_changed": (
                arms["merged"]["top6_parent_sids"] != arms["separate"]["top6_parent_sids"]
            ),
            "mild_parent_changed_vs_separate": (
                arms["separate"]["top6_parent_sids"] != arms["mild"]["top6_parent_sids"]
            ),
        })
        print(f"{len(records)}/{test.molecule_id.nunique()} {mid} mixed={mixed_count}", flush=True)
    rows = [row for row in records if "error" not in row]
    summary = {
        "scope": "engine-only candidate generation; not full 0.477 ranking, Recall@25 or LB",
        "n_molecules": len(rows),
        "n_truth_in_reference_excluded": sum(row["truth_in_reference_excluded"] for row in rows),
        "n_truth_in_pool": sum(row["truth_in_pool"] for row in rows),
        "n_molecules_with_mild_mix": sum(row["mixed_views"] > 0 for row in rows),
        "mixed_views_total": sum(row["mixed_views"] for row in rows),
        "separate_parent_changed": sum(row["separate_parent_changed"] for row in rows),
        "mild_parent_changed_vs_separate": sum(
            row["mild_parent_changed_vs_separate"] for row in rows
        ),
        "arms": {
            arm: {
                "generated_candidates": sum(
                    row["arms"][arm]["generated_candidates"] for row in rows
                ),
                "truth_generated": sum(row["arms"][arm]["truth_generated"] for row in rows),
                "truth_in_pool_or_generated": sum(
                    row["arms"][arm]["truth_in_pool_or_generated"] for row in rows
                ),
            } for arm in ("merged", "separate", "mild")
        },
        "elapsed_seconds": time.monotonic() - started,
        "inputs": {
            "test_sha256": sha256(test_path),
            "labels_sha256": sha256(labels_path),
            "library_spec_meta_sha256": sha256(library_dir / "spec_meta.parquet"),
            "pool_meta_sha256": sha256(pool_dir / "pool_meta.parquet"),
            "adapter_sha256": sha256(Path(__file__).with_name("ce_mild_mixup.py")),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per-molecule.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-dir", type=Path,
        default=root / "data/derived/hypothesis1-assets/casmi26-attributed-v4g-reproduction-assets/"
        "casmi26-v4b-models/code",
    )
    parser.add_argument(
        "--library-dir", type=Path,
        default=root / "data/derived/hypothesis1-local/attempt1-kaggle/working/"
        "external_reference_42zwn464",
    )
    parser.add_argument("--pool-dir", type=Path, default=Path("/kaggle/working/work"))
    parser.add_argument(
        "--test", type=Path,
        default=root / "data/derived/candidate-rescue-corrected-v2/test.parquet",
    )
    parser.add_argument(
        "--labels", type=Path,
        default=root / "data/derived/candidate-rescue-corrected-v2/labels-local-only.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "data/derived/ce-views-20261117/generator-mild-external80",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    report = evaluate(
        code_dir=args.code_dir, library_dir=args.library_dir, pool_dir=args.pool_dir,
        test_path=args.test, labels_path=args.labels, output_dir=args.output, limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

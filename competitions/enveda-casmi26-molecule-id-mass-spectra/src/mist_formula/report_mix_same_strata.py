"""Stratify paired M1/A validation ranks by source, instrument, and mix eligibility."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from train_mix_same import Cache


def mode(values: np.ndarray) -> int:
    counts = Counter(int(value) for value in values)
    return min(counts, key=lambda x: (-counts[x], x))


def metrics(rows: list[dict]) -> dict:
    a = np.asarray([row["A_rank"] for row in rows])
    m1 = np.asarray([row["M1_rank"] for row in rows])
    if not len(rows):
        return {"molecules": 0}
    return {
        "molecules": len(rows),
        "A_mrr25": float(np.mean(1 / a)),
        "M1_mrr25": float(np.mean(1 / m1)),
        "delta_mrr25": float(np.mean(1 / m1 - 1 / a)),
        "A_top1": float(np.mean(a == 1)),
        "M1_top1": float(np.mean(m1 == 1)),
        "improved": int(np.sum(m1 < a)),
        "worsened": int(np.sum(m1 > a)),
        "unchanged": int(np.sum(m1 == a)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = {
        int(row["mol_idx"]): row
        for line in args.baseline_jsonl.read_text().splitlines()
        if (row := json.loads(line))
    }
    val = json.loads(args.validation_json.read_text())
    val = val.get("validation", val)
    treatment = dict(zip(val["query_mols"], val["ranks"], strict=True))
    panel = pd.read_parquet(args.panel)
    if set(baseline) != set(treatment) or set(panel.mol_idx) != set(treatment):
        raise ValueError("A/M1 panel molecule IDs differ")
    data = Cache(args.cache, args.precise_mzs, args.source_metadata)
    fp = np.load(args.cache / "fingerprints.npy", mmap_mode="r")
    val_groups = defaultdict(list)
    for row in np.flatnonzero(data.spec.split.eq("validation")):
        key = (int(data.mol_idx[row]), int(data.adduct[row]),
               int(data.ingest_id[row]), int(data.instrument_id[row]),
               int(data.meta[row, 3]))
        val_groups[key].append(int(row))

    def eligible(row: int) -> bool:
        if data.instrument_id[row] < 0:
            return False
        key = (int(data.mol_idx[row]), int(data.adduct[row]),
               int(data.ingest_id[row]), int(data.instrument_id[row]),
               int(data.meta[row, 3]))
        for other in val_groups[key]:
            if other == row or bool(data.meta[row, 2] > 0.5) != bool(data.meta[other, 2] > 0.5):
                continue
            if data.meta[row, 2] > 0.5 and abs(data.meta[row, 1] - data.meta[other, 1]) > 0.1:
                continue
            if abs(data.meta[row, 0] - data.meta[other, 0]) <= max(data.meta[row, 0] * 20e-6, 1e-7):
                return True
        return False

    records = []
    categories = json.loads((args.source_metadata / "categories.json").read_text())
    for item in panel.itertuples(index=False):
        mol = int(item.mol_idx)
        rows = np.asarray(item.query_cache_rows, dtype=np.int64)
        source_id = mode(data.ingest_id[rows])
        instrument_id = mode(data.instrument_id[rows])
        candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
        target = np.asarray(fp[mol], dtype=bool)
        rivals = np.asarray(fp[candidates[candidates != mol]], dtype=bool)
        similarity = (rivals & target).sum(axis=1) / np.maximum(
            (rivals | target).sum(axis=1), 1
        )
        records.append({
            "mol_idx": mol,
            "molecule_key": baseline[mol]["molecule_key"],
            "A_rank": int(baseline[mol]["A_rank"]),
            "M1_rank": int(treatment[mol]),
            "query_spectra": len(rows),
            "dominant_source": categories["ingest_lib"][source_id],
            "dominant_instrument": (
                categories["instrument_type"][instrument_id]
                if instrument_id >= 0 else "<missing>"
            ),
            "validation_measurement_mix_eligible": any(eligible(int(row)) for row in rows),
            "max_isomer_fp_similarity": float(similarity.max()),
        })
    strata = defaultdict(list)
    for row in records:
        for dimension, name in (
            ("source", row["dominant_source"]),
            ("instrument", row["dominant_instrument"]),
            ("validation_measurement_mix_eligible",
             str(row["validation_measurement_mix_eligible"])),
        ):
            strata[(dimension, name)].append(row)
        value = row["max_isomer_fp_similarity"]
        band = "<0.3" if value < 0.3 else "0.3-0.5" if value < 0.5 else (
            "0.5-0.7" if value < 0.7 else ">=0.7"
        )
        strata[("max_isomer_fp_similarity", band)].append(row)
    summary = {
        "overall": metrics(records),
        "strata": {
            dimension: {
                name: metrics(rows)
                for (kind, name), rows in sorted(strata.items()) if kind == dimension
            }
            for dimension in (
                "source", "instrument", "validation_measurement_mix_eligible",
                "max_isomer_fp_similarity",
            )
        },
        "eligibility_note": (
            "validation measurement has a same-molecule partner in validation under M1 rules; "
            "validation spectra were not used in training"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "strata.json").write_text(json.dumps(summary, indent=2))
    (args.output / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

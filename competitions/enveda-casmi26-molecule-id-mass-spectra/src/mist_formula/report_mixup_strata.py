"""Explain a mixup panel result by measurement source and isomer difficulty."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from mixup_pairs import PartnerPool
from train_mixup_weak import Cache


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"molecules": 0}
    a = np.asarray([row["A_rank"] for row in rows])
    b = np.asarray([row["M2_rank"] for row in rows])
    delta = 1 / b - 1 / a
    return {
        "molecules": len(rows),
        "A_mrr25": float(np.mean(1 / a)),
        "M2_mrr25": float(np.mean(1 / b)),
        "delta_mrr25": float(np.mean(delta)),
        "A_top1": float(np.mean(a == 1)),
        "M2_top1": float(np.mean(b == 1)),
        "improved": int(np.sum(b < a)),
        "worsened": int(np.sum(b > a)),
        "unchanged": int(np.sum(b == a)),
    }


def mode(values: np.ndarray) -> int:
    counts = Counter(int(value) for value in values)
    return min(counts, key=lambda x: (-counts[x], x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = {
        int(row["mol_idx"]): row
        for line in args.baseline_jsonl.read_text().splitlines()
        if (row := json.loads(line))
    }
    validation = json.loads(args.validation_json.read_text())
    validation = validation.get("validation", validation)
    treatment = dict(zip(validation["query_mols"], validation["ranks"], strict=True))
    panel = pd.read_parquet(args.panel)
    if set(baseline) != set(treatment) or set(panel.mol_idx) != set(treatment):
        raise ValueError("Fixed A/M2 panel molecule IDs differ")
    data = Cache(args.cache, args.precise_mzs)
    pool = PartnerPool(data, args.metadata_dir, mode="cross")
    formula_id = pd.factorize(data.mol.formula, sort=False)[0]
    group_lookup = {}
    for group, start in enumerate(pool.offsets[:-1]):
        row = int(pool.rows[start])
        key = (
            int(formula_id[data.mol_idx[row]]),
            int(data.adduct[row]),
            int(pool.ingest_id[row]),
            int(pool.instrument_id[row]),
            int(pool.ce_kind[row]),
            int(data.meta[row, 3]),
        )
        group_lookup[key] = group

    fp = np.load(args.cache / "fingerprints.npy", mmap_mode="r")
    records = []
    for item in panel.itertuples(index=False):
        mol = int(item.mol_idx)
        query_rows = np.asarray(item.query_cache_rows, np.int64)
        source_id = mode(pool.ingest_id[query_rows])
        instrument_id = mode(pool.instrument_id[query_rows])
        candidates = np.asarray(item.candidate_mol_indices, np.int64)
        target = np.asarray(fp[mol], dtype=bool)
        rivals = np.asarray(fp[candidates[candidates != mol]], dtype=bool)
        similarity = (rivals & target).sum(axis=1) / np.maximum(
            (rivals | target).sum(axis=1), 1
        )
        max_similarity = float(similarity.max())
        eligible = False
        for row in query_rows:
            key = (
                int(formula_id[mol]),
                int(data.adduct[row]),
                int(pool.ingest_id[row]),
                int(pool.instrument_id[row]),
                int(pool.ce_kind[row]),
                int(data.meta[row, 3]),
            )
            group = group_lookup.get(key)
            if group is None:
                continue
            candidate_rows = pool.rows[pool.offsets[group] : pool.offsets[group + 1]]
            ce_close = (
                np.abs(pool.ce_ev[candidate_rows] - pool.ce_ev[row]) <= pool.ce_tol
                if pool.ce_kind[row] == 1 else np.ones(len(candidate_rows), bool)
            )
            mz_close = (
                np.abs(data.meta[candidate_rows, 0] - data.meta[row, 0])
                / data.meta[row, 0] * 1e6 <= pool.precursor_ppm
            )
            if np.any(ce_close & mz_close):
                eligible = True
                break
        records.append({
            "mol_idx": mol,
            "molecule_key": baseline[mol]["molecule_key"],
            "A_rank": int(baseline[mol]["A_rank"]),
            "M2_rank": int(treatment[mol]),
            "query_spectra": len(query_rows),
            "dominant_source": (
                pool.categories["ingest_lib"][source_id] if source_id >= 0 else "<missing>"
            ),
            "dominant_instrument": (
                pool.categories["instrument_type"][instrument_id]
                if instrument_id >= 0 else "<missing>"
            ),
            "max_isomer_fp_similarity": max_similarity,
            "train_analog_partner_eligible": eligible,
        })
    strata = defaultdict(list)
    for row in records:
        strata[("source", row["dominant_source"])].append(row)
        strata[("instrument", row["dominant_instrument"])].append(row)
        strata[("train_analog_eligible", str(row["train_analog_partner_eligible"]))].append(row)
        value = row["max_isomer_fp_similarity"]
        if value < 0.3:
            band = "<0.3"
        elif value < 0.5:
            band = "0.3-0.5"
        elif value < 0.7:
            band = "0.5-0.7"
        else:
            band = ">=0.7"
        strata[("max_isomer_fp_similarity", band)].append(row)
    summary = {
        "overall": metrics(records),
        "strata": {
            dimension: {
                name: metrics(rows)
                for (kind, name), rows in sorted(strata.items()) if kind == dimension
            }
            for dimension in (
                "source", "instrument", "train_analog_eligible", "max_isomer_fp_similarity"
            )
        },
        "notes": {
            "dominant_measurement": "mode among each validation molecule's query spectra",
            "train_analog_eligible": (
                "whether any validation query spectrum would match a training molecule under "
                "strict M2 metadata and ppm criteria; validation spectra are never mixed "
                "in training"
            ),
            "fp_similarity": (
                "maximum 2048-bit Morgan Tanimoto against the other 24 same-formula candidates"
            ),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "strata.json").write_text(json.dumps(summary, indent=2))
    (args.output / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

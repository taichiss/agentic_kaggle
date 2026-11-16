"""Count exactly which training spectra can be mixed under M1 matching rules."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from train_mix_same import Cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = Cache(args.cache, args.precise_mzs, args.source_metadata)
    train_rows = np.flatnonzero(data.spec.split.eq("train"))
    eligible = np.zeros(len(data.spec), dtype=np.bool_)
    max_group_size = 0
    for rows in data.same_mol_adduct_source_instrument_polarity.values():
        n = len(rows)
        max_group_size = max(max_group_size, n)
        if n < 2:
            continue
        row_ids = np.asarray(rows, dtype=np.int64)
        mz = data.meta[row_ids, 0]
        ce = data.meta[row_ids, 1] * 100
        known = data.meta[row_ids, 2] > 0.5
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            mz_ok = np.abs(mz[start:stop, None] - mz[None, :]) <= np.maximum(
                mz[start:stop, None] * 20e-6, 1e-7
            )
            ce_ok = (
                (known[start:stop, None] == known[None, :]) &
                (~known[start:stop, None] | (np.abs(ce[start:stop, None] - ce[None, :]) <= 10))
            )
            pair = mz_ok & ce_ok
            pair[np.arange(stop-start), np.arange(start, stop)] = False
            eligible[row_ids[start:stop]] = pair.any(axis=1)
    total_by_source = Counter()
    eligible_by_source = Counter()
    for source_id, name in enumerate(data.ingest_categories):
        source_rows = train_rows[data.ingest_id[train_rows] == source_id]
        total_by_source[name] = len(source_rows)
        eligible_by_source[name] = int(eligible[source_rows].sum())
    eligible_mols = np.unique(data.mol_idx[train_rows[eligible[train_rows]]])
    result = {
        "train_spectra": len(train_rows),
        "eligible_spectra": int(eligible[train_rows].sum()),
        "train_molecules": len(data.train_mols),
        "eligible_molecules": len(eligible_mols),
        "max_matching_group_size": max_group_size,
        "total_spectra_by_source": total_by_source,
        "eligible_spectra_by_source": eligible_by_source,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

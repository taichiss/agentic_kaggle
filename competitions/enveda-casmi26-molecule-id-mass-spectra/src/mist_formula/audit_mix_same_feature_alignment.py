"""Check mixed peak IDs stay aligned between feature and formula paths."""

import argparse
import json
from pathlib import Path

import numpy as np
from train_mix_same import Cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    data = Cache(args.cache, args.precise_mzs, args.source_metadata)
    rng = np.random.default_rng(74147)
    train_rows = np.flatnonzero(data.spec.split.eq("train"))
    checked = different = 0
    for row in rng.choice(train_rows, size=args.samples, replace=True):
        row = int(row)
        other = data.partner(row, rng)
        if other < 0:
            continue
        checked += 1
        n1, n2 = int(data.length[row]), int(data.length[other])
        mz = np.concatenate((data.mz[row, :n1], data.mz[other, :n2])).astype(np.float64)
        alpha = rng.uniform(0.25, 0.75)
        intensity = np.concatenate((
            alpha * np.asarray(data.x[row, :n1, 1], dtype=np.float32) ** 2,
            (1 - alpha) * np.asarray(data.x[other, :n2, 1], dtype=np.float32) ** 2,
        )).astype(np.float64)
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
        group = np.cumsum(np.r_[True, np.diff(mz) > mz[1:] * 20e-6]) - 1
        total = np.bincount(group, weights=intensity)
        selected_feature = np.argsort(-total.astype(np.float32), kind="stable")[:96]
        selected_formula = np.argsort(-total, kind="stable")[:96]
        if not np.array_equal(selected_feature, selected_formula):
            different += 1
    print(json.dumps({"checked": checked, "different_peak_selection": different}))


if __name__ == "__main__":
    main()

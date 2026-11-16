"""Audit same-molecule mix partners against source and instrument metadata."""

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
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()
    data = Cache(args.cache, args.precise_mzs, args.sidecar)
    ingest = np.load(args.sidecar / "ingest_id.npy", mmap_mode="r")
    instrument = np.load(args.sidecar / "instrument_id.npy", mmap_mode="r")
    categories = json.loads((args.sidecar / "categories.json").read_text())
    rng = np.random.default_rng(431321)
    train_rows = np.flatnonzero(data.spec.split.eq("train"))
    sampled = rng.choice(train_rows, size=args.samples, replace=True)
    counts = Counter()
    source_counts = Counter()
    for row in sampled:
        row = int(row)
        other = data.partner(row, rng)
        if other < 0:
            counts["ineligible"] += 1
            continue
        counts["eligible"] += 1
        source = categories["ingest_lib"][ingest[row]]
        source_counts[source] += 1
        if ingest[row] != ingest[other]:
            counts["different_source"] += 1
        if instrument[row] != instrument[other]:
            counts["different_instrument"] += 1
        if data.meta[row, 3] != data.meta[other, 3]:
            counts["different_polarity"] += 1
        if (ingest[row] == ingest[other]
                and instrument[row] == instrument[other]
                and data.meta[row, 3] == data.meta[other, 3]):
            counts["both_match"] += 1
    report = {"samples": args.samples, "counts": counts,
              "source_counts_for_eligible": source_counts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()

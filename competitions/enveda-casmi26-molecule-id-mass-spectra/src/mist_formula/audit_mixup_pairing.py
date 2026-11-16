"""Measure how often the strict cross-molecule mixup policy finds a partner."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from mixup_pairs import PartnerPool
from train_mixup_weak import Cache


def run(cache: Path, precise_mzs: Path, metadata_dir: Path, output: Path, probes: int, seed: int):
    data = Cache(cache, precise_mzs)
    pool = PartnerPool(data, metadata_dir, mode="cross")
    rng = np.random.default_rng(seed)
    sampled_mols = rng.choice(data.train_mols, size=probes, replace=True)
    sampled_rows = np.fromiter(
        (rng.choice(data.train_spectra[int(m)]) for m in sampled_mols),
        dtype=np.int64,
        count=probes,
    )
    paired = 0
    breakdown = Counter()
    for row in sampled_rows:
        source_id = int(pool.ingest_id[row])
        source = (
            pool.categories["ingest_lib"][source_id] if source_id >= 0 else "<missing>"
        )
        breakdown[(source, "probes")] += 1
        partner = pool.sample_partner(int(row), rng)
        if partner is None:
            continue
        paired += 1
        breakdown[(source, "paired")] += 1
        assert data.spec.split.iloc[partner] == "train"
        assert data.mol_idx[row] != data.mol_idx[partner]
        assert data.mol.formula.iloc[data.mol_idx[row]] == data.mol.formula.iloc[
            data.mol_idx[partner]
        ]
        assert data.adduct[row] == data.adduct[partner]
        assert data.meta[row, 3] == data.meta[partner, 3]
        assert pool.ingest_id[row] == pool.ingest_id[partner]
        assert pool.instrument_id[row] == pool.instrument_id[partner]
        assert pool.ce_kind[row] == pool.ce_kind[partner]
        if pool.ce_kind[row] == 1:
            assert abs(float(pool.ce_ev[row]) - float(pool.ce_ev[partner])) <= pool.ce_tol
        assert (
            abs(float(data.meta[row, 0]) - float(data.meta[partner, 0]))
            / float(data.meta[row, 0])
            * 1e6
            <= pool.precursor_ppm
        )
    train_rows = np.asarray(data.spec.split.eq("train"))
    known_source = pool.ingest_id >= 0
    known_instrument = pool.instrument_id >= 0
    valid_ce = pool.ce_kind <= 1
    report = {
        "probes": probes,
        "seed": seed,
        "sampling": "uniform training molecule, then one of its accepted training spectra",
        "strict_partner_probes": paired,
        "strict_partner_coverage": paired / probes,
        "expected_effective_mix_fraction_at_25pct_attempt": 0.25 * paired / probes,
        "train_spectrum_rows": int(train_rows.sum()),
        "train_rows_known_source": int((train_rows & known_source).sum()),
        "train_rows_known_instrument": int((train_rows & known_instrument).sum()),
        "train_rows_valid_ce_or_both_missing": int((train_rows & valid_ce).sum()),
        "train_rows_indexed": pool.stats["indexed_rows"],
        "library_probe_coverage": {
            source: {
                "probes": breakdown[(source, "probes")],
                "paired": breakdown[(source, "paired")],
                "coverage": breakdown[(source, "paired")] / breakdown[(source, "probes")],
            }
            for source in sorted({source for source, _ in breakdown})
        },
        "pairing_policy": {
            "same_formula": True,
            "distinct_train_molecule": True,
            "same_library_instrument_adduct_polarity_ce_status": True,
            "ce_tolerance_ev": pool.ce_tol,
            "precursor_tolerance_ppm": pool.precursor_ppm,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    if args.probes <= 0:
        parser.error("--probes must be positive")
    run(**vars(args))


if __name__ == "__main__":
    main()

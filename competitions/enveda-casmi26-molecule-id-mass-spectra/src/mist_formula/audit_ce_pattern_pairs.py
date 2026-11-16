"""Replay 12k strict-M1 and exact multi-CE-pattern pairing, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ce_pattern_pairs import PatternCache
from train_mix_same_ce_strict import Cache as StrictCache


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_pair(data: PatternCache, row: int, partner: int) -> None:
    if data.spec.split.iloc[row] != "train" or data.spec.split.iloc[partner] != "train":
        raise AssertionError("Partnering leaked out of training split")
    if data.mol_idx[row] != data.mol_idx[partner]:
        raise AssertionError("Partnering crossed molecules")
    if data.adduct[row] != data.adduct[partner]:
        raise AssertionError("Partnering crossed adducts")
    if data.ingest_id[row] != data.ingest_id[partner]:
        raise AssertionError("Partnering crossed sources")
    if data.instrument_id[row] < 0 or data.instrument_id[row] != data.instrument_id[partner]:
        raise AssertionError("Partnering crossed or omitted instruments")
    if data.meta[row, 3] != data.meta[partner, 3]:
        raise AssertionError("Partnering crossed polarity")
    if data.ce_kind[row] != data.ce_kind[partner]:
        raise AssertionError("Partnering crossed CE kind")
    if data.ce_kind[row] == 2 and (
        data.pattern_id[row] < 0 or data.pattern_id[row] != data.pattern_id[partner]
    ):
        raise AssertionError("Partnering crossed multi-CE patterns")
    precursor = float(data.meta[row, 0])
    if abs(float(data.meta[partner, 0]) - precursor) > max(precursor * 20e-6, 1e-7):
        raise AssertionError("Partner precursor differs by >20 ppm")


def replay(args: argparse.Namespace) -> dict:
    strict = StrictCache(args.cache, args.precise_mzs, args.source_metadata)
    pattern = PatternCache(
        args.cache, args.precise_mzs, args.source_metadata, args.pattern_metadata
    )
    if not np.array_equal(strict.train_mols, pattern.train_mols):
        raise AssertionError("Molecule-disjoint split changed")
    rng = np.random.default_rng(args.seed)
    mix_strict = np.random.default_rng(args.seed + 300_007)
    mix_pattern = np.random.default_rng(args.seed + 300_007)
    strict_by_source, pattern_by_source = Counter(), Counter()
    attempted_by_source = Counter()
    by_kind, by_pattern, kind2_by_source = Counter(), Counter(), Counter()
    comparison = Counter()
    examples = []
    strict_attempted = strict_applied = attempted = applied = 0
    for step in range(1, args.steps + 1):
        ids = rng.choice(strict.train_mols, size=args.batch_size, replace=False)
        rows = np.asarray(
            [rng.choice(strict.train_spectra[int(mol)]) for mol in ids], dtype=np.int64
        )
        # The rank-loss negative draw consumes the same main RNG stream as training.
        for mol in ids:
            same = strict.same_formula[strict.mol.formula.iloc[mol]]
            if len(same) > 1:
                negative = int(rng.choice(same))
                while negative == mol:
                    negative = int(rng.choice(same))
            else:
                rng.choice(strict.train_mols)
        strict_mask = mix_strict.random(len(rows)) < args.mix_probability
        pattern_mask = mix_pattern.random(len(rows)) < args.mix_probability
        strict_attempted += int(strict_mask.sum())
        attempted += int(pattern_mask.sum())
        strict_pairs = np.full(len(rows), -1, dtype=np.int64)
        pattern_pairs = np.full(len(rows), -1, dtype=np.int64)
        for i in np.flatnonzero(strict_mask):
            row = int(rows[i])
            partner = strict.partner(row, mix_strict)
            if partner >= 0:
                mix_strict.uniform(args.mix_alpha_min, args.mix_alpha_max)
                strict_applied += 1
                strict_pairs[i] = partner
                strict_by_source[strict.ingest_categories[int(strict.ingest_id[row])]] += 1
        for i in np.flatnonzero(pattern_mask):
            row = int(rows[i])
            source = pattern.ingest_categories[int(pattern.ingest_id[row])]
            attempted_by_source[source] += 1
            partner = pattern.partner(row, mix_pattern)
            if partner < 0:
                continue
            assert_pair(pattern, row, partner)
            mix_pattern.uniform(args.mix_alpha_min, args.mix_alpha_max)
            applied += 1
            pattern_pairs[i] = partner
            pattern_by_source[source] += 1
            kind = int(pattern.ce_kind[row])
            by_kind[kind] += 1
            if kind == 2:
                by_pattern[int(pattern.pattern_id[row])] += 1
                kind2_by_source[source] += 1
                if len(examples) < 20:
                    examples.append({
                        "step": step, "batch_position": int(i),
                        "anchor_cache_row": row, "partner_cache_row": partner,
                        "anchor_raw_row": int(pattern.raw_row_index[row]),
                        "partner_raw_row": int(pattern.raw_row_index[partner]),
                        "molecule_index": int(pattern.mol_idx[row]),
                        "source": source, "pattern_id": int(pattern.pattern_id[row]),
                    })
        both = (strict_pairs >= 0) & (pattern_pairs >= 0)
        comparison["both_applied"] += int(both.sum())
        comparison["same_partner"] += int(np.sum(both & (strict_pairs == pattern_pairs)))
        comparison["strict_only"] += int(np.sum((strict_pairs >= 0) & (pattern_pairs < 0)))
        comparison["pattern_only"] += int(np.sum((strict_pairs < 0) & (pattern_pairs >= 0)))
        if step % args.log_every == 0 or step == args.steps:
            print(json.dumps({
                "step": step, "strict_applied": strict_applied,
                "pattern_applied": applied, "pattern_kind2": by_kind[2],
            }), flush=True)
    if strict_applied != args.expected_strict_applied:
        raise AssertionError(
            f"Strict replay changed: {strict_applied} != {args.expected_strict_applied}"
        )
    metadata = json.loads((args.pattern_metadata / "manifest.json").read_text())
    train = np.asarray(pattern.spec.split.eq("train"))
    known_instrument = pattern.instrument_id >= 0
    valid_kind2 = pattern.pattern_id >= 0
    report = {
        "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
        "mix_probability": args.mix_probability,
        "alpha_range": [args.mix_alpha_min, args.mix_alpha_max],
        "same_molecule_query_and_negative_stream": True,
        "separate_mix_rng_streams_after_first_policy_difference": True,
        "cache_spectra_sha256": sha(args.cache / "spectra.parquet"),
        "strict_trainer_sha256": sha(Path(__file__).with_name("train_mix_same_ce_strict.py")),
        "pattern_partner_source_sha256": sha(Path(__file__).with_name("ce_pattern_pairs.py")),
        "signature_builder_sha256": sha(
            Path(__file__).with_name("prepare_ce_pattern_signatures.py")
        ),
        "pattern_manifest_sha256": sha(args.pattern_metadata / "manifest.json"),
        "pattern_id_sha256": metadata["pattern_id_sha256"],
        "raw_row_index_sha256": metadata["raw_row_index_sha256"],
        "raw_train_sha256": metadata["raw_train_sha256"],
        "raw_ce_columns_sha256": metadata["raw_ce_columns_sha256"],
        "kind2_train_rows": int(np.sum(train & (pattern.ce_kind == 2))),
        "kind2_valid_train_rows": int(np.sum(train & valid_kind2)),
        "kind2_valid_known_instrument_train_rows": int(np.sum(
            train & valid_kind2 & known_instrument
        )),
        "kind2_valid_train_rows_in_groups_ge2": int(sum(
            len(rows) for rows in pattern.same_mol_pattern.values() if len(rows) >= 2
        )),
        "kind2_valid_train_groups_ge2": int(sum(
            len(rows) >= 2 for rows in pattern.same_mol_pattern.values()
        )),
        "strict": {
            "attempted": strict_attempted, "applied": strict_applied,
            "applied_by_source": dict(sorted(strict_by_source.items())),
        },
        "pattern": {
            "attempted": attempted, "applied": applied,
            "attempted_by_source": dict(sorted(attempted_by_source.items())),
            "applied_by_source": dict(sorted(pattern_by_source.items())),
            "applied_by_ce_kind": {str(k): v for k, v in sorted(by_kind.items())},
            "kind2_applied_by_source": dict(sorted(kind2_by_source.items())),
            "kind2_applied_by_pattern": {
                str(k): v for k, v in sorted(by_pattern.items())
            },
        },
        "positionwise_comparison": dict(comparison),
        "multi_ce_pair_examples": examples,
        "interpretation": (
            "This is an exact 12k RNG pairing replay, not model training. Both arms "
            "share the molecule/negative stream; after the first newly accepted "
            "multi-CE pair, separate mix RNG streams diverge. A future training run "
            "would retain strict M1 architecture, targets and shared peak merging."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--pattern-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mix-probability", type=float, default=0.25)
    parser.add_argument("--mix-alpha-min", type=float, default=0.25)
    parser.add_argument("--mix-alpha-max", type=float, default=0.75)
    parser.add_argument("--expected-strict-applied", type=int, default=59_258)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()
    result = replay(args)
    print(json.dumps({
        "strict": result["strict"], "pattern": result["pattern"],
    }), flush=True)


if __name__ == "__main__":
    main()

"""Replay old and CE-strict M1 pairing with the same fixed anchor stream."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from train_mix_same import Cache as OriginalCache
from train_mix_same_ce_strict import Cache as StrictCache


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(args: argparse.Namespace) -> dict:
    old = OriginalCache(args.cache, args.precise_mzs, args.source_metadata)
    strict = StrictCache(args.cache, args.precise_mzs, args.source_metadata)
    if not np.array_equal(old.train_mols, strict.train_mols):
        raise AssertionError("Old and strict cache splits differ")
    rng = np.random.default_rng(args.seed)
    mix_old = np.random.default_rng(args.seed + 300_007)
    mix_strict = np.random.default_rng(args.seed + 300_007)
    kinds = strict.ce_kind
    by_source_old = Counter()
    by_source_strict = Counter()
    kind_pairs_old = Counter()
    kind_pairs_strict = Counter()
    comparisons = Counter()
    samples = []
    old_attempted = old_applied = strict_attempted = strict_applied = 0
    for step in range(1, args.steps + 1):
        ids = rng.choice(old.train_mols, size=args.batch_size, replace=False)
        rows = np.asarray(
            [rng.choice(old.train_spectra[int(i)]) for i in ids], np.int64
        )
        # Consume the exact negative-selection RNG calls in train_mix_same.py.
        for mol_id in ids:
            same = old.same_formula[old.mol.formula.iloc[mol_id]]
            if len(same) > 1:
                other = int(rng.choice(same))
                while other == mol_id:
                    other = int(rng.choice(same))
            else:
                rng.choice(old.train_mols)
        old_mask = mix_old.random(len(rows)) < args.mix_probability
        strict_mask = mix_strict.random(len(rows)) < args.mix_probability
        old_attempted += int(old_mask.sum())
        strict_attempted += int(strict_mask.sum())
        old_pairs = np.full(len(rows), -1, dtype=np.int64)
        strict_pairs = np.full(len(rows), -1, dtype=np.int64)
        for i in np.flatnonzero(old_mask):
            anchor = int(rows[i])
            partner = old.partner(anchor, mix_old)
            if partner >= 0:
                mix_old.uniform(args.mix_alpha_min, args.mix_alpha_max)
                old_pairs[i] = partner
                old_applied += 1
                source = old.ingest_categories[int(old.ingest_id[anchor])]
                by_source_old[source] += 1
                kind_pair = (int(kinds[anchor]), int(kinds[partner]))
                kind_pairs_old[kind_pair] += 1
                if max(kind_pair) > 1 and len(samples) < 20:
                    samples.append({
                        "step": step, "batch_position": int(i),
                        "anchor_cache_row": anchor, "partner_cache_row": partner,
                        "molecule_index": int(old.mol_idx[anchor]),
                        "source": source, "ce_kind_anchor_partner": kind_pair,
                    })
        for i in np.flatnonzero(strict_mask):
            anchor = int(rows[i])
            partner = strict.partner(anchor, mix_strict)
            if partner >= 0:
                mix_strict.uniform(args.mix_alpha_min, args.mix_alpha_max)
                strict_pairs[i] = partner
                strict_applied += 1
                source = strict.ingest_categories[int(strict.ingest_id[anchor])]
                by_source_strict[source] += 1
                kind_pairs_strict[(int(kinds[anchor]), int(kinds[partner]))] += 1
        both = (old_pairs >= 0) & (strict_pairs >= 0)
        comparisons["both_applied"] += int(both.sum())
        comparisons["same_partner"] += int(np.sum(both & (old_pairs == strict_pairs)))
        comparisons["different_partner"] += int(np.sum(both & (old_pairs != strict_pairs)))
        comparisons["old_only_applied"] += int(np.sum((old_pairs >= 0) & (strict_pairs < 0)))
        comparisons["strict_only_applied"] += int(np.sum((old_pairs < 0) & (strict_pairs >= 0)))
        comparisons["both_attempted"] += int(np.sum(old_mask & strict_mask))
        if step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step": step, "old_applied": old_applied,
                              "strict_applied": strict_applied}), flush=True)
    old_manifest = json.loads(args.old_manifest.read_text())
    old_mix = old_manifest["mixup"]
    if old_attempted != old_mix["attempted"] or old_applied != old_mix["applied"]:
        raise AssertionError("Original M1 pairing replay does not match training manifest")
    if any(a > 1 or b > 1 or a != b for a, b in kind_pairs_strict):
        raise AssertionError("Strict M1 mixed a ce_kind=2 row or crossed CE kinds")
    train = np.asarray(old.spec.split.eq("train"))
    known = np.asarray(old.instrument_id >= 0)
    report = {
        "old_manifest_sha256": sha(args.old_manifest),
        "old_trainer_sha256": sha(Path(__file__).with_name("train_mix_same.py")),
        "strict_trainer_sha256": sha(Path(__file__).with_name("train_mix_same_ce_strict.py")),
        "source_metadata_ce_kind_sha256": sha(args.source_metadata / "ce_kind.npy"),
        "cache_spectra_sha256": sha(args.cache / "spectra.parquet"),
        "seed": args.seed, "steps": args.steps,
        "batch_size": args.batch_size,
        "mix_probability": args.mix_probability,
        "alpha_range": [args.mix_alpha_min, args.mix_alpha_max],
        "same_anchor_sampling_stream": True,
        "separate_mix_rng_streams_after_first_policy_difference": True,
        "kind0_1_known_ce_flag_parity_asserted_by_strict_cache": True,
        "train_spectra_by_ce_kind": {
            str(int(kind)): int(np.sum(train & (kinds == kind)))
            for kind in np.unique(kinds)
        },
        "known_instrument_train_spectra_by_ce_kind": {
            str(int(kind)): int(np.sum(train & known & (kinds == kind)))
            for kind in np.unique(kinds)
        },
        "old": {
            "attempted": old_attempted, "applied": old_applied,
            "applied_by_source": dict(by_source_old),
            "applied_ce_kind_pairs": {f"{a}:{b}": count for (a, b), count in kind_pairs_old.items()},
            "applied_with_ce_kind_gt_1": int(sum(
                count for (a, b), count in kind_pairs_old.items() if max(a, b) > 1
            )),
        },
        "strict": {
            "attempted": strict_attempted, "applied": strict_applied,
            "applied_by_source": dict(by_source_strict),
            "applied_ce_kind_pairs": {
                f"{a}:{b}": count for (a, b), count in kind_pairs_strict.items()
            },
            "applied_with_ce_kind_gt_1": 0,
            "applied_across_ce_kind_0_1": 0,
        },
        "positionwise_comparison": dict(comparisons),
        "excluded_original_pair_examples": samples,
        "interpretation": (
            "The original manifest is exactly reproduced. Anchor rows and negatives use "
            "the same RNG stream, but separate mix RNG streams diverge after rejected "
            "pairs; positionwise partner differences include downstream RNG effects. "
            "Old applied pairs containing ce_kind>1 are the directly invalid pairs."
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
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mix-probability", type=float, default=0.25)
    parser.add_argument("--mix-alpha-min", type=float, default=0.25)
    parser.add_argument("--mix-alpha-max", type=float, default=0.75)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()
    report = replay(args)
    print(json.dumps({"old": report["old"], "strict": report["strict"],
                      "comparison": report["positionwise_comparison"]}), flush=True)


if __name__ == "__main__":
    main()

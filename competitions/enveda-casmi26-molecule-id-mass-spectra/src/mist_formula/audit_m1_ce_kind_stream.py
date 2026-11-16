"""Replay exact M1/weak-M1 pairing RNG and count disallowed CE-kind-2 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from train_mix_same import Cache


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    m1_manifest = json.loads((args.m1 / "manifest.json").read_text())
    weak_manifest = json.loads((args.weak / "manifest.json").read_text())
    if (
        m1_manifest["seed"] != weak_manifest["seed"]
        or m1_manifest["steps"] != weak_manifest["steps"]
        or weak_manifest["steps"] != 12_000
        or m1_manifest["batch_size"] != weak_manifest["batch_size"]
        or weak_manifest["batch_size"] != 128
        or m1_manifest["mixup"]["probability"] != weak_manifest["mixup"]["probability"]
        or weak_manifest["mixup"]["probability"] != 0.25
    ):
        raise ValueError("M1 and weak-M1 RNG inputs differ")
    data = Cache(args.cache, args.precise_mzs, args.sidecar)
    ce_kind = np.load(args.sidecar / "ce_kind.npy", mmap_mode="r")
    ce_ev = np.load(args.sidecar / "ce_ev.npy", mmap_mode="r")
    if len(ce_kind) != len(data.spec):
        raise ValueError("CE-kind sidecar length differs from spectrum cache")
    seed = weak_manifest["seed"]
    rng = np.random.default_rng(seed)
    mix_rng = np.random.default_rng(seed + 300_007)
    attempted = applied = 0
    attempted_by_source = Counter()
    applied_by_source = Counter()
    kind_pairs = Counter()
    invalid_by_source = Counter()
    invalid_anchor_molecules = set()
    examples = []
    expected_m1 = {row["step"]: row for row in json.loads((args.m1 / "history.json").read_text())}
    expected_weak = {
        row["step"]: row for row in json.loads((args.weak / "history.json").read_text())
    }
    for step in range(1, 12_001):
        ids = rng.choice(data.train_mols, size=128, replace=False)
        rows = np.asarray([rng.choice(data.train_spectra[int(i)]) for i in ids], np.int64)
        for mol_id in ids:
            same = data.same_formula[data.mol.formula.iloc[mol_id]]
            if len(same) > 1:
                other = int(rng.choice(same))
                while other == mol_id:
                    other = int(rng.choice(same))
            else:
                other = int(rng.choice(data.train_mols))
        masks = mix_rng.random(len(rows)) < 0.25
        attempted += int(masks.sum())
        for i in np.flatnonzero(masks):
            anchor = int(rows[i])
            source = data.ingest_categories[int(data.ingest_id[anchor])]
            attempted_by_source[source] += 1
            partner = data.partner(anchor, mix_rng)
            if partner < 0:
                continue
            # This draw is required to preserve the next step's exact RNG state.
            mix_rng.uniform(0.85, 0.95)
            applied += 1
            applied_by_source[source] += 1
            kind_a, kind_b = int(ce_kind[anchor]), int(ce_kind[partner])
            kind_pairs[f"{kind_a}/{kind_b}"] += 1
            if kind_a == 2 or kind_b == 2:
                invalid_by_source[source] += 1
                invalid_anchor_molecules.add(int(data.mol_idx[anchor]))
                if len(examples) < 12:
                    examples.append({
                        "anchor_cache_row": anchor,
                        "partner_cache_row": partner,
                        "mol_idx": int(data.mol_idx[anchor]),
                        "source": source,
                        "kind_pair": f"{kind_a}/{kind_b}",
                        "anchor_ce_ev": (
                            None if not np.isfinite(ce_ev[anchor]) else float(ce_ev[anchor])
                        ),
                        "partner_ce_ev": (
                            None if not np.isfinite(ce_ev[partner]) else float(ce_ev[partner])
                        ),
                        "anchor_meta_ce_mean_ev": float(data.meta[anchor, 1] * 100),
                        "partner_meta_ce_mean_ev": float(data.meta[partner, 1] * 100),
                    })
        if step in expected_m1:
            for history in (expected_m1, expected_weak):
                event = history[step]
                if event["mix_attempted"] != attempted or event["mix_applied"] != applied:
                    raise AssertionError(f"Exact RNG stream disagrees with training at step {step}")
    for manifest in (m1_manifest, weak_manifest):
        mixed = manifest["mixup"]
        if (mixed["attempted"] != attempted or mixed["applied"] != applied
                or mixed["attempts_by_source"] != dict(attempted_by_source)
                or mixed["applied_by_source"] != dict(applied_by_source)):
            raise AssertionError("Replayed RNG counts differ from training manifest")
    invalid = sum(invalid_by_source.values())
    report = {
        "replay": "exact: all logged steps and final per-source counts match M1 and weak-M1",
        "train_steps": 12_000,
        "sampled_training_queries": 12_000 * 128,
        "mix_attempted": attempted,
        "mix_applied": applied,
        "invalid_kind2_applied": invalid,
        "invalid_kind2_fraction_of_applied": invalid / applied,
        "invalid_kind2_fraction_of_all_queries": invalid / (12_000 * 128),
        "invalid_kind2_unique_anchor_molecules": len(invalid_anchor_molecules),
        "kind_pair_counts": dict(sorted(kind_pairs.items())),
        "invalid_kind2_by_source": dict(sorted(invalid_by_source.items())),
        "attempted_by_source": dict(sorted(attempted_by_source.items())),
        "applied_by_source": dict(sorted(applied_by_source.items())),
        "examples": examples,
        "sidecar_ce_kind_sha256": sha(args.sidecar / "ce_kind.npy"),
        "m1_manifest_sha256": sha(args.m1 / "manifest.json"),
        "weak_manifest_sha256": sha(args.weak / "manifest.json"),
        "train_source_sha256": sha(Path(__file__).with_name("train_mix_same.py")),
        "policy": (
            "Strict M2/M3 policy excludes ce_kind=2 (multi/invalid); old M1 uses mean CE "
            "metadata and did not exclude these pairs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "mix_attempted": attempted, "mix_applied": applied,
        "invalid_kind2_applied": invalid,
        "invalid_kind2_fraction_of_applied": invalid / applied,
        "kind_pair_counts": report["kind_pair_counts"],
        "invalid_kind2_by_source": report["invalid_kind2_by_source"],
    }), flush=True)


if __name__ == "__main__":
    main()

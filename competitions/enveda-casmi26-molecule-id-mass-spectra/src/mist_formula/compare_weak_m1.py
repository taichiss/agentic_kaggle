"""Pair a weak same-molecule mixup run against A and strict M1 on fixed panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def score(rows: list[dict], seed: int) -> dict:
    if not rows:
        return {"molecules": 0}
    result = {"molecules": len(rows)}
    vectors = {
        name: np.asarray([
            0 if row[f"{name}_rank"] is None else 1 / row[f"{name}_rank"]
            for row in rows
        ])
        for name in ("A", "M1", "WEAK")
    }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(rows), size=(10_000, len(rows)))
    for name in ("A", "M1", "WEAK"):
        ranks = [row[f"{name}_rank"] for row in rows]
        result[name] = {
            "mrr25": float(vectors[name].mean()),
            "top1": sum(value == 1 for value in ranks) / len(rows),
            "present": sum(value is not None for value in ranks),
        }
    for reference in ("A", "M1"):
        delta = vectors["WEAK"] - vectors[reference]
        ci = np.quantile(delta[sampled].mean(axis=1), [0.025, 0.975])
        result[f"WEAK_vs_{reference}"] = {
            "delta_mrr25": float(delta.mean()),
            "paired_bootstrap_95pct": [float(value) for value in ci],
            "improved": int((delta > 0).sum()),
            "worsened": int((delta < 0).sum()),
            "unchanged": int((delta == 0).sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-1000", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--weak-manifest", type=Path, required=True)
    parser.add_argument("--m1-strata", type=Path, required=True)
    parser.add_argument("--parent-80", type=Path, required=True)
    parser.add_argument("--prior-parent-80", type=Path, required=True)
    parser.add_argument("--np26", type=Path, required=True)
    parser.add_argument("--prior-np26", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()

    baseline_rows = jsonl(args.baseline_1000)
    m1 = json.loads(args.m1_manifest.read_text())["validation"]
    weak = json.loads(args.weak_manifest.read_text())["validation"]
    a = {int(row["mol_idx"]): int(row["A_rank"]) for row in baseline_rows}
    m1_ranks = dict(zip(m1["query_mols"], m1["ranks"], strict=True))
    weak_ranks = dict(zip(weak["query_mols"], weak["ranks"], strict=True))
    if len(a) != 1000 or set(a) != set(m1_ranks) or set(a) != set(weak_ranks):
        raise ValueError("The fixed 1,000 molecule panel differs between A, M1, and WEAK")
    source_rows = {int(row["mol_idx"]): row for row in jsonl(args.m1_strata)}
    if set(a) != set(source_rows):
        raise ValueError("M1 source strata panel differs from the fixed 1,000")
    panel1000 = []
    for mol_id in m1["query_mols"]:
        previous = source_rows[mol_id]
        if previous["A_rank"] != a[mol_id] or previous["M1_rank"] != m1_ranks[mol_id]:
            raise ValueError(f"M1 strata ranks differ at molecule {mol_id}")
        panel1000.append({
            "mol_idx": mol_id,
            "A_rank": a[mol_id],
            "M1_rank": m1_ranks[mol_id],
            "WEAK_rank": weak_ranks[mol_id],
            "dominant_source": previous["dominant_source"],
            "dominant_instrument": previous["dominant_instrument"],
            "validation_measurement_mix_eligible": previous["validation_measurement_mix_eligible"],
            "max_isomer_fp_similarity": previous["max_isomer_fp_similarity"],
        })

    panel80 = jsonl(args.parent_80)
    if len(panel80) != 80 or not all(
        all(f"{name}_rank" in row for name in ("A", "M1", "WEAK")) for row in panel80
    ):
        raise ValueError("Parent proxy must hold 80 shared A/M1/WEAK rows")
    prior80 = {row["molecule_id"]: row for row in jsonl(args.prior_parent_80)}
    if set(prior80) != {row["molecule_id"] for row in panel80} or any(
        row["A_rank"] != prior80[row["molecule_id"]]["A_rank"]
        or row["M1_rank"] != prior80[row["molecule_id"]]["M1_rank"]
        for row in panel80
    ):
        raise ValueError("A/M1 ranks differ from the saved parent proxy")
    panel26 = jsonl(args.np26)
    if len(panel26) != 26 or not all(
        all(f"{name}_rank" in row for name in ("A", "M1", "WEAK")) for row in panel26
    ):
        raise ValueError("NP26 must hold 26 shared A/M1/WEAK rows")
    prior26 = {row["mol_idx"]: row for row in jsonl(args.prior_np26)}
    if set(prior26) != {row["mol_idx"] for row in panel26} or any(
        row["A_rank"] != prior26[row["mol_idx"]]["A_rank"]
        or row["M1_rank"] != prior26[row["mol_idx"]]["M1_rank"]
        for row in panel26
    ):
        raise ValueError("A/M1 ranks differ from the saved NP26 evaluation")
    strata = defaultdict(list)
    for row in panel1000:
        strata["source", row["dominant_source"]].append(row)
        strata["instrument", row["dominant_instrument"]].append(row)
        strata["source_instrument", (
            row["dominant_source"] + "/" + row["dominant_instrument"]
        )].append(row)
        strata["mix_eligible", str(row["validation_measurement_mix_eligible"])].append(row)
        similarity = row["max_isomer_fp_similarity"]
        band = (
            "<0.3" if similarity < 0.3 else
            "0.3-0.5" if similarity < 0.5 else
            "0.5-0.7" if similarity < 0.7 else ">=0.7"
        )
        strata["max_isomer_fp_similarity", band].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paired_1000 = args.output.parent / "per-molecule-1000.jsonl"
    paired_1000.write_text("".join(json.dumps(row) + "\n" for row in panel1000))
    report = {
        "input_sha256": {
            name: sha(getattr(args, name))
            for name in (
                "baseline_1000", "m1_manifest", "weak_manifest",
                "m1_strata", "parent_80", "prior_parent_80", "np26", "prior_np26",
            )
        },
        "bootstrap_seed": args.seed,
        "per_molecule_1000_sha256": sha(paired_1000),
        "panel1000": score(panel1000, args.seed),
        "parent80": score(panel80, args.seed),
        "np26_all": score(panel26, args.seed),
        "np26_multi_candidate": score(
            [row for row in panel26 if row["candidate_count"] > 1], args.seed
        ),
        "np26_candidate_bands": {
            "1": score([row for row in panel26 if row["candidate_count"] == 1], args.seed),
            "2-5": score([row for row in panel26 if 2 <= row["candidate_count"] <= 5], args.seed),
            "6-24": score([row for row in panel26 if 6 <= row["candidate_count"] <= 24], args.seed),
            "25": score([row for row in panel26 if row["candidate_count"] == 25], args.seed),
        },
        "panel1000_strata": {
            dimension: {
                category: score(rows, args.seed)
                for (kind, category), rows in sorted(strata.items()) if kind == dimension
            }
            for dimension in (
                "source", "instrument", "source_instrument", "mix_eligible",
                "max_isomer_fp_similarity",
            )
        },
        "limitations": (
            "WEAK was selected after observing A/M1/M2/M3 local scores, so this is an "
            "exploratory comparison. The 80 parent panel and 24 multi-candidate NP queries "
            "are small; NP26 was assembled post hoc. Do not compare absolute MRR across "
            "the three panels or interpret this as an unbiased test estimate."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "panel1000": report["panel1000"],
        "parent80": report["parent80"],
        "np26_multi_candidate": report["np26_multi_candidate"],
    }))


if __name__ == "__main__":
    main()

"""Paired fixed-1000 and saved-parent-80 evaluation of CE-strict M1."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(first: np.ndarray, second: np.ndarray, *, seed: int) -> dict:
    first_rr = np.where(first > 0, 1 / np.maximum(first, 1), 0)
    second_rr = np.where(second > 0, 1 / np.maximum(second, 1), 0)
    delta = first_rr - second_rr
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(delta), size=(5000, len(delta)))
    ci = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
    return {
        "queries": len(delta),
        "first_mrr25": float(first_rr.mean()),
        "second_mrr25": float(second_rr.mean()),
        "delta_mrr25_first_minus_second": float(delta.mean()),
        "paired_bootstrap_95pct": [float(ci[0]), float(ci[1])],
        "first_top1": float(np.mean(first == 1)),
        "second_top1": float(np.mean(second == 1)),
        "first_better": int(np.sum(first_rr > second_rr)),
        "second_better": int(np.sum(first_rr < second_rr)),
        "equal": int(np.sum(first_rr == second_rr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A-panel-ranks", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--parent-per-molecule", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    A_rows = [json.loads(line) for line in args.A_panel_ranks.open()]
    old = json.loads(args.old_manifest.read_text())
    new = json.loads(args.new_manifest.read_text())
    parent = json.loads(args.parent_report.read_text())
    parent_rows = [json.loads(line) for line in args.parent_per_molecule.open()]
    audit = json.loads(args.pair_audit.read_text())
    if (old["panel_sha256"] != new["panel_sha256"]
            or old["cache_spectra_sha256"] != new["cache_spectra_sha256"]
            or old["cache_molecules_sha256"] != new["cache_molecules_sha256"]
            or old["formula_csv_sha256"] != new["formula_csv_sha256"]
            or old["seed"] != new["seed"]
            or old["steps"] != 12000 or new["steps"] != 12000
            or old["checkpoint_sha256"] != sha(args.old_checkpoint)
            or new["checkpoint_sha256"] != sha(args.new_checkpoint)
            or old["mixup"]["applied"] != audit["old"]["applied"]
            or new["mixup"]["applied"] != audit["strict"]["applied"]
            or new["trainer_source_sha256"] != audit["strict_trainer_sha256"]
            or audit["strict"]["applied_with_ce_kind_gt_1"] != 0
            or audit["strict"]["applied_across_ce_kind_0_1"] != 0
            or not new["mixup"]["peak_merge_policy"].startswith(
                "shared mixup_pairs.merge_peaks"
            )):
        raise AssertionError("Training inputs, run counts or checkpoint SHA differ")
    old_val, new_val = old["validation"], new["validation"]
    if not (len(A_rows) == old_val["queries"] == new_val["queries"] == 1000):
        raise AssertionError("Fixed 1000 panel size differs")
    A_by_mol = {int(row["mol_idx"]): int(row["A_rank"]) for row in A_rows}
    old_by_mol = dict(zip(old_val["query_mols"], old_val["ranks"], strict=True))
    new_by_mol = dict(zip(new_val["query_mols"], new_val["ranks"], strict=True))
    if not (len(A_by_mol) == len(old_by_mol) == len(new_by_mol)
            and set(A_by_mol) == set(old_by_mol) == set(new_by_mol)):
        raise AssertionError("Fixed panel molecule identities differ")
    ids = [int(x) for x in new_val["query_mols"]]
    fixed = {
        "A": np.asarray([A_by_mol[x] for x in ids], dtype=int),
        "M1_original": np.asarray([old_by_mol[x] for x in ids], dtype=int),
        "M1_ce_strict": np.asarray([new_by_mol[x] for x in ids], dtype=int),
    }
    if (len(parent_rows) != parent["rows"] or parent["rows"] != 80
            or parent["checkpoints_sha256"]["M1_original"] != sha(args.old_checkpoint)
            or parent["checkpoints_sha256"]["M1_ce_strict"] != sha(args.new_checkpoint)):
        raise AssertionError("Saved parent proxy or checkpoint mismatch")
    parent_ranks = {
        name: np.asarray(
            [row[name + "_rank"] or 0 for row in parent_rows], dtype=int
        )
        for name in ("A", "M1_original", "M1_ce_strict")
    }
    report = {
        "checkpoint_sha256": {
            "M1_original": sha(args.old_checkpoint),
            "M1_ce_strict": sha(args.new_checkpoint),
        },
        "manifest_sha256": {
            "M1_original": sha(args.old_manifest),
            "M1_ce_strict": sha(args.new_manifest),
        },
        "A_panel_ranks_sha256": sha(args.A_panel_ranks),
        "fixed_panel_sha256": new["panel_sha256"],
        "parent_report_sha256": sha(args.parent_report),
        "parent_per_molecule_sha256": sha(args.parent_per_molecule),
        "pair_audit_sha256": sha(args.pair_audit),
        "fixed1000": {
            "strict_vs_A": summarize(fixed["M1_ce_strict"], fixed["A"], seed=args.seed),
            "strict_vs_original_M1": summarize(
                fixed["M1_ce_strict"], fixed["M1_original"], seed=args.seed
            ),
        },
        "saved_parent80": {
            "strict_vs_A": summarize(
                parent_ranks["M1_ce_strict"], parent_ranks["A"], seed=args.seed
            ),
            "strict_vs_original_M1": summarize(
                parent_ranks["M1_ce_strict"], parent_ranks["M1_original"], seed=args.seed
            ),
            "truth_present": parent["conditions"]["A"]["present"],
            "parent_control_sha256": parent["control_sha256"],
            "rank1_protected": True,
        },
        "note": "The fixed1000 panel scores the model alone; the selected parent80 proxy applies 0.477 parent RRF once at weight 0.25 with rank1 fixed. Neither is hidden Public LB.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fixed1000": report["fixed1000"],
                      "saved_parent80": report["saved_parent80"]}), flush=True)


if __name__ == "__main__":
    main()

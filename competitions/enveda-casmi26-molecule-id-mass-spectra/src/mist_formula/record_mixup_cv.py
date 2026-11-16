"""Idempotently append the six M1/M2/M3 local CV records to the CASMI ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/derived/mist-mixup-20261116"
LEDGER = ROOT / "strategy/cv-results.jsonl"
PANEL_SHA = "da870db0c4317b431ea482c416bd7ff4a5c0c8ccec6140e8641349c018bf20db"
FOLDERS = {"m1": "M1-strict", "m2": "M2-full-corrected", "m3": "M3-full"}
DESCRIPTIONS = {
    "m1": "Same-molecule distinct measured spectrum mixup; one unchanged binary FP target.",
    "m2": "Same-formula distinct training molecule weak mixup (anchor 85–95%); anchor FP target.",
    "m3": "Distinct same-formula molecules 50:50; soft FP, both positives weighted equally.",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_rows() -> list[dict]:
    proxy_path = BASE / "parent-proxy-all/report.json"
    proxy = json.loads(proxy_path.read_text())
    if proxy["rows"] != 80 or proxy["weight"] != 0.25:
        raise ValueError("Saved 0.477-family proxy changed")
    a_proxy = proxy["conditions"]["A"]["mrr25"]
    a_checkpoint = ROOT / "data/derived/mist-duration-20261107/dataset-12k/final.pt"
    a_proxy_report_path = (
        ROOT / "data/derived/mist-bcf-20261110/parent-proxy-evaluation/report.json"
    )
    a_proxy_report = json.loads(a_proxy_report_path.read_text())
    if (
        abs(a_proxy_report["conditions"]["A"]["mrr25"] - a_proxy) > 1e-12
        or proxy["checkpoints_sha256"]["A"] != sha(a_checkpoint)
        or a_proxy_report["parent_control_sha256"] != proxy["control_sha256"]
    ):
        raise ValueError("A baseline proxy and Mixup proxy do not agree")
    rows = [
        {
            "experiment_id": "EXP-0025-mist-12k",
            "split_id": "saved-0477-family-80-proxy",
            "metric": "MRR@25",
            "value": a_proxy,
            "comparison_kind": "approximate",
            "comparison_note": (
                "Frozen 0.479 A checkpoint on the same saved labeled 80-molecule 0.477-family "
                "parent candidate lists, formula-local rank fusion weight 0.25 and rank-1 "
                "protection. Selected proxy with only 54/80 reference structures in candidate "
                "lists; not hidden test/Public LB or generalization CV."
            ),
            "metrics_sha256": sha(a_proxy_report_path),
            "checkpoint_sha256": sha(a_checkpoint),
            "parent_control_sha256": proxy["control_sha256"],
            "n_groups": 80,
            "truth_present": proxy["conditions"]["A"]["present"],
            "top1": proxy["conditions"]["A"]["top1"],
            "missing_groups": proxy["conditions"]["A"]["missing_groups"],
        }
    ]
    for arm, folder in FOLDERS.items():
        upper = arm.upper()
        directory = BASE / folder
        manifest_path = directory / "manifest.json"
        paired_path = directory / "paired-vs-A.json"
        manifest = json.loads(manifest_path.read_text())
        paired = json.loads(paired_path.read_text())
        panel = manifest["validation"]
        checkpoint_sha = sha(directory / "final.pt")
        if (
            checkpoint_sha != manifest["checkpoint_sha256"]
            or checkpoint_sha != proxy["checkpoints_sha256"][upper]
            or manifest["panel_sha256"] != PANEL_SHA
            or manifest["seed"] != 20261107
            or manifest["steps"] != 12000
        ):
            raise ValueError(f"{upper} checkpoint, split, seed or update count differs")
        if (
            panel["queries"] != paired["rows"] != 1000
            or abs(panel["mrr25"] - paired["treatment_mrr25"]) > 1e-12
            or abs(panel["top1"] - paired["treatment_top1"]) > 1e-12
            or abs(paired["baseline_mrr25"] - 0.39125450535562456) > 1e-12
        ):
            raise ValueError(f"{upper} model-only panel and paired report disagree")
        mixup = manifest["mixup"]
        attempted = mixup.get("attempted", mixup.get("partner_pool", {}).get("attempted"))
        applied = mixup.get("applied", mixup.get("partner_pool", {}).get("paired"))
        if not attempted or applied is None or applied > attempted:
            raise ValueError(f"{upper} mixup coverage missing")
        exp_id = f"EXP-20261116-mixup-{arm}"
        common = {
            "experiment_id": exp_id,
            "metric": "MRR@25",
            "checkpoint_sha256": checkpoint_sha,
            "manifest_sha256": sha(manifest_path),
            "seed": manifest["seed"],
            "steps": manifest["steps"],
            "baseline_experiment_id": "EXP-20261110-bcf-a",
            "paired_bootstrap_unit": "molecule",
            "paired_bootstrap_seed": 20261116,
        }
        rows.append(
            {
                **common,
                "split_id": "same-formula-1000-v1",
                "value": panel["mrr25"],
                "comparison_kind": "synthetic_diagnostic",
                "comparison_note": (
                    f"{DESCRIPTIONS[arm]} Same frozen-A 2048-bit architecture, 12000 updates, "
                    "molecule-disjoint validation split and 25 same-formula candidates. "
                    "Model-only local diagnostic; not Public LB. One training seed; "
                    "bootstrap interval conditions on these trained weights."
                ),
                "metrics_sha256": sha(paired_path),
                "panel_sha256": PANEL_SHA,
                "n_groups": 1000,
                "top1": panel["top1"],
                "top25": panel["top25"],
                "baseline_value": paired["baseline_mrr25"],
                "delta_mrr25_vs_A": paired["delta_mrr25"],
                "paired_bootstrap_95pct_delta_vs_A": paired["paired_bootstrap_95pct"],
                "improved_vs_A": paired["improved"],
                "worsened_vs_A": paired["worsened"],
                "unchanged_vs_A": paired["unchanged"],
                "mix_attempted": attempted,
                "mix_applied": applied,
                "effective_mix_fraction": applied / (12000 * 128),
            }
        )
        condition = proxy["conditions"][upper]
        difference = condition["paired_vs_A"]
        if abs(condition["mrr25"] - a_proxy - difference["delta_mrr25"]) > 1e-12:
            raise ValueError(f"{upper} parent-proxy difference disagrees")
        rows.append(
            {
                **common,
                "split_id": "saved-0477-family-80-proxy",
                "value": condition["mrr25"],
                "comparison_kind": "approximate",
                "comparison_note": (
                    f"{DESCRIPTIONS[arm]} Same saved labeled 80-molecule 0.477-family parent "
                    "candidate lists, formula-local rank fusion weight 0.25 and rank-1 protection. "
                    "Selected proxy with only 54/80 reference structures in candidate lists; "
                    "not the hidden test set, actual Public LB or generalization CV. "
                    "Bootstrap interval conditions on these trained weights and this proxy."
                ),
                "metrics_sha256": sha(proxy_path),
                "parent_control_sha256": proxy["control_sha256"],
                "n_groups": 80,
                "truth_present": condition["present"],
                "top1": condition["top1"],
                "missing_groups": condition["missing_groups"],
                "baseline_value": a_proxy,
                "delta_mrr25_vs_A": difference["delta_mrr25"],
                "paired_bootstrap_95pct_delta_vs_A": difference["bootstrap_95pct"],
                "improved_vs_A": difference["improved"],
                "worsened_vs_A": difference["worsened"],
                "unchanged_vs_A": difference["unchanged"],
            }
        )
    return rows


def record() -> dict:
    incoming = make_rows()
    with LEDGER.open("r+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        existing_text = stream.read()
        existing = [json.loads(line) for line in existing_text.splitlines() if line.strip()]
        by_key = {
            (row.get("experiment_id"), row.get("split_id"), row.get("metric")): row
            for row in existing
        }
        if len(by_key) != len(existing):
            raise ValueError("Existing CV ledger has duplicate experiment/split/metric keys")
        missing = []
        skipped = []
        for row in incoming:
            key = (row["experiment_id"], row["split_id"], row["metric"])
            prior = by_key.get(key)
            if prior is None:
                missing.append(row)
            elif prior == row:
                skipped.append(key)
            else:
                raise ValueError(f"Conflicting CV ledger row already exists: {key}")
        if missing:
            stream.seek(0, 2)
            if existing_text and not existing_text.endswith("\n"):
                stream.write("\n")
            stream.write(
                "".join(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in missing
                )
            )
            stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
    final = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    if len(final) != len(existing) + len(missing):
        raise AssertionError("CV ledger length differs after append")
    return {
        "ledger": str(LEDGER),
        "appended": len(missing),
        "already_present": len(skipped),
        "total": len(final),
    }


if __name__ == "__main__":
    print(json.dumps(record()))

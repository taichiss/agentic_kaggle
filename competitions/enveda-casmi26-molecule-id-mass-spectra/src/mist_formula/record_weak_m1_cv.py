"""Idempotently record weak-M1 local CV and proxy results in the CASMI ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/derived/mist-mixup-20261116"
WEAK = BASE / "M1-weak-085095"
LEDGER = ROOT / "strategy/cv-results.jsonl"
EXPERIMENT = "EXP-20261117-mixup-m1-weak-085095"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return abs(left - right) < 1e-12


def make_rows() -> list[dict]:
    manifest_path = WEAK / "manifest.json"
    comparison_path = WEAK / "paired-comparison/report.json"
    proxy_path = WEAK / "parent-proxy/report.json"
    np26_path = WEAK / "np26-evaluation/report.json"
    np26_panel_path = BASE / "np26-panel/panel.parquet"
    overlap_path = BASE / "np26-evaluation/structure-overlap-audit.json"
    parity_path = WEAK / "frozen-inference-parity.json"
    contract_path = WEAK / "comparability-report.json"
    manifest = json.loads(manifest_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    proxy = json.loads(proxy_path.read_text())
    np26 = json.loads(np26_path.read_text())
    overlap = json.loads(overlap_path.read_text())
    parity = json.loads(parity_path.read_text())
    contract = json.loads(contract_path.read_text())
    checkpoint_sha = sha(WEAK / "final.pt")
    if checkpoint_sha != manifest["checkpoint_sha256"]:
        raise ValueError("Weak checkpoint differs from its manifest")
    if (
        comparison["input_sha256"]["weak_manifest"] != sha(manifest_path)
        or contract["checkpoints_sha256"]["WEAK"] != checkpoint_sha
        or not contract["same_non_alpha_inputs"]
        or not contract["same_logged_partner_stream"]
        or parity["parity"] != "exact"
        or parity["queries"] != 1000
        or manifest["seed"] != 20261107
        or manifest["steps"] != 12000
        or manifest["mixup"]["alpha_range"] != [0.85, 0.95]
        or manifest["mixup"]["probability"] != 0.25
    ):
        raise ValueError("Weak M1 training parity or contract differs")
    fixed = comparison["panel1000"]
    parent = comparison["parent80"]
    natural = comparison["np26_multi_candidate"]
    if (
        manifest["validation"]["queries"] != fixed["molecules"] != 1000
        or not close(manifest["validation"]["mrr25"], fixed["WEAK"]["mrr25"])
        or not close(proxy["conditions"]["WEAK"]["mrr25"], parent["WEAK"]["mrr25"])
        or not close(np26["multi_candidate_only"]["WEAK"]["mrr25"],
                     natural["WEAK"]["mrr25"])
        or np26["query_molecules"] != 26
        or natural["molecules"] != 24
        or np26["single_candidate_molecules"] != 2
        or np26["panel_sha256"] != sha(np26_panel_path)
        or overlap["overlap_counts"]["any_structure_key"] != 0
        or proxy["weight"] != 0.25
        or proxy["conditions"]["WEAK"]["missing_groups"] != 0
    ):
        raise ValueError("Comparison panels or overlap audit differ")
    common = {
        "experiment_id": EXPERIMENT,
        "metric": "MRR@25",
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": sha(manifest_path),
        "trainer_sha256": contract["same_trainer_source_sha256"],
        "seed": manifest["seed"],
        "steps": manifest["steps"],
        "mix_probability": manifest["mixup"]["probability"],
        "mix_alpha_range": manifest["mixup"]["alpha_range"],
        "mix_attempted": manifest["mixup"]["attempted"],
        "mix_applied": manifest["mixup"]["applied"],
        "baseline_experiment_id": "EXP-20261110-bcf-a",
        "paired_bootstrap_unit": "molecule",
        "paired_bootstrap_seed": comparison["bootstrap_seed"],
        "metrics_sha256": sha(comparison_path),
    }

    def paired(result: dict) -> dict:
        versus_a = result["WEAK_vs_A"]
        versus_m1 = result["WEAK_vs_M1"]
        return {
            "value": result["WEAK"]["mrr25"],
            "n_groups": result["molecules"],
            "top1": result["WEAK"]["top1"],
            "baseline_value": result["A"]["mrr25"],
            "strict_m1_value": result["M1"]["mrr25"],
            "delta_mrr25_vs_A": versus_a["delta_mrr25"],
            "paired_bootstrap_95pct_delta_vs_A": versus_a["paired_bootstrap_95pct"],
            "improved_vs_A": versus_a["improved"],
            "worsened_vs_A": versus_a["worsened"],
            "unchanged_vs_A": versus_a["unchanged"],
            "delta_mrr25_vs_M1": versus_m1["delta_mrr25"],
            "paired_bootstrap_95pct_delta_vs_M1": versus_m1["paired_bootstrap_95pct"],
        }

    return [
        {
            **common,
            **paired(fixed),
            "split_id": "same-formula-1000-v1",
            "comparison_kind": "synthetic_diagnostic",
            "comparison_note": (
                "Exploratory weak same-molecule spectrum mixup, anchor coefficient 0.85–0.95. "
                "Same M1 partner stream, A architecture, molecule-disjoint split and 25 "
                "same-formula candidates. Model-only, one training seed; the coefficient "
                "was chosen after seeing earlier local scores. Not Public LB; the molecule "
                "bootstrap holds trained weights fixed."
            ),
            "panel_sha256": manifest["panel_sha256"],
            "top25": manifest["validation"]["top25"],
        },
        {
            **common,
            **paired(parent),
            "split_id": "saved-0477-family-80-proxy",
            "comparison_kind": "approximate",
            "comparison_note": (
                "Exploratory weak same-molecule spectrum mixup on the saved labeled "
                "80-molecule 0.477-family parent. Same candidate sets, formula-local fusion "
                "weight 0.25 and rank-1 protection. Only 54/80 truths are in candidate "
                "lists. Selected proxy, not hidden test, Public LB or generalization CV."
            ),
            "parent_control_sha256": proxy["control_sha256"],
            "truth_present": parent["WEAK"]["present"],
            "missing_groups": proxy["conditions"]["WEAK"]["missing_groups"],
            "proxy_report_sha256": sha(proxy_path),
        },
        {
            **common,
            **paired(natural),
            "split_id": "np-examples-timstof-26-variable-candidates-v1",
            "comparison_kind": "synthetic_diagnostic",
            "comparison_note": (
                "Post hoc enveda-np-examples timsTOF auxiliary panel: 26 molecule-disjoint "
                "queries, 117 measured spectra; evaluate 24 multi-candidate molecules "
                "after excluding two forced-rank-1 singletons. Candidate counts vary "
                "from 1 to 25 using same-formula cache structures, not hidden competition "
                "candidates. Raw InChIKey14 and tautomer-canonical structure train "
                "overlap 0/26. Small sample and post hoc selection; no comparison of "
                "absolute MRR with the 1000/80 panels or Public LB."
            ),
            "panel_sha256": np26["panel_sha256"],
            "query_molecules_total": 26,
            "query_spectra_total": 117,
            "single_candidate_excluded": 2,
            "candidate_count_range": [1, 25],
            "structure_overlap_audit_sha256": sha(overlap_path),
            "np26_report_sha256": sha(np26_path),
        },
    ]


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
        new = []
        already = []
        for row in incoming:
            key = (row["experiment_id"], row["split_id"], row["metric"])
            prior = by_key.get(key)
            if prior is None:
                new.append(row)
            elif prior == row:
                already.append(key)
            else:
                raise ValueError(f"Conflicting CV ledger entry: {key}")
        if new:
            stream.seek(0, 2)
            if existing_text and not existing_text.endswith("\n"):
                stream.write("\n")
            stream.write("".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in new
            ))
            stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
    final = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    if len(final) != len(existing) + len(new):
        raise AssertionError("CV ledger length differs after append")
    return {"ledger": str(LEDGER), "appended": len(new),
            "already_present": len(already), "total": len(final)}


if __name__ == "__main__":
    print(json.dumps(record()))

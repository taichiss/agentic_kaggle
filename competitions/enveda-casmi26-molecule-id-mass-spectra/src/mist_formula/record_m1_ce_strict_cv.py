"""Idempotently record CE-strict M1's three local evaluations with provenance."""

import fcntl
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/derived/mist-mixup-20261116/M1-ce-strict-sharedmerge"
LEDGER = ROOT / "strategy/cv-results.jsonl"
EXPERIMENT = "EXP-20261117-mixup-m1-ce-strict"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_rows() -> list[dict]:
    manifest_path = BASE / "manifest.json"
    checkpoint_path = BASE / "final.pt"
    audit_path = BASE / "pair-audit.json"
    comparison_path = BASE / "comparison-report.json"
    parent_path = BASE / "parent-proxy/report.json"
    natural_path = BASE / "natural-source-eval/report.json"
    natural_rows_path = BASE / "natural-source-eval/per-molecule.jsonl"
    old_path = ROOT / "data/derived/mist-mixup-20261116/M1-strict/final.pt"
    A_path = ROOT / "data/derived/mist-duration-20261107/dataset-12k/final.pt"
    source_metadata_path = ROOT / "data/derived/mist-mixup-20261116/source-metadata/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    parent = json.loads(parent_path.read_text())
    natural = json.loads(natural_path.read_text())
    checkpoint_sha = sha(checkpoint_path)
    if (manifest["checkpoint_sha256"] != checkpoint_sha
            or manifest["trainer_source_sha256"] != audit["strict_trainer_sha256"]
            or manifest["mixup"]["attempted"] != audit["strict"]["attempted"]
            or manifest["mixup"]["applied"] != audit["strict"]["applied"]
            or audit["old"]["attempted"] != 383036
            or audit["old"]["applied"] != 190452
            or audit["strict"]["applied_with_ce_kind_gt_1"] != 0
            or audit["strict"]["applied_across_ce_kind_0_1"] != 0
            or comparison["checkpoint_sha256"]["M1_ce_strict"] != checkpoint_sha
            or comparison["checkpoint_sha256"]["M1_original"] != sha(old_path)
            or parent["checkpoints_sha256"]["M1_ce_strict"] != checkpoint_sha
            or parent["checkpoints_sha256"]["M1_original"] != sha(old_path)
            or parent["checkpoints_sha256"]["A"] != sha(A_path)
            or natural["checkpoint_sha256"] != checkpoint_sha
            or natural["query_training_structure_overlap"] != 0
            or natural["parent_RRF_applied"] is not False):
        raise AssertionError("CE-strict input, replay or checkpoint audit mismatch")
    fixed = comparison["fixed1000"]["strict_vs_A"]
    proxy = comparison["saved_parent80"]["strict_vs_A"]
    primary = natural["metrics"]["multi_candidate_primary"]
    if (fixed["queries"] != 1000 or proxy["queries"] != 80
            or comparison["saved_parent80"]["truth_present"] != 54
            or primary["molecule_source_pairs"] != 5425
            or primary["unique_structure_keys"] != 4108
            or natural["metrics"]["single_candidate_trivial"]["molecule_source_pairs"] != 1687
            or natural["metrics"]["source_x_timsTOF_only"]["enveda-np-examples"][
                "molecule_source_pairs"
            ] != 24):
        raise AssertionError("CE-strict local panel membership differs")
    common = {
        "experiment_id": EXPERIMENT,
        "metric": "MRR@25",
        "checkpoint_sha256": checkpoint_sha,
        "manifest_sha256": sha(manifest_path),
        "trainer_sha256": manifest["trainer_source_sha256"],
        "shared_peak_merge_source_sha256": manifest["shared_peak_merge_source_sha256"],
        "pair_audit_sha256": sha(audit_path),
        "source_metadata_manifest_sha256": sha(source_metadata_path),
        "raw_train_sha256": manifest["raw_train_sha256"],
        "cache_spectra_sha256": manifest["cache_spectra_sha256"],
        "cache_molecules_sha256": manifest["cache_molecules_sha256"],
        "precise_mzs_sha256": manifest["precise_mzs_sha256"],
        "formula_csv_sha256": manifest["formula_csv_sha256"],
        "ingest_id_sha256": manifest["ingest_id_sha256"],
        "instrument_id_sha256": manifest["instrument_id_sha256"],
        "ce_kind_sha256": manifest["ce_kind_sha256"],
        "ce_ev_sha256": manifest["ce_ev_sha256"],
        "A_checkpoint_sha256": sha(A_path),
        "M1_original_checkpoint_sha256": sha(old_path),
        "seed": manifest["seed"], "steps": manifest["steps"],
        "mix_attempted": audit["strict"]["attempted"],
        "mix_applied": audit["strict"]["applied"],
        "baseline_experiment_id": "EXP-0025-mist-12k",
        "comparison_note_common": (
            "Combined repair versus original M1: exclude ce_kind=2, pair same CE kind, "
            "compare kind1 sidecar ce_ev within 10 eV, and use shared strongest-m/z "
            "20ppm peak merge. Model architecture, seed, split and 12000 updates are fixed. "
            "The local difference cannot be attributed to CE filtering alone."
        ),
    }
    return [
        {
            **common,
            "split_id": "same-formula-1000-v1",
            "comparison_kind": "synthetic_diagnostic",
            "comparison_note": "Molecule-disjoint 1000-query fixed 25-candidate same-formula panel; model-only, not parent reranking or Public LB.",
            "metrics_sha256": sha(comparison_path),
            "evaluation_report_sha256": sha(comparison_path),
            "panel_sha256": manifest["panel_sha256"],
            "A_panel_ranks_sha256": comparison["A_panel_ranks_sha256"],
            "value": fixed["first_mrr25"],
            "top1": fixed["first_top1"],
            "n_groups": fixed["queries"],
            "baseline_value": fixed["second_mrr25"],
            "delta_mrr25_vs_A": fixed["delta_mrr25_first_minus_second"],
            "paired_bootstrap_95pct_delta_vs_A": fixed["paired_bootstrap_95pct"],
            "paired_bootstrap_unit": "molecule",
            "improved_vs_A": fixed["first_better"],
            "worsened_vs_A": fixed["second_better"],
            "unchanged_vs_A": fixed["equal"],
        },
        {
            **common,
            "split_id": "saved-0477-family-80-proxy",
            "comparison_kind": "approximate",
            "comparison_note": "Selected saved 0.477 parent proxy: same candidate lists, formula-local RRF weight 0.25 once, rank1 fixed; truth in candidates for only 54/80. Not hidden test or Public LB.",
            "metrics_sha256": sha(comparison_path),
            "evaluation_report_sha256": sha(parent_path),
            "paired_comparison_report_sha256": sha(comparison_path),
            "per_molecule_sha256": sha(BASE / "parent-proxy/per-molecule.jsonl"),
            "parent_control_sha256": parent["control_sha256"],
            "parent_spectra_sha256": parent["spectra_sha256"],
            "parent_labels_sha256": parent["labels_sha256"],
            "frozen_inference_source_sha256": parent["model_source_sha256"],
            "value": proxy["first_mrr25"],
            "top1": proxy["first_top1"],
            "n_groups": proxy["queries"],
            "truth_present": 54,
            "baseline_value": proxy["second_mrr25"],
            "delta_mrr25_vs_A": proxy["delta_mrr25_first_minus_second"],
            "paired_bootstrap_95pct_delta_vs_A": proxy["paired_bootstrap_95pct"],
            "paired_bootstrap_unit": "molecule",
            "improved_vs_A": proxy["first_better"],
            "worsened_vs_A": proxy["second_better"],
            "unchanged_vs_A": proxy["equal"],
        },
        {
            **common,
            "split_id": "natural-sources-5425-variable-candidates-posthoc-v1",
            "comparison_kind": "synthetic_diagnostic",
            "comparison_note": "Post-hoc GNPS/RIKEN/MassBank/enveda-np-examples panel: 5425 molecule-source pairs, 4108 structure keys, 2-25 same-formula candidates. Excludes 1687 forced-rank1 singletons. No parent rank exists; model-only. Structure-key bootstrap keeps source repeats together. Query/train structure overlap 0. Only 24 multi-candidate timsTOF pairs. Not comparable by absolute MRR to fixed25, parent80 or Public LB.",
            "metrics_sha256": sha(natural_path),
            "evaluation_report_sha256": sha(natural_path),
            "per_molecule_sha256": sha(natural_rows_path),
            "panel_sha256": natural["panel_sha256"],
            "prior_ranks_sha256": natural["prior_ranks_sha256"],
            "prior_report_sha256": natural["prior_report_sha256"],
            "frozen_inference_source_sha256": natural["frozen_model_sha256"],
            "value": primary["STRICT_mrr25"],
            "top1": primary["STRICT_top1"],
            "n_groups": 5425,
            "n_unique_structures": 4108,
            "single_candidate_excluded": 1687,
            "baseline_value": primary["STRICT_vs_A"]["second_mrr25"],
            "delta_mrr25_vs_A": primary["STRICT_vs_A"]["delta_mrr25_first_minus_second"],
            "paired_bootstrap_95pct_delta_vs_A": primary["STRICT_vs_A"]["delta_cluster_bootstrap_95pct"],
            "paired_bootstrap_unit": "structure_key",
            "paired_bootstrap_seed": natural["bootstrap"]["seed"],
            "improved_vs_A": primary["STRICT_vs_A"]["first_better"],
            "worsened_vs_A": primary["STRICT_vs_A"]["second_better"],
            "unchanged_vs_A": primary["STRICT_vs_A"]["same_rank"],
            "timsTOF_multi_candidate_queries": 24,
        },
    ]


def main() -> None:
    entries = make_rows()
    written = []
    with LEDGER.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        previous = stream.read()
        existing = {
            (row["experiment_id"], row.get("split_id"), row.get("metric")): row
            for line in previous.splitlines()
            if (row := json.loads(line))
        }
        stream.seek(0, 2)
        for entry in entries:
            key = (entry["experiment_id"], entry["split_id"], entry["metric"])
            if key in existing:
                if existing[key] != entry:
                    raise ValueError(f"conflicting CV result exists for {key}")
                continue
            if previous and not previous.endswith("\n"):
                stream.write("\n")
                previous += "\n"
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written.append(entry["split_id"])
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    receipt = {
        "experiment_id": EXPERIMENT,
        "written": written,
        "entries": len(entries),
        "cv_results_sha256": sha(LEDGER),
    }
    (BASE / "cv-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

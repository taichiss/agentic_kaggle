"""Register the three fixed A75:M1 25 local diagnostics without duplicate rows."""

import fcntl
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/derived/mist-mixup-20261116"
EXPERIMENT_ID = "EXP-20261117-mixup-a75m1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_rows() -> list[dict]:
    grid_path = BASE / "adaptive-grid-local.json"
    natural_path = BASE / "natural-source-full/m4-evaluation/report.json"
    per_molecule_path = BASE / "natural-source-full/m4-evaluation/per-molecule.jsonl"
    A_path = ROOT / "data/derived/mist-formula-20261107/steps-18000/step-12000.pt"
    A_manifest_path = ROOT / "data/derived/mist-formula-20261107/steps-18000/manifest.json"
    M1_path = BASE / "M1-strict/final.pt"
    M1_manifest_path = BASE / "M1-strict/manifest.json"
    evaluator_path = ROOT / "src/mist_formula/evaluate_natural_source_m4.py"
    blend_source_path = ROOT / "src/mist_formula/explore_mixup_adaptive.py"
    manifest_path = BASE / "M4-a75m1/manifest.json"
    grid = json.loads(grid_path.read_text())
    natural = json.loads(natural_path.read_text())
    A_manifest = json.loads(A_manifest_path.read_text())
    M1_manifest = json.loads(M1_manifest_path.read_text())
    A_sha, M1_sha = sha(A_path), sha(M1_path)
    if (A_manifest["milestones"]["12000"]["checkpoint_sha256"] != A_sha
            or M1_manifest["checkpoint_sha256"] != M1_sha
            or natural["A_checkpoint_sha256"] != A_sha
            or natural["M1_checkpoint_sha256"] != M1_sha
            or natural["query_training_structure_overlap"] != 0
            or natural["score_rank_parity"] != "A and M1 reproduced every saved natural-panel rank"
            or natural["parent_RRF_applied_to_natural_panel"] is not False):
        raise AssertionError("M4 checkpoint, split, or natural-panel audit mismatch")
    if (A_manifest["panel_sha256"] != M1_manifest["panel_sha256"]
            or grid["panel_queries"] != 1000
            or grid["proxy_queries"] != 80
            or grid["proxy_truth_present"] != 54):
        raise AssertionError("M4 saved panels or training split changed")
    blend = grid["score_blends"]["A75+M125"]
    primary = natural["metrics"]["multi_candidate_primary"]
    source_tims = natural["metrics"]["source_x_timsTOF_only"]
    if (natural["scored_molecule_source_pairs"] != 7112
            or primary["molecule_source_pairs"] != 5425
            or primary["unique_structure_keys"] != 4108
            or natural["metrics"]["single_candidate_trivial"]["molecule_source_pairs"] != 1687
            or source_tims["enveda-np-examples"]["molecule_source_pairs"] != 24):
        raise AssertionError("M4 natural-source panel membership changed")
    if (abs(blend["panel"]["mrr25"] - grid["baseline_A_mrr25_panel"]
            - blend["panel"]["delta_mrr25_vs_A"]) > 1e-12
            or abs(blend["proxy"]["mrr25"] - grid["baseline_A_mrr25_proxy"]
                   - blend["proxy"]["delta_mrr25_vs_A"]) > 1e-12
            or abs(primary["M4_mrr25"] - primary["M4_vs_A"]["second_mrr25"]
                   - primary["M4_vs_A"]["delta_mrr25_first_minus_second"]) > 1e-12):
        raise AssertionError("M4 paired MRR arithmetic mismatch")
    manifest = {
        "condition": EXPERIMENT_ID,
        "score_recipe": "z-score A and M1 candidate soft-Tanimoto scores separately within each full-formula group; 0.75*A + 0.25*M1",
        "candidate_fp": "same fixed 2048-bit Morgan FP candidates for A and M1",
        "same_training_split": A_manifest["split"] == M1_manifest["split"],
        "A_checkpoint_sha256": A_sha,
        "A_training_manifest_sha256": sha(A_manifest_path),
        "M1_checkpoint_sha256": M1_sha,
        "M1_training_manifest_sha256": sha(M1_manifest_path),
        "blend_source_sha256": sha(blend_source_path),
        "natural_evaluator_sha256": sha(evaluator_path),
        "same_formula_1000_and_parent80_evaluation_report_sha256": sha(grid_path),
        "natural_5425_evaluation_report_sha256": sha(natural_path),
        "natural_per_molecule_sha256": sha(per_molecule_path),
        "same_formula_1000_panel_scores_sha256": grid["evidence_sha256"]["panel_scores"],
        "saved_parent80_scores_sha256": grid["evidence_sha256"]["proxy_scores"],
        "natural_panel_sha256": natural["panel_sha256"],
        "natural_parent_RRF_applied": False,
        "saved_parent80_parent_RRF_once_weight": 0.25,
        "note": "The A75:M1 ratio was selected after earlier local experiments. These are local diagnostics, not Public LB scores.",
    }
    if not manifest["same_training_split"]:
        raise AssertionError("A and M1 training split differs")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    common = {
        "experiment_id": EXPERIMENT_ID,
        "metric": "MRR@25",
        "manifest_sha256": sha(manifest_path),
        "A_checkpoint_sha256": A_sha,
        "M1_checkpoint_sha256": M1_sha,
        "A_training_manifest_sha256": sha(A_manifest_path),
        "M1_training_manifest_sha256": sha(M1_manifest_path),
        "blend_weight_A": 0.75,
        "blend_weight_M1": 0.25,
        "baseline_experiment_id": "EXP-0025-mist-12k",
    }
    panel = blend["panel"]
    proxy = blend["proxy"]
    natural_vs_A = primary["M4_vs_A"]
    return [
        {
            **common,
            "split_id": "same-formula-1000-v1",
            "comparison_kind": "synthetic_diagnostic",
            "comparison_note": "Molecule-disjoint fixed 1000-query, 25-candidate same-formula panel. A/M1 within-formula z-score fusion, model-only; no parent RRF. A75:M1 ratio was examined after earlier results. Molecule bootstrap holds checkpoints fixed; not Public LB.",
            "metrics_sha256": sha(grid_path),
            "evaluation_report_sha256": sha(grid_path),
            "panel_scores_sha256": grid["evidence_sha256"]["panel_scores"],
            "panel_sha256": A_manifest["panel_sha256"],
            "value": panel["mrr25"],
            "top1": panel["top1"],
            "n_groups": 1000,
            "baseline_value": grid["baseline_A_mrr25_panel"],
            "delta_mrr25_vs_A": panel["delta_mrr25_vs_A"],
            "paired_bootstrap_95pct_delta_vs_A": panel["molecule_bootstrap_95pct_delta"],
            "paired_bootstrap_unit": "molecule",
            "improved_vs_A": panel["improved"],
            "worsened_vs_A": panel["worsened"],
            "unchanged_vs_A": panel["unchanged"],
        },
        {
            **common,
            "split_id": "saved-0477-family-80-proxy",
            "comparison_kind": "approximate",
            "comparison_note": "Saved selected 80-query 0.477-family parent lists. A/M1 z-score fusion within full-formula groups, parent RRF once at 0.25, rank 1 protected. Only 54/80 truths are in candidate lists; not hidden test, Public LB, or generalization CV.",
            "metrics_sha256": sha(grid_path),
            "evaluation_report_sha256": sha(grid_path),
            "proxy_scores_sha256": grid["evidence_sha256"]["proxy_scores"],
            "parent_control_sha256": grid["evidence_sha256"]["control"],
            "value": proxy["mrr25"],
            "top1": proxy["top1"],
            "n_groups": 80,
            "truth_present": 54,
            "baseline_value": grid["baseline_A_mrr25_proxy"],
            "delta_mrr25_vs_A": proxy["delta_mrr25_vs_A"],
            "paired_bootstrap_95pct_delta_vs_A": proxy["molecule_bootstrap_95pct_delta"],
            "paired_bootstrap_unit": "molecule",
            "improved_vs_A": proxy["improved"],
            "worsened_vs_A": proxy["worsened"],
            "unchanged_vs_A": proxy["unchanged"],
        },
        {
            **common,
            "split_id": "natural-sources-5425-variable-candidates-posthoc-v1",
            "comparison_kind": "synthetic_diagnostic",
            "comparison_note": "Post-hoc four-natural-source panel: 5425 molecule-source pairs (4108 distinct structures) with 2-25 same-formula candidates; 1687 forced-rank-1 singletons excluded. Model-only fusion because no parent ranks exist for these queries; random candidate order is not a parent rank. Structure-key bootstrap keeps repeated source measurements together. Query/true-label training-structure overlap 0; training structures occur only as decoys. Only 24 multi-candidate timsTOF pairs. Do not compare absolute MRR to fixed-25, saved parent80, or Public LB.",
            "metrics_sha256": sha(natural_path),
            "evaluation_report_sha256": sha(natural_path),
            "per_molecule_sha256": sha(per_molecule_path),
            "panel_sha256": natural["panel_sha256"],
            "value": primary["M4_mrr25"],
            "top1": primary["M4_top1"],
            "n_groups": 5425,
            "n_unique_structures": 4108,
            "single_candidate_excluded": 1687,
            "baseline_value": natural_vs_A["second_mrr25"],
            "delta_mrr25_vs_A": natural_vs_A["delta_mrr25_first_minus_second"],
            "paired_bootstrap_95pct_delta_vs_A": natural_vs_A["delta_cluster_bootstrap_95pct"],
            "paired_bootstrap_unit": "structure_key",
            "paired_bootstrap_seed": natural["bootstrap"]["seed"],
            "improved_vs_A": natural_vs_A["first_better"],
            "worsened_vs_A": natural_vs_A["second_better"],
            "unchanged_vs_A": natural_vs_A["same_rank"],
            "timsTOF_multi_candidate_queries": 24,
        },
    ]


def main() -> None:
    entries = make_rows()
    ledger = ROOT / "strategy/cv-results.jsonl"
    written = []
    with ledger.open("a+", encoding="utf-8") as stream:
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
                    raise ValueError(f"conflicting CV result exists: {key}")
                continue
            if previous and not previous.endswith("\n"):
                stream.write("\n")
                previous += "\n"
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written.append(entry["split_id"])
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    receipt = {
        "experiment_id": EXPERIMENT_ID,
        "written": written,
        "entries": len(entries),
        "cv_results_sha256": sha(ledger),
    }
    output = BASE / "M4-a75m1/cv-receipt.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

"""G2 + fixed MIST A retains its pinned parent and audits its output."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/mist_formula/build_separate_ce_overlay.py"
spec = importlib.util.spec_from_file_location("build_separate_ce_overlay", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def actual_args(tmp_path: Path) -> argparse.Namespace:
    parent_dir = ROOT / "data/derived/ce-views-20261117/kernels/separate"
    return argparse.Namespace(
        parent_notebook=parent_dir / "ce-generator-separate.ipynb",
        parent_build_receipt=parent_dir / "build-receipt.json",
        parent_metadata=parent_dir / "kernel-metadata.json",
        inference_module=ROOT / "src/mist_formula/frozen/a_0479_model.py",
        checkpoint=ROOT / "data/derived/mist-duration-20261107/dataset-12k/final.pt",
        dataset=module.A_DATASET,
        kernel="owner/ce-separate-mist-a-test",
        model_name="mist-a-ce-separate",
        output=tmp_path,
        experiment_id="EXP-G2-A-TEST",
    )


def test_build_retains_g2_parent_and_fixed_a_settings(tmp_path):
    args = actual_args(tmp_path)
    if not args.parent_notebook.is_file() or not args.checkpoint.is_file():
        pytest.skip("Local G2 parent or fixed A checkpoint is unavailable")
    receipt = module.build(args)
    output = tmp_path / "trained-model-overlay.ipynb"
    notebook = json.loads(output.read_text())
    parent = json.loads(args.parent_notebook.read_text())
    assert notebook["cells"][: len(parent["cells"])] == parent["cells"]
    assert len(notebook["cells"]) == len(parent["cells"]) + 3
    overlay = "".join(notebook["cells"][len(parent["cells"]) + 1]["source"])
    audit = "".join(notebook["cells"][-1]["source"])
    assert "_weight = 0.25" in overlay
    assert "_protected = 1" in overlay
    assert "_within_formula = True" in overlay
    assert "sorted(_a) == sorted(_b)" in audit
    assert "_a[0] == _b[0]" in audit
    assert "g2_generator_audit_sha256" in audit
    assert receipt["control_sha256"] == module.G2_NOTEBOOK_SHA256
    assert receipt["checkpoint_sha256"] == module.A_CHECKPOINT_SHA256
    assert receipt["module_sha256"] == module.INFERENCE_SHA256
    assert receipt["weight"] == module.FUSION_WEIGHT
    assert receipt["parent_adapter_sha256"] == module.G2_ADAPTER_SHA256
    assert receipt["candidate_set_sha256"] == "computed from the parent output at runtime"
    assert receipt["final_notebook_sha256"] == module.sha(output)
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"][-1] == module.A_DATASET
    assert metadata["is_private"]


def test_bad_dataset_and_parent_receipt_are_rejected(tmp_path):
    args = actual_args(tmp_path)
    if not args.parent_notebook.is_file():
        pytest.skip("Local G2 parent is unavailable")
    args.dataset = "owner/other-checkpoint"
    with pytest.raises(ValueError, match="Fixed A dataset"):
        module.build(args)
    args.dataset = module.A_DATASET
    bad_receipt = json.loads(args.parent_build_receipt.read_text())
    bad_receipt["mode"] = "mild"
    bad_path = tmp_path / "bad-parent-receipt.json"
    bad_path.write_text(json.dumps(bad_receipt))
    args.parent_build_receipt = bad_path
    with pytest.raises(ValueError, match="G2 parent receipt mismatch: mode"):
        module.build(args)


def test_runtime_audit_binds_parent_candidates_and_g2_log(tmp_path):
    pandas = pytest.importorskip("pandas")
    work, comp = tmp_path / "work", tmp_path / "comp"
    work.mkdir()
    comp.mkdir()
    pandas.DataFrame({"molecule_id": ["m1"], "smiles": ["C;CC;CCC"]}).to_csv(
        work / "model-control.csv", index=False
    )
    pandas.DataFrame({"molecule_id": ["m1"], "smiles": ["C;CCC;CC"]}).to_csv(
        work / "submission.csv", index=False
    )
    pandas.DataFrame({"molecule_id": ["m1"], "smiles": [""]}).to_csv(
        comp / "sample_submission.csv", index=False
    )
    (work / "model-overlay-report.json").write_text(json.dumps({
        "checkpoint_sha256": module.A_CHECKPOINT_SHA256,
        "inference_sha256": module.INFERENCE_SHA256,
        "weight": 0.25,
        "protected_top_n": 1,
        "within_formula": True,
        "errors": [],
        "top1_changed": 0,
        "attempted_rows": 1,
        "changed_rows": 1,
        "submission_sha256": module.sha(work / "submission.csv"),
    }))
    (work / "execution_receipt.json").write_text(json.dumps({
        "source_sha256": module.G2_FROZEN_SHA256,
        "processed_molecules": 1,
        "is_full_run": True,
    }))
    (work / "ce-generator-audit.json").write_text(json.dumps({
        "mode": "separate",
        "adapter_source_sha256": module.G2_ADAPTER_SHA256,
        "generator_calls": 1,
        "mixed_views_total": 0,
        "queries_with_mixed_views": 0,
        "changed_generator_parent_queries": 1,
        "generated_candidates_total": 2,
    }))
    source = module.audit_source(
        model_name="model", experiment_id="EXP-G2-A-TEST",
        parent_sha=module.G2_NOTEBOOK_SHA256,
        frozen_sha=module.G2_FROZEN_SHA256,
        checkpoint_sha=module.A_CHECKPOINT_SHA256,
        weight=module.FUSION_WEIGHT,
    ) + module.G2_AUDIT.replace("@@G2_ADAPTER_SHA@@", repr(module.G2_ADAPTER_SHA256))
    exec(source.replace("/kaggle/working", str(work)), {"COMP": str(comp)})
    audit = json.loads((work / "model-dynamic-audit.json").read_text())
    assert audit["ce_parent_output_sha256"] == module.sha(work / "model-control.csv")
    assert audit["g2_generator_audit_sha256"] == module.sha(work / "ce-generator-audit.json")
    assert audit["g2_changed_parent_queries"] == 1
    assert audit["top1_changed"] == 0

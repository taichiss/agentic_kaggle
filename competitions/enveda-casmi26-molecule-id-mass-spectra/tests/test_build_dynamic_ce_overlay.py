"""A CE-view parent can change candidates before the fixed MIST reranker runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/mist_formula/build_dynamic_ce_overlay.py"
spec = importlib.util.spec_from_file_location("build_dynamic_ce_overlay", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_actual_ce_parent_keeps_all_cells_and_attaches_dynamic_audit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    parent_dir = root / "data/derived/ce-views-20261117/kernel-g1"
    parent = parent_dir / "ce-analog-each.ipynb"
    checkpoint = root / "data/derived/mist-duration-20261107/dataset-12k/final.pt"
    if not parent.is_file() or not checkpoint.is_file():
        pytest.skip("Local CE parent and fixed A checkpoint are unavailable")
    args = __import__("argparse").Namespace(
        parent_notebook=parent,
        parent_build_receipt=parent_dir / "build-receipt.json",
        parent_metadata=parent_dir / "kernel-metadata.json",
        parent_notebook_sha256=module.sha(parent),
        inference_module=root / "src/mist_formula/frozen/a_0479_model.py",
        checkpoint=checkpoint,
        dataset="owner/fixed-a-checkpoint",
        kernel="owner/ce-each-mist-a-test",
        model_name="ce-mist-a-test",
        output=tmp_path,
        weight=0.25,
        experiment_id="EXP-CE-TEST",
    )
    receipt = module.build(args)
    notebook = json.loads((tmp_path / "trained-model-overlay.ipynb").read_text())
    original = json.loads(parent.read_text())
    assert notebook["cells"][: len(original["cells"])] == original["cells"]
    assert len(notebook["cells"]) == len(original["cells"]) + 3
    assert (
        receipt["parent_frozen_sha256"]
        == json.loads(args.parent_build_receipt.read_text())["variant_frozen_sha256"]
    )
    assert receipt["checkpoint_sha256"] == module.A_CHECKPOINT_SHA256
    assert receipt["candidate_set_sha256"] == "computed from the parent output at runtime"
    assert receipt["final_notebook_sha256"] == module.sha(tmp_path / "trained-model-overlay.ipynb")
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"][-1] == "owner/fixed-a-checkpoint"
    assert metadata["is_private"]


def test_runtime_audit_hashes_new_candidate_set_without_control_hash(tmp_path):
    pandas = pytest.importorskip("pandas")
    work, comp = tmp_path / "work", tmp_path / "comp"
    work.mkdir()
    comp.mkdir()
    base = pandas.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["C;CC;CCC", "O"]})
    final = pandas.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["C;CCC;CC", "O"]})
    sample = pandas.DataFrame({"molecule_id": ["m1", "m2"], "smiles": ["", ""]})
    base.to_csv(work / "model-control.csv", index=False)
    final.to_csv(work / "submission.csv", index=False)
    sample.to_csv(comp / "sample_submission.csv", index=False)
    frozen_sha = "f" * 64
    report = {
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
    }
    (work / "model-overlay-report.json").write_text(json.dumps(report))
    (work / "execution_receipt.json").write_text(
        json.dumps({"source_sha256": frozen_sha, "processed_molecules": 2})
    )
    source = module.audit_source(
        model_name="model",
        experiment_id="EXP-CE-TEST",
        parent_sha="p" * 64,
        frozen_sha=frozen_sha,
        checkpoint_sha=module.A_CHECKPOINT_SHA256,
        weight=0.25,
    ).replace("/kaggle/working", str(work))
    exec(source, {"COMP": str(comp)})
    audit = json.loads((work / "model-dynamic-audit.json").read_text())
    members = "m1\tC;CC;CCC\nm2\tO"
    assert audit["dynamic_candidate_set_sha256"] == hashlib.sha256(members.encode()).hexdigest()
    assert audit["ce_parent_output_sha256"] == module.sha(work / "model-control.csv")
    assert audit["top1_changed"] == 0

    final.loc[0, "smiles"] = "C;CC;CO"
    final.to_csv(work / "submission.csv", index=False)
    with pytest.raises(AssertionError, match="candidate set"):
        exec(source, {"COMP": str(comp)})

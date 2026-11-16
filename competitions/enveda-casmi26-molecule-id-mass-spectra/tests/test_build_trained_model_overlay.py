"""Checks for the pinned trained-model submission notebook builder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "public_notebooks"
    / "build_trained_model_overlay.py"
)


def test_build_is_append_only_and_pins_checkpoint(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("overlay_builder", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parent = {"cells": [{"cell_type": "code", "source": ["print('parent')\n"]}]}
    control = tmp_path / "control.ipynb"
    control.write_text(json.dumps(parent), encoding="utf-8")
    metadata = tmp_path / "kernel-metadata.json"
    metadata.write_text(json.dumps({"dataset_sources": ["owner/existing"]}), encoding="utf-8")
    inference = tmp_path / "inference.py"
    inference.write_text("def score_candidates(*args):\n    return [0.0]\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "result"
    args = argparse.Namespace(
        control_notebook=control,
        control_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
        control_metadata=metadata,
        module=inference,
        checkpoint=checkpoint,
        dataset="owner/new-model",
        kernel="owner/new-kernel",
        model_name="testmodel",
        output=output,
        weight=0.25,
        protect_top_n=1,
        within_formula=False,
    )
    receipt = module.build(args)
    result = json.loads((output / "trained-model-overlay.ipynb").read_text())
    result_metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert result["cells"][:1] == parent["cells"]
    assert len(result["cells"]) == 3
    assert receipt["checkpoint_sha256"] in "".join(result["cells"][-1]["source"])
    assert result_metadata["dataset_sources"] == ["owner/existing", "owner/new-model"]

"""Focused tests for the EXP-0014 dual-head Kaggle packager."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

COMPETITION_ROOT = Path(__file__).parents[1]
SCRIPTS = COMPETITION_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "prepare_temporal_link_abcd_submissions.py"
SPEC = importlib.util.spec_from_file_location("prepare_temporal_link_abcd_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


def test_config_declares_each_controlled_arm_once() -> None:
    config_path = COMPETITION_ROOT / "configs/exp-0014-tgraph3-link-ensemble-abcd.toml"
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    variants = packaging._variants(config)

    assert [variant["arm"] for variant in variants] == ["A", "B", "C", "D"]
    assert {variant["mode"] for variant in variants} == packaging.ALLOWED_MODES
    assert config["inference"]["minimum_component_nodes"] == 7


def test_notebook_keeps_paths_separate_from_inference_arguments() -> None:
    notebook = packaging._notebook(
        dataset_id="owner/dataset",
        title="ABCD fixture",
        mode="agreement_gate",
        manifest_sha256="a" * 64,
        logit_bound=0.15,
        minimum_component_nodes=7,
    )
    source = "".join(notebook["cells"][1]["source"])
    compile(source, "fixture.ipynb", "exec")
    before_command, command = source.split("command = [", 1)

    assert "--temporal-link-mode" not in before_command
    assert "--minimum-component-nodes" not in before_command
    assert "--temporal-graph-checkpoint" in command
    assert "--temporal-graph-attention-checkpoint" in command
    assert "'agreement_gate'" in command
    assert "'7'" in command
    assert notebook["metadata"]["agentic_kaggle"]["manifest_sha256"] == "a" * 64
    json.dumps(notebook)

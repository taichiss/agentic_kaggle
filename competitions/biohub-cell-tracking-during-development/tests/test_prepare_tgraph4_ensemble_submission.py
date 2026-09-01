"""Focused tests for the EXP-0015 T_graph=4 ensemble packager."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

COMPETITION_ROOT = Path(__file__).parents[1]
SCRIPTS = COMPETITION_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "prepare_tgraph4_ensemble_submission.py"
SPEC = importlib.util.spec_from_file_location("prepare_tgraph4_ensemble_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_base_bundle(root: Path, source_sha256: str) -> tuple[Path, str]:
    bundle = root / "base"
    bundle.mkdir()
    weights = bundle / "edge_predictor_best.pth"
    weights.write_bytes(b"frozen-host-fixture")
    weights_sha256 = _sha256(weights)
    (bundle / "config.json").write_text("{}\n", encoding="utf-8")
    (bundle / "checkpoint-metadata.json").write_text(
        json.dumps(
            {
                "weights_sha256": weights_sha256,
                "source_checkpoint_sha256": source_sha256,
                "completed_epochs": 30,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "ORGANIZER-LICENSE").write_text("fixture\n", encoding="utf-8")
    expanded = bundle / "tracking_cellmot_models"
    required_tracking_members = (
        "tracking_cellmot/__init__.py",
        "tracking_cellmot/models/__init__.py",
        "tracking_cellmot/models/simple_node_transformer.py",
        "tracking_cellmot/models/temporal_unet.py",
    )
    for relative in required_tracking_members:
        path = expanded / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    return bundle, weights_sha256


def _graph_payload(
    *,
    experiment_id: str,
    completed_epoch: int,
    architecture: str,
    graph_window_size: int,
    base_sha256: str,
    cache_fingerprint: str,
    source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph_config: dict[str, Any] = {
        "node_feature_dim": 32,
        "hidden_dim": 64,
        "top_k": 8,
        "radius_um": 15.0,
        "distance_scale_um": 10.0,
        "dropout": 0.1,
        "middle_coord_atol": 1.0e-4,
        "image_window_size": 2,
        "graph_window_size": graph_window_size,
        "architecture": architecture,
        "ownership": "right_transition",
        "schema_version": 1,
    }
    if architecture == "candidate_attention":
        graph_config.update(
            {
                "attention_heads": 4,
                "residual_logit_bound": 0.15,
            }
        )
    metadata = {
        "experiment_id": experiment_id,
        "completed_epochs": completed_epoch,
        "cache_fingerprint": cache_fingerprint,
        "ownership": "right_transition",
        "image_window_size": 2,
        "graph_window_size": graph_window_size,
        "source_raw_checkpoint_sha256": source_sha256,
    }
    payload = {
        "schema_version": 1,
        "config": graph_config,
        "state_dict": {},
        "base_checkpoint_sha256": base_sha256,
        "metadata": metadata,
    }
    wrapper = {
        "experiment_id": experiment_id,
        "completed_epochs": completed_epoch,
        "temporal_graph": payload,
    }
    return wrapper, payload


def _fixture(tmp_path: Path) -> dict[str, Any]:
    source_sha256 = "2" * 64
    primary_cache = "3" * 64
    fallback_cache = "4" * 64
    bundle, base_sha256 = _write_base_bundle(tmp_path, source_sha256)
    head_records = {
        "mlp": ("EXP-0015", 4, "mlp", 4, primary_cache),
        "attention": (
            "EXP-0015-ATTN",
            5,
            "candidate_attention",
            4,
            primary_cache,
        ),
        "fallback_mlp": ("EXP-0009", 3, "mlp", 3, fallback_cache),
        "fallback_attention": (
            "EXP-0012",
            3,
            "candidate_attention",
            3,
            fallback_cache,
        ),
    }
    checkpoint_paths: dict[str, Path] = {}
    payloads: dict[Path, tuple[dict[str, Any], dict[str, Any]]] = {}
    checkpoint_sha256: dict[str, str] = {}
    for name, (experiment, epoch, architecture, window, cache) in head_records.items():
        path = tmp_path / f"{name}.pth"
        path.write_bytes(f"checkpoint fixture {name}".encode())
        checkpoint_paths[name] = path
        checkpoint_sha256[name] = _sha256(path)
        payloads[path.resolve()] = _graph_payload(
            experiment_id=experiment,
            completed_epoch=epoch,
            architecture=architecture,
            graph_window_size=window,
            base_sha256=base_sha256,
            cache_fingerprint=cache,
            source_sha256=source_sha256,
        )

    graph_source = tmp_path / "temporal_graph"
    graph_source.mkdir()
    (graph_source / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    (graph_source / "model.py").write_text("# fixture\n", encoding="utf-8")
    inference_script = tmp_path / "run_kaggle_inference.py"
    inference_script.write_text(
        """\
def _verify_bundle_manifest(bundle):
    return bundle

FLAGS = (
    '--temporal-graph-checkpoint',
    '--temporal-graph-attention-checkpoint',
    '--temporal-graph-fallback-checkpoint',
    '--temporal-graph-fallback-attention-checkpoint',
    '--temporal-link-mode',
    '--minimum-component-nodes',
)
""",
        encoding="utf-8",
    )

    config_path = tmp_path / "exp-0015.toml"
    config_path.write_text(
        f"""\
schema_version = 1
experiment_id = "EXP-0015"

[source]
base_experiment_id = "EXP-0004"
base_checkpoint_completed_epochs = 30
base_checkpoint_path = "unused/edge_predictor_best.pth"
base_checkpoint_sha256 = "{base_sha256}"
source_checkpoint_sha256 = "{source_sha256}"

[data]
fold = 0
image_window_size = 2
graph_window_size = 4
cache_fingerprint = "{primary_cache}"
cache_manifest_sha256 = "{"5" * 64}"

[fallback]
graph_window_size = 3
cache_fingerprint = "{fallback_cache}"
cache_manifest_sha256 = "{"6" * 64}"

[heads.mlp]
experiment_id = "EXP-0015"
completed_epoch = 4
checkpoint_path = "unused/mlp.pth"
checkpoint_sha256 = "{checkpoint_sha256["mlp"]}"

[heads.attention]
experiment_id = "EXP-0015-ATTN"
completed_epoch = 5
checkpoint_path = "unused/attention.pth"
checkpoint_sha256 = "{checkpoint_sha256["attention"]}"

[heads.fallback_mlp]
experiment_id = "EXP-0009"
completed_epoch = 3
checkpoint_path = "unused/fallback-mlp.pth"
checkpoint_sha256 = "{checkpoint_sha256["fallback_mlp"]}"

[heads.fallback_attention]
experiment_id = "EXP-0012"
completed_epoch = 3
checkpoint_path = "unused/fallback-attention.pth"
checkpoint_sha256 = "{checkpoint_sha256["fallback_attention"]}"

[ensemble]
mlp_weight = 0.5
attention_weight = 0.5
center_over_valid_candidates = true
logit_bound = 0.15

[inference]
minimum_component_nodes = 7
preserve_division_components = true

[submission]
dataset_id = "suzukitaichi/biohub-exp-0015-tgraph4-link-5050"
dataset_title = "Biohub EXP-0015 Fixture Dataset"
kernel_id = "suzukitaichi/biohub-exp-0015-tgraph4-bounded-logit-5050-submit"
kernel_title = "Biohub EXP-0015 Fixture Kernel"
submission_message = "fixture"

[output]
bundle_dir = "unused/output"
""",
        encoding="utf-8",
    )
    return {
        "config_path": config_path,
        "base_bundle": bundle,
        "checkpoint_paths": checkpoint_paths,
        "payloads": payloads,
        "graph_source": graph_source,
        "inference_script": inference_script,
    }


def _install_checkpoint_loader(monkeypatch: pytest.MonkeyPatch, fixture) -> None:
    def fake_validate(path: Path, **expected):
        wrapper, payload = fixture["payloads"][path.resolve()]
        assert payload["base_checkpoint_sha256"] == expected["expected_base_sha256"]
        assert wrapper["completed_epochs"] == expected["expected_completed_epoch"]
        assert wrapper["experiment_id"] == expected["expected_experiment_id"]
        return wrapper, payload

    monkeypatch.setattr(packaging, "_validate_graph_checkpoint", fake_validate)


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture):
    _install_checkpoint_loader(monkeypatch, fixture)
    checkpoints = fixture["checkpoint_paths"]
    return packaging.prepare(
        fixture["config_path"],
        tmp_path / "output",
        base_bundle_override=fixture["base_bundle"],
        temporal_graph_checkpoint_override=checkpoints["mlp"],
        temporal_graph_attention_checkpoint_override=checkpoints["attention"],
        temporal_graph_fallback_checkpoint_override=checkpoints["fallback_mlp"],
        temporal_graph_fallback_attention_checkpoint_override=(checkpoints["fallback_attention"]),
        temporal_graph_source=fixture["graph_source"],
        inference_script=fixture["inference_script"],
    )


def test_notebook_compiles_with_four_explicit_head_arguments() -> None:
    notebook = packaging._notebook(
        dataset_id="owner/dataset",
        title="EXP-0015 fixture",
        manifest_sha256="a" * 64,
        logit_bound=0.15,
        minimum_component_nodes=7,
    )
    source = "".join(notebook["cells"][1]["source"])
    compile(source, "fixture.ipynb", "exec")
    before_command, command = source.split("command = [", 1)

    assert "--temporal-link-mode" not in before_command
    assert "--temporal-graph-fallback-checkpoint" in command
    assert "--temporal-graph-fallback-attention-checkpoint" in command
    assert "'bounded_logit_5050'" in command
    assert "'7'" in command
    assert notebook["metadata"]["agentic_kaggle"]["experiment_id"] == "EXP-0015"
    json.dumps(notebook)


def test_prepare_writes_content_addressed_dataset_and_one_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    result = _prepare(tmp_path, monkeypatch, fixture)
    dataset_dir = Path(result["dataset_dir"])
    kernel_dir = Path(result["kernel_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_metadata = json.loads(
        (dataset_dir / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    kernel_metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert set(manifest["temporal_graph_heads"]) == set(packaging.HEAD_SPECS)
    assert manifest["temporal_contract"]["primary_graph_window_size"] == 4
    assert manifest["temporal_contract"]["fallback_graph_window_size"] == 3
    assert manifest["ensemble"]["mode"] == "bounded_logit_5050"
    assert manifest["postprocess"]["minimum_component_nodes"] == 7
    for head_name, spec in packaging.HEAD_SPECS.items():
        checkpoint_name = spec["checkpoint_name"]
        assert manifest["temporal_graph_heads"][head_name]["checkpoint_sha256"] == _sha256(
            dataset_dir / checkpoint_name
        )
        assert checkpoint_name in manifest["files"]
    assert dataset_metadata["id"] == packaging.DEFAULT_DATASET_ID
    assert kernel_metadata["id"] == packaging.DEFAULT_KERNEL_ID
    assert kernel_metadata["dataset_sources"] == [packaging.DEFAULT_DATASET_ID]
    assert kernel_metadata["competition_sources"] == [packaging.COMPETITION_SLUG]
    assert len(list(kernel_dir.glob("*.ipynb"))) == 1

    notebook = json.loads(next(kernel_dir.glob("*.ipynb")).read_text(encoding="utf-8"))
    compile("".join(notebook["cells"][1]["source"]), "packaged.ipynb", "exec")
    assert result["manifest_sha256"] == _sha256(dataset_dir / "manifest.json")


def test_prepare_rejects_candidate_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _, payload = fixture["payloads"][fixture["checkpoint_paths"]["fallback_attention"].resolve()]
    payload["config"]["radius_um"] = 16.0
    _install_checkpoint_loader(monkeypatch, fixture)
    checkpoints = fixture["checkpoint_paths"]

    with pytest.raises(ValueError, match="different candidate contract"):
        packaging.prepare(
            fixture["config_path"],
            tmp_path / "rejected",
            base_bundle_override=fixture["base_bundle"],
            temporal_graph_checkpoint_override=checkpoints["mlp"],
            temporal_graph_attention_checkpoint_override=checkpoints["attention"],
            temporal_graph_fallback_checkpoint_override=checkpoints["fallback_mlp"],
            temporal_graph_fallback_attention_checkpoint_override=(
                checkpoints["fallback_attention"]
            ),
            temporal_graph_source=fixture["graph_source"],
            inference_script=fixture["inference_script"],
        )

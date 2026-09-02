"""Focused tests for the EXP-0016 T_graph=5 ensemble packager."""

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
import prepare_tgraph4_ensemble_submission as shared_packaging  # noqa: E402

SCRIPT = SCRIPTS / "prepare_tgraph5_ensemble_submission.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_tgraph5_ensemble_test",
    SCRIPT,
)
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
    caches = {
        "primary": "3" * 64,
        "t4_fallback": "4" * 64,
        "t3_fallback": "5" * 64,
    }
    bundle, base_sha256 = _write_base_bundle(tmp_path, source_sha256)
    head_records = {
        "mlp": ("EXP-0016-MLP", 5, "mlp", 5, caches["primary"]),
        "attention": (
            "EXP-0016-ATTN",
            5,
            "candidate_attention",
            5,
            caches["primary"],
        ),
        "t4_fallback_mlp": (
            "EXP-0015-MLP",
            8,
            "mlp",
            4,
            caches["t4_fallback"],
        ),
        "t4_fallback_attention": (
            "EXP-0015-ATTN",
            5,
            "candidate_attention",
            4,
            caches["t4_fallback"],
        ),
        "t3_fallback_mlp": (
            "EXP-0009",
            3,
            "mlp",
            3,
            caches["t3_fallback"],
        ),
        "t3_fallback_attention": (
            "EXP-0012",
            3,
            "candidate_attention",
            3,
            caches["t3_fallback"],
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
    inference_flags = tuple(str(spec["inference_flag"]) for spec in packaging.HEAD_SPECS.values())
    inference_script.write_text(
        "def _verify_bundle_manifest(bundle):\n"
        "    return bundle\n\n"
        f"FLAGS = {inference_flags!r}\n"
        "MODE = '--temporal-link-mode'\n"
        "BOUND = '--ensemble-logit-bound'\n"
        "MINIMUM = '--minimum-component-nodes'\n",
        encoding="utf-8",
    )

    head_tables = []
    for name, (experiment, epoch, _architecture, _window, _cache) in head_records.items():
        head_tables.append(
            f"""\
[heads.{name}]
experiment_id = "{experiment}"
completed_epoch = {epoch}
checkpoint_path = "unused/{name}.pth"
checkpoint_sha256 = "{checkpoint_sha256[name]}"
"""
        )
    config_path = tmp_path / "exp-0016.toml"
    config_path.write_text(
        f"""\
schema_version = 1
experiment_id = "EXP-0016"

[source]
base_experiment_id = "EXP-0004"
base_checkpoint_completed_epochs = 30
base_checkpoint_path = "unused/edge_predictor_best.pth"
base_checkpoint_sha256 = "{base_sha256}"
source_checkpoint_sha256 = "{source_sha256}"

[data]
fold = 0
image_window_size = 2
graph_window_size = 5
cache_fingerprint = "{caches["primary"]}"
cache_manifest_sha256 = "{"6" * 64}"

[fallback_t4]
graph_window_size = 4
cache_fingerprint = "{caches["t4_fallback"]}"
cache_manifest_sha256 = "{"7" * 64}"

[fallback_t3]
graph_window_size = 3
cache_fingerprint = "{caches["t3_fallback"]}"
cache_manifest_sha256 = "{"8" * 64}"

{"".join(head_tables)}
[ensemble]
mlp_weight = 0.5
attention_weight = 0.5
center_over_valid_candidates = true
logit_bound = 0.15

[inference]
base_postprocess_profile = "public-applicable-v1"
minimum_component_nodes = 7
preserve_division_components = true

[controls]
freeze_image_model = true
freeze_detection_head = true
freeze_host_edge_scorer = true
freeze_candidate_generation = true
freeze_division_policy = true
freeze_postprocess = true
effective_postprocess_profile = "public-applicable-v1-min-component-7"

[submission]
dataset_id = "{packaging.DEFAULT_DATASET_ID}"
dataset_title = "Biohub EXP-0016 Fixture Dataset"
kernel_id = "{packaging.DEFAULT_KERNEL_ID}"
kernel_title = "Biohub EXP-0016 Fixture Kernel"
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


def _install_checkpoint_loader(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
) -> None:
    def fake_validate(path: Path, **expected):
        wrapper, payload = fixture["payloads"][path.resolve()]
        assert payload["base_checkpoint_sha256"] == expected["expected_base_sha256"]
        assert wrapper["completed_epochs"] == expected["expected_completed_epoch"]
        assert wrapper["experiment_id"] == expected["expected_experiment_id"]
        return wrapper, payload

    monkeypatch.setattr(shared_packaging, "_validate_graph_checkpoint", fake_validate)


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
):
    _install_checkpoint_loader(monkeypatch, fixture)
    checkpoints = fixture["checkpoint_paths"]
    return packaging.prepare(
        fixture["config_path"],
        tmp_path / "output",
        base_bundle_override=fixture["base_bundle"],
        temporal_graph_checkpoint_override=checkpoints["mlp"],
        temporal_graph_attention_checkpoint_override=checkpoints["attention"],
        temporal_graph_t4_fallback_checkpoint_override=(checkpoints["t4_fallback_mlp"]),
        temporal_graph_t4_fallback_attention_checkpoint_override=(
            checkpoints["t4_fallback_attention"]
        ),
        temporal_graph_fallback_checkpoint_override=checkpoints["t3_fallback_mlp"],
        temporal_graph_fallback_attention_checkpoint_override=(
            checkpoints["t3_fallback_attention"]
        ),
        temporal_graph_source=fixture["graph_source"],
        inference_script=fixture["inference_script"],
    )


def test_notebook_compiles_with_six_content_addressed_head_arguments() -> None:
    notebook = packaging._notebook(
        dataset_id="owner/dataset",
        title="EXP-0016 fixture",
        manifest_sha256="a" * 64,
    )
    source = "".join(notebook["cells"][1]["source"])
    compile(source, "fixture.ipynb", "exec")
    before_command, command = source.split("command = [", 1)

    assert "--temporal-link-mode" not in before_command
    for head_name, spec in packaging.HEAD_SPECS.items():
        flag = str(spec["inference_flag"])
        checkpoint_name = str(spec["checkpoint_name"])
        assert command.count(repr(flag)) == 1
        assert checkpoint_name in before_command
        assert f"head_{head_name}['checkpoint_sha256']" in before_command
    assert repr(packaging.TEMPORAL_LINK_MODE) in command
    assert repr(str(packaging.ENSEMBLE_LOGIT_BOUND)) in command
    assert repr(str(packaging.MINIMUM_COMPONENT_NODES)) in command
    metadata = notebook["metadata"]["agentic_kaggle"]
    assert metadata["experiment_id"] == "EXP-0016"
    assert set(metadata["head_checkpoints"]) == set(packaging.HEAD_SPECS)
    assert metadata["fallback_order"] == [5, 4, 3, "host"]
    json.dumps(notebook)


def test_prepare_writes_six_head_content_hashes_and_one_offline_gpu_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    assert set(result["head_checkpoint_sha256"]) == set(packaging.HEAD_SPECS)
    assert manifest["temporal_contract"]["fallback_order"] == [5, 4, 3, "host"]
    assert {
        tier: contract["graph_window_size"]
        for tier, contract in manifest["temporal_contract"]["tiers"].items()
    } == {"primary": 5, "t4_fallback": 4, "t3_fallback": 3}
    assert manifest["ensemble"]["mode"] == "bounded_logit_5050"
    assert manifest["ensemble"]["mlp_weight"] == 0.5
    assert manifest["ensemble"]["attention_weight"] == 0.5
    assert manifest["postprocess"]["minimum_component_nodes"] == 7
    for head_name, spec in packaging.HEAD_SPECS.items():
        checkpoint_name = str(spec["checkpoint_name"])
        actual_sha256 = _sha256(dataset_dir / checkpoint_name)
        head = manifest["temporal_graph_heads"][head_name]
        assert head["checkpoint"] == checkpoint_name
        assert head["checkpoint_sha256"] == actual_sha256
        assert manifest["files"][checkpoint_name]["sha256"] == actual_sha256
        assert result["head_checkpoint_sha256"][head_name] == actual_sha256
    assert dataset_metadata["id"] == packaging.DEFAULT_DATASET_ID
    assert kernel_metadata["id"] == packaging.DEFAULT_KERNEL_ID
    assert kernel_metadata["dataset_sources"] == [packaging.DEFAULT_DATASET_ID]
    assert kernel_metadata["competition_sources"] == [packaging.COMPETITION_SLUG]
    assert kernel_metadata["enable_gpu"] == "true"
    assert kernel_metadata["enable_internet"] == "false"
    assert len(list(kernel_dir.glob("*.ipynb"))) == 1

    notebook = json.loads(next(kernel_dir.glob("*.ipynb")).read_text(encoding="utf-8"))
    compile("".join(notebook["cells"][1]["source"]), "packaged.ipynb", "exec")
    assert result["manifest_sha256"] == _sha256(dataset_dir / "manifest.json")


def test_prepare_rejects_candidate_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _, payload = fixture["payloads"][fixture["checkpoint_paths"]["t3_fallback_attention"].resolve()]
    payload["config"]["radius_um"] = 16.0

    with pytest.raises(ValueError, match="different candidate contract"):
        _prepare(tmp_path, monkeypatch, fixture)


def test_prepare_rejects_wrong_tier_cache_and_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _, payload = fixture["payloads"][fixture["checkpoint_paths"]["t4_fallback_mlp"].resolve()]
    payload["metadata"]["cache_fingerprint"] = "9" * 64
    payload["metadata"]["graph_window_size"] = 5

    with pytest.raises(ValueError, match="cache fingerprint mismatch|graph window mismatch"):
        _prepare(tmp_path, monkeypatch, fixture)


def test_prepare_rejects_checkpoint_byte_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["checkpoint_paths"]["attention"].write_bytes(b"mutated")

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        _prepare(tmp_path, monkeypatch, fixture)


def test_prepare_rejects_missing_t4_runtime_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    source = fixture["inference_script"].read_text(encoding="utf-8")
    fixture["inference_script"].write_text(
        source.replace("--temporal-graph-t4-fallback-checkpoint", "--missing-t4"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing T_graph=5 wiring"):
        _prepare(tmp_path, monkeypatch, fixture)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("mlp_weight = 0.5", "mlp_weight = 0.6", "50:50"),
        (
            "minimum_component_nodes = 7",
            "minimum_component_nodes = 6",
            "minimum_component_nodes=7",
        ),
        (
            "freeze_host_edge_scorer = true",
            "freeze_host_edge_scorer = false",
            "frozen Host controls",
        ),
    ],
)
def test_prepare_rejects_nonfixed_submission_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    source = fixture["config_path"].read_text(encoding="utf-8")
    assert old in source
    fixture["config_path"].write_text(
        source.replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _prepare(tmp_path, monkeypatch, fixture)

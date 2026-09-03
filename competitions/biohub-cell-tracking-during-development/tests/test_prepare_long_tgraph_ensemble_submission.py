"""Focused tests for content-addressed T_graph=10/20 submission packaging."""

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

SCRIPT = SCRIPTS / "prepare_long_tgraph_ensemble_submission.py"
SPEC = importlib.util.spec_from_file_location("prepare_long_tgraph_test", SCRIPT)
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
    required_tracking_members = (
        "tracking_cellmot/__init__.py",
        "tracking_cellmot/models/__init__.py",
        "tracking_cellmot/models/simple_node_transformer.py",
        "tracking_cellmot/models/temporal_unet.py",
    )
    for relative in required_tracking_members:
        path = bundle / "tracking_cellmot_models" / relative
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
    cache_manifest_sha256: str,
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
        graph_config.update({"attention_heads": 4, "residual_logit_bound": 0.15})
    metadata = {
        "experiment_id": experiment_id,
        "completed_epochs": completed_epoch,
        "cache_fingerprint": cache_fingerprint,
        "cache_manifest_sha256": cache_manifest_sha256,
        "ownership": "right_transition",
        "image_window_size": 2,
        "graph_window_size": graph_window_size,
        "source_raw_checkpoint_sha256": source_sha256,
    }
    if graph_window_size >= 6:
        metadata.update(
            {
                "cache_schema_version": 4,
                "feature_schema": "tgraph-long-history-linear-aggregate-features-v1",
                "feature_width": 122,
            }
        )
    payload = {
        "schema_version": 1,
        "config": graph_config,
        "state_dict": {},
        "base_checkpoint_sha256": base_sha256,
        "metadata": metadata,
    }
    return (
        {
            "experiment_id": experiment_id,
            "completed_epochs": completed_epoch,
            "temporal_graph": payload,
        },
        payload,
    )


def _fixture(
    tmp_path: Path,
    *,
    experiment_id: str,
    primary_window: int,
    fallback_windows: tuple[int, ...],
) -> dict[str, Any]:
    source_sha256 = "2" * 64
    bundle, base_sha256 = _write_base_bundle(tmp_path, source_sha256)
    windows = (primary_window, *fallback_windows)
    cache_by_window = {window: f"{index:x}" * 64 for index, window in enumerate(windows, 3)}
    manifest_by_window = {
        window: f"{index:x}" * 64 for index, window in enumerate(windows, 10)
    }
    tier_names = ("primary", *(f"t{window}" for window in fallback_windows))
    tiers = tuple(zip(tier_names, windows, strict=True))
    head_paths: dict[str, Path] = {}
    payloads: dict[Path, tuple[dict[str, Any], dict[str, Any]]] = {}
    head_tables = []
    for tier, window in tiers:
        for architecture in ("mlp", "candidate_attention"):
            name = packaging._head_name(tier, architecture)
            suffix = "ATTN" if architecture == "candidate_attention" else "MLP"
            head_experiment = f"FIXTURE-T{window}-{suffix}"
            path = tmp_path / f"{name}.pth"
            path.write_bytes(f"checkpoint fixture {name}".encode())
            head_paths[name] = path
            payloads[path.resolve()] = _graph_payload(
                experiment_id=head_experiment,
                completed_epoch=5,
                architecture=architecture,
                graph_window_size=window,
                base_sha256=base_sha256,
                cache_fingerprint=cache_by_window[window],
                cache_manifest_sha256=manifest_by_window[window],
                source_sha256=source_sha256,
            )
            architecture_fields = (
                "architecture = \"candidate_attention\"\n"
                "attention_heads = 4\n"
                "residual_logit_bound = 0.15\n"
                if architecture == "candidate_attention"
                else ""
            )
            head_tables.append(
                f"""\
[heads.{name}]
experiment_id = "{head_experiment}"
completed_epoch = 5
checkpoint_path = "unused/{name}.pth"
checkpoint_sha256 = "{_sha256(path)}"
{architecture_fields}
"""
            )

    fallback_tables = []
    for window in fallback_windows:
        fallback_tables.append(
            f"""\
[fallbacks.t{window}]
graph_window_size = {window}
cache_fingerprint = "{cache_by_window[window]}"
cache_manifest_sha256 = "{manifest_by_window[window]}"
"""
        )
    config_path = tmp_path / f"{experiment_id.lower()}.toml"
    config_path.write_text(
        f"""\
schema_version = 1
experiment_id = "{experiment_id}"

[source]
base_experiment_id = "EXP-0004"
base_checkpoint_completed_epochs = 30
base_checkpoint_path = "unused/edge_predictor_best.pth"
base_checkpoint_sha256 = "{base_sha256}"
source_checkpoint_sha256 = "{source_sha256}"

[data]
fold = 0
image_window_size = 2
graph_window_size = {primary_window}
cache_fingerprint = "{cache_by_window[primary_window]}"
cache_manifest_sha256 = "{manifest_by_window[primary_window]}"

{"".join(fallback_tables)}
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
dataset_id = "owner/{experiment_id.lower()}-dataset"
dataset_title = "{experiment_id} Fixture Dataset"
kernel_id = "owner/{experiment_id.lower()}-kernel"
kernel_title = "{experiment_id} Fixture Kernel"
submission_message = "fixture"

[output]
bundle_dir = "unused/output"
""",
        encoding="utf-8",
    )
    graph_source = tmp_path / "temporal_graph"
    graph_source.mkdir()
    (graph_source / "__init__.py").write_text("# fixture\n", encoding="utf-8")
    (graph_source / "model.py").write_text("# history_pairs fixture\n", encoding="utf-8")
    inference_script = tmp_path / "run_kaggle_inference.py"
    inference_script.write_text(
        "def _verify_bundle_manifest(bundle):\n"
        "    return bundle\n\n"
        "FLAGS = ('--temporal-graph-checkpoint', "
        "'--temporal-graph-attention-checkpoint', "
        "'--temporal-graph-fallback-stack', '--temporal-link-mode', "
        "'--ensemble-logit-bound', '--minimum-component-nodes')\n"
        "history_pairs = []\n",
        encoding="utf-8",
    )
    return {
        "config_path": config_path,
        "base_bundle": bundle,
        "head_paths": head_paths,
        "payloads": payloads,
        "graph_source": graph_source,
        "inference_script": inference_script,
    }


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture: dict[str, Any]):
    def fake_validate(path: Path, **expected):
        wrapper, payload = fixture["payloads"][path.resolve()]
        assert payload["base_checkpoint_sha256"] == expected["expected_base_sha256"]
        assert wrapper["completed_epochs"] == expected["expected_completed_epoch"]
        assert wrapper["experiment_id"] == expected["expected_experiment_id"]
        return wrapper, payload

    monkeypatch.setattr(shared_packaging, "_validate_graph_checkpoint", fake_validate)
    return packaging.prepare(
        fixture["config_path"],
        tmp_path / "output",
        base_bundle_override=fixture["base_bundle"],
        head_overrides=fixture["head_paths"],
        temporal_graph_source=fixture["graph_source"],
        inference_script=fixture["inference_script"],
    )


@pytest.mark.parametrize(
    ("experiment_id", "primary_window", "fallback_windows", "head_count"),
    [
        ("EXP-0017", 10, (5, 4, 3), 8),
        ("EXP-0018", 20, (10, 5, 4, 3), 10),
    ],
)
def test_prepare_writes_content_addressed_long_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_id: str,
    primary_window: int,
    fallback_windows: tuple[int, ...],
    head_count: int,
) -> None:
    fixture = _fixture(
        tmp_path,
        experiment_id=experiment_id,
        primary_window=primary_window,
        fallback_windows=fallback_windows,
    )
    result = _prepare(tmp_path, monkeypatch, fixture)
    dataset_dir = Path(result["dataset_dir"])
    kernel_dir = Path(result["kernel_dir"])
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    descriptor = json.loads(
        (dataset_dir / packaging.FALLBACK_STACK_NAME).read_text(encoding="utf-8")
    )
    kernel_metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())

    assert len(manifest["temporal_graph_heads"]) == head_count
    assert len(result["head_checkpoint_sha256"]) == head_count
    assert manifest["temporal_contract"]["fallback_order"] == [
        primary_window,
        *fallback_windows,
        "host",
    ]
    assert [entry["graph_window_size"] for entry in descriptor["fallbacks"]] == list(
        fallback_windows
    )
    assert manifest["fallback_stack"]["sha256"] == _sha256(
        dataset_dir / packaging.FALLBACK_STACK_NAME
    )
    assert manifest["files"][packaging.FALLBACK_STACK_NAME]["sha256"] == (
        manifest["fallback_stack"]["sha256"]
    )
    assert kernel_metadata["enable_gpu"] == "true"
    assert kernel_metadata["enable_internet"] == "false"
    notebook = json.loads(next(kernel_dir.glob("*.ipynb")).read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    compile(source, "packaged.ipynb", "exec")
    command = source.split("command = [", 1)[1]
    assert command.count(repr("--temporal-graph-fallback-stack")) == 1
    assert "--temporal-graph-t4-fallback-checkpoint" not in command
    assert notebook["metadata"]["agentic_kaggle"]["fallback_order"] == [
        primary_window,
        *fallback_windows,
        "host",
    ]
    assert result["manifest_sha256"] == _sha256(dataset_dir / "manifest.json")


def test_prepare_rejects_incomplete_controlled_fallback_ladder(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        experiment_id="EXP-0017",
        primary_window=10,
        fallback_windows=(5, 4, 3),
    )
    source = fixture["config_path"].read_text(encoding="utf-8")
    start = source.index("[fallbacks.t4]")
    end = source.index("[fallbacks.t3]")
    fixture["config_path"].write_text(source[:start] + source[end:], encoding="utf-8")

    with pytest.raises(ValueError, match="requires fallback tiers"):
        packaging.prepare(fixture["config_path"])


def test_prepare_rejects_missing_generic_inference_wiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        experiment_id="EXP-0017",
        primary_window=10,
        fallback_windows=(5, 4, 3),
    )
    source = fixture["inference_script"].read_text(encoding="utf-8")
    fixture["inference_script"].write_text(
        source.replace("--temporal-graph-fallback-stack", "--missing-stack"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing long T_graph wiring"):
        _prepare(tmp_path, monkeypatch, fixture)


def test_prepare_rejects_long_feature_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        experiment_id="EXP-0017",
        primary_window=10,
        fallback_windows=(5, 4, 3),
    )
    _, payload = fixture["payloads"][fixture["head_paths"]["primary_mlp"].resolve()]
    payload["metadata"]["feature_schema"] = "wrong-schema"

    with pytest.raises(ValueError, match="long-window feature schema mismatch"):
        _prepare(tmp_path, monkeypatch, fixture)


def test_parse_head_overrides_is_unique() -> None:
    assert packaging._parse_head_overrides(["primary_mlp=/tmp/mlp.pth"])[
        "primary_mlp"
    ] == Path("/tmp/mlp.pth")
    with pytest.raises(ValueError, match="unique NAME=PATH"):
        packaging._parse_head_overrides(
            ["primary_mlp=/tmp/a.pth", "primary_mlp=/tmp/b.pth"]
        )

"""Focused tests for EXP-0009 Kaggle Dataset/Notebook packaging."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

COMPETITION_ROOT = Path(__file__).parents[1]
SCRIPTS = COMPETITION_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "prepare_temporal_graph_submission.py"
SPEC = importlib.util.spec_from_file_location("prepare_temporal_graph_submission_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_bundle(root: Path, *, tracking_layout: str = "expanded") -> tuple[Path, str]:
    bundle = root / "base"
    bundle.mkdir()
    weights = bundle / "edge_predictor_best.pth"
    weights.write_bytes(b"flattened frozen host epoch 30")
    weights_sha256 = _sha256(weights)
    (bundle / "config.json").write_text(
        json.dumps(
            {
                "unet_out_channels": 32,
                "downsample": [1, 4, 4],
                "window_size": 2,
            }
        ),
        encoding="utf-8",
    )
    (bundle / "checkpoint-metadata.json").write_text(
        json.dumps(
            {
                "completed_epochs": 30,
                "weights_sha256": weights_sha256,
                "source_checkpoint_sha256": "7" * 64,
            }
        ),
        encoding="utf-8",
    )
    tracking_files = {
        "tracking_cellmot/__init__.py": b"from .models import TemporalUNet3D\n",
        "tracking_cellmot/models/__init__.py": b"class TemporalUNet3D:\n    pass\n",
        "tracking_cellmot/models/simple_node_transformer.py": b"class Transformer:\n    pass\n",
        "tracking_cellmot/models/temporal_unet.py": b"class UNet:\n    pass\n",
    }
    if tracking_layout == "expanded":
        for relative, contents in tracking_files.items():
            destination = bundle / "tracking_cellmot_models" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
    elif tracking_layout == "archive":
        with zipfile.ZipFile(bundle / "tracking_cellmot_models.zip", "w") as archive:
            for relative, contents in tracking_files.items():
                archive.writestr(relative, contents)
    else:
        raise ValueError(f"unsupported tracking fixture layout: {tracking_layout}")
    (bundle / "ORGANIZER-LICENSE").write_text("BSD fixture\n", encoding="utf-8")
    (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    return bundle, weights_sha256


def _graph_checkpoint(
    path: Path,
    *,
    base_sha256: str,
    completed_epochs: int = 5,
) -> None:
    payload = {
        "schema_version": 1,
        "config": {
            "node_feature_dim": 32,
            "hidden_dim": 64,
            "top_k": 8,
            "radius_um": 15.0,
            "distance_scale_um": 10.0,
            "dropout": 0.1,
            "middle_coord_atol": 0.0001,
            "image_window_size": 2,
            "graph_window_size": 3,
        },
        "state_dict": {"residual_mlp.fixture": torch.zeros(1)},
        "base_checkpoint_sha256": base_sha256,
        "metadata": {
            "completed_epochs": completed_epochs,
            "experiment_id": "EXP-0009",
        },
    }
    torch.save(
        {
            "temporal_graph": payload,
            "completed_epochs": completed_epochs,
            "experiment_id": "EXP-0009",
        },
        path,
    )


def _experiment_config(path: Path, base_sha256: str) -> None:
    path.write_text(
        f'''schema_version = 1
experiment_id = "EXP-0009"

[source]
base_experiment_id = "EXP-0004"
base_checkpoint_completed_epochs = 30
base_checkpoint_path = "unused/base/edge_predictor_best.pth"
base_checkpoint_sha256 = "{base_sha256}"

[data]
fold = 0

[submission]
competition = "biohub-cell-tracking-during-development"

[[submission.milestones]]
completed_epoch = 5
checkpoint = "checkpoint_epoch_0005.pth"
variant = "early-lb-probe"
dataset_id = "fixture-owner/biohub-exp-0009-e5"
kernel_id = "fixture-owner/biohub-exp-0009-e5-submit"
postprocess_profile = "public-applicable-v1"

[[submission.milestones]]
completed_epoch = 30
checkpoint = "checkpoint_epoch_0030.pth"
variant = "final-30e"
dataset_id = "fixture-owner/biohub-exp-0009-e30"
kernel_id = "fixture-owner/biohub-exp-0009-e30-submit"
postprocess_profile = "public-applicable-v1"

[output]
artifact_dir = "artifacts/EXP-0009"
''',
        encoding="utf-8",
    )


def _graph_source(root: Path) -> Path:
    source = root / "temporal_graph"
    source.mkdir()
    (source / "__init__.py").write_text("from .model import Head\n", encoding="utf-8")
    (source / "model.py").write_text("class Head:\n    pass\n", encoding="utf-8")
    return source


def _inference_script(path: Path) -> Path:
    path.write_text(
        "# fixture\n"
        "OPTION = '--temporal-graph-checkpoint'\n"
        "def _verify_bundle_manifest(_bundle):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    return path


def _fixtures(tmp_path: Path) -> dict[str, Path | str]:
    base_bundle, base_sha256 = _base_bundle(tmp_path)
    graph_checkpoint = tmp_path / "checkpoint_epoch_0005.pth"
    _graph_checkpoint(graph_checkpoint, base_sha256=base_sha256)
    config = tmp_path / "experiment.toml"
    _experiment_config(config, base_sha256)
    return {
        "base_bundle": base_bundle,
        "base_sha256": base_sha256,
        "graph_checkpoint": graph_checkpoint,
        "config": config,
        "graph_source": _graph_source(tmp_path),
        "inference": _inference_script(tmp_path / "run_kaggle_inference.py"),
    }


def test_prepare_uses_milestone_ids_and_builds_verified_offline_bundle(
    tmp_path: Path,
) -> None:
    fixture = _fixtures(tmp_path)
    output = tmp_path / "output"
    result = packaging.prepare(
        fixture["config"],
        5,
        output,
        base_bundle_override=fixture["base_bundle"],
        graph_checkpoint_override=fixture["graph_checkpoint"],
        temporal_graph_source=fixture["graph_source"],
        inference_script=fixture["inference"],
    )

    assert result["dataset_id"] == "fixture-owner/biohub-exp-0009-e5"
    assert result["kernel_id"] == "fixture-owner/biohub-exp-0009-e5-submit"
    assert result["base_weights_sha256"] == fixture["base_sha256"]
    dataset = output / "dataset"
    assert (dataset / "edge_predictor_best.pth").read_bytes() == (
        fixture["base_bundle"] / "edge_predictor_best.pth"
    ).read_bytes()
    assert (dataset / "temporal_graph_checkpoint.pth").read_bytes() == (
        fixture["graph_checkpoint"]
    ).read_bytes()
    with zipfile.ZipFile(dataset / "temporal_graph.zip") as archive:
        assert archive.namelist() == [
            "temporal_graph/__init__.py",
            "temporal_graph/model.py",
        ]
    with zipfile.ZipFile(dataset / "tracking_cellmot_models.zip") as archive:
        assert archive.namelist() == [
            "tracking_cellmot/__init__.py",
            "tracking_cellmot/models/__init__.py",
            "tracking_cellmot/models/simple_node_transformer.py",
            "tracking_cellmot/models/temporal_unet.py",
        ]

    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["completed_epoch"] == 5
    assert (dataset / "dataset-metadata.json").is_file()
    assert "dataset-metadata.json" not in manifest["files"]
    assert manifest["base"]["weights_sha256"] == fixture["base_sha256"]
    assert (
        manifest["temporal_graph"]["base_checkpoint_sha256"]
        == fixture["base_sha256"]
    )
    assert set(manifest["files"]["tracking_cellmot_models.zip"]["members"]) == {
        "tracking_cellmot/__init__.py",
        "tracking_cellmot/models/__init__.py",
        "tracking_cellmot/models/simple_node_transformer.py",
        "tracking_cellmot/models/temporal_unet.py",
    }
    assert set(manifest["files"]["temporal_graph.zip"]["members"]) == {
        "temporal_graph/__init__.py",
        "temporal_graph/model.py",
    }
    for relative, details in manifest["files"].items():
        packaged = dataset / relative
        assert packaged.stat().st_size == details["bytes"]
        assert _sha256(packaged) == details["sha256"]

    kernel_metadata = json.loads(
        (output / "kernel/kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert kernel_metadata["enable_internet"] == "false"
    assert kernel_metadata["dataset_sources"] == [result["dataset_id"]]
    notebook = json.loads(next((output / "kernel").glob("*.ipynb")).read_text())
    notebook_source = "".join(notebook["cells"][1]["source"])
    assert "--temporal-graph-checkpoint" in notebook_source
    test_candidates_source, command_source = notebook_source.split("command = [", 1)
    assert "--postprocess-profile" not in test_candidates_source
    assert "--temporal-graph-checkpoint" not in test_candidates_source
    assert "--postprocess-profile" in command_source
    assert "--temporal-graph-checkpoint" in command_source
    assert "manifest['files']" in notebook_source
    assert result["manifest_sha256"] in notebook_source

    repeated_output = tmp_path / "repeated-output"
    packaging.prepare(
        fixture["config"],
        5,
        repeated_output,
        base_bundle_override=fixture["base_bundle"],
        graph_checkpoint_override=fixture["graph_checkpoint"],
        temporal_graph_source=fixture["graph_source"],
        inference_script=fixture["inference"],
    )
    assert _sha256(dataset / "tracking_cellmot_models.zip") == _sha256(
        repeated_output / "dataset/tracking_cellmot_models.zip"
    )
    assert _sha256(dataset / "temporal_graph.zip") == _sha256(
        repeated_output / "dataset/temporal_graph.zip"
    )


def test_prepare_rejects_graph_checkpoint_for_a_different_base(tmp_path: Path) -> None:
    fixture = _fixtures(tmp_path)
    _graph_checkpoint(fixture["graph_checkpoint"], base_sha256="0" * 64)

    with pytest.raises(ValueError, match="graph/base checkpoint SHA-256 mismatch"):
        packaging.prepare(
            fixture["config"],
            5,
            tmp_path / "output",
            base_bundle_override=fixture["base_bundle"],
            graph_checkpoint_override=fixture["graph_checkpoint"],
            temporal_graph_source=fixture["graph_source"],
            inference_script=fixture["inference"],
        )


def test_generated_manifest_accepts_kaggle_expanded_archive_layout(
    tmp_path: Path,
) -> None:
    fixture = _fixtures(tmp_path)
    output = tmp_path / "output"
    integrated_inference = COMPETITION_ROOT / "scripts/run_kaggle_inference.py"
    packaging.prepare(
        fixture["config"],
        5,
        output,
        base_bundle_override=fixture["base_bundle"],
        graph_checkpoint_override=fixture["graph_checkpoint"],
        temporal_graph_source=fixture["graph_source"],
        inference_script=integrated_inference,
    )
    dataset = output / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    archive_names = [
        name for name, entry in manifest["files"].items() if "members" in entry
    ]
    assert set(archive_names) == {
        "temporal_graph.zip",
        "tracking_cellmot_models.zip",
    }
    for archive_name in archive_names:
        archive_path = dataset / archive_name
        expanded = dataset / Path(archive_name).stem
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(expanded)
        archive_path.unlink()

    assert not (dataset / "tracking_cellmot_models.zip").exists()
    assert (dataset / "tracking_cellmot_models/tracking_cellmot/__init__.py").is_file()
    assert not (dataset / "temporal_graph.zip").exists()
    assert (dataset / "temporal_graph/temporal_graph/__init__.py").is_file()

    inference_spec = importlib.util.spec_from_file_location(
        "expanded_bundle_inference_test",
        dataset / "run_kaggle_inference.py",
    )
    assert inference_spec is not None and inference_spec.loader is not None
    inference = importlib.util.module_from_spec(inference_spec)
    inference_spec.loader.exec_module(inference)
    inference._verify_bundle_manifest(dataset)

    notebook = json.loads(next((output / "kernel").glob("*.ipynb")).read_text())
    notebook_source = "".join(notebook["cells"][1]["source"])
    assert "verification_module._verify_bundle_manifest(bundle)" in notebook_source
    assert "assert packaged.is_file()" not in notebook_source


def test_prepare_rejects_checkpoint_epoch_that_differs_from_milestone(
    tmp_path: Path,
) -> None:
    fixture = _fixtures(tmp_path)
    _graph_checkpoint(
        fixture["graph_checkpoint"],
        base_sha256=fixture["base_sha256"],
        completed_epochs=6,
    )

    with pytest.raises(ValueError, match="completed epoch mismatch"):
        packaging.prepare(
            fixture["config"],
            5,
            tmp_path / "output",
            base_bundle_override=fixture["base_bundle"],
            graph_checkpoint_override=fixture["graph_checkpoint"],
            temporal_graph_source=fixture["graph_source"],
            inference_script=fixture["inference"],
        )

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "prepare_backbone_ab_submission.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_backbone_ab_submission", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
packager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = packager
SPEC.loader.exec_module(packager)
from backbone_ab.checkpointing import (  # noqa: E402
    DecoderProfile,
    InferenceProfile,
    write_inference_profile,
)


def _slugify(title: str) -> str:
    return "-".join(title.lower().split())


def test_fresh_package_directories_remove_only_exact_children(tmp_path: Path) -> None:
    output_root = tmp_path / "package"
    dataset = output_root / "dataset"
    kernel = output_root / "kernel"
    (dataset / "__pycache__").mkdir(parents=True)
    (dataset / "__pycache__" / "smoke.cpython-312.pyc").write_bytes(b"cache")
    kernel.mkdir()
    (kernel / "stale.ipynb").write_text("stale", encoding="utf-8")
    preserved = output_root / "keep.txt"
    preserved.write_text("keep", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    fresh_dataset, fresh_kernel = packager._fresh_package_directories(output_root)

    assert fresh_dataset == dataset
    assert fresh_kernel == kernel
    assert list(dataset.iterdir()) == []
    assert list(kernel.iterdir()) == []
    assert preserved.read_text(encoding="utf-8") == "keep"
    assert outside.read_text(encoding="utf-8") == "outside"


def test_fresh_package_directories_refuse_symlink_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "package"
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    output_root.mkdir()
    (output_root / "dataset").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked package directory"):
        packager._fresh_package_directories(output_root)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_fresh_package_directories_refuse_known_broad_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    competition_root = tmp_path / "repo" / "competitions" / "biohub"
    competition_root.mkdir(parents=True)
    repository_root = competition_root.parents[1]
    marker = repository_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(packager, "COMPETITION_ROOT", competition_root)

    with pytest.raises(ValueError, match="broad output root"):
        packager._fresh_package_directories(repository_root)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_zip_tree_is_deterministic_and_excludes_python_caches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "nested" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (source / "nested" / "module.pyc").write_bytes(b"compiled")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "package.cpython-312.pyc").write_bytes(b"compiled")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    packager._zip_tree(first, source, "example")
    os.utime(source / "package.py", (2_000_000_000, 2_000_000_000))
    packager._zip_tree(second, source, "example")

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "example/nested/module.py",
            "example/package.py",
        ]
        assert all(info.date_time == packager._ZIP_TIMESTAMP for info in archive.infolist())
        assert not any(
            "__pycache__" in name or name.endswith(".pyc")
            for name in archive.namelist()
        )


def test_kernel_title_and_notebook_ids_are_slug_stable() -> None:
    kernel_id = "suzukitaichi/biohub-exp-0007b-e5-submit"

    title, slug = packager._kaggle_title(kernel_id)
    notebook = packager._notebook(title)

    assert slug == kernel_id.split("/", 1)[1]
    assert _slugify(title) == slug
    assert [cell["id"] for cell in notebook["cells"]] == [
        "package-overview",
        "run-inference",
    ]
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5


@pytest.mark.parametrize(
    "identifier",
    [
        "missing-owner-separator",
        "owner/Has-Uppercase",
        "owner/double--hyphen",
        "owner/",
    ],
)
def test_kaggle_title_rejects_identifiers_that_cannot_round_trip(identifier: str) -> None:
    with pytest.raises(ValueError):
        packager._kaggle_title(identifier)


def test_exp7_identifier_requires_the_exact_selected_epoch_token() -> None:
    packager._validate_exp7_identifier(
        "owner/biohub-exp-0007a-epoch5", 5, kernel=False
    )
    packager._validate_exp7_identifier(
        "owner/biohub-exp-0007a-epoch5-submit", 5, kernel=True
    )

    with pytest.raises(ValueError, match="epoch5"):
        packager._validate_exp7_identifier(
            "owner/biohub-exp-0007a-epoch50", 5, kernel=False
        )


def test_corrected_prepare_fails_before_reusing_an_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "package"
    marker = output_root / "dataset" / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")
    checkpoint = SimpleNamespace(
        state_dict={},
        metadata={},
        sha256="0" * 64,
        source_format="wrapped",
    )
    monkeypatch.setattr(packager, "load_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setitem(
        sys.modules,
        "dynamic_network_architectures",
        SimpleNamespace(__path__=[str(tmp_path)]),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    with pytest.raises(ValueError, match="validated selection"):
        packager.prepare(
            packager.COMPETITION_ROOT
            / "configs/exp-0007a-corrected-spatial-50e.toml",
            tmp_path / "checkpoint_epoch_0005.pth",
            output_root,
            "owner/biohub-exp-0007a-epoch5",
            "owner/biohub-exp-0007a-epoch5-submit",
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_rejects_manual_corrected_v2_packaging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = (
        packager.COMPETITION_ROOT
        / "configs/exp-0007a-corrected-spatial-50e.toml"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_backbone_ab_submission.py",
            "--config",
            str(config),
            "--checkpoint",
            "checkpoint_epoch_0005.pth",
        ],
    )

    with pytest.raises(SystemExit):
        packager.main()


def test_cli_rejects_corrected_contract_when_model_api_is_implicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        packager.COMPETITION_ROOT
        / "configs/exp-0007a-corrected-spatial-50e.toml"
    ).read_text(encoding="utf-8")
    config = tmp_path / "implicit-corrected.toml"
    config.write_text(
        source.replace('model_api = "corrected_v2"\n', ""),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_backbone_ab_submission.py",
            "--config",
            str(config),
            "--checkpoint",
            "checkpoint_epoch_0005.pth",
        ],
    )

    with pytest.raises(SystemExit):
        packager.main()


def test_selection_cli_forwards_only_explicit_finalization_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_prepare_selected(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(packager, "prepare_selected", fake_prepare_selected)
    selection = tmp_path / "selection.json"
    report = tmp_path / "summary.json"
    output = tmp_path / "package"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_backbone_ab_submission.py",
            "--selection-json",
            str(selection),
            "--report-summary",
            str(report),
            "--output-root",
            str(output),
            "--dataset-id",
            "owner/biohub-exp-0007a-epoch30",
            "--kernel-id",
            "owner/biohub-exp-0007a-epoch30-submit",
            "--require-report-score-above",
            "0.5",
            "--report-score-tolerance",
            "0.000001",
        ],
    )

    assert packager.main() == 0
    assert captured == {
        "args": (
            selection.resolve(),
            report.resolve(),
            output.resolve(),
            "owner/biohub-exp-0007a-epoch30",
            "owner/biohub-exp-0007a-epoch30-submit",
        ),
        "kwargs": {
            "require_report_score_above": 0.5,
            "report_score_tolerance": 0.000001,
        },
    }


def test_prepare_selected_propagates_the_selected_epoch_and_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    competition_root = tmp_path / "competition"
    (competition_root / "data").mkdir(parents=True)
    selected = competition_root / "artifacts/EXP-0007A/checkpoint_epoch_0030.pth"
    binding = packager.FinalizationBinding(
        selection_path=competition_root / "selection.json",
        report_summary_path=competition_root / "summary.json",
        report_config_path=competition_root / "selected-report-screen.toml",
        experiment_config_path=(
            competition_root / "configs/exp-0007a-corrected-spatial-50e.toml"
        ),
        checkpoint_path=selected,
        inference_profile_path=competition_root / "selected_inference_profile.json",
        manifest_path=competition_root / "validation_split.json",
        completed_epoch=30,
        checkpoint_sha256="a" * 64,
        inference_profile_sha256="b" * 64,
        experiment_config_sha256="c" * 64,
        manifest_sha256="d" * 64,
        selection_sha256="e" * 64,
        report_summary_sha256="f" * 64,
        report_config_sha256="1" * 64,
        report_score=0.7,
    )
    captured = {}
    monkeypatch.setattr(packager, "COMPETITION_ROOT", competition_root)
    monkeypatch.setattr(
        packager,
        "validate_selection_report_binding",
        lambda *_args, **_kwargs: binding,
    )

    def fake_prepare(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"completed_epochs": 30}

    monkeypatch.setattr(packager, "prepare", fake_prepare)
    output = competition_root / "data/kaggle-exp-0007a-epoch30"

    result = packager.prepare_selected(
        binding.selection_path,
        binding.report_summary_path,
        output,
        "owner/biohub-exp-0007a-epoch30",
        "owner/biohub-exp-0007a-epoch30-submit",
        require_report_score_above=packager.EXP7A_EPOCH5_REPORT_BASELINE,
        report_score_tolerance=0.0,
    )

    assert result == {"completed_epochs": 30}
    assert captured["args"][:3] == (
        binding.experiment_config_path,
        binding.checkpoint_path,
        output,
    )
    assert captured["args"][5] == binding.inference_profile_path
    assert captured["kwargs"]["expected_completed_epoch"] == 30
    assert captured["kwargs"]["selection_provenance"]["completed_epoch"] == 30


def test_prepare_selected_rejects_a_non_improving_report_before_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = SimpleNamespace(report_score=packager.EXP7A_EPOCH5_REPORT_BASELINE)
    monkeypatch.setattr(
        packager,
        "validate_selection_report_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        packager,
        "prepare",
        lambda *_args, **_kwargs: pytest.fail("package must not be generated"),
    )

    with pytest.raises(ValueError, match="does not exceed gate"):
        packager.prepare_selected(
            tmp_path / "selection.json",
            tmp_path / "summary.json",
            tmp_path / "data/kaggle-exp-0007a-epoch30",
            "owner/biohub-exp-0007a-epoch30",
            "owner/biohub-exp-0007a-epoch30-submit",
            require_report_score_above=packager.EXP7A_EPOCH5_REPORT_BASELINE,
            report_score_tolerance=0.0,
        )


def test_prepare_selected_applies_tolerance_above_the_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = SimpleNamespace(
        report_score=packager.EXP7A_EPOCH5_REPORT_BASELINE + 0.0000001
    )
    monkeypatch.setattr(
        packager,
        "validate_selection_report_binding",
        lambda *_args, **_kwargs: binding,
    )

    with pytest.raises(ValueError, match="does not exceed gate"):
        packager.prepare_selected(
            tmp_path / "selection.json",
            tmp_path / "summary.json",
            tmp_path / "data/kaggle-exp-0007a-epoch30",
            "owner/biohub-exp-0007a-epoch30",
            "owner/biohub-exp-0007a-epoch30-submit",
            require_report_score_above=packager.EXP7A_EPOCH5_REPORT_BASELINE,
            report_score_tolerance=0.000001,
        )


def test_prepare_selected_rejects_an_unpinned_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        packager,
        "validate_selection_report_binding",
        lambda *_args, **_kwargs: SimpleNamespace(report_score=0.7),
    )

    with pytest.raises(ValueError, match="pinned epoch-5 report baseline"):
        packager.prepare_selected(
            tmp_path / "selection.json",
            tmp_path / "summary.json",
            tmp_path / "data/kaggle-exp-0007a-epoch30",
            "owner/biohub-exp-0007a-epoch30",
            "owner/biohub-exp-0007a-epoch30-submit",
            require_report_score_above=0.5,
            report_score_tolerance=0.0,
        )


def test_corrected_package_records_selected_source_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    competition_root = tmp_path / "competition"
    config_path = competition_root / "configs/exp-0007a-corrected-spatial-50e.toml"
    config_path.parent.mkdir(parents=True)
    source_config = (
        packager.COMPETITION_ROOT
        / "configs/exp-0007a-corrected-spatial-50e.toml"
    )
    config_path.write_bytes(source_config.read_bytes())
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    organizer = competition_root / config["source"]["organizer_repository_path"]
    for relative in (
        "tracking_cellmot/__init__.py",
        "tracking_cellmot/models/__init__.py",
        "tracking_cellmot/models/temporal_unet.py",
        "tracking_cellmot/models/simple_node_transformer.py",
    ):
        source = organizer / "src" / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("VALUE = 1\n", encoding="utf-8")
    script = competition_root / "scripts/run_kaggle_inference.py"
    script.parent.mkdir(parents=True)
    script.write_text("VALUE = 1\n", encoding="utf-8")
    backbone = competition_root / "src/backbone_ab/backbones.py"
    backbone.parent.mkdir(parents=True)
    backbone.write_text("VALUE = 1\n", encoding="utf-8")
    dynamic_root = tmp_path / "dynamic_network_architectures"
    dynamic_root.mkdir()
    (dynamic_root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (competition_root / "data").mkdir(exist_ok=True)

    checkpoint_path = competition_root / "artifacts/EXP-0007A/checkpoint_epoch_0030.pth"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"source-checkpoint")
    checkpoint_sha = packager.sha256_file(checkpoint_path)
    resume_config = json.loads(json.dumps(config))
    resume_config["train"].pop("epochs")
    split_sha = "d" * 64
    checkpoint = SimpleNamespace(
        state_dict={"weight": "state"},
        metadata={
            "completed_epochs": 30,
            "experiment_id": "EXP-0007A",
            "model_contract": "corrected_v2",
            "resume_fingerprint": packager.canonical_json_sha256(resume_config),
            "validation_subset_manifest_sha256": split_sha,
        },
        sha256=checkpoint_sha,
        source_format="wrapped",
    )
    profile_path = competition_root / "selected_inference_profile.json"
    profile = InferenceProfile(
        experiment_id="EXP-0007A",
        model_api="corrected_v2",
        checkpoint_sha256=checkpoint_sha,
        experiment_config_sha256=packager.sha256_file(config_path),
        source_revision=config["source"]["organizer_revision"],
        downsample=(1, 4, 4),
        window_size=2,
        detection_threshold=0.1,
        detection_tta=True,
        pool_kernel_um=5.0,
        edge_activation="parent_softmax_with_null",
        edge_threshold=0.15,
        max_detections_per_frame=1024,
        decoder=DecoderProfile(1, 2, 0.25, 0.75),
    )
    write_inference_profile(profile_path, profile)
    provenance = {
        "completed_epoch": 30,
        "checkpoint_sha256": checkpoint_sha,
        "inference_profile_sha256": profile.sha256,
        "experiment_config_sha256": packager.sha256_file(config_path),
        "manifest_sha256": split_sha,
        "selection_sha256": "e" * 64,
        "report_summary_sha256": "f" * 64,
        "report_config_sha256": "1" * 64,
        "candidate_epochs": list(range(5, 51, 5)),
        "report_score": 0.7,
        "report_score_baseline": 0.5688260117,
        "report_score_tolerance": 0.0,
        "report_score_gate_exclusive": 0.5688260117,
    }

    monkeypatch.setattr(packager, "COMPETITION_ROOT", competition_root)
    monkeypatch.setattr(packager, "load_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        packager.subprocess,
        "check_output",
        lambda *_args, **_kwargs: config["source"]["organizer_revision"] + "\n",
    )
    monkeypatch.setitem(
        sys.modules,
        "dynamic_network_architectures",
        SimpleNamespace(__path__=[str(dynamic_root)]),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(save=lambda _state, path: path.write_bytes(b"packaged-state")),
    )
    output = competition_root / "data/kaggle-exp-0007a-epoch30"

    result = packager.prepare(
        config_path,
        checkpoint_path,
        output,
        "owner/biohub-exp-0007a-epoch30",
        "owner/biohub-exp-0007a-epoch30-submit",
        profile_path,
        selection_provenance=provenance,
        expected_completed_epoch=30,
    )

    dataset = output / "dataset"
    metadata = json.loads(
        (dataset / "checkpoint-metadata.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert result["completed_epochs"] == 30
    assert metadata["source_checkpoint_sha256"] == checkpoint_sha
    assert metadata["experiment_config_sha256"] == packager.sha256_file(config_path)
    assert metadata["validation_subset_manifest_sha256"] == split_sha
    assert manifest["completed_epochs"] == 30
    assert {
        "config.json",
        "inference_profile.json",
        "selection-provenance.json",
        "tracking_cellmot_models.zip",
        "backbone_ab.zip",
        "dynamic_network_architectures.zip",
    } <= manifest["files"].keys()

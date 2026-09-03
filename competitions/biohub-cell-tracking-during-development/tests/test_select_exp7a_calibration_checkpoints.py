"""Leakage guards for EXP-0007A multi-checkpoint calibration selection."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

COMPETITION_ROOT = Path(__file__).parents[1]
MODULE_PATH = COMPETITION_ROOT / "scripts/select_exp7a_calibration_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("select_exp7a_calibration", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
from backbone_ab.checkpointing import (  # noqa: E402
    DecoderProfile,
    InferenceProfile,
    load_inference_profile,
    write_inference_profile,
)
from backbone_ab.finalization import validate_selection_report_binding  # noqa: E402


def _template(name: str) -> str:
    return (COMPETITION_ROOT / "configs" / name).read_text(encoding="utf-8")


def _install_experiment_config(root: Path) -> Path:
    path = root / "configs/exp-0007a-corrected-spatial-50e.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            COMPETITION_ROOT
            / "configs/exp-0007a-corrected-spatial-50e.toml"
        ).read_bytes()
    )
    return path


def _install_templates(root: Path) -> tuple[Path, Path, str]:
    manifest = root / "artifacts/EXP-0007/validation_split.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibration": ["synthetic-calibration"],
                "report": ["synthetic-report"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha = module.sha256_file(manifest)
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    calibration = config_dir / "exp-0007a-calibration-screen-epoch50.toml"
    report = config_dir / "exp-0007a-report-screen-epoch50.toml"
    pinned_sha = (
        "2fcede60f353645de37768be3e735a332f552fa9253d51d1154564ba4ce5cc1c"
    )
    calibration.write_text(
        _template(calibration.name).replace(pinned_sha, manifest_sha),
        encoding="utf-8",
    )
    report.write_text(
        _template(report.name).replace(pinned_sha, manifest_sha),
        encoding="utf-8",
    )
    return calibration, report, manifest_sha


def _install_screen_config(root: Path) -> Path:
    path = root / "candidate-screen.toml"
    path.write_text("schema_version = 1\n", encoding="utf-8")
    return path


def _bind_screen_summary(summary: dict, screen_config: Path) -> dict:
    summary["config_path"] = str(screen_config)
    summary["screen_config_sha256"] = module.sha256_file(screen_config)
    return summary


def _write_bound_profile(
    path: Path,
    *,
    checkpoint: Path,
    experiment_config: Path,
    detection_threshold: float = 0.1,
    edge_threshold: float = 0.15,
    null_parent_threshold: float = 0.25,
    division_threshold: float = 0.75,
) -> InferenceProfile:
    with experiment_config.open("rb") as file:
        config = tomllib.load(file)
    inference = config["inference"]
    decoder = inference["decoder"]
    profile = InferenceProfile(
        experiment_id=config["experiment_id"],
        model_api="corrected_v2",
        checkpoint_sha256=module.sha256_file(checkpoint),
        experiment_config_sha256=module.sha256_file(experiment_config),
        source_revision=config["source"]["organizer_revision"],
        downsample=tuple(config["train"]["downsample"]),
        window_size=config["data"]["window_size"],
        detection_threshold=detection_threshold,
        detection_tta=inference["det_tta"],
        pool_kernel_um=inference["pool_kernel_um"],
        edge_activation=inference["edge_activation"],
        edge_threshold=edge_threshold,
        max_detections_per_frame=inference["max_detections_per_frame"],
        decoder=DecoderProfile(
            decoder["max_parents_per_node"],
            decoder["max_children_per_node"],
            null_parent_threshold,
            division_threshold,
        ),
        postprocess_profile=inference["postprocess_profile"],
    )
    write_inference_profile(path, profile)
    return profile


def _calibration_summary(
    *,
    checkpoint: Path,
    epoch: int,
    profile: InferenceProfile,
    manifest_sha: str,
    screen_config: Path,
    score: float = 0.7,
) -> dict:
    return _bind_screen_summary(
        {
            "completed_epochs": epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": module.sha256_file(checkpoint),
            "inference_profile_sha256": profile.sha256,
            "screen": {
                "dataset_subset": "calibration",
                "threshold_selection": "screen_sweep",
                "dataset_subset_manifest_sha256": manifest_sha,
            },
            "best_threshold_result": {
                "detection_threshold": profile.detection_threshold,
                "edge_threshold": profile.edge_threshold,
                "null_parent_threshold": profile.decoder.null_parent_threshold,
                "division_threshold": profile.decoder.division_threshold,
                "competition_score": score,
                "adjusted_edge_jaccard": score,
                "edge_jaccard": score,
                "node_recall": 0.95,
            },
        },
        screen_config,
    )


def test_candidate_configs_are_calibration_only_and_epoch_specific() -> None:
    rendered = module.render_calibration_config(
        _template("exp-0007a-calibration-screen-epoch50.toml"),
        epoch=35,
        checkpoint="artifacts/EXP-0007A/checkpoint_epoch_0035.pth",
        artifact_dir="artifacts/EXP-0007A/checkpoint-selection/epoch-0035",
    )
    config = tomllib.loads(rendered)

    assert config["data"]["dataset_subset"] == "calibration"
    assert config["model"]["checkpoint"].endswith("checkpoint_epoch_0035.pth")
    assert "inference_profile" not in config["model"]
    assert config["output"]["artifact_dir"].endswith("epoch-0035")


def test_renderers_reject_profile_reuse_and_report_sweeps() -> None:
    calibration_text = _template("exp-0007a-calibration-screen-epoch50.toml")
    calibration_text = calibration_text.replace(
        'checkpoint = "artifacts/EXP-0007A/checkpoint_epoch_0050.pth"',
        'checkpoint = "artifacts/EXP-0007A/checkpoint_epoch_0050.pth"\n'
        'inference_profile = "stale-profile.json"',
    )
    with pytest.raises(ValueError, match="not reuse a profile"):
        module.render_calibration_config(
            calibration_text,
            epoch=50,
            checkpoint="checkpoint.pth",
            artifact_dir="artifacts/screen",
        )

    report_text = _template("exp-0007a-report-screen-epoch50.toml").replace(
        "detection_tta = true",
        "detection_tta = true\nedge_thresholds = [0.1, 0.2]",
    )
    calibration = tomllib.loads(
        _template("exp-0007a-calibration-screen-epoch50.toml")
    )
    with pytest.raises(ValueError, match="must not contain threshold sweeps"):
        module.render_selected_report_config(
            report_text,
            epoch=50,
            checkpoint="checkpoint.pth",
            inference_profile="profile.json",
            artifact_dir="artifacts/report",
            calibration_manifest=calibration["data"]["dataset_subset_manifest"],
            calibration_manifest_sha256=calibration["data"][
                "dataset_subset_manifest_sha256"
            ],
        )


def test_selected_report_is_bound_to_one_checkpoint_and_profile() -> None:
    calibration = tomllib.loads(
        _template("exp-0007a-calibration-screen-epoch50.toml")
    )
    rendered = module.render_selected_report_config(
        _template("exp-0007a-report-screen-epoch50.toml"),
        epoch=25,
        checkpoint="artifacts/EXP-0007A/checkpoint_epoch_0025.pth",
        inference_profile=(
            "artifacts/EXP-0007A/checkpoint-selection/"
            "selected_inference_profile.json"
        ),
        artifact_dir=(
            "artifacts/EXP-0007A/checkpoint-selection/selected-report-epoch-0025"
        ),
        calibration_manifest=calibration["data"]["dataset_subset_manifest"],
        calibration_manifest_sha256=calibration["data"][
            "dataset_subset_manifest_sha256"
        ],
    )
    config = tomllib.loads(rendered)

    assert config["experiment_id"] == "EXP-0007A-selected-report-epoch0025"
    assert config["data"]["dataset_subset"] == "report"
    assert config["model"]["checkpoint"].endswith("checkpoint_epoch_0025.pth")
    assert config["model"]["inference_profile"].endswith(
        "selected_inference_profile.json"
    )
    assert "detection_thresholds" not in config["inference"]


def test_calibration_record_rejects_report_summary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"weights")
    experiment_config = _install_experiment_config(tmp_path)
    screen_config = _install_screen_config(tmp_path)
    summary = _bind_screen_summary(
        {
            "completed_epochs": 5,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": "a" * 64,
            "inference_profile_sha256": "b" * 64,
            "screen": {
                "dataset_subset": "report",
                "threshold_selection": "screen_sweep",
            },
            "best_threshold_result": {},
        },
        screen_config,
    )

    with pytest.raises(ValueError, match="calibration"):
        module.calibration_record(
            summary,
            expected_epoch=5,
            expected_checkpoint=checkpoint,
            expected_experiment_config=experiment_config,
            expected_manifest_sha256="c" * 64,
            expected_screen_config=screen_config,
            profile_path=tmp_path / "profile.json",
        )


def test_calibration_record_rejects_checkpoint_changed_after_screen(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"original-weights")
    experiment_config = _install_experiment_config(tmp_path)
    profile = _write_bound_profile(
        tmp_path / "profile.json",
        checkpoint=checkpoint,
        experiment_config=experiment_config,
    )
    screen_config = _install_screen_config(tmp_path)
    summary = _calibration_summary(
        checkpoint=checkpoint,
        epoch=5,
        profile=profile,
        manifest_sha="c" * 64,
        screen_config=screen_config,
    )
    checkpoint.write_bytes(b"replacement-weights")

    with pytest.raises(ValueError, match="checkpoint SHA"):
        module.calibration_record(
            summary,
            expected_epoch=5,
            expected_checkpoint=checkpoint,
            expected_experiment_config=experiment_config,
            expected_manifest_sha256="c" * 64,
            expected_screen_config=screen_config,
            profile_path=tmp_path / "profile.json",
        )


def test_calibration_record_rejects_tampered_profile_content(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"weights")
    experiment_config = _install_experiment_config(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile = _write_bound_profile(
        profile_path,
        checkpoint=checkpoint,
        experiment_config=experiment_config,
    )
    screen_config = _install_screen_config(tmp_path)
    summary = _calibration_summary(
        checkpoint=checkpoint,
        epoch=5,
        profile=profile,
        manifest_sha="c" * 64,
        screen_config=screen_config,
    )
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["profile"]["detection_threshold"] = 0.9
    profile_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="profile hash mismatch"):
        module.calibration_record(
            summary,
            expected_epoch=5,
            expected_checkpoint=checkpoint,
            expected_experiment_config=experiment_config,
            expected_manifest_sha256="c" * 64,
            expected_screen_config=screen_config,
            profile_path=profile_path,
        )


def test_calibration_record_rejects_stale_experiment_or_screen_config(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"weights")
    experiment_config = _install_experiment_config(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile = _write_bound_profile(
        profile_path,
        checkpoint=checkpoint,
        experiment_config=experiment_config,
    )
    screen_config = _install_screen_config(tmp_path)
    summary = _calibration_summary(
        checkpoint=checkpoint,
        epoch=5,
        profile=profile,
        manifest_sha="c" * 64,
        screen_config=screen_config,
    )

    experiment_config.write_text("changed-config\n", encoding="utf-8")
    with pytest.raises(ValueError, match="experiment config"):
        module.calibration_record(
            summary,
            expected_epoch=5,
            expected_checkpoint=checkpoint,
            expected_experiment_config=experiment_config,
            expected_manifest_sha256="c" * 64,
            expected_screen_config=screen_config,
            profile_path=profile_path,
        )

    experiment_config = _install_experiment_config(tmp_path)
    screen_config.write_text("schema_version = 1\nchanged = true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="screen config SHA"):
        module.calibration_record(
            summary,
            expected_epoch=5,
            expected_checkpoint=checkpoint,
            expected_experiment_config=experiment_config,
            expected_manifest_sha256="c" * 64,
            expected_screen_config=screen_config,
            profile_path=profile_path,
        )


def test_calibration_record_rejects_thresholds_from_another_profile(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"weights")
    experiment_config = _install_experiment_config(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile = _write_bound_profile(
        profile_path,
        checkpoint=checkpoint,
        experiment_config=experiment_config,
    )
    screen_config = _install_screen_config(tmp_path)
    summary = _calibration_summary(
        checkpoint=checkpoint,
        epoch=5,
        profile=profile,
        manifest_sha="c" * 64,
        screen_config=screen_config,
    )
    summary["best_threshold_result"]["edge_threshold"] = 0.99

    with pytest.raises(ValueError, match="edge_threshold"):
        module.calibration_record(
            summary,
            expected_epoch=5,
            expected_checkpoint=checkpoint,
            expected_experiment_config=experiment_config,
            expected_manifest_sha256="c" * 64,
            expected_screen_config=screen_config,
            profile_path=profile_path,
        )


def test_generate_only_writes_ten_configs_without_running_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration, report, _ = _install_templates(tmp_path)
    _install_experiment_config(tmp_path)
    called = False

    def fail_if_called(_: Path) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(module, "_run_screen", fail_if_called)
    monkeypatch.setattr(module, "COMPETITION_ROOT", tmp_path)
    result = module.run(
        calibration_template=calibration,
        report_template=report,
        output_root=tmp_path / "artifacts/EXP-0007A/checkpoint-selection",
        generate_only=True,
    )

    assert len(result["candidate_configs"]) == 10
    assert len(result["missing_checkpoints"]) == 10
    assert not called
    for path in result["candidate_configs"]:
        config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        assert config["data"]["dataset_subset"] == "calibration"


def test_generate_only_rejects_a_score_affecting_report_protocol_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration, report, _ = _install_templates(tmp_path)
    _install_experiment_config(tmp_path)
    report.write_text(
        report.read_text(encoding="utf-8").replace(
            "max_match_distance_um = 7.0", "max_match_distance_um = 70.0"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "COMPETITION_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_run_screen",
        lambda _path: pytest.fail("invalid protocol must fail before evaluation"),
    )

    with pytest.raises(ValueError, match="max_match_distance_um"):
        module.run(
            calibration_template=calibration,
            report_template=report,
            output_root=tmp_path / "artifacts/EXP-0007A/checkpoint-selection",
            generate_only=True,
        )


def test_runner_selects_on_calibration_then_validates_report_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration, report, manifest_sha = _install_templates(tmp_path)
    experiment_config = _install_experiment_config(tmp_path)
    monkeypatch.setattr(module, "COMPETITION_ROOT", tmp_path)
    artifact_root = tmp_path / "artifacts/EXP-0007A"
    output_root = artifact_root / "checkpoint-selection"
    for epoch in module.DEFAULT_EPOCHS:
        checkpoint = artifact_root / f"checkpoint_epoch_{epoch:04d}.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{epoch}".encode())

    subsets: list[str] = []

    def fake_screen(config_path: Path) -> None:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        subset = config["data"]["dataset_subset"]
        subsets.append(subset)
        output = tmp_path / config["output"]["artifact_dir"]
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = tmp_path / config["model"]["checkpoint"]
        epoch = int(checkpoint.stem.rsplit("_", 1)[1])
        if subset == "calibration":
            score = 0.9 if epoch == 30 else 0.5 + epoch / 1000
            profile = _write_bound_profile(
                output / "inference_profile.json",
                checkpoint=checkpoint,
                experiment_config=experiment_config,
            )
            summary = _calibration_summary(
                checkpoint=checkpoint,
                epoch=epoch,
                profile=profile,
                manifest_sha=manifest_sha,
                screen_config=config_path,
                score=score,
            )
        else:
            profile = load_inference_profile(
                tmp_path / config["model"]["inference_profile"]
            )
            summary = _bind_screen_summary(
                {
                    "completed_epochs": epoch,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": module.sha256_file(checkpoint),
                    "inference_profile_sha256": profile.sha256,
                    "screen": {
                        "dataset_subset": "report",
                        "threshold_selection": "fixed_profile",
                        "dataset_subset_manifest": config["data"][
                            "dataset_subset_manifest"
                        ],
                        "dataset_subset_manifest_sha256": manifest_sha,
                    },
                    "best_threshold_result": {
                        "detection_threshold": profile.detection_threshold,
                        "edge_threshold": profile.edge_threshold,
                        "null_parent_threshold": (
                            profile.decoder.null_parent_threshold
                        ),
                        "division_threshold": profile.decoder.division_threshold,
                        "competition_score": 0.7,
                    },
                },
                config_path,
            )
        (output / "summary.json").write_text(
            json.dumps(summary) + "\n", encoding="utf-8"
        )

    monkeypatch.setattr(module, "_run_screen", fake_screen)
    result = module.run(
        calibration_template=calibration,
        report_template=report,
        output_root=output_root,
    )

    assert result["selected"]["completed_epoch"] == 30
    assert result["candidate_epochs"] == list(range(5, 51, 5))
    assert result["report_used_for_selection"] is False
    assert subsets == ["calibration"] * 10 + ["report"]
    assert sum(record["selected"] for record in result["records"]) == 1
    report_summary = output_root / "selected-report-epoch-0030/summary.json"
    binding = validate_selection_report_binding(
        output_root / "selection.json",
        report_summary,
        competition_root=tmp_path,
    )
    assert binding.completed_epoch == 30
    assert binding.report_score == 0.7

    partial = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    partial["candidate_epochs"] = partial["candidate_epochs"][:-1]
    partial["records"] = partial["records"][:-1]
    partial_path = output_root / "partial-selection.json"
    partial_path.write_text(json.dumps(partial) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete checkpoint sweep"):
        validate_selection_report_binding(
            partial_path,
            report_summary,
            competition_root=tmp_path,
        )

    wrong = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    for record in wrong["records"]:
        record["selected"] = record["completed_epoch"] == 5
    wrong["selected"] = {
        key: value for key, value in wrong["records"][0].items() if key != "selected"
    }
    wrong_path = output_root / "wrong-selection.json"
    wrong_path.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="highest-ranked calibration record"):
        validate_selection_report_binding(
            wrong_path,
            report_summary,
            competition_root=tmp_path,
        )

    stale_report = json.loads(report_summary.read_text(encoding="utf-8"))
    stale_report["checkpoint_sha256"] = "f" * 64
    report_summary.write_text(json.dumps(stale_report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="report checkpoint SHA"):
        module.run(
            calibration_template=calibration,
            report_template=report,
            output_root=output_root,
        )
    assert subsets == ["calibration"] * 10 + ["report"]


def test_immutable_selection_file_rejects_changed_content(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    module._write_immutable(path, json.dumps({"selected": 5}) + "\n")

    with pytest.raises(FileExistsError, match="immutable"):
        module._write_immutable(path, json.dumps({"selected": 10}) + "\n")

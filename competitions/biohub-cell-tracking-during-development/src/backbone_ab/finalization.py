"""Fail-closed binding checks for EXP-0007 checkpoint finalization."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .checkpointing import load_inference_profile, sha256_file

THRESHOLD_SWEEP_KEYS = {
    "detection_thresholds",
    "edge_thresholds",
    "null_parent_thresholds",
    "division_thresholds",
}
COMPETITION_METRIC = "adjusted_edge_jaccard + 0.1 * division_jaccard"
SELECTION_SCORE_FIELDS = (
    "competition_score",
    "adjusted_edge_jaccard",
    "edge_jaccard",
    "node_recall",
)


@dataclass(frozen=True)
class FinalizationBinding:
    """Content-addressed paths selected without consulting report scores."""

    selection_path: Path
    report_summary_path: Path
    report_config_path: Path
    experiment_config_path: Path
    checkpoint_path: Path
    inference_profile_path: Path
    manifest_path: Path
    completed_epoch: int
    checkpoint_sha256: str
    inference_profile_sha256: str
    experiment_config_sha256: str
    manifest_sha256: str
    selection_sha256: str
    report_summary_sha256: str
    report_config_sha256: str
    report_score: float

    def provenance(
        self,
        *,
        report_score_baseline: float,
        report_score_tolerance: float,
    ) -> dict[str, Any]:
        """Return portable package provenance without local absolute paths."""
        values = asdict(self)
        for key in tuple(values):
            if key.endswith("_path"):
                values[key] = Path(values[key]).name
        values["candidate_epochs"] = list(range(5, 51, 5))
        values["report_score_baseline"] = float(report_score_baseline)
        values["report_score_tolerance"] = float(report_score_tolerance)
        values["report_score_gate_exclusive"] = float(
            report_score_baseline + report_score_tolerance
        )
        return values


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        value = tomllib.load(file)
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version in {path}")
    return value


def _workspace_file(root: Path, value: str | Path, *, label: str) -> Path:
    root = root.resolve(strict=True)
    path = Path(value)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes competition workspace: {candidate}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _same_path(actual: str | Path, expected: Path, root: Path, *, label: str) -> None:
    if _workspace_file(root, actual, label=label) != expected:
        raise ValueError(f"{label} does not match the selected path")


def _require_sha(value: object, expected: str, *, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} SHA-256 mismatch")


def _finite_score(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _selection_key(record: dict[str, Any], index: int) -> tuple[float, ...]:
    scores = tuple(
        _finite_score(record.get(field), label=f"selection record {index} {field}")
        for field in SELECTION_SCORE_FIELDS
    )
    epoch = _finite_score(
        record.get("completed_epoch"), label=f"selection record {index} completed_epoch"
    )
    return (*scores, -epoch)


def validate_profile_experiment_binding(
    profile: Any,
    experiment_config: dict[str, Any],
    *,
    expected_experiment_id: str = "EXP-0007A",
) -> None:
    """Reject profiles carrying non-threshold defaults from another experiment."""
    if profile.experiment_id != expected_experiment_id:
        raise ValueError("selected profile experiment_id mismatch")
    if profile.model_api != "corrected_v2":
        raise ValueError("selected profile must use corrected_v2 inference")
    if profile.source_revision != experiment_config["source"]["organizer_revision"]:
        raise ValueError("selected profile organizer revision mismatch")
    inference = experiment_config["inference"]
    decoder = inference["decoder"]
    expected = {
        "downsample": tuple(int(value) for value in experiment_config["train"]["downsample"]),
        "window_size": int(experiment_config["data"]["window_size"]),
        "detection_tta": bool(
            inference.get("det_tta", inference.get("detection_tta", False))
        ),
        "pool_kernel_um": float(inference["pool_kernel_um"]),
        "edge_activation": str(inference["edge_activation"]),
        "max_detections_per_frame": int(inference["max_detections_per_frame"]),
        "postprocess_profile": str(inference.get("postprocess_profile", "none")),
    }
    for field, expected_value in expected.items():
        if getattr(profile, field) != expected_value:
            raise ValueError(f"selected profile {field} does not match experiment config")
    expected_decoder = {
        "max_parents_per_node": int(decoder["max_parents_per_node"]),
        "max_children_per_node": int(decoder["max_children_per_node"]),
    }
    for field, expected_value in expected_decoder.items():
        if getattr(profile.decoder, field) != expected_value:
            raise ValueError(
                f"selected profile decoder.{field} does not match experiment config"
            )


def validate_screen_experiment_binding(
    screen_config: dict[str, Any],
    experiment_config: dict[str, Any],
    *,
    expected_subset: str,
) -> None:
    """Pin score-affecting screen settings to the tracked EXP-0007 protocol."""
    if expected_subset not in {"calibration", "report"}:
        raise ValueError(f"unsupported validation subset: {expected_subset}")
    screen_source = screen_config.get("source", {})
    experiment_source = experiment_config.get("source", {})
    for key in ("organizer_repository_path", "organizer_revision"):
        if screen_source.get(key) != experiment_source.get(key):
            raise ValueError(f"screen source.{key} does not match experiment config")

    screen_data = screen_config.get("data", {})
    experiment_data = experiment_config.get("data", {})
    expected_data = {
        "train_dir": experiment_data.get("train_dir"),
        "fold": experiment_data.get("fold"),
        "validation_group": experiment_data.get("validation_group"),
        "dataset_subset": expected_subset,
        "transition_strategy": "densest-plus-all-divisions",
    }
    for key, expected in expected_data.items():
        if screen_data.get(key) != expected:
            raise ValueError(f"screen data.{key} does not match validation protocol")

    expected_downsample = tuple(int(value) for value in experiment_config["train"]["downsample"])
    actual_downsample = tuple(
        int(value) for value in screen_config.get("model", {}).get("downsample", [])
    )
    if actual_downsample != expected_downsample:
        raise ValueError("screen model.downsample does not match experiment config")

    screen_inference = screen_config.get("inference", {})
    experiment_inference = experiment_config.get("inference", {})
    expected_tta = bool(
        experiment_inference.get(
            "det_tta", experiment_inference.get("detection_tta", False)
        )
    )
    if bool(screen_inference.get("detection_tta", False)) != expected_tta:
        raise ValueError("screen detection_tta does not match experiment config")

    evaluation = screen_config.get("evaluation", {})
    expected_evaluation = {
        "metric": COMPETITION_METRIC,
        "max_match_distance_um": 7.0,
        "estimated_total_allocation": "uniform-per-frame",
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"screen evaluation.{key} does not match validation protocol")


def validate_selection_report_binding(
    selection_path: Path,
    report_summary_path: Path,
    *,
    competition_root: Path,
    expected_candidate_epochs: tuple[int, ...] = tuple(range(5, 51, 5)),
    expected_experiment_id: str = "EXP-0007A",
) -> FinalizationBinding:
    """Bind a complete calibration selection to its single fixed-profile report."""
    root = competition_root.resolve(strict=True)
    selection_path = _workspace_file(
        root, selection_path, label="checkpoint selection"
    )
    report_summary_path = _workspace_file(
        root, report_summary_path, label="selected report summary"
    )
    selection = _load_json(selection_path)
    summary = _load_json(report_summary_path)

    if selection.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint selection schema")
    if selection.get("selection_basis") != "calibration_manifest_only":
        raise ValueError("selection must be based only on the calibration manifest")
    if selection.get("report_used_for_selection") is not False:
        raise ValueError("report data must not be used for checkpoint selection")
    candidate_epochs = tuple(int(value) for value in selection.get("candidate_epochs", []))
    if candidate_epochs != expected_candidate_epochs:
        raise ValueError(
            "finalization requires the complete checkpoint sweep: "
            f"expected {list(expected_candidate_epochs)}, got {list(candidate_epochs)}"
        )
    records = selection.get("records")
    if not isinstance(records, list) or len(records) != len(expected_candidate_epochs):
        raise ValueError("selection must contain one record per required checkpoint")
    record_epochs = [int(record.get("completed_epoch", -1)) for record in records]
    if tuple(record_epochs) != expected_candidate_epochs:
        raise ValueError("selection records do not match the required checkpoint epochs")
    selected_records = [record for record in records if record.get("selected") is True]
    if len(selected_records) != 1:
        raise ValueError("selection records must mark exactly one checkpoint")

    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selection is missing the selected checkpoint record")
    completed_epoch = int(selected.get("completed_epoch", 0))
    if completed_epoch not in expected_candidate_epochs:
        raise ValueError("selected checkpoint is outside the complete checkpoint sweep")
    marked = selected_records[0]
    ranked = max(
        enumerate(records),
        key=lambda item: _selection_key(item[1], item[0]),
    )[1]
    ranked_without_marker = {
        key: value for key, value in ranked.items() if key != "selected"
    }
    if selected != ranked_without_marker:
        raise ValueError("selected checkpoint is not the highest-ranked calibration record")
    for key in (
        "completed_epoch",
        "checkpoint",
        "checkpoint_sha256",
        "inference_profile",
        "inference_profile_sha256",
        "dataset_subset_manifest_sha256",
    ):
        if marked.get(key) != selected.get(key):
            raise ValueError(f"marked selection disagrees with selected.{key}")

    checkpoint_path = _workspace_file(
        root, selected["checkpoint"], label="selected checkpoint"
    )
    expected_name = f"checkpoint_epoch_{completed_epoch:04d}.pth"
    if checkpoint_path.name != expected_name:
        raise ValueError(
            f"selected checkpoint filename must be {expected_name}, got {checkpoint_path.name}"
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    _require_sha(
        selected.get("checkpoint_sha256"),
        checkpoint_sha256,
        label="selected checkpoint",
    )

    profile_path = _workspace_file(
        root,
        selection["selected_inference_profile"],
        label="selected inference profile",
    )
    if profile_path.name != "selected_inference_profile.json":
        raise ValueError("finalization requires the immutable selected profile copy")
    profile = load_inference_profile(profile_path)
    _require_sha(
        selected.get("inference_profile_sha256"),
        profile.sha256,
        label="selected inference profile",
    )
    if profile.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("selected inference profile is bound to another checkpoint")
    source_profile_path = _workspace_file(
        root,
        selected["inference_profile"],
        label="calibration-selected inference profile",
    )
    source_profile = load_inference_profile(source_profile_path)
    if source_profile.sha256 != profile.sha256:
        raise ValueError("selected profile copy differs from its calibration source")

    report_config_path = _workspace_file(
        root, selection["selected_report_config"], label="selected report config"
    )
    report_config = _load_toml(report_config_path)
    expected_report_id = f"EXP-0007A-selected-report-epoch{completed_epoch:04d}"
    if report_config.get("experiment_id") != expected_report_id:
        raise ValueError("selected report config experiment_id mismatch")
    if report_config["data"].get("dataset_subset") != "report":
        raise ValueError("selected report config must use the report subset")
    if THRESHOLD_SWEEP_KEYS & report_config.get("inference", {}).keys():
        raise ValueError("selected report config must not sweep thresholds")
    _same_path(
        report_config["model"]["checkpoint"],
        checkpoint_path,
        root,
        label="report checkpoint",
    )
    _same_path(
        report_config["model"]["inference_profile"],
        profile_path,
        root,
        label="report inference profile",
    )
    experiment_config_path = _workspace_file(
        root,
        report_config["model"]["experiment_config"],
        label="experiment config",
    )
    experiment_config = _load_toml(experiment_config_path)
    if experiment_config.get("experiment_id") != expected_experiment_id:
        raise ValueError("finalization experiment_id mismatch")
    if experiment_config.get("backbone", {}).get("contract") != "corrected_v2":
        raise ValueError("finalization requires the corrected_v2 model contract")
    if int(experiment_config.get("train", {}).get("epochs", 0)) != 50:
        raise ValueError("finalization requires the tracked 50-epoch experiment config")
    experiment_config_sha256 = sha256_file(experiment_config_path)
    if profile.experiment_config_sha256 != experiment_config_sha256:
        raise ValueError("selected profile is bound to another experiment config")
    validate_profile_experiment_binding(
        profile,
        experiment_config,
        expected_experiment_id=expected_experiment_id,
    )
    validate_screen_experiment_binding(
        report_config,
        experiment_config,
        expected_subset="report",
    )

    manifest_name = selection.get("calibration_manifest")
    manifest_sha256 = selection.get("calibration_manifest_sha256")
    if not isinstance(manifest_name, str) or not isinstance(manifest_sha256, str):
        raise ValueError("selection is missing split manifest provenance")
    if selection.get("report_manifest") != manifest_name:
        raise ValueError("calibration and report manifest paths differ")
    if selection.get("report_manifest_sha256") != manifest_sha256:
        raise ValueError("calibration and report manifest SHA-256 values differ")
    if selected.get("dataset_subset_manifest_sha256") != manifest_sha256:
        raise ValueError("selected calibration record split manifest SHA-256 mismatch")
    manifest_path = _workspace_file(root, manifest_name, label="split manifest")
    _require_sha(
        _content_sha256(manifest_path), manifest_sha256, label="split manifest"
    )
    if report_config["data"].get("dataset_subset_manifest") != manifest_name:
        raise ValueError("selected report config uses another split manifest")
    if (
        report_config["data"].get("dataset_subset_manifest_sha256")
        != manifest_sha256
    ):
        raise ValueError("selected report config uses another split manifest SHA-256")

    expected_summary = (
        root / report_config["output"]["artifact_dir"] / "summary.json"
    ).resolve(strict=True)
    if report_summary_path != expected_summary:
        raise ValueError("report summary is outside the selected report artifact")
    _same_path(
        summary["checkpoint"], checkpoint_path, root, label="report summary checkpoint"
    )
    if int(summary.get("completed_epochs", 0)) != completed_epoch:
        raise ValueError("report summary completed epoch does not match selection")
    _require_sha(
        summary.get("checkpoint_sha256"), checkpoint_sha256, label="report checkpoint"
    )
    _require_sha(
        summary.get("inference_profile_sha256"),
        profile.sha256,
        label="report inference profile",
    )
    _same_path(
        summary["config_path"],
        report_config_path,
        root,
        label="report summary config",
    )
    report_config_sha256 = sha256_file(report_config_path)
    _require_sha(
        summary.get("screen_config_sha256"),
        report_config_sha256,
        label="report screen config",
    )
    screen = summary.get("screen", {})
    if screen.get("dataset_subset") != "report":
        raise ValueError("report summary must use the report subset")
    if screen.get("threshold_selection") != "fixed_profile":
        raise ValueError("report summary must use one fixed inference profile")
    if screen.get("dataset_subset_manifest") != manifest_name:
        raise ValueError("report summary split manifest path mismatch")
    if screen.get("dataset_subset_manifest_sha256") != manifest_sha256:
        raise ValueError("report summary split manifest SHA-256 mismatch")
    best = summary.get("best_threshold_result")
    if not isinstance(best, dict):
        raise ValueError("report summary is missing fixed-profile metrics")
    expected_thresholds = {
        "detection_threshold": profile.detection_threshold,
        "edge_threshold": profile.edge_threshold,
        "null_parent_threshold": profile.decoder.null_parent_threshold,
        "division_threshold": profile.decoder.division_threshold,
    }
    for key, expected in expected_thresholds.items():
        if float(best.get(key, math.nan)) != float(expected):
            raise ValueError(f"report summary {key} does not match selected profile")
    report_score = _finite_score(
        best.get("competition_score"), label="report competition score"
    )

    return FinalizationBinding(
        selection_path=selection_path,
        report_summary_path=report_summary_path,
        report_config_path=report_config_path,
        experiment_config_path=experiment_config_path,
        checkpoint_path=checkpoint_path,
        inference_profile_path=profile_path,
        manifest_path=manifest_path,
        completed_epoch=completed_epoch,
        checkpoint_sha256=checkpoint_sha256,
        inference_profile_sha256=profile.sha256,
        experiment_config_sha256=experiment_config_sha256,
        manifest_sha256=manifest_sha256,
        selection_sha256=sha256_file(selection_path),
        report_summary_sha256=sha256_file(report_summary_path),
        report_config_sha256=report_config_sha256,
        report_score=report_score,
    )

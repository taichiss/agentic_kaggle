#!/usr/bin/env python
"""Select EXP-0007A epoch 5..50 using calibration data, then report once."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_ROOT = SCRIPT_DIR.parent
SOURCE_ROOT = COMPETITION_ROOT / "src"
DEFAULT_CALIBRATION_TEMPLATE = (
    COMPETITION_ROOT / "configs/exp-0007a-calibration-screen-epoch50.toml"
)
DEFAULT_REPORT_TEMPLATE = (
    COMPETITION_ROOT / "configs/exp-0007a-report-screen-epoch50.toml"
)
DEFAULT_OUTPUT_ROOT = (
    COMPETITION_ROOT / "artifacts/EXP-0007A/checkpoint-selection-calibration"
)
DEFAULT_EPOCHS = tuple(range(5, 51, 5))
EVALUATOR = SCRIPT_DIR / "evaluate_backbone_ab_competition_screen.py"

sys.path.insert(0, str(SOURCE_ROOT))
from backbone_ab.checkpointing import (  # noqa: E402
    load_inference_profile,
    sha256_file,
)
from backbone_ab.finalization import (  # noqa: E402
    validate_profile_experiment_binding,
    validate_screen_experiment_binding,
    validate_selection_report_binding,
)
from checkpoint_selection import (  # noqa: E402
    annotate_checkpoint_selection,
    select_best_checkpoint,
)


def _load_toml_text(text: str) -> dict[str, Any]:
    data = tomllib.loads(text)
    if data.get("schema_version") != 1:
        raise ValueError("screen template must use schema_version=1")
    return data


def _replace_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=\s*.*$")
    rendered, replacements = pattern.subn(f"{key} = {json.dumps(value)}", text)
    if replacements != 1:
        raise ValueError(f"expected exactly one {key!r} in screen template")
    return rendered


def render_calibration_config(
    template_text: str,
    *,
    epoch: int,
    checkpoint: str,
    artifact_dir: str,
) -> str:
    """Render one calibration-only checkpoint screen from the tracked template."""
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    template = _load_toml_text(template_text)
    if template["data"].get("dataset_subset") != "calibration":
        raise ValueError("calibration template must use the calibration subset")
    if "inference_profile" in template["model"]:
        raise ValueError("calibration template must sweep thresholds, not reuse a profile")
    rendered = _replace_scalar(
        template_text,
        "experiment_id",
        f"EXP-0007A-checkpoint-calibration-epoch{epoch:04d}",
    )
    rendered = _replace_scalar(rendered, "checkpoint", checkpoint)
    rendered = _replace_scalar(rendered, "artifact_dir", artifact_dir)
    parsed = _load_toml_text(rendered)
    if parsed["data"]["dataset_subset"] != "calibration":
        raise AssertionError("rendered checkpoint screen escaped calibration subset")
    return rendered


def render_selected_report_config(
    template_text: str,
    *,
    epoch: int,
    checkpoint: str,
    inference_profile: str,
    artifact_dir: str,
    calibration_manifest: str,
    calibration_manifest_sha256: str,
) -> str:
    """Render the only report screen, bound to the calibration-selected state."""
    template = _load_toml_text(template_text)
    if template["data"].get("dataset_subset") != "report":
        raise ValueError("report template must use the report subset")
    if not template["model"].get("inference_profile"):
        raise ValueError("report template must declare one fixed inference_profile")
    sweep_keys = {
        "detection_thresholds",
        "edge_thresholds",
        "null_parent_thresholds",
        "division_thresholds",
    }
    if sweep_keys & template["inference"].keys():
        raise ValueError("report template must not contain threshold sweeps")
    if template["data"].get("dataset_subset_manifest") != calibration_manifest:
        raise ValueError("calibration and report templates must share one split manifest")
    if (
        template["data"].get("dataset_subset_manifest_sha256")
        != calibration_manifest_sha256
    ):
        raise ValueError("calibration and report templates must share one manifest SHA-256")
    rendered = _replace_scalar(
        template_text,
        "experiment_id",
        f"EXP-0007A-selected-report-epoch{epoch:04d}",
    )
    rendered = _replace_scalar(rendered, "checkpoint", checkpoint)
    rendered = _replace_scalar(rendered, "inference_profile", inference_profile)
    rendered = _replace_scalar(rendered, "artifact_dir", artifact_dir)
    parsed = _load_toml_text(rendered)
    if parsed["data"]["dataset_subset"] != "report":
        raise AssertionError("rendered final screen escaped report subset")
    return rendered


def calibration_record(
    summary: dict[str, Any],
    *,
    expected_epoch: int,
    expected_checkpoint: Path,
    expected_experiment_config: Path,
    expected_manifest_sha256: str,
    expected_screen_config: Path,
    profile_path: Path,
) -> dict[str, Any]:
    """Validate one calibration result and convert it to a selection record."""
    if summary.get("completed_epochs") != expected_epoch:
        raise ValueError(
            f"checkpoint epoch mismatch: expected {expected_epoch}, "
            f"got {summary.get('completed_epochs')}"
        )
    if Path(summary["checkpoint"]).resolve() != expected_checkpoint.resolve():
        raise ValueError("calibration summary references the wrong checkpoint")
    screen = summary.get("screen", {})
    if screen.get("dataset_subset") != "calibration":
        raise ValueError("checkpoint selection may only consume calibration summaries")
    if screen.get("threshold_selection") != "screen_sweep":
        raise ValueError("each candidate must calibrate its own threshold profile")
    if screen.get("dataset_subset_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("calibration summary split manifest SHA-256 mismatch")
    if Path(summary.get("config_path", "")).resolve() != expected_screen_config.resolve():
        raise ValueError("calibration summary references the wrong screen config")
    if summary.get("screen_config_sha256") != sha256_file(expected_screen_config):
        raise ValueError("calibration summary screen config SHA-256 mismatch")
    actual_checkpoint_sha256 = sha256_file(expected_checkpoint)
    if summary.get("checkpoint_sha256") != actual_checkpoint_sha256:
        raise ValueError("calibration summary checkpoint SHA does not match checkpoint bytes")
    profile = load_inference_profile(profile_path)
    if summary.get("inference_profile_sha256") != profile.sha256:
        raise ValueError("calibration summary profile SHA does not match profile content")
    if profile.checkpoint_sha256 != actual_checkpoint_sha256:
        raise ValueError("inference profile is bound to a different checkpoint")
    actual_config_sha256 = sha256_file(expected_experiment_config)
    if profile.experiment_config_sha256 != actual_config_sha256:
        raise ValueError("inference profile is bound to a different experiment config")
    with expected_experiment_config.open("rb") as file:
        experiment_config = tomllib.load(file)
    validate_profile_experiment_binding(profile, experiment_config)
    best = summary.get("best_threshold_result")
    if not isinstance(best, dict):
        raise ValueError("calibration summary is missing best threshold metrics")
    expected_thresholds = {
        "detection_threshold": profile.detection_threshold,
        "edge_threshold": profile.edge_threshold,
        "null_parent_threshold": profile.decoder.null_parent_threshold,
        "division_threshold": profile.decoder.division_threshold,
    }
    for key, expected in expected_thresholds.items():
        if float(best.get(key, float("nan"))) != float(expected):
            raise ValueError(f"calibration summary {key} does not match its profile")
    return {
        "completed_epoch": expected_epoch,
        "checkpoint": str(expected_checkpoint),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "inference_profile": str(profile_path),
        "inference_profile_sha256": summary["inference_profile_sha256"],
        "dataset_subset_manifest_sha256": expected_manifest_sha256,
        **best,
    }


def _write_immutable(path: Path, content: str | bytes) -> None:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace immutable selection file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _relative_to_competition(path: Path) -> str:
    try:
        return path.resolve().relative_to(COMPETITION_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside the competition workspace: {path}") from exc


def _run_screen(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(EVALUATOR), "--config", str(config_path)],
        cwd=COMPETITION_ROOT,
        check=True,
    )


def run(
    *,
    calibration_template: Path = DEFAULT_CALIBRATION_TEMPLATE,
    report_template: Path = DEFAULT_REPORT_TEMPLATE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    epochs: Sequence[int] = DEFAULT_EPOCHS,
    generate_only: bool = False,
    run_report: bool = True,
) -> dict[str, Any]:
    """Run all calibration candidates and evaluate report only after selection."""
    epochs = tuple(int(epoch) for epoch in epochs)
    if not epochs or len(set(epochs)) != len(epochs) or any(epoch <= 0 for epoch in epochs):
        raise ValueError("epochs must be unique positive integers")
    calibration_text = calibration_template.read_text(encoding="utf-8")
    report_text = report_template.read_text(encoding="utf-8")
    calibration_data = _load_toml_text(calibration_text)
    report_data = _load_toml_text(report_text)
    calibration_manifest = calibration_data["data"]["dataset_subset_manifest"]
    if report_data["data"].get("dataset_subset_manifest") != calibration_manifest:
        raise ValueError("calibration and report templates must share one split manifest")
    calibration_manifest_sha256 = calibration_data["data"].get(
        "dataset_subset_manifest_sha256"
    )
    if (
        not isinstance(calibration_manifest_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", calibration_manifest_sha256)
    ):
        raise ValueError("calibration template must pin a lowercase manifest SHA-256")
    if (
        report_data["data"].get("dataset_subset_manifest_sha256")
        != calibration_manifest_sha256
    ):
        raise ValueError("calibration and report templates must share one manifest SHA-256")
    actual_manifest_sha256 = sha256_file(COMPETITION_ROOT / calibration_manifest)
    if actual_manifest_sha256 != calibration_manifest_sha256:
        raise ValueError("split manifest bytes do not match the pinned SHA-256")
    experiment_config = (
        COMPETITION_ROOT / calibration_data["model"]["experiment_config"]
    )
    if not experiment_config.is_file():
        raise FileNotFoundError(f"experiment config is missing: {experiment_config}")
    if (
        report_data.get("model", {}).get("experiment_config")
        != calibration_data.get("model", {}).get("experiment_config")
    ):
        raise ValueError("calibration and report templates must use one experiment config")
    with experiment_config.open("rb") as file:
        experiment_data = tomllib.load(file)
    validate_screen_experiment_binding(
        calibration_data,
        experiment_data,
        expected_subset="calibration",
    )
    validate_screen_experiment_binding(
        report_data,
        experiment_data,
        expected_subset="report",
    )

    candidates = []
    missing_checkpoints = []
    for epoch in epochs:
        checkpoint = COMPETITION_ROOT / (
            f"artifacts/EXP-0007A/checkpoint_epoch_{epoch:04d}.pth"
        )
        candidate_dir = output_root / f"epoch-{epoch:04d}"
        candidate_config = candidate_dir / "candidate-screen.toml"
        rendered = render_calibration_config(
            calibration_text,
            epoch=epoch,
            checkpoint=_relative_to_competition(checkpoint),
            artifact_dir=_relative_to_competition(candidate_dir),
        )
        _write_immutable(candidate_config, rendered)
        candidates.append((epoch, checkpoint, candidate_dir, candidate_config))
        if not checkpoint.is_file():
            missing_checkpoints.append(checkpoint)

    if generate_only:
        return {
            "mode": "generate_only",
            "candidate_configs": [str(item[3]) for item in candidates],
            "missing_checkpoints": [str(path) for path in missing_checkpoints],
        }
    if missing_checkpoints:
        rendered_missing = "\n".join(str(path) for path in missing_checkpoints)
        raise FileNotFoundError(
            "all calibration checkpoints must exist before selection:\n"
            f"{rendered_missing}"
        )

    records = []
    for epoch, checkpoint, candidate_dir, candidate_config in candidates:
        summary_path = candidate_dir / "summary.json"
        profile_path = candidate_dir / "inference_profile.json"
        if not summary_path.is_file() or not profile_path.is_file():
            _run_screen(candidate_config)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records.append(
            calibration_record(
                summary,
                expected_epoch=epoch,
                expected_checkpoint=checkpoint,
                expected_experiment_config=experiment_config,
                expected_manifest_sha256=calibration_manifest_sha256,
                expected_screen_config=candidate_config,
                profile_path=profile_path,
            )
        )

    annotated = annotate_checkpoint_selection(records)
    selected = dict(select_best_checkpoint(records))
    selected_epoch = int(selected["completed_epoch"])
    selected_profile_source = Path(selected["inference_profile"])
    selected_profile = output_root / "selected_inference_profile.json"
    _write_immutable(selected_profile, selected_profile_source.read_bytes())
    copied_profile = load_inference_profile(selected_profile)
    if copied_profile.sha256 != selected["inference_profile_sha256"]:
        raise ValueError("copied selected profile does not match the selected profile SHA")
    if copied_profile.checkpoint_sha256 != selected["checkpoint_sha256"]:
        raise ValueError("copied selected profile is bound to a different checkpoint")

    report_artifact = output_root / f"selected-report-epoch-{selected_epoch:04d}"
    report_config = output_root / "selected-report-screen.toml"
    report_rendered = render_selected_report_config(
        report_text,
        epoch=selected_epoch,
        checkpoint=_relative_to_competition(Path(selected["checkpoint"])),
        inference_profile=_relative_to_competition(selected_profile),
        artifact_dir=_relative_to_competition(report_artifact),
        calibration_manifest=calibration_manifest,
        calibration_manifest_sha256=calibration_manifest_sha256,
    )
    _write_immutable(report_config, report_rendered)

    result = {
        "schema_version": 1,
        "selection_basis": "calibration_manifest_only",
        "calibration_manifest": calibration_manifest,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "report_manifest": report_data["data"]["dataset_subset_manifest"],
        "report_manifest_sha256": report_data["data"][
            "dataset_subset_manifest_sha256"
        ],
        "candidate_epochs": list(epochs),
        "records": annotated,
        "selected": selected,
        "selected_inference_profile": str(selected_profile),
        "selected_report_config": str(report_config),
        "report_used_for_selection": False,
    }
    _write_immutable(
        output_root / "selection.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )

    if run_report:
        report_summary = report_artifact / "summary.json"
        if not report_summary.is_file():
            _run_screen(report_config)
        validate_selection_report_binding(
            output_root / "selection.json",
            report_summary,
            competition_root=COMPETITION_ROOT,
            expected_candidate_epochs=epochs,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-template", type=Path, default=DEFAULT_CALIBRATION_TEMPLATE)
    parser.add_argument("--report-template", type=Path, default=DEFAULT_REPORT_TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, nargs="+", default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="write candidate configs without requiring checkpoints or using the GPU",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="select and freeze the checkpoint/profile without evaluating report",
    )
    args = parser.parse_args()
    result = run(
        calibration_template=args.calibration_template.resolve(),
        report_template=args.report_template.resolve(),
        output_root=args.output_root.resolve(),
        epochs=args.epochs,
        generate_only=args.generate_only,
        run_report=not args.skip_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

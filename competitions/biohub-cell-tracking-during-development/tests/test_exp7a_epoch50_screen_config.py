"""Static checks for the EXP-0007 calibration/report flow."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

COMPETITION_ROOT = Path(__file__).parents[1]
CONFIG_ROOT = COMPETITION_ROOT / "configs"
SPLIT_SHA256 = "2fcede60f353645de37768be3e735a332f552fa9253d51d1154564ba4ce5cc1c"


def _load(name: str) -> dict:
    with (CONFIG_ROOT / name).open("rb") as file:
        return tomllib.load(file)


def test_epoch50_screen_uses_its_own_checkpoint_and_artifacts() -> None:
    calibration = _load("exp-0007a-calibration-screen-epoch50.toml")
    report = _load("exp-0007a-report-screen-epoch50.toml")

    expected_experiment = "configs/exp-0007a-corrected-spatial-50e.toml"
    expected_checkpoint = "artifacts/EXP-0007A/checkpoint_epoch_0050.pth"
    for config in (calibration, report):
        assert config["model"]["experiment_config"] == expected_experiment
        assert config["model"]["checkpoint"] == expected_checkpoint
        assert config["data"]["dataset_subset_manifest"] == (
            "artifacts/EXP-0007/validation_split.json"
        )

    assert calibration["data"]["dataset_subset"] == "calibration"
    assert report["data"]["dataset_subset"] == "report"
    assert calibration["output"]["artifact_dir"].endswith(
        "calibration-screen-epoch50"
    )
    assert report["output"]["artifact_dir"].endswith("report-screen-epoch50")


def test_epoch50_report_consumes_only_the_calibrated_immutable_profile() -> None:
    epoch5 = _load("exp-0007a-calibration-screen.toml")
    calibration = _load("exp-0007a-calibration-screen-epoch50.toml")
    report = _load("exp-0007a-report-screen-epoch50.toml")

    for key in (
        "detection_thresholds",
        "edge_thresholds",
        "null_parent_thresholds",
        "division_thresholds",
    ):
        assert calibration["inference"][key] == epoch5["inference"][key]
        assert key not in report["inference"]
    assert report["model"]["inference_profile"] == (
        "artifacts/EXP-0007A/calibration-screen-epoch50/inference_profile.json"
    )


def test_all_exp7_screens_pin_the_recovered_split_manifest_hash() -> None:
    screens = sorted(CONFIG_ROOT.glob("exp-0007*screen*.toml"))

    assert screens
    for path in screens:
        with path.open("rb") as file:
            config = tomllib.load(file)
        assert config["data"]["dataset_subset_manifest_sha256"] == SPLIT_SHA256


def test_backbone_reports_are_fixed_and_calibrations_are_sweeps() -> None:
    for path in sorted(CONFIG_ROOT.glob("exp-0007[abc]-*screen*.toml")):
        with path.open("rb") as file:
            config = tomllib.load(file)
        subset = config["data"]["dataset_subset"]
        if subset == "calibration":
            assert "inference_profile" not in config["model"]
        else:
            assert config["model"]["inference_profile"]
            assert not {
                "detection_thresholds",
                "edge_thresholds",
                "null_parent_thresholds",
                "division_thresholds",
            } & config["inference"].keys()


def test_epoch50_training_config_changes_only_the_continuation_target() -> None:
    epoch5 = json.loads(json.dumps(_load("exp-0007a-corrected-spatial.toml")))
    epoch50 = json.loads(json.dumps(_load("exp-0007a-corrected-spatial-50e.toml")))

    assert epoch5["train"].pop("epochs") == 5
    assert epoch50["train"].pop("epochs") == 50
    assert epoch50 == epoch5

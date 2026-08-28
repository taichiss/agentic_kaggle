"""Safety tests for the legacy-only detection calibration helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

COMPETITION_ROOT = Path(__file__).parents[1]
MODULE_PATH = COMPETITION_ROOT / "scripts/calibrate_backbone_ab_detection.py"
SPEC = importlib.util.spec_from_file_location("legacy_detection_calibration", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_corrected_v2_is_redirected_to_non_leaking_screen() -> None:
    config = COMPETITION_ROOT / "configs/exp-0007a-corrected-spatial.toml"

    with pytest.raises(ValueError, match="calibration/report split"):
        module.run(config, Path("unused-checkpoint.pth"), [0.1])

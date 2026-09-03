"""Lightweight tests for organizer-log to W&B metric parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/run_host_baseline_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_host_baseline_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_epoch_history_parses_organizer_output() -> None:
    module = _load_script()
    stdout = (
        "  Epoch   4/5 | edge=0.0153 | det=0.4415 | test_loss=0.0175 | "
        "acc=0.9962 | recall=0.3148 | best=0.3136 * | train=0.4s test=0.4s\n"
    )

    assert module._epoch_history(stdout) == [
        {
            "epoch": 4,
            "train/edge_loss": 0.0153,
            "train/detection_loss": 0.4415,
            "validation/loss": 0.0175,
            "validation/accuracy": 0.9962,
            "validation/node_recall": 0.3148,
            "validation/best_acc_recall": 0.3136,
            "runtime/epoch_train_seconds": 0.4,
            "runtime/epoch_validation_seconds": 0.4,
        }
    ]

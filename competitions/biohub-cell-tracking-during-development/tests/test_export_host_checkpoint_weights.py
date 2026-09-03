"""Focused tests for periodic host checkpoint export."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).parents[1] / "scripts/export_host_checkpoint_weights.py"
SPEC = importlib.util.spec_from_file_location("export_host_checkpoint_weights_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def _config(path: Path) -> None:
    path.write_text(
        json.dumps({"unet_out_channels": 2, "window_size": 2}),
        encoding="utf-8",
    )


def test_exports_ema_state_dict_with_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ema_checkpoint_epoch_0030.pth"
    model_config = tmp_path / "config.json"
    output = tmp_path / "exported"
    state = {"layer.weight": torch.arange(4, dtype=torch.float32)}
    torch.save(
        {
            "completed_epochs": 30,
            "model_state_dict": state,
            "ema_state_dict": {"num_updates": 10},
            "ema_validation": {"acc_recall": 0.9},
        },
        checkpoint,
    )
    _config(model_config)

    metadata = exporter.export_checkpoint(
        checkpoint,
        model_config,
        output,
        expected_completed_epoch=30,
        weight_view="ema",
    )

    restored = torch.load(
        output / "edge_predictor_best.pth", map_location="cpu", weights_only=True
    )
    assert torch.equal(restored["layer.weight"], state["layer.weight"])
    assert metadata["weight_view"] == "ema"
    assert metadata["source_checkpoint"] == checkpoint.name
    assert metadata["weights_sha256"] == exporter._sha256(
        output / "edge_predictor_best.pth"
    )


def test_rejects_mislabeled_weight_view(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_epoch_0030.pth"
    model_config = tmp_path / "config.json"
    torch.save(
        {
            "completed_epochs": 30,
            "model_state_dict": {"layer.weight": torch.ones(1)},
        },
        checkpoint,
    )
    _config(model_config)

    with pytest.raises(ValueError, match="EMA checkpoint wrapper"):
        exporter.export_checkpoint(
            checkpoint,
            model_config,
            tmp_path / "exported",
            expected_completed_epoch=30,
            weight_view="ema",
        )

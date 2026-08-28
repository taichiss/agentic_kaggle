"""Focused tests for optional T_graph=3 submission inference wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

COMPETITION_ROOT = Path(__file__).parents[1]
SCRIPT = COMPETITION_ROOT / "scripts/run_kaggle_inference.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_kaggle_inference_graph_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_graph_checkpoint_preserves_base_and_first_transition() -> None:
    module = _load_script()
    base = torch.tensor([[[1.0], [2.0]]])
    current = SimpleNamespace(edge_logits=base)

    assert module._owned_transition_logits(None, object(), current) is base

    class FailingHead:
        def __call__(self, *_args):
            raise AssertionError("the first transition must use frozen host logits")

    assert module._owned_transition_logits(FailingHead(), None, current) is base


def test_graph_head_owns_only_complete_right_transition() -> None:
    module = _load_script()
    base = torch.tensor([[[1.0]]])
    refined = torch.tensor([[[4.0]]])
    previous = object()
    current = SimpleNamespace(edge_logits=base)
    calls: list[tuple[object, object]] = []

    class RecordingHead:
        def __call__(self, left, right):
            calls.append((left, right))
            return SimpleNamespace(edge_logits=refined)

    assert module._owned_transition_logits(RecordingHead(), previous, current) is refined
    assert calls == [(previous, current)]


def test_pair_stream_reuses_middle_nodes_and_keeps_legacy_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    sys.path.insert(0, str(COMPETITION_ROOT / "src"))
    monkeypatch.setattr(
        module,
        "_zarr_metadata",
        lambda _path: {
            "shape": (3, 1, 4, 8),
            "dtype": np.dtype("uint16"),
            "scale": (2.0, 1.0, 0.5),
            "q_low": 0.0,
            "q_high": 2.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_load_frame",
        lambda _dataset, frame, _shape, _dtype, _downsample: torch.full(
            (1, 1, 2), float(frame)
        ),
    )
    monkeypatch.setattr(
        module,
        "_detect",
        lambda _logits, frame, _threshold, _kernel: np.asarray(
            [[frame, 0, 0, frame % 2]], dtype=np.int16
        ),
    )

    class FrozenHost:
        def encode(self, images):
            maps = images.unsqueeze(2).repeat(1, 1, 2, 1, 1, 1)
            logits = [torch.zeros(1, 1, 1, 1, 2) for _ in range(2)]
            return maps, logits

        def index_features(self, feature_maps, _coords, _mask):
            return feature_maps.mean(dim=(-3, -2, -1)).unsqueeze(1)

        def predict_edges(
            self,
            source_features,
            target_features,
            *_args,
        ):
            return torch.zeros(
                1,
                source_features.shape[1],
                target_features.shape[1],
            )

    calls = []

    class InspectingGraphHead:
        def __call__(self, previous, current):
            torch.testing.assert_close(
                previous.target_coords_um,
                current.source_coords_um,
            )
            torch.testing.assert_close(
                current.source_coords_um[0, 0],
                torch.tensor([0.0, 0.0, 2.0]),
            )
            assert torch.equal(previous.target_mask, current.source_mask)
            calls.append((previous, current))
            return SimpleNamespace(edge_logits=current.edge_logits)

    arguments = (
        FrozenHost(),
        tmp_path,
        torch.device("cpu"),
        (1, 4, 4),
        2,
        0.99,
        0.5,
        5.0,
        False,
        3,
    )
    legacy = module.predict_dataset(*arguments)
    explicit_none = module.predict_dataset(*arguments, temporal_graph_head=None)
    with_graph = module.predict_dataset(
        *arguments,
        temporal_graph_head=InspectingGraphHead(),
    )

    np.testing.assert_array_equal(explicit_none[0], legacy[0])
    assert explicit_none[1] == legacy[1]
    np.testing.assert_array_equal(with_graph[0], legacy[0])
    assert with_graph[1] == legacy[1]
    assert len(calls) == 1


def test_checkpoint_resolution_is_opt_in_and_cli_has_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.delenv(module.TEMPORAL_GRAPH_ENV, raising=False)
    monkeypatch.delenv("TEMPORAL_GRAPH_CHECKPOINT", raising=False)

    assert module._resolve_temporal_graph_checkpoint(None, tmp_path, {}) is None
    configured = {"temporal_graph": {"checkpoint": "graph.pth"}}
    assert module._resolve_temporal_graph_checkpoint(None, tmp_path, configured) == (
        tmp_path / "graph.pth"
    )

    monkeypatch.setenv(module.TEMPORAL_GRAPH_ENV, "/env/graph.pth")
    assert module._resolve_temporal_graph_checkpoint(None, tmp_path, configured) == Path(
        "/env/graph.pth"
    )
    assert module._resolve_temporal_graph_checkpoint(
        Path("cli.pth"), tmp_path, configured
    ) == Path("cli.pth")


def test_graph_checkpoint_rejects_different_frozen_host(tmp_path: Path) -> None:
    module = _load_script()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "edge_predictor_best.pth").write_bytes(b"host-a")

    sys.path.insert(0, str(COMPETITION_ROOT / "src"))
    from temporal_graph import TemporalGraphResidualHead

    head = TemporalGraphResidualHead(node_feature_dim=2, hidden_dim=4, top_k=1)
    checkpoint = tmp_path / "graph.pth"
    torch.save(
        head.checkpoint_payload(base_checkpoint_sha256="0" * 64),
        checkpoint,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._load_temporal_graph_head(checkpoint, bundle, torch.device("cpu"))

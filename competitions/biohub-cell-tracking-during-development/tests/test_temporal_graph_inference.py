"""Focused tests for optional T_graph=3/4/5 submission inference wiring."""

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


def _graph_config(graph_window_size: int, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "node_feature_dim": 2,
        "top_k": 8,
        "radius_um": 15.0,
        "distance_scale_um": 10.0,
        "middle_coord_atol": 1.0e-4,
        "image_window_size": 2,
        "graph_window_size": graph_window_size,
        "ownership": "right_transition",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _synthetic_frozen_host_stream(
    module,
    monkeypatch: pytest.MonkeyPatch,
    *,
    frames: int,
    empty_frames: frozenset[int] = frozenset(),
):
    monkeypatch.setattr(
        module,
        "_zarr_metadata",
        lambda _path: {
            "shape": (frames, 1, 4, 8),
            "dtype": np.dtype("uint16"),
            "scale": (2.0, 1.0, 0.5),
            "q_low": 0.0,
            "q_high": 1.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_load_frame",
        lambda _dataset, frame, _shape, _dtype, _downsample: torch.full(
            (1, 1, 2), float(frame)
        ),
    )

    def detect(_logits, frame, _threshold, _kernel):
        if frame in empty_frames:
            return np.empty((0, 4), dtype=np.int16)
        return np.asarray([[frame, 0, 0, frame]], dtype=np.int16)

    monkeypatch.setattr(module, "_detect", detect)

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

    return FrozenHost()


def _pair_frames(pair) -> tuple[int, int]:
    source = round(float(pair.source_features[0, 0, 0]))
    target = round(float(pair.target_features[0, 0, 0]))
    return source, target


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


def test_t4_uses_t3_fallback_until_prior_pair_is_available() -> None:
    module = _load_script()
    base = torch.tensor([[[1.0]]])
    fallback_logits = torch.tensor([[[2.0]]])
    primary_logits = torch.tensor([[[3.0]]])
    prior, previous = object(), object()
    current = SimpleNamespace(edge_logits=base)
    calls: list[tuple[str, object, object, object | None]] = []

    class PrimaryHead:
        config = _graph_config(4)

        def __call__(self, left, right, *, prior_pair):
            calls.append(("primary", left, right, prior_pair))
            return SimpleNamespace(edge_logits=primary_logits)

    class FallbackHead:
        config = _graph_config(3)

        def __call__(self, left, right):
            calls.append(("fallback", left, right, None))
            return SimpleNamespace(edge_logits=fallback_logits)

    head = PrimaryHead()
    fallback = FallbackHead()
    assert (
        module._owned_transition_logits(
            head,
            previous,
            current,
            temporal_graph_fallback_head=fallback,
        )
        is fallback_logits
    )
    assert (
        module._owned_transition_logits(
            head,
            previous,
            current,
            prior_pair=prior,
            temporal_graph_fallback_head=fallback,
        )
        is primary_logits
    )
    assert calls == [
        ("fallback", previous, current, None),
        ("primary", previous, current, prior),
    ]


def test_t5_uses_t3_then_t4_fallback_before_primary() -> None:
    module = _load_script()
    base = torch.tensor([[[1.0]]])
    t3_logits = torch.tensor([[[2.0]]])
    t4_logits = torch.tensor([[[3.0]]])
    primary_logits = torch.tensor([[[4.0]]])
    oldest, prior, previous = object(), object(), object()
    current = SimpleNamespace(edge_logits=base)
    calls: list[tuple[str, object, object, object | None, object | None]] = []

    class PrimaryHead:
        config = _graph_config(5)

        def __call__(self, left, right, *, prior_pair, oldest_pair):
            calls.append(("primary", left, right, prior_pair, oldest_pair))
            return SimpleNamespace(edge_logits=primary_logits)

    class T3FallbackHead:
        config = _graph_config(3)

        def __call__(self, left, right):
            calls.append(("t3", left, right, None, None))
            return SimpleNamespace(edge_logits=t3_logits)

    class T4FallbackHead:
        config = _graph_config(4)

        def __call__(self, left, right, *, prior_pair):
            calls.append(("t4", left, right, prior_pair, None))
            return SimpleNamespace(edge_logits=t4_logits)

    head = PrimaryHead()
    t3_fallback = T3FallbackHead()
    t4_fallback = T4FallbackHead()
    assert (
        module._owned_transition_logits(
            head,
            previous,
            current,
            temporal_graph_fallback_head=t3_fallback,
            temporal_graph_t4_fallback_head=t4_fallback,
        )
        is t3_logits
    )
    assert (
        module._owned_transition_logits(
            head,
            previous,
            current,
            prior_pair=prior,
            temporal_graph_fallback_head=t3_fallback,
            temporal_graph_t4_fallback_head=t4_fallback,
        )
        is t4_logits
    )
    assert (
        module._owned_transition_logits(
            head,
            previous,
            current,
            prior_pair=prior,
            oldest_pair=oldest,
            temporal_graph_fallback_head=t3_fallback,
            temporal_graph_t4_fallback_head=t4_fallback,
        )
        is primary_logits
    )
    assert calls == [
        ("t3", previous, current, None, None),
        ("t4", previous, current, prior, None),
        ("primary", previous, current, prior, oldest),
    ]


def test_t4_fallback_candidate_contract_mismatch_fails_closed() -> None:
    module = _load_script()
    primary = SimpleNamespace(config=_graph_config(4))
    fallback = SimpleNamespace(config=_graph_config(3, top_k=4))

    with pytest.raises(ValueError, match="different candidate contracts: top_k"):
        module._validate_temporal_graph_fallback_contract(primary, fallback)


def test_t5_rejects_a_non_t4_second_fallback() -> None:
    module = _load_script()
    primary = SimpleNamespace(config=_graph_config(5))
    fallback = SimpleNamespace(config=_graph_config(3))

    with pytest.raises(ValueError, match="fallback must use T_graph=4"):
        module._validate_temporal_graph_fallback_contract(
            primary,
            fallback,
            primary_window_size=5,
            fallback_window_size=4,
        )


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


def test_t4_pair_stream_rolls_two_history_pairs_and_uses_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    sys.path.insert(0, str(COMPETITION_ROOT / "src"))
    host = _synthetic_frozen_host_stream(module, monkeypatch, frames=5)
    fallback_calls: list[tuple[tuple[int, int], tuple[int, int]]] = []
    primary_calls: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = []

    class PrimaryHead:
        config = _graph_config(4)

        def __call__(self, previous, current, *, prior_pair):
            primary_calls.append(
                (_pair_frames(prior_pair), _pair_frames(previous), _pair_frames(current))
            )
            return SimpleNamespace(edge_logits=current.edge_logits)

    class FallbackHead:
        config = _graph_config(3)

        def __call__(self, previous, current):
            fallback_calls.append((_pair_frames(previous), _pair_frames(current)))
            return SimpleNamespace(edge_logits=current.edge_logits)

    coords, edges = module.predict_dataset(
        host,
        tmp_path,
        torch.device("cpu"),
        (1, 4, 4),
        2,
        0.99,
        0.5,
        5.0,
        False,
        5,
        PrimaryHead(),
        FallbackHead(),
    )

    assert len(coords) == 5
    assert len(edges) == 4
    assert fallback_calls == [((0, 1), (1, 2))]
    assert primary_calls == [
        ((0, 1), (1, 2), (2, 3)),
        ((1, 2), (2, 3), (3, 4)),
    ]


def test_t5_pair_stream_rolls_three_history_pairs_and_both_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    sys.path.insert(0, str(COMPETITION_ROOT / "src"))
    host = _synthetic_frozen_host_stream(module, monkeypatch, frames=6)
    t3_calls: list[tuple[tuple[int, int], tuple[int, int]]] = []
    t4_calls: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = []
    primary_calls: list[
        tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ]
    ] = []

    class PrimaryHead:
        config = _graph_config(5)

        def __call__(self, previous, current, *, prior_pair, oldest_pair):
            primary_calls.append(
                (
                    _pair_frames(oldest_pair),
                    _pair_frames(prior_pair),
                    _pair_frames(previous),
                    _pair_frames(current),
                )
            )
            return SimpleNamespace(edge_logits=current.edge_logits)

    class T3FallbackHead:
        config = _graph_config(3)

        def __call__(self, previous, current):
            t3_calls.append((_pair_frames(previous), _pair_frames(current)))
            return SimpleNamespace(edge_logits=current.edge_logits)

    class T4FallbackHead:
        config = _graph_config(4)

        def __call__(self, previous, current, *, prior_pair):
            t4_calls.append(
                (
                    _pair_frames(prior_pair),
                    _pair_frames(previous),
                    _pair_frames(current),
                )
            )
            return SimpleNamespace(edge_logits=current.edge_logits)

    coords, edges = module.predict_dataset(
        host,
        tmp_path,
        torch.device("cpu"),
        (1, 4, 4),
        2,
        0.99,
        0.5,
        5.0,
        False,
        6,
        PrimaryHead(),
        T3FallbackHead(),
        T4FallbackHead(),
    )

    assert len(coords) == 6
    assert len(edges) == 5
    assert t3_calls == [((0, 1), (1, 2))]
    assert t4_calls == [((0, 1), (1, 2), (2, 3))]
    assert primary_calls == [
        ((0, 1), (1, 2), (2, 3), (3, 4)),
        ((1, 2), (2, 3), (3, 4), (4, 5)),
    ]


def test_t4_pair_stream_resets_history_after_an_empty_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    sys.path.insert(0, str(COMPETITION_ROOT / "src"))
    host = _synthetic_frozen_host_stream(
        module,
        monkeypatch,
        frames=6,
        empty_frames=frozenset({2}),
    )
    fallback_calls: list[tuple[tuple[int, int], tuple[int, int]]] = []
    primary_calls: list[object] = []

    class PrimaryHead:
        config = _graph_config(4)

        def __call__(self, previous, current, *, prior_pair):
            primary_calls.append((previous, current, prior_pair))
            return SimpleNamespace(edge_logits=current.edge_logits)

    class FallbackHead:
        config = _graph_config(3)

        def __call__(self, previous, current):
            fallback_calls.append((_pair_frames(previous), _pair_frames(current)))
            return SimpleNamespace(edge_logits=current.edge_logits)

    module.predict_dataset(
        host,
        tmp_path,
        torch.device("cpu"),
        (1, 4, 4),
        2,
        0.99,
        0.5,
        5.0,
        False,
        6,
        PrimaryHead(),
        FallbackHead(),
    )

    assert fallback_calls == [((3, 4), (4, 5))]
    assert primary_calls == []


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

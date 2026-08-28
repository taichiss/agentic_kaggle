"""Focused tests for the frozen-host temporal graph residual core."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

COMPETITION_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(COMPETITION_ROOT / "src"))

from temporal_graph import (  # noqa: E402
    FrozenPair,
    ParentCandidates,
    RightTransitionTriplet,
    TemporalGraphConfig,
    TemporalGraphResidualHead,
    build_candidate_features,
    build_parent_candidates,
    candidate_feature_dim,
    expected_previous_parent_statistics,
    refine_logits,
)


def _features(count: int, width: int = 2, *, offset: float = 0.0) -> torch.Tensor:
    values = torch.arange(count * width, dtype=torch.float32).reshape(1, count, width)
    return values + offset


def _pair(
    source_x: list[float],
    target_x: list[float],
    edge_logits: torch.Tensor,
    *,
    source_mask: list[bool] | None = None,
    target_mask: list[bool] | None = None,
    feature_dim: int = 2,
    source_offset: float = 0.0,
    target_offset: float = 10.0,
) -> FrozenPair:
    source_coords = torch.zeros(1, len(source_x), 3)
    target_coords = torch.zeros(1, len(target_x), 3)
    if source_x:
        source_coords[0, :, 2] = torch.tensor(source_x)
    if target_x:
        target_coords[0, :, 2] = torch.tensor(target_x)
    return FrozenPair(
        source_features=_features(len(source_x), feature_dim, offset=source_offset),
        target_features=_features(len(target_x), feature_dim, offset=target_offset),
        source_coords_um=source_coords,
        target_coords_um=target_coords,
        source_mask=torch.tensor(
            [source_mask if source_mask is not None else [True] * len(source_x)],
            dtype=torch.bool,
        ),
        target_mask=torch.tensor(
            [target_mask if target_mask is not None else [True] * len(target_x)],
            dtype=torch.bool,
        ),
        edge_logits=edge_logits,
    )


def test_parent_candidates_apply_radius_topk_and_padding() -> None:
    source = torch.tensor([[[0.0, 0.0, 0.0], [0, 0, 1], [0, 0, 3], [0, 0, 10]]])
    target = torch.tensor([[[0.0, 0.0, 1.8], [0, 0, 3]]])

    candidates = build_parent_candidates(
        source,
        target,
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[True, False]]),
        top_k=3,
        radius_um=1.3,
    )

    assert candidates.source_index.shape == (1, 2, 3)
    assert candidates.source_index[0, 0].tolist() == [1, 2, 0]
    assert candidates.valid_mask[0, 0].tolist() == [True, True, False]
    assert not candidates.valid_mask[0, 1].any()
    torch.testing.assert_close(
        candidates.distance_um[0, 0, :2], torch.tensor([0.8, 1.2])
    )
    dense = candidates.dense_mask()
    assert dense.shape == (1, 4, 2)
    assert torch.nonzero(dense[0, :, 0]).flatten().tolist() == [1, 2]


def test_expected_previous_position_entropy_and_empty_history_fallback() -> None:
    previous = _pair(
        [0.0, 2.0],
        [1.0, 5.0],
        torch.zeros(1, 2, 2),
        target_mask=[True, False],
    )

    statistics = expected_previous_parent_statistics(previous)

    torch.testing.assert_close(
        statistics.expected_position_um[0, 0], torch.tensor([0.0, 0.0, 1.0])
    )
    torch.testing.assert_close(
        statistics.entropy[0, 0, 0], torch.log(torch.tensor(2.0))
    )
    torch.testing.assert_close(
        statistics.expected_position_um[0, 1], previous.target_coords_um[0, 1]
    )
    assert statistics.has_history[0, 0, 0]
    assert not statistics.has_history[0, 1, 0]
    assert statistics.entropy[0, 1, 0] == 0


def test_candidate_features_use_right_transition_and_constant_velocity() -> None:
    previous = _pair(
        [0.0],
        [1.0],
        torch.zeros(1, 1, 1),
        feature_dim=1,
    )
    current = _pair(
        [1.0],
        [3.0],
        torch.tensor([[[2.0]]]),
        feature_dim=1,
        source_offset=20.0,
        target_offset=30.0,
    )
    candidates = build_parent_candidates(
        current.source_coords_um,
        current.target_coords_um,
        current.source_mask,
        current.target_mask,
        top_k=1,
        radius_um=5.0,
    )

    batch = build_candidate_features(
        previous,
        current,
        candidates,
        distance_scale_um=10.0,
    )

    assert batch.features.shape == (1, 1, 1, candidate_feature_dim(1))
    # Three one-channel appearance views precede displacement and velocity residual.
    torch.testing.assert_close(batch.features[0, 0, 0, 3:6], torch.tensor([0.0, 0.0, 0.2]))
    torch.testing.assert_close(batch.features[0, 0, 0, 6:9], torch.tensor([0.0, 0.0, 0.1]))
    assert batch.features[0, 0, 0, -3] == 2.0  # frozen host logit
    assert batch.features[0, 0, 0, -1] == 1.0  # has history


def test_triplet_contract_owns_right_transition_and_rejects_reordered_middle() -> None:
    previous = _pair([0.0], [1.0, 2.0], torch.zeros(1, 1, 2))
    current = _pair(
        [1.0, 2.0],
        [3.0],
        torch.zeros(1, 2, 1),
        source_offset=20.0,
    )
    triplet = RightTransitionTriplet(previous, current)

    assert triplet.frame_count == 3
    assert triplet.owned_transition is current

    reordered = _pair(
        [2.0, 1.0],
        [3.0],
        torch.zeros(1, 2, 1),
        source_offset=20.0,
    )
    with pytest.raises(ValueError, match="coordinates or node order"):
        RightTransitionTriplet(previous, reordered)


def test_zero_initialized_head_preserves_host_logits_and_detaches_host() -> None:
    previous = _pair([0.0], [1.0, 3.0], torch.zeros(1, 1, 2))
    base_logits = torch.tensor([[[3.0, -1.0], [0.5, 2.0]]], requires_grad=True)
    current = _pair(
        [1.0, 3.0],
        [2.0, 4.0],
        base_logits,
        source_offset=20.0,
    )
    model = TemporalGraphResidualHead(
        TemporalGraphConfig(
            node_feature_dim=2,
            hidden_dim=8,
            top_k=1,
            radius_um=10.0,
        )
    )

    output = model(previous, current)

    assert torch.equal(output.edge_logits, base_logits.detach())
    assert not output.candidate_residual.any()
    output.edge_logits.sum().backward()
    assert base_logits.grad is None
    assert model.residual_mlp[-1].weight.grad is not None
    assert model.residual_mlp[-1].bias.grad is not None


def test_refine_logits_changes_only_candidate_entries() -> None:
    base = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2).requires_grad_()
    candidates = ParentCandidates(
        source_index=torch.tensor([[[0], [1]]]),
        valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        distance_um=torch.ones(1, 2, 1),
        source_count=3,
    )
    residual = torch.tensor([[[5.0], [7.0]]], requires_grad=True)

    refined = refine_logits(base, candidates, residual)

    expected = base.detach().clone()
    expected[0, 0, 0] += 5.0
    expected[0, 1, 1] += 7.0
    torch.testing.assert_close(refined, expected)
    outside = ~candidates.dense_mask()
    assert torch.equal(refined[outside], base.detach()[outside])
    refined.sum().backward()
    assert base.grad is None
    torch.testing.assert_close(residual.grad, torch.ones_like(residual))


def test_empty_previous_and_empty_owned_transition_are_finite() -> None:
    previous = _pair([], [1.0, 2.0], torch.empty(1, 0, 2))
    current = _pair(
        [1.0, 2.0],
        [],
        torch.empty(1, 2, 0),
        source_offset=20.0,
    )
    model = TemporalGraphResidualHead(
        node_feature_dim=2,
        hidden_dim=4,
        top_k=3,
        radius_um=5.0,
    )

    output = model(previous, current)

    assert output.edge_logits.shape == (1, 2, 0)
    assert output.candidate_residual.shape == (1, 0, 3)
    assert torch.isfinite(output.edge_logits).all()
    statistics = output.candidate_features.previous_statistics
    torch.testing.assert_close(
        statistics.expected_position_um, previous.target_coords_um.float()
    )
    assert not statistics.has_history.any()
    assert not statistics.entropy.any()


def test_checkpoint_payload_is_plain_and_round_trips() -> None:
    model = TemporalGraphResidualHead(
        node_feature_dim=2,
        hidden_dim=4,
        top_k=2,
        radius_um=9.0,
    )
    with torch.no_grad():
        model.residual_mlp[-1].bias.fill_(1.25)
    payload = model.checkpoint_payload(
        base_checkpoint_sha256="host-checkpoint-sha",
        metadata={"experiment": "synthetic"},
    )

    assert isinstance(payload, dict)
    assert isinstance(payload["config"], dict)
    assert payload["base_checkpoint_sha256"] == "host-checkpoint-sha"
    assert all(value.device.type == "cpu" for value in payload["state_dict"].values())

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    restored = TemporalGraphResidualHead.from_checkpoint_payload(loaded)

    assert restored.config == model.config
    assert restored.config.image_window_size == 2
    assert restored.config.graph_window_size == 3
    assert restored.config.ownership == "right_transition"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)

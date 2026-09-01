"""Tests for controlled MLP/Attention temporal-link combinations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

COMPETITION_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(COMPETITION_ROOT / "src"))

from temporal_graph import (  # noqa: E402
    FrozenPair,
    TemporalGraphConfig,
    TemporalGraphLinkEnsemble,
)


class StubHead(nn.Module):
    def __init__(self, residual: torch.Tensor, *, architecture: str) -> None:
        super().__init__()
        self.config = TemporalGraphConfig(
            node_feature_dim=2,
            hidden_dim=8,
            top_k=2,
            radius_um=10.0,
            architecture=architecture,
            attention_heads=2,
            residual_logit_bound=(
                0.15 if architecture == "candidate_attention" else None
            ),
        )
        self.register_buffer("residual", residual)

    def forward_candidate_features(self, features) -> torch.Tensor:
        assert features.valid_mask.shape == self.residual.shape
        return self.residual * features.valid_mask.to(self.residual.dtype)


def _pair(
    source_x: list[float],
    target_x: list[float],
    logits: torch.Tensor,
) -> FrozenPair:
    source_coords = torch.zeros(1, len(source_x), 3)
    target_coords = torch.zeros(1, len(target_x), 3)
    source_coords[0, :, 2] = torch.tensor(source_x)
    target_coords[0, :, 2] = torch.tensor(target_x)
    return FrozenPair(
        source_features=torch.zeros(1, len(source_x), 2),
        target_features=torch.zeros(1, len(target_x), 2),
        source_coords_um=source_coords,
        target_coords_um=target_coords,
        source_mask=torch.ones(1, len(source_x), dtype=torch.bool),
        target_mask=torch.ones(1, len(target_x), dtype=torch.bool),
        edge_logits=logits,
    )


def _triplet() -> tuple[FrozenPair, FrozenPair]:
    previous = _pair([0.0], [0.0, 3.0], torch.zeros(1, 1, 2))
    current = _pair([0.0, 3.0], [1.0], torch.tensor([[[0.1], [0.0]]]))
    return previous, current


def test_single_head_modes_apply_only_the_selected_residual() -> None:
    previous, current = _triplet()
    mlp_residual = torch.tensor([[[-0.4, 0.4]]])
    attention_residual = torch.tensor([[[-0.1, 0.1]]])
    mlp = StubHead(mlp_residual, architecture="mlp")
    attention = StubHead(
        attention_residual,
        architecture="candidate_attention",
    )

    mlp_output = TemporalGraphLinkEnsemble(
        mlp,
        attention,
        mode="mlp",
    )(previous, current)
    attention_output = TemporalGraphLinkEnsemble(
        mlp,
        attention,
        mode="bounded_attention",
    )(previous, current)

    torch.testing.assert_close(mlp_output.candidate_residual, mlp_residual)
    torch.testing.assert_close(
        attention_output.candidate_residual,
        attention_residual,
    )


def test_5050_centers_and_bounds_the_combined_candidate_residual() -> None:
    previous, current = _triplet()
    mlp = StubHead(torch.tensor([[[4.0, -2.0]]]), architecture="mlp")
    attention = StubHead(
        torch.tensor([[[0.1, -0.1]]]),
        architecture="candidate_attention",
    )
    output = TemporalGraphLinkEnsemble(
        mlp,
        attention,
        mode="bounded_logit_5050",
        logit_bound=0.15,
    )(previous, current)

    assert float(output.candidate_residual.abs().max()) <= 0.15 + 1.0e-6
    torch.testing.assert_close(
        output.candidate_residual.sum(dim=-1),
        torch.zeros(1, 1),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_agreement_gate_uses_bounded_attention_only_on_shared_override() -> None:
    previous, current = _triplet()
    mlp = StubHead(torch.tensor([[[-2.0, 2.0]]]), architecture="mlp")
    attention_residual = torch.tensor([[[-0.15, 0.15]]])
    attention = StubHead(
        attention_residual,
        architecture="candidate_attention",
    )
    output = TemporalGraphLinkEnsemble(
        mlp,
        attention,
        mode="agreement_gate",
    )(previous, current)

    torch.testing.assert_close(output.candidate_residual, attention_residual)
    assert output.edge_logits.argmax(dim=1).item() == 1


def test_agreement_gate_preserves_host_when_heads_disagree() -> None:
    previous, current = _triplet()
    mlp = StubHead(torch.tensor([[[-2.0, 2.0]]]), architecture="mlp")
    attention = StubHead(
        torch.tensor([[[0.15, -0.15]]]),
        architecture="candidate_attention",
    )
    output = TemporalGraphLinkEnsemble(
        mlp,
        attention,
        mode="agreement_gate",
    )(previous, current)

    assert not output.candidate_residual.any()
    assert torch.equal(output.edge_logits, current.edge_logits)


def test_ensemble_rejects_candidate_contract_mismatch() -> None:
    mlp = StubHead(torch.zeros(1, 1, 2), architecture="mlp")
    attention = StubHead(
        torch.zeros(1, 1, 2),
        architecture="candidate_attention",
    )
    attention.config = TemporalGraphConfig(
        node_feature_dim=2,
        hidden_dim=8,
        top_k=1,
        radius_um=10.0,
        architecture="candidate_attention",
        attention_heads=2,
        residual_logit_bound=0.15,
    )

    with pytest.raises(ValueError, match="different candidate contracts"):
        TemporalGraphLinkEnsemble(mlp, attention, mode="agreement_gate")

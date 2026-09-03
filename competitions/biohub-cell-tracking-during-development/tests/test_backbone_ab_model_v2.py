"""Model-level tests for the corrected-v2 detector/linker contract."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

COMPETITION_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(COMPETITION_ROOT / "src"))

from backbone_ab.backbones import (  # noqa: E402
    CorrectedTrackingModel,
    FrameSharedBackbone,
    RelativeTimeTemporalAttention,
    build_backbone,
    build_joint_model,
)
from backbone_ab.contracts import EncodedWindow, LinkOutput  # noqa: E402


class _DummyBackbone(nn.Module):
    def __init__(self, feature_dim: int = 4) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.repeat(1, 1, self.feature_dim, 1, 1, 1)


class _DummyTransformer(nn.Module):
    def __init__(self, feat_dim: int, **_: object) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.projection = nn.Linear(feat_dim, 1)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        coords_source: torch.Tensor,
        coords_target: torch.Tensor,
        mask_source: torch.Tensor,
        mask_target: torch.Tensor,
    ) -> torch.Tensor:
        del coords_source, coords_target, mask_source, mask_target
        source_score = self.projection(source)
        target_score = self.projection(target).transpose(1, 2)
        return source_score + target_score


def _host_module() -> SimpleNamespace:
    return SimpleNamespace(SimpleNodeTransformer=_DummyTransformer, _POS_EMBED_DIM=8)


def _model(
    feature_dim: int = 4,
    *,
    radius_um: float | None = None,
    top_k: int | None = None,
) -> CorrectedTrackingModel:
    return CorrectedTrackingModel(
        _host_module(),
        _DummyBackbone(feature_dim),
        feature_dim,
        spatial_embedding_dim=8,
        link_candidate_radius_um=radius_um,
        link_candidate_top_k=top_k,
    )


def test_link_output_exposes_target_major_parent_choices() -> None:
    output = LinkOutput(
        edge_logits=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        null_parent_logits=torch.full((1, 4), -2.0),
        division_logits=torch.zeros(1, 3),
    )

    assert output.parent_logits.shape == (1, 4, 4)
    torch.testing.assert_close(output.parent_logits[..., -1], output.null_parent_logits)
    torch.testing.assert_close(
        output.parent_logits[..., :-1],
        output.edge_logits.transpose(1, 2),
    )


def test_relative_time_attention_starts_as_exact_identity() -> None:
    torch.manual_seed(3)
    attention = RelativeTimeTemporalAttention(channels=8, heads=2)
    features = torch.randn(2, 3, 8, 2, 2, 2)

    output = attention(features, torch.tensor([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]))

    torch.testing.assert_close(output, features)
    torch.testing.assert_close(attention.residual_gate, torch.zeros(8))
    output.sum().backward()
    assert attention.residual_gate.grad is not None


def test_build_nodes_recomputes_spatial_and_physical_features_samplewise() -> None:
    model = _model()
    features = torch.zeros(2, 2, 4, 3, 3, 3, requires_grad=True)
    ramp = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)
    features = features + ramp
    detection_logits = torch.zeros(2, 2, 1, 3, 3, 3, requires_grad=True)
    encoded = EncodedWindow(features=features, detection_logits=detection_logits)
    coords = torch.tensor(
        [
            [[[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]], [[1.0, 1.0, 1.0], [0, 0, 0]]],
            [[[0.5, 1.0, 1.5], [0, 0, 0]], [[2.0, 1.0, 0.0], [0, 0, 0]]],
        ],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [[[True, True], [True, False]], [[True, False], [True, False]]]
    )
    image_shape = torch.tensor([[100, 3, 3, 3], [100, 3, 3, 3]])
    voxel_size = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

    nodes = model.build_nodes(
        encoded,
        coords,
        masks,
        image_shape,
        voxel_size,
        frame_indices=torch.tensor([[40, 41], [70, 71]]),
        delta_t=2.5,
    )

    assert len(nodes) == 2
    torch.testing.assert_close(nodes[0].appearance[0, 0], torch.full((4,), 6.5))
    torch.testing.assert_close(
        nodes[0].physical_coords_um[0, 0],
        torch.tensor([0.5, 1.0, 1.5]),
    )
    torch.testing.assert_close(
        nodes[0].physical_coords_um[1, 0],
        torch.tensor([1.0, 3.0, 6.0]),
    )
    assert torch.count_nonzero(nodes[0].appearance[1, 1]) == 0
    assert nodes[0].spatial_position.shape == (2, 2, 24)
    torch.testing.assert_close(nodes[0].frame_role, torch.zeros_like(nodes[0].frame_role))
    torch.testing.assert_close(nodes[1].delta_t[0, 0], torch.tensor([2.5]))
    assert not nodes[0].detection_probability.requires_grad
    assert not nodes[0].division_probability.requires_grad
    assert nodes[0].division_logits.requires_grad

    shifted_time_nodes = model.build_nodes(
        encoded,
        coords,
        masks,
        image_shape,
        voxel_size,
        frame_indices=torch.tensor([[400, 401], [700, 701]]),
        delta_t=2.5,
    )
    torch.testing.assert_close(
        nodes[0].spatial_position,
        shifted_time_nodes[0].spatial_position,
    )


def test_common_encode_build_link_contract_and_detached_context() -> None:
    torch.manual_seed(5)
    model = _model()
    images = torch.randn(1, 2, 3, 3, 3, requires_grad=True)
    encoded = model.encode_window(images)
    coords = torch.tensor([[[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], [[1, 1, 1], [2, 2, 2]]]])
    masks = torch.ones(1, 2, 2, dtype=torch.bool)
    source, target = model.build_nodes(
        encoded,
        coords,
        masks,
        (100, 3, 3, 3),
        (1.0, 1.0, 1.0),
        frame_indices=(8, 9),
    )

    output = model.link_pair(source, target)

    assert encoded.features.shape == (1, 2, 4, 3, 3, 3)
    assert encoded.detection_logits.shape == (1, 2, 1, 3, 3, 3)
    assert model.transformer.feat_dim == 32
    assert output.edge_logits.shape == (1, 2, 2)
    assert output.null_parent_logits.shape == (1, 2)
    assert output.division_logits.shape == (1, 2)
    output.parent_logits.sum().backward()
    assert model.detect_head.weight.grad is None
    assert model.division_head[0].weight.grad is None
    assert model.transformer.projection.weight.grad is not None
    assert model.null_parent_head[0].weight.grad is not None


def test_link_pair_is_finite_with_fully_masked_samples() -> None:
    model = _model()
    encoded = model.encode_window(torch.randn(2, 2, 3, 3, 3))
    coords = torch.zeros(2, 2, 2, 3)
    masks = torch.tensor(
        [[[True, False], [True, False]], [[False, False], [False, False]]]
    )
    source, target = model.build_nodes(
        encoded,
        coords,
        masks,
        (100, 3, 3, 3),
        (1.0, 1.0, 1.0),
    )

    output = model.link_pair(source, target)

    assert torch.isfinite(output.edge_logits).all()
    assert torch.isfinite(output.null_parent_logits).all()
    assert (output.edge_logits[0, 1] == -1.0e4).all()
    assert (output.edge_logits[1] == -1.0e4).all()
    assert output.candidate_mask is not None
    assert not output.candidate_mask[1].any()
    parent_probability = output.parent_logits[1].softmax(dim=-1)
    torch.testing.assert_close(
        parent_probability[..., -1],
        torch.ones_like(parent_probability[..., -1]),
    )


def test_candidate_mask_makes_parent_probability_invariant_to_far_sources() -> None:
    torch.manual_seed(11)
    model = _model(radius_um=15.0, top_k=32)
    count = 80
    encoded = EncodedWindow(
        features=torch.zeros(1, 2, 4, 3, 3, 3),
        detection_logits=torch.zeros(1, 2, 1, 3, 3, 3),
    )
    coords = torch.zeros(1, 2, count, 3)
    coords[0, 0, :, 2] = torch.arange(1, count + 1)
    base_masks = torch.zeros(1, 2, count, dtype=torch.bool)
    base_masks[0, 0, :15] = True
    base_masks[0, 1, 0] = True
    extended_masks = base_masks.clone()
    extended_masks[0, 0] = True

    base = model.build_nodes(
        encoded,
        coords,
        base_masks,
        (100, 3, 3, 100),
        (1.0, 1.0, 1.0),
    )
    extended = model.build_nodes(
        encoded,
        coords,
        extended_masks,
        (100, 3, 3, 100),
        (1.0, 1.0, 1.0),
    )
    base_output = model.link_pair(base[0], base[1])
    extended_output = model.link_pair(extended[0], extended[1])

    assert base_output.candidate_mask is not None
    assert extended_output.candidate_mask is not None
    assert int(base_output.candidate_mask.sum()) == 15
    assert int(extended_output.candidate_mask.sum()) == 15
    base_probability = base_output.parent_logits[0, 0].softmax(dim=-1)
    extended_probability = extended_output.parent_logits[0, 0].softmax(dim=-1)
    torch.testing.assert_close(base_probability, extended_probability)


def test_candidate_mask_combines_radius_and_nearest_top_k() -> None:
    model = _model(radius_um=50.0, top_k=7)
    count = 80
    encoded = EncodedWindow(
        features=torch.zeros(1, 2, 4, 3, 3, 3),
        detection_logits=torch.zeros(1, 2, 1, 3, 3, 3),
    )
    coords = torch.zeros(1, 2, count, 3)
    coords[0, 0, :, 2] = torch.arange(1, count + 1)
    masks = torch.zeros(1, 2, count, dtype=torch.bool)
    masks[0, 0] = True
    masks[0, 1, 0] = True
    source, target = model.build_nodes(
        encoded,
        coords,
        masks,
        (100, 3, 3, 100),
        (1.0, 1.0, 1.0),
    )

    output = model.link_pair(source, target)

    assert output.candidate_mask is not None
    selected = torch.nonzero(output.candidate_mask[0, :, 0]).flatten()
    torch.testing.assert_close(selected, torch.arange(7))
    assert (output.edge_logits[0, 7:, 0] == -1.0e4).all()


@pytest.mark.parametrize("mode", ["identity", "per_voxel_mha"])
def test_corrected_factory_selects_temporal_fusion_independently_of_name(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    import backbone_ab.backbones as module

    class _DummySpatial(nn.Module):
        def __init__(self, *_: object, **__: object) -> None:
            super().__init__()

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image

    class _DummyTemporal(_DummySpatial):
        pass

    monkeypatch.setattr(module, "NNUNetFeatureExtractor", _DummySpatial)
    monkeypatch.setattr(module, "CorrectedTemporalNNUNetFeatureExtractor", _DummyTemporal)
    config = {
        "name": "nnunet",
        "contract": "corrected_v2",
        "input_channels": 1,
        "feature_dim": 4,
        "nnunet": {},
        "temporal_fusion": {"mode": mode, "stages": [2, 3], "heads": 4},
    }

    backbone = build_backbone(config)

    if mode == "identity":
        assert isinstance(backbone, FrameSharedBackbone)
    else:
        assert isinstance(backbone, _DummyTemporal)


def test_joint_factory_uses_corrected_model_without_changing_legacy_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backbone_ab.backbones as module

    monkeypatch.setattr(module, "build_backbone", lambda _: _DummyBackbone())
    model = build_joint_model(
        {
            "name": "nnunet",
            "contract": "corrected_v2",
            "feature_dim": 4,
            "link_candidate_radius_um": 15.0,
            "link_candidate_top_k": 32,
        },
        _host_module(),
    )

    assert isinstance(model, CorrectedTrackingModel)
    assert model.link_candidate_radius_um == 15.0
    assert model.link_candidate_top_k == 32

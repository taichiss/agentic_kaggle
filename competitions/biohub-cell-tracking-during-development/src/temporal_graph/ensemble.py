"""Controlled inference-only combinations of two temporal-link heads."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .candidates import (
    build_candidate_features,
    build_parent_candidates,
    refine_logits,
)
from .contracts import FrozenPair, RightTransitionTriplet, TemporalGraphOutput
from .model import TemporalGraphResidualHead

TEMPORAL_LINK_MODES = (
    "mlp",
    "bounded_attention",
    "bounded_logit_5050",
    "agreement_gate",
)


class TemporalGraphLinkEnsemble(nn.Module):
    """Apply one of four frozen, attribution-safe temporal-link policies.

    Both heads share the exact same candidate set and feature tensor. The
    image model, frozen host logits, detections, division policy, and graph
    post-processing therefore remain outside this comparison.
    """

    def __init__(
        self,
        mlp_head: TemporalGraphResidualHead,
        attention_head: TemporalGraphResidualHead,
        *,
        mode: str,
        logit_bound: float = 0.15,
    ) -> None:
        super().__init__()
        if mode not in TEMPORAL_LINK_MODES:
            raise ValueError(f"unsupported temporal link mode: {mode}")
        if not math.isfinite(logit_bound) or logit_bound <= 0.0:
            raise ValueError("logit_bound must be finite and positive")
        if mlp_head.config.architecture != "mlp":
            raise ValueError("mlp_head must use the MLP architecture")
        if attention_head.config.architecture != "candidate_attention":
            raise ValueError(
                "attention_head must use the candidate_attention architecture"
            )
        if attention_head.config.residual_logit_bound is None:
            raise ValueError("attention_head must have a residual logit bound")

        comparable_fields = (
            "node_feature_dim",
            "top_k",
            "radius_um",
            "distance_scale_um",
            "middle_coord_atol",
            "image_window_size",
            "graph_window_size",
            "ownership",
        )
        mismatches = [
            name
            for name in comparable_fields
            if getattr(mlp_head.config, name) != getattr(attention_head.config, name)
        ]
        if mismatches:
            raise ValueError(
                "temporal heads use different candidate contracts: "
                + ", ".join(mismatches)
            )

        self.mlp_head = mlp_head
        self.attention_head = attention_head
        self.mode = mode
        self.logit_bound = float(logit_bound)

    @staticmethod
    def _center_and_bound(
        residual: torch.Tensor,
        valid_mask: torch.Tensor,
        bound: float,
    ) -> torch.Tensor:
        valid = valid_mask.to(residual.dtype)
        mean = (residual * valid).sum(dim=-1, keepdim=True) / valid.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        centered = (residual - mean) * valid
        return bound * torch.tanh(centered / bound) * valid

    @staticmethod
    def _masked_parent(logits: torch.Tensor, current: FrozenPair) -> torch.Tensor:
        valid = current.source_mask.unsqueeze(-1) & current.target_mask.unsqueeze(1)
        return logits.masked_fill(~valid, -torch.inf).argmax(dim=1)

    def forward(
        self,
        previous_pair: FrozenPair | RightTransitionTriplet,
        current_pair: FrozenPair | None = None,
        *,
        history_pairs: Sequence[FrozenPair] | None = None,
        prior_pair: FrozenPair | None = None,
        oldest_pair: FrozenPair | None = None,
    ) -> TemporalGraphOutput:
        graph_window_size = self.mlp_head.config.graph_window_size
        if isinstance(previous_pair, RightTransitionTriplet):
            if current_pair is not None:
                raise ValueError("current_pair must be omitted when passing a triplet")
            previous = previous_pair.previous_to_source
            current = previous_pair.source_to_target
        else:
            if current_pair is None:
                raise ValueError("current_pair is required")
            previous = previous_pair
            current = current_pair
            RightTransitionTriplet(
                previous,
                current,
                middle_coord_atol=self.mlp_head.config.middle_coord_atol,
            )

        candidates = build_parent_candidates(
            current.source_coords_um.detach(),
            current.target_coords_um.detach(),
            current.source_mask.detach(),
            current.target_mask.detach(),
            top_k=self.mlp_head.config.top_k,
            radius_um=self.mlp_head.config.radius_um,
        )
        features = build_candidate_features(
            previous,
            current,
            candidates,
            history_pairs=history_pairs,
            prior_pair=prior_pair,
            oldest_pair=oldest_pair,
            graph_window_size=graph_window_size,
            distance_scale_um=self.mlp_head.config.distance_scale_um,
            middle_coord_atol=self.mlp_head.config.middle_coord_atol,
        )

        if self.mode == "mlp":
            residual = self.mlp_head.forward_candidate_features(features)
        elif self.mode == "bounded_attention":
            residual = self.attention_head.forward_candidate_features(features)
        else:
            mlp_residual = self.mlp_head.forward_candidate_features(features)
            attention_residual = self.attention_head.forward_candidate_features(
                features
            )
            averaged = 0.5 * mlp_residual + 0.5 * attention_residual
            if self.mode == "bounded_logit_5050":
                residual = self._center_and_bound(
                    averaged,
                    candidates.valid_mask,
                    self.logit_bound,
                )
            else:
                mlp_logits = refine_logits(
                    current.edge_logits,
                    candidates,
                    mlp_residual,
                )
                attention_logits = refine_logits(
                    current.edge_logits,
                    candidates,
                    attention_residual,
                )
                host_parent = self._masked_parent(current.edge_logits, current)
                mlp_parent = self._masked_parent(mlp_logits, current)
                attention_parent = self._masked_parent(attention_logits, current)
                agreed_parent = mlp_parent.unsqueeze(-1)
                agreed_is_candidate = (
                    (candidates.source_index == agreed_parent)
                    & candidates.valid_mask
                ).any(dim=-1)
                override = (
                    mlp_parent.eq(attention_parent)
                    & mlp_parent.ne(host_parent)
                    & agreed_is_candidate
                    & current.target_mask
                )
                # MLP acts only as an agreement veto. The applied correction
                # comes from the already centered and bounded Attention head;
                # every other target column remains the frozen host exactly.
                residual = attention_residual * override.unsqueeze(-1).to(
                    attention_residual.dtype
                )

        logits = refine_logits(current.edge_logits, candidates, residual)
        return TemporalGraphOutput(logits, residual, features)

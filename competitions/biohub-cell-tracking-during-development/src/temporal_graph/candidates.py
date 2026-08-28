"""Bounded parent candidates and O(NK) temporal graph features."""

from __future__ import annotations

import torch

from .contracts import (
    CandidateFeatureBatch,
    FrozenPair,
    ParentCandidates,
    PreviousParentStatistics,
    RightTransitionTriplet,
)


def build_parent_candidates(
    source_coords_um: torch.Tensor,
    target_coords_um: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    top_k: int,
    radius_um: float,
) -> ParentCandidates:
    """Select the nearest valid sources inside a physical-radius gate.

    Output is always padded to exactly ``top_k`` slots and is target-major.
    Invalid slots use source index zero and distance zero; ``valid_mask`` is
    authoritative. The bounded output supports O(NK) downstream features.
    """
    if source_coords_um.ndim != 3 or source_coords_um.shape[-1] != 3:
        raise ValueError("source_coords_um must have shape (B,N_source,3)")
    if target_coords_um.ndim != 3 or target_coords_um.shape[-1] != 3:
        raise ValueError("target_coords_um must have shape (B,N_target,3)")
    if source_mask.shape != source_coords_um.shape[:2] or source_mask.dtype != torch.bool:
        raise ValueError("source_mask must be boolean with shape (B,N_source)")
    if target_mask.shape != target_coords_um.shape[:2] or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be boolean with shape (B,N_target)")
    if source_coords_um.shape[0] != target_coords_um.shape[0]:
        raise ValueError("source and target coordinates must share the batch axis")
    if source_coords_um.device != target_coords_um.device:
        raise ValueError("source and target coordinates must be on the same device")
    if (
        source_mask.device != source_coords_um.device
        or target_mask.device != source_coords_um.device
    ):
        raise ValueError("candidate inputs must be on the same device")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")

    batch, source_count = source_coords_um.shape[:2]
    target_count = target_coords_um.shape[1]
    device = source_coords_um.device
    output_index = torch.zeros(batch, target_count, top_k, dtype=torch.long, device=device)
    output_mask = torch.zeros(batch, target_count, top_k, dtype=torch.bool, device=device)
    output_distance = torch.zeros(
        batch,
        target_count,
        top_k,
        dtype=torch.float32,
        device=device,
    )
    if batch == 0 or source_count == 0 or target_count == 0:
        return ParentCandidates(output_index, output_mask, output_distance, source_count)

    distances = torch.cdist(
        target_coords_um.detach().float(),
        source_coords_um.detach().float(),
    )
    eligible = target_mask.unsqueeze(-1) & source_mask.unsqueeze(1)
    eligible &= distances <= float(radius_um)
    bounded = distances.masked_fill(~eligible, torch.inf)
    selected_count = min(top_k, source_count)
    selected_distance, selected_index = torch.topk(
        bounded,
        k=selected_count,
        dim=-1,
        largest=False,
        sorted=True,
    )
    selected_valid = torch.isfinite(selected_distance)
    output_index[..., :selected_count] = torch.where(
        selected_valid, selected_index, torch.zeros_like(selected_index)
    )
    output_mask[..., :selected_count] = selected_valid
    output_distance[..., :selected_count] = torch.where(
        selected_valid, selected_distance, torch.zeros_like(selected_distance)
    )
    return ParentCandidates(output_index, output_mask, output_distance, source_count)


def expected_previous_parent_statistics(
    previous_pair: FrozenPair,
    *,
    eps: float = 1.0e-8,
) -> PreviousParentStatistics:
    """Return expected parent position and entropy for each middle-frame node.

    Parent probabilities are normalized across valid previous-frame sources.
    A node with no valid history falls back to its own position, with zero
    entropy and ``has_history=False``. No all-masked softmax is evaluated.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    batch, previous_count, middle_count = previous_pair.edge_logits.shape
    fallback = previous_pair.target_coords_um.detach().float()
    entropy = fallback.new_zeros(batch, middle_count, 1)
    has_history = torch.zeros(
        batch,
        middle_count,
        1,
        dtype=torch.bool,
        device=fallback.device,
    )
    if batch == 0 or previous_count == 0 or middle_count == 0:
        return PreviousParentStatistics(fallback, entropy, has_history)

    logits = previous_pair.edge_logits.detach().float()
    valid = previous_pair.source_mask.unsqueeze(-1) & previous_pair.target_mask.unsqueeze(1)
    valid &= torch.isfinite(logits)
    masked_logits = logits.masked_fill(~valid, -torch.inf)
    has_parent = valid.any(dim=1)
    maximum = masked_logits.amax(dim=1)
    maximum = torch.where(has_parent, maximum, torch.zeros_like(maximum))
    weights = torch.where(
        valid,
        torch.exp(masked_logits - maximum.unsqueeze(1)),
        torch.zeros_like(masked_logits),
    )
    probabilities = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    expected = torch.einsum(
        "bps,bpc->bsc",
        probabilities,
        previous_pair.source_coords_um.detach().float(),
    )
    expected = torch.where(has_parent.unsqueeze(-1), expected, fallback)
    entropy_values = -(
        probabilities * probabilities.clamp_min(eps).log()
    ).sum(dim=1)
    entropy = torch.where(
        has_parent, entropy_values, torch.zeros_like(entropy_values)
    ).unsqueeze(-1)
    has_history = has_parent.unsqueeze(-1)
    return PreviousParentStatistics(expected, entropy, has_history)


def _gather_source(values: torch.Tensor, candidates: ParentCandidates) -> torch.Tensor:
    """Gather ``(B,S,...)`` values into target-major ``(B,T,K,...)``."""
    batch, source_count = values.shape[:2]
    if batch != candidates.batch_size or source_count != candidates.source_count:
        raise ValueError("source values do not match ParentCandidates")
    trailing = values.shape[2:]
    output_shape = (
        candidates.batch_size,
        candidates.target_count,
        candidates.top_k,
        *trailing,
    )
    if source_count == 0:
        return values.new_zeros(output_shape)
    batch_index = torch.arange(batch, device=values.device)[:, None, None]
    return values[batch_index, candidates.source_index]


def _gather_edge_logits(
    edge_logits: torch.Tensor,
    candidates: ParentCandidates,
) -> torch.Tensor:
    if edge_logits.shape != (
        candidates.batch_size,
        candidates.source_count,
        candidates.target_count,
    ):
        raise ValueError("edge_logits do not match ParentCandidates")
    if candidates.source_count == 0:
        return edge_logits.new_zeros(candidates.source_index.shape)
    return torch.gather(edge_logits.transpose(1, 2), 2, candidates.source_index)


def candidate_feature_dim(node_feature_dim: int) -> int:
    """Return the stable feature width used by :func:`build_candidate_features`."""
    if isinstance(node_feature_dim, bool) or not isinstance(node_feature_dim, int):
        raise TypeError("node_feature_dim must be an integer")
    if node_feature_dim <= 0:
        raise ValueError("node_feature_dim must be positive")
    return 3 * node_feature_dim + 10


def build_candidate_features(
    previous_pair: FrozenPair,
    current_pair: FrozenPair,
    candidates: ParentCandidates,
    *,
    distance_scale_um: float = 10.0,
    middle_coord_atol: float = 1.0e-4,
) -> CandidateFeatureBatch:
    """Build O(N_target * K) features for the owned right transition.

    Feature order is: previous-view middle appearance, current-view source
    appearance, target appearance, normalized displacement (3), normalized
    constant-velocity residual (3), distance, frozen base logit, previous
    parent entropy, and a history-availability indicator.
    """
    if distance_scale_um <= 0:
        raise ValueError("distance_scale_um must be positive")
    triplet = RightTransitionTriplet(
        previous_pair,
        current_pair,
        middle_coord_atol=middle_coord_atol,
    )
    current = triplet.owned_transition
    if candidates.batch_size != current.batch_size:
        raise ValueError("candidate and transition batch dimensions differ")
    if candidates.source_count != current.source_count:
        raise ValueError("candidate source axis differs from the owned transition")
    if candidates.target_count != current.target_count:
        raise ValueError("candidate target axis differs from the owned transition")

    history = expected_previous_parent_statistics(previous_pair)
    previous_view = _gather_source(
        previous_pair.target_features.detach().float(), candidates
    )
    current_source = _gather_source(current.source_features.detach().float(), candidates)
    source_coords = _gather_source(current.source_coords_um.detach().float(), candidates)
    expected_previous = _gather_source(history.expected_position_um, candidates)
    previous_entropy = _gather_source(history.entropy, candidates)
    has_history = _gather_source(history.has_history, candidates).float()

    target_view = current.target_features.detach().float().unsqueeze(2).expand(
        -1, -1, candidates.top_k, -1
    )
    target_coords = current.target_coords_um.detach().float().unsqueeze(2).expand(
        -1, -1, candidates.top_k, -1
    )
    displacement = (target_coords - source_coords) / float(distance_scale_um)
    velocity = source_coords - expected_previous
    predicted_target = source_coords + velocity
    velocity_residual = (target_coords - predicted_target) / float(distance_scale_um)
    distance = candidates.distance_um.unsqueeze(-1) / float(distance_scale_um)
    base_logit = _gather_edge_logits(current.edge_logits.detach().float(), candidates).unsqueeze(-1)

    features = torch.cat(
        [
            previous_view,
            current_source,
            target_view,
            displacement,
            velocity_residual,
            distance,
            base_logit,
            previous_entropy,
            has_history,
        ],
        dim=-1,
    )
    expected_dim = candidate_feature_dim(current.feature_dim)
    if features.shape[-1] != expected_dim:
        raise RuntimeError("candidate feature construction drifted from its public contract")
    features = features * candidates.valid_mask.unsqueeze(-1).to(features.dtype)
    return CandidateFeatureBatch(features, candidates, history)


def refine_logits(
    base_logits: torch.Tensor,
    candidates: ParentCandidates,
    candidate_residual: torch.Tensor,
) -> torch.Tensor:
    """Scatter residuals onto candidates while preserving every other logit.

    The base tensor is detached deliberately: this boundary guarantees the
    host model remains frozen even when callers pass tensors requiring grad.
    """
    expected = (
        candidates.batch_size,
        candidates.source_count,
        candidates.target_count,
    )
    if base_logits.shape != expected:
        raise ValueError(f"base_logits must have shape {expected}")
    if candidate_residual.shape != candidates.source_index.shape:
        raise ValueError("candidate_residual must have shape (B,N_target,K)")
    base = base_logits.detach()
    if not candidates.valid_mask.any():
        return base + candidate_residual.sum().to(base.dtype) * 0.0

    dense_delta = torch.zeros_like(base)
    batch, target, slot = torch.nonzero(candidates.valid_mask, as_tuple=True)
    source = candidates.source_index[batch, target, slot]
    values = candidate_residual[batch, target, slot].to(base.dtype)
    dense_delta.index_put_((batch, source, target), values, accumulate=True)
    return base + dense_delta

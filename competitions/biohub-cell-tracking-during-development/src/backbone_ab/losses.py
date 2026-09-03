"""Sparse-GT-aware linker and division losses for the corrected model contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

UNKNOWN_DIVISION = -1
WEAK_NON_DIVISION = 0
DIVISION = 1


@dataclass(frozen=True)
class ParentSupervision:
    """Target-wise parent classes and a mask of trustworthy labels."""

    classes: torch.Tensor
    mask: torch.Tensor


def mask_non_candidate_parents(
    supervision: ParentSupervision,
    candidate_mask: torch.Tensor | None,
) -> ParentSupervision:
    """Ignore annotated parents excluded by the bounded candidate graph.

    Candidate masks are source-major ``(B,S,T)``. Null classes remain valid
    because they do not refer to a source candidate.
    """
    if candidate_mask is None:
        return supervision
    if candidate_mask.ndim != 3:
        raise ValueError("candidate_mask must have shape (B,S,T)")
    batch, sources, targets = candidate_mask.shape
    if supervision.classes.shape != (batch, targets):
        raise ValueError("candidate mask and parent supervision dimensions differ")
    keep = supervision.mask.clone()
    non_null = keep & (supervision.classes < sources)
    if non_null.any():
        sample, target = torch.nonzero(non_null, as_tuple=True)
        source = supervision.classes[sample, target]
        available = candidate_mask[sample, source, target]
        keep[sample, target] = available
    return ParentSupervision(classes=supervision.classes, mask=keep)


@torch.no_grad()
def candidate_parent_counts(
    supervision: ParentSupervision,
    candidate_mask: torch.Tensor,
) -> tuple[int, int]:
    """Count supervised non-null parents retained by the candidate graph."""
    if candidate_mask.ndim != 3:
        raise ValueError("candidate_mask must have shape (B,S,T)")
    batch, sources, targets = candidate_mask.shape
    if supervision.classes.shape != (batch, targets):
        raise ValueError("candidate mask and parent supervision dimensions differ")
    non_null = supervision.mask & (supervision.classes < sources)
    total = int(non_null.sum().item())
    if total == 0:
        return 0, 0
    sample, target = torch.nonzero(non_null, as_tuple=True)
    source = supervision.classes[sample, target]
    survived = int(candidate_mask[sample, source, target].sum().item())
    return survived, total


def build_parent_supervision(
    gt_edges: torch.Tensor,
    source_matches: torch.Tensor,
    target_matches: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    reliable_null_target: torch.Tensor,
) -> ParentSupervision:
    """Map sparse GT edges to proposal parents without inventing negatives.

    A matched target is supervised only when its single annotated parent is
    present in the source proposals. Zero-parent GT columns remain unknown:
    sparse labels cannot distinguish a birth from a missing annotation. The
    null class is supervised only when training deliberately dropped the
    target's otherwise-present annotated parent proposal.
    """
    if gt_edges.ndim != 3:
        raise ValueError("gt_edges must have shape (B,Gs,Gt)")
    batch, source_count = source_matches.shape
    if target_matches.shape[0] != batch:
        raise ValueError("source and target matches must share a batch dimension")
    target_count = target_matches.shape[1]
    classes = torch.full(
        (batch, target_count), source_count, dtype=torch.long, device=gt_edges.device
    )
    supervised = torch.zeros(
        batch, target_count, dtype=torch.bool, device=gt_edges.device
    )

    for sample in range(batch):
        for target_index in range(target_count):
            if not bool(target_mask[sample, target_index]):
                continue
            if bool(reliable_null_target[sample, target_index]):
                supervised[sample, target_index] = True
                continue
            gt_target = int(target_matches[sample, target_index].item())
            if gt_target < 0 or gt_target >= gt_edges.shape[2]:
                continue
            parents = torch.nonzero(
                gt_edges[sample, :, gt_target] > 0,
                as_tuple=False,
            ).flatten()
            if parents.numel() != 1:
                continue
            candidates = torch.nonzero(
                source_mask[sample]
                & (source_matches[sample] == int(parents[0].item())),
                as_tuple=False,
            ).flatten()
            if candidates.numel() != 1:
                # A missing or ambiguous proposal is unknown, not null.
                continue
            classes[sample, target_index] = candidates[0]
            supervised[sample, target_index] = True
    return ParentSupervision(classes=classes, mask=supervised)


def parent_classification_loss(
    parent_logits: torch.Tensor,
    supervision: ParentSupervision,
) -> torch.Tensor:
    """Cross-entropy over trustworthy target-wise parent labels."""
    if parent_logits.ndim != 3:
        raise ValueError("parent_logits must have shape (B,T,S+1)")
    if parent_logits.shape[:2] != supervision.classes.shape:
        raise ValueError("parent logits and supervision target dimensions differ")
    if not supervision.mask.any():
        return parent_logits.sum() * 0.0
    with torch.autocast(device_type=parent_logits.device.type, enabled=False):
        return F.cross_entropy(
            parent_logits.float()[supervision.mask],
            supervision.classes[supervision.mask],
        )


def build_division_states(
    gt_edges: torch.Tensor,
    source_matches: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    """Return positive, weak-negative, and unknown division states.

    Annotated out-degree >1 is positive, exactly one is a weak negative, and
    zero is unknown. Unmatched and padded proposals are also unknown.
    """
    if gt_edges.ndim != 3:
        raise ValueError("gt_edges must have shape (B,Gs,Gt)")
    states = torch.full_like(source_matches, UNKNOWN_DIVISION, dtype=torch.int8)
    for sample in range(source_matches.shape[0]):
        for source_index in range(source_matches.shape[1]):
            if not bool(source_mask[sample, source_index]):
                continue
            gt_source = int(source_matches[sample, source_index].item())
            if gt_source < 0 or gt_source >= gt_edges.shape[1]:
                continue
            children = int((gt_edges[sample, gt_source] > 0).sum().item())
            if children > 1:
                states[sample, source_index] = DIVISION
            elif children == 1:
                states[sample, source_index] = WEAK_NON_DIVISION
    return states


def three_state_division_loss(
    division_logits: torch.Tensor,
    states: torch.Tensor,
    *,
    weak_negative_weight: float = 0.1,
) -> torch.Tensor:
    """Compute separately normalised positive and weak-negative BCE terms."""
    if division_logits.shape != states.shape:
        raise ValueError("division logits and states must have the same shape")
    if weak_negative_weight < 0:
        raise ValueError("weak_negative_weight must be non-negative")
    positive = states == DIVISION
    negative = states == WEAK_NON_DIVISION
    with torch.autocast(device_type=division_logits.device.type, enabled=False):
        logits = division_logits.float()
        positive_loss = (
            F.binary_cross_entropy_with_logits(
                logits[positive], torch.ones_like(logits[positive])
            )
            if positive.any()
            else logits.sum() * 0.0
        )
        negative_loss = (
            F.binary_cross_entropy_with_logits(
                logits[negative], torch.zeros_like(logits[negative])
            )
            if negative.any()
            else logits.sum() * 0.0
        )
    return positive_loss + weak_negative_weight * negative_loss


@torch.no_grad()
def parent_accuracy(
    parent_logits: torch.Tensor,
    supervision: ParentSupervision,
) -> tuple[int, int]:
    """Return correct and supervised-target counts."""
    predictions = parent_logits.argmax(dim=-1)
    correct = int(
        ((predictions == supervision.classes) & supervision.mask).sum().item()
    )
    return correct, int(supervision.mask.sum().item())

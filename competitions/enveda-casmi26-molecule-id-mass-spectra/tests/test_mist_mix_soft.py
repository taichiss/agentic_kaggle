"""M3 loss must preserve A on clean rows and reward both mixed sources."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mist_formula"))
from train_mix_soft import replace_positive_decoys, soft_mix_loss, tanimoto  # noqa: E402


def test_clean_row_matches_frozen_a_loss():
    logits = torch.tensor([[0.2, -0.7, 0.8, -0.4]], requires_grad=True)
    anchor = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    partner = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    negative = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    pos_weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss, bce, ranking = soft_mix_loss(
        logits, anchor, partner, negative, torch.tensor([0.0]), pos_weight, 0.5
    )
    expected_bce = F.binary_cross_entropy_with_logits(logits, anchor, pos_weight=pos_weight)
    p = logits.sigmoid()
    expected_rank = F.softplus(10 * (tanimoto(p, negative) - tanimoto(p, anchor))).mean()
    torch.testing.assert_close(bce, expected_bce)
    torch.testing.assert_close(ranking, expected_rank)
    torch.testing.assert_close(loss, expected_bce + 0.5 * expected_rank)


def test_half_mix_has_soft_target_and_two_positive_ranks():
    logits = torch.tensor([[0.2, -0.7, 0.8, -0.4]], requires_grad=True)
    anchor = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    partner = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    negative = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    pos_weight = torch.ones(4)
    loss, bce, ranking = soft_mix_loss(
        logits, anchor, partner, negative, torch.tensor([0.5]), pos_weight, 0.5
    )
    expected_bce = F.binary_cross_entropy_with_logits(logits, 0.5 * (anchor + partner))
    p = logits.sigmoid()
    negative_score = tanimoto(p, negative)
    expected_rank = 0.5 * F.softplus(10 * (negative_score - tanimoto(p, anchor)))
    expected_rank += 0.5 * F.softplus(10 * (negative_score - tanimoto(p, partner)))
    torch.testing.assert_close(bce, expected_bce)
    torch.testing.assert_close(ranking, expected_rank.mean())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_partner_positive_is_never_used_as_negative():
    data = SimpleNamespace(
        mol=pd.DataFrame({"formula": ["C2H6O", "C2H6O", "C2H6O", "C3H8O"]}),
        same_formula=defaultdict(list, {"C2H6O": [0, 1, 2]}),
        train_mols=np.arange(4),
    )
    negatives = [1]
    replaced, fallback = replace_positive_decoys(
        np.array([0]), np.array([1]), negatives, data, np.random.default_rng(1)
    )
    assert negatives == [2]
    assert replaced == 1 and fallback == 0

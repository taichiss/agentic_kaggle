"""Fixed 0.479 within-formula rank fusion for B, C and score-level fusion."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def formula(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None


def fuse_scores(B: np.ndarray, C: np.ndarray, indices: list[int]) -> np.ndarray | None:
    """Standardize within one formula group, then combine candidate scores."""
    b, c = np.asarray(B)[indices], np.asarray(C)[indices]
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        return None

    def z(v):
        sd = v.std()
        return (v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v)

    return 0.5 * z(b) + 0.5 * z(c)


def rerank(smiles: list[str], scores: np.ndarray, *, weight: float = 0.25):
    """Return permutation, keeping rank 1 and each formula's position slots."""
    if not 0 <= weight <= 1 or len(smiles) != len(scores):
        raise ValueError("invalid fusion inputs")
    groups: dict[str, list[int]] = {}
    for i in range(1, len(smiles)):
        key = formula(smiles[i])
        if key is not None:
            groups.setdefault(key, []).append(i)
    order = list(range(len(smiles)))
    missing_groups = 0
    eligible_groups = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        values = np.asarray(scores)[indices]
        if not np.isfinite(values).all():
            missing_groups += 1
            continue
        eligible_groups += 1
        model_order = sorted(indices, key=lambda i: (-float(scores[i]), i))
        ranks = {i: j for j, i in enumerate(model_order)}
        fused = {
            i: (1 - weight) / (1 + base_rank) + weight / (1 + ranks[i])
            for base_rank, i in enumerate(indices)
        }
        selected = sorted(indices, key=lambda i: (-fused[i], i))
        for slot, i in zip(indices, selected, strict=True):
            order[slot] = i
    if order[0] != 0 or sorted(order) != list(range(len(smiles))):
        raise AssertionError("fusion changed protected candidate identity")
    return order, {"eligible_groups": eligible_groups, "missing_groups": missing_groups}


def rerank_fused(smiles: list[str], B: np.ndarray, C: np.ndarray, *, weight: float = 0.25):
    """Apply the parent rank blend once, after B/C score fusion."""
    scores = np.full(len(smiles), -np.inf, dtype=np.float32)
    groups: dict[str, list[int]] = {}
    for i in range(len(smiles)):
        key = formula(smiles[i])
        if key is not None:
            groups.setdefault(key, []).append(i)
    for indices in groups.values():
        if sum(i > 0 for i in indices) < 2:
            continue
        values = fuse_scores(B, C, indices)
        if values is not None:
            for index, value in zip(indices, values, strict=True):
                if index > 0:
                    scores[index] = value
    return rerank(smiles, scores, weight=weight)

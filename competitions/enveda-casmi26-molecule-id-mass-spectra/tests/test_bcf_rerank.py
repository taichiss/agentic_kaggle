import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/mist_formula"))
import bcf_rerank  # noqa: E402
from bcf_rerank import rerank, rerank_fused  # noqa: E402
from leader_rerank import rerank_indices  # noqa: E402


def test_formula_slots_and_leader_are_fixed():
    smiles = ["CCO", "CCO", "COC", "CCN", "NCC", "C(C)O"]
    scores = np.array([0, 0.1, 0.9, 0.2, 0.8, 0.7])
    order, audit = rerank(smiles, scores)
    assert order[0] == 0
    assert sorted(order) == list(range(len(smiles)))
    assert [smiles[i].count("N") for i in order] == [s.count("N") for s in smiles]
    assert audit["missing_groups"] == 0


def test_missing_c_preserves_affected_group_and_fusion_is_once():
    smiles = ["CCO", "COC", "CCO", "CCN", "NCC"]
    b = np.array([0.0, 0.2, 0.9, 0.1, 0.8])
    c = np.array([0.0, 0.8, 0.1, -np.inf, -np.inf])
    order, audit = rerank_fused(smiles, b, c)
    assert order[0] == 0
    assert order[3:] == [3, 4]
    assert audit["missing_groups"] == 1


def test_complete_scores_match_original_0479_protected_reranker():
    smiles = ["CCO", "COC", "C(C)O", "CCN", "NCC", "CC=O"]
    rng = np.random.default_rng(20261110)
    for _ in range(30):
        scores = rng.normal(size=len(smiles))
        actual, _ = rerank(smiles, scores)
        expected, _ = rerank_indices(smiles, scores, leader_mode="protected")
        assert actual == expected


def test_fusion_standardization_includes_protected_top1(monkeypatch):
    original = bcf_rerank.fuse_scores
    seen = []

    def capture(b, c, indices):
        seen.append(indices)
        return original(b, c, indices)

    monkeypatch.setattr(bcf_rerank, "fuse_scores", capture)
    rerank_fused(
        ["CCO", "COC", "C(C)O"],
        np.array([0.9, 0.2, 0.3]),
        np.array([0.1, 0.8, 0.7]),
    )
    assert [0, 1, 2] in seen

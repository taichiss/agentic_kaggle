from __future__ import annotations

from casmi26.metrics import (
    SubmissionRow,
    canonical_inchikey14,
    mrr_at_k_breakdown,
    score_submission,
)

# 既知の分子 SMILES
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
GLUCOSE = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
# 同じ分子の別の正当な SMILES (芳香環表記違い)
ASPIRIN_ALT = "CC(=O)Oc1ccccc1C(=O)O"


def test_canonical_inchikey14_handles_equivalent_smiles():
    """芳香環表記違いの SMILES は同じ InChIKey14 になる。"""
    k1 = canonical_inchikey14(ASPIRIN)
    k2 = canonical_inchikey14(ASPIRIN_ALT)
    assert k1 is not None
    assert len(k1) == 14
    assert k1 == k2


def test_canonical_inchikey14_returns_none_on_invalid():
    assert canonical_inchikey14("not_a_smiles_123!!!") is None
    assert canonical_inchikey14("") is None


def test_mrr_at_k_rank1_scores_one():
    rows = [SubmissionRow("m", [ASPIRIN], ASPIRIN)]
    r = mrr_at_k_breakdown(rows)
    assert r.mrr == 1.0
    assert r.top1_accuracy == 1.0
    assert r.rank_counts[1] == 1


def test_mrr_at_k_rank2_scores_half():
    rows = [SubmissionRow("m", [CAFFEINE, ASPIRIN], ASPIRIN)]
    r = mrr_at_k_breakdown(rows)
    assert r.mrr == 0.5
    assert r.rank_counts[2] == 1


def test_mrr_at_k_rank25_scores_point04():
    cands = [CAFFEINE] * 24 + [ASPIRIN]
    rows = [SubmissionRow("m", cands, ASPIRIN)]
    r = mrr_at_k_breakdown(rows)
    assert abs(r.mrr - 1 / 25) < 1e-9
    assert r.rank_counts[25] == 1


def test_mrr_at_k_miss_scores_zero():
    cands = [CAFFEINE] * 25
    rows = [SubmissionRow("m", cands, ASPIRIN)]
    r = mrr_at_k_breakdown(rows)
    assert r.mrr == 0.0
    assert r.rank_counts[0] == 1


def test_mrr_at_k_mixed():
    """3 分子: rank 1 / rank 2 / miss → MRR = (1.0 + 0.5 + 0) / 3 = 0.5。"""
    rows = [
        SubmissionRow("m1", [ASPIRIN], ASPIRIN),
        SubmissionRow("m2", [CAFFEINE, ASPIRIN], ASPIRIN),
        SubmissionRow("m3", [CAFFEINE] * 25, ASPIRIN),
    ]
    r = mrr_at_k_breakdown(rows)
    assert abs(r.mrr - 0.5) < 1e-9
    assert abs(r.top1_accuracy - 1 / 3) < 1e-9
    assert r.rank_counts == {
        **{i: 0 for i in range(26)},
        0: 1,
        1: 1,
        2: 1,
    }


def test_mrr_at_k_candidates_beyond_k_ignored():
    """25 位より後ろにある正解は miss 扱い。"""
    cands = [CAFFEINE] * 25 + [ASPIRIN]
    rows = [SubmissionRow("m", cands, ASPIRIN)]
    r = mrr_at_k_breakdown(rows)
    assert r.mrr == 0.0


def test_mrr_at_k_shorter_k():
    """k_max=5 で rank 10 の正解は miss。"""
    cands = [CAFFEINE] * 9 + [ASPIRIN]
    rows = [SubmissionRow("m", cands, ASPIRIN)]
    r = mrr_at_k_breakdown(rows, k_max=5)
    assert r.mrr == 0.0
    r25 = mrr_at_k_breakdown(rows, k_max=25)
    assert abs(r25.mrr - 0.1) < 1e-9


def test_class_breakdown_sum_equals_overall_mrr():
    """クラス寄与の総和は全体 MRR と一致。"""
    rows = [
        SubmissionRow("m1", [ASPIRIN], ASPIRIN),  # rank 1
        SubmissionRow("m2", [CAFFEINE, ASPIRIN], ASPIRIN),  # rank 2
        SubmissionRow("m3", [CAFFEINE] * 25, ASPIRIN),  # miss
        SubmissionRow("m4", [GLUCOSE], GLUCOSE),  # rank 1
    ]
    labels = ["class1", "class2", "class3", "class1"]
    r = mrr_at_k_breakdown(rows, class_labels=labels)
    assert r.class_contributions is not None
    assert r.class_mrr is not None
    assert r.class_counts == {"class1": 2, "class2": 1, "class3": 1}
    assert r.class_mrr["class1"] == 1.0
    assert r.class_mrr["class2"] == 0.5
    assert r.class_mrr["class3"] == 0.0
    total = sum(r.class_contributions.values())
    assert abs(total - r.mrr) < 1e-9
    # (1 + 1 + 0.5 + 0) / 4 = 0.625
    assert abs(r.mrr - 0.625) < 1e-9


def test_score_submission_dict_interface():
    submission = {
        "m1": [ASPIRIN],
        "m2": [CAFFEINE, ASPIRIN],
    }
    truth = {
        "m1": ASPIRIN,
        "m2": ASPIRIN,
        "m3": GLUCOSE,  # 提出に無い分子は 0 点
    }
    r = score_submission(submission, truth)
    # (1 + 0.5 + 0) / 3
    assert abs(r.mrr - 0.5) < 1e-9
    assert r.count == 3

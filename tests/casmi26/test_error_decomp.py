from __future__ import annotations

import math

import pandas as pd
import pytest

from casmi26.analysis import (
    StageSpec,
    assign_classes,
    attach_canonical_keys,
    decompose,
    default_stages,
    to_markdown,
)
from casmi26.metrics import SubmissionRow, canonical_inchikey14, mrr_at_k


def _long(rankings: dict[str, list[str]]) -> pd.DataFrame:
    rows = [
        {"molecule_id": m, "candidate_key": k, "rank": r}
        for m, keys in rankings.items()
        for r, k in enumerate(keys, start=1)
    ]
    return pd.DataFrame(rows, columns=["molecule_id", "candidate_key", "rank"])


def _filler(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(n)]


@pytest.fixture
def scenario():
    """5 分子で 4 段の落ち方を 1 つずつ作る。

    m_top1   : 全段通過、最終 rank 1
    m_pool   : 候補 pool に正解なし
    m_fp     : pool にはあるが FP 順位 150 (top-100 外)
    m_sim    : FP top-100 だが simulator 後に rank 30
    m_ranker : simulator top-25 だが最終 rank 4
    """
    truth = {m: f"T_{m}" for m in ["m_top1", "m_pool", "m_fp", "m_sim", "m_ranker"]}
    pool = {m: _filler("X", 10) + [t] for m, t in truth.items() if m != "m_pool"}
    pool["m_pool"] = _filler("X", 10)
    fp = {
        "m_top1": [truth["m_top1"]] + _filler("X", 5),
        "m_pool": _filler("X", 5),
        "m_fp": _filler("X", 149) + [truth["m_fp"]],
        "m_sim": _filler("X", 9) + [truth["m_sim"]],
        "m_ranker": _filler("X", 2) + [truth["m_ranker"]],
    }
    sim = {
        "m_top1": [truth["m_top1"]],
        "m_pool": _filler("X", 5),
        "m_fp": _filler("X", 30),
        "m_sim": _filler("X", 29) + [truth["m_sim"]],
        "m_ranker": _filler("X", 4) + [truth["m_ranker"]],
    }
    ranker = {
        "m_top1": [truth["m_top1"]] + _filler("X", 24),
        "m_pool": _filler("X", 25),
        "m_fp": _filler("X", 25),
        "m_sim": _filler("X", 25),
        "m_ranker": _filler("X", 3) + [truth["m_ranker"]] + _filler("Y", 21),
    }
    outputs = {
        "pool": _long(pool),
        "fp": _long(fp),
        "simulator": _long(sim),
        "ranker": _long(ranker),
    }
    return truth, outputs


def test_each_molecule_lands_in_its_failing_stage(scenario):
    truth, outputs = scenario
    res = decompose(truth, outputs)
    bucket = dict(zip(res.per_molecule["molecule_id"], res.per_molecule["bucket"], strict=True))
    assert bucket == {
        "m_top1": "top1",
        "m_pool": "pool",
        "m_fp": "fp",
        "m_sim": "simulator",
        "m_ranker": "ranker",
    }


def test_mrr_and_loss_shares_add_up(scenario):
    truth, outputs = scenario
    res = decompose(truth, outputs)
    # m_top1 = 1.0, m_ranker = 1/4、他は圏外
    assert res.mrr == pytest.approx((1.0 + 0.25) / 5)
    assert res.stage_table["loss_share"].sum() == pytest.approx(1.0 - res.mrr)
    share = res.stage_table.groupby("bucket")["loss_share"].sum()
    assert share["ranker"] == pytest.approx(0.75 / 5)
    assert share["pool"] == pytest.approx(1.0 / 5)
    assert share["top1"] == pytest.approx(0.0)


def test_ceilings_are_cumulative_pass_rates(scenario):
    truth, outputs = scenario
    res = decompose(truth, outputs)
    assert res.ceilings == pytest.approx({"pool": 0.8, "fp": 0.6, "simulator": 0.4})


def test_final_mrr_matches_official_scorer():
    """最終段の MRR が breakdown.mrr_at_k (公式スコアラ再現) と一致する。"""
    smiles = [
        "CCO",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "C1=CC=C(C=C1)C=O",
        "CCC(=O)O",
    ]
    keys = [canonical_inchikey14(s) for s in smiles]
    truth_smiles = {"a": smiles[0], "b": smiles[1], "c": smiles[2]}
    sub = {
        "a": [smiles[0], smiles[3]],
        "b": [smiles[3], smiles[4], smiles[1]],
        "c": [smiles[3], smiles[4]],
    }
    official = mrr_at_k(
        SubmissionRow(molecule_id=m, candidate_smiles=sub[m], truth_smiles=truth_smiles[m])
        for m in truth_smiles
    )
    final = attach_canonical_keys(
        pd.DataFrame(
            [
                {"molecule_id": m, "smiles": s, "rank": r}
                for m, ss in sub.items()
                for r, s in enumerate(ss, start=1)
            ]
        )
    )
    truth = {m: canonical_inchikey14(s) for m, s in truth_smiles.items()}
    res = decompose(truth, {"ranker": final}, stages=[StageSpec("ranker", 1)])
    assert keys[0] == truth["a"]
    assert res.mrr == pytest.approx(official)
    assert res.mrr == pytest.approx((1.0 + 1 / 3) / 3)


def test_rank_beyond_k_max_counts_as_miss():
    truth = {"m": "T"}
    final = _long({"m": _filler("X", 30) + ["T"]})
    res = decompose(truth, {"ranker": final}, stages=[StageSpec("ranker", 1)])
    row = res.per_molecule.iloc[0]
    assert math.isnan(row["final_rank"])
    assert row["rank_bin"] == "miss"
    assert res.mrr == 0.0


def test_molecule_absent_from_outputs_is_scored_zero(scenario):
    truth, outputs = scenario
    truth = {**truth, "m_absent": "T_absent"}
    res = decompose(truth, outputs)
    assert res.count == 6
    absent = res.per_molecule.set_index("molecule_id").loc["m_absent"]
    assert absent["bucket"] == "pool"


def test_recovered_flags_fusion_rescue():
    """FP で落ちても最終ランカーが拾ったら recovered として数える。"""
    truth = {"m": "T"}
    outputs = {
        "pool": _long({"m": ["T"]}),
        "fp": _long({"m": _filler("X", 200) + ["T"]}),
        "simulator": _long({"m": ["T"]}),
        "ranker": _long({"m": ["X0000", "T"]}),
    }
    res = decompose(truth, outputs)
    row = res.per_molecule.iloc[0]
    assert row["bucket"] == "fp"
    assert bool(row["recovered"])
    assert row["final_rank"] == 2


def test_classes_and_rank_table(scenario):
    truth, outputs = scenario
    classes = assign_classes(
        truth,
        reference_keys={"T_m_top1"},
        database_keys={"T_m_top1", "T_m_fp", "T_m_sim", "T_m_ranker"},
    )
    assert classes["m_top1"] == "class1"
    assert classes["m_pool"] == "class3"
    assert classes["m_ranker"] == "class2"

    res = decompose(truth, outputs, classes=classes)
    rt = res.rank_table
    assert rt.loc["class1", "rank1"] == 1
    assert rt.loc["class2", "rank4-10"] == 1
    assert rt.loc["class3", "miss"] == 1
    assert rt["contribution"].sum() == pytest.approx(res.mrr)


def test_missing_stage_output_raises(scenario):
    truth, outputs = scenario
    del outputs["simulator"]
    with pytest.raises(ValueError, match="simulator"):
        decompose(truth, outputs)


def test_stage_subset_without_simulator(scenario):
    """simulator 段がないパイプラインでも段を指定すれば動く。"""
    truth, outputs = scenario
    stages = [s for s in default_stages() if s.name != "simulator"]
    res = decompose(truth, outputs, stages=stages)
    assert "simulator" not in set(res.per_molecule["bucket"])
    assert res.stage_table["loss_share"].sum() == pytest.approx(1.0 - res.mrr)


def test_markdown_report_mentions_every_stage(scenario):
    truth, outputs = scenario
    md = to_markdown(decompose(truth, outputs))
    for name in ("pool", "fp", "simulator", "ranker"):
        assert name in md
    assert "MRR@25: 0.2500" in md

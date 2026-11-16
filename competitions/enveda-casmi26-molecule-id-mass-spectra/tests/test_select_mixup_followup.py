"""Preregistered fifth-slot choice must not depend on incomplete scores."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src/mist_formula/select_mixup_followup.py"
spec = importlib.util.spec_from_file_location("select_mixup_followup", SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def rows(*, a: str = "0.479", m1: str = "0.479", m2: str = "0.478", m3: str = "0.477"):
    values = {"A": a, "M1": m1, "M2": m2, "M3": m3}
    return [
        {
            "ref": ref,
            "experiment_id": experiment,
            "status": "SubmissionStatus.COMPLETE",
            "public_score": values[name],
        }
        for name, (ref, experiment) in subject.REFERENCES.items()
    ]


def test_choose_weak_m1_when_cross_isomer_scores_do_not_clear_both_controls():
    assert subject.select(rows(m1="0.481", m2="0.480", m3="0.479"))["selected"] == "weak_M1"


def test_choose_m2_on_displayed_tie_above_both_controls():
    result = subject.select(rows(m2="0.480", m3="0.480"))
    assert result["selected"] == "A75+M225"
    assert result["winner"] == "M2"


def test_choose_higher_m3_when_it_clears_both_controls():
    result = subject.select(rows(m2="0.480", m3="0.481"))
    assert result["selected"] == "A75+M325"
    assert result["public_lb_displayed"]["M3"] == "0.481"


def test_refuse_incomplete_or_mismatched_ledger_entry():
    incomplete = rows(m2="0.480")
    incomplete[2]["status"] = "SubmissionStatus.PENDING"
    with pytest.raises(ValueError, match="not complete"):
        subject.select(incomplete)
    swapped = rows()
    swapped[2]["experiment_id"] = "wrong"
    with pytest.raises(ValueError, match="different experiment"):
        subject.select(swapped)


def test_latest_complete_record_overrides_prior_timeout():
    complete = rows(m2="0.480")
    timeout = dict(complete[2], status="timeout", public_score=None)
    result = subject.select([timeout, *complete])
    assert result["selected"] == "A75+M225"

from __future__ import annotations

import numpy as np
import pytest

from casmi26.analysis.ms_diag import (
    C13,
    CATEGORIES,
    H,
    Peaks,
    ScoreVariant,
    annotate,
    deisotope_mask,
    explain,
    ion_masses,
    tolerance,
)
from casmi26.baseline.chem import PROTON
from casmi26.baseline.fragments import fragment_masses


def _peaks(mz, it, prec, adduct="[M+H]+", mode=1.0):
    it = np.asarray(it, float)
    return Peaks(np.asarray(mz, float), it / it.sum(), prec, adduct, mode)


def _cat(cats, i):
    return CATEGORIES[cats[i]]


def test_isotope_is_separated_from_hydrogen_shift():
    """断片 F の +1.00336 ピークは ¹³C 同位体と判定し、+1H 断片とは数えない。"""
    frag = np.array([100.0, 200.0])
    f_ion = 100.0 + PROTON
    pk = _peaks([f_ion, f_ion + C13, f_ion + H + 0.03, 200.0 + PROTON], [1.0, 0.1, 0.2, 0.5], 300.0)
    cats = annotate(pk, frag)
    assert _cat(cats, 0) == "frag_exact"
    assert _cat(cats, 1) == "isotope_of_peak"
    assert _cat(cats, 2) == "unexplained"  # 0.03 Da ずれた +1H は許容差外
    assert _cat(cats, 3) == "frag_exact"


def test_precursor_isotope_above_and_neutral_loss():
    prec = 301.0
    pk = _peaks(
        [prec, prec + C13, prec + 5.0, prec - 18.010565, 50.0],
        [1.0, 0.2, 0.1, 0.3, 0.05],
        prec,
    )
    cats = annotate(pk, np.array([10.0]))
    assert [_cat(cats, i) for i in range(5)] == [
        "precursor",
        "precursor_isotope",
        "above_precursor",
        "neutral_loss",
        "unexplained",
    ]


def test_adduct_carrier_explains_sodiated_fragment():
    from casmi26.analysis.ms_diag import ADDUCT_CARRIERS

    na = ADDUCT_CARRIERS["[M+Na]+"][0]
    pk = _peaks([150.0 + na], [1.0], 400.0, adduct="[M+Na]+")
    assert _cat(annotate(pk, np.array([150.0])), 0) == "adduct_ion"


def test_ion_masses_negative_mode_and_ppm_tolerance():
    ions = ion_masses(np.array([100.0]), -1.0, [0])
    assert ions[0] == pytest.approx(100.0 - PROTON)
    tol = tolerance(np.array([100.0, 1000.0]), None, 10.0, 0.003)
    assert tol == pytest.approx([0.003, 0.01])


def test_deisotope_mask_drops_weaker_c13_peak():
    pk = _peaks([100.0, 100.0 + C13, 150.0], [1.0, 0.2, 0.5], 300.0)
    assert list(deisotope_mask(pk, np.full(3, 0.01))) == [True, False, True]


def test_explain_v0_matches_v17_and_bond_cap():
    from casmi26.baseline.fragments import explain_score

    frag = fragment_masses("CCOC(=O)C")
    mz = np.array([frag[1] + PROTON, frag[-1] + PROTON, 77.7])
    it = np.array([0.5, 1.0, 0.2])
    pk = _peaks(mz, it, float(frag[-1] + PROTON))
    v0 = ScoreVariant("V0")
    assert explain(pk, frag, 5, v0) == pytest.approx(explain_score(frag, mz, pk.intensity))
    # 結合数が上限を超えると分子全体の質量しか使わない (V17 の挙動)
    capped = explain(pk, frag, 99, v0)
    assert capped < explain(pk, frag, 99, ScoreVariant("V1", max_bonds=120))


def test_three_breaks_add_fragments():
    smi = "CC(C)CC1CCCCC1"
    f2, f3 = fragment_masses(smi, max_breaks=2), fragment_masses(smi, max_breaks=3)
    assert set(np.round(f2, 6)) <= set(np.round(f3, 6))
    assert len(f3) > len(f2)

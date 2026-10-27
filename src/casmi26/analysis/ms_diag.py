"""MS/MS スペクトルのピーク帰属と、MetFrag-lite 系スコアの変種 (EXP-0003)。

ピーク帰属は質量スペクトルの原理に沿った優先順で行い、各ピークを最初に当てはまった
区分に入れる (CATEGORIES の順)。¹³C 同位体の判定を H 移動つき断片より先に置くのは、
+1H (1.00783 Da) と ¹³C (1.00336 Da) の差 0.0045 Da が既定の許容差 0.01 Da より小さく、
H 移動で同位体ピークを「説明できた」ことにしてしまう偽陽性を切り分けるため。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from casmi26.baseline.chem import MASS, PROTON
from casmi26.baseline.spectra import clean

C13 = 1.0033548  # ¹³C − ¹²C
H = MASS["H"]
E = 0.00054857990

# precursor からの中性ロス (転位を伴うものを含む)。名前 → 質量
NEUTRAL_LOSSES: dict[str, float] = {
    "H2O": 18.010565,
    "NH3": 17.026549,
    "CO": 27.994915,
    "CO2": 43.989829,
    "HCOOH": 46.005479,
    "CH3OH": 32.026215,
    "C2H4O2": 60.021129,  # 酢酸
    "C2H2O": 42.010565,  # ケテン (アセチル基)
    "CH2O": 30.010565,
    "CH3•": 15.023475,  # ラジカル脱離 (メトキシ基など)
    "pentose": 132.042259,  # C5H8O4
    "deoxyhexose": 146.057909,  # C6H10O4 (ラムノース等)
    "hexose": 162.052824,  # C6H10O5
    "glucuronic": 176.032088,  # C6H8O6
    "2xH2O": 36.021129,
    "H2O+CO": 46.005479,
}
# 断片からの連続ロス (小さいものだけ)
CONSECUTIVE_LOSSES = {"H2O": 18.010565, "NH3": 17.026549, "CO": 27.994915, "CO2": 43.989829}

# adduct ごとの、断片に残りうる電荷担体 (proton 以外)
ADDUCT_CARRIERS: dict[str, list[float]] = {
    "[M+Na]+": [MASS["Na"] - E],
    "[M+K]+": [MASS["K"] - E],
    "[M+NH4]+": [MASS["N"] + 4 * MASS["H"] - E],
    "[M+CH2O2-H]-": [MASS["C"] + 2 * MASS["H"] + 2 * MASS["O"] - PROTON],
    "[M+Cl]-": [MASS["Cl"] + E],
}

CATEGORIES = (
    "precursor",
    "precursor_isotope",
    "above_precursor",
    "frag_exact",
    "isotope_of_peak",
    "frag_hshift",
    "neutral_loss",
    "consecutive_loss",
    "frag_3break",
    "adduct_ion",
    "unexplained",
)


@dataclass(frozen=True)
class Peaks:
    mz: np.ndarray
    intensity: np.ndarray  # 和が 1 になるよう正規化
    precursor_mz: float
    adduct: str
    mode: float  # +1 / -1


def prepare_peaks(
    mz: np.ndarray, it: np.ndarray, precursor_mz: float, adduct: str, mode: float
) -> Peaks:
    """MetFrag-lite と同じ前処理 (base peak 比 0.2% 未満を除去、上位 256 本、強度の和 = 1)。"""
    m2, i2 = clean(np.asarray(mz, np.float32), np.asarray(it, np.float32), 0.002, 256, 1.0, False)
    return Peaks(np.asarray(m2, float), np.asarray(i2, float), precursor_mz, adduct, mode)


def _match(target: np.ndarray, mz: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """mz の各要素について、target のどれかと tol 以内で一致するか。"""
    if len(target) == 0 or len(mz) == 0:
        return np.zeros(len(mz), bool)
    t = np.sort(target)
    idx = np.searchsorted(t, mz)
    ok = np.zeros(len(mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(t) - 1)
        ok |= np.abs(t[k] - mz) <= tol
    return ok


def _nearest_ppm(target: np.ndarray, mz: np.ndarray) -> np.ndarray:
    t = np.sort(target)
    idx = np.searchsorted(t, mz)
    best = np.full(len(mz), np.inf)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(t) - 1)
        best = np.minimum(best, np.abs(t[k] - mz))
    return best / mz * 1e6


def ion_masses(
    frag_mass: np.ndarray, mode: float, h_shifts: Sequence[int], carriers: Sequence[float] = ()
) -> np.ndarray:
    """中性断片質量 → イオン m/z (1 価)。proton 付加/脱離に加え、任意の電荷担体。"""
    if len(frag_mass) == 0:
        return np.zeros(0)
    charge = PROTON if mode > 0 else -PROTON
    out = [frag_mass + dh * H + charge for dh in h_shifts]
    out += [frag_mass + dh * H + c for c in carriers for dh in h_shifts]
    return np.concatenate(out)


def tolerance(mz: np.ndarray, abs_da: float | None, ppm: float | None, floor: float) -> np.ndarray:
    if abs_da is not None:
        return np.full(len(mz), abs_da)
    assert ppm is not None
    return np.maximum(mz * ppm * 1e-6, floor)


def annotate(
    pk: Peaks,
    frag2: np.ndarray,
    frag3: np.ndarray | None = None,
    *,
    tol_da: float = 0.01,
) -> np.ndarray:
    """各ピークの帰属区分 (CATEGORIES の index) を返す。"""
    mz, n = pk.mz, len(pk.mz)
    tol = np.full(n, tol_da)
    cat = np.full(n, len(CATEGORIES) - 1)
    free = np.ones(n, bool)

    def assign(name: str, hit: np.ndarray) -> None:
        nonlocal free
        sel = free & hit
        cat[sel] = CATEGORIES.index(name)
        free &= ~sel

    p = pk.precursor_mz
    assign("precursor", np.abs(mz - p) <= tol)
    assign("precursor_isotope", (np.abs(mz - p - C13) <= tol) | (np.abs(mz - p - 2 * C13) <= tol))
    assign("above_precursor", mz > p + 0.5)
    assign("frag_exact", _match(ion_masses(frag2, pk.mode, [0]), mz, tol))
    # ¹³C 同位体: 1.00336 下に、より強い「説明済み」ピークがある
    explained = cat != CATEGORIES.index("unexplained")
    iso = np.zeros(n, bool)
    for i in range(n):
        j = np.flatnonzero(np.abs(mz - (mz[i] - C13)) <= tol[i])
        iso[i] = bool(len(j)) and bool((explained[j] & (pk.intensity[j] > pk.intensity[i])).any())
    assign("isotope_of_peak", iso)
    assign("frag_hshift", _match(ion_masses(frag2, pk.mode, [-2, -1, 1, 2]), mz, tol))
    nl_neutral = (p - (PROTON if pk.mode > 0 else -PROTON)) - np.array(
        list(NEUTRAL_LOSSES.values())
    )
    assign("neutral_loss", _match(ion_masses(nl_neutral, pk.mode, [0]), mz, tol))
    consec = np.concatenate([frag2 - v for v in CONSECUTIVE_LOSSES.values()])
    assign("consecutive_loss", _match(ion_masses(consec, pk.mode, [-1, 0, 1]), mz, tol))
    if frag3 is not None:
        assign("frag_3break", _match(ion_masses(frag3, pk.mode, [-2, -1, 0, 1, 2]), mz, tol))
    carriers = ADDUCT_CARRIERS.get(pk.adduct, [])
    if carriers:
        ions = np.concatenate([frag2 + c for c in carriers])
        assign("adduct_ion", _match(ions, mz, tol))
    return cat


def exact_match_ppm(pk: Peaks, frag2: np.ndarray, tol_da: float = 0.01) -> np.ndarray:
    """H 移動なしの断片に 0.01 Da 以内で一致したピークの質量誤差 (ppm)。"""
    ions = ion_masses(frag2, pk.mode, [0])
    if len(ions) == 0 or len(pk.mz) == 0:
        return np.zeros(0)
    ppm = _nearest_ppm(ions, pk.mz)
    return ppm[ppm * pk.mz * 1e-6 <= tol_da]


@dataclass(frozen=True)
class ScoreVariant:
    """MetFrag-lite スコアの変種。V17 は abs_da=0.01, h_shifts=±2, 他すべて False。"""

    name: str
    max_bonds: int = 34
    abs_da: float | None = 0.01
    ppm: float | None = None
    ppm_floor: float = 0.003
    h_shifts: tuple[int, ...] = (-2, -1, 0, 1, 2)
    deisotope: bool = False
    drop_precursor: bool = False
    neutral_losses: bool = False
    adduct_carriers: bool = False


def deisotope_mask(pk: Peaks, tol: np.ndarray) -> np.ndarray:
    """¹³C 同位体と判定したピークを False にするマスク。"""
    keep = np.ones(len(pk.mz), bool)
    for i in range(len(pk.mz)):
        j = np.flatnonzero(np.abs(pk.mz - (pk.mz[i] - C13)) <= tol[i])
        if len(j) and (pk.intensity[j] > pk.intensity[i]).any():
            keep[i] = False
    return keep


@dataclass(frozen=True)
class PreparedPeaks:
    """変種ごとに 1 回だけ作る、候補に依存しない前処理済みピーク。"""

    peaks: Peaks
    mz: np.ndarray
    weight: np.ndarray  # sqrt(強度)
    tol: np.ndarray
    total: float


def prepare_for(pk: Peaks, v: ScoreVariant) -> PreparedPeaks:
    tol = tolerance(pk.mz, v.abs_da, v.ppm, v.ppm_floor)
    keep = np.ones(len(pk.mz), bool)
    if v.deisotope and len(pk.mz):
        keep &= deisotope_mask(pk, tol)
    if v.drop_precursor:
        keep &= pk.mz <= pk.precursor_mz - 0.5
    w = np.sqrt(pk.intensity[keep])
    return PreparedPeaks(pk, pk.mz[keep], w, tol[keep], float(w.sum()))


def explain_prepared(pp: PreparedPeaks, frag: np.ndarray, n_bonds: int, v: ScoreVariant) -> float:
    """sqrt(強度) 重みで説明できたピークの割合 (V17 の explain_score を変種で一般化)。"""
    if pp.total <= 0 or len(pp.mz) == 0:
        return 0.0
    if n_bonds > v.max_bonds and len(frag):
        frag = frag[-1:]  # V17: 結合数が上限を超えると分子全体の質量だけ
    pk = pp.peaks
    carriers = ADDUCT_CARRIERS.get(pk.adduct, []) if v.adduct_carriers else []
    ions = ion_masses(frag, pk.mode, v.h_shifts, carriers)
    if v.neutral_losses and len(frag):
        extra = [frag[-1] - np.array(list(NEUTRAL_LOSSES.values()))]
        extra += [frag - x for x in CONSECUTIVE_LOSSES.values()]
        ions = np.concatenate([ions, ion_masses(np.concatenate(extra), pk.mode, [-1, 0, 1])])
    ok = _match(ions, pp.mz, pp.tol)
    return float(pp.weight[ok].sum() / pp.total)


def explain(pk: Peaks, frag: np.ndarray, n_bonds: int, v: ScoreVariant) -> float:
    return explain_prepared(prepare_for(pk, v), frag, n_bonds, v)

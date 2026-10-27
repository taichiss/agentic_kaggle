"""スペクトル前処理と entropy similarity (numba)。

`haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline` (V17) の Cell 3 を移植。
差分: ライブラリ側スペクトルは検索のたびに clean せず、`clean_library` で 1 回だけ
前処理して保持する (検索結果は同一)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numba import njit, prange

from casmi26.baseline.chem import neutral_mass


@dataclass(frozen=True)
class CleanConfig:
    floor: float = 0.002  # base peak 比 0.2% 未満を捨てる
    topk: int = 256
    power: float = 1.0
    ent_weight: bool = True
    mz_tol: float = 0.01
    # > 0 なら m/z > precursor_mz - precursor_window のピーク (precursor 本体・同位体・
    # precursor より重いピーク) を類似度計算の前に除く。0 は V17 と同じ (除かない)
    precursor_window: float = 0.0


@njit(cache=True, fastmath=True)
def clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i
            j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk :]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v2 = it[idx[i]] ** power
        oi[i] = v2
        s += v2
    if s > 0:
        for i in range(c):
            oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0:
                S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c):
                    oi[i] /= s2
    return om, oi


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0
    j = 0
    n = len(qmz)
    m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    SAB = 0.0
    tot = 0.0
    buf = np.empty(n + m, np.float64)
    b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]
            i += 1
        elif d > tol:
            buf[b] = cp[j]
            j += 1
        else:
            buf[b] = qp[i] + cp[j]
            i += 1
            j += 1
        b += 1
    while i < n:
        buf[b] = qp[i]
        i += 1
        b += 1
    while j < m:
        buf[b] = cp[j]
        j += 1
        b += 1
    for x in range(b):
        tot += buf[x]
    if tot <= 0:
        return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0:
            SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True, parallel=True)
def search(qmz, qp, cand, off, allmz, allin, tol):
    """前処理済みライブラリの cand 行に対する entropy similarity。"""
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]
        b = off[c + 1]
        if b > a:
            out[k] = entropy_sim(qmz, qp, allmz[a:b], allin[a:b], tol)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def search_shift(qmz, qp, cand, off, allmz, allin, tol, shift):
    """直接一致と precursor 質量差だけ shift した一致の大きい方 (analog 伝播用)。"""
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]
        b = off[c + 1]
        if b <= a:
            continue
        cm = allmz[a:b]
        cp = allin[a:b]
        s1 = entropy_sim(qmz, qp, cm, cp, tol)
        sh = shift[k]
        if -0.001 < sh < 0.001:
            out[k] = s1
            continue
        s2 = entropy_sim(qmz, qp, cm + sh, cp, tol)
        out[k] = s1 if s1 > s2 else s2
    return out


@njit(cache=True, parallel=True)
def _clean_counts(off, mz, it, floor, topk, power, ent_weight):
    n = len(off) - 1
    cnt = np.zeros(n, np.int64)
    for r in prange(n):
        om, _ = clean(
            mz[off[r] : off[r + 1]], it[off[r] : off[r + 1]], floor, topk, power, ent_weight
        )
        cnt[r] = len(om)
    return cnt


@njit(cache=True, parallel=True)
def _clean_fill(off, mz, it, new_off, out_mz, out_it, floor, topk, power, ent_weight):
    n = len(off) - 1
    for r in prange(n):
        om, oi = clean(
            mz[off[r] : off[r + 1]], it[off[r] : off[r + 1]], floor, topk, power, ent_weight
        )
        a = new_off[r]
        for i in range(len(om)):
            out_mz[a + i] = om[i]
            out_it[a + i] = oi[i]


def drop_precursor_region(
    off: np.ndarray, mz: np.ndarray, it: np.ndarray, prec: np.ndarray, window: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各スペクトルから m/z > prec - window のピークを除く (flat 配列のまま)。"""
    row = np.repeat(np.arange(len(off) - 1), np.diff(off))
    keep = mz <= (prec[row] - window)
    cnt = np.bincount(row[keep], minlength=len(off) - 1)
    new_off = np.zeros(len(off), np.int64)
    np.cumsum(cnt, out=new_off[1:])
    return new_off, mz[keep], it[keep]


def clean_flat(
    off: np.ndarray, mz: np.ndarray, it: np.ndarray, cfg: CleanConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """flat 配列 (offsets + values) のスペクトル群を一括 clean。"""
    args = (cfg.floor, cfg.topk, cfg.power, cfg.ent_weight)
    cnt = _clean_counts(off, mz, it, *args)
    new_off = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=new_off[1:])
    out_mz = np.empty(new_off[-1], np.float32)
    out_it = np.empty(new_off[-1], np.float32)
    _clean_fill(off, mz, it, new_off, out_mz, out_it, *args)
    return new_off, out_mz, out_it


@dataclass
class CleanLibrary:
    """train.parquet 全行の前処理済みスペクトル。行番号は train.parquet の行順と一致。"""

    off: np.ndarray
    mz: np.ndarray
    it: np.ndarray
    nm: np.ndarray  # neutral mass (adduct 不明なら nan)

    @property
    def n_peaks(self) -> np.ndarray:
        return np.diff(self.off)


def clean_library(train_path: Path, cache_path: Path | None, cfg: CleanConfig) -> CleanLibrary:
    """row group ごとに読んで clean し npz にキャッシュする (全 4 億ピークの一括読込を避ける)。"""
    if cache_path is not None and cache_path.exists():
        z = np.load(cache_path)
        return CleanLibrary(off=z["off"], mz=z["mz"], it=z["it"], nm=z["nm"])
    pf = pq.ParquetFile(train_path)
    offs, mzs, its, nms = [np.zeros(1, np.int64)], [], [], []
    base = 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(
            g, columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct"]
        )
        mzc = t.column("ms2_mzs").combine_chunks()
        itc = t.column("ms2_normalized_intensities").combine_chunks()
        off = mzc.offsets.to_numpy().astype(np.int64)
        off = off - off[0]
        mz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        it = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        prec = t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
        if cfg.precursor_window > 0:
            off, mz, it = drop_precursor_region(off, mz, it, prec, cfg.precursor_window)
        o2, m2, i2 = clean_flat(off, mz, it, cfg)
        offs.append(o2[1:] + base)
        base += int(o2[-1])
        mzs.append(m2)
        its.append(i2)
        add = np.asarray(t.column("adduct").to_pylist(), dtype=object)
        nms.append(neutral_mass(prec, add))
        del t, mzc, itc, mz, it
    lib = CleanLibrary(
        off=np.concatenate(offs),
        mz=np.concatenate(mzs),
        it=np.concatenate(its),
        nm=np.concatenate(nms),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, off=lib.off, mz=lib.mz, it=lib.it, nm=lib.nm)
    return lib

"""PubChem 本体 (約 1.2 億 CID) の質量順索引と、窓ごとの文献量上位 N 件の取り出し (EXP-0009)。

索引は scripts/casmi26_build_pubchem_index.py が PubChem FTP の Extras から作る:
    mass.npy     float64  質量順に並べたモノアイソトピック質量
    cid.npy      int32    同じ順の CID
    lsid.npy     float32  ln(1 + SID 件数)  (登録の多さ)
    lpmid.npy    float32  ln(1 + PubMed 件数)  (文献の多さ)
    smiles.bin   SMILES を改行区切りで CID 順に連結したもの
    smiles_off.npy int64  CID → smiles.bin の開始位置 (長さは次の CID の位置との差)

SMILES は全部メモリに載せず、窓に入った候補の分だけ memmap から読む。
文献量の prior は公開 kernel と同じ 0.25 × (lsid + lpmid)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRIOR_WEIGHT = 0.25


@dataclass(frozen=True)
class PubChemHit:
    cid: int
    smiles: str
    mass: float
    prior: float  # PRIOR_WEIGHT × (lsid + lpmid)


class PubChemIndex:
    def __init__(self, root: Path) -> None:
        self.mass = np.load(root / "mass.npy", mmap_mode="r")
        self.cid = np.load(root / "cid.npy", mmap_mode="r")
        self.prior = PRIOR_WEIGHT * (
            np.load(root / "lsid.npy", mmap_mode="r").astype(np.float32)
            + np.load(root / "lpmid.npy", mmap_mode="r").astype(np.float32)
        )
        # 完全版は CID 添字の位置 (smiles_off.npy)、軽量版は質量順の並び添字 (smiles_pos_off.npy)
        self.by_pos = (root / "smiles_pos_off.npy").exists()
        self.off = np.load(
            root / ("smiles_pos_off.npy" if self.by_pos else "smiles_off.npy"), mmap_mode="r"
        )
        self.smi = np.memmap(root / "smiles.bin", dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self.mass)

    def smiles_of(self, cid: int) -> str:
        """完全版のみ: CID から SMILES。"""
        a, b = int(self.off[cid]), int(self.off[cid + 1])
        return bytes(self.smi[a : b - 1]).decode("ascii", errors="replace")

    def smiles_at(self, pos: int) -> str:
        """質量順の並び添字から SMILES (完全版・軽量版の両方)。"""
        if not self.by_pos:
            return self.smiles_of(int(self.cid[pos]))
        a, b = int(self.off[pos]), int(self.off[pos + 1])
        return bytes(self.smi[a : b - 1]).decode("ascii", errors="replace")

    def top_n(self, target: float, half: float, n: int) -> list[PubChemHit]:
        """target ± half の窓から、文献量 prior の上位 n 件。"""
        lo = int(np.searchsorted(self.mass, target - half, "left"))
        hi = int(np.searchsorted(self.mass, target + half, "right"))
        if hi <= lo or n <= 0:
            return []
        pri = np.asarray(self.prior[lo:hi])
        k = min(n, hi - lo)
        top = np.argpartition(-pri, k - 1)[:k] if k < hi - lo else np.arange(hi - lo)
        top = top[np.argsort(-pri[top], kind="stable")]
        out = []
        for t in top:
            out.append(
                PubChemHit(
                    int(self.cid[lo + t]),
                    self.smiles_at(lo + t),
                    float(self.mass[lo + t]),
                    float(pri[t]),
                )
            )
        return out

    def window_size(self, target: float, half: float) -> int:
        lo = int(np.searchsorted(self.mass, target - half, "left"))
        hi = int(np.searchsorted(self.mass, target + half, "right"))
        return hi - lo


def collect_candidates(
    index: PubChemIndex,
    neutral_masses: list[np.ndarray],
    *,
    pool_ppm: float,
    pool_min_da: float,
    top: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """各 query の窓 (パイプラインと同じ中心・幅) から文献量の上位 top 件を集める。

    返り値: (smiles, mass, prior, 窓の大きさのリスト)。CID で重複を除く。
    """
    hits: dict[int, tuple[str, float, float]] = {}
    sizes: list[int] = []
    for nm in neutral_masses:
        v = nm[np.isfinite(nm)]
        if not len(v):
            continue
        t = float(np.median(v))
        half = max(t * pool_ppm / 1e6, pool_min_da)
        sizes.append(index.window_size(t, half))
        for h in index.top_n(t, half, top):
            hits[h.cid] = (h.smiles, h.mass, h.prior)
    return (
        np.array([v[0] for v in hits.values()], object),
        np.array([v[1] for v in hits.values()], np.float64),
        np.array([v[2] for v in hits.values()], np.float32),
        sizes,
    )

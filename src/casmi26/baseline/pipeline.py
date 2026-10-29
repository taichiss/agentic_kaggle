"""fold-safe な baseline パイプライン (haideptry V17 の Channel 1〜3)。

V17 との差分 (詳細は experiments/EXP-0001/README.md):
    - ライブラリ・analog 代表スペクトル・候補 pool の train 由来構造は、すべて
      ホールドアウト identity group を除いた側だけから作る
    - Channel 4 (FPNet) は使わない: 公開重みは train 全体で学習済みでリークする
    - 学習済みランカー (rank_train.npz の HistGBM) は使わない: 同じく train 全体で
      作られている。代わりに固定重みの線形融合 score = w_lib*lib + w_analog*ap + w_frag*frag
    - analog の Tanimoto は 6,930-bit 複合指紋ではなく Morgan r2 2048-bit
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rdkit import DataStructs

from casmi26.baseline.chem import morgan_fp
from casmi26.baseline.fragments import explain_score, fragment_masses
from casmi26.baseline.spectra import CleanConfig, CleanLibrary, clean, search, search_shift
from casmi26.metrics.breakdown import canonical_inchikey14


@dataclass(frozen=True)
class BaselineConfig:
    ppm_win: float = 8.5  # ライブラリ検索 (Channel 1) の窓。候補窓も pool_ppm 未指定ならこれ
    ppm_fallback: float = 30.0
    # 候補窓: 半幅 = max(target × pool_ppm, pool_min_da)。None / 0 は V17 と同じ
    pool_ppm: float | None = None
    pool_min_da: float = 0.0
    # precursor m/z の系統ずれ (Da)。中性質量を出す前に引く。0 は V17 と同じ
    mz_offset_pos: float = 0.0
    mz_offset_neg: float = 0.0
    analog_win: float = 200.0
    n_analog: int = 100
    sim_power: float = 4.0
    w_lib: float = 1.0
    w_analog: float = 1.0
    w_frag: float = 0.25
    workers: int = 11
    clean: CleanConfig = field(default_factory=CleanConfig)


@dataclass(frozen=True)
class Query:
    molecule_id: str
    truth_key: str
    spectra: Sequence[tuple[np.ndarray, np.ndarray]]  # 生の (m/z, 強度)
    neutral_masses: np.ndarray
    ion_mode: float  # +1 positive / -1 negative (平均)
    precursor_mzs: np.ndarray | None = None  # spectra と同順。precursor_window > 0 で使う


class SpectralIndex:
    """許可された行だけで作るライブラリ検索 (Channel 1) と analog 代表集合 (Channel 2)。"""

    def __init__(self, lib: CleanLibrary, allowed: np.ndarray, row_keys: np.ndarray) -> None:
        self.lib = lib
        self.row_keys = row_keys
        ok = allowed & np.isfinite(lib.nm) & (lib.n_peaks > 0)
        rows = np.flatnonzero(ok)
        o = np.argsort(lib.nm[rows], kind="mergesort")
        self.rows = rows[o]
        self.snm = lib.nm[self.rows]

        # 構造ごとにピーク数最大のスペクトルを 1 本 (V17 build_rep)
        df = pd.DataFrame({"row": rows, "key": row_keys[rows], "np": lib.n_peaks[rows]})
        rep = df.sort_values(["key", "np"], ascending=[True, False]).drop_duplicates("key")
        rep_rows = rep["row"].to_numpy()
        o = np.argsort(lib.nm[rep_rows])
        self.rep_rows = rep_rows[o]
        self.rep_nm = lib.nm[self.rep_rows]
        self.rep_keys = row_keys[self.rep_rows]
        self.keys = frozenset(rep["key"].tolist())

    def lib_hits(
        self, specs: Sequence[tuple[np.ndarray, np.ndarray]], target: float, cfg: BaselineConfig
    ) -> dict[str, float]:
        tol = target * cfg.ppm_win / 1e6
        lo, hi = np.searchsorted(self.snm, [target - tol, target + tol])
        cand = self.rows[lo:hi]
        agg: dict[str, float] = {}
        if len(cand) == 0:
            return agg
        for qm, qp in specs:
            sc = search(qm, qp, cand, self.lib.off, self.lib.mz, self.lib.it, cfg.clean.mz_tol)
            for k, s in zip(self.row_keys[cand], sc, strict=True):
                if s > agg.get(k, -1.0):
                    agg[k] = float(s)
        return agg

    def analogs(
        self, specs: Sequence[tuple[np.ndarray, np.ndarray]], target: float, cfg: BaselineConfig
    ) -> list[tuple[str, float]]:
        lo, hi = np.searchsorted(self.rep_nm, [target - cfg.analog_win, target + cfg.analog_win])
        cand = self.rep_rows[lo:hi]
        if len(cand) == 0:
            return []
        shift = (target - self.rep_nm[lo:hi]).astype(np.float32)
        best = np.zeros(len(cand), np.float32)
        for qm, qp in specs:
            sc = search_shift(
                qm, qp, cand, self.lib.off, self.lib.mz, self.lib.it, cfg.clean.mz_tol, shift
            )
            np.maximum(best, sc, out=best)
        top = np.argsort(-best)[: cfg.n_analog]
        return [(str(self.rep_keys[lo + i]), float(best[i])) for i in top]


class CandidatePool:
    """COCONUT ∪ (ホールドアウト外の train 構造) を中性質量でソートした候補 pool。

    known_key は train 由来構造の scorer_key (公式 RDKit で正規化済み)。
    COCONUT 由来は None で、窓に入ったものだけ canonical_inchikey14 を計算する。
    """

    def __init__(
        self,
        mass: np.ndarray,
        smiles: np.ndarray,
        known_key: np.ndarray,
        source: np.ndarray,
        prior: np.ndarray | None = None,
    ) -> None:
        o = np.argsort(mass, kind="mergesort")
        self.mass = mass[o]
        self.smiles = smiles[o]
        self.known_key = known_key[o]
        self.source = source[o]
        # 文献量の prior (PubChem 由来の候補だけ有限値、他は nan)。EXP-0009
        self.prior = prior[o] if prior is not None else None

    def window(self, target: float, ppm: float, min_da: float = 0.0) -> np.ndarray:
        half = max(target * ppm / 1e6, min_da)
        lo = np.searchsorted(self.mass, target - half, "left")
        hi = np.searchsorted(self.mass, target + half, "right")
        return np.arange(lo, hi)


def _z(x: np.ndarray) -> np.ndarray:
    """query 内で標準化 (分散 0 なら 0)。"""
    sd = float(x.std())
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _pmap(fn: Callable, items: list, workers: int, chunksize: int) -> list:
    if workers <= 1 or len(items) < 200:
        return [fn(x) for x in items]
    # numba のスレッドを抱えた親から fork しないよう spawn を使う
    with mp.get_context("spawn").Pool(workers) as p:
        return p.map(fn, items, chunksize=chunksize)


@dataclass
class BaselineOutput:
    stages: dict[str, pd.DataFrame]
    diag: pd.DataFrame
    database_keys: set[str]
    # 候補ごとの生スコア (学習済みランカー用)。順位は同点を pool 順で割るので特徴量に使わない
    features: pd.DataFrame | None = None


def run_baseline(
    queries: Sequence[Query],
    index: SpectralIndex,
    pool: CandidatePool,
    key_to_smiles: dict[str, str],
    cfg: BaselineConfig,
    log: Callable[[str], None] = print,
) -> BaselineOutput:
    """全 query を処理し、誤差分解用の段出力 (long format) を返す。"""
    c = cfg.clean
    # Phase 1: スペクトル検索と候補窓 (numba)
    per_q = []
    for i, q in enumerate(queries):
        specs = []
        for si, (mz, it) in enumerate(q.spectra):
            if c.precursor_window > 0 and q.precursor_mzs is not None:
                keep = np.asarray(mz) <= q.precursor_mzs[si] - c.precursor_window
                mz, it = np.asarray(mz)[keep], np.asarray(it)[keep]
            qm, qp = clean(
                np.asarray(mz, np.float32),
                np.asarray(it, np.float32),
                c.floor,
                c.topk,
                c.power,
                c.ent_weight,
            )
            if len(qm):
                specs.append((qm, qp))
        nms = q.neutral_masses[np.isfinite(q.neutral_masses)]
        target = float(np.median(nms)) if len(nms) else float("nan")
        lib_hits: dict[str, float] = {}
        analogs: list[tuple[str, float]] = []
        cand = np.zeros(0, np.int64)
        if np.isfinite(target):
            lib_hits = index.lib_hits(specs, target, cfg)
            analogs = index.analogs(specs, target, cfg)
            pool_ppm = cfg.pool_ppm if cfg.pool_ppm is not None else cfg.ppm_win
            cand = pool.window(target, pool_ppm, cfg.pool_min_da)
            if len(cand) == 0:
                cand = pool.window(target, cfg.ppm_fallback)
        per_q.append((target, lib_hits, analogs, cand))
        if (i + 1) % 100 == 0:
            log(f"  spectral search {i + 1}/{len(queries)}")

    # Phase 2: COCONUT 由来候補の canonical key (並列)
    all_cand = np.unique(np.concatenate([p[3] for p in per_q])) if per_q else np.zeros(0, int)
    need = [int(j) for j in all_cand if pool.known_key[j] is None]
    log(f"  canonicalize {len(need):,} COCONUT candidates")
    keys = _pmap(canonical_inchikey14, [str(pool.smiles[j]) for j in need], cfg.workers, 64)
    cand_key: dict[int, str | None] = {int(j): pool.known_key[j] for j in all_cand}
    cand_key.update(dict(zip(need, keys, strict=True)))

    # Phase 3: 断片質量 (並列、MetFrag-lite)
    cand_smiles = sorted({str(pool.smiles[j]) for j in all_cand})
    log(f"  fragment masses for {len(cand_smiles):,} candidates")
    frags = dict(
        zip(cand_smiles, _pmap(fragment_masses, cand_smiles, cfg.workers, 16), strict=True)
    )

    # Phase 4: 指紋 (Tanimoto 用)
    fp_cache: dict[str, DataStructs.ExplicitBitVect | None] = {}

    def fp(smi: str) -> DataStructs.ExplicitBitVect | None:
        if smi not in fp_cache:
            fp_cache[smi] = morgan_fp(smi)
        return fp_cache[smi]

    # Phase 5: 特徴量と融合
    stage_names = ["pool", "ranker", "channel_lib", "channel_analog", "channel_frag"]
    if pool.prior is not None:
        stage_names += ["ranker_prior", "analog_prior", "analog_frag_prior", "prior_only"]
    rows: dict[str, list[tuple[str, str, int, str]]] = {n: [] for n in stage_names}
    diag = []
    feat_rows: list[pd.DataFrame] = []
    for q, (target, lib_hits, analogs, cand) in zip(queries, per_q, strict=True):
        seen: set[str] = set()
        idx, ckeys = [], []
        for j in cand:
            k = cand_key.get(int(j))
            if k is None or k in seen:
                continue
            seen.add(k)
            idx.append(int(j))
            ckeys.append(k)
        n = len(idx)
        if n == 0:
            diag.append((q.molecule_id, target, 0, 0.0, 0.0))
            continue
        lv = np.array([lib_hits.get(k, 0.0) for k in ckeys], np.float32)

        afps, w = [], []
        for k, sim in analogs:
            f = fp(key_to_smiles.get(k, ""))
            if f is not None:
                afps.append(f)
                w.append(max(sim, 0.0))
        ap = np.zeros(n, np.float32)
        if afps:
            wp = np.asarray(w, np.float32) ** cfg.sim_power
            for t, ci in enumerate(idx):
                f = fp(str(pool.smiles[ci]))
                if f is not None:
                    tan = np.asarray(DataStructs.BulkTanimotoSimilarity(f, afps), np.float32)
                    ap[t] = float((tan * wp).max())

        peaks = []
        for mz, it in q.spectra:
            m2, i2 = clean(
                np.asarray(mz, np.float32), np.asarray(it, np.float32), c.floor, c.topk, 1.0, False
            )
            peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
        fr = np.array(
            [
                max(
                    (
                        explain_score(frags[str(pool.smiles[j])], a, b, q.ion_mode, c.mz_tol)
                        for a, b in peaks
                    ),
                    default=0.0,
                )
                for j in idx
            ],
            np.float32,
        )

        score = cfg.w_lib * lv + cfg.w_analog * ap + cfg.w_frag * fr
        # 文献量の prior: 同じ構造キーの PubChem 候補の値。窓の上位 N 件に無い構造は、
        # 上位 N 件の最小値 (= それ以下) とみなす
        pr = np.zeros(n, np.float32)
        if pool.prior is not None:
            best: dict[str, float] = {}
            for j in cand:
                k, v = cand_key.get(int(j)), float(pool.prior[int(j)])
                if k is not None and np.isfinite(v):
                    best[k] = max(best.get(k, -np.inf), v)
            floor = min(best.values()) if best else 0.0
            pr = np.array([best.get(k, floor) for k in ckeys], np.float32)
        feat_rows.append(
            pd.DataFrame(
                {
                    "molecule_id": q.molecule_id,
                    "candidate_key": ckeys,
                    "smiles": [str(pool.smiles[j]) for j in idx],
                    "source": [str(pool.source[j]) for j in idx],
                    "mass_ppm": [(float(pool.mass[j]) / target - 1) * 1e6 for j in idx],
                    "lib": lv,
                    "analog": ap,
                    "frag": fr,
                    "prior": pr,
                }
            )
        )
        ranked = [
            ("pool", np.zeros(n)),
            ("ranker", score),
            ("channel_lib", lv),
            ("channel_analog", ap),
            ("channel_frag", fr),
        ]
        if pool.prior is not None:
            ranked += [
                ("ranker_prior", _z(score) + pr),
                ("analog_prior", _z(ap) + pr),
                ("analog_frag_prior", _z(ap) + 0.5 * _z(fr) + pr),
                ("prior_only", pr),
            ]
        for name, sc in ranked:
            order = np.argsort(-sc, kind="stable")
            rows[name].extend(
                (q.molecule_id, ckeys[o], r, str(pool.smiles[idx[o]]))
                for r, o in enumerate(order, start=1)
            )
        diag.append(
            (q.molecule_id, target, n, float(lv.max()), float(analogs[0][1]) if analogs else 0.0)
        )

    stages = {
        n: pd.DataFrame(v, columns=["molecule_id", "candidate_key", "rank", "smiles"])
        for n, v in rows.items()
    }
    db_keys = {k for k in cand_key.values() if k is not None}
    return BaselineOutput(
        stages=stages,
        diag=pd.DataFrame(
            diag, columns=["molecule_id", "target_mass", "n_candidates", "best_lib", "best_analog"]
        ),
        database_keys=db_keys,
        features=pd.concat(feat_rows, ignore_index=True) if feat_rows else None,
    )

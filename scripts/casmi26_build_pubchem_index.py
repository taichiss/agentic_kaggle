"""PubChem FTP の Extras から、質量順の候補索引と文献量を作る (EXP-0009)。

入力 (data/input/external/pubchem/、https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/):
    CID-Mass.gz    CID, 分子式, モノアイソトピック質量, 精密質量
    CID-SMILES.gz  CID, SMILES
    CID-PMID.gz    CID, PMID, 種別   (行数を CID ごとに数える)
    CID-SID.gz     CID, SID, ...     (同上)

除外: 荷電種 (分子式が + / - で終わる)、複数成分 (SMILES に '.')、質量 100–1500 Da の外。
出力: data/working/casmi26/pubchem_index/ (casmi26.baseline.pubchem の docstring 参照)

使い方:
    uv run python scripts/casmi26_build_pubchem_index.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/input/external/pubchem")
OUT = Path("data/working/casmi26/pubchem_index")
CHUNK = 5_000_000
MASS_RANGE = (100.0, 1500.0)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def chunks(path: Path, usecols: list[int], names: list[str], dtype: dict):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=usecols,
        names=names,
        dtype=dtype,
        chunksize=CHUNK,
        quoting=3,
        na_filter=False,
        on_bad_lines="skip",
    )


def count_by_cid(path: Path, max_cid: int) -> np.ndarray:
    cnt = np.zeros(max_cid + 2, np.int32)
    n = 0
    for df in chunks(path, [0], ["cid"], {"cid": np.int64}):
        c = df["cid"].to_numpy()
        c = c[(c > 0) & (c <= max_cid)]
        cnt += np.bincount(c, minlength=len(cnt)).astype(np.int32)
        n += len(df)
    log(f"  {path.name}: {n:,} rows")
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. 質量 (CID 添字の配列)。最大 CID は先に 1 回走査して決める
    cids, masses, charged = [], [], []
    for df in chunks(
        args.src / "CID-Mass.gz",
        [0, 1, 2],
        ["cid", "formula", "mono"],
        {"cid": np.int64, "formula": str, "mono": np.float64},
    ):
        cids.append(df["cid"].to_numpy())
        masses.append(df["mono"].to_numpy())
        f = df["formula"].str
        charged.append((f.endswith("+") | f.endswith("-")).to_numpy())
    cid_all = np.concatenate(cids)
    max_cid = int(cid_all.max())
    mass = np.full(max_cid + 1, np.nan)
    mass[cid_all] = np.concatenate(masses)
    bad = np.zeros(max_cid + 1, bool)
    bad[cid_all[np.concatenate(charged)]] = True
    del cids, masses, charged, cid_all
    log(f"CID-Mass: max CID {max_cid:,}, charged {int(bad.sum()):,}")

    # 2. SMILES を CID 順に連結。欠番の CID は長さ 0 (次の CID と同じ開始位置) で表す
    off = np.full(max_cid + 2, -1, np.int64)
    has = np.zeros(max_cid + 1, bool)
    pos, last = 0, 0
    with open(args.out / "smiles.bin", "wb") as fo:
        for df in chunks(
            args.src / "CID-SMILES.gz", [0, 1], ["cid", "smiles"], {"cid": np.int64, "smiles": str}
        ):
            df = df[(df["cid"] > 0) & (df["cid"] <= max_cid)]
            c = df["cid"].to_numpy()
            if len(c) == 0:
                continue
            if c[0] <= last or (np.diff(c) <= 0).any():
                raise ValueError("CID-SMILES is expected to be strictly sorted by CID")
            last = int(c[-1])
            ln = df["smiles"].str.len().to_numpy(np.int64) + 1
            off[c] = pos + np.cumsum(ln) - ln
            pos += int(ln.sum())
            multi = df["smiles"].str.contains(".", regex=False).to_numpy()
            bad[c[multi]] = True
            has[c[~multi]] = True
            fo.write(("\n".join(df["smiles"].tolist()) + "\n").encode("ascii", errors="replace"))
    off[max_cid + 1] = pos
    # 欠番 (-1) を後ろの CID の開始位置で埋める
    o = np.where(off < 0, np.iinfo(np.int64).max, off)
    off = np.minimum.accumulate(o[::-1])[::-1]
    np.save(args.out / "smiles_off.npy", off)
    log(f"CID-SMILES: {int(has.sum()):,} single-component structures")

    # 3. 文献量
    lsid = np.log1p(count_by_cid(args.src / "CID-SID.gz", max_cid)).astype(np.float32)
    lpmid = np.log1p(count_by_cid(args.src / "CID-PMID.gz", max_cid)).astype(np.float32)

    # 4. 有効な CID を質量順に並べて保存
    ok = has & ~bad & np.isfinite(mass) & (mass >= MASS_RANGE[0]) & (mass <= MASS_RANGE[1])
    cid = np.flatnonzero(ok).astype(np.int32)
    order = np.argsort(mass[cid], kind="mergesort")
    cid = cid[order]
    np.save(args.out / "cid.npy", cid)
    np.save(args.out / "mass.npy", mass[cid])
    np.save(args.out / "lsid.npy", lsid[cid])
    np.save(args.out / "lpmid.npy", lpmid[cid])
    meta = {
        "source": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_cid": max_cid,
        "n_structures": int(len(cid)),
        "mass_range": MASS_RANGE,
        "excluded": "charged formulas, multi-component SMILES",
        "prior": "0.25 * (ln(1+SID count) + ln(1+PMID count))",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"index: {len(cid):,} structures → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

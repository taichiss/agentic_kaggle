"""PubChem 完全版の索引から、Kaggle に上げる軽量版を作る (EXP-0009)。

質量を BIN_DA 幅の区間に分け、区間ごとに文献量 prior の上位 PER_BIN 件だけを残す。
窓 (±max(5 ppm, 2 mDa)) の上位 100 件は、ほぼ必ずこの中に入る (区間の上位 PER_BIN 件に
入らない構造が窓の上位 100 件に入るのは、窓の端の区間に強い構造が偏ったときだけ)。
SMILES は質量順の並び添字で引く形式 (smiles_pos_off.npy) で保存する。

使い方:
    uv run python scripts/casmi26_slim_pubchem_index.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from casmi26.baseline.pubchem import PubChemIndex

FULL = Path("data/working/casmi26/pubchem_index")
OUT = Path("data/working/casmi26/pubchem_slim")
BIN_DA = 0.001
PER_BIN = 400
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, default=FULL)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--per-bin", type=int, default=PER_BIN)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    full = PubChemIndex(args.full)
    mass = np.asarray(full.mass)
    prior = np.asarray(full.prior, dtype=np.float32)
    b = np.floor(mass / BIN_DA).astype(np.int64)
    order = np.lexsort((-prior, b))  # 区間ごとに prior の降順
    bs = b[order]
    start = np.r_[0, np.flatnonzero(np.diff(bs)) + 1]
    rank = np.arange(len(bs)) - np.repeat(start, np.diff(np.r_[start, len(bs)]))
    keep = np.sort(order[rank < args.per_bin])  # 質量順に戻す
    log(f"keep {len(keep):,} / {len(mass):,} structures")

    np.save(args.out / "mass.npy", mass[keep])
    np.save(args.out / "cid.npy", np.asarray(full.cid)[keep])
    np.save(args.out / "lsid.npy", np.load(args.full / "lsid.npy", mmap_mode="r")[keep])
    np.save(args.out / "lpmid.npy", np.load(args.full / "lpmid.npy", mmap_mode="r")[keep])
    off = np.zeros(len(keep) + 1, np.int64)
    with open(args.out / "smiles.bin", "wb") as fo:
        pos = 0
        for i0 in range(0, len(keep), 1_000_000):
            part = [full.smiles_at(int(p)) + "\n" for p in keep[i0 : i0 + 1_000_000]]
            ln = np.fromiter((len(x) for x in part), np.int64, len(part))
            off[i0 + 1 : i0 + 1 + len(part)] = pos + np.cumsum(ln)
            pos += int(ln.sum())
            fo.write("".join(part).encode("ascii", errors="replace"))
    np.save(args.out / "smiles_pos_off.npy", off)
    meta = json.loads((args.full / "meta.json").read_text())
    meta.update(
        {
            "slim": {"bin_da": BIN_DA, "per_bin": args.per_bin, "kept": int(len(keep))},
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

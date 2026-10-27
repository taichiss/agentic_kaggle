"""EXP-0001 の提出 kernel (A: 3 チャネル固定融合 / B: analog 単独) を生成する。

`src/casmi26` のうち推論に要るモジュールを notebook に埋め込むので、
コード用の Kaggle Dataset は作らない。生成物:
    enveda-casmi26/kernels/<slug>/notebook.ipynb
    enveda-casmi26/kernels/<slug>/kernel-metadata.json

使い方:
    uv run python scripts/casmi26_prepare_kernel.py
    # ローカル試走 (Kaggle と同じ notebook コードを data/input で回す)
    CASMI_INPUT=data/input uv run python enveda-casmi26/kernels/<slug>/run_local.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

OWNER = "yuitomurano"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
DATASETS = [
    "prvsiyan/coconut-casmi26-candidates",
    "dmitriigluzdov/casmi26-identity-disjoint-validation-folds",
    "metric/rdkit-2026-3-3-wheel",
]
KERNELS = {
    "casmi26-exp0001-a-fusion": "fusion",
    "casmi26-exp0001-b-analog": "analog",
}
EMBED = [
    "casmi26/__init__.py",
    "casmi26/metrics/__init__.py",
    "casmi26/metrics/breakdown.py",
    "casmi26/baseline/__init__.py",
    "casmi26/baseline/chem.py",
    "casmi26/baseline/fragments.py",
    "casmi26/baseline/spectra.py",
    "casmi26/baseline/pipeline.py",
    "casmi26/baseline/infer.py",
]
SRC = Path("src")
OUT = Path("enveda-casmi26/kernels")

SETUP = r"""
import glob, os, subprocess, sys, time
from pathlib import Path

T0 = time.time()
ROOTS = [r for r in ("/kaggle/input", os.environ.get("CASMI_INPUT")) if r and os.path.isdir(r)]
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("data/working/kernel")
WORK.mkdir(parents=True, exist_ok=True)


def find(name, must=""):
    hits = [h for r in ROOTS for h in glob.glob(f"{r}/**/{name}", recursive=True) if must in h]
    if not hits:
        raise FileNotFoundError(f"{name} ({must}) not found under {ROOTS}")
    return Path(sorted(hits, key=len)[0])


# 公式スコアラと同じ RDKit 2026.03.3 を入れる (Kaggle 上のみ)
if os.path.isdir("/kaggle/input"):
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    whl = glob.glob("/kaggle/input/**/rdkit-2026.3.3-*.whl", recursive=True)
    whl = [w for w in whl if tag in w]
    if whl:
        pip = [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"]
        subprocess.run(pip + [whl[0]], check=False)
import rdkit
print("rdkit", rdkit.__version__, "python", sys.version.split()[0])

# 埋め込んだ casmi26 パッケージを展開
PKG = WORK / "casmi26_src"
for rel, text in EMBEDDED.items():
    p = PKG / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
sys.path.insert(0, str(PKG))
"""

RUN = r"""
import pandas as pd
from casmi26.baseline.infer import MODES, InputPaths, run_test, to_submission, validate_submission
from casmi26.baseline.pipeline import BaselineConfig

paths = InputPaths(
    train=find("train.parquet", "enveda"),
    test=find("test.parquet", "enveda"),
    sample_submission=find("sample_submission.csv", "enveda"),
    coconut_dir=find("coco_meta.pkl").parent,
    structures=find("structures.parquet", "identity-disjoint-validation-folds"),
)
print(paths)
cfg = BaselineConfig(workers=max(1, (os.cpu_count() or 2)))
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
out = run_test(paths, cfg, log=log)

sample = pd.read_csv(paths.sample_submission)
for mode, stage in MODES.items():
    out.stages[stage].to_parquet(WORK / f"stage_{stage}.parquet", index=False)
sub = to_submission(out.stages[MODES[MODE]], sample)
validate_submission(sub, sample)
sub.to_csv(WORK / "submission.csv", index=False)
out.diag.to_csv(WORK / "diag.csv", index=False)
n_real = out.stages[MODES[MODE]].groupby("molecule_id").size()
n_pad = int((n_real.reindex(sample["molecule_id"]).fillna(0) < 25).sum())
log(f"MODE={MODE} submission.csv {sub.shape} / molecules padded to 25: {n_pad}")
print(out.diag["n_candidates"].describe())
"""


def _code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def build(slug: str, mode: str, git_head: str) -> Path:
    embedded = {rel: (SRC / rel).read_text(encoding="utf-8") for rel in EMBED}
    title = {"fusion": "A: lib + analog + 0.25 frag の固定融合", "analog": "B: analog チャネル単独"}
    cells = [
        _md(
            f"# CASMI26 EXP-0001 {title[mode]}\n\n"
            f"- 生成元: `scripts/casmi26_prepare_kernel.py` (git `{git_head}`)\n"
            "- fold-safe に組み直した公開 baseline (haideptry V17) の Channel 1–3。"
            "FPNet と学習済みランカーは使わない\n"
            "- ローカル CV: fusion np=0.194 / fold0=0.096、analog np=0.535 / fold0=0.598"
        ),
        _code(f"MODE = {mode!r}\nEMBEDDED = {json.dumps(embedded, ensure_ascii=False)}"),
        _code(SETUP),
        _code(RUN),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    d = OUT / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), "utf-8")
    meta = {
        "id": f"{OWNER}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": DATASETS,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    # ローカル試走用: notebook のコードセルをそのまま順に exec する
    (d / "run_local.py").write_text(
        "import json, pathlib\n"
        "nb = json.loads((pathlib.Path(__file__).parent / 'notebook.ipynb').read_text())\n"
        "if __name__ == '__main__':  # spawn した子プロセスで再実行しない\n"
        "    g = {'__name__': '__main__'}\n"
        "    for c in nb['cells']:\n"
        "        if c['cell_type'] == 'code':\n"
        "            exec(''.join(c['source']), g)\n",
        "utf-8",
    )
    return d


def main() -> int:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    for slug, mode in KERNELS.items():
        print(build(slug, mode, head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

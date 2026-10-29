from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _gen():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts/casmi26_prepare_kernel.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_embedded_modules_are_import_closed():
    """notebook に埋め込むモジュールが import する casmi26.* は、すべて埋め込まれている。"""
    gen = _gen()
    embedded = {rel[:-3].replace("/", ".").removesuffix(".__init__") for rel in gen.EMBED}
    needed: set[str] = set()
    for rel in gen.EMBED:
        src = (ROOT / "src" / rel).read_text(encoding="utf-8")
        needed |= set(re.findall(r"^\s*from (casmi26[\w.]*) import", src, re.M))
        needed |= set(re.findall(r"^\s*import (casmi26[\w.]*)", src, re.M))
    assert needed <= embedded, sorted(needed - embedded)


def test_every_kernel_has_a_cv_note_and_known_pool():
    gen = _gen()
    for slug, (mode, pool) in gen.KERNELS.items():
        assert (mode, pool) in gen.CV_NOTE, slug
        assert pool in {"v17", "exp0008", "exp0009"}, slug

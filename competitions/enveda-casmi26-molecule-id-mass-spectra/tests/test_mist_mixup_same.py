"""Physical peak mixing must precede nonlinear spectrum features."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
SOURCE = Path(__file__).resolve().parents[1] / "src/mist_formula"
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location("mist_mix_same", SOURCE / "train_mix_same.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def test_overlap_merges_before_intensity_transform():
    data = object.__new__(subject.Cache)
    data.length = np.array([2, 2])
    data.mz = np.zeros((2, 96), dtype=np.float32)
    data.mz[0, :2] = [50, 70]
    data.mz[1, :2] = [50.0002, 80]
    data.x = np.zeros((2, 96, 4), dtype=np.float16)
    data.x[0, :2, 1] = np.sqrt([1, 0.5])
    data.x[1, :2, 1] = np.sqrt([1, 0.5])
    data.meta = np.zeros((2, 4), dtype=np.float32)
    data.meta[:, 0] = 0.1  # precursor m/z = 100

    features, mzs, count = data.mixed_spectrum(0, 1, 0.5)

    assert count == 3
    np.testing.assert_allclose(mzs[:3], [50.0001, 70, 80], atol=1e-5)
    np.testing.assert_allclose(features[:3, 1] ** 2, [1, 0.25, 0.25], atol=2e-3)
    np.testing.assert_allclose(features[:3, 3], (100 - mzs[:3]) / 1000, atol=1e-4)

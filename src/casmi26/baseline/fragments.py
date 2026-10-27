"""MetFrag-lite: 1〜2 本の結合切断で生じる断片質量がピークをどれだけ説明するか。

baseline V17 Cell 6 の移植 (Kaggle 公開 notebook, Apache 2.0)。
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger

from casmi26.baseline.chem import MASS, PROTON

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

H_ATOM = MASS["H"]


def _mol_graph(smi: str) -> tuple[np.ndarray, list[tuple[int, int]], int] | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    n = m.GetNumAtoms()
    w = np.zeros(n)
    for a in m.GetAtoms():
        w[a.GetIdx()] = MASS.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H_ATOM
    if (w == 0).any():
        return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n


def _components(n: int, bonds: list[tuple[int, int]], drop: set[int]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop:
            continue
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    cur.append(v)
        comps.append(cur)
    return comps


def fragment_masses(smi: str, max_breaks: int = 2, max_bonds: int = 34) -> np.ndarray:
    try:
        g = _mol_graph(smi)
    except Exception:
        return np.zeros(0)
    if g is None:
        return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds:
        return np.array([w.sum()])
    out = {float(w.sum())}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    if max_breaks >= 3:
        for i in range(nb):
            for j in range(i + 1, nb):
                for k in range(j + 1, nb):
                    for c in _components(n, bonds, {i, j, k}):
                        out.add(float(w[c].sum()))
    return np.array(sorted(out))


def explain_score(
    frag_mass: np.ndarray,
    peak_mz: np.ndarray,
    peak_int: np.ndarray,
    mode: float = 1.0,
    tol: float = 0.01,
    h_shifts: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> float:
    """sqrt(強度) 重みで、断片 ion に ±tol で一致したピークの割合。"""
    if len(frag_mass) == 0 or len(peak_mz) == 0:
        return 0.0
    charge = PROTON if mode > 0 else -PROTON
    ion = np.sort(np.concatenate([frag_mass + dh * H_ATOM + charge for dh in h_shifts]))
    w = np.sqrt(np.asarray(peak_int, float))
    tot = w.sum()
    if tot <= 0:
        return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)

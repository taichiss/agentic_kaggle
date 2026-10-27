"""adduct → neutral mass 変換と構造指紋 (baseline V17 Cell 4 / Cell 6 の移植)。"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

MASS = dict(
    C=12.0,
    H=1.00782503207,
    N=14.0030740048,
    O=15.9949146196,
    P=30.97376163,
    S=31.97207100,
    F=18.99840322,
    Cl=34.96885268,
    Br=78.9183371,
    I=126.904473,
    Na=22.9897692809,
    K=38.96370668,
    Si=27.9769265325,
    B=11.0093054,
    Se=79.9165213,
)
E = 0.00054857990
PROTON = MASS["H"] - E
H2O = 2 * MASS["H"] + MASS["O"]
NH4 = MASS["N"] + 4 * MASS["H"]
FORMATE = MASS["C"] + 2 * MASS["H"] + 2 * MASS["O"]
ACETATE = 2 * MASS["C"] + 4 * MASS["H"] + 2 * MASS["O"]

# adduct: (n_M, charge, 付加質量)
ADDUCTS: dict[str, tuple[int, int, float]] = {
    "[M+H]+": (1, 1, PROTON),
    "[M+NH4]+": (1, 1, NH4 - E),
    "[M+Na]+": (1, 1, MASS["Na"] - E),
    "[M+K]+": (1, 1, MASS["K"] - E),
    "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON),
    "[M]+": (1, 1, -E),
    "[M-H2O]+": (1, 1, -E - H2O),
    "[M+CH3OH+H]+": (1, 1, PROTON + MASS["C"] + 4 * MASS["H"] + MASS["O"]),
    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS["C"] + 3 * MASS["H"] + MASS["N"]),
    "[M-H]-": (1, 1, -PROTON),
    "[M-H2O-H]-": (1, 1, -PROTON - H2O),
    "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON),
    "[M+Cl]-": (1, 1, MASS["Cl"] + E),
    "[M]-": (1, 1, E),
    "[M-2H]-": (1, 2, -2 * PROTON),
    "[M+Na-2H]-": (1, 1, MASS["Na"] - 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON),
    "[2M+Na]+": (2, 1, MASS["Na"] - E),
    "[2M+NH4]+": (2, 1, NH4 - E),
    "[2M+K]+": (2, 1, MASS["K"] - E),
    "[2M-H]-": (2, 1, -PROTON),
    "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON),
    "[2M+Na-2H]-": (2, 1, MASS["Na"] - 2 * PROTON),
    "[3M+H]+": (3, 1, PROTON),
    "[3M-H]-": (3, 1, -PROTON),
}


def neutral_mass(mz: np.ndarray, adduct: np.ndarray) -> np.ndarray:
    """precursor m/z と adduct から中性分子の monoisotopic mass。未知 adduct は nan。"""
    out = np.full(len(mz), np.nan)
    ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = ad == a
        if m.any():
            out[m] = (mz[m] * z - d) / n
    return out


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def morgan_fp(smiles: str) -> DataStructs.ExplicitBitVect | None:
    """Morgan r2 2048-bit。baseline は 6,930-bit の複合指紋だが、ここでは Tanimoto 用途に絞る。"""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return _MORGAN.GetFingerprint(mol)

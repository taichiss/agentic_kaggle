import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("data/derived/saner-multipositive/cache-parallel")
spec = pd.read_parquet(
    root / "spectra.parquet", columns=["mol_idx", "adduct_id", "split", "length"]
)
sidecar = Path("data/derived/mist-mixup-20261116/source-metadata")
source = np.load(sidecar / "ingest_id.npy", mmap_mode="r")
kind = np.load(sidecar / "ce_kind.npy", mmap_mode="r")
energy = np.load(sidecar / "ce_ev.npy", mmap_mode="r")
pattern = np.load(
    "data/derived/mist-mixup-20261116/ce-pattern-signatures/pattern_id.npy",
    mmap_mode="r",
)
metadata = np.memmap(root / "metadata.bin", mode="r", dtype="<f4", shape=(len(spec), 4))
source_id = json.loads((sidecar / "categories.json").read_text())["ingest_lib"].index("enveda-180")
selected = np.flatnonzero(
    (source == source_id) & spec.split.eq("validation").to_numpy() & (spec.adduct_id.to_numpy() > 0)
)
frame = spec.iloc[selected].copy()
frame["row"] = selected
frame["kind"] = kind[selected]
frame["energy"] = energy[selected]
frame["pattern"] = pattern[selected]
frame["polarity"] = metadata[selected, 3].astype(int)
frame["view_kind"] = np.where(
    frame.pattern.isin([0, 1]),
    "combo",
    np.where(
        frame.kind.eq(1) & frame.energy.isin([20.0, 40.0, 60.0]),
        frame.energy.astype("Int64").astype(str),
        "other",
    ),
)
frame = frame[frame.view_kind.ne("other")]
group_keys = ["mol_idx", "adduct_id", "polarity"]
groups = frame.groupby(group_keys + ["view_kind"]).row.first().unstack()
groups = groups.dropna(subset=["20", "40", "60", "combo"])
groups = groups.sample(min(200, len(groups)), random_state=20261117)
peaks = np.memmap(root / "peaks.bin", mode="r", dtype="<f2", shape=(len(spec), 96, 4))
precise = np.memmap(
    "data/derived/mist-formula-20261107/precise-mzs.bin",
    mode="r",
    dtype="<f4",
    shape=(len(spec), 96),
)


def vector(row):
    row = int(row)
    length = int(spec.length.iloc[row])
    return {round(float(precise[row, j]), 2): float(peaks[row, j, 1]) ** 2 for j in range(length)}


cosines = []
overlaps = []
for _, views in groups.iterrows():
    single = [vector(views[str(e)]) for e in (20, 40, 60)]
    target = vector(views["combo"])
    synthetic = {
        mass: sum(v.get(mass, 0) for v in single) / 3
        for mass in set().union(*(set(v) for v in single))
    }
    masses = set(synthetic) | set(target)
    left = np.array([synthetic.get(mass, 0) for mass in masses])
    right = np.array([target.get(mass, 0) for mass in masses])
    cosines.append(float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right))))
    overlaps.append(len(set(target) & set(synthetic)) / max(len(target), 1))

print(
    json.dumps(
        {
            "groups": len(groups),
            "cosine_10_50_90pct": np.quantile(cosines, [0.1, 0.5, 0.9]).tolist(),
            "target_peak_overlap_10_50_90pct": np.quantile(overlaps, [0.1, 0.5, 0.9]).tolist(),
            "cosine_over_0_99": int((np.asarray(cosines) > 0.99).sum()),
        }
    )
)

# Formula-conditioned Transformer experiment

`model.py` is the single-file inference module. Its public function is:

```python
score_candidates(spectra: list[dict], smiles: list[str], checkpoint_path, device="cpu")
```

It returns one higher-is-better float per SMILES, in input order. Spectra are
`test.parquet` row dictionaries for a single `molecule_id`. Invalid SMILES score
`-inf`. Model and RDKit fingerprints are cached in-process. Checkpoints contain
the exact CHNO composition catalog and do not need the source CSV at inference.
The score is validated for ranking candidates with the same full molecular
formula. Scores from different formulas have not been calibrated against each
other, so the submission wrapper should rerank only within formula groups.

## Training contract

The experiment reuses `saner-multipositive/cache-parallel`: its 2,430,328
accepted official spectra, 273,833 molecules, molecule-disjoint train/validation
split, and 2048-bit stereo-free tautomer-canonical Morgan fingerprints. The
common panel at `data/derived/model-comparison-20261107/common-panel.parquet`
has 1,000 validation identities and 25 same-formula candidates per identity.
Ties are scored pessimistically. These candidate-pool diagnostics are not the
Kaggle hidden-test MRR.

The shared cache stores m/z as float16, which cannot support 20 ppm formula
assignment. `prepare_mzs.py` replays the official `train.parquet` rows and writes
float32 selected m/z. It requires exact agreement with the cache's peak tensor,
adduct, and length and 1e-6 absolute agreement for float32 metadata; the
identity must additionally match by
InChIKey14 or molecular formula. It verifies all accepted rows and the output
size before replacing the final file. The source file may express ion formulas
with an extra H while the cache stores the neutral structure formula.

Formula candidates are at most three CHNO subformulas per peak and neutral loss,
bounded by the candidate parent composition. The CSV supplies compositions
only; masses are recomputed from monoisotopic atomic masses. Positive and
negative proton/radical ions are considered; Na/K channels are enabled only
for their matching precursor adducts. Doubly charged precursors do not receive
neutral-loss candidates. Peaks without a CHNO match still use raw m/z, intensity,
and neutral-loss Fourier features. The model is trained from random weights
using training molecules only, with fingerprint BCE and same-formula ranking
loss.

Run after the precise-m/z preparation succeeds:

```bash
uv run --offline --with torch==2.10.0+cu129 --with pandas --with pyarrow \
  --with numpy --index https://download.pytorch.org/whl/cu129 \
  --index https://pypi.org/simple python src/mist_formula/train.py \
  --raw-train data/input/train.parquet \
  --cache data/derived/saner-multipositive/cache-parallel \
  --precise-mzs data/derived/mist-formula-20261107/precise-mzs.bin \
  --formula-csv '<user supplied CSV path>' \
  --panel data/derived/model-comparison-20261107/common-panel.parquet \
  --output data/derived/mist-formula-20261107/full --steps 3000
```

The ignored output directory receives a checkpoint, history, and manifest with
source and panel hashes, split, seed, steps, and validation metric. Keep the
checkpoint and any submission artifacts out of Git.

## 2026-11-07 run

The 6,000-step seed-20261107 run completed on the RTX 5070 Ti in about 635
seconds, batch size 128. Final checkpoint SHA-256:
`4c4ec2dd701a13e677be1ee64e8bf3d287d12ffdb5ba73a6c6132d728b5d2f09`.
On the common 1,000-molecule, 25-same-formula-candidate panel, the score path
used by `score_candidates` achieved MRR@25 **0.3011775**, top-1 **0.143**.
An earlier training-loop report of 0.2976066 averaged per-spectrum probabilities
before scoring, whereas inference averages logits before the sigmoid; the
nonlinearity changes ranks. The inference-parity value is the comparable one.
Official-test smoke on the first 12 molecule IDs scored all 300 supplied
candidates in 2.71 seconds on CPU. This validates loading and format, not hidden
test accuracy.

The same-split random8 versus random4+hard4 mining experiment and its
exploratory random4+hard8 arm are specified in
[`docs/mist-hard-negatives-20261107.md`](../../docs/mist-hard-negatives-20261107.md).

## 2026-11-07 step-duration comparison

The 2× and 3× variants retain every setting of the 6,000-step run, including
the seed, 128-molecule batch, learning rate, random same-formula negative
sampling, input cache, and fixed 1,000-molecule panel. One run starts from
random weights and proceeds to 18,000 updates. `--milestone-steps 12000 18000`
preserves and evaluates both checkpoints along that single trajectory. This
avoids treating a weights-only checkpoint as an optimizer-state resume. The
6,000-step control checkpoint and loss history are compared with the original
run to verify that the first part of the trajectory was reproduced. The captured
step-6,000 `last.pt` is byte-identical to the original run's `last.pt` (SHA-256
`656d57dc819b0dc1f92b7a802771b878f49147dc97fd4e7af2e052c6ac567b94`);
all 25 shared logged loss events through step 6,000 match exactly.

The training process was started before later hard-negative edits to
`train.py`. Its exact source is frozen in the ignored run directory as
`data/derived/mist-formula-20261107/steps-18000/train-source.py` (SHA-256
`5ff8378da5ccc0a8ae9f3ad2d317663402f9faf1461a3cad7e0b65db9a2088c7`).
The current on-disk `train.py` may therefore differ from the running source.
The inference module SHA-256 is
`b40d1ab02c8135831f9b17b002a228f133fe7164c8365113744ea5e126a13c50`.

The two Kaggle variants use the same pinned 0.477 parent notebook, formula-local
rank fusion with weight 0.25, and rank-1 protection as the 6,000-step MIST
submission. Each checkpoint has a separate private Dataset and Notebook. The
parent control CSV must match SHA-256
`39309f6c46d25e9c934dad8871d211ad8afebda07f4441c596714cbe7b6fd9b2`
before submission. Local panel MRR is a model-only diagnostic; Public LB
measures the full parent pipeline with the MIST overlay.

The 12,000-step checkpoint SHA-256 is
`a63e61f2d1dc19df98f7acb5a65199e3c3377e56fcb4da1347553ff68db7f6bb`
and its panel MRR@25/top-1 are **0.391255 / 0.221**. The 18,000-step checkpoint
SHA-256 is `c63467c1fa41c5e21162f85f7c0a28460d1ea4d1b815a6adf63b86af365da5da`
and its panel MRR@25/top-1 are **0.422447 / 0.257**. The 6,000-step control
is **0.301177 / 0.143**. These are one-seed observations on the same artificial
candidate panel; Kaggle Public LB is recorded separately.

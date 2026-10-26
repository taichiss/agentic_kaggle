---
competition: enveda-CASMI26-molecule-id-mass-spectra
checked_at: 2026-10-25
tools_covered:
  - MassSpecGym
  - ICEBERG
  - GLACIER
  - FIORA
  - FRIGID
source_urls:
  - https://proceedings.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Paper-Datasets_and_Benchmarks_Track.pdf
  - https://github.com/pluskal-lab/MassSpecGym
  - https://huggingface.co/datasets/roman-bushuiev/MassSpecGym
  - https://arxiv.org/abs/2304.13136
  - https://arxiv.org/abs/2606.29161
  - https://github.com/coleygroup/ms-pred
  - https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1.full
  - https://www.nature.com/articles/s41467-025-57422-4
  - https://www.biorxiv.org/content/10.1101/2024.04.22.590551v1.full
  - https://github.com/BAMeScience/fiora
  - https://zenodo.org/records/16984129
  - https://arxiv.org/abs/2604.16648
  - https://arxiv.org/html/2604.16648v1
  - https://github.com/coleygroup/FRIGID
  - https://zenodo.org/records/23543494
  - https://zenodo.org/records/23303289
  - https://arxiv.org/html/2606.19624v1
  - https://www.sota2.com/research/sota/de-novo-structural-elucidation-on-massspecgym-test
---

# Forward MS Predictors & Benchmarks Survey (CASMI26)

Focus: tools that are **approved for competition use** and that can act as a forward
spectrum simulator for re-ranking SMILES candidates in the CASMI26 pipeline (metric: MRR@25).
SIRIUS/CSI:FingerID is **forbidden** (NIST training data) and is excluded from recommendations.

Public-LB top at time of survey: **0.467**. Public kernels generally retrieve ~700k candidates
by neutral mass (COCONUT + train) and score them with (a) library cosine, (b) analog Tanimoto,
(c) fragment-coverage, (d) a spectrum→fingerprint transformer, fused through a GBDT ranker.
Forward MS predictors enter channel (c) and as a re-rank channel beside the library cosine.

---

## 1. MassSpecGym (benchmark)

- **Paper:** Bushuiev et al., *MassSpecGym: A benchmark for the discovery and identification of molecules*, **NeurIPS 2024 Datasets & Benchmarks (Spotlight)**.
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Paper-Datasets_and_Benchmarks_Track.pdf
- **Code / data:** https://github.com/pluskal-lab/MassSpecGym (MIT), HF dataset `roman-bushuiev/MassSpecGym`.
- **Task:** benchmark covering three MS/MS challenges — (1) *de novo* molecular structure generation, (2) molecule retrieval, (3) spectrum simulation (forward MS). Each with an optional "chemical formula given" bonus track.
- **Dataset:** 231,104 spectra / 28,929 unique molecules. Train 194,119 / val 19,429 / test 17,556. Scaffold-aware split.
- **License:** code MIT; dataset reused per publisher terms (GNPS-sourced, CC-compatible).
- **Leaderboard snapshot (as of 2026-10):**
  - *De novo* (test, exact-match Top-1): **MSFlow 32.0%** (flow-matching), MIST+CDDD 24.1%, **FRIGID ~18%**, DiffMS 2.3%, MADGEN 1.3%, MS-BART 1.1%.
  - *Retrieval, formula-known track* (Hit@1 / Hit@5 / Hit@20): **MIST vFRIGID 53.8 / 65.3 / 74.7**, MIST vDiffMS 46.4 / 54.7 / 66.3, FRIGID (generative retrieval) 45.3 / 58.3 / 69.5, **ICEBERG 36.2 / 60.7 / 78.8**, DiffMS 32.3 / 54.4 / 75.5.
- **Caveat:** the follow-up *MassSpecGym in the Wild* (arXiv 2606.19624, 2026) finds that ≥17/26 downstream papers inflated their numbers via a padding-mask bug in batched MIST; use re-audited scores above rather than original paper tables.
- **Relevance to CASMI26:** the labelled public corpus most CASMI26 kernels implicitly train on. ICEBERG, GLACIER, FRIGID, MIST and MIST-vFRIGID are all MassSpecGym-benchmarked and MassSpecGym-trainable → compliance-safe.

---

## 2. ICEBERG (Coley group — forward MS2)

- **Papers:**
  - Goldman, Li, Coley, *Generating Molecular Fragmentation Graphs with Autoregressive Neural Networks*, **arXiv:2304.13136**, submitted 2023-04-25, revised 2024-01-08. (ICEBERG 1.)
  - Goldman et al., *Neural Spectral Prediction for Structure Elucidation with Tandem Mass Spectrometry* (ICEBERG 2 / 2.1), **bioRxiv 2025-05-28** (https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1.full).
- **Task:** forward MS2 prediction (SMILES → predicted MS/MS peaks + intensities). Used in CASMI26 as a *scoring/re-ranking* channel and inside FRIGID as the cycle-consistency simulator.
- **Architecture:** two-stage. ICEBERG-**Generate**: GNN-based autoregressive fragmentation-graph generator that models each breakage event; ICEBERG-**Score**: predicts intensity + ±6 H-shift rearrangements per fragment. Trained independently.
- **Training data:** NIST'20 (ICEBERG 1 & 2), NIST'23, MassSpecGym (added in v2.1). The *MassSpecGym-only* checkpoint is the compliance-safe one for CASMI26.
- **Released weights:** https://github.com/coleygroup/ms-pred (MIT). MassSpecGym checkpoints `msg_all` and `msg_simulation` on Dropbox; NIST checkpoints only released against NIST license proof — **do not use NIST checkpoints in CASMI26**.
- **Adducts supported (v2.1):** positive `[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH3+H]+`, `[M-H2O+H]+`, `[M-2H2O+H]+`; negative `[M-H]-`, `[M+Cl]-`, `[M-H2O-H]-`, `[M-CO2-H]-`. Encoded as one-hot adduct vector. **Clear advantage over GLACIER for CASMI26's natural-products adduct mix.**
- **Performance:** Top-1 retrieval 40.0% on `[M+H]+` random split (ICEBERG 2.1); 33.5% on scaffold split; Top-10 92%. Retrieval Hit@1/5/20 on MassSpecGym = 36.2 / 60.7 / 78.8 (Table 3 of *In the Wild*).
- **Limitations:** molecules ≤1500 Da; element whitelist (C,H,N,O,S,P,F,Cl,Br,I,Si,B,Se,Fe,Co,As,Na,K); Orbitrap-HCD trained — CASMI26 hidden test is **timsTOF**, so expect domain shift. Rearrangements limited to ±6 H.
- **Inference:** GLACIER paper reports ~8× speedup *over ICEBERG*; ICEBERG itself is tractable (minutes per thousand candidates on A100).

---

## 3. GLACIER (Coley group — forward MS2, single-stage)

- **Paper:** Wang, Wang, Coley, *GLACIER: Rethinking Mass Spectrum Prediction as an Object Detection Problem*, **arXiv:2606.29161** (v1 2026-06-28, v2 2026-10-05).
- **Code / weights:** same repo as ICEBERG → https://github.com/coleygroup/ms-pred. MassSpecGym checkpoint on Dropbox. License: code MIT, paper CC BY-NC-ND 4.0.
- **Task:** forward MS2 prediction, single-shot (no autoregressive fragment-graph expansion).
- **Architecture:** single-stage transformer over the molecular graph, framed as *object detection*: each fragment subgraph is a "box" with a predicted intensity. Supports multi-breakpoint fragmentation in one pass.
- **Training data:** **MassSpecGym only** (hence compliance-safe out-of-the-box).
- **Performance:** Top-1 retrieval **70.0% (with contrastive finetuning) / 69.7% (without)** on MassSpecGym; 52.5 / 38.5% on NIST'20. Previous SOTA on MassSpecGym was 64.0%. ~8× faster inference than ICEBERG.
- **Limitations — IMPORTANT for CASMI26:**
  - Released public checkpoint is trained on `[M+H]+` only. Public-LB kernel analyses have confirmed this is a bottleneck: natural-product spectra with `[M-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M-H2O+H]+` cannot be scored with the public GLACIER weights.
  - Single-step model → may miss sequential/rearrangement-heavy fragmentation that ICEBERG explicitly models.
  - Paper license CC BY-NC-ND 4.0; code is MIT and the derived weights are usable for Kaggle non-commercial work, but **check the kernel-metadata license annotation**.
- **Role in CASMI26:** fastest `[M+H]+` scorer. Best used as a cheap first-pass re-ranker; defer other adducts to ICEBERG / FIORA.

---

## 4. FIORA / FIORA-OS (BAM / Benner group — forward MS/MS, GNN)

- **Paper:** Nowatzky, Russo, Lisec, Kister, Reinert, Muth, Benner, *FIORA: Local neighborhood-based prediction of compound mass spectra from single fragmentation events*, **Nature Communications 2025** (`s41467-025-57422-4`); preprint bioRxiv 2024-04-22.
- **Code:** https://github.com/BAMeScience/fiora — **MIT License**. Weights bundled in `fiora/resources/models`; MSnLib training data on **Zenodo 16984129**. R port on CRAN (`fioRa`).
- **Approval:** confirmed OK by competition host in Kaggle discussion **topic 751373** per project context.
- **Task:** forward MS/MS prediction + retention-time (RT) and collision-cross-section (CCS) estimation (bonus for timsTOF!).
- **Architecture:** 6-layer Relational GCN (RGCN) on the molecular graph; fragmentation is cast as an **edge-level** probability-of-cleavage task per bond. Fragment intensities and bond-stability properties are jointly predicted.
- **Training data:** NIST 2017 + MS-Dial public → 74,401 spectra / 10,692 compounds (≥50% peak coverage, ≤1000 Da). The MIT-licensed release ships the **MS-Dial-trained weights** (NIST excluded) and more recent MSnLib-trained checkpoints (Zenodo v7). → Public weights are compliance-safe for CASMI26.
- **Adducts:** `[M+H]+`, `[M]+`, `[M-H]-`, `[M]-`. **Covers both positive and negative mode.** Fills the gap left by GLACIER's `[M+H]+`-only limitation. Does not natively cover `[M+Na]+`, `[M+NH4]+`, `[M+K]+`.
- **Performance:** cosine similarity vs truth on library test = **0.88 (vs ICEBERG 0.84, CFM-ID 0.81)**; CASMI 2016 **0.87 (vs 0.83 / 0.81)**; CASMI 2022 0.52 vs 0.53 / 0.55 (dropped due to high-coverage spectra needing multi-step cascades).
- **Inference:** ~10k spectra / 5 min on A100 GPU, 20× faster than CPU. "A few seconds" per structure in practice.
- **CCS prediction** is directly relevant to CASMI26 (timsTOF): FIORA can predict CCS for each candidate, giving an *orthogonal* evidence channel most public kernels do not yet use.
- **Limitations:** single-step fragmentation only; weaker on CASMI 2022-type spectra with low primary coverage; superclass coverage diverse but biased toward lipids / small drugs.

---

## 5. FRIGID (Coley group — diffusion decoder on MIST backbone)

- **Paper:** Bohde, Liu, Manjrekar, Lederbauer, Ji, Wang, Coley, *FRIGID: Scaling Diffusion-Based Molecular Generation from Mass Spectra at Training and Inference Time*, **arXiv:2604.16648** (2026-04-17, rev 2026-07-27).
- **Code:** https://github.com/coleygroup/FRIGID — **Apache 2.0** per repo LICENSE (note: paper redistribution is CC BY-NC-SA 4.0; the per-paper phrasing in some write-ups says "CC BY-NC"; **the usable artifact under Apache-2.0 is the code+weights**, so weight-reuse is compliance-safe. Approval per competition host in Kaggle topic **748882**).
- **Weights:** FRIGID decoder + ICEBERG models on **Zenodo 23543494**; MIST encoder checkpoints on **Zenodo 23303289**.
- **Task:** *de novo* structural elucidation from MS/MS → generates SAFE-string SMILES. **Not a forward predictor by itself**, but uses ICEBERG forward predictions for cycle-consistency refinement.
- **Architecture:**
  - **Encoder:** MIST — spectrum → 4096-bit radius-2 Morgan fingerprint.
  - **Decoder:** Masked Diffusion Language Model (MDLM) over BPE-tokenised SAFE strings (vocab 1,880, max 256 tokens). Fingerprint active-bit indices → 3 self-attention layers → **cross-attention into every Transformer block**. Precursor chemical formula injected as a hard constraint.
  - **Inference-time refinement (optional):** ICEBERG simulates MS/MS for each generated candidate; peaks absent in the experimental spectrum (hallucinations) are mapped back to atoms via a 20 ppm atom-level consistency score, those atoms' SAFE tokens are remasked and re-denoised. Typically 10 refinement rounds.
- **Training data:** FRIGID-base DLM pre-trained on **up to 1 B unlabeled molecules** (SAFE corpus); MIST encoder fine-tuned on MassSpecGym / NPLIB1 training splits (test and validation molecules explicitly removed).
- **Performance:**
  - MassSpecGym *de novo* test: Top-1 **18.29% (FRIGID with 10 refinement rounds)**, Top-10 22.0%; FRIGID-base 16.09% Top-1.
  - NPLIB1: Top-1 **25.03%**, Top-10 33.37%; tripling prior SOTA.
  - As a *retrieval* method (via MIST vFRIGID encoder scores): **Hit@1 53.76% / Hit@5 65.32% / Hit@20 74.71% on MassSpecGym retrieval** — this is currently the strongest retrieval score on the public leaderboard.
- **Inference cost:** FRIGID-base 6.58 s/spectrum on 6000 Ada; +10 refinement rounds 41.3 s/spectrum; full inference-time scaling up to 86.5 s/spectrum. **Not usable for all 700k candidates**, but perfectly sized for generating tail-filler candidates for the top ~1k spectra.
- **Adducts:** paper does not restrict adducts explicitly; conditioning is on precursor formula + MIST fingerprint rather than on adduct type. In practice MIST training data (MassSpecGym) covers all adducts in the dataset.
- **Limitations:** depends on accurate formula / fingerprint upstream; refinement latency grows linearly with rounds; no built-in uncertainty estimate.

---

## Cross-cutting analysis (~800 words)

### 1. Division of labour between ICEBERG, GLACIER, FIORA, FRIGID

Three of these four tools solve **forward MS2 prediction (SMILES → spectrum)**; FRIGID is the odd one out — it is an *inverse* decoder that *uses* a forward predictor (ICEBERG) internally.

- **GLACIER** is the single-stage, blazing-fast forward predictor; its MassSpecGym checkpoint is the only one of these models trained **exclusively** on MassSpecGym, so it's the cleanest "no-NIST" scoring channel. Role: cheap, high-throughput re-ranker.
- **ICEBERG** is the physically-grounded, two-stage forward predictor with the broadest **adduct vocabulary** (10+ adducts across positive and negative mode) and explicit H-shift rearrangements. Role: accurate scorer for non-`[M+H]+` candidates and fragment-coverage computation. It also doubles as FRIGID's internal simulator.
- **FIORA** is a GNN edge-level forward predictor with its own weights and MIT license, plus **RT and CCS prediction**. Role: independent spectral score + CCS evidence channel (orthogonal to the Coley-stack).
- **FRIGID** is a diffusion *decoder*; it generates fresh candidate SMILES conditioned on the spectrum. Role: *candidate generation*, not scoring. Produces the structures that ICEBERG/GLACIER/FIORA then rank.

A clean CASMI26 pipeline looks like: COCONUT+train retrieval → GLACIER fast score (for `[M+H]+` partition) + ICEBERG score (for everything else) + FIORA score (independent cross-check, plus CCS evidence) → FRIGID generates tail candidates for spectra where the retrieval pool is weak → GBDT ranker fuses all channels for MRR@25.

### 2. GLACIER's `[M+H]+` restriction — compliance-safe extension

The public GLACIER checkpoint only handles `[M+H]+`. Public-LB analyses confirm this hurts natural-product spectra, which commonly ionise as `[M-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M+K]+`, `[M-H2O+H]+`.

- **ICEBERG 2.1** out-of-the-box covers `[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH3+H]+`, `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M-H]-`, `[M+Cl]-`, `[M-H2O-H]-`, `[M-CO2-H]-`. **Use the MassSpecGym checkpoint** (`msg_all`) — avoid the NIST ones.
- **FIORA** covers `[M+H]+`, `[M]+`, `[M-H]-`, `[M]-`. Positive+negative mode, MIT licence, native CCS prediction. Fills GLACIER's negative-mode gap directly.
- Neither FIORA nor out-of-the-box FRIGID explicitly expose `[M+NH4]+` or `[M+K]+` as discrete adduct tokens; use ICEBERG for those.

So: the compliance-safe route is to **partition candidates by adduct → route to the best model**: `[M+H]+` to GLACIER (speed) or ICEBERG; `[M-H]-` to FIORA + ICEBERG; `[M+Na]+`/`[M+NH4]+`/`[M+K]+` to ICEBERG; FRIGID agnostic to adduct via formula conditioning.

### 3. MassSpecGym leaderboard

- *De novo* (exact-match Top-1): **MSFlow 32% (flow-matching)**, MIST+CDDD 24.1%, FRIGID 18.3%, DiffMS 2.3%. Flow-matching + fingerprint-conditioned decoders dominate; vanilla seq2seq transformers score ~0%.
- *Retrieval* (Hit@1 / @5 / @20): **MIST vFRIGID 53.8 / 65.3 / 74.7**, MIST vDiffMS 46.4 / 54.7 / 66.3, FRIGID 45.3 / 58.3 / 69.5, **ICEBERG 36.2 / 60.7 / 78.8**, DiffMS 32.3 / 54.4 / 75.5. The retrieval winner is a MIST-based diffusion encoder; ICEBERG wins at Hit@20 which is directly MRR@25-relevant.

Caveat: *MassSpecGym In the Wild* (2606.19624) finds ≥17/26 papers inflated numbers due to a MIST padding-mask bug — always re-audit before trusting a headline number.

### 4. FRIGID as a tail-filler

Mechanistically FRIGID is a masked-diffusion decoder generating **SAFE strings** conditioned on (a) formula and (b) a MIST-predicted Morgan fingerprint cross-attended into every transformer block. ICEBERG cycle-consistency flags hallucinated peaks and re-masks the offending atoms for re-denoising (default 10 rounds, 20 ppm tolerance).

Each spectrum yields a **set of structurally diverse candidates** — exactly the shape of distribution you want to fill ranks 20–25 in an MRR@25 submission when retrieval+scoring has already locked in top-10. Latency is 6.6–86 s/spectrum, so running on the top ~1-2k hardest spectra is tractable within Kaggle 12-h GPU budgets.

### 5. Blind spots public kernels miss

Three concrete blind spots:
1. **Negative-mode spectra.** Public GLACIER-only kernels silently fail these. FIORA (`[M-H]-`, `[M]-`) + ICEBERG (`[M-H]-`, `[M+Cl]-`, …) fix this.
2. **timsTOF CCS evidence.** No current public kernel uses CCS, yet the hidden test is **timsTOF-measured**. FIORA predicts CCS natively — this is a free, orthogonal evidence channel.
3. **Structures outside the 712k COCONUT+train pool.** Retrieval-only pipelines are capped by their candidate pool. FRIGID can *generate* candidates de novo (fingerprint-conditioned), especially useful for novel natural products that are below mass-tolerance matches.

---

## actionable_points

Priority order for integration (compliance: MassSpecGym / MSnLib / Apache-2.0 / MIT only; **no NIST, no SIRIUS**):

1. **FIORA forward predictor + CCS head** — ~1.5 person-days.
   - MIT license, bundled weights, covers `[M-H]-` which the public kernels miss.
   - Add as (a) new spectral-cosine channel in GBDT, (b) a CCS-match channel (new, no competitor has it).
   - Expected MRR@25 lift: **+0.015 to +0.030** (negative-mode partition alone ≥5% of test set per metadata; CCS is net-new signal).

2. **ICEBERG 2.1 MassSpecGym checkpoint** (replace GLACIER-only path) — ~1 person-day.
   - Use `msg_all` weights only (avoid NIST). Route `[M+Na]+`, `[M+NH4]+`, `[M+K]+`, `[M-H2O+H]+` through ICEBERG.
   - Expected MRR@25 lift: **+0.010 to +0.020** on the sodiated/ammoniated adduct subset.

3. **FRIGID tail-fill for hard spectra** — ~2–3 person-days.
   - Run FRIGID (10 refinement rounds) only on spectra where current top-5 cosine < threshold or formula candidates <50. Merge generated SMILES into retrieval pool before GBDT.
   - Expected MRR@25 lift: **+0.005 to +0.015** (tail ranks 15–25).

4. **MIST vFRIGID retrieval encoder as a 5th GBDT channel** — ~1 person-day.
   - Precompute MIST fingerprint per spectrum (Apache-2.0 weights, MassSpecGym-trained), score dot-product vs candidate Morgan fingerprints.
   - Expected MRR@25 lift: **+0.010 to +0.020** (currently SOTA on MassSpecGym retrieval Hit@1).

5. **Keep GLACIER as a fast speed-tier scorer** only, limited to `[M+H]+` candidates — 0.5 person-days of refactoring.

Total realistic uplift from this integration pass: **+0.04 to +0.08 MRR@25** on top of the 0.467 public baseline, under conservative stacking assumptions.

**Explicitly forbidden / not recommended:** SIRIUS, CSI:FingerID, NIST-weighted ICEBERG checkpoints, any model whose paper only reports NIST training. "Unverified" items: FIORA license on the Zenodo weights v7 (listed as MIT in GitHub repo, but Zenodo record license tag not independently confirmed in this survey); FRIGID paper's exact license statement (code is Apache-2.0 per repo LICENSE, paper redistribution is CC BY-NC-SA 4.0 — the usable *artifact* is the Apache-2.0 code+weights). Confirm both before shipping a submission.

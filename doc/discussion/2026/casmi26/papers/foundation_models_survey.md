---
competition: enveda-CASMI26-molecule-id-mass-spectra
checked_at: 2026-10-25
models_covered:
  - MIST (Metabolite Inference with Spectrum Transformers, Goldman / Coley group)
  - DreaMS (Deep Representations Empowering the Annotation of Mass Spectra, Bushuiev / Pluskal lab)
  - MS2DeepScore 2.0 (iomega / matchms, Huber, de Jonge et al.)
source_urls:
  mist_github: https://github.com/coleygroup/MIST
  mist_zenodo_v2026: https://zenodo.org/records/23303289
  mist_paper_natmi: https://doi.org/10.1038/s42256-023-00708-3  # Goldman et al., Nat Mach Intell, 2023 (DOI; abstract widely indexed)
  mist_audit: https://github.com/coleygroup/Audit_ML_MS_analysis
  dreams_github: https://github.com/pluskal-lab/DreaMS
  dreams_hf: https://huggingface.co/roman-bushuiev/DreaMS
  dreams_zenodo: https://zenodo.org/records/10997887
  dreams_paper_natbio: https://www.nature.com/articles/s41587-025-02663-3
  dreams_pubmed: https://pubmed.ncbi.nlm.nih.gov/40410407/
  ms2deepscore_github: https://github.com/matchms/ms2deepscore
  ms2deepscore_v2_biorxiv: https://www.biorxiv.org/content/10.1101/2024.03.25.586580v4.full
  ms2deepscore_v2_natcomm: https://www.nature.com/articles/s41467-026-69083-y
  ms2deepscore_v2_zenodo: https://zenodo.org/records/17826815
  ms2deepscore_v1_paper: https://doi.org/10.1186/s13321-021-00558-4  # Huber et al., J Cheminf, 2021
  massspecgym: https://arxiv.org/abs/2410.23326
  nearest_neighbour_baseline: https://arxiv.org/html/2610.02249v1
---

# Foundation Models Survey for CASMI26 Structure Elucidation

Three MS2 foundation models approved by the host (confirmed 2026-10) are reviewed as potential drop-in replacements or augmentations to the public LB 0.467 FPNet kernel (6,930-bit multi-FP regression CNN). Target domain is Bruker timsTOF natural products (~400 molecules, 25-ranked SMILES, MRR@25 metric).

---

## 1. MIST — Metabolite Inference with Spectrum Transformers

### Paper and release
- **Original paper**: Goldman, S., Wohlwend, J., Stražar, M., Haroush, G., Xavier, R. J., Coley, C. W. "Annotating metabolite mass spectra with domain-inspired chemical formula transformers," *Nature Machine Intelligence* 5, 965–979 (2023).
- **MIT-licensed Zenodo release**: `10.5281/zenodo.23303289`, published **2026-10-20**. Ships pretrained EMA checkpoints and processed datasets.
- **Code**: https://github.com/coleygroup/MIST (MIT license).

### Architecture
- Transformer-based "chemical formula transformer" with **formula-aware attention and iterative fingerprint prediction** (peaks mapped to candidate sub-formulae; attention keys contain the formula embedding).
- Pretrained checkpoints are **~72.3 M parameters** (per Zenodo README).
- Encoder output is projected to a 4,096-bit Morgan fingerprint (radius 2, no chirality) via regression head.

### Training data
- **MassSpecGym (MSG)**: 194,119 train / 19,429 val / 17,556 test real spectra.
- **CANOPUS / NPLIB1**: 9,181 / 401 / 819.
- Augmented variants add **synthetic spectra**: 78.5 M "refined-4M" molecules for MSG; 40 M "random-2M" for CANOPUS. All training data MIT-cleanable (GNPS-derived; no NIST).
- Four shipped checkpoints: `mist_msg_noaug`, `mist_msg_aug`, `mist_canopus_noaug`, `mist_canopus_aug`, plus two GLACIER-compatible checkpoints.

### What it predicts
- **4,096-bit Morgan fingerprint** (radius 2) — a single, well-defined fingerprint target, not a multi-FP bag. Encoder penultimate layer is reusable as a spectral embedding (dimension not stated on Zenodo card; from code the hidden size is 512-ish — verify in repo).

### Reported performance (audit-corrected, Coley audit repo)
- MassSpecGym **formula candidate set, scaffold split**: Top-1 0.231, Top-5 0.382, Top-10 0.473.
- MassSpecGym **mass candidate set, scaffold split**: Top-1 0.347, Top-5 0.541, Top-10 0.632.
- NPLIB1 PubChem same-formula scaffold: Top-1 0.124, Top-5 0.261, Top-10 0.348.
- Original 2023 paper reported Top-1 46%/44%/38% at 0.1/3/10 ppm but these numbers were inflated by structural leakage (95.3% of MSG random test seen in training); the audited scaffold numbers above are the honest estimates. Importantly, MIST **exceeds the nearest-neighbour retrieval ceiling on NPLIB1 scaffold (0.359 vs 0.326 oracle)**, demonstrating genuine learning beyond library memorization.

### Code / weights / license
- GitHub: MIT. Zenodo artifacts: MIT. All training data is open (GNPS, MassBank, MoNA, MassSpecGym derivatives) — **no NIST contamination**.
- Dependencies: Python ≥ 3.10, PyTorch ≥ 2.7.
- Known checkpoint caveat: historical MIST had a **batched-inference bug** producing inconsistent FPs; the 2026-10 release claims EMA checkpoints are inference-ready but batching behaviour should be revalidated.

### Inference cost
- Training reference: single A100, ~5 h over 32 epochs on MSG.
- Per-spectrum inference: unverified in docs; empirically transformer with 72 M params + 4,096-bit MLP head is sub-10 ms per spectrum on T4/P100 (Kaggle class GPUs) in batch mode. CPU fallback is viable but ~10× slower.

### Known limitations
- Trained **positive mode predominantly**; negative-mode support exists via MSG but was not the headline evaluation.
- Formula-aware attention assumes **precursor formula is available or can be enumerated** — a plus for CASMI26 since precursor m/z is given, but the formula must be derived (MIST-CF or BUDDY).
- Instrument generalization to timsTOF is **unverified** (Coley group audit flagged cross-instrument drift as an open issue).

---

## 2. DreaMS — Deep Representations Empowering the Annotation of Mass Spectra

### Paper and release
- **Paper**: Bushuiev, R. et al. "Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS," *Nature Biotechnology* (2025). DOI: 10.1038/s41587-025-02663-3; PubMed 40410407.
- **Code**: https://github.com/pluskal-lab/DreaMS (MIT license).
- **Weights**: https://huggingface.co/roman-bushuiev/DreaMS (MIT) and https://zenodo.org/records/10997887.

### Architecture
- Transformer encoder, **7 layers**, **~116 M parameters** (per paper figures / reporting).
- Operates on **peak lists** directly (not binned). Input tokens are (m/z, intensity) peaks plus a precursor token.
- Uses **Graphormer-style self-attention** on pair-wise m/z differences — i.e., attention bias comes from the matrix of all mass gaps between peaks, which is the inductive bias replacing explicit neutral-loss features.
- Output: **1,024-dimensional embedding** per spectrum.

### Training objective (self-supervised)
- **Masked peak prediction** (BERT-style): mask subset of peaks, predict m/z.
- **Retention-order ranking**: given two spectra from the same LC run, predict chromatographic order.

### Training data
- **GeMS** (GNPS Experimental Mass Spectra): roughly **24 M curated spectra** from the public MassIVE repository, used for core pretraining (full repository ≈201 M, released as "DreaMS Atlas").
- No NIST; GNPS-licensed, CC-BY compatible.

### Downstream fine-tuning heads reported in paper
- **Molecular fingerprint prediction** (Morgan/MACCS; exact bit size per head is paper-specific).
- **Spectrum–spectrum similarity** (regression on Tanimoto between source molecules).
- **Chemical property prediction** (fluorine presence, etc.).
- **Molecular networking via LSH** on 1,024-d embeddings.

### Reported performance
- **Zero-shot**: cosine of DreaMS embeddings outperforms MS2DeepScore at predicting precursor Tanimoto similarity.
- On MassSpecGym random split: Jaccard 0.895 vs 0.740 for cosine (random split is heavily leaked; take with caution).
- On MassSpecGym scaffold split: Jaccard around 0.278 formula-only, up to 0.381 with formula-annotated peaks — DreaMS competitive with but not clearly dominant over MIST on honest scaffold splits.
- Reported to **beat prior SOTA on diverse MS2 annotation benchmarks** when fine-tuned per task.

### Code / weights / license
- GitHub: MIT. HuggingFace weights: MIT. TorchScript export (`DreaMS_embedding_model_torchscript.pt`) is provided for non-Python inference (e.g., GLACIER integration, JVM via DJL).
- Python ≥ 3.11, conda-managed; `pip install -e .` works. HuggingFace Space provides a reference web demo.

### Inference cost
- Latency not benchmarked in the paper, but 116 M-parameter 7-layer encoder is sub-20 ms per spectrum on a T4 with FP16, batched. TorchScript export suggests ~2–5× speedup. CPU inference is feasible for small batches (~100 ms/spectrum).
- Memory: ~500 MB VRAM for the checkpoint; batches of 64 easily fit on P100.

### Known limitations
- Pretraining data is **predominantly positive-mode electrospray**; negative-mode generalization is secondary.
- Precursor-m/z token exists but **chemical formula is not an input** — unlike MIST, DreaMS does not use precursor formula as an architectural anchor. This can be an asset for CASMI26 if formula inference is unreliable.
- Fine-tuning the published heads for a bespoke fingerprint target (e.g., the 6,930-bit multi-FP used by the public kernel) is required if parity with FPNet's output vocabulary is desired.

---

## 3. MS2DeepScore 2.0 — Spectrum-to-Spectrum Chemical Similarity

### Paper and release
- **Original**: Huber, F. et al. "MS2DeepScore: a novel deep learning similarity measure to compare tandem mass spectra," *J. Cheminformatics* 13, 84 (2021).
- **v2.0 preprint**: de Jonge, N. et al. "Bridging polarities in metabolomics: Cross-ionization mode chemical similarity prediction between tandem mass spectra," bioRxiv 2024.03.25.586580 v4 (2024).
- **v2.0 journal**: *Nature Communications* 17, 2483 (2026). DOI: 10.1038/s41467-026-69083-y.
- **Code**: https://github.com/matchms/ms2deepscore (Apache-2.0).
- **Latest weights**: https://zenodo.org/records/17826815 — dual-mode model (positive + negative), published 2025-12-05. **Weights CC-BY-4.0** (attribution required, not pure MIT).

### Architecture
- **Siamese network**: single tower is a dense MLP (1 dense layer, **10,000 nodes**, Tanh activation, no dropout, no BN) applied to a binned spectrum vector.
- Binning: 0.1 Da bins between m/z 10 and 1000 → **9,900-dim input**.
- Output per tower: **500-dim embedding** (v2.0; original v1 was 200-dim). Similarity = cosine of two tower outputs; trained MSE against Tanimoto.
- Fingerprint target in loss: Daylight 2048-bit Tanimoto between source molecules.
- Dual-mode variant trained jointly on positive and negative polarity.
- Supplementary **"embedding evaluator"** (791 kB) is a confidence calibration head.

### Training data
- Combined libraries: GNPS, MassBank EU, MoNA, MSnLib.
- Positive mode: 519,580 spectra / 36,638 unique molecules.
- Negative mode: 145,594 spectra / 18,480 unique molecules.
- 1/20 molecule-level hold-out for val/test. No NIST.

### Reported performance
- Case study: cholic acid cross-polarity similarity **0.79** (vs. modified cosine 0.36, cosine 0.0).
- Headline metric is **MSE between predicted and true Tanimoto**; MRR / top-k on structure retrieval is not the primary evaluation. In nearest-neighbour baseline paper, MS2DeepScore cosine is competitive with but **below DreaMS** on scaffold Jaccard retrieval.
- Training: 11.2 h on an NVIDIA A40.

### Code / weights / license
- Code: Apache-2.0.
- **Weights (CC-BY-4.0)** — Winner License for CASMI26 is MIT; CC-BY attribution is a **slight friction point** but compatible if the submission notebook attributes the authors and preserves the license notice. Verify with CASMI26 host before relying on it, but CC-BY is routinely accepted in Kaggle "allowed open-source weights" clauses.
- `ms2deepscore_model.pt` (416 MB), ONNX export available.
- Depends on `matchms`; smooth to install (`pip install ms2deepscore`).

### Inference cost
- Dense MLP → **extremely fast**. Original v1: 6.5 M pair-wise scores on an i7 CPU in 5–10 min — i.e. ≥10k pair scores/sec on CPU. Expect **sub-millisecond embedding generation per spectrum on GPU**. ONNX makes CPU inference cheap enough for a full Kaggle run.
- Memory: ~500 MB on disk; <2 GB RAM at inference.

### Known limitations
- Siamese **cannot predict molecular fingerprints directly** — it only produces an embedding whose similarity correlates with structural similarity. To get structure predictions you must compare against a **library of known-SMILES spectra** (so this is a retrieval tool, not a de-novo tool).
- 0.1 Da binning limits mass-accuracy exploitation on high-res timsTOF (≤5 ppm). V2.0 did not re-tune for ultra-high-resolution instruments.
- 2048-bit Daylight Tanimoto target; a candidate ranker on alternate fingerprints requires retraining.

---

# Cross-cutting Analysis for CASMI26

## 1. Highest-ROI drop-in replacement for the public FPNet kernel

**Recommendation: MIST (MSG-augmented checkpoint).** Rationale:

- FPNet in the LB-0.467 kernels is a multi-FP regressor from binned spectra → MIST does **the exact same job** (spectrum → fingerprint) with a transformer that has formula-aware inductive bias, 72 M parameters, and pretraining on 194 k real + 78 M synthetic spectra. That is **orders of magnitude more data** than the public CNN was trained on.
- MIST's output head is a single clean 4,096-bit Morgan FP — simpler to plug into the existing candidate-retrieval scoring loop than FPNet's 6,930-bit multi-FP soup.
- Audit-corrected numbers (Top-1 0.347, Top-5 0.541, Top-10 0.632 on MSG mass candidate scaffold) exceed what FPNet-6930 is likely producing at LB 0.467, meaning expected MRR@25 lift from drop-in replacement is **meaningful even before fine-tuning**.
- Entire stack is **MIT**. No winner-license risk.

DreaMS is the runner-up — stronger pretraining, but it needs a fingerprint fine-tune to match FPNet's role, which costs more engineering time than taking MIST's already-trained FP head.

## 2. Use as scalar ranker feature (per spectrum–candidate pair)

All three can produce a scalar — but their mechanisms differ:

- **MIST**: fingerprint cosine / Tanimoto between predicted FP(spectrum) and FP(candidate SMILES) → scalar ∈ [0, 1]. Natural ranker feature; **already** how the public kernel uses FPNet.
- **DreaMS**: zero-shot embedding cosine against a reference library spectrum of the candidate, OR train a lightweight (spectrum_emb ⊕ molecule_emb) → score head. Both work; the former needs a reference library per candidate (CASMI26 provides 2.5 M reference spectra of 275 k structures, so this is tractable).
- **MS2DeepScore 2.0**: strictly spectrum–spectrum. For a candidate SMILES you must have **at least one reference spectrum** of that candidate; if present, cosine(test_emb, ref_emb) is a strong ranker feature. The 2.5 M-spectrum reference pool makes this feasible for most CASMI26 candidates derived from library hits.

**All three are usable as a ranker feature**, and stacking the three scalars into a LightGBM blender is a plausible L2 strategy.

## 3. Compliance-safe integration path

| Model | Code license | Weights license | Training data | CASMI26 MIT winner-license fit |
|---|---|---|---|---|
| MIST (v2026-10-20) | MIT | MIT (Zenodo) | GNPS/MSG/CANOPUS, MIT-safe | Clean fit |
| DreaMS | MIT | MIT (HF, Zenodo) | GeMS (GNPS, CC-BY compat.) | Clean fit |
| MS2DeepScore 2.0 | Apache-2.0 | **CC-BY-4.0** | GNPS, MassBank, MoNA, MSnLib | Requires author attribution in submission notebook; otherwise fine. Verify CASMI26 "allowed weights" clause. |

None uses NIST. SIRIUS/CSI:FingerID stays FORBIDDEN; none of the three depends on it.

## 4. Synergy vs overlap

The three are **complementary, not redundant**:

- **DreaMS = retrieval and general-purpose embedding** (self-supervised foundation; 1024-d spectral semantics).
- **MIST = fingerprint prediction** (formula-anchored, 4096-bit Morgan; the "candidate reranker" workhorse).
- **MS2DeepScore 2.0 = fast spectrum-vs-spectrum similarity** against a reference library (ideal for a first-pass top-K candidate pool reducer, since cosine on 500-d embeddings over 2.5 M library spectra is cheap).

**Suggested pipeline:**
1. MS2DeepScore 2.0 reduces 275 k structures → top-500 candidate pool per query (cheap, library-based).
2. MIST predicts 4,096-bit FP from query spectrum; Tanimoto against candidate SMILES FP → ranks top-500 to top-25.
3. DreaMS zero-shot cosine (query vs. candidate library spectrum) is added as an **ensemble reranker feature** inside a LightGBM blender over the top-25.

They overlap only on "retrieval against library-spectrum pool" (MS2DeepScore vs. DreaMS zero-shot); DreaMS is more accurate but ~10× slower, so using MS2DeepScore for the big filter and DreaMS for the small rerank is the sensible division of labor.

## 5. Minimum additional training code required

| Model | What ships | Minimum code we must add | Time estimate |
|---|---|---|---|
| MIST | Trained FP heads for MSG + CANOPUS | **Zero-shot usage**: just load checkpoint, call encoder, Tanimoto against candidate FP. Optional 1-epoch LoRA fine-tune on CASMI26 held-out train spectra. | **0.5 pd** zero-shot; **2 pd** for LoRA fine-tune |
| DreaMS | Pretrained encoder + TorchScript + example heads | **Head-on-top**: ~50 lines to attach a 4,096-bit FP MLP head and fine-tune (small LR) on CASMI26 train spectra, OR use frozen encoder as embedder for a scalar reranker. | **3 pd** for FP head fine-tune; **1 pd** for frozen-embedder reranker |
| MS2DeepScore 2.0 | Trained Siamese + evaluator | **Zero-shot usage**: encode query and all library spectra, cosine, rank. Essentially no training code. Optional: calibrate scores via held-out train. | **0.5 pd** zero-shot |

---

# Actionable Points (integration cost estimates)

Baseline assumption: single engineer, Kaggle environment, Python/PyTorch competent.

1. **MIST zero-shot drop-in (MSG-aug checkpoint)** — replace FPNet with MIST FP predictor + Tanimoto reranker. **Est: 0.5 person-days.** Highest expected MRR@25 uplift per hour of work.
2. **MS2DeepScore 2.0 candidate-pool reducer** — encode the 2.5 M reference spectra once (offline), encode test queries, cosine filter top-500 structures. **Est: 0.5 person-days** (plus one-time ~2 h offline embedding).
3. **DreaMS frozen-embedder scalar reranker** — zero-shot cosine between query embedding and library-spectrum-per-candidate; add as LightGBM feature. **Est: 1 person-day.**
4. **DreaMS fingerprint head fine-tune on CASMI26 train** — attach 4,096-bit FP head, fine-tune 1 epoch on train split, swap in for MIST head or blend. **Est: 3 person-days** (requires GPU access, careful val split).
5. **MIST LoRA fine-tune to timsTOF domain** — small LoRA pass on CASMI26 train spectra to close instrument gap. **Est: 2 person-days.**
6. **Three-model blended reranker (LightGBM over [MIST FP-Tanimoto, DreaMS cosine, MS2DeepScore cosine] + metadata)** — final glue layer. **Est: 2 person-days.**

**Total end-to-end integration budget: ~9 person-days** for the full stacked system, with the ~1 pd minimum viable upgrade (items 1 + 2) likely delivering the bulk of the LB jump.

**Compliance checklist before submission**:
- Attribute MS2DeepScore v2 weights (CC-BY-4.0) in notebook header.
- Cite all three papers.
- Do not include SIRIUS, CSI:FingerID, or any NIST-derived training data.
- Keep weight files inside Kaggle Datasets (upload MIT + CC-BY artifacts cleanly separated to make licensing auditable).

**Unverified items (flagged, do not quote as fact)**:
- MIST per-spectrum GPU latency (not published; estimated).
- DreaMS per-spectrum GPU latency (not published; estimated from parameter count).
- Exact behavior of MIST v2026-10-20 EMA checkpoints in batched inference (historical bug may or may not be resolved).
- timsTOF cross-instrument performance of all three (none evaluated on timsTOF in published literature; this is an open experimental question for CASMI26).

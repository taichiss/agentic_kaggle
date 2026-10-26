---
competition: enveda-CASMI26-molecule-id-mass-spectra
checked_at: 2026-10-25
metric: MRR@25 (reciprocal rank of first hit; 0 if truth not in top-25)
methods_covered:
  - LambdaRank / LambdaMART
  - ListNet / ListMLE
  - Softmax listwise / InfoNCE (DPR-style)
  - WARP (Weston et al.)
  - ApproxNDCG / NeuralNDCG (Qin 2008 / Bruch 2019 / Pobrotyn 2021)
  - Direct IR-metric optimization frameworks
  - CASMI 2016/2017/2022 winners (CSI:FingerID, CSI:IOKR, MS-FINDER)
source_urls:
  - https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
  - https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf
  - https://lightgbm.readthedocs.io/en/latest/Parameters.html
  - https://ffineis.github.io/blog/2021/05/01/lambdarank-lightgbm.html
  - https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf
  - https://icml.cc/Conferences/2008/papers/167.pdf
  - https://aclanthology.org/2020.emnlp-main.550.pdf
  - https://www.ijcai.org/Proceedings/11/Papers/460.pdf
  - https://research.google.com/pubs/archive/37180.pdf
  - https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2008-164.pdf
  - https://dl.acm.org/doi/pdf/10.1145/3331184.3331347
  - https://arxiv.org/pdf/2102.07831
  - https://arxiv.org/pdf/2008.13373
  - https://github.com/allegro/allRank
  - https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/losses/SoftmaxLoss
  - https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/losses/ApproxNDCGLoss
  - https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1
  - https://doi.org/10.3390/metabo9080160
  - https://www.pnas.org/doi/pdf/10.1073/pnas.1509788112
  - https://arxiv.org/pdf/2205.02169
---

# Learning-to-Rank Survey for CASMI26

**Goal**: identify a listwise loss that directly optimizes MRR@25 to replace the
pointwise `objective=binary` baseline currently dominant on the LB (0.44–0.47).

The CASMI26 setup, viewed as an LTR problem:
- **Query** = one unknown MS/MS spectrum (one molecule).
- **Documents** = ≤25 candidate SMILES scored per query (final submission is the top-25 ranked list).
- **Training queries** = each training molecule paired with 1 positive (true SMILES) + N PubChem negatives.
- **Labels** = binary (positive=1, negatives=0).
- **Metric** = MRR@25 = (1 / rank of positive) if positive ∈ top-25, else 0.

The binary + single-positive + rank-of-first-hit structure matters: it collapses
NDCG, MAP, and MRR to **almost identical quantities**, and it fits the dense-retrieval
"one query, one positive, many negatives" template better than the usual
multi-grade LTR template that LambdaMART was designed for.

---

## 1. LambdaRank / LambdaMART

**References.**
- Burges, Shaked, Renshaw et al. "Learning to Rank with Nonsmooth Cost Functions."
  NeurIPS 2006 (original LambdaRank).
- Burges, "From RankNet to LambdaRank to LambdaMART: An Overview."
  Microsoft Research Technical Report **MSR-TR-2010-82**, June 2010.
- Burges et al., "Learning to Rank using an Ensemble of Lambda-Gradient Models,"
  JMLR W&CP, 2011 (Yahoo! LTR Challenge winner).

**1-paragraph summary.** LambdaRank starts from RankNet's pairwise cross-entropy
loss over pairs (i,j) with scores s_i, s_j:

```
C_ij = log(1 + exp(-sigma * (s_i - s_j)))
```

Instead of differentiating a non-smooth IR metric, LambdaRank *defines* the
gradient (the "lambda") as the RankNet pairwise gradient **scaled by |ΔZ|**,
where ΔZ is the change in the target IR metric if items i and j are swapped:

```
lambda_ij = (-sigma / (1 + exp(sigma * (s_i - s_j)))) * |Delta Z_ij|
lambda_i  = sum_{j: rel_i > rel_j} lambda_ij  -  sum_{j: rel_i < rel_j} lambda_ij
```

Z is typically NDCG, which makes misranking at the top of the list dominate the
gradient. LambdaMART is simply LambdaRank with gradient-boosted regression
trees as the scorer. Burges' 2011 paper explicitly notes that **|ΔZ| can be
swapped for any IR metric** (NDCG, MAP, MRR), making LambdaRank
metric-agnostic in principle.

**LightGBM config (verbatim from docs).**
- `objective=lambdarank` — "objective function for lambdarank."
- `lambdarank_truncation_level` (default 30) — "controls the number of
  top-results to focus on during training. ... closely related to the desirable
  cutoff k in the metric NDCG@k. ... optimal setting ... slightly higher than
  k (e.g., k + 3)."
- `lambdarank_norm` (default true) — normalises lambdas across queries.
- `label_gain` (default `0,1,3,7,15,...`) — gain mapping; `gain(label=i) = 2^i-1`.
- `group` / `query` — row counts per query_id; data must be grouped by query_id.
- Labels must be non-negative integers (binary 0/1 is valid).

**`lambda_rank_truncation_level=25` for MRR@25?** Not an exact match but close:
LightGBM's truncation controls the outer loop of pair enumeration — only pairs
where at least one item is in the top-K contribute, and ΔNDCG is computed with
log2(1+rank) discount at NDCG@K. Setting it to 25 (or 28 following Burges'
"k+3" rule) focuses training on the slice that matters for MRR@25.

**Can LambdaRank optimize MRR directly in LightGBM?** The *framework* supports
arbitrary |ΔZ|, but **mainline LightGBM only implements the NDCG delta**
(`src/objective/rank_objective.hpp`, `GetGradientsForOneQuery`, uses
`delta_pair_NDCG`). There is no `objective=mrrrank` and no `lambdarank_metric`
switch — the hard-coded delta is NDCG. Options:
1. **Rely on NDCG@25-weighted gradients as a surrogate for MRR@25.** With
   binary labels and exactly one positive per query, NDCG@25 reduces to
   `1 / log2(1 + rank_of_positive)` (vs MRR's `1 / rank_of_positive`). The two
   quantities are monotonic in rank_of_positive, so a model that improves one
   generally improves the other. This is almost certainly good enough.
2. **Patch LightGBM.** Replace `DCGCalculator`'s discount with `1/rank` and
   recompile. Possible but introduces maintenance cost; not recommended unless
   NDCG-trained lambdarank falls short on CV.
3. **Switch to a framework that exposes the delta.** `allRank`'s `lambdaLoss.py`
   accepts a `weighing_scheme` callable; TF-Ranking's `LambdaWeight` has
   `MRRLambdaWeight`. Both allow true MRR-weighted lambdas.

**Suitability for CASMI26.** **High.** LightGBM lambdarank is a 5-line change
from the existing public kernels, directly addresses the identified gap, and
NDCG@25 is numerically very close to MRR@25 under the binary single-positive
regime. The compound-virtual-screening paper
(https://arxiv.org/pdf/2205.02169) found LambdaMART + enrichment-weighted
lambdas outperformed binary classification baselines on a structurally similar
ligand-ranking task.

---

## 2. ListNet / ListMLE

**References.**
- Cao, Qin, Liu, Tsai, Li. "Learning to Rank: From Pairwise Approach to
  Listwise Approach." ICML 2007. (Microsoft tech report MSR-TR-2007-40.)
- Xia, Liu, Wang, Zhang, Li. "Listwise Approach to Learning to Rank: Theory
  and Algorithm." ICML 2008.

**1-paragraph summary.** ListNet models the probability that an item is ranked
first (the Plackett–Luce "top-one" probability) using a softmax over scores
within a query. Loss is the cross-entropy between the predicted and ground-truth
top-one distributions:

```
P_s(i) = exp(s_i) / sum_j exp(s_j)
P_y(i) = phi(y_i) / sum_j phi(y_j)              # phi monotone, e.g. identity
L_ListNet = - sum_i P_y(i) * log P_s(i)
```

ListMLE replaces the top-one surrogate with the full Plackett–Luce likelihood
of the ground-truth permutation pi* and maximises it:

```
L_ListMLE = - sum_{k=1..n} log( exp(s_{pi*(k)}) / sum_{j>=k} exp(s_{pi*(j)}) )
```

ListMLE is convex, continuous, differentiable, and runs in O(n log n). Both
losses treat each query independently.

**Implementations.**
- `allRank` (PyTorch, Allegro): `allrank/models/losses/listNet.py`,
  `listMLE.py`, `binary_listNet.py`. Also `pointwise`, `rankNet`, `lambdaLoss`,
  `approxNDCG`, `neuralNDCG`, `ordinal`, `bce`.
- `tensorflow_ranking.keras.losses.ListMLELoss`, `SoftmaxLoss`
  (SoftmaxLoss is explicitly identified as a listwise variant equivalent to
  ListNet with log-softmax).

**Cost on 25 candidates × hundreds of thousands of queries.** Trivial. ListNet
and ListMLE are O(n log n) per query. With n=25 (26 if we include the
positive), a single epoch over 300k queries is on the order of 10^7 ops — a
few minutes on a single GPU, or even a modern CPU.

**Suitability for CASMI26.** **High for the ListNet top-one form**; this
collapses, under binary single-positive labels, to **exactly the softmax /
InfoNCE loss** (section 3 below). Vanilla ListMLE needs a full permutation of
labels, which we do not have (we have only one positive), so ListMLE is
awkward here.

---

## 3. Softmax listwise / Contrastive listwise (DPR, InfoNCE, ColBERT style)

**References.**
- Karpukhin, Oguz, Min, Lewis, Wu, Edunov, Chen, Yih. "Dense Passage Retrieval
  for Open-Domain Question Answering." EMNLP 2020.
- van den Oord, Li, Vinyals. "Representation Learning with Contrastive
  Predictive Coding." arXiv:1807.03748 (InfoNCE formulation).
- Khattab, Zaharia. "ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT." SIGIR 2020.

**1-paragraph summary.** With one positive x+ and N negatives {x−_1, …, x−_N}
per query q, DPR / InfoNCE optimises:

```
L_InfoNCE = - log( exp(s(q, x+) / tau) / ( exp(s(q, x+) / tau) + sum_i exp(s(q, x-_i) / tau) ) )
```

This is **exactly ListNet's top-one cross-entropy** when the ground-truth
distribution is a one-hot on the positive. The temperature `tau` is a scaling
knob (equivalent to multiplying scores before softmax). The dense-retrieval
community prefers this over LambdaRank because: (i) they use neural scorers
where single-positive single-query softmax is naturally differentiable; (ii)
they typically train on **implicit** feedback with only positives known, which
exactly matches our CASMI26 setup; (iii) in-batch negatives and hard-negative
mining give very strong training signal; (iv) LambdaMART is tightly coupled to
GBDT trees — awkward to combine with learned embeddings.

**Equivalence to ListNet.** Yes. With binary labels and exactly one positive,
`L_ListNet(top-one)` and `L_InfoNCE` are identical up to the temperature tau.
Different communities, same math.

**Why it dominates in retrieval but not in web search.** Web-search LambdaMART
has graded labels (0..4) and gets a real benefit from NDCG weighting. DPR-style
retrieval has binary labels and 1 positive — exactly our regime — so the
softmax listwise loss is the natural fit and matches the eval metric closely
(recall@k, MRR@k).

**Suitability for CASMI26.** **Very high if we also train a learned scorer**
(e.g., a small MLP over MS/MS features × candidate SMILES features), **lower
if we stick with GBDT** since LightGBM has no first-class listwise softmax
objective and we'd have to implement it ourselves via a custom objective
callback. Still, implementing softmax-CE as a LightGBM custom `fobj` is
~30 lines of NumPy.

---

## 4. WARP (Weighted Approximately Ranked Pairwise)

**References.**
- Weston, Bengio, Usunier. "WSABIE: Scaling Up to Large Vocabulary Image
  Annotation." IJCAI 2011. (The paper that introduced the WARP loss.)
- Usunier, Buffoni, Gallinari. "Ranking with Ordered Weighted Pairwise
  Classification." ICML 2009 (OWPC, the theoretical predecessor).

**1-paragraph summary.** For each training example (q, x+), sample random
negatives x− uniformly until one *violates the margin*: `s(q, x−) + 1 > s(q, x+)`.
If a violator is found after S samples, estimate the positive's current rank as
`rank ≈ floor((N-1) / S)` and apply a hinge-style update scaled by a
rank-decreasing weight L(rank):

```
L_WARP = L(rank) * max(0, 1 + s(q, x-) - s(q, x+))
L(k)   = sum_{j=1..k} 1/j     # harmonic weighting approximates precision@k
```

The harmonic weighting puts more gradient at the top of the list, giving
precision@k-flavoured optimisation. WARP is best known in recommender systems
(implemented in LightFM, Spotlight).

**Implementations.** `LightFM.loss='warp'`, `spotlight.losses.warp_loss`,
various PyTorch gists. No LightGBM support (not applicable — WARP is
inherently stochastic/online).

**Suitability for CASMI26.** **Medium.** WARP optimises a precision@k-like
quantity, not MRR directly, and the online rejection-sampling is less
sample-efficient than softmax CE when we already have N fixed candidates per
query. If we had an infinite candidate pool to subsample from (we do —
PubChem), WARP's hard-negative mining is a plus. In practice, DPR-style
softmax with explicit hard-negative mining dominates WARP on modern retrieval
benchmarks.

---

## 5. Direct MRR optimisation / ApproxNDCG / NeuralNDCG

**References.**
- Qin, Liu, Li. "A General Approximation Framework for Direct Optimization of
  Information Retrieval Measures." Microsoft Tech Report **MSR-TR-2008-164**;
  published in Information Retrieval Journal 2010 (ApproxNDCG, ApproxMAP;
  framework also covers MRR).
- Bruch, Zoghi, Bendersky, Najork. "Revisiting Approximate Metric Optimization
  in the Age of Deep Neural Networks." SIGIR 2019.
- Pobrotyn, Białobrzeski. "NeuralNDCG: Direct Optimisation of a Ranking Metric
  via Differentiable Relaxation of Sorting." arXiv:2102.07831, 2021.
- Bruch, Han, Bendersky, Najork. "A Stochastic Treatment of Learning to Rank
  Scoring Functions." WSDM 2020 (XE_NDCG — another cross-entropy with gain).

**1-paragraph summary.** Qin et al.'s framework replaces the non-differentiable
rank function `pi(i)` with a sigmoid-based approximation:

```
rank(i) ≈ 1 + sum_{j != i} sigmoid( -alpha * (s_i - s_j) )
```

Plug this into any IR metric and backprop. **For MRR this gives**:

```
L_ApproxMRR = - sum_q (1 / rank_approx(positive_q))
```

ApproxNDCG is the same idea with NDCG. Bruch 2019 showed ApproxNDCG, when
properly tuned (alpha, large batch size), **beats LambdaRank, ListMLE and
ListNet on web-search LTR benchmarks**. NeuralNDCG (Pobrotyn 2021) replaces
the sigmoid-rank with a differentiable sorting operator (NeuralSort /
SoftSort), closer to true NDCG at the cost of O(n^2) per query. The Pobrotyn
paper reports NeuralNDCG matches ApproxNDCG when alpha is tuned.

**Direct MRR surrogate.** Qin's framework *does* support MRR — the paper
explicitly derives ApproxMRR. However, I did not find a mainstream
production implementation of ApproxMRR; `allRank` and `TF-Ranking` only ship
ApproxNDCG. Rolling our own ApproxMRR is ~20 lines of PyTorch and is
scientifically defensible because of the Qin paper.

**Suitability for CASMI26.** **High if we train a neural scorer.** With binary
single-positive labels, L_ApproxMRR has an especially simple form (sum over
queries of −1/rank_approx(positive)). Not usable with LightGBM out of the box
without a custom objective.

---

## 6. CASMI 2016 / 2017 / 2022 winners — did they use LTR?

**References.**
- Schymanski, Ruttkies, Krauss et al. "Critical Assessment of Small Molecule
  Identification 2016: Automated methods." *J. Cheminform.* 2017.
- Dührkop, Fleischauer, Ludwig et al. "SIRIUS 4 / CSI:FingerID." PNAS 2015 and
  follow-ups. (CSI:FingerID is the dominant ML pipeline across CASMI 2016+.)
- Brouard et al. "Improved Small Molecule Identification through Learning
  Combinations of Kernel Regression Models." *Metabolites* 2019.
  (CSI:IOKR variants.)
- Tsugawa et al. MS-FINDER (CASMI 2016 Cat. 3 winner — rule-based, not LTR).

**1-paragraph summary.** CASMI 2016 Category 2 (In-silico fragmentation only)
was won by the **CSI:IOKR** team with Input–Output Kernel Regression (41%
challenge wins) — this is kernel-regression over molecular fingerprint output
space, **not** a learning-to-rank method per se, though the final step
(matching predicted fingerprint to PubChem candidates) is implicitly a
ranking. CSI:FingerID (Dührkop et al.) uses an ensemble of SVMs to predict
each fingerprint bit, then scores candidates by Platt-calibrated Tanimoto
similarity — again, pointwise rather than listwise. CASMI 2016 Category 3
("Full Information") was won by **MS-FINDER** (Team Kind, 76% wins) using
rule-based fragment matching. CASMI 2022 results and solutions are listed on
the Fiehn Lab page; the leading automated solutions still trace back to the
CSI/SIRIUS lineage. **To our knowledge, no CASMI winner has used LambdaMART,
ListNet, or softmax listwise loss as the primary ranker.** This is the
competitive gap for CASMI26.

**Suitability for CASMI26.** Prior CASMI winners are a strong feature-engineering
reference (fingerprint prediction, Tanimoto scoring) but **leave the ranker
itself under-optimized** — all of them are either pointwise classifiers or
pointwise regressors. Replacing the final pointwise scoring step with a
listwise LTR objective is a logically clean upgrade that CASMI history
supports.

---

## Cross-cutting recommendation (ROI synthesis)

### 1. Single highest-ROI method

**LightGBM `objective=lambdarank` with `lambdarank_truncation_level=25` (or 28),
grouped by query_id, binary labels, keeping existing features.** Rationale:
- Smallest diff from the current public kernels (same features, same library,
  swap objective + add `group` column).
- Under binary single-positive labels, NDCG@25-weighted lambda gradient and
  MRR@25-weighted lambda gradient are monotonic in `rank_of_positive`;
  improving NDCG@25 improves MRR@25.
- LambdaMART beat RankSVM and GBDT-regression on the ligand-ranking task in
  Furui & Ohue 2022 (arXiv:2205.02169) — a chem-informatics analogue.
- Zero new dependencies, zero re-engineering of the inference pipeline.

Fallback / stretch upgrade if the above saturates: implement **softmax listwise
cross-entropy (= ListNet top-one = InfoNCE)** either as a LightGBM custom
objective (NumPy fobj) or as a small PyTorch MLP head on top of the existing
features. This is the loss that matches CASMI26's structure most tightly.

### 2. Training data construction: how many negatives N?

- **N = 24 negatives per query** so each group has exactly 25 candidates — this
  mirrors the submission format (top-25) and matches `lambdarank_truncation_level=25`.
- Pick negatives by **retrieving top-24 PubChem candidates by precursor mass +
  adduct tolerance**. Hard negatives (chemically similar to the positive) are
  what the ranker needs to learn to separate; random PubChem molecules are
  trivially rejected on precursor mass alone.
- If you have budget, **mix "easy" and "hard" negatives**: 20 hard
  (precursor-matched) + 4 random, roughly matching DPR's hard-negative mining
  recipe. More than ~32 negatives shows diminishing returns in DPR-style
  training and multiplies training cost linearly.
- With ~300k training queries × 25 candidates = 7.5M rows — comfortable for
  LightGBM on a single machine.

### 3. Loss function form (concrete)

**Primary (LightGBM lambdarank):**
```
# For each pair (i,j) within a query with rel_i=1, rel_j=0 (and labels truncated to top-25):
Delta_NDCG_ij = (2^1 - 2^0) * |1/log2(1 + rank_i) - 1/log2(1 + rank_j)| / IdealDCG
lambda_ij     = -sigma / (1 + exp(sigma * (s_i - s_j))) * |Delta_NDCG_ij|
grad_i        = sum over j (sign-adjusted lambda_ij)
```

**Fallback (PyTorch softmax listwise):**
```
# scores: (B, K) tensor of candidate scores per query; positive is always index 0
# L = - log softmax(scores)[:, 0] = log sumexp(scores) - scores[:, 0]
loss = torch.logsumexp(scores / tau, dim=1) - scores[:, 0] / tau
```

**Research / stretch (ApproxMRR, custom):**
```
# scores (B, K); positive at index 0; alpha ~ 10
rank_approx_pos = 1 + sum_{j != 0} sigmoid(-alpha * (scores[:, 0] - scores[:, j]))
loss = -(1 / rank_approx_pos).mean()
```

### 4. Expected MRR lift vs pointwise baseline

Direct CASMI benchmarks don't exist. Reference points:
- Furui & Ohue 2022 (ligand virtual screening, LambdaMART vs GBDT regression):
  "lambdarank" improved NDCG@k over regression baselines across all tested
  datasets; absolute lifts of 0.02–0.05 NDCG.
- Bruch 2019 (web search, ApproxNDCG vs LambdaMART): ApproxNDCG +0.3–1% NDCG.
- DPR 2020 (QA retrieval, softmax listwise vs BM25 pointwise): large (+10 to
  +20 point) MRR@20 gains, but confounded with representation learning.

A realistic planning number: **+0.02 to +0.05 absolute MRR@25** on CASMI26 from
swapping `binary` → `lambdarank` *alone* (features unchanged). Larger lifts
require also upgrading the feature set with harder negatives and
query-specific normalisation. From the LB 0.44–0.47 band, this would put a
clean lambdarank submission around **0.47–0.52**, i.e. competitive without any
feature-engineering heroics.

### 5. Integration pattern — can we just swap the objective?

Almost, but not quite. The existing public kernels are structured as:

```
X_train: (n_candidates, n_features)   # one row per candidate across all queries
y_train: binary (1 = truth, 0 = other)
clf = HistGradientBoostingClassifier() or LGBMClassifier(objective='binary')
clf.fit(X_train, y_train)
scores_per_candidate = clf.predict_proba(X_test)[:, 1]
```

To switch to lambdarank you need to:

1. Add a `query_id` column (one id per training molecule).
2. **Sort the rows by `query_id`** so all rows of a query are contiguous.
3. Compute `group = X_train.groupby('query_id').size().values` — the per-query
   row counts LightGBM expects.
4. Replace `LGBMClassifier(objective='binary')` with
   `LGBMRanker(objective='lambdarank', metric='ndcg', ndcg_eval_at=[25],
   lambdarank_truncation_level=25, label_gain=[0,1])`.
5. Call `ranker.fit(X_train, y_train, group=group)`.
6. Inference is unchanged: `ranker.predict(X_test)` returns per-row scores;
   sort within each test query to produce the top-25 submission.

This is a **~15-line diff** on any of the public kernels. The feature table
only needs the extra `query_id` column; feature engineering is reusable.

---

## actionable_points — code sketches

### A. Drop-in LightGBM `lambdarank` config

```python
import lightgbm as lgb
import pandas as pd

df = (
    df.sort_values("query_id")
      .reset_index(drop=True)
)
group = df.groupby("query_id", sort=False).size().values

feature_cols = [c for c in df.columns if c not in ("query_id", "label")]
X = df[feature_cols]
y = df["label"].astype(int)   # 0 or 1

params = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[25],
    lambdarank_truncation_level=25,     # match MRR@25 cutoff
    lambdarank_norm=True,
    label_gain=[0, 1],                  # binary labels
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    n_estimators=2000,
    early_stopping_round=100,
    verbose=-1,
)

ranker = lgb.LGBMRanker(**params)
ranker.fit(
    X, y,
    group=group,
    eval_set=[(X_val, y_val)],
    eval_group=[group_val],
    eval_at=[25],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
)

# Inference: per-query top-25 by predicted score (descending)
test_df["score"] = ranker.predict(test_df[feature_cols])
submission = (
    test_df.sort_values(["query_id", "score"], ascending=[True, False])
           .groupby("query_id")
           .head(25)
)
```

### B. PyTorch softmax listwise (ListNet top-one / InfoNCE) loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxListwiseLoss(nn.Module):
    """
    scores: (B, K) tensor, scores[:, 0] is the positive, scores[:, 1:] negatives.
    Equivalent to ListNet top-one CE and to InfoNCE with 1 positive + (K-1) negatives.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.tau = temperature

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        logits = scores / self.tau
        # logsumexp - positive_logit, averaged over queries
        return (torch.logsumexp(logits, dim=1) - logits[:, 0]).mean()

# Example usage with a small MLP scorer over (query_feats, candidate_feats):
class PairMLP(nn.Module):
    def __init__(self, d_in, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, pair_feats):           # (B, K, d_in)
        B, K, D = pair_feats.shape
        s = self.net(pair_feats.reshape(B * K, D)).reshape(B, K)
        return s

loss_fn = SoftmaxListwiseLoss(temperature=1.0)
model   = PairMLP(d_in=X_train.shape[-1])
opt     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for batch in loader:                          # batch["pairs"]: (B, 25, D)
    scores = model(batch["pairs"])            # (B, 25); index 0 is positive
    loss   = loss_fn(scores)
    opt.zero_grad(); loss.backward(); opt.step()
```

### C. PyTorch ApproxMRR (research / stretch; Qin 2008 framework)

```python
class ApproxMRRLoss(nn.Module):
    """
    Smooth MRR surrogate. scores: (B, K), positive at column 0.
    alpha controls sharpness; alpha -> infty recovers true argmax ranking.
    """
    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        pos    = scores[:, 0:1]                      # (B, 1)
        others = scores[:, 1:]                       # (B, K-1)
        # Smoothed rank of positive among all candidates:
        rank_approx = 1.0 + torch.sigmoid(-self.alpha * (pos - others)).sum(dim=1)
        return -(1.0 / rank_approx).mean()
```

### D. LightGBM custom softmax listwise objective (if sticking with GBDT)

```python
import numpy as np

def softmax_listwise_fobj(preds, dataset):
    """
    LightGBM custom objective: softmax cross-entropy per query group.
    Requires dataset.get_group() to return per-query row counts and the
    first row of each group to be the positive (label=1).
    Returns (grad, hess) each of shape (n_rows,).
    """
    y = dataset.get_label()
    groups = dataset.get_group()
    grad = np.zeros_like(preds, dtype=np.float64)
    hess = np.zeros_like(preds, dtype=np.float64)

    start = 0
    for g in groups:
        end = start + int(g)
        s = preds[start:end]
        p = np.exp(s - s.max())
        p /= p.sum()
        # Gradient of softmax CE: p - y
        grad[start:end] = p - y[start:end]
        # Diagonal of Hessian: p * (1 - p)
        hess[start:end] = np.clip(p * (1.0 - p), 1e-6, None)
        start = end
    return grad, hess

params = dict(
    objective=softmax_listwise_fobj,
    metric="None",
    num_leaves=63,
    learning_rate=0.05,
    verbose=-1,
)
# Data must be sorted by query_id and the positive placed first within each group.
```

---

## Final verdict

- **Day-1 submission upgrade**: swap `objective=binary` →
  `objective=lambdarank`, add `query_id` + `group`, set
  `lambdarank_truncation_level=25`, `label_gain=[0,1]`. Expect +0.02–0.05 MRR
  with no feature changes.
- **Day-3 upgrade if lambdarank saturates**: implement
  `softmax_listwise_fobj` as a LightGBM custom objective (section D) — this
  is the loss that is mathematically closest to MRR@25 under binary
  single-positive labels.
- **Research direction if we move to a neural scorer**: PyTorch
  softmax-listwise (section B) is strictly dominant over LightGBM for learned
  embeddings; ApproxMRR (section C) is the principled MRR-direct surrogate if
  softmax-listwise plateaus.
- **Do not** try to patch LightGBM's `|ΔZ|` to use `1/rank` instead of
  `1/log2(1+rank)` unless CV evidence demands it. NDCG@25 ≈ MRR@25 for
  single-positive binary queries.

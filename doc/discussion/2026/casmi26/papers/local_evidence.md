---
competition: enveda-CASMI26-molecule-id-mass-spectra
checked_at: 2026-10-25
source: ローカルで取得した外部 Kaggle Dataset (Dmitrii Gluzdov 公開) の README / TECHNICAL_NOTES / SUMMARY.json / 実データ
---

# CASMI26 ローカル実地調査サマリ

> 本ファイルは公開論文のサーベイではなく、**取得した Kaggle Dataset の中に書かれていた一次情報の抜粋とローカル検証結果**。
> 文献サーベイ (`foundation_models_survey.md`, `forward_predictors_survey.md`, `ltr_survey.md`) とは独立に、
> CASMI26 公開エコシステムが提供するベースライン数字と CV 設計のソースドキュメントをまとめる。

## 1. CV 設計の正本仕様 (Dmitrii Gluzdov, 2026-10-19)

ソース: `data/input/external/casmi26/casmi26-identity-disjoint-validation-folds/TECHNICAL_NOTES.md`

### 分割アルゴリズム
- **RDKit 2026.03.3** (公式スコアラ pin)
- `rdMolStandardize.TautomerEnumerator` で tautomer canonicalize → scorer key
- scorer key と stored key のどちらでも同一構造扱いしてグループ化 → **274,184 identity groups**
- 1.6% の構造で stored key と scorer key が差分（canonicalization 対象）

### 2 系列の fold

| 系列 | 方式 | fold サイズ | グループ数 |
|---|---|---|---:|
| `fold_identity` | 同一 scored identity を 1 fold に閉じる | 55,512–55,514 | 274,184 |
| `fold_scaffold` | 加えて Bemis-Murcko scaffold も 1 fold に閉じる | 55,512–55,514 | 140,160 |

### Validation Protocol (4 ステップ、公式仕様)
1. **ホールドアウト identity を全部除く**: モデル学習、retrieval 用 reference spectrum library、label-fitted feature すべてから
2. **候補 pool に残すかの選択**: ホールドアウト構造を残す → "known structure, no reference spectrum" シミュ、除く → "structure absent from DB" シミュ
3. **MRR@25 の前に candidate coverage (正解が候補に入っている率) を報告**
4. **公開 pretrained は全 fold に露出済み**: clean estimate には "4 fold で学習された checkpoint" が必要 → `fold-safe FPNet` を使う

### 検証
- 160 件サンプルの分子で "cross-fold neighbors at Tanimoto >= 0.7" を全ライブラリ検索して **0 件**
- 全スペクトル・全構造の fold alignment エラー: 0 件
- identity/analogue クロスフォール違反: 0 件

### Analogue fold (さらに厳しい)
- Morgan Tanimoto >= 0.7 で結合した連結成分単位で split
- 266,158 unique FP, 29.3B eligible pairs, 195,770 threshold edges
- 194,357 連結成分、最大 8,148 構造の component

## 2. 公開 FPNet の性能 (natural product panel benchmark)

ソース: `data/input/external/casmi26/casmi26-natural-product-panel-benchmark/README.md`

### パネル定義
- **250 分子**: Enveda timsTOF で測った natural products（hidden test に最も近い distribution）
- **1,939,000 PubChem 候補**（10 ppm window）が各分子に紐付いている

### 3 つの状況別 MRR@25

| 状況 | 隠すもの | hidden test で言えば | 公開パイプラインの MRR@25 |
|---|---|---|---:|
| Other reference | 同一分子のこのライブラリ内スペクトル | 他の instrument で測った reference がある分子 | **0.980** |
| No reference | 全 reference spectrum | reference なしの既知構造 | **0.988** |
| **Not in the pool** | 全スペクトル + 構造を候補 pool からも除く | **PubChem にのみ存在 or 完全新規** | **0.361** |

### FPNet 単体性能 (PubChem 検索、natural product panel 250 分子)

| 設定 | MRR@25 |
|---|---:|
| FPNet alone (fingerprint score only, raw) | **0.512** |
| FPNet + popularity prior `0.25 × (log_substances + log_pubmed)` | **0.969** |
| 5,000-row shortlist 版 (notebook 実装) | 0.951 |

### "Not in the pool" での補完
| 設定 | MRR@25 |
|---|---:|
| Main list (ranker 系) 単独 | 0.361 |
| Best PubChem proposal を promote | **0.718** |

### FPNet のリーク確認 (critical)

| FPNet | Natural product panel top-1 accuracy |
|---|---:|
| **全 train で学習 (`casmi26-fpnet-full1`, `v4b-models`)** | **89%** ← train 全体に overfit |
| **Fold-safe (ホールドアウト構造を学習から除外)** | **42%** ← CV に使える真の性能 |

**含意:** 公開 kernel の多くが `casmi26-fpnet-full1` で CV を回しているが、これは **MRR@25 が +0.47 虚偽に見える** 構造の reproduction。bio-hub L2 (作っただけの Validation が採用判定経路に組み込まれていない) と同じ失敗パターン。

**本プロジェクトでの対策:**
- CV 用 FPNet は必ず **`dmitriigluzdov/casmi26-fold-safe-fpnet`** を使う (学習コード公開済み、Winner License 整合)
- `casmi26-fpnet-full1` / `v4b-models` は "ベースライン再現確認" にのみ使い、**ablation 判定には使わない**

## 3. Popularity Prior の構造 (Dmitrii Gluzdov, 2026-10-20)

ソース: `data/input/external/casmi26/casmi26-pubchem-popularity-prior/FILE_DESCRIPTIONS.json` (dataset ページ description)

### なぜ popularity が効くか (hostが示したデータ)
- Natural product panel の正解分子 = PubChem の同一 formula 内で "best-documented" である確率:
  - **Panel natural products: 66%**
  - Training natural products: 24%
  - その他の training molecules: 2%
- 中央値の正解は候補の **99.8% (panel NP) / 99.0% (training NP) / 66.0% (other)** を上回る documentation 量

### 使い方 (公開 kernel の数式)
```
score_final = standardized(FPNet_score) + 0.25 × (log_substances + log_pubmed)
     + 0.50 × (bonus for spectral-library depositor / NP membership / LOTUS refs / synonyms)
```

### 得られる特徴量
| 特徴 | 定義 |
|---|---|
| `lsid` | ln(1 + PubChem substance 件数) |
| `lpmid` | ln(1 + PubMed article 件数) |
| `lpatent` | ln(1 + patent 件数) |
| `lmesh` | ln(1 + MeSH term 件数) |
| `lsynonyms` | ln(1 + synonym 件数) |
| `n_depositors`, `n_vendors` | depositor / vendor 数 |
| `src_mask` | bit mask (1 spectral library, 2 metabolite/pathway, 4 NP DB, 8 ChEBI, 16 LIPID MAPS, 32 drug, 128 toxicology, 256 vendor, 512 Wikidata) |
| `np` | bit mask (1 COCONUT, 2 LOTUS, 4 NPAtlas) |
| `lotus_refs` | LOTUS 文献 references 数 |
| `formula_isomers` | 同一 formula を持つ候補数 (crowding 指標) |

### 統合済み: PubChem tier 106M 構造 + 候補 pool 710k の全部

### 本プロジェクトでの含意
- popularity prior は **確実に +数%** 来る。公開 kernel が全員使っている理由がデータで裏取り可能
- 15 列くらいある副次特徴量 (lmesh, lsynonyms, depositors 等) は全員活用しきれていない → listwise ranker に全部突っ込むのが差分
- "panel で 0.512 → 0.969" の gap のうち、少なくとも +0.25 分は popularity 単独で説明がつく

## 4. COCONUT 候補 pool の中身

ソース: `data/input/external/casmi26/coconut-casmi26-candidates/`

| ファイル | 中身 |
|---|---|
| `coco_meta.pkl` | 436,389 分子のメタ (SMILES, inchikey14, formula 等) |
| `coco_mass.npy` | neutral exact mass (float64), sorted → `np.searchsorted` で ±10 ppm window retrieval |
| `coco_fp.npy` | 436,389 × 867 uint8。multi-FP 行列 |
| `fp_bits.npy` | 6,930 bit index。FPNet A/B の出力次元と対応 |

- mass 範囲: 100 - 1,300 Da、median 420 Da
- test precursor_mz 範囲 (245-460 Da、median 329) と整合、十分な候補が得られる

## 5. 公開 LB 0.467 の分解の裏付け（panel benchmark での校正）

公開 LB 0.467 は hidden test で測定された数字だが、natural product panel benchmark (250 分子) ではもっと高い数字が出ている:
- FPNet + popularity 全部入り: **0.969**（panel benchmark、ただし train 露出リーク含む）
- hidden test (public LB): 0.467

**gap の内訳 (仮説):**
1. Hidden test の分布が panel と違う (Class 3 の割合が多い等)
2. Panel の 250 分子はほぼ Class 1/2 で、hidden test は Class 3 が 15-30% 入ってる

→ **Metric Sandbox でシミュレートした "Class 3 を 5% → 20% に上げると +0.023 MRR" は現実的な期待値**（panel 0.969 と LB 0.467 の gap のうち Class 3 分を埋める）。

## 6. 本プロジェクトの戦略への反映

### 確定したベースライン再現パス
1. `prvsiyan/coconut-casmi26-candidates` + `casmi26-pubchemlite-cc0-candidate-fingerprints` の候補 pool
2. `dmitriigluzdov/casmi26-fold-safe-fpnet` の FPNet (CV 用 clean checkpoint)
3. `dmitriigluzdov/casmi26-pubchem-popularity-prior` の 15+ 特徴量
4. **CV 判定は `casmi26-identity-disjoint-validation-folds` の fold_identity (primary) + fold_scaffold (secondary) + enveda_np_examples ホールドアウト (target domain)** で両方上がったものだけ採用

### 差分注入ポイントの最新優先順
1. **Learning-to-rank ranker**: pointwise classification → LambdaRank/listwise で MRR@25 直最適化（詳細は `ltr_survey.md`）
2. **DreaMS / MIST / MS2DeepScore 2.0** を ranker の特徴量に注入（詳細は `foundation_models_survey.md`）
3. **FIORA / FRIGID** で ICEBERG/GLACIER の adduct blind spot 埋め（詳細は `forward_predictors_survey.md`）
4. **Class 3 専用補助**: Panel 0.969 vs LB 0.467 gap のうち Class 3 分子分を埋めるのが最大の勝ち筋

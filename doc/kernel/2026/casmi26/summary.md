---
competition: enveda-CASMI26-molecule-id-mass-spectra
checked_at: 2026-10-25
source_notebooks: 12 (pulled to raw/)
---

# CASMI 2026 公開 Kernel / Model 分析まとめ

## 1. 公開エコシステムの地図

### 1.1 公開 LB の構造
- **公開 kernel LB の頭打ち: 0.467** (`alexchilton/casmi26-lb-0-467-one-knob-rrf-offset-3-to-1`、RRF offset を 3→1 に下げただけ)
- **実際の総合トップは "nofreewill42" の 0.484** (独自アプローチ、公開ソース未共有、topic 745072 で言及)
- 公開 kernel 0.44-0.47 クラスタは **同一系統の fork 群**（E006 ベース → PubChem + ICEBERG/GLACIER + Neural rerank）
- 第 2 系統: Analog Propagation (`prvsiyan` 151 vote, LB 0.456)

### 1.2 共通使用の外部データセット
| Kaggle dataset slug | 内容 | サイズ | 用途 |
|---|---|---:|---|
| `dmitriigluzdov/casmi26-pubchem-popularity-prior` | PubChem CID-PMID/CID-Patent/CID-SID 件数 (93 vote) | - | ranking prior の popularity 項 |
| `prvsiyan/coconut-casmi26-candidates` | COCONUT 2.0 462k 候補 + 指紋 | 148 MB | 候補 pool 本体 |
| `franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints` | PubChemLite CC0 subset + FP | 113 MB | ライセンスクリーンな代替 pool |
| `ahmedberatozer/casmi26-iceberg`, `-glacier` | ms-pred ICEBERG/GLACIER forward model (coleygroup/ms-pred) | 64/56 MB | スペクトラム simulator |
| `prvsiyan/casmi26-forward-reference-stack` | ICEBERG/GLACIER オフラインスタック | 3.8 GB | 同上フル版 |
| `dmitriigluzdov/casmi26-fold-safe-fpnet` | fold-safe FPNet (identity-stratified) | - | fingerprint 回帰ネットワーク、学習コードなしで winner 不可だがローカル参考 |
| `dmitriigluzdov/casmi26-identity-disjoint-validation-folds` | inchikey14 ベースの disjoint validation fold | 47 MB | **CV 設計のゴールデンリファレンス** |
| `dmitriigluzdov/casmi26-natural-product-knowledge-table` | NP pathway, class 情報 | 84 MB | ranking features |
| `dmitriigluzdov/casmi26-natural-product-panel-benchmark` | 自前 NP ベンチマーク | 175 MB | 検証プロキシ |
| `ahmedberatozer/casmi26-dreamsfp-d2` | DreaMS fingerprint 変換 D2 | 450 MB | DreaMS 系 |

### 1.3 公開エコシステムの用語辞典（公開 kernel で頻出）
| 用語 | 意味 |
|---|---|
| **E006** | `dmitriigluzdov` のベースラインエンジン名 (library + FPNet + LightGBM) |
| **DG / v15 / v16 / v4g / v4n** | 同系統の派生版 |
| **N1A / N1 Challenger** | ICEBERG/GLACIER が blind な分子をカバーするための補助エンジン (natural product 向け) |
| **ICEBERG / GLACIER** | coleygroup/ms-pred の forward MS predictor。GLACIER は `[M+H]+` 限定、ICEBERG は汎用 |
| **PubChemLite** | PubChem の CC0 subset。ライセンス安全 |
| **Popularity Prior** | PubChem CID の PMID/Patent/SID 件数ログ和（ranking 特徴量） |
| **RRF** | Reciprocal Rank Fusion。複数 ranker 出力の融合。`score = Σ 1/(offset + rank_i)` |
| **Dual-gate** | PubChem 候補を lib_max 等でゲート → 条件満たすときのみ promote |
| **genfix / Features-frozen generation** | ICEBERG/GLACIER の追加候補を既存 ranker スコアを乱さず挿入するトリック |

## 2. 共通パイプライン (LB 0.44-0.47 の全 kernel が収束している 5 段)

```
[Query spectrum]
       ↓
[1. Candidate Retrieval]
   - Library search (train spectra, ±8.5 ppm, entropy-weighted cosine)
   - Analog propagation (mass shift ±200 Da, sim^4 Tanimoto weighting)
   - COCONUT 2.0 pool (462k) + train scaffolds (275k) ≒ 711k 候補
   - PubChemLite / PubChem (105M) をゲート付きでマージ
       ↓
[2. Fingerprint Prediction (ML)]
   - FPNet A/B (6,930-bit multi-FP 回帰 CNN)
   - 各候補の構造 FP との f·z dot product スコア
   - Fold-safe FPNet 版あり
       ↓
[3. Simulator Overlay (Forward MS Prediction)]
   - ICEBERG / GLACIER を使い、各候補の予測 MS を本物とコサイン
   - GLACIER は [M+H]+ のみ
   - 同一 molecular formula 内で再順位付け
       ↓
[4. Learning-to-rank]
   - LightGBM / HistGradientBoosting ranker
   - 31 特徴量: lib sim, analog sims, fragment scores, FPNet logits,
                popularity (log CID-PMID + CID-Patent + CID-SID),
                rank agreement, formula share, z-scores
   - Dual class prior (W1 ∈ {0.30, 0.60}), 4-seed bagging
       ↓
[5. Final Fusion (RRF)]
   - score = 1/(offset + rank_baseline) + w / (offset + rank_neural)
   - offset = 1.0, w = 0.25 (現 LB 0.467 のパラメータ)
   - 上位 40 → top 25 に圧縮
       ↓
[submission.csv]
```

### 2.1 現時点のボトルネック
Explore エージェント分析と `topic 752356` (Aryan Awasthi の客観分析) が一致して指摘:
1. **ICEBERG/GLACIER の coverage** が ~60-70% しかない。スコア無しの分子は baseline order に戻るため MRR 伸びない
2. **Class 3 (全く新規の分子) の LB は横ばい**（topic 751300 で複数チームが報告）。ローカルで +0.003-0.004 出ても LB は flat
3. **`[M+H]+` 以外の adduct に対する証拠が薄い**（GLACIER が blind）

### 2.2 公開 kernel で未着手 / 未実装のモジュール
- `casmi-2026-modular-retrieval-ranking-overhaul` の `class LearnedRanker` は **スケルトンだけで trained checkpoint 未公開**
- **ListNet / LambdaRank 系の MRR 直最適化 ranker はほぼ全 kernel 未実装**（HistGBM / LightGBM の pointwise 分類ロスのみ）
- **DreaMS 系の pretrained spectrum encoder** は `hengck23/dreams-structure-from-spectrum-retrival-demo` でデモだけ (9 vote)。本格活用した高スコア kernel なし
- **MIST 系** は 10/20 公開（Zenodo 23303289 MIT）だが、CASMI26 公開 kernel で統合しているものはまだ見ない

## 3. 代替アプローチの 2 系統

### 3.1 Analog Propagation (prvsiyan, 151 vote, LB 0.456)
- 2 engine 切り替え: Engine 1 (spectral library) + Engine 2 (PubChem retrieval + FPNet)
- Class 1 / Class 2 / Class 3 でルーティング
- Hongbin Gu の V465 v15 が元ネタ
- Reproducible, documented, ドキュメントとして価値あり

### 3.2 DreaMS Retrieval (hengck23, 9 vote)
- MassSpecGym 学習済み MS2 SSL foundation model
- Query spectrum → 1024-dim embedding → KNN
- **spectrum ↔ reference spectra の cosine**（候補の構造 FP との cosine ではない）
- Class 1 (リファレンススペクトル存在) のみ有効、Class 2/3 は未対応
- 単体で LB 0.46+ に届いた報告はなし。**他モジュールと組み合わせる余地が残っている**

### 3.3 De novo Generation (inversion 公式チュートリアル, 397 vote)
- 6-layer Transformer encoder + 6-layer causal decoder、autoregressive SMILES 生成
- 候補 pool 不要、純生成
- LB 不明だが、公開 kernel 群では主流に採用されていない（組み合わせが難しい）
- MRR@25 の **tail 候補 (10-25 位)** を埋めるための補助として研究価値あり

## 4. 公開 Kernel の勝ち筋評価マトリクス

| 系統 | 公開 LB 到達 | 改善余地 | 独自モジュール注入先 |
|---|---:|---|---|
| **E006 fusion 系** | 0.467 | ICEBERG coverage, Class 3, learned ranker | **LambdaRank/ListNet ranker, Class 3 専用補助** |
| **Analog Propagation** | 0.456 | 同上 | 同上 |
| **DreaMS retrieval** | ~0.3 | Class 2/3 の構造予測 | **DreaMS embedding を既存 ranker の 1 特徴に注入** |
| **De novo seq2seq** | - | 25 候補の tail を埋める | **主系統の 20-25 位スロットに注入** |

## 5. ノートブック別詳細レポート

### 5.1 `inversion/casmi-denovo-tutorial-notebook` (397 vote)
- 公式チュートリアル。Transformer seq2seq。
- 教育目的、LB 競争力なし。

### 5.2 `haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline` (266 vote, LB 0.30-0.336)
- **最もシンプルで改造しやすい ranker 実装**
- 31 特徴量 → HistGradientBoosting
- Cell 8 の `rank_features()` と `rank_proba()` が差分注入の起点

### 5.3 `dmitriigluzdov/casmi-26-from-spectra-to-structures` (163 vote, LB 0.417)
- E006 系の elaborate 版 (128 cells)
- LightGBM 4-booster + popularity re-rank + PubChem gated merge + GLACIER/ICEBERG rerank
- Cell 104-110 がランカー構造の中核

### 5.4 `dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures` (172 vote)
- EDA 専用。validation fold 設計、popularity prior の効果計測。
- `fold-safe FPNet` の必要性の裏付け

### 5.5 `sanpier/pubchem-neural025-lb-0-466` (78 vote, LB 0.466)
- 5 段パイプラインの決定版
- `NEURAL_WEIGHT=0.25, RRF_OFFSET=3.0` → 本家。offset=1 版が +0.001 で 0.467 に

### 5.6 `casmi26-lb-0-467-one-knob-rrf-offset-3-to-1` (9 vote, LB 0.467)
- 公開 kernel 現状最強。パラメータ 1 変数いじり。
- 教訓: 公開 LB 伸びは「1 knob tuning」の収穫逓減フェーズ

### 5.7 `flexonafft/casmi26-closing-the-evidence-gap` (90 vote)
- ICEBERG/GLACIER coverage gap を埋める試み
- N1A Challenger experiment、+0 improvement の negative result 報告

### 5.8 `prvsiyan/analog-propagation-casmi-2026-baseline` (151 vote, LB 0.456)
- 第 2 系統の筆頭
- Class 1/2/3 ルーティング付き

### 5.9 `kozykappa/casmi-2026-modular-retrieval-ranking-overhaul` (27 vote)
- `class LearnedRanker` スケルトン **実装例として参照価値**
- 学習済 checkpoint は未公開 → **自前で埋めればここが差分注入ポイント**

### 5.10 `mohammadhmozafary/casmi-2026-domain-knowledge-for-non-chemists` (26 vote)
- 化学ドメイン知識の primer
- **collision_energy NCE↔eV 変換式 `eV ≈ NCE × precursor_mz / 500`**
- Adduct normalization、ppm vs Da の使い分け

### 5.11 `kitopl/casmi-2026-data-pitfalls-that-break-your-cv` (9 vote)
- **本コンペ最重要 EDA**
- test.parquet (配布) と hidden test の分布差
- inchikey14 canonicalization gap (3,167 構造が scoring 時 collapse)
- `precursor_error_ppm` を query 選択に使うな (label leakage)

### 5.12 `hengck23/dreams-structure-from-spectrum-retrival-demo` (9 vote)
- DreaMS retrieval デモ
- `embedding_model_state_dict.pt` + `MassSpecGym_DreaMS_embeddings.npy`
- MIT ライセンス、MassSpecGym 学習データなので CASMI26 OK

## 6. actionable_points（本プロジェクトへの転用方針）

### 6.1 すぐ使うもの
1. **候補 pool**: `prvsiyan/coconut-casmi26-candidates` + `franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints` を attach
2. **CV manifest**: `dmitriigluzdov/casmi26-identity-disjoint-validation-folds` を正本として採用（inchikey14 ベース）
3. **popularity prior**: `dmitriigluzdov/casmi26-pubchem-popularity-prior` の log(CID-PMID + CID-Patent + CID-SID) を特徴量
4. **ベースライン ranker**: `haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline` を最弱ベースライン（kaggle_flow_v1.md §4 Week 1 の "最弱ベースライン提出"）に流用

### 6.2 差分注入ポイント（本コンペの勝ち筋候補）
1. **LambdaRank / ListNet 系の MRR 直最適化 ranker**: 既存 LightGBM の lambdarank 目的関数差し替え、または PyTorch で listwise loss 実装
2. **DreaMS embedding を既存 ranker の特徴量に追加**: cosine(query, reference) ではなく cosine(query, candidate 候補の予測 embedding) を使う
3. **Class 3 (新規分子) 向け補助**: Analog Propagation の方向性を強化、または MIST (10/20 公開) の統合
4. **ICEBERG/GLACIER coverage gap 埋め**: 自前で簡易 forward model を学習、または近傍探索 heuristic
5. **MRR@25 top-3 専用の 2 段 reranker**: 上位 10 候補だけに適用する軽量 neural reranker

### 6.3 絶対にやってはいけない
1. **NIST データ・NIST 派生モデル（SIRIUS/CSI:FingerID 等）を使う** → 失格
2. **`precursor_error_ppm` を query 側の filter に使う** → label leakage、CV が架空に上がる
3. **`ingest_lib` ホールドアウトを信じる** → `enveda-np-examples` が全ライブラリと 100% コンタミ
4. **`inchikey14` を未 canonical で判定** → 1.15% の構造が scoring 時に collapse、0.48% MRR 損
5. **他人の pretrained 重みを再現訓練コードなしで winner 提出する** → 失格リスク (topic 745072, 749217)
6. **`[M+H]+` の adduct 偏重設計のまま test に出す** → `[M-H]-` 等への対応不足で Class 3 が落ちる

## 7. 戦略への反映（strategy/current.md へ追記予定）

- **勝ち筋**: 「E006 共通パイプラインの Stage 4 (ranker)」に **LambdaRank / listwise loss / DreaMS embedding 統合** を注入する差分モジュール設計
- **ベースライン**: Fast Spectral Cosine Baseline を再現 → MRR のローカル測定 → 差分の因果効果を測る
- **CV**: `casmi26-identity-disjoint-validation-folds` を正本採用、inchikey14 canonical key で scoring

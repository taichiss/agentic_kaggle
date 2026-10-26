# Enveda CASMI 2026 — strategy/current.md

## 位置づけ
- **一次情報源:** [doc/strategy/kaggle_flow_v1.md](../../doc/strategy/kaggle_flow_v1.md)
- 本ファイルは **フローへのポインタ + コンペ固有差分だけ**を記述する。フロー自体の内容を複製しない

## 現在地
- **Phase:** Phase C — Week 0（開始チェックリスト、全項目潰すまで Week 1 に進まない）
- **実行フロー:** [doc/strategy/kaggle_flow_v1.md §3](../../doc/strategy/kaggle_flow_v1.md) の Week 0

## コンペ固有サマリ (詳細は [../overview.md](../overview.md))
- **評価指標:** MRR@25（1 分子につき 25 候補の順位付きリスト、最初の正解順位の逆数）
- **提出:** `submission.csv` 400 行、`molecule_id, smiles(; 区切り 25 本)`、Notebook 提出
- **データ量:** train 2.54M スペクトル / 27.8 万ユニーク SMILES、test 1,213 スペクトル / 400 molecule_id
- **ドメインシフト:** 配布 `test.parquet` は 100% `enveda-180` (合成 drug-like)。hidden test は timsTOF 計測の natural products で分布が別物
- **外部データ:** 可（**CC BY-NC OK (10/17 緩和)**）。PubChem / PubChemLite / COCONUT / ChEBI / LOTUS / MassBank / MoNA / GNPS など使用可。**NIST / SIRIUS / CSI:FingerID は失格**
- **事前学習モデル:** 可。MS2DeepScore 2.0 / FIORA-OS / ICEBERG / GLACIER / DreaMS SSL / CFM-ID 4 / MIST (10/20 MIT) / FRIGID (NC) / TabPFN v2 すべて OK
- **Submit 制限:** 5/day、Final 2（**Final 2 本は完全再現必須。学習コードを遡れない他人重みは Final に出さない**）

## kaggle_flow_v1.md §1 中核原則への対応差分

### L1 (指標感度を実測した根拠を直観で覆さない)
- MRR@25 は $\sum 1/\mathrm{rank}_u / 400$。1 位→圏外で 1 分子あたり -1.0 点 ≒ 全体 -0.0025 点
- Week 0 Metric Sandbox で、"候補順位の 1 位精度 vs 2〜25 位の順位最適化" の交換レートを合成データで実測し、`enveda-casmi26/metric_sensitivity.md` に残す
- 以降「ranker 変更」「候補数削減」「cutoff 変更」の各判断に、この感度への整合を 1 行で `registry.csv.notes` に書く

### L2 (Validation は採用判定経路に組み込まれているか)
- CASMI 系は train の `inchikey14` 単位でホールドアウトを切る群別 CV が正しい（**同一分子が train/val に分かれるとリーク**）
- **併せて instrument_type ホールドアウト** が必須（test は全 timsTOF、train に Orbitrap/QTOF が混ざるため LB との乖離が出やすい）
- 公開 kernel が降ってきた瞬間、その出力を `inchikey14 × instrument_type` CV に落とすアダプタを先に書く（bio-hub L2 の失敗を繰り返さない）

### L3 (公開 kernel の後処理をいじるモードに降りたら負け)
- CASMI 系で公開されがちな基礎パイプライン: `spectrum embedding → fingerprint 回帰 → DB 検索 → Tanimoto 類似度ランク`
- 本コンペの差分注入先候補: ① test timsTOF ドメインに合わせた domain-adapted spectrum encoder, ② MRR@25 特化の reranker (contrastive / listwise)
- PP sweep は 5 件まで。それ以降は差分モジュール設計に時間を回す

### L4 (公開 kernel 採用時のチェックポイント)
- CP1〜CP3 を §6 のまま適用。CP3 (改修先モジュールが決まるまで PP sweep 禁止) を忘れない

### L5 (再現性)
- Winner License (OSI 承認、商用利用可、methodology ドキュメント義務) が課されるため、再現性は Week 0 から徹底
- 全提出で `git rev-parse HEAD` + `pip freeze` + `kernel-metadata.json` を `enveda-casmi26/experiments/EXP-XXXX/env.json` に保存

## 採用規律 (このコンペの明文化)
1. **両ホールドアウトで勝ったものだけ採用**（`inchikey14 disjoint fold` AND `enveda-np-examples ホールドアウト`、`casmi26-identity-disjoint-validation-folds` 正本）
2. **候補 25 本の下位 (10 位以降) をいじる変更は、1 位 accuracy を悪化させないか確認必須**
3. **公開 kernel 出力を使う場合、まず CV アダプタに通して LB を鵜呑みにしない**
4. **MRR の rank 1〜3 を触る変更を優先、rank 10〜25 の順位微調整は後回し**
5. **canonical_inchikey14 でローカルスコアリング**（RDKit 2026.03.3 pin。生の inchikey14 比較は 0.48% MRR 損失する）
6. **Final 2 本には学習コード公開済みの重みのみ使用**（`fpnet-full1`, `v4b-models` 等の未公開学習コード重みは CV 検証用に留める）
7. **CV 評価には fold-safe FPNet のみ使用**（`dmitriigluzdov/casmi26-fold-safe-fpnet`）。公開 `casmi26-fpnet-full1` は train 全体で学習 → **42% top-1 を 89% top-1 に偽装するリーク** が発生する（natural product panel で確認、local_evidence.md §2 参照）
8. **MIST の報告数字は audit-corrected scores のみ信じる**（MassSpecGym In the Wild で 17/26 downstream 論文が padding mask バグで overstate している）

## 勝ち筋の仮説（kernel/Discussion 分析 2026-10-25 時点）

### 公開 LB の現状
- 公開 kernel 頭打ち = **LB 0.467** (`alexchilton`, RRF offset 変数 1 つの微調整)
- 公開情報の総合トップ = **LB 0.484** (`nofreewill42`, 独自アプローチ、ソース未公開)
- Class 3 (完全新規) で全チーム停滞中（topic 751300）

### 公開 LB 0.44-0.47 の共通 5 段パイプライン
```
[Candidate Retrieval (library + analog + PubChemLite/COCONUT)]
  → [FPNet fingerprint 予測]
  → [ICEBERG/GLACIER simulator overlay]
  → [LightGBM/HistGBM pointwise ranker]
  → [RRF fusion → top 25]
```

### 差分注入先の優先順位（本プロジェクトの差分モジュール設計案、文献サーベイ結論を反映）

> 詳細は [`doc/discussion/2026/casmi26/papers/survey_summary.md`](../../doc/discussion/2026/casmi26/papers/survey_summary.md) を参照。
> 全アップグレード一覧表は同文書 §1.1 に。

#### 保守的合計 **+0.065 ~ +0.135 MRR** (項目 1-5 で LB 0.467 → 0.53-0.60 を狙う)

| # | モジュール | 期待 MRR 上昇 | 工数 (pd) | ライセンス | 公開 kernel 空白度 |
|---|---|---|---:|---|---|
| **1** | **LightGBM LambdaRank 置換** (`objective=lambdarank, lambdarank_truncation_level=25, label_gain=[0,1]`) | **+0.02-0.05** | **0.5** | FOSS | **完全空白** |
| 2 | FIORA + CCS channel (timsTOF 固有 orthogonal signal) | +0.015-0.030 | 1.5 | MIT | 完全空白 |
| 3 | ICEBERG 2.1 `msg_all` で non-`[M+H]+` adduct 対応 | +0.010-0.020 | 1 | MIT | 部分 |
| 4 | MIST (Zenodo 23303289, 10/20 公開) retrieval encoder | +0.010-0.020 | 1 | MIT | **完全空白** |
| 5 | MS2DeepScore 2.0 プール reducer (275k→500) | +0.010 | 0.5 | Apache + CC-BY (attribution) | 完全空白 |
| 6 | DreaMS cosine blender feature | +0.005-0.010 | 1 | MIT | ほぼ空白 |
| 7 | FRIGID tail-fill (hard spectra の rank 20-25 補完) | +0.005-0.015 | 2-3 | Apache | 完全空白 |
| 8 | Softmax listwise custom fobj (Day-3 upgrade) | +0.01 (vs LambdaRank) | 1 | - | 完全空白 |

#### 文献由来の重要発見
- **MIST vFRIGID が MassSpecGym retrieval で Hit@1 53.8% の SOTA** (2026 発表)、公開 kernel で誰も使っていない → 現時点の **空白地帯最大**
- **CASMI 歴代 (2016/2017/2022) の勝者は誰も LTR を使っていない** (CSI:FingerID は SVM pointwise、MS-FINDER は rule-based) → **LTR 導入が最大のアービトラージ機会**
- **binary single-positive の場合、NDCG@25 と MRR@25 は rank_of_positive に対して monotonic** → LightGBM の `objective=lambdarank` (NDCG delta 固定) で MRR@25 も improve する (ltr_survey.md 根拠)
- **MassSpecGym In the Wild 警告** (arXiv 2606.19624): 17/26 の downstream 論文が MIST padding mask バグで数値を overstate、本プロジェクトは **audit-corrected scores** しか信じない
- **CCS (Collision Cross Section) が timsTOF 固有信号で、公開 kernel で誰も使っていない**。FIORA が native 予測可能

#### 推奨実行順序 (Week 1-3 の 5 日分)
- **Day 1 (ベースライン再現 + LambdaRank 導入)**: `haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline` 再現 → LightGBM 7 行変更 → CV で +0.02 確認 → submit
- **Day 2-3 (MIST + MS2DeepScore 2.0 統合)**: MIST MSG-aug checkpoint をローカル推論、MS2DeepScore 2.0 で 2.5M reference encode → candidate pool reducer、両モデルを LambdaRank 特徴量に追加
- **Day 4-5 (FIORA + ICEBERG 2.1 統合)**: FIORA MIT weights で `[M-H]-` の forward prediction → cosine + CCS 特徴量追加、ICEBERG 2.1 `msg_all` で `[M+Na]+/[M+NH4]+/[M+K]+` の分子を再 scoring

### 外部データの最優先取得（Week 0 で完了済）

**取得済み** (`data/input/external/casmi26/`):
- [x] `prvsiyan/coconut-casmi26-candidates` (404 MB, COCONUT 462k 候補 + FP)
- [x] `dmitriigluzdov/casmi26-identity-disjoint-validation-folds` (61 MB, **CV 正本**)
- [x] `franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints` (345 MB)
- [x] `dmitriigluzdov/casmi26-natural-product-knowledge-table` (130 MB)
- [x] `dmitriigluzdov/casmi26-natural-product-panel-benchmark` (223 MB, FPNet 性能校正データ)
- [x] `dmitriigluzdov/casmi26-pubchem-popularity-prior` の pool_* 系 (190 MB)

**Week 1-2 で取得予定**:
- [ ] MIST Zenodo 23303289 (10/20 公開 MIT, Kaggle Dataset 化は未確認、Zenodo 直接)
- [ ] DreaMS SSL backbone (HuggingFace `roman-bushuiev/DreaMS` or Zenodo 10997887)
- [ ] MS2DeepScore 2.0 weights (Zenodo 17826815, CC-BY-4.0 attribution 要)
- [ ] FIORA weights (GitHub `BAMeScience/fiora` + Zenodo 16984129, MIT)
- [ ] ICEBERG 2.1 `msg_all` checkpoint (coleygroup/ms-pred, Dropbox 経由)
- [ ] GLACIER MSG checkpoint (同上)
- [ ] `dmitriigluzdov/casmi26-fold-safe-fpdata` (CV 用の fold-safe FPNet, 公開 FPNet はリーク！)

### 絶対回避項目（Discussion / kernel 分析から確定）
- NIST, SIRIUS/CSI:FingerID 不使用（失格）
- `precursor_error_ppm` を query 選択の filter に使う（label leakage）
- `ingest_lib` ベースのホールドアウト（natural products で 100% コンタミ）
- 生の `inchikey14` 比較でローカルスコアリング（canonical 不整合）
- 他人の学習コード非公開重みを Final 提出に含む（Winner 失格リスク）

## 直近アクション (Week 0 チェックリスト進行中)
1. 指標因数分解実装 `src/<pkg>/metrics/breakdown.py`
2. Metric Sandbox 実測 → `enveda-casmi26/metric_sensitivity.md`
3. 指標ハッシュ → `enveda-casmi26/frozen/metric_hash.txt`
4. CV 設計 → `enveda-casmi26/frozen/cv_manifest.json`
5. 文献サーベイ → `doc/discussion/2026/casmi26/`
6. 採用規律の明文化 (本ファイル上記)

## 旧コンペの戦略メモ (ポインタのみ)
- bio-hub Lessons: [bio-hub/matome.md](../../bio-hub/matome.md)
- BirdCLEF+ 2026 当時の current.md: [strategy/current.birdclef2026.md.bak](../../strategy/current.birdclef2026.md.bak)

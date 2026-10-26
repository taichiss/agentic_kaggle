---
competition: enveda-CASMI26-molecule-id-mass-spectra
checked_at: 2026-10-25
metric: MRR@25 on 400 molecules
scope: 文献サーベイ 3 本 + ローカル実地調査 1 本の統合
sources:
  - papers/foundation_models_survey.md
  - papers/forward_predictors_survey.md
  - papers/ltr_survey.md
  - papers/local_evidence.md
---

# CASMI26 文献サーベイ 統合シンセシス

> 本ファイルは 4 つのサーベイ文書を単一のロードマップに統合した正本。
> 戦略判断、工数見積もり、優先順位の正本として `strategy/current.md` から参照される。

## 1. 公開 LB 0.467 を超えるためのロードマップ

### 1.1 全アップグレード候補の一覧表（マージ版）

| # | モジュール | 期待 MRR 上昇 | 工数 (pd) | ライセンス | 公開 kernel 空白度 | 根拠文書 |
|---|---|---|---:|---|---|---|
| **1** | **LightGBM LambdaRank 置換** (`objective=lambdarank`, `lambdarank_truncation_level=25`, `label_gain=[0,1]`) | **+0.02–0.05** | **0.5** | FOSS | **完全空白**（全 kernel pointwise） | [ltr_survey](ltr_survey.md#5-integration-pattern) |
| 2 | FIORA + CCS channel (timsTOF 固有 orthogonal signal) | +0.015–0.030 | 1.5 | MIT | 完全空白 | [forward_predictors_survey](forward_predictors_survey.md#4-fiora--fiora-os) |
| 3 | ICEBERG 2.1 `msg_all` で non-`[M+H]+` adduct 対応 | +0.010–0.020 | 1 | MIT | 部分 | [forward_predictors_survey](forward_predictors_survey.md#2-iceberg) |
| 4 | MIST (Zenodo 23303289, 10/20 公開) retrieval encoder | +0.010–0.020 | 1 | MIT | **完全空白** | [foundation_models_survey](foundation_models_survey.md#1-mist) |
| 5 | MS2DeepScore 2.0 プール reducer (275k→500) | +0.010 | 0.5 | Apache + CC-BY (attribution 必要) | 完全空白 | [foundation_models_survey](foundation_models_survey.md#3-ms2deepscore-20) |
| 6 | DreaMS cosine blender feature | +0.005–0.010 | 1 | MIT | ほぼ空白（hengck23 デモのみ） | [foundation_models_survey](foundation_models_survey.md#2-dreams) |
| 7 | FRIGID tail-fill (hard spectra の rank 20-25 補完) | +0.005–0.015 | 2–3 | Apache | 完全空白 | [forward_predictors_survey](forward_predictors_survey.md#5-frigid) |
| 8 | Softmax listwise custom fobj (Day-3 upgrade) | +0.01 (vs LambdaRank) | 1 | - | 完全空白 | [ltr_survey](ltr_survey.md#3-softmax-listwise) |

### 1.2 保守的合計期待値
- **項目 1 + 2 + 3 + 4 + 5 の合計: +0.065 〜 +0.135 MRR**
- 現 LB 0.467 + 保守下限 +0.065 = **0.53**（gold line ~0.471 を 0.06 上回る）
- 現 LB 0.467 + 保守上限 +0.135 = **0.60**（nofreewill42 の非公開 top 0.484 を上回る）
- 工数: 項目 1-5 全部で **~4.5 person-days**

### 1.3 推奨実行順序

**Day 1 (ベースライン再現 + LambdaRank 導入):**
1. 最弱ベースライン再現: `haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline` を identity-disjoint fold 0 で CV 測定
2. LightGBM 7 行変更: `objective='binary'` → `objective='lambdarank'`, `group=` 追加, `lambdarank_truncation_level=25`
3. 両ホールドアウト ({identity-disjoint, enveda-np-examples}) で検証、CV >= 現値 を確認して submit

**Day 2-3 (MIST + MS2DeepScore 2.0 統合):**
4. MIST Zenodo 23303289 の MSG-aug checkpoint をローカル推論
5. MS2DeepScore 2.0 で 2.5M reference spectra を encode → cosine で candidate pool reducer 作成
6. 両モデルを LambdaRank の特徴量として追加

**Day 4-5 (FIORA + ICEBERG 2.1 統合):**
7. FIORA MIT weights で `[M-H]-` の forward prediction → cosine 特徴量 + CCS 特徴量を追加
8. ICEBERG 2.1 `msg_all` で `[M+Na]+/[M+NH4]+/[M+K]+/[M-H2O+H]+` の分子を再 scoring

**Day 6+ (Stretch goals):**
9. DreaMS cosine blender feature
10. FRIGID tail-fill (rank 20-25 補完)
11. Softmax listwise custom fobj (LambdaRank saturate したら)

## 2. 各技術の役割分担（複数モデルを使う根拠）

### 2.1 アーキテクチャ全体図（最終目標）

```
[Query MS2 spectrum]
       │
       ├─► MS2DeepScore 2.0 ──► cosine vs 2.5M library spectra ──► top-500 candidate pool
       │                                                              │
       │                                                              ▼
       ├─► MIST (vFRIGID) ─────► 4096-bit Morgan FP prediction ──┐
       │                                                           │
       ├─► DreaMS SSL ─────────► 1024-d embedding ──────────┐      │
       │                                                     │      │
       └─► Candidate adduct = [M+H]+? ────────┐              │      │
                                              │              │      │
                                   [YES]      │  [NO]        │      │
                                    │         │   │          │      │
                                    ▼         │   ▼          │      │
                              GLACIER  ◄──────┘   ICEBERG   │      │
                              (fast)              2.1 msg_all│      │
                                    │             (10 adducts)│    │
                                    │                  │      │     │
                                    └──┬───────────────┘      │     │
                                       │                      │     │
                                       ▼                      │     │
                                 forward MS cosine            │     │
                                       │                      │     │
                                       │ FIORA (independent + │     │
                                       │ CCS for timsTOF) ◄───┘     │
                                       │       │                    │
                                       └───┬───┘                    │
                                           │                        │
                                           ▼                        │
                              [Feature table per (query, candidate)]│
                                           │                        │
                                           ▼                        │
                                  LambdaRank (LightGBM) ◄───────────┘
                                  objective=lambdarank
                                  truncation_level=25
                                           │
                                           ▼
                    [Top-25 SMILES per molecule → submission.csv]
```

### 2.2 ベースライン (公開 LB 0.467) との差分
- **候補 pool reducer**: 公開 kernel は mass window + analog propagation。本案は + MS2DeepScore 2.0 で 10× quality
- **FP prediction**: 公開 kernel は FPNet (6,930-bit, 学習コード非公開)。本案は MIST (4,096-bit, MIT, MSG-aug 78.5M synthetic 学習)
- **Forward MS**: 公開 kernel は GLACIER (`[M+H]+` only)。本案は GLACIER + ICEBERG 2.1 + FIORA の 3 本
- **Ranker**: 公開 kernel は pointwise classification。本案は LambdaRank (NDCG@25 weighted λ gradient)
- **新規信号**: 公開 kernel にない CCS (timsTOF 固有、FIORA から)、DreaMS 1024-d embedding
- **De novo tail-filler**: 公開 kernel になし。本案は FRIGID で hard spectra の rank 20-25 補完

## 3. 文献由来の重要発見（戦略に反映すべき）

### 3.1 MassSpecGym In the Wild の警告 (arXiv 2606.19624)
- **17/26 の downstream 論文が MIST の padding mask バグで数値を overstate している**
- 本プロジェクトでは **audit-corrected scores** しか信じない（MIST の実力は Top-1 0.347 on MSG mass-candidate scaffold）
- `foundation_models_survey.md` に記載済みの数値は audit-corrected

### 3.2 MassSpecGym retrieval SOTA (2026 公開)
- **MIST vFRIGID: Hit@1 53.8% / Hit@5 65.3% / Hit@20 74.7%** on MSG retrieval (formula-known track)
- 公開 kernel で MIST vFRIGID を使っているのはゼロ件 → **現時点の空白地帯最大**
- CASMI26 の timsTOF ドメインに持ってくる際の domain gap が open question

### 3.3 LTR 文献の決定打
- **binary single-positive の場合、NDCG@25 と MRR@25 は rank_of_positive に対して monotonic** → LambdaRank で NDCG@25 を optimize すれば MRR@25 も improve する
- LightGBM の `objective=lambdarank` は NDCG delta 固定だが、本コンペ用途には十分
- CASMI 歴代の勝者は **誰も LTR を使っていない**（CSI:FingerID は SVM pointwise、MS-FINDER は rule-based） → **LTR 導入が最大のアービトラージ機会**

### 3.4 Forward predictor の adduct partition 戦略
- GLACIER の `[M+H]+` 限定は仕様。拡張しようとしても weights 自体が blind
- 代わりに **adduct 別ルーティング**: `[M+H]+` → GLACIER (fast), `[M-H]-` → FIORA, その他 → ICEBERG 2.1
- test に含まれる adduct 分布 (`[M+H]+` 79%, `[M-H]-` 16%, その他 5%) を考慮すると、`[M-H]-` 対応 (FIORA) だけで +5% ~ 15% の partition に効く

### 3.5 CCS (Collision Cross Section) の timsTOF 固有信号
- **全 test spectra が timsTOF** = CCS 情報が train / test に共通で含まれるはず
- FIORA は候補分子から CCS を予測できる → 予測 CCS vs 観測 CCS の match が orthogonal な evidence
- **公開 kernel で CCS を使っているのはゼロ件** → これも空白地帯

## 4. ライセンス / Winner 要件のチェックリスト

### 4.1 使用予定 OSS/モデル のライセンス matrix
| コンポーネント | コード | 重み | 学習データ | Winner MIT 整合 |
|---|---|---|---|---|
| MIST | MIT | MIT (Zenodo 23303289) | MSG + CANOPUS + synthetic | ✓ Clean |
| DreaMS | MIT | MIT (HF + Zenodo) | GeMS (GNPS) | ✓ Clean |
| MS2DeepScore 2.0 | Apache 2.0 | CC-BY-4.0 | GNPS/MassBank/MoNA/MSnLib | ✓ attribution 必要 |
| ICEBERG | MIT | MIT (`msg_all` Dropbox) | MassSpecGym のみ (NIST checkpoint 除外) | ✓ Clean |
| GLACIER | MIT | MIT (MSG checkpoint) | MassSpecGym のみ | ✓ Clean |
| FIORA | MIT | MIT (MS-Dial 学習) | MS-Dial (NIST 除外) | ✓ Clean |
| FRIGID | Apache 2.0 | Zenodo 23543494 (要確認) | MSG + NPLIB1 + 1B SAFE | ✓ Clean (paper 著作権と別) |
| LightGBM | MIT | - | - | ✓ Clean |
| RDKit | BSD 3-Clause | - | - | ✓ Clean |

### 4.2 Final 2 本の reproducibility 要件（Discussion topic 745072, 748882）
- Winner はこれら OSS 全てを **元の学習コード + 許可された学習データで再現可能** に提示する必要あり
- 本案のアーキテクチャは **全て公開済みの training code + 公開済みの training data** で構成されている → 問題なし
- `ahmedberatozer/casmi26-fpnet-full1` 系は学習コード未公開のため **Final 提出に含めない**（CV 検証用のみ）

## 5. 本プロジェクトの絶対ルール（文献結論から確定）

### 5.1 絶対使わない
- NIST / NIST 派生 (SIRIUS/CSI:FingerID 含む) — Discussion topic 750330, 751373 で明示
- ICEBERG の NIST-trained checkpoint (`nist_all`, `nist_simulation`)
- FPNet-full1 など学習コード非公開重み — Final に出す場合リスク、CV 検証にのみ使う

### 5.2 絶対使う
- `canonical_inchikey14` scoring (RDKit 2026.03.3 pin) — `src/casmi26/metrics/breakdown.py` 既実装
- Identity-disjoint fold (`dmitriigluzdov/casmi26-identity-disjoint-validation-folds`) — `frozen/cv_manifest.json` 正本
- **Fold-safe FPNet** (`dmitriigluzdov/casmi26-fold-safe-fpnet`) for CV 評価 — public FPNet はリーク
- Enveda-np-examples ホールドアウト (hidden test に最も近い distribution)

### 5.3 判断保留 (Week 1 で情報収集してから決める)
- MIT MSG `[M+K]+` adduct 対応（ICEBERG 2.1 に含まれていない adduct の fallback 戦略）
- MS2DeepScore 2.0 の CC-BY-4.0 weight 使用時の attribution ノート記法
- MIST の 72M params inference time を Kaggle Notebook 9h GPU budget に収める戦略

## 6. 疑問が残っている open question

1. **MIST の timsTOF 域での性能**: 全文献で MSG (主に Orbitrap) 学習、timsTOF cross-domain は未評価。local CV で測定する必要あり
2. **DreaMS fine-tune の必要性**: zero-shot で十分か、fingerprint head を add して fine-tune すべきか
3. **FRIGID の inference cost**: 10 refinement rounds で 41.3 s/spectrum → 1,213 spectra × 41.3 s = 14 時間 → 全スペクトルは無理、hard spectra に絞る必要（基準は何？）
4. **LambdaRank の group サイズ**: 推奨は 25 だが、学習時にもっと多くの negative を入れる (例: 50-100) ことで性能上がるか
5. **分子式 constraints**: 全パイプラインで `precursor_formula` を使うか？ MIST は formula-aware attention、FRIGID は formula conditioning、FIORA は不要。CASMI26 では precursor_mz が与えられている → formula を enumeration できる → 使うべき

## 7. 次アクション (Week 1 開始の最初の 3 件)

1. **誤差分解ツール実装** `src/casmi26/analysis/error_decomp.py`
   - GT 分子に対して "候補 pool に入っているか / FP prediction で top-N に入るか / forward MS score で top-25 に入るか / ranker で top-1 を取れるか" の 4 段分類
   - Class 1 (library hit) / Class 2 (PubChem hit) / Class 3 (de novo generation 必要) ラベル付けを実装
2. **最弱ベースライン再現**: `haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline` を identity-disjoint fold 0 で CV 測定、LB に 1 本 submit
3. **LambdaRank 置換の PoC**: 既存 kernel の最終 ranker を LightGBM lambdarank に変える 15 行変更、CV で +0.02 MRR 確認

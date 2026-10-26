---
competition: enveda-CASMI26-molecule-id-mass-spectra
checked_at: 2026-10-25
topics_reviewed: 12 (raw/)
---

# CASMI 2026 Discussion 要約

## 1. 外部データ・事前学習重み (最新確定ルール)

### 1.1 使用可（ホスト公式確認済み）
| カテゴリ | 項目 | ソース topic | 備考 |
|---|---|---|---|
| **ライセンス緩和** | **CC BY-NC 系 OK** (2026-10-17 更新) | 750330 | 以前は NC 禁止、以後緩和 |
| スペクトルライブラリ | GNPS, MassBank, MoNA | 749217 | 全部可 |
| | GNPS-NIST14-MATCHES | 749217 | **名前に NIST を含むが可** |
| | MassSpecGym | 741359, 749217 | 可 |
| | Enveda-180 (Zenodo) | 748882 | 可 |
| 構造 DB | PubChem, PubChemLite, COCONUT, ChEBI, LIPID MAPS | 741359, 749217 | 可 |
| | LOTUS, NPAtlas | 748882 | 独自ライセンスで可 |
| 事前学習モデル | **MS2DeepScore 2.0, FIORA-OS** | 751373 | GNPS/MassBank/MoNA 学習 → OK |
| | **GLACIER / ICEBERG (coleygroup/ms-pred)** | 748882 | MassSpecGym 学習 → OK |
| | **DreaMS SSL backbone (ssl_model.ckpt)** | 748882 | **ただし train 内に NIST 派生あり**、議論中（Marie 回答: "weights 自体が NIST 派生でなければ OK"、749217） |
| | **CFM-ID 4** | 749217 | 可 |
| | **MIST (Zenodo 23303289, 2026-10-20 公開)** | 751373 | MIT, MassSpecGym 学習、議論中（要再確認） |
| | **FRIGID (Zenodo 19685145, CC BY-NC 4.0)** | 748882 | ホスト確認: OK（Marie 回答済） |
| | **TabPFN v2 (Apache 2.0)** | 748882 | OK |
| | 化学言語モデル / 構造グラフモデル (PubChem のみ学習、スペクトル不使用) | 751373 | OK, prior として使用可 |
| ranking prior | **PubChem CID-PMID / CID-Patent / CID-SID 件数** | 748882 | 可、公開 kernel 全員採用 |

### 1.2 使用不可（絶対禁止）
| 項目 | ソース | 理由 |
|---|---|---|
| **NIST ライブラリ全体** | 750330 | オープン再配布不可 |
| **NIST 派生データ** (nist20_mona_clean_merged_spectra_dreams_hidden_nist20.hdf5 等) | 749217 | 同上 |
| **SIRIUS / CSI:FingerID** | 751373 | 訓練データが完全公開でなく NIST を含むため |
| **TabPFN 2.5 以降** | 748882 | 非商用ライセンス |
| 商用プロプラ DB (NIST 以外も含む非公開ソース) | 741359 | オープンアクセスでない |

### 1.3 判断基準（ホスト確定）
- "approved list は exhaustive ではない。NIST 派生でなければ可" (749217, Marie)
- 判定は **model weights が NIST derived か否か** だけ。訓練コーパスに NIST が混入している models は、weights 自体が NIST 派生かどうかで判断
- 新しい DB / モデルを使う場合は、Discussion で宣言するのが安全（過去事例の黙認は保証されない）

## 2. Winner 要件（Reproducibility 関連）

### 2.1 確定した運用（topic 745072, 748882, 749217）
- **Winner は "元の学習コード + 許可された学習データ" を提示する必要あり**
- **第三者 pretrained 重みを使うには、その重みの original training code + permitted training data にアクセス可能である必要**
- → **ahmedberatozer の `-v4b-models`, `-fpnet-full1`, `-v2-pool`, `-v3-models` 系の学習コードは未公開** → 現状のまま winner に出すとリスク
- Marie 公式回答: 公開 Kaggle Dataset 内のコードは Foundational Rule 6.b の OSS 扱い。ただしコードがない重みだけでは不可

### 2.2 2026-10-21 時点の運用変更（Marie 回答）
- 既往の非再現提出は medal 剥奪はされないが prize eligibility 失う
- **Final 2 本の reproducibility チェックが本質的な審査対象**
- 残り期間（2026-12-14 まで）: 再現性のある 2 本を Final として選ぶ

### 2.3 本プロジェクトへの含意（kaggle_flow_v1.md §1 L5 と直結）
- 他人の重みを使う場合、training code を遡れないものは **CV 検証用に留め、Final 提出には使わない**
- 自前学習の重みは全て `experiments/EXP-XXXX/env.json` に training code の git HEAD + 学習データ ソースを紐付ける
- Winner License (MIT) 提出前提で、使用 OSS の license matrix を Week 4 に用意する

## 3. CV / データリーク関連

### 3.1 Class 分類（公式明示でないが Discussion 共通認識）
| クラス | 定義 | 割合 (推定) | 公開 LB 到達 |
|---|---|---|---|
| Class 1 | train にリファレンススペクトルあり | ~16-17% (hidden test) | 単独で MRR 寄与 ~0.16 が天井 |
| Class 2 | 構造のみ既知、スペクトルは train になし | 大半 | PubChem retrieval + ranker の寄与 |
| Class 3 | 完全新規化合物 | 小さいが決定的 | **全員伸び悩み** |

### 3.2 Class 3 の LB 停滞（topic 751300）
- 複数チーム: 「ローカル改善 +0.003-0.004 でも LB flat」
- Mansur: "train から切った validation はもう信じていない。真の新規性に対して楽観的すぎる"
- Tra Crew: "class 3 lever で LB +0.003-0.004 は取れたが、以降停滞"
- **本コンペの最大の壁は Class 3** → ここが差分の勝ち筋

### 3.3 配布 test と hidden test の分布差（kernel 分析からの傍証）
- 配布 `test.parquet` は 100% `enveda-180`（合成 drug-like）
- hidden test は timsTOF 計測の natural products
- `kitopl` の "Data Pitfalls" で詳細。validation 設計には **natural products × inchikey14 holdout** が必須

### 3.4 inchikey14 canonicalization（kernel 分析からの傍証）
- train の `inchikey14` は RDKit tautomer canonicalization 未適用
- 公式スコアラは RDKit 2026.03.3 で canonicalize
- → **3,167 構造 (1.15%) が同一スコアキーに collapse** → 0.48% MRR の "見えない損失"
- 対策: ローカルスコアで scoring キーを `canonical_inchikey14` で統一

## 4. スコアラ・LB 挙動

### 4.1 MRR 計算の挙動（topic 748882, 751474）
- "first-match MRR" 挙動確認中: ランク 1 が同じで下位候補を変えても MRR 下がらない保証があるかはホスト未回答
- public/private split の stratification 詳細は非公開
- 公式スコアラの script version pinning 問い合わせ中 (751474)

### 4.2 Public vs Private split
- 2026-10-13 時点では **public 31% / private 69%**（ホストは詳細非回答だが、議論内で言及）
- 配布 test 1,213 スペクトル / 400 molecule_id だが、hidden test は再構成（class 分布含めて）

## 5. 本プロジェクトへの actionable_points

### 5.1 戦略に取り入れるべき最新情報
1. **CC BY-NC 緩和（10/17）**: HMDB など NC 系データが使える → ランキング特徴量の幅が広がる
2. **MIT 系の MIST は使える**（10/20 公開 Zenodo 23303289）。CASMI26 公開 kernel ではまだ本格採用されていない → 差分注入先候補
3. **DreaMS SSL backbone は使える**（公式確定）、ただし training data に NIST 派生が混入している点をホストが weights 単位で判定 → OK 認定済み (749217)
4. **FRIGID (CC BY-NC)**: MIST ベースの diffusion decoder、"de novo + retrieval" のハイブリッドができる → 公開 kernel で未活用

### 5.2 CV 設計の正本（Discussion と kernel 分析の統合）
- **inchikey14 ベースの disjoint fold を正本採用**: `dmitriigluzdov/casmi26-identity-disjoint-validation-folds`
- **canonical_inchikey14** で scoring（RDKit 2026.03.3 pin）
- **natural products ホールドアウト** を追加（配布 test は enveda-180 偏重で LB を代表しない）
- **instrument_type (timsTOF) 偏重** を考慮したサブセット CV も用意

### 5.3 Final 提出の運用ルール
- Final 2 本は **両方とも完全再現可能** に揃える
- 使用した他人重みは全て "training code + permitted training data" を遡れるものに限定
- 使えない重み (fpnet-full1 等) を使った実験は CV 検証用に留める
- Final 選定前に Winner License (MIT) 準拠の license matrix を作る

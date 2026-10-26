---
competition: Enveda CASMI 2026 — Molecule ID From Mass Spectra
slug: enveda-CASMI26-molecule-id-mass-spectra
source_url: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
checked_at: 2026-10-22
deadline: 2026-12-14 23:59 UTC
reward: 50,000 USD
teams_at_check: 3589
---

# Phase A overview — Enveda CASMI 2026 — Molecule ID From Mass Spectra

> **公式本文の確認状況:** Evaluation (MRR@25) と Rules はユーザー提供分で確定。Submit は Notebook 提出想定。

## 1. 基本情報
| 項目 | 値 |
|---|---|
| コンペ名 | Enveda CASMI 2026 — Molecule ID From Mass Spectra |
| slug | `enveda-CASMI26-molecule-id-mass-spectra` |
| 主催 | Enveda Therapeutics, Inc. (5700 Flatiron Parkway, Boulder, CO 80301) |
| 締切 | 2026-12-14 23:59 UTC（残 53 日） |
| 賞金総額 | 50,000 USD（1st 16k / 2nd 12k / 3rd 9k / 4th 7k / 5th 6k） |
| カテゴリ | Featured |
| ユーザー状態 | エントリー済み |

## 2. 評価指標 (確定)

**Mean Reciprocal Rank @ 25 (MRR@25)**

$$\mathrm{MRR}@25 = \frac{1}{U}\sum_{u=1}^{U} \frac{1}{\mathrm{rank}_u}$$

- $U$: molecule 数（= test の unique molecule_id = **400**）
- $\mathrm{rank}_u$: 分子 $u$ の **ranked list の中で最初に正解となった構造の順位** (1〜25)
- 25 候補中に正解がなければ $0$ 点
- 1 位 = 1.0 / 2 位 = 0.5 / 3 位 = 0.333 / 5 位 = 0.2 / 10 位 = 0.1 / 25 位 = 0.04
- 正解は **1 分子につき 1 構造のみ**（複数正解・部分点なし）

**指標感度の第一近似 (Week 0 で実測して裏取り):**
- 1 分子の rank が 1→2 に落ちる = 0.5 点損 ≒ 全体の `0.5/400 = 0.00125` 点損
- 1 分子の rank が 1→5 に落ちる = 0.8 点損 ≒ 全体の `0.002` 点損
- 「圏外→1 位」= +1.0 点 ≒ 全体で +0.0025 点
- **順位 1〜3 の精度が最大の勝ち筋**。4 位以下の順位最適化は限界効用が急減（1/4=0.25 → 1/25=0.04 の間で単調減）
- 400 分子 × 25 候補 = 10,000 候補の順位付け。1 位の精度が 1% 変わる (= 4 分子が 1 位→圏外) と MRR は 0.01 動く

## 3. 提出形式
- **ファイル名:** `submission.csv`
- **行数:** 400 行（test の unique molecule_id 数）
- **列:** `molecule_id`, `smiles`
- **smiles 列:** `;` 区切りで**最大 25 SMILES**を順位順に並べる
- **提出経路:** **Notebook 提出 (Code Competition)** 想定（ユーザー申告）
- **1 日あたり submit 数:** 5 / 日
- **Final Submission:** 最大 2 本選択

## 4. 制約 (確定: 公式 Rules より)

### データライセンス
- **CC BY-NC 4.0**（Competition Use and Non-Commercial & Academic Research）
- 配布データの再配布・商用利用禁止、未参加者への共有禁止

### 外部データ / 外部ツール (§2.6)
- **外部データ使用は原則可**（Host が明示禁止しない限り）
- 条件: "publicly available and equally accessible" または Reasonableness Standard を満たす
- **「賞金額を超えるライセンス料のデータセット購入は不可」** という例示あり → PubChem (無料), HMDB (無料), MoNA (無料), GNPS (無料), MassBank (無料) は OK
- NIST (有料商用) は Reasonableness Standard でグレー。使うなら Discussion で宣言するのが安全
- AMLT (AutoML tools) 使用可

### 事前学習モデル
- 明示禁止なし → 使用可。Open Source Initiative 準拠ライセンスのモデル (MIST, SpecTUS, MS2DeepScore 等) は問題なし
- 「組込み外部データ/事前学習重みが incompatible license の場合、winner はそれら部分のオープンソース化義務を免除」(§2.5.a) → **ライセンスが CC-NC や GPL でも winner 時に差し障りが出ない設計**

### Winner License (§2.5.a)
- 自分のソースコード・モデル部分は **OSI 承認 open source ライセンスで提供必須**（商用利用を制限しないもの、= MIT / Apache 2.0 / BSD 等）
- 詳細な手法ドキュメント（methodology, 前処理, loss, 学習, hyperparams, 再現可能なコードリポジトリ）の提出義務あり → **実験再現性は Week 0 から徹底する（bio-hub L5 の失敗を繰り返さない）**

### Code / 公開共有
- Private 共有 (チーム外) 禁止
- Public 共有 (フォーラム / Notebook) は OK、OSI 承認オープンソースライセンスでの共有扱いになる

### Team
- Team 最大 5 名、merger 可（Submission 合計が制限内）

### 推論時間 / GPU
- Rules には明示なし。Notebook 提出型の Kaggle 慣例（例: BirdCLEF は 90min CPU）に従う可能性 → **Overview > Code Requirements タブで要確認**（未取得）

## 5. 配布データ (ローカル配置済み)

配置先: `data/input/enveda-CASMI26-molecule-id-mass-spectra/`

| ファイル | サイズ | 行数 | 列数 |
|---|---:|---:|---:|
| `train.parquet` | 約 3.0 GB | 2,539,608 | 19 |
| `test.parquet` | 約 4.8 MB | 1,213 | 12 |
| `sample_submission.csv` | 約 44 KB | 400 | 2 |

### train スキーマ (19 列)
`ingest_lib, normalized_smiles, inchikey, inchikey14, molecular_formula, ionization_mode, instrument_type, adduct, adduct_orig, precursor_mz, precursor_error_ppm, ms2_mzs, ms2_normalized_intensities, num_peaks, base_peak_intensity, collision_energy_ev, collision_energy_orig, collision_energy_orig_units`

- unique SMILES: 277,566
- unique inchikey14: 275,810
- unique inchikey: 275,950

### test スキーマ (12 列)
`molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_orig, collision_energy_ev, collision_energy_orig_units`

- unique molecule_id: **400** (= 提出行数)
- unique spectrum_id: 1,213
- 1 molecule_id あたり 1〜9 スペクトル (median 3, mean 3.03)

### ドメインシフト (train → test)
| 項目 | train | test |
|---|---|---|
| instrument_type | timsTOF 45%, Orbitrap 29%, QTOF 系多数 (表記ゆれあり) | **全て timsTOF** |
| ionization_mode | positive 75% / negative 25% | positive 81% / negative 19% |
| adduct top3 | `[M+H]+` 52% / `[M-H]-` 18% / `[2M+Na]+` 7% | `[M+H]+` 79% / `[M-H]-` 16% / `[M+CH2O2-H]-` 3% |
| precursor_mz | min 2 / max 2.43e6 (**外れ値混入**) median 350 | 245–460 (median 329) **狭レンジ** |
| num_peaks | median 42 / max 73,318 (**外れ値**) | median 230 / max 3,259 |
| collision_energy_orig_units | 要確認 | **全て eV** |
| ingest_lib | enveda-180 46%, pluskal_ms2 21%, riken 14%, gnps 9%, massbank 4%, mona 4%, ほか | — |

## 6. 過去類似コンペ (`doc/solution/`)

- `doc/solution/` は 2023 / 2024 / 2025 とも **全て BirdCLEF 系** — 分子 ID / MS2 系の過去上位 writeup はストックなし
- 外部 CASMI 歴代（2012-2024）および MIST / SIRIUS+CSI:FingerID / MS2DeepScore / SpecTUS / MassFormer / MS2Mol の文献サーベイが必要
- Week 0「過去コンペ横断分析」では、`doc/solution/2024/` の BirdCLEF データは使えないため、`doc/discussion/2026/casmi26/` に **論文サーベイ** を 2〜3 本落とす形で代替する

## 7. 参加判断 → **参加確定**

- 公式ルールが確定し、「外部データ可 / 事前学習モデル可 / Code Competition」が揃った
- 評価指標 MRR@25 は 1〜3 位の精度に強く依存 → 「上位候補の生成 + ランク付け」が明確に勝ち筋
- 残 53 日で kaggle_flow_v1.md Week 0〜4 の 1 サイクルがちょうど収まる

## 8. 残っている未確認項目 (Phase C で潰す)

1. **Overview > Code Requirements タブ**: 推論時間制限 (CPU/GPU) の明示値。見つけ次第 overview.md 追記
2. **Overview > Timeline タブ**: Entry Deadline / Team Merger Deadline の正確な日時
3. **Discussion フォーラムの重要スレッド**: 禁止外部データ / 禁止モデルがアナウンスされていないか

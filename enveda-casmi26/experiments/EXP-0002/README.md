# EXP-0002: CV と LB のずれの切り分け (Class 1 模擬 CV + クラス構成の逆算)

- **日付:** 2026-10-27
- **コード:** `scripts/casmi26_baseline_cv.py --reference other`, `scripts/casmi26_cv_lb_bridge.py`
- **前提:** EXP-0001 の A (固定融合) LB 0.191 / B (analog 単独) LB 0.247。CV の差 +0.34 に対し LB の差は +0.056 だった
- **LB:** 新規提出なし

## 仮説

CV と LB の幅のずれは、CV にない 2 つのクラスで説明できる。

1. **Class 1** (reference スペクトルが train にある): ライブラリ一致が正解を指すので A が有利
2. **pool 外** (正解が候補 pool = COCONUT ∪ train 構造にない): A も B も 0 点

## やったこと

1. **Class 1 模擬 CV (`--reference other`):** ホールドアウト構造のスペクトルのうち、query と同じソース (np: enveda-np-examples、fold0: timsTOF) のものだけライブラリから除き、他装置・他ライブラリの reference は残す
2. **逆算:** クラス別 MRR を CV で実測し、2 本の LB を連立して hidden test のクラス構成 (p1, p2, p3) を解く

   LB_A = p1·A1 + p2·A2 、 LB_B = p1·B1 + p2·B2 、 p3 = 1 − p1 − p2 (pool 外は 0 点)

   不確かさは LB の SE ±0.03 と CV 側の分子 bootstrap (4,000 回) で出した

## 結果

### クラス別 MRR@25 (CV 実測)

| | Class 1 (other reference) | Class 2 (reference なし) | pool 外 |
|---|---:|---:|---:|
| A 固定融合 | **0.914** (np, 250 分子) | 0.194 (np) | 0 |
| B analog 単独 | 0.637 (np) | **0.535** (np) | 0 |
| (参考) ライブラリ単独 | 0.927 (np) | 0.053 (np) | 0 |

- np/other では 250 分子すべてに別ライブラリの reference があり、全員 Class 1 になった
- fold0/other で Class 1 になったのは 4 分子だけ (enveda-180 構造は timsTOF 以外の reference をほぼ持たない)。fold0 の Class 1 MRR (A 0.875 / B 0.542) は参考値

### 逆算したクラス構成 (np の実測値を使用)

| | 点推定 | 10–90% 区間 |
|---|---:|---:|
| p1 Class 1 | **0.148** | 0.087 – 0.208 |
| p2 Class 2 | 0.285 | 0.164 – 0.408 |
| p3 pool 外 | **0.567** | 0.485 – 0.648 |

- bootstrap の 99.8% で 3 つとも 0–1 の範囲に収まった (解として成立する)
- fold0 の値で解いても p1 0.192 / p2 0.239 / p3 0.569 で、ほぼ同じ
- **独立した照合:** p1 = 0.148 は Discussion での独立推定 (Class 1 ≈ 16–17%) とよく一致する

## 解釈

1. **ずれは説明できた。** CV にない Class 1 (A 有利) と pool 外 (両方 0 点) を入れると、A・B 両方の LB が同時に再現できるクラス構成が、外部推定と矛盾しない形で存在する
2. **最大のボトルネックは候補 pool。** hidden test の約 57% は、正解が COCONUT ∪ train 構造にないと推定される。今の pool のままでは、ランカーが完璧でも MRR の上限は p1 + p2 ≈ 0.43 にとどまる。公開 kernel の LB 0.467 がこの上限を超えているのは、PubChem 系の候補 (106M 構造) を混ぜているためと整合する
   - 注: ここでの「pool 外」は「この pool にない」という意味で、Discussion の Class 3 (完全新規) とは別物。PubChem には入っている分子が多いはず
3. **ランカーはクラスで使うべきチャネルが違う。** Class 1 ではライブラリ (A 0.914 vs B 0.637)、Class 2 では analog (B 0.535 vs A 0.194)。ライブラリ一致を信じるかどうかを学習で切り替えるランカーが要る
   - 理想的に切り替えられた場合の予測: LB ≈ 0.148 × 0.914 + 0.285 × 0.535 ≈ **0.288** (B から +0.04)

## 注意 (この推定の限界)

- 未知数 2 つに式 2 本なので、ちょうど解けてしまう。モデルが正しいかの検定にはならない。外部推定 (p1 ≈ 16–17%) との一致が唯一の独立な確認
- CV のクラス別 MRR が hidden test でもそのまま成り立つ前提。np holdout は hidden test に最も近い proxy だが、同じ分布とは限らない
- 次の提出で予測 LB を事前に書き、外れるかを見れば、このモデルを反証できる (例: 上の 0.288)

## 次に試すこと

1. **候補 pool の拡張 (最優先):** PubChemLite (手元にある `casmi26-pubchemlite-cc0-candidate-fingerprints`) や PubChem tier を pool に足し、上限 0.43 を引き上げる。候補数が増えて Class 1/2 の順位が下がる影響も同時に測る
2. **学習済みランカー:** ライブラリ一致の信頼度で切り替えを学習させる (nested 構成)。目標は上の予測 0.288
3. **CV の採用判定に Class 1 系列を追加:** np/keep (Class 2) と np/other (Class 1) の両方を、p1 / p2 で重み付けして見る

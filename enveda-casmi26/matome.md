# Enveda CASMI 2026 振り返りまとめ

**競技:** Enveda CASMI 2026 — Molecule ID From Mass Spectra
**評価指標:** MRR@25
**自前最良 LB:** TBD
**乗合いで到達した最終 LB:** TBD
**上位の到達:** TBD
**作成日:** TBD（コンペ終了後）

---

> **本ファイルはコンペ終了後に埋める。** `doc/strategy/kaggle_flow_v1.md` §8 の振り返りテンプレに沿う。

## 1. 自前アプローチと上位解法の決定的な違い
- 共通の土台 (使った重み・公開 kernel)
- 技術差分 3 軸 (このコンペのパイプライン段で切る: 例 Spectrum encoder / Candidate retrieval / Reranker)
- 上位の段差ごとの効き目要素

## 2. Validation 設計の振り返り
- 機能していた部分
- 機能しなくなった時期と原因
- 教訓（次回に残すべきルール）

## 3. 中核原則 (§1) への準拠と逸脱
- L1〜L5 のうちどれを守り、どれを破ったか
- 破った理由と、それが結果にどう効いたか

## 4. 次版 (kaggle_flow_vN+1.md) に追加すべき Lessons
- 本コンペで新しく見つけた制約・原則

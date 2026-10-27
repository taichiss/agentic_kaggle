# Enveda CASMI 2026 — strategy/todo.md

## Phase C — Week 0 開始チェックリスト

> 全項目潰すまで Week 1 に進まない (`doc/strategy/kaggle_flow_v1.md` §3)

- [x] コンペ基本情報 (overview.md 1 枚) — [enveda-casmi26/overview.md](../overview.md)
- [x] データ全量確認 — 配布 train 2.54M rows / 19 cols、test 1,213 rows / 400 molecules、SHA256 一致確認済み（`frozen/cv_manifest.json`）
- [x] 指標の因数分解実装 `src/casmi26/metrics/breakdown.py`（11 件 pytest green、[`tests/casmi26/test_breakdown.py`](../../tests/casmi26/test_breakdown.py)）
- [x] Metric Sandbox 実測 → [`enveda-casmi26/metric_sensitivity.md`](../metric_sensitivity.md) + `metric_sensitivity.json`
- [x] 指標ハッシュ freeze → [`enveda-casmi26/frozen/metric_hash.txt`](../frozen/metric_hash.txt)
- [x] CV 設計の確定 → [`enveda-casmi26/frozen/cv_manifest.json`](../frozen/cv_manifest.json)（identity-disjoint 5fold + enveda-np-examples holdout）
- [x] 公開 kernel 分析 → `doc/kernel/2026/casmi26/summary.md` (12 本分析、E006 共通パイプライン特定)
- [x] Discussion 分析 → `doc/discussion/2026/casmi26/summary.md` (外部データ・事前学習重み・CV 落とし穴の確定ルール)
- [x] **過去 CASMI/MassSpecGym 系の文献サーベイ** → `doc/discussion/2026/casmi26/papers/` に 4 本完了
  - [x] [foundation_models_survey.md](../../doc/discussion/2026/casmi26/papers/foundation_models_survey.md) — MIST / DreaMS / MS2DeepScore 2.0
  - [x] [forward_predictors_survey.md](../../doc/discussion/2026/casmi26/papers/forward_predictors_survey.md) — MassSpecGym / ICEBERG / GLACIER / FIORA / FRIGID
  - [x] [ltr_survey.md](../../doc/discussion/2026/casmi26/papers/ltr_survey.md) — LambdaRank / ListNet / Softmax / ApproxMRR + 4 コードスケッチ
  - [x] [local_evidence.md](../../doc/discussion/2026/casmi26/papers/local_evidence.md) — ローカル実地調査 (CV folds / Panel benchmark / Popularity prior)
  - [x] [survey_summary.md](../../doc/discussion/2026/casmi26/papers/survey_summary.md) — 全体シンセシス（LB 0.467 → 0.53-0.60 の 5 日ロードマップ）
- [x] 採用規律の明文化 → [strategy/current.md](current.md)（8 項目、公開 FPNet リーク / MIST padding mask 警告を含む）

## Phase C 残作業

- [ ] **RDKit version diff test**: ローカル 2026.09.1 と公式 pin 2026.03.3 の `canonical_inchikey14` 挙動差分測定（Week 1 冒頭に実装、差分が有意なら 2026.03.3 を pin）

## Phase D (Week 1) — 誤差分解ゲート (`doc/strategy/kaggle_flow_v1.md` §4)

- [x] 誤差分解ツール: `src/casmi26/analysis/error_decomp.py` + CLI `scripts/casmi26_error_decomp.py`（2026-10-27、pytest 11 件 green、合成データでのみ動作確認）
  - GT 分子に対して「候補 pool に入っているか / FPNet 回帰で top-N に入っているか / Simulator (ICEBERG/GLACIER) overlay で top-25 に入っているか / ranker で top-1 を取れているか」の 4 段分類
  - Class 1/2/3 ラベル付きで集計
  - 段の損失 = 最初に落ちた段に (1 - 1/rank) を全額計上。合計は 1 - MRR に一致
- [x] 誤差分解を実データで回す → EXP-0001 (np / fold0 × pool keep / drop の 4 条件)
- [x] 最弱ベースラインの fold-safe 再現: V17 の ch1–3 のみ (FPNet・学習済みランカーはリークするため不使用)。np/keep=0.1944, fold0/keep=0.0955
- [ ] 最弱ベースラインの Kaggle 提出: kernel A (固定融合) / B (analog 単独) を `enveda-casmi26/kernels/` に用意しローカル試走済み。push と submit はユーザー確認待ち。予想は EXP-0001/README.md に固定済み
- [x] 誤差分解でボトルネック特定 → ランカー段が損失の 80–90%。[strategy/current.md](current.md) に反映
- [ ] Week 1 Exit 条件: ボトルネック順位の特定 → keep 条件では達成。Class 3 比率 (drop 条件の現実的な見積もり) は未確定
- [ ] **学習済みランカー (nested 構成)**: fold-train 側から学習 query を作り、HistGBM pointwise vs LightGBM LambdaRank を同一特徴量で比較 (roadmap #1)
- [ ] Class 3 比率の見積もり: train 構造を含む COCONUT か PubChemLite pool で drop 条件を作り直す
- [ ] `src/casmi26/metrics/breakdown.py` の mypy エラー (RDLogger.DisableLog の attr-defined)。指標ハッシュ freeze 対象なので、直すなら metric_hash.txt を同時更新

## 外部データ取得状況（Phase C 完了部）

**取得済み** (`data/input/external/casmi26/`):
- [x] `prvsiyan/coconut-casmi26-candidates` — 404 MB (COCONUT 462k 候補 + FP)
- [x] `dmitriigluzdov/casmi26-identity-disjoint-validation-folds` — 61 MB (**CV 正本**)
- [x] `franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints` — 345 MB
- [x] `dmitriigluzdov/casmi26-natural-product-knowledge-table` — 130 MB
- [x] `dmitriigluzdov/casmi26-natural-product-panel-benchmark` — 223 MB
- [x] `dmitriigluzdov/casmi26-pubchem-popularity-prior` — **部分取得** (bulk zip が 404、pool_* 系 14 ファイル 190 MB のみ個別取得。PubChem tier 全体 ~12 GB は Week 1 必要時に判断)

**保留 / 未取得**:
- [ ] `prvsiyan/casmi26-forward-reference-stack` (3.8 GB, ICEBERG/GLACIER オフラインスタック) — Week 1 の Simulator 段再現時に取得
- [ ] MIST Zenodo 23303289 (10/20 公開 MIT) — Week 2 以降、差分モジュール設計時に取得
- [ ] DreaMS SSL backbone (hengck23 kernel 内のパス) — Week 2 以降
- [ ] `ahmedberatozer/casmi26-fpnet-full1` 等の FPNet 系 — 学習コード未公開のため **Final 提出不可**、ローカル CV 検証用のみ

## 保留 / 未確定

- [ ] Overview > Code Requirements タブで推論時間制限の明示値を確認
- [ ] Overview > Timeline タブで Entry Deadline / Team Merger Deadline 確定
- [ ] 禁止外部データ・禁止モデルが Discussion でアナウンスされていないか継続監視
- [ ] 公式スコアラ (Kaggle Notebook 版) との 1-to-1 比較テスト（RDKit 版差分吸収のため）

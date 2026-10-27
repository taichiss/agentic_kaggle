# Enveda CASMI 2026 — strategy/experiments.md

> 代表実験の解釈ログ。`experiments/registry.csv` の中から深堀りする分だけここに書く。
> 1 行 1 実験の数値記録は `experiments/registry.csv` 側で管理する（本ファイルは "なぜその数値になったか" を残す場所）。

## テンプレ

### EXP-XXXX: <タイトル>
- **日付:** YYYY-MM-DD
- **git HEAD:** `<sha>`
- **config:** `experiments/EXP-XXXX/config.yaml`
- **CV score:** inchikey14-gkf5=X, instrument-holdout=Y
- **LB score:** Z
- **仮説:** （何を検証したかったか）
- **結果:** （数字と、期待との差分）
- **解釈:** （なぜその結果になったと考えるか）
- **指標感度との整合:** （Week 0 Metric Sandbox の感度に照らしてこの変化は妥当か）
- **次に試すこと:** （この実験から派生する次アクション）

### EXP-0001: fold-safe baseline 再現 + 誤差分解
- **日付:** 2026-10-27
- **git HEAD:** `b4f8f2e` + 未コミット差分 (本実験のコミットで確定)
- **config:** `enveda-casmi26/experiments/EXP-0001/<条件>/summary.json` の `config`、詳細は [EXP-0001/README.md](../experiments/EXP-0001/README.md)
- **CV score:** np/keep=0.1944, fold0/keep=0.0955 (np/drop=0.0042, fold0/drop=0.0003)
- **LB score:** A 固定融合 0.191 / B analog 単独 0.247 (2026-10-27)
- **仮説:** haideptry V17 のうちリークなしで組める ch1–3 だけで、どの段が MRR を落としているかを分解できる
- **結果:** keep 条件の pool カバー率は 99.6–100%、損失の 0.80–0.90 がランカー段。単一チャネルでは analog 0.535 (np) / frag 0.617 (fold0) なのに、固定重み融合は 0.194 / 0.096
- **解釈:** identity-disjoint CV では正解の reference スペクトルは必ず除外されるので、ライブラリ一致 (94–99% の分子で発生、類似度 median 約 0.6) は常に train 側の別異性体を指す。固定重みの融合ではこれが正解を押しのける (融合 top-1 の 66–76% がライブラリ top-1 と一致)。V17 の 0.30–0.34 は FPNet と学習済みランカー込みの数字で、本実験は再現値ではない
- **指標感度との整合:** rank 2–10 に 112 (np) / 84 (fold0) 分子が滞留。metric_sensitivity.md §4 の「rank 1–3 を直す変更が最優先」と整合し、次はランカー
- **次に試すこと:** nested 構成で fold-train 側から学習 query を作り、HistGBM pointwise と LightGBM LambdaRank を同じ特徴量で比較 (roadmap #1)。ライブラリ一致は lvmax 等の相対特徴量で学習させる
- **LB 照合 (2026-10-27):** A=0.191, B=0.247。B − A = +0.056 で事前基準 (+0.03) を超え、CV の方向は LB に引き継がれた。ただし CV の差 (+0.34) は LB の約 6 倍で、B の CV (0.535) は LB を大きく上回る。pool 外分子 (Class 3) と Class 1 が CV にないことが原因と推定 (未検証)

### EXP-0002: CV と LB のずれの切り分け
- **日付:** 2026-10-27
- **config:** `--reference other` (query と同じソースの reference だけ除く)、逆算は `scripts/casmi26_cv_lb_bridge.py`。詳細は [EXP-0002/README.md](../experiments/EXP-0002/README.md)
- **CV score:** Class 1 模擬 (np/other): A 0.914 / B 0.637。Class 2 (np/keep, EXP-0001): A 0.194 / B 0.535
- **LB score:** 新規提出なし (EXP-0001 の A 0.191 / B 0.247 を使用)
- **仮説:** CV と LB の幅のずれは、CV にない Class 1 (A 有利) と pool 外 (両方 0 点) で説明できる
- **結果:** 逆算したクラス構成は Class 1 0.148 [0.09–0.21] / Class 2 0.285 [0.16–0.41] / pool 外 0.567 [0.48–0.65] (10–90%)。fold0 で解いてもほぼ同じ
- **解釈:** p1 が Discussion の独立推定 (16–17%) と一致し、ずれは説明できた。hidden test の約 57% は正解が COCONUT ∪ train にないと推定され、今の pool では MRR 上限が約 0.43。Class 1 はライブラリ、Class 2 は analog が強い
- **指標感度との整合:** pool 外の分子を pool に入れて rank 1 にできれば 1 分子 +0.0025。57% (推定 228 分子) が対象で、ランカー改善より上限が大きい
- **次に試すこと:** PubChemLite / PubChem tier で pool 拡張 (最優先)、ライブラリ一致の信頼度で切り替える学習済みランカー (予測 LB ≈ 0.288)

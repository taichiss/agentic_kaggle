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

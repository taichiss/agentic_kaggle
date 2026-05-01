# BirdCLEF+2026 Workspace

この README は人間向けの作業入口です。  
AI 向けの短い指示は `AGENTS.md`、詳細な運用ルールは `.codex/codex.md` にあります。

## 目的

このワークスペースは BirdCLEF+2026 だけに固定せず、Kaggle の音声・分類系コンペへ流用できる形で管理します。  
公式情報、過去コンペ知見、実験ログ、実装、品質ゲートを分けて蓄積する前提です。

## ディレクトリ構成

```text
.
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── data/
│   └── input/
├── doc/
│   ├── overview/
│   ├── discussion/
│   │   ├── 2025/
│   │   └── 2026/
│   ├── kernel/
│   │   └── 2026/
│   └── solution/
│       ├── 2023/
│       ├── 2024/
│       └── 2025/
├── strategy/
│   ├── current.md
│   ├── experiments.md
│   └── todo.md
├── scripts/
│   ├── archgate.py
│   ├── bootstrap_quality.sh
│   └── quality_gate.py
└── tests/
    └── test_archgate.py
```

## 各ディレクトリの役割

- `data/input/`
  - 競技データの配置先です。必要なら `data/input/<competition-slug>/` で分けます。
- `doc/overview/`
  - 公式ルール、評価指標、提出形式、制約、データ説明の要約を置きます。
- `doc/discussion/`
  - Kaggle Discussion 由来の知見を保存します。
- `doc/kernel/`
  - Notebook / Kernel 由来の知見を保存します。
- `doc/solution/`
  - 過去コンペの上位解法まとめです。
  - 各年の `summary.md` は年ごとの横断要約です。
  - 各年の `*.csv` は比較表です。
  - `1th` から `10th` などの拡張子なしファイルは順位別の raw writeup です。
- `strategy/`
  - 現在の方針、実験ログ、未着手タスクを管理します。
- `scripts/`
  - 品質ゲートとアーキテクチャ制約のチェック用スクリプトです。
- `tests/`
  - ルール系スクリプトの最低限の自動テストです。

## 最初にやること

1. 依存を入れる。  
   `bash scripts/bootstrap_quality.sh`
2. 現在対象のコンペ情報を確認する。  
   `strategy/current.md`
3. 公式情報を確認し、足りなければ `doc/overview/` に追記する。
4. 過去コンペの知見を使う場合は `doc/solution/` の `summary.md` から読む。

## 日常の確認手順

### 作業開始時

1. `strategy/current.md` で以下を確認する。
   - `competition`
   - `metric`
   - `submission_format`
   - `constraints`
   - `confirmed_facts`
2. `strategy/todo.md` で今やることを確認する。
3. `data/input/` に必要データが揃っているか確認する。

### 調査時

1. 公式情報は `doc/overview/` に残す。
2. Discussion の要点は `doc/discussion/` に残す。
3. Kernel の要点は `doc/kernel/` に残す。
4. 過去 solution の再利用ポイントは `strategy/current.md` か別メモへ要約して転記する。

### 実験前

1. 仮説を `strategy/current.md` に書く。
2. 実験条件の差分を明文化する。
3. GPU 実行前は以下を確認する。
   - `nvidia-smi`
   - `uv run python -c "import torch; print(torch.cuda.is_available())"`

### 実験後

1. 結果を `strategy/experiments.md` に残す。
2. 次アクションを `strategy/todo.md` に更新する。
3. 方針が変わったら `strategy/current.md` を更新する。

## 品質確認コマンド

- フォーマット  
  `uv run python scripts/quality_gate.py format`
- lint  
  `uv run python scripts/quality_gate.py lint`
- typecheck  
  `uv run python scripts/quality_gate.py typecheck`
- architecture check  
  `uv run python scripts/quality_gate.py arch`
- tests  
  `uv run python scripts/quality_gate.py test`
- 全部まとめて  
  `uv run python scripts/quality_gate.py all`

## `doc/solution/` の見方

- 年全体の傾向を見る  
  `sed -n '1,120p' doc/solution/2025/summary.md`
- 年別の格納物を見る  
  `find doc/solution -maxdepth 2 -mindepth 2 -type f | sort`
- 順位別 raw writeup を見る  
  `sed -n '1,120p' doc/solution/2025/1th`

## 補足

- `AGENTS.md` は AI 用の短いポインタです。人間向け説明はこの README を優先してください。
- `.codex/codex.md` にはより厳密な運用ルール、品質ハーネス、計画運用ルールがあります。

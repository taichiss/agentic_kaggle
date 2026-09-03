# agentic_kaggle

`agentic_kaggle` は、Kaggle コンペティションをエージェントと再現可能に進めるための
ADR（Architecture Decision Record）駆動ワークスペースです。特定のコンペやデータ形式には
依存せず、コンペごとの事実・戦略・コード・評価を分離します。

## 設計の中心

- 継続的な設計判断は `docs/adr/` に残し、コード・テスト・運用ルールから参照する。
- エージェントへの文章指示だけに頼らず、`harness_check` と CI で構造を検証する。
- コンペ固有資産は `competitions/<competition-slug>/` に閉じ込める。
- 生データ、モデル、予測、提出物、認証情報は Git に入れない。
- Notebook は探索に使い、再利用する処理はコンペ配下の `src/` へ移す。
- 公式情報、第三者情報、ローカル実験結果を区別して意思決定する。

## クイックスタート

```bash
uv sync --extra dev
uv run kaggle-init playground-series-s6e1 \
  --title "Playground Series S6E1" \
  --metric RMSE \
  --metric-direction minimize \
  --competition-url https://www.kaggle.com/competitions/playground-series-s6e1
uv run harness_check
uv run pytest
```

`kaggle-init` は `templates/competition/` から独立したコンペワークスペースを生成します。
生成後は `competition.toml` の未確認項目を公式情報で埋め、`strategy/current.md` に最初の
検証計画を記録してください。

## TRUST-LB ワークフロー

この基盤は、提出ファイルの軽量検査後に Kaggle Leaderboard を主フィードバックとして使えます。

```bash
uv run kaggle-lb submit playground-series-s6e1 \
  --file competitions/playground-series-s6e1/submissions/exp-0001.csv \
  --experiment-id EXP-0001 \
  --message "baseline"
```

コマンドは Kaggle CLI で提出し、submission を特定して scoring 完了まで待ち、public/private
score を表示します。結果は対象コンペの `strategy/lb-submissions.jsonl` に追記されます。
Code Competition では、完了済み Notebook の所有者/slug と version を明示します。`--file` は
Notebook が生成した出力と同名のローカル CSV を指定し、提出前検査と台帳の provenance に使います。

```bash
uv run kaggle-lb submit <competition-slug> \
  --file <downloaded-kernel-output>/submission.csv \
  --kernel <owner>/<notebook> \
  --kernel-version <version> \
  --experiment-id EXP-0002 \
  --message "notebook baseline"
```

最新状態だけ確認する場合は次を使います。

```bash
uv run kaggle-lb status playground-series-s6e1 --latest
```

事前に Kaggle CLI の認証を設定し、Web 上で Competition Rules を承諾してください。提出は
quota を消費するため、`kaggle-lb submit` が明示的に実行されたときだけ行われます。

## リポジトリ構成

```text
agentic_kaggle/
├── AGENTS.md                       # エージェント向けの短い入口
├── .agents/                        # 開発フロー、ハーネス、状態管理
├── docs/
│   └── adr/                        # 継続的な設計判断の正本
├── templates/
│   └── competition/                # コンペ初期化テンプレート
├── competitions/
│   └── <competition-slug>/         # コンペ固有の知識・実装・評価
├── src/agentic_kaggle/             # プラットフォーム本体
└── tests/                           # プラットフォームの軽量テスト
```

コンペワークスペースの標準構成は次のとおりです。

```text
competitions/<slug>/
├── competition.toml                # 公式仕様と制約
├── strategy/                        # 現在方針、実験台帳、TODO
├── docs/{overview,discussion,kernel}/
├── configs/                         # 再現可能な実験設定
├── src/                             # 再利用するコンペ固有コード
├── tests/                           # コンペ固有テスト
├── evals/                           # 小さな評価 fixture と期待値
├── notebooks/                       # 探索専用
├── data/                            # Git 管理外
├── artifacts/                       # Git 管理外
└── submissions/                     # Git 管理外
```

## ADR ワークフロー

1. 既存 ADR と対象コンペの `competition.toml`、`strategy/` を確認する。
2. 複数の実装へ影響する方針、データ境界、評価契約を変える場合は ADR を `Proposed` で追加する。
3. 選択肢、決定、影響、検証方法を記載する。
4. 合意後に `Accepted` とし、小さな実装・テスト・評価 fixture へ落とす。
5. 方針変更時は過去 ADR を書き換えず、新しい ADR から `Superseded` にする。

詳細は [ADR index](docs/adr/README.md) と
[development workflow](.agents/development-workflow.md) を参照してください。

## 標準検証

```bash
uv run ruff check .
uv run harness_check
uv run pytest
```

コンペ固有の学習や推論は各コンペの `README.md` にコマンドを固定し、seed、fold、設定、
入力バージョン、コード commit、CV/LB 結果を `strategy/experiments.md` に残します。

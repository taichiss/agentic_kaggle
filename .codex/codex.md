## INSTRUCTION
- Python 環境は `uv` で構築・実行する。
- 返答は日本語で行う。
- 本プロジェクトは特定コンペ専用ではなく、複数の Kaggle コンペへ流用する前提の作業用ワークスペースである。
- コンペが切り替わったら、まず現在対象のコンペ名、評価指標、提出形式、制約、利用可能データを確認し、古いコンペ前提を引きずらない。
- 公式情報、配布データ、Discussion / Kernel 由来の知見、ローカル実験結果は分けて管理する。
- ルート `AGENTS.md` は短いポインタ文書として保ち、詳細ルールはこのファイルやスクリプト側に逃がす。

## ディレクトリ運用
- 競技データは `data/input/` に配置する。必要に応じて `data/input/<competition-slug>/` のようにコンペ別に整理する。
- 公式ルール、評価指標、提出仕様、データ説明の要約は `doc/overview/` に保存する。
- 現行コンペや過去コンペの Discussion 由来の知見は `doc/discussion/` に保存する。
- 現行コンペや過去コンペの Notebook / Kernel 由来の知見は `doc/kernel/` に保存する。
- 過去コンペの上位解法まとめは `doc/solution/` に年別で保存する。
- その時点の実行戦略、仮説、優先順位、実験計画、判断履歴は `.codex/strategy/` に保存・更新する。
- ルート `strategy/` は互換用コピーとして扱い、戦略文書の正本は `.codex/strategy/` とする。

## `doc/` の現状構成
- `doc/overview/`: 現時点では `.gitkeep` のみ。今後、公式情報の要約を置く場所。
- `doc/discussion/`: `2025/`, `2026/` ディレクトリはあるが、現時点では実データ未格納で `.gitkeep` のみ。
- `doc/kernel/2026/`: BirdCLEF 2026 向けの参考 Notebook がある。`perch-v2-starter-train-infer.ipynb` や `pantanal-distill-birdclef2026-improvement*.ipynb` をここで管理する。
- `doc/solution/`: 過去コンペの上位 solution を年別 (`2023/`, `2024/`, `2025/`) に整理している。

## `doc/solution/` の読み方
- 各年ディレクトリには `summary.md` があり、その年の上位解法の横断要約を読む入口として使う。
- 各年ディレクトリには `*.csv` もあり、上位解法の比較表・要点整理を表形式で確認できる。
- `1th` から `10th` のような順位名ファイルは、各順位の solution writeup の生データ置き場として使う。
- 順位ファイルは拡張子なしだが、実体は UTF-8 のテキスト / Markdown ライクな内容で、`sed`, `less`, `rg` で直接読める。
- 年によっては未収集順位があり、`2024` は `5th`, `7th`, `10th` がなく、`2025` は `3th`, `8th` がない。欠番は未整理データとして扱う。
- `2025/image.png` は 2025 solution 関連の補助画像。

## `doc/` の確認方法
- 全体構成の確認: `find doc -maxdepth 3 -type d | sort`
- 全ファイル一覧の確認: `find doc -maxdepth 3 -type f | sort`
- `solution` 年別の格納物確認: `find doc/solution -maxdepth 2 -mindepth 2 -type f | sort`
- 要約の確認: `sed -n '1,120p' doc/solution/2025/summary.md`
- 順位別 raw データの確認: `sed -n '1,120p' doc/solution/2025/1th`
- ファイル種別の確認: `file doc/solution/2023/1th doc/solution/2024/1th doc/solution/2025/1th`

## `doc/solution/` の運用ルール
- まず `summary.md` と `*.csv` を見て年ごとの全体傾向を掴み、その後に順位別 raw データを読む。
- 順位別 raw データから再利用する知見は、そのまま使わず `.codex/strategy/` や別メモに要約して転記する。
- 新しい順位データを追加する場合も、`summary.md` と `*.csv` の更新有無を合わせて確認する。

## 品質ハーネス
- 最速フィードバックは `PostToolUse` フックで取り、`bash .codex/hooks/post-tool-use.sh` から `uv run python scripts/quality_gate.py format` を呼ぶ。
- 速いフィードバックは pre-commit で取り、`.pre-commit-config.yaml` から `lint`, `typecheck`, `arch` を実行する。
- 遅いフィードバックは CI で取り、`.github/workflows/ci.yml` から `uv run python scripts/quality_gate.py all` を実行する。
- 人間レビューはフォーマットや単純違反の発見ではなく、仮説、設計、リーク、CV 設計、再現性の確認に集中させる。

## アーキテクチャガード
- Python コードは原則 `src/` 配下に置き、レイヤーは `core`, `providers`, `data`, `features`, `models`, `training`, `inference` を使う。
- 外部I/O、Kaggle API、ロギング、テレメトリ、feature flag のような横断的関心事は `providers/` 経由で注入する。
- レイヤー依存の検証は `scripts/archgate.py` を正本とし、import 方向を機械的にチェックする。
- ルールを変える場合は口頭で済ませず、`scripts/archgate.py` と `AGENTS.md` / `.codex/codex.md` を同時に更新する。

## AGENTS / 計画運用
- ルート `AGENTS.md` は 50 行以下を目安に、最低限のコマンド、禁止事項、参照先だけを書く。
- 現状説明、技術スタック解説、冗長なスタイルガイドは `AGENTS.md` に書かない。コード、設定、テスト、リンターを真実のソースとする。
- 実装前に短い計画を先に確定し、1回の変更では 1 機能または 1 論点だけを進める。
- 完了を宣言する前に、`format`, `lint`, `typecheck`, `arch`, `test` のどこまで実行したかを明示する。

## 知識蓄積ルール
- Discussion / Kernel の内容は丸写しせず、再利用しやすい要約に落とし込む。
- 各メモには最低限 `competition`, `source_url`, `checked_at`, `summary`, `actionable_points` を残す。
- 過去コンペの知見を使う場合は、現行コンペへ転用できる理由と差分も明記する。
- 重要な制約事項は必ず残す。例: 外部データ可否、事前学習モデル制限、推論時間制限、ライセンス、CV の注意点。
- 複数ソースが食い違う場合は、共通点、相違点、信頼度を明記する。

## 戦略運用ルール
- 新しい公式情報、EDA の発見、Discussion、Kernel、実験結果を得たら、会話だけで終わらせず `.codex/strategy/` を更新する。
- `.codex/strategy/` には少なくとも以下を維持する。
- `.codex/strategy/current.md`: 現在の勝ち筋、主要仮説、優先順位、評価方針、直近アクション。
- `.codex/strategy/experiments.md`: 実験ログ、設定、結果、解釈、次に試すこと。
- `.codex/strategy/todo.md`: 未着手タスク、保留事項、調査待ち事項。
- 戦略文書では、確認済みの事実と未検証の仮説を混同しない。
- 優先順位は `official information > raw data inspection > reproducible local result > third-party discussion/kernel` の順で判断する。

## 標準ワークフロー
1. 現在対象のコンペ名、目的、評価指標、提出形式、制約を確定する。
2. `data/input/` の中身を確認し、未展開ファイルや不足データを整理する。
3. 公式情報を `doc/overview/` に要約する。
4. 関連する現行 / 過去コンペの Discussion / Kernel を `doc/discussion/` と `doc/kernel/` に蓄積する。
5. それらをもとに `.codex/strategy/current.md` を更新し、現行コンペで採る方針を明文化する。
6. EDA、特徴量設計、CV、学習、推論、提出を戦略に沿って進める。本質的でないので安易にモデルアンサンブルを行わない。
7. 新しい結果が出るたびに `.codex/strategy/experiments.md` と `.codex/strategy/todo.md` を更新する。

## 実験・実装ルール
- まず小さく再現可能なベースラインを作る。
- 実験ごとに seed、fold、特徴量、モデル、前処理、後処理、推論設定、スコアを残す。
- CV と LB の乖離が大きい場合は、リーク、分布差、評価手順のズレを優先的に疑う。
- 大きな方針変更をする場合は、コードだけ先に変えず `.codex/strategy/` の記述も更新する。
- 長時間学習や重い前処理の前に、入出力パス、空き容量、必要計算資源を確認する。

## GPU 実行環境の再発防止ルール
- GPU 実行前に必ず同じシェル / 同じ環境で `nvidia-smi` を確認する。
- GPU 実行前に必ず同じシェル / 同じ環境で `uv run python` から `torch.cuda.is_available()` を確認する。
- `torch.cuda.is_available()` が `False` の場合は実行環境ミスマッチとみなし、学習や重い推論を開始しない。
- `/dev/nvidia*`, `/proc/driver/nvidia/version`, `/dev/dxg` の有無も必要に応じて確認する。
- `nvidia-smi` は動くのに `torch` が GPU を認識しない場合は、WSL や仮想環境の不一致を疑い、ユーザーの実行環境を優先する。
- ユーザーが「自分で実行する」と明言した場合、こちらは実行せず、コマンド提示と確認事項の整理に徹する。
- 事前確認が通るまで `nohup` などでのバックグラウンド実行は行わない。

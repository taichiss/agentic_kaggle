## INSTRUCTION
- Python 環境は `uv` で構築・実行する。
- 返答は日本語で行う。
- 本プロジェクトは特定コンペ専用ではなく、複数の Kaggle コンペへ流用する前提の作業用ワークスペースである。
- コンペが切り替わったら、まず現在対象のコンペ名、評価指標、提出形式、制約、利用可能データを確認し、古いコンペ前提を引きずらない。
- 公式情報、配布データ、Discussion / Kernel 由来の知見、ローカル実験結果は分けて管理する。

## ディレクトリ運用
- 競技データは `data/input/` に配置する。必要に応じて `data/input/<competition-slug>/` のようにコンペ別に整理する。
- 公式ルール、評価指標、提出仕様、データ説明の要約は `doc/overview/` に保存する。
- 現行コンペや過去コンペの Discussion 由来の知見は `doc/discussion/` に保存する。
- 現行コンペや過去コンペの Notebook / Kernel 由来の知見は `doc/kernel/` に保存する。
- その時点の実行戦略、仮説、優先順位、実験計画、判断履歴は `strategy/` に保存・更新する。
- `.codex/strategy/` は過去の名残として扱い、新規の戦略文書は `strategy/` を正本とする。

## 知識蓄積ルール
- Discussion / Kernel の内容は丸写しせず、再利用しやすい要約に落とし込む。
- 各メモには最低限 `competition`, `source_url`, `checked_at`, `summary`, `actionable_points` を残す。
- 過去コンペの知見を使う場合は、現行コンペへ転用できる理由と差分も明記する。
- 重要な制約事項は必ず残す。例: 外部データ可否、事前学習モデル制限、推論時間制限、ライセンス、CV の注意点。
- 複数ソースが食い違う場合は、共通点、相違点、信頼度を明記する。

## 戦略運用ルール
- 新しい公式情報、EDA の発見、Discussion、Kernel、実験結果を得たら、会話だけで終わらせず `strategy/` を更新する。
- `strategy/` には少なくとも以下を維持する。
- `strategy/current.md`: 現在の勝ち筋、主要仮説、優先順位、評価方針、直近アクション。
- `strategy/experiments.md`: 実験ログ、設定、結果、解釈、次に試すこと。
- `strategy/todo.md`: 未着手タスク、保留事項、調査待ち事項。
- 戦略文書では、確認済みの事実と未検証の仮説を混同しない。
- 優先順位は `official information > raw data inspection > reproducible local result > third-party discussion/kernel` の順で判断する。

## 標準ワークフロー
1. 現在対象のコンペ名、目的、評価指標、提出形式、制約を確定する。
2. `data/input/` の中身を確認し、未展開ファイルや不足データを整理する。
3. 公式情報を `doc/overview/` に要約する。
4. 関連する現行 / 過去コンペの Discussion / Kernel を `doc/discussion/` と `doc/kernel/` に蓄積する。
5. それらをもとに `strategy/current.md` を更新し、現行コンペで採る方針を明文化する。
6. EDA、特徴量設計、CV、学習、推論、提出を戦略に沿って進める。
7. 新しい結果が出るたびに `strategy/experiments.md` と `strategy/todo.md` を更新する。

## 実験・実装ルール
- まず小さく再現可能なベースラインを作る。
- 実験ごとに seed、fold、特徴量、モデル、前処理、後処理、推論設定、スコアを残す。
- CV と LB の乖離が大きい場合は、リーク、分布差、評価手順のズレを優先的に疑う。
- 大きな方針変更をする場合は、コードだけ先に変えず `strategy/` の記述も更新する。
- 長時間学習や重い前処理の前に、入出力パス、空き容量、必要計算資源を確認する。

## GPU 実行環境の再発防止ルール
- GPU 実行前に必ず同じシェル / 同じ環境で `nvidia-smi` を確認する。
- GPU 実行前に必ず同じシェル / 同じ環境で `uv run python` から `torch.cuda.is_available()` を確認する。
- `torch.cuda.is_available()` が `False` の場合は実行環境ミスマッチとみなし、学習や重い推論を開始しない。
- `/dev/nvidia*`, `/proc/driver/nvidia/version`, `/dev/dxg` の有無も必要に応じて確認する。
- `nvidia-smi` は動くのに `torch` が GPU を認識しない場合は、WSL や仮想環境の不一致を疑い、ユーザーの実行環境を優先する。
- ユーザーが「自分で実行する」と明言した場合、こちらは実行せず、コマンド提示と確認事項の整理に徹する。
- 事前確認が通るまで `nohup` などでのバックグラウンド実行は行わない。

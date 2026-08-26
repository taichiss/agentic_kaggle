# Agent Harness

エージェントが自分の変更を決定的に検証するための共通ハーネスです。

`harness_check` は次を検査します。

- 必須の入口文書、ADR、テンプレート、CI が存在する。
- ADR の番号、状態、必須セクション、index 登録が正しい。
- `competitions/<slug>/competition.toml` とディレクトリ名が一致する。
- コンペワークスペースに戦略、知識、コード、評価の境界がある。
- `.gitignore` にデータ、artifact、submission、secret の防止規則がある。

文章で守り続ける必要がある規則は、可能な限りこのチェックかテストへ移します。
コンペの精度やモデル品質そのものは一律に判定せず、各コンペの `evals/` と README に
評価コマンドを定義します。

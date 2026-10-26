# Kaggle Strategy Flow v1 (Week 4 構成)

**版:** v1
**初版:** 2026-10-22
**由来:** bio-hub (Biohub Cell Tracking) の反省からの Lessons Learned
**位置づけ:** 複数コンペ横断で守るべき 1 本の戦略フロー。コンペ別の `strategy/current.md` はこのフローへのポインタと差分だけを書く。

---

## 0. この文書の使い方

- **コンペ開始時**: Week 0 のチェックリストを上から全て潰す。1 項目でも skip したら以降の Week に進まない。
- **実験 1 件ごと**: 「採用判定の制約 (§5)」を全部満たしているかを確認してから registry.csv に commit する。
- **公開 kernel にスイッチした瞬間**: §6 のチェックポイントを強制実行する。これを飛ばすのが bio-hub 最大の事故原因だった。
- **コンペ終了後**: §8 の振り返りテンプレを埋め、新しい Lessons Learned を次版 (`kaggle_flow_v2.md`) に反映する。
- **一次情報源**: 本ファイル。`CLAUDE.md` と `.codex/strategy/` はここを参照する側であり、内容を複製しない。

---

## 1. 中核原則 (bio-hub Lessons Learned の結晶)

### L1. 評価指標との整合を実測した根拠は、後からの直観で覆さない
bio-hub では S1 Metric Sandbox で「division FP 1 件 = edge FP 2.7 本相当」と自分で実測していたのに、ILP で `division_weight=0.05` と**分裂に消極的な設定を選んだ**。実測した感度を無視した設計判断は、以降のすべての実験を無意味にする。

**制約:**
- Week 0 で指標因数分解を**必ず実測** (架空の理論値ではなく自前データで)。
- 以降のモデル/損失/後処理の各ハイパーパラメータは、「指標感度に対して整合しているか」の 1 行コメントを `registry.csv` の `notes` に書く。
- 実測根拠を覆す変更をする場合、`notes` に **「指標感度 X の実測を覆す。覆す理由は Y」** を明記する。書けないならその変更はしない。

### L2. Validation は「作っただけ」ではなく「採用判定経路に組み込まれているか」で測る
bio-hub では LOEO + metric freeze まで作ったのに、公開 kernel 流用後に **「公開 kernel 出力を fold に落とすアダプタ」を書かず**、47 実験中 42 件が LB 一発勝負に退化した。

**制約:**
- `registry.csv` の `local_score` 列を**必須入力**にする。空欄での commit を `scripts/register_experiment.py` で reject する。
- 公開 kernel / 他人の重みを使う場合も、その出力を自前 CV fold で再評価するアダプタを**最初に書く**。
- LB は「CV と乖離した時だけ疑うための検証」として温存する。LB で順位を決めない。

### L3. 「公開 kernel の後処理をいじる」モードに降りたら負け
bio-hub では EXP-0035〜0048 の 14 件連続 PP sweep、EXP-0083a〜0083c の TTA 乗合いで **LB 0.001 を追う** 作業に時間を溶かし、自前モデルは v12 (LB 0.835) で停止した。1 位との差を縮めるには「公開 kernel にないモジュールを自分で設計する」経路しかない。

**制約:**
- PP sweep に入る前に **誤差分解 (§4 Week 1)** を必ず通し、「どこを触れば指標が動くか」を仮説ドリブンで特定する。
- 「公開 kernel の後処理ハイパーパラメータだけを変える」実験は 1 コンペあたり 5 件まで。それ以上は自前モジュール設計に時間を振る。
- 自分の価値を**モジュールとして注入する**設計を Week 3 で必ず試す (bio-hub なら Soon net 相当)。

### L4. 公開 kernel にスイッチした瞬間を独立チェックポイントとして扱う
bio-hub では CAL-public-base (LB 0.939) で公開 kernel 採用の意思決定をした瞬間、S0〜S3 の validation 設計を再構築する責務が発生していたが、気づかずに PP sweep に突入した。

**制約:** §6 の 3 項目チェックポイントを強制実行する。

### L5. 実験再現性は "コード消失" でゼロになる
bio-hub の EXP-0066 "best; コード消失" に象徴されるとおり、`notes` だけでは再現できない。

**制約:** 全提出に `git rev-parse HEAD`、`kernel-metadata.json` snapshot、環境ハッシュを紐付ける。§5 参照。

---

## 2. ディレクトリ規約 (コンペ共通)

```
<comp-root>/              # 例: bio-hub/, birdclef-2026/
  overview.md             # 公式ルール・指標・提出形式・制約の 1 枚要約
  matome.md               # コンペ終了後の振り返りまとめ (§8)
  frozen/                 # metric_hash.txt, cv_manifest.json (改変検知)

experiments/
  registry.csv            # 全実験の 1 行記録。local_score 必須
  EXP-XXXX/               # 各実験の config, output snapshot, 誤差分解結果

strategy/
  current.md              # ← このファイルへのポインタ + コンペ固有差分だけ
  experiments.md          # 代表実験の解釈ログ (registry.csv の深堀り)
  todo.md                 # 未着手・保留・調査待ち

doc/strategy/
  kaggle_flow_v1.md       # ← 本ファイル (ワークスペース全体で 1 本)
```

**コンペ固有の `strategy/current.md` には、「このコンペは kaggle_flow_v1.md の Week X にいる」「§Y の制約に対する差分は Z」だけ書く。** フロー自体を書き直さない。

---

## 3. Week 0 — 開始チェックリスト (全項目必須)

着手前に全部潰す。1 項目でも skip したら Week 1 に進まない。

- [ ] **コンペ基本情報**: 名前、slug、deadline、評価指標、提出形式、制約 (CPU/GPU 時間、外部データ可否、事前学習重み可否、ライセンス) を `<comp-root>/overview.md` に 1 枚で書く
- [ ] **データ全量確認**: `data/input/<slug>/` の中身を `du -sh`, `find -type f | head`, zarr/HDF5 なら shape まで確認。未展開や欠落をゼロにする
- [ ] **指標の因数分解実装**: `src/<pkg>/metrics/breakdown.py` に **指標の構成要素ごとの計算** を実装 (bio-hub なら edge TP/FP/FN と division TP/FP/FN を分けて出す)
- [ ] **Metric Sandbox 実測**: 合成データ or GT で「要素 1 件の変化が総合スコアに与える感度」を**自分で測る** (bio-hub S1 相当)。結果を `<comp-root>/metric_sensitivity.md` に残す
- [ ] **指標ハッシュ freeze**: 指標実装の SHA256 を `<comp-root>/frozen/metric_hash.txt` に保存。CI で改変検知
- [ ] **CV 設計の確定**: データリーク (時間、サイト、個体、グループ) を列挙し、それを閉じる split を設計する。split manifest を `<comp-root>/frozen/cv_manifest.json` に freeze
- [ ] **過去コンペ横断分析**: `doc/solution/` の該当年 summary + 1-10 位の writeup から勝因要素を表に落とす (bio-hub では未実施 → 1 位の Soon net 相当モジュールを事前に想像できなかった)
- [ ] **採用規律の明文化**: 「両フォールドで勝ったものだけ採用」など、このコンペの採用規律を `strategy/current.md` に書く

---

## 4. Week 1 — 誤差分解ゲートを先に作る (= 公開 kernel 推論より先)

bio-hub の最大の順序ミス: 公開 kernel 推論 (S4) を先にやって、誤差分解 (S5) が未着手のままだった。**順序を反転させる**。

- [ ] **誤差分解ツール**: `src/<pkg>/analysis/error_decomp.py` に「GT のある正解イベントを、モデルのどの段でロスしたか」を分類するスクリプトを書く
  - bio-hub 例: GT division に対して「検出が出ているか / linker 候補に入っているか / ILP で選ばれているか」の 3 段で分類
  - BirdCLEF 例: GT positive window に対して「backbone embedding で分離しているか / head で確率が出ているか / 後処理で潰されたか」
- [ ] **最弱ベースライン提出**: 一番シンプルな提出を 1 本作って LB に出す。目的は提出パスの検証だけで、スコアは問わない
- [ ] **誤差分解でボトルネック特定**: 最弱ベースラインの誤差分解結果から、「どの段を改善すれば指標がどれだけ動くか」の見積もりを `strategy/current.md` に書く
- [ ] **Week 1 Exit 条件**: ボトルネック順位が特定できていること。できていないうちは Week 2 に進まない

---

## 5. Week 2 — Validation アダプタ構築 (全実験の必須経路化)

- [ ] **CV アダプタ**: 任意の推論出力 (submission.csv, geff, parquet など) を CV fold に当てはめて自動スコアリングする CLI を `src/<pkg>/cli/eval_oof.py` に実装
- [ ] **公開 kernel 互換**: 他人の kernel 出力でも同じアダプタで評価できることを確認 (出力フォーマットの変換層を書く)
- [ ] **registry.csv の必須入力ガード**: `scripts/register_experiment.py` を書き、`local_score` が空 or `git rev-parse HEAD` が未記録なら commit を reject
- [ ] **提出メタデータ snapshot**: 全提出で `kernel-metadata.json` + `pip freeze` + `git rev-parse HEAD` を `experiments/EXP-XXXX/env.json` に保存
- [ ] **Week 2 Exit 条件**: 「LB に出す前に必ず CV アダプタを通す」が運用として回っていること。LB 直叩き実験がゼロ件であること

---

## 6. 公開 kernel 採用チェックポイント (降ってきた時に実行)

公開 kernel のスコアに乗ろうと判断した瞬間、以下 3 項目を**強制実行**する。これを飛ばすと bio-hub と同じ道を辿る。

- [ ] **CP1. 公開 kernel の出力を自前 CV fold に落として LOEO/GroupKFold スコアを再計算**する。LB スコアを鵜呑みにしない
- [ ] **CP2. 公開 kernel を誤差分解ツール (§4) に通し**、「どの段にまだ改善余地があるか」を特定する
- [ ] **CP3. 改修先モジュールを決める**。決まるまで PP sweep を禁止する。公開 kernel のハイパーパラメータいじりに降りない

---

## 7. Week 3 — 自分の価値をモジュールとして注入する (Week 4 は提出固め)

### Week 3: 独自モジュール設計
- [ ] 誤差分解で特定したボトルネック段に、公開 kernel にはない**専用モジュール**を 1 つ設計する
  - bio-hub 1 位の Soon net (分裂専用判定モデル) がこれに該当
  - 公開 kernel の出力の上に ranking / re-scoring / veto として乗せる形が実装コストが低い
- [ ] 独自モジュールの学習データは、公開 kernel の出力 + GT を**少数クラス oversample** して組む (bio-hub なら division 8×)
- [ ] CV アダプタ (§5) で独自モジュール有無を比較し、ボトルネック段のスコア変化を測る
- [ ] 既存の ILP / 後処理ハイパーパラメータは、評価指標から**逆算**して grid 掃引する (bio-hub なら `division_weight` と `disappearance_weight`)

### Week 4: 提出固め
- [ ] CPU/GPU 時間制約の実測。余裕 20% を確保
- [ ] 推論経路の冗長化排除 (キャッシュ共有、重複計算の統合)
- [ ] 最終 2 提出の選定: **CV Best** と **Public Best** (CV Best が優先)
- [ ] 両提出の `kernel-metadata.json` + `git rev-parse HEAD` を `<comp-root>/final_submissions.md` に記録
- [ ] コンペ終了後、§8 の振り返りを埋める

---

## 8. 振り返りテンプレート (コンペ終了後)

`<comp-root>/matome.md` に以下を埋める。bio-hub の `matome.md` が v1 のサンプル。

```markdown
# <comp-name> 振り返りまとめ

## 1. アプローチと上位解法の決定的な違い
- 共通の土台 (使った重み・公開 kernel)
- 技術差分 3 軸 (このコンペのパイプライン段で切る)
- 上位の段差ごとの効き目要素

## 2. Validation 設計の振り返り
- 機能していた部分
- 機能しなくなった時期と原因
- 教訓 (次回に残すべきルール)

## 3. 中核原則 (§1) への準拠と逸脱
- L1〜L5 のうちどれを守り、どれを破ったか
- 破った理由と、それが結果にどう効いたか

## 4. 次版 (kaggle_flow_vN+1.md) に追加すべき Lessons
- 本コンペで新しく見つけた制約・原則
```

---

## 9. 更新運用

- 本ファイルの修正は **コンペ振り返り (§8) を踏まえて** のみ行う。コンペ最中には変えない (ブレを防ぐため)
- 大きな追加・変更は `kaggle_flow_v2.md`, `v3.md` と**版を切る**。過去版は履歴として残す
- 版を切ったら `CLAUDE.md` の参照先ポインタを更新する

---

## 10. 付録: bio-hub Lessons Learned 原文へのリンク

- 全体まとめ: [bio-hub/matome.md](../../bio-hub/matome.md)
- v1〜v12 開発レポート: [doc/report_v1_v12/report.md](../../doc/report_v1_v12/report.md)
- 実験レジストリ: [experiments/registry.csv](../../experiments/registry.csv)
- 旧戦略 (bio-hub 当時の): [strategy/current.md](../../strategy/current.md), [strategy/experiments.md](../../strategy/experiments.md), [strategy/todo.md](../../strategy/todo.md)

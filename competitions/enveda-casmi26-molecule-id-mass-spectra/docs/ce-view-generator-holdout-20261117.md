# CE ビュー別の候補生成: Enveda 分子分離検証

2026-11-17。0.477 の凍結した生成エンジンについて、クエリの衝突エネルギー (CE) の扱いだけを変え、候補集合に正解構造が入るかを比較した。これは候補生成の診断であり、親ノートブックの MRR、順位、Public LB を測ったものではない。

## 対象と漏洩防止

- 既存の分子単位で分離された validation split の `enveda-180`、timsTOF から、単一 20/40/60 eV と、その複合測定を同じ分子・アダクトで持つ群を抽出した。対象は 12,053 群、10,778 分子。seed `20261117` で 200 **異なる分子**を選び、各分子の 4 スペクトルをクエリにした。
- 正解構造キーに対応する参照ライブラリの **全 SID と全スペクトル**を検索対象から外し、候補プールの正解 PID も除いた。正解が含まれる場合は生成器が再生成した候補によるものになる。
- 同じ凍結候補プール、分子質量、生成規則、上限 6 親を使用した。生成だけの高速経路は、最初の 2 群で 0.477 `Engine.run` の生成候補キーと完全一致した。G4 も別途最初の 2 群で親エンジンと一致した。

## 結果

| 条件 | 生成候補に正解 | 全候補に正解 | 平均生成候補数 | G0 から候補集合が変化 |
| --- | ---: | ---: | ---: | ---: |
| G0: 従来の統合スペクトル | 8/200 | 8/200 | 5.555 | 対照 |
| G2: 実測 CE を個別検索 | 7/200 | 7/200 | 6.040 | 69/200 |
| G3: G2 + 同一分子の穏やかな CE Mixup | 7/200 | 7/200 | 6.200 | G2 比 11/200 |
| G4: 統合スペクトルの最上位親を保持し、残りを CE 別に選択 | 8/200 | 8/200 | 5.875 | 77/200 |

| 比較 | 正解を救済 | 正解を喪失 | 候補集合が変化 | 生成候補の追加 / 除去 (群をまたぐ出現数) |
| --- | ---: | ---: | ---: | ---: |
| G0 → G2 | 1 | 2 | 69/200 | +583 / −486 |
| G2 → G3 | 0 | 0 | 11/200 | +46 / −14 |
| G0 → G4 | 3 | 3 | 77/200 | +563 / −499 |
| G2 → G4 | 3 | 2 | 65/200 | +413 / −446 |
| G3 → G4 | 3 | 2 | 63/200 | +393 / −458 |

G0→G2 の正解判定が変わった分子キーは `YZOFEDHCGUJBGP` (喪失)、`FSKUOCYHKMUTRH` (救済)、`JRGITTKNEWLGFM` (喪失)。G2→G3 では正解判定が変わった分子はない。G4 は異なる生成親 SID を 200/200 群で選び、候補集合を 77/200 群で変えたが、G0 比の正解包含率は同じだった。

## テストセットへの適用可能性

ラベルを読まずに `test.parquet` の CE 配列長 1、20/40/60 eV、同一分子・アダクト・装置で数えると、400 分子・513 群のうち 279 分子・312 群に 2 種以上の単一 CE がある。3 種すべてを持つのは 118 分子・119 群。以前の external80 の 5/80 という適用数は、テスト全体の CE ビュー分布を代表しない。

この検証分子は Enveda のスクリーニング化合物であり、コンペのテスト分子とは化学空間が異なる。200 分子の生成候補包含だけでは順位改善や LB 改善を断定できない。候補集合変化の探索としては G4 が G3 より大きいが、正解包含の純増は確認されなかった。提出後には同じ候補群で MRR と LB の差を別途記録する。

## 再現手順と成果物

評価コードは [`evaluate_ce_generator_holdout.py`](../src/public_notebooks/evaluate_ce_generator_holdout.py) と [`evaluate_ce_diverse_holdout.py`](../src/public_notebooks/evaluate_ce_diverse_holdout.py)。それぞれ `--parent-crosscheck 2` で全エンジンとの候補キー一致を検査する。依存環境は以下の固定コマンドで実行した。

```bash
uv run --offline --with torch==2.10.0+cu129 --with numpy --with pandas --with pyarrow --with rdkit --with numba --index https://download.pytorch.org/whl/cu129 --index https://pypi.org/simple python src/public_notebooks/evaluate_ce_generator_holdout.py --limit 200 --parent-crosscheck 2
uv run --offline --with torch==2.10.0+cu129 --with numpy --with pandas --with pyarrow --with rdkit --with numba --index https://download.pytorch.org/whl/cu129 --index https://pypi.org/simple python src/public_notebooks/evaluate_ce_diverse_holdout.py --parent-crosscheck 2
```

- G0/G2/G3: `data/derived/ce-views-20261117/enveda-generator-holdout/report.json` (SHA256 `1b9ff598bbc8c53827588850614eb0803281bbe6172e4c5a2d0d15ccbbae66c6`)、`per-group.jsonl` (`1cca0fec5d262995f3fe5f5123caca965eb9344601e4dcd670b14652ae75db2a`)、`per-molecule-deltas.jsonl` (`8cf03b45a751afd811ec054beddf0f55e569c904986358af2d8ccc987ebb59e3`)
- G4: `data/derived/ce-views-20261117/enveda-ce-diverse-holdout/report.json` (`6a64343f0839ef746e12ef95ef44d9b4080853ecbcc7982a35711fb79fc99dbf`)、`per-group.jsonl` (`868987af2b23faf1036c71edf512ccf10114d276b0259186853cb3b2b7a823d3`)

各 `report.json` に入力データ、親エンジン、アダプター、プール、評価コードの SHA256 を記録した。`data/derived` の生成物はローカルの非追跡ファイルとして保持する。

# Enveda-180 の実測CE別スペクトル監査

2026-11-17。生成側の次サイクルに使う実測CE別の入力を、既存の分子非重複split上で監査した。元データを新たに分割せず、候補・正解ラベル・テストの構造は読まない。

## 入力と採用条件

- `data/derived/saner-multipositive/cache-parallel/{molecules,spectra}.parquet` は273,833構造、2,430,328スペクトル。分子の`key`は重複せず、各スペクトルの`split`は対応分子の`split`と一致する。
- `data/derived/mist-mixup-20261116/source-metadata/` のライブラリ・装置・CE sidecar、および `ce-pattern-signatures/` の原CE表記を、保存済みcacheと同じ行順で参照する。Enveda-180の1,153,785行は全件`timsTOF`。採用する既知アダクト行は797,404行。
- 一組のキーは既存split、同一分子`mol_idx`、同一アダクトID、同一極性。単一CEが正確に20、40、60 eVで各1本以上あり、元の複数CE表記が正確に`20,40,60`または`-60,-40,-20`の統合測定もある群だけを採用する。変換後の平均CEだけで統合測定を判定しない。
- アダクトID1は`[M+H]+`、ID2は`[M-H]-`（`src/mist_formula/model.py`の語彙）。採用群内の前駆体m/z幅は全件20 ppm未満で、別の前駆体を混ぜる群はない。

| 既存split | Enveda-180既知アダクト行 | 20/40/60 eV単独測定がそろう群 | 対応する元の3-CE統合測定もある群 | 一意の分子 | 正／負極性 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 716,818 | 107,879 | **107,858** | 96,150 | 88,432／19,426 |
| validation | 80,586 | 12,209 | **12,207** | 10,866 | 9,996／2,211 |

完全群の前駆体m/z幅の最大はtrainで7.78 ppm、validationで3.52 ppm。構造keyはtrainとvalidationを跨がない。trainの完全群は`[M+H]+`が87,254、`[M-H]-`が19,335、validationではそれぞれ9,854、2,199。残りは別の既知アダクトである。

## 実験への使い方と境界

trainの107,858群をCE別モデルの学習候補、validationの12,207群を同装置・未見分子の診断に使える。20/40/60 eVの単独測定を**同じ構造の異なる測定view**として扱う。元の3-CE統合測定は独立した取得だと確認できていない。validationの完全群から固定seedで抽出した200群では、0.01 Daに丸めた統合測定ピークが単独3本のいずれかにも存在する割合の中央値は100%、単独3本の単純平均強度とのcosine中央値は0.936だった。これは統合測定が単独測定から計算された可能性を示す診断であり、生成過程の証明ではない。**独立した正解スペクトルとして自己教師あり学習の評価には使わない。** まずCE別viewの選択・別入力・予測集約を比較し、その後に線形ピークMixupを評価する。

生成側の効果を測るときは、既存のルール型生成器とプールの候補を固定しない。正解構造が候補群に含まれる率、生成候補だけでの救済数、候補内の順位、最終MRR@25を分けて記録する。既存の0.479対照で使った**固定候補集合**の評価は順位付けの対照であり、候補生成の改善を検出できない。別ソースや未知アダクトへの一般化は、このEnveda-180内部のvalidationだけでは確認できない。

監査コードは`src/mist_formula/audit_ce_views.py`。再実行コマンドは競技DIRから：

```bash
uv run --offline --with pandas --with pyarrow --with numpy python src/mist_formula/audit_ce_views.py
uv run --offline --with pandas --with pyarrow --with numpy python src/mist_formula/diagnose_ce_composite.py
```

機械可読の集計結果はコミット対象外の`data/derived/ce-views-20261117/data-audit.json`（SHA-256 `000e65fea3feb620f29bafe87219f2c2ed289a758dbad6bb3dcf1bd533a5603d`）に保存した。入力cache、CE sidecar、pattern定義の完全なSHA-256は同JSONに記録する。両スクリプトのRuff check/formatは成功した。

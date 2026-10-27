# CASMI Spectrum Explorer

CASMI の Parquet をローカルで読み、MS/MS スペクトルと分子構造を探索するツールです。
ピークの値・測定メタデータを確認し、SMILES による構造描画と、Morgan fingerprint
（radius 2、2048 bit）の Tanimoto 類似度による類似構造の検索ができます。対応する
スペクトルがある比較候補は、鏡像プロットで並べられます。
比較側に複数の測定がある場合は、付加イオンや衝突エネルギーを確認して選べます。

## 起動

リポジトリルートから実行します。依存関係は `uv` が用意します。

```bash
uv run --extra eda python competitions/enveda-casmi26-molecule-id-mass-spectra/eda/server.py
```

表示される `http://127.0.0.1:8765` を開きます。既定の入力は
`competitions/enveda-casmi26-molecule-id-mass-spectra/data/input/test.parquet` です。
別のCASMI形式のParquetを表示するには `--data /path/to/train.parquet` を付けます。
指定ファイルがないときはエラーで停止します。合成デモは `--demo` で明示的に選べます。

大量データを誤ってすべてメモリに読み込まないよう、既定では**先頭5,000行**だけを
対象とします。`--max-rows 10000` で変更できます。検索と類似度比較は読み込んだ範囲
だけが対象であり、並びに偏りがある場合は結果にも偏りが生じます。
表示を軽くするため、各スペクトルのグラフと数値表は強度上位5,000ピークまでを
表示し、元の配列長・欠損・表示件数を画面に記載します。元のParquetは変更しません。

## データの解釈

- 公開テストは表示用サンプルで、スコアに使われる非公開データではありません。
  構造ラベルがないので、手入力SMILESは候補であり正解を意味しません。
- ラベル付き学習Parquetでは `normalized_smiles` と `inchikey14` を使用します。
  類似度は構造の近さであり、同定の確信度やスペクトル一致度ではありません。
- `data/input/np-examples-preview.parquet` がある環境では、構造ラベル付きの画面を
  `--data competitions/enveda-casmi26-molecule-id-mass-spectra/data/input/np-examples-preview.parquet`
  で開けます。これは第三者Hugging Faceミラーから取得した一部で、公式データとの
  一致は未検証です。出典と取得範囲は同じディレクトリの provenance JSON に記録しています。
- 実データや生成物は Git 管理しません。アプリは localhost で動き、Kaggle への提出は行いません。

## 動作確認

```bash
uv run --extra eda --extra dev pytest -q competitions/enveda-casmi26-molecule-id-mass-spectra/tests/test_eda.py
```

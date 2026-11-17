# CE適合Enveda 200分子のEngine1順位診断（2026-11-17）

## 評価の範囲

以前の候補生成診断と**同じ200分子**を、凍結した0.477親のEngine1、FPNet、V1FE特徴量、LightGBM順位モデル、pool人気度補正0.15まで通した。200分子は分子を跨がない既存validation splitの`enveda-180`から、同一分子・同一アダクトの実測20/40/60 eVをすべて持つ群をseed `20261117`で選んだ。**200/200群**が3つの単一CEに適合する。G0は従来の統合スペクトル、G4は統合スペクトルの最上位生成親を保持して残りの親をCE別に選択、G5はG0とG4の生成候補の和集合である。

各分子について、正解構造の参照SIDをすべて除外した（200 SID、合計1,558参照スペクトル）。正解のpool PIDも200/200群で除外し、正解がpoolまたは参照の自己一致で入らないようにした。各モードの候補生成段階での正解包含判定は、保存済み独立診断と**200/200群で一致**した。凍結Engine1の全候補をスコア順に並べ、重複キーを除いた上位60にpool人気度による再配置を適用し、その上位25で正解順位を数えた。

| 条件 | 生成候補に正解 | Engine1 Top25に正解 | Engine1 Top1 | Engine1 MRR@25 | G0との差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0・従来の統合スペクトル | 8/200 | 8/200 | 3/200 | 0.026667 | — |
| G4・CE別に生成親を多様化 | 8/200 | 8/200 | 5/200 | 0.032500 | +0.005833 |
| G5・元候補とCE別候補の和集合 | 11/200 | 11/200 | 6/200 | 0.041667 | +0.015000 |

G4は正解を3件追加した一方で3件失った。G5はその3件の追加を保ち、3件の喪失を回復した。G0→G5で正解順位が改善したのは3/200分子、悪化0、残り197は不変。改善3件のEngine1順位はいずれも圏外→1位だった。G4→G5ではG4が失った正解が1位、2位、3位に戻った。G0対G4の分子単位差は改善3・悪化3・不変194である。

分子を単位にした5,000回の対応ブートストラップ（seed `20261117`、2.5/97.5パーセンタイル）95%区間は、G4−G0が**[−0.014167, +0.027500]**、G5−G0が**[0, +0.035000]**。G5の差はわずか3分子に依存し、区間は0を含む。G5の設計はこのパネルの先行候補包含結果も参照して選んでいるため、ここでの差を独立検証の汎化性能とは扱わない。

## 解釈と限界

この順位は**Engine1だけ**の値であり、Engine2・PubChem・最終融合・固定MIST A再順位付け後の提出Top25ではない。親モデルの学習元に同じ構造が含まれたかは監査できておらず、参照SIDとpool PIDの除外だけでは「モデル未見分子CV」にならない。また、正解pool PIDを意図的に除くため絶対MRRは通常のコンペ課題より低く、Public LB 0.477等と直接比較できない。Enveda-180はスクリーニング化合物で、テスト分子の化学空間とも異なる。

この診断から言えるのは、同じCE適合200群の凍結Engine1で、G5の候補追加による3件の正解包含が**実際に上位25まで届いた**ことだけである。最終提出の良否は5本目の公開LBと、後段まで再現した別パネルの評価で確認する。

## 再現情報

評価コードは [`evaluate_ce_union_engine1_holdout.py`](../src/public_notebooks/evaluate_ce_union_engine1_holdout.py)。既存の6件診断器を200群へ拡張し、同じG4/G5記録の分子・入力スペクトル行との一致を検査した。実行時間はCUDAで529.5秒。入力のモデル・参照・pool・アダプターのSHA-256はローカルの`data/derived/ce-views-20261117/enveda200-engine1-rank/report.json`に記録した。分子別出力`per-group.jsonl`のSHA-256は`224f00658195fade00fbe1b37c57a4735279dd9be2cf6ef35085140883c0174a`。このローカル生成物はGit対象外である。以下の全評価コマンドには、凍結したKaggleモデル・参照・poolを`/kaggle/input`と`/kaggle/working`に配置し、導出済みG4/G5記録と親mountを用意した環境が必要である。

```bash
uv run --offline --with torch==2.10.0+cu129 --with numpy --with pandas --with pyarrow --with rdkit --with numba --with lightgbm --index https://download.pytorch.org/whl/cu129 --index https://pypi.org/simple python src/public_notebooks/evaluate_ce_union_engine1_holdout.py --device cuda
# 保存済みの分子別結果から集計と区間だけを再計算する場合
uv run --offline --with torch==2.10.0+cu129 --with numpy --with pandas --with pyarrow --with rdkit --with numba --with lightgbm --index https://download.pytorch.org/whl/cu129 --index https://pypi.org/simple python src/public_notebooks/evaluate_ce_union_engine1_holdout.py --postprocess-only
```

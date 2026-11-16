# 5本目のMixup提出条件の選択

初回M1・M2・M3のPublic LBが揃った後、競技DIRで次を実行する。

```bash
python3 src/mist_formula/select_mixup_followup.py
```

このスクリプトは`strategy/lb-submissions.jsonl`の各提出IDの**最後の記録**だけを読み、A (`58077626`)、M1 (`58435231`)、M2 (`58434412`)、M3 (`58435174`) の実験ID・`SubmissionStatus.COMPLETE`・表示Public LBを照合する。いずれかが未確定なら`{"state":"waiting"}`をJSONで出し、終了コード2で止まる。台帳やKaggleには書き込まない。

確定後は`{"state":"ready", "selected": ...}`を出す。M2またはM3の表示値がAとM1の両方を**厳密に上回る**場合は高い方を選び、同点ならM2を選ぶ。選択値は`A75+M225`または`A75+M325`。それ以外は`weak_M1`。3桁表示の同点を改善とはみなさない。選択後も、対応するprivate Notebookの遠隔監査とCSVの一致を確認してから競技へ提出する。規則を固定した元文書は`docs/mist-spectrum-mixup-results-20261116.md`。

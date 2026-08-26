# Kaggle Code Competition workflow

The final submission is created by a Kaggle Notebook with internet disabled and must write
`submission.csv`. Do not put downloaded `.ipynb` files, embedded weights, output CSVs, or Kaggle
credentials in this tracked directory.

## Prepare a private submission kernel

Create an ignored working directory under `data/submission-kernel/`, initialize metadata, add the
reviewed inference notebook and any attached public dataset/model references, then push explicitly:

```bash
COMP=competitions/biohub-cell-tracking-during-development
mkdir -p "$COMP/data/submission-kernel"
uv run kaggle kernels init -p "$COMP/data/submission-kernel"
# Edit kernel-metadata.json and the generated notebook/script.
uv run kaggle kernels push -p "$COMP/data/submission-kernel"
uv run kaggle kernels status <owner>/<kernel-slug>
uv run kaggle kernels output <owner>/<kernel-slug> -p "$COMP/artifacts/kernel-output"
uv run kaggle competitions submissions biohub-cell-tracking-during-development
```

Before pushing, confirm in `kernel-metadata.json` that the competition data source is attached,
internet is disabled, the accelerator matches the tested config, and the kernel is private unless
publication is intentional. Notebook push consumes compute and a competition submission consumes a
quota; neither is automated by `fetch_assets.py`.

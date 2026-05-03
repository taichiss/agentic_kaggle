# BirdCLEF 2026 Perch Probe CV Models

## Notebook compatibility
- `perch_meta.parquet` and `perch_arrays.npz` are saved at the dataset root.
- These root files contain the notebook-compatible `full59` cache.
- CV model bundles live under `cv_models/<pattern>/`.
- `train_audio_mlp_head_bundle.joblib` is saved at the dataset root for ver8 submit.
- `train_audio_mlp_head_meta.json` stores its training metadata.

## Patterns
- `main_all66_sitebalanced3`: oof_macro_auc=0.882219, splitter=site_balanced_file, n_splits=3
- `secondary_all66_filegkf5`: oof_macro_auc=0.891374, splitter=file_group_kfold, n_splits=5
- `stress_all66_siteholdout3`: oof_macro_auc=0.857185, splitter=site_holdout, n_splits=3

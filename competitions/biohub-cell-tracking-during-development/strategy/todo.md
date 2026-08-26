# Todo

## Bootstrap

- [x] Authenticate Kaggle CLI without placing credentials in the repository.
- [x] Accept the live competition Rules.
- [x] Download and inventory official data.
- [x] Pull the versioned public notebook references.
- [x] Record the organizer baseline commit revision.

## Baseline

- [x] Run graph I/O and metric smoke checks on one training dataset.
- [x] Adapt the organizer detector/linker for bounded local smoke runs.
- [x] Generate and validate a complete sample-based test submission locally.
- [x] Package the pipeline for an offline Kaggle Notebook.
- [x] Push the notebook with explicit user approval and log the LB result.
- [x] Submit the epoch-5 organizer checkpoint as the first model-based LB anchor.
- [x] Compare epoch-19 raw and artifact-free post-processed submissions: 0.805 to 0.869 public.
- [x] Complete the 50-epoch training series with checkpoints every five completed epochs.
- [ ] Calibrate detection counts and edge thresholds from the 0.869 model-based anchor.

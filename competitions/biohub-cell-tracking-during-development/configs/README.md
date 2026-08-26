# Experiment Configs

`baseline.toml` records the verified geometry and evaluation boundary while leaving algorithm
choices explicitly `unimplemented`. `exp-0001-host-smoke.toml` records the deliberately tiny
organizer-baseline contract test; its thresholds and model size are not quality candidates.
`exp-0002-wandb-extended.toml` extends the same contract to five epochs and records epoch trends in
W&B without uploading competition data, predictions, or checkpoints.

Each new experiment must copy the config to a new file and record its experiment ID in
`strategy/experiments.md`. Keep secrets, absolute machine paths, downloaded data, and artifacts out
of configs.

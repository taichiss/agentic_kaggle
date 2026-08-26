# Experiment Configs

`baseline.toml` records the verified geometry and evaluation boundary while leaving algorithm
choices explicitly `unimplemented`. `exp-0001-host-smoke.toml` records the deliberately tiny
organizer-baseline contract test; its thresholds and model size are not quality candidates.

Each new experiment must copy the config to a new file and record its experiment ID in
`strategy/experiments.md`. Keep secrets, absolute machine paths, downloaded data, and artifacts out
of configs.

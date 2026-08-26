# Experiment Configs

`baseline.toml` records the verified geometry and evaluation boundary while leaving algorithm
choices explicitly `unimplemented`. Do not turn an untested community method into a confirmed
baseline merely by changing the label.

Each new experiment must copy the config to a new file and record its experiment ID in
`strategy/experiments.md`. Keep secrets, absolute machine paths, downloaded data, and artifacts out
of configs.

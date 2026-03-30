# Splat-Nav Wrapper Workspace

This workspace keeps the upstream Splat-Nav repository intact and adds an outer engineering layer for reproducibility, logging, and onboarding.
The `scripts/` folder is wrapper automation; core paper algorithms stay in the upstream source tree.

## Scope

1. Reproduce Splat-Nav on this machine (offline/simulation only).
2. Keep all heavy artifacts on `/data/howard`.
3. Do not touch other user environments and do not upgrade CUDA/drivers.

## Key Paths

1. Wrapper workspace: `/home/howard/Splat-Nav`
2. Upstream repo (preserved): `/data/howard/repos/splatnav-official`
3. Upstream symlink (recommended entry): `/home/howard/Splat-Nav/splatnav-official`
4. Upstream symlink (same target): `/home/howard/Splat-Nav/third_party/splatnav-official`
5. Runs output: `/home/howard/Splat-Nav/runs` -> `/data/howard/splatnav/runs`
6. Docs: `/home/howard/Splat-Nav/docs`

## Scripts

1. `scripts/create_run_dir.sh <topic>`
2. `scripts/run_in_tmux.sh <window_name> <topic> "<command>"`
3. `scripts/bootstrap_env.sh [env_name]`
4. `scripts/env_health_check.sh [env_name]`

## Make Targets

```bash
make env-bootstrap
make env-check
make data-download
make data-link
make smoke-plan SCENE=old_union CONFIG=/path/to/config.yml METHOD=sfc-1
```

## Typical Workflow

```bash
# 1) Bootstrap environment in tmux
bash scripts/run_in_tmux.sh env_bootstrap bootstrap_env "bash scripts/bootstrap_env.sh splatnav"

# 2) Health check
bash scripts/run_in_tmux.sh env_check env_health "bash scripts/env_health_check.sh splatnav"
```

## Result Convention

All run logs and artifacts go to:

```text
runs/YYYY-MM-DD_HHMM_<topic>/
```

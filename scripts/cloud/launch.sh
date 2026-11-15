#!/usr/bin/env bash
# Start one detached training run per seed.
# Usage: launch.sh --config configs/terrain_v2.toml --seeds 0,1,2 --n-envs 8 [--nohup|--tmux] [--resume]
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python scripts/launch_seeds.py launch "$@"

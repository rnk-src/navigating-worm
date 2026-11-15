#!/usr/bin/env bash
# Prepare a fresh Ubuntu machine for training. Run it inside a checkout of this repo.
# It installs the system packages and uv, syncs the locked environment, runs the checks
# and prints the core count and the env throughput. It is safe to run twice.
# Usage: bootstrap.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
run() {
  echo "+ $*"
  [ "$DRY" = 1 ] || "$@"
}

SUDO=""
[ "$(id -u)" = 0 ] || SUDO="sudo"
if command -v apt-get > /dev/null; then
  run $SUDO apt-get update
  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git build-essential libgl1 libegl1 libosmesa6 tmux rsync curl
else
  echo "no apt-get on this system: skip the system packages"
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv > /dev/null; then
  echo "+ curl -LsSf https://astral.sh/uv/install.sh | sh"
  [ "$DRY" = 1 ] || curl -LsSf https://astral.sh/uv/install.sh | sh
fi

run uv sync --frozen
export MUJOCO_GL="${MUJOCO_GL:-egl}"  # render tests skip when no GL backend works
run make check
echo "cores: $(getconf _NPROCESSORS_ONLN)"
run uv run python scripts/bench_env.py --world 8m --steps 2000

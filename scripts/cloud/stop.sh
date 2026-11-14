#!/usr/bin/env bash
# Send SIGTERM to every run. Each run saves a checkpoint and ends.
# Continue a run with: launch.sh ... --resume
# Usage: stop.sh [--runs DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python scripts/launch_seeds.py stop "$@"

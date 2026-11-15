#!/usr/bin/env bash
# Print the steps, level, speed and state of every run, with the end of each log.
# Usage: status.sh [--tail N] [--runs DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python scripts/launch_seeds.py status "$@"

#!/usr/bin/env bash
# Runs on the training VM. After --max-hours it stops the training and DELETES this
# server through the Hetzner API, so that a forgotten VM cannot bill without limit.
# Results that are not copied off the VM by then are lost.
# The API token comes from a file with mode 600. It never appears in an argument list.
# Usage: failsafe.sh --max-hours 12 [--grace-seconds 120] [--token-file FILE] [--runs DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."

MAX_HOURS=12
GRACE=120
TOKEN_FILE=/root/.hcloud_token
RUNS=runs
while [ $# -gt 0 ]; do
  case "$1" in
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --grace-seconds) GRACE="$2"; shift 2 ;;
    --token-file) TOKEN_FILE="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

sleep "$(awk "BEGIN { print int($MAX_HOURS * 3600) }")"
echo "failsafe: the time limit of $MAX_HOURS h is over. Stop the training."
export PATH="$HOME/.local/bin:$PATH"
bash scripts/cloud/stop.sh --runs "$RUNS" || true  # each run saves a checkpoint
sleep "$GRACE"

SERVER_ID="$(curl -fsS http://169.254.169.254/hetzner/v1/metadata/instance-id)"
echo "failsafe: delete server $SERVER_ID"
curl -fsS -X DELETE --config - "https://api.hetzner.cloud/v1/servers/$SERVER_ID" << CONFIG
header = "Authorization: Bearer $(cat "$TOKEN_FILE")"
CONFIG

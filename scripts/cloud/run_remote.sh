#!/usr/bin/env bash
# Train on a rented Hetzner Cloud VM, copy the results back, then DELETE the VM.
# Run it on your own machine, from this repo. It needs the hcloud CLI, ssh and rsync.
#
#   export HCLOUD_TOKEN=...   # in the shell only; never in a file of this repo
#   scripts/cloud/run_remote.sh --ssh-key NAME --config configs/terrain_v2.toml \
#       --seeds 0,1,2 --n-envs 15
#
# Billing stops only when the server is deleted. The script deletes it only after the
# results are on this machine and complete. If anything fails, the server stays and the
# script says so. A failsafe on the VM deletes it after --max-hours in every case.
#
# Options: --type ccx63  --location LOC  --image ubuntu-24.04  --max-hours 12
#          --results DIR (default runs/cloud_<server name>)  --poll-seconds 300
#          --dry-run (print every command, run nothing)  --keep (do not delete)
#          --max-poll-failures 12 (give up after this many failed status checks in a row)
#          --server NAME (attach to a server that already trains: wait, copy, verify, delete)
set -euo pipefail
cd "$(dirname "$0")/../.."

TYPE=ccx63
LOCATION=""
IMAGE=ubuntu-24.04
SSH_KEY=""
CONFIG=configs/terrain_v2.toml
SEEDS=0,1,2
N_ENVS=15
MAX_HOURS=12
POLL=300
RESULTS=""
DRY=0
KEEP=0
SERVER=""
MAX_POLL_FAILURES=12
GONE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --type) TYPE="$2"; shift 2 ;;
    --location) LOCATION="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --n-envs) N_ENVS="$2"; shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --poll-seconds) POLL="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --keep) KEEP=1; shift ;;
    --server) SERVER="$2"; shift 2 ;;
    --max-poll-failures) MAX_POLL_FAILURES="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$SSH_KEY" ] || [ -n "$SERVER" ] || { echo "pass --ssh-key NAME (a key of the Hetzner project)" >&2; exit 2; }
[ -f "$CONFIG" ] || { echo "no config file $CONFIG" >&2; exit 2; }
if [ "$DRY" = 0 ] && [ -z "${HCLOUD_TOKEN:-}" ]; then
  echo "set HCLOUD_TOKEN in the environment of this shell" >&2
  exit 2
fi
if [ "$DRY" = 0 ]; then
  for tool in hcloud ssh rsync git; do
    command -v "$tool" > /dev/null || { echo "install $tool first" >&2; exit 2; }
  done
fi

NAME="${SERVER:-worm-train-$(date -u +%Y%m%d-%H%M%S)}"
RUN_NAME="$(sed -n 's/^name *= *"\(.*\)"/\1/p' "$CONFIG" | head -1)"
RESULTS="${RESULTS:-runs/cloud_$NAME}"
KNOWN_HOSTS="$(mktemp)"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$KNOWN_HOSTS"
  -o ConnectTimeout=10 -o ServerAliveInterval=30)
ON_VM='export PATH="$HOME/.local/bin:$PATH"; cd worm'
IP="<vm-ip>"
CREATED=0
DELETED=0

report() {
  rm -f "$KNOWN_HOSTS"
  [ "$CREATED" = 1 ] || return 0
  if [ "$GONE" = 1 ]; then
    echo "Server $NAME no longer exists. Check: hcloud server list"
  elif [ "$DELETED" = 1 ]; then
    echo "Server $NAME: deleted."
  else
    echo "WARNING: server $NAME is NOT deleted. It bills until you delete it:"
    echo "    hcloud server delete $NAME"
  fi
}
trap report EXIT

run() {
  echo "+ $*"
  [ "$DRY" = 1 ] || "$@"
}
remote() {
  echo "+ ssh root@$IP '$1'"
  [ "$DRY" = 1 ] || ssh "${SSH_OPTS[@]}" "root@$IP" "$1" < /dev/null
}

if [ -n "$SERVER" ]; then
  echo "attach to server $NAME"
  [ "$DRY" = 1 ] || CREATED=1
  [ "$DRY" = 1 ] || IP="$(hcloud server ip "$NAME")"
else
  # 1. Create the server.
  create=(hcloud server create --name "$NAME" --type "$TYPE" --image "$IMAGE"
    --ssh-key "$SSH_KEY" --label purpose=worm-train)
  [ -z "$LOCATION" ] || create+=(--location "$LOCATION")
  run "${create[@]}"
  [ "$DRY" = 1 ] || CREATED=1
  [ "$DRY" = 1 ] || IP="$(hcloud server ip "$NAME")"

  # 2. Wait for SSH.
  echo "+ wait for ssh on root@$IP"
  if [ "$DRY" = 0 ]; then
    for attempt in $(seq 1 60); do
      ssh "${SSH_OPTS[@]}" "root@$IP" true < /dev/null 2> /dev/null && break
      [ "$attempt" != 60 ] || { echo "no ssh after 60 tries" >&2; exit 1; }
      sleep 5
    done
  fi

  # 3. Copy the committed tree to the VM.
  git diff --quiet HEAD 2> /dev/null || echo "note: uncommitted changes are not copied"
  echo "+ git archive HEAD | ssh root@$IP 'mkdir -p worm && tar -x -C worm'"
  [ "$DRY" = 1 ] || git archive HEAD | ssh "${SSH_OPTS[@]}" "root@$IP" 'mkdir -p worm && tar -x -C worm'

  # 4. Install the failsafe first, so that a later failure cannot leave the VM for ever.
  echo "+ ssh root@$IP 'umask 077; cat > /root/.hcloud_token'   (the token goes through stdin)"
  if [ "$DRY" = 0 ]; then
    printf '%s' "$HCLOUD_TOKEN" |
      ssh "${SSH_OPTS[@]}" "root@$IP" 'umask 077; cat > /root/.hcloud_token'
  fi
  # The parentheses matter: without them a waiting shell keeps the ssh session open.
  remote "cd worm && (nohup setsid bash scripts/cloud/failsafe.sh --max-hours $MAX_HOURS > /root/failsafe.log 2>&1 < /dev/null &)"

  # 5. Prepare the machine, then start the training.
  remote "cd worm && bash scripts/cloud/bootstrap.sh"
  remote "$ON_VM && bash scripts/cloud/launch.sh --config $CONFIG --seeds $SEEDS --n-envs $N_ENVS --nohup --no-tensorboard"
fi

# 6. Wait until no run is going. status.sh exits with 3 while one is.
echo "+ poll every $POLL s: ssh root@$IP 'bash scripts/cloud/status.sh --exit-code'"
failures=0
while [ "$DRY" = 0 ]; do
  code=0
  ssh "${SSH_OPTS[@]}" "root@$IP" "$ON_VM && bash scripts/cloud/status.sh --tail 0 --exit-code" \
    < /dev/null || code=$?
  [ "$code" != 0 ] || break
  if [ "$code" = 3 ]; then
    failures=0
  else
    failures=$((failures + 1))
    # Only a "not found" answer means the server is gone. A network fault is not proof.
    answer="$(hcloud server describe "$NAME" 2>&1 > /dev/null || true)"
    if grep -qi "not found" <<< "$answer"; then
      GONE=1
      echo "The server is gone. The failsafe may have deleted it. Its results are lost." >&2
      exit 1
    fi
    if [ "$failures" -ge "$MAX_POLL_FAILURES" ]; then
      echo "$failures status checks failed in a row. Give up; the server still exists." >&2
      exit 1
    fi
    echo "status check failed with code $code ($failures in a row); try again"
  fi
  sleep "$POLL"
done

# 7. Copy the results back and verify them. Checkpoints stay on the VM.
verify() {
  local seed dir file
  for seed in ${SEEDS//,/ }; do
    dir="$RESULTS/${RUN_NAME}_seed$seed"
    for file in policy.zip vecnormalize.pkl meta.json; do
      [ -s "$dir/$file" ] || { echo "missing: $dir/$file"; return 1; }
    done
  done
}
VERIFIED=0
for attempt in 1 2 3; do
  [ "$DRY" = 1 ] || mkdir -p "$RESULTS"
  run rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude='checkpoints/' --include='*/' \
    --include='policy.zip' --include='vecnormalize.pkl' --include='*.json' --include='*.toml' \
    --include='*.log' --include='tb/***' --exclude='*' "root@$IP:worm/runs/" "$RESULTS/" || true
  echo "+ verify policy.zip, vecnormalize.pkl and meta.json of every seed in $RESULTS"
  if [ "$DRY" = 1 ] || verify; then
    VERIFIED=1
    break
  fi
  echo "try $attempt: the results are not complete"
  sleep 10
done
if [ "$VERIFIED" = 0 ]; then
  echo "The results are not complete after 3 tries. The server stays." >&2
  exit 1
fi
echo "The results are in $RESULTS."

# 8. Delete the server, then show what is left.
if [ "$KEEP" = 1 ]; then
  echo "--keep: the server stays."
  exit 0
fi
run hcloud server delete "$NAME"
[ "$DRY" = 1 ] || DELETED=1
run hcloud server list -l purpose=worm-train

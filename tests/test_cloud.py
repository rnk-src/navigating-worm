"""The cloud scripts, run against fake `hcloud`, `ssh`, `rsync` and `curl`. No network."""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CLOUD = ROOT / "scripts" / "cloud"
TOKEN = "secret-token-do-not-print-123"
CONFIG = "configs/terrain_v2_smoke.toml"

# run_remote.sh copies `git archive HEAD`. The copy on the VM is not a git checkout.
needs_git = pytest.mark.skipif(not (ROOT / ".git").exists(), reason="not a git checkout")

FAKE = """#!/usr/bin/env bash
echo "{name} $*" >> "$FAKE_LOG"
{body}
"""
# The fake rsync makes the files that a finished run would bring back.
RSYNC_OK = """
dest="${@: -1}"
for seed in 0 1; do
  mkdir -p "$dest/terrain_v2_smoke_seed$seed"
  for f in policy.zip vecnormalize.pkl meta.json; do
    echo x > "$dest/terrain_v2_smoke_seed$seed/$f"
  done
done
"""


def fakes(tmp_path: Path, rsync_body: str = RSYNC_OK) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bodies = {
        "hcloud": 'if [ "$1 $2" = "server ip" ]; then echo 127.0.0.1; fi',
        "ssh": "cat > /dev/null || true",
        "rsync": rsync_body,
        "curl": 'case "$*" in *metadata*) echo 4242;; esac; cat > /dev/null || true',
        "sleep": "",
    }
    for name, body in bodies.items():
        path = bin_dir / name
        path.write_text(FAKE.format(name=name, body=body))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_LOG": str(tmp_path / "calls.log"),
        "HCLOUD_TOKEN": TOKEN,
    }


def run_remote(
    env: dict[str, str], tmp_path: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(CLOUD / "run_remote.sh"), "--ssh-key", "my-key", "--config", CONFIG]
        + ["--seeds", "0,1", "--n-envs", "2", "--results", str(tmp_path / "results"), *extra],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def positions(text: str, *needles: str) -> list[int]:
    found = [text.find(needle) for needle in needles]
    assert all(at >= 0 for at in found), [n for n, at in zip(needles, found, strict=True) if at < 0]
    return found


@pytest.mark.parametrize("script", ["bootstrap.sh", "run_remote.sh", "failsafe.sh"])
def test_cloud_scripts_parse(script: str) -> None:
    assert subprocess.run(["bash", "-n", str(CLOUD / script)]).returncode == 0
    assert "set -euo pipefail" in (CLOUD / script).read_text()


def test_bootstrap_dry_run_lists_uv_sync_and_make_check() -> None:
    done = subprocess.run(
        ["bash", str(CLOUD / "bootstrap.sh"), "--dry-run"], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    order = positions(done.stdout, "uv sync --frozen", "make check", "bench_env.py")
    assert order == sorted(order)


def test_run_remote_dry_run_orders_create_launch_sync_verify_delete(tmp_path: Path) -> None:
    env = fakes(tmp_path)
    del env["HCLOUD_TOKEN"]  # a dry run needs no token
    done = run_remote(env, tmp_path, "--dry-run")
    assert done.returncode == 0, done.stderr
    order = positions(
        done.stdout,
        "hcloud server create",
        "failsafe.sh",
        "bootstrap.sh",
        "launch.sh",
        "status.sh",
        "rsync",
        "verify",
        "hcloud server delete",
        "hcloud server list",
    )
    assert order == sorted(order)
    assert "--type ccx63" in done.stdout and "purpose=worm-train" in done.stdout
    assert not (tmp_path / "calls.log").exists()  # a dry run calls nothing


@needs_git
def test_run_remote_creates_trains_verifies_and_deletes(tmp_path: Path) -> None:
    env = fakes(tmp_path)
    done = run_remote(env, tmp_path, "--poll-seconds", "0")
    assert done.returncode == 0, done.stdout + done.stderr
    calls = (tmp_path / "calls.log").read_text()
    order = positions(
        calls,
        "hcloud server create",
        "launch.sh",
        "rsync",
        "hcloud server delete",
        "hcloud server list",
    )
    assert order == sorted(order)
    assert "deleted" in done.stdout
    assert (tmp_path / "results" / "terrain_v2_smoke_seed1" / "policy.zip").exists()


@needs_git
def test_run_remote_skips_delete_when_verify_fails(tmp_path: Path) -> None:
    env = fakes(tmp_path, rsync_body="")  # the sync brings nothing back
    done = run_remote(env, tmp_path, "--poll-seconds", "0")
    assert done.returncode != 0
    calls = (tmp_path / "calls.log").read_text()
    assert "hcloud server delete" not in calls
    assert calls.count("rsync ") == 3  # three tries
    assert "NOT deleted" in done.stdout + done.stderr


@needs_git
def test_run_remote_keep_flag_skips_the_delete(tmp_path: Path) -> None:
    done = run_remote(fakes(tmp_path), tmp_path, "--poll-seconds", "0", "--keep")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "hcloud server delete" not in (tmp_path / "calls.log").read_text()


def test_run_remote_needs_a_token_and_an_ssh_key(tmp_path: Path) -> None:
    env = fakes(tmp_path)
    del env["HCLOUD_TOKEN"]
    done = run_remote(env, tmp_path)
    assert done.returncode != 0 and "HCLOUD_TOKEN" in done.stderr
    assert not (tmp_path / "calls.log").exists()


def test_failsafe_builds_the_delete_request_from_metadata(tmp_path: Path) -> None:
    env = fakes(tmp_path)
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN)
    done = subprocess.run(
        ["bash", str(CLOUD / "failsafe.sh"), "--max-hours", "0", "--grace-seconds", "0"]
        + ["--token-file", str(token_file), "--runs", str(tmp_path / "runs")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "calls.log").read_text()
    order = positions(calls, "metadata/instance-id", "-X DELETE")
    assert order == sorted(order)
    assert "https://api.hetzner.cloud/v1/servers/4242" in calls
    assert TOKEN not in calls + done.stdout + done.stderr  # the token goes through stdin


@needs_git
def test_no_token_appears_in_any_log_or_committed_file(tmp_path: Path) -> None:
    done = run_remote(fakes(tmp_path), tmp_path, "--poll-seconds", "0")
    assert done.returncode == 0, done.stdout + done.stderr
    assert TOKEN not in done.stdout + done.stderr + (tmp_path / "calls.log").read_text()
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    for name in tracked.stdout.split():
        path = ROOT / name
        if path.suffix in {".sh", ".py", ".toml", ".md", ".yml"} and path.exists():
            assert not re.search(r"HCLOUD_TOKEN=[\"']?[A-Za-z0-9]{20,}", path.read_text()), name


@needs_git
def test_run_remote_can_attach_to_a_server_that_already_trains(tmp_path: Path) -> None:
    done = run_remote(fakes(tmp_path), tmp_path, "--poll-seconds", "0", "--server", "worm-train-x")
    assert done.returncode == 0, done.stdout + done.stderr
    calls = (tmp_path / "calls.log").read_text()
    for skipped in ("server create", "bootstrap.sh", "failsafe.sh", "launch.sh", "tar -x"):
        assert skipped not in calls
    order = positions(calls, "status.sh", "rsync", "hcloud server delete worm-train-x")
    assert order == sorted(order)
    assert "Server worm-train-x: deleted." in done.stdout

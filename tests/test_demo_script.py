"""The demo script starts, serves its state once and exits when asked to smoke-test."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_demo_smoke_serves_state_and_exits_zero() -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo.py"),
         "--policy", "scripted", "--port", "0", "--no-render", "--smoke"],
        capture_output=True,
        text=True,
        timeout=120,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    state = json.loads(done.stdout.strip().splitlines()[-1])
    assert state["policy"] == "scripted"
    assert len(state["head"]) == 2

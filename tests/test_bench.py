"""The throughput benchmark script runs and reports a rate."""

import re
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "bench_env.py"


def test_bench_env_prints_control_steps_per_second_for_each_world() -> None:
    for world in ("flat", "4m"):
        done = subprocess.run(
            [sys.executable, str(SCRIPT), "--world", world, "--steps", "40"],
            capture_output=True, text=True, timeout=120,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr
        match = re.search(rf"{world}: (\d+) control steps/s", done.stdout)
        assert match and int(match.group(1)) > 10

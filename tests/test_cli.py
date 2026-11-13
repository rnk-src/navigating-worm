"""End-to-end: train a tiny policy with the CLI, then evaluate it with the CLI."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_train_then_eval_route_writes_a_result_table(tmp_path: Path) -> None:
    trained = run("train.py", "--steps", "256", "--n-envs", "1", "--out", str(tmp_path / "run"))
    assert trained.returncode == 0, trained.stderr
    assert (tmp_path / "run" / "policy.zip").exists()

    report = tmp_path / "eval.json"
    evaluated = run(
        "eval_route.py",
        "--policy", str(tmp_path / "run"),
        "--runs", "2",
        "--length", "4",
        "--timeout", "2",
        "--json", str(report),
    )  # fmt: skip
    assert evaluated.returncode == 0, evaluated.stderr
    data = json.loads(report.read_text())
    assert data["runs_total"] == 2
    assert data["successes"] == sum(row["success"] for row in data["runs"])
    assert [row["total"] for row in data["runs"]] == [2, 2]  # 4 m at 2 m spacing
    # An untrained policy cannot cover 2 m in 2 s, so both runs must fail honestly.
    assert data["successes"] == 0
    frictions = {row["mu_forward"] for row in data["runs"]}
    assert len(frictions) == 2  # random friction per run
    assert "success: 0/2" in evaluated.stdout


def test_eval_route_can_record_a_video_of_the_first_run(tmp_path: Path) -> None:
    import pytest

    from worm.env import WormGoalEnv

    try:
        probe = WormGoalEnv(render_mode="rgb_array")
        probe.reset(seed=0)
        probe.render()
        probe.close()
    except Exception as error:  # no GL backend on this machine
        pytest.skip(f"rendering unavailable: {error}")

    run("train.py", "--steps", "256", "--n-envs", "1", "--out", str(tmp_path / "run"))
    video = tmp_path / "route.mp4"
    evaluated = run(
        "eval_route.py",
        "--policy", str(tmp_path / "run"),
        "--runs", "1",
        "--length", "2",
        "--timeout", "1",
        "--video", str(video),
    )  # fmt: skip
    assert evaluated.returncode == 0, evaluated.stderr
    import imageio.v2 as imageio

    frames = imageio.mimread(video)
    assert len(frames) == 20  # 1 s at 20 frames per second
    assert frames[0].shape == (480, 640, 3)

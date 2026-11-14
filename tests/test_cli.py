"""End-to-end: train a tiny policy with the CLI, then evaluate it with the CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_eval_route_can_record_a_video_of_the_first_run(tmp_path: Path, rendering: None) -> None:
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


def test_showcase_writes_a_sped_up_overview_video(tmp_path: Path, rendering: None) -> None:
    run("train.py", "--steps", "256", "--n-envs", "1", "--out", str(tmp_path / "run"))
    video = tmp_path / "showcase.mp4"
    made = run(
        "showcase.py",
        "--policy", str(tmp_path / "run"),
        "--out", str(video),
        "--timeout", "2",
        "--speed", "2",
    )  # fmt: skip
    assert made.returncode == 0, made.stderr
    import imageio.v2 as imageio

    frames = imageio.mimread(video)
    assert len(frames) == 20  # 2 s of sim at 20 Hz, every 2nd frame kept
    assert frames[0].shape == (480, 1280, 3)  # close-up on the left, overview on the right
    left, right = frames[0][:, :640], frames[0][:, 640:]
    green = lambda img: int((img[..., 1].astype(int) - img[..., 0].astype(int) > 40).sum())  # noqa: E731
    assert green(left) > 5 * green(right) > 0  # the worm is much larger in the close-up
    assert "reached 0/9" in made.stdout  # an untrained policy reaches nothing in 2 s
    red = frames[0][..., 0].astype(int) - frames[0][..., 1].astype(int) > 80
    assert red.sum() > 200  # the waypoint markers are in the picture


def test_make_world_writes_a_map_png(tmp_path: Path) -> None:
    out = tmp_path / "map.png"
    made = run(
        "make_world.py", "--seed", "3", "--size", "4", "--mix", "1", "0", "0", "--out", str(out)
    )
    assert made.returncode == 0, made.stderr
    import imageio.v2 as imageio

    image = imageio.imread(out)
    assert image.shape == (160, 160, 3)  # 4 m at 40 pixels per metre
    assert "seed 3" in made.stdout and "concrete 100 %" in made.stdout


def test_plan_map_writes_a_plan_png_or_reports_why_not(tmp_path: Path) -> None:
    out = tmp_path / "plan.png"
    world = ["--seed", "3", "--size", "6", "--rocks", "6"]
    made = run("plan_map.py", *world, "--start", "0", "0", "--goal", "2", "1.5", "--out", str(out))
    assert made.returncode == 0, made.stderr
    import imageio.v2 as imageio

    image = imageio.imread(out)
    assert image.shape == (240, 240, 3)
    blue = (image[..., 2].astype(int) - image[..., 0].astype(int) > 120).sum()
    assert blue > 100  # the path is drawn
    assert "waypoints" in made.stdout

    outside = run("plan_map.py", *world, "--start", "0", "0", "--goal", "9", "9")
    assert outside.returncode == 2
    assert "infeasible: OUT_OF_BOUNDS" in outside.stdout


def test_navigate_drives_a_planned_route_with_the_scripted_controller(tmp_path: Path) -> None:
    report = tmp_path / "nav.json"
    world = ["--seed", "3", "--size", "4", "--rocks", "0", "--ditches", "0", "--max-slope", "5"]
    done = run(
        "navigate.py", "--policy", "scripted", *world, "--goal", "0.8", "0.6", "--json", str(report)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    data = json.loads(report.read_text())
    assert data["success"] and data["reason"] == "reached" and data["policy"] == "scripted"
    assert set(data["class_seconds"]) == {"concrete", "grass", "sand"}
    assert sum(data["class_seconds"].values()) == pytest.approx(data["sim_seconds"], abs=0.05)
    assert data["sim_seconds"] > 3.0
    assert "reached" in done.stdout

    outside = run("navigate.py", "--policy", "scripted", *world, "--goal", "9", "9")
    assert outside.returncode == 2 and "infeasible: OUT_OF_BOUNDS" in outside.stdout


def test_navigate_can_record_a_video_with_the_map_beside_the_follow_camera(
    tmp_path: Path, rendering: None
) -> None:
    video = tmp_path / "nav.mp4"
    world = ["--seed", "3", "--size", "4", "--rocks", "0", "--ditches", "0", "--max-slope", "5"]
    done = run(
        "navigate.py", "--policy", "scripted", *world, "--goal", "0.5", "0.3",
        "--video", str(video), "--speed", "4",
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    import imageio.v2 as imageio

    frames = imageio.mimread(video)
    assert len(frames) >= 5 and frames[0].shape == (480, 1120, 3)  # 640 camera + 480 map

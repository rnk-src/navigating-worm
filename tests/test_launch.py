import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from worm import launch as ln
from worm.training import load_policy

ROOT = Path(__file__).parent.parent
SMOKE = ROOT / "configs" / "terrain_v2_smoke.toml"


def wait_for_exit(pids: list[int], seconds: float = 120.0) -> None:
    deadline = time.monotonic() + seconds
    while any(ln.is_training(pid) for pid in pids):
        assert time.monotonic() < deadline, "the runs did not end"
        time.sleep(0.2)


def test_launch_seeds_builds_one_command_per_seed_with_its_own_out_and_log() -> None:
    runs = ln.build_commands(SMOKE, seeds=[0, 2], n_envs=3, runs=Path("runs"), python="python")
    assert [run.seed for run in runs] == [0, 2]
    assert [run.out for run in runs] == [
        Path("runs/terrain_v2_smoke_seed0"),
        Path("runs/terrain_v2_smoke_seed2"),
    ]
    assert [run.log for run in runs] == [
        Path("runs/terrain_v2_smoke_seed0.log"),
        Path("runs/terrain_v2_smoke_seed2.log"),
    ]
    assert runs[1].command == (
        "python",
        "scripts/train.py",
        "--config",
        str(SMOKE),
        "--seed",
        "2",
        "--n-envs",
        "3",
        "--out",
        "runs/terrain_v2_smoke_seed2",
    )
    resumed = ln.build_commands(SMOKE, [0], 3, resume=True, steps=512, python="python")[0]
    assert resumed.command[-3:] == ("--steps", "512", "--resume")


def test_tmux_command_runs_the_training_detached_with_its_log() -> None:
    run = ln.build_commands(SMOKE, [1], 2, runs=Path("runs"), python="python")[0]
    command = ln.tmux_command(run)
    assert command[:5] == ["tmux", "new-session", "-d", "-s", "terrain_v2_smoke_seed1"]
    assert command[5].startswith("exec python scripts/train.py --config ")
    assert command[5].endswith("> runs/terrain_v2_smoke_seed1.log 2>&1")


def test_launch_refuses_existing_run_without_resume(tmp_path: Path) -> None:
    out = tmp_path / "terrain_v2_smoke_seed0"
    out.mkdir()
    (out / "progress.json").write_text("{}")
    with pytest.raises(FileExistsError, match="--resume"):
        ln.launch(SMOKE, [0], 1, runs=tmp_path, tensorboard=False)
    assert not (tmp_path / "terrain_v2_smoke_launch.json").exists()


def test_launch_nohup_mode_runs_the_smoke_config_for_two_seeds(tmp_path: Path) -> None:
    record = ln.launch(SMOKE, [0, 1], 1, steps=512, runs=tmp_path, tensorboard=False)
    launched = json.loads(record.read_text())
    assert record == tmp_path / "terrain_v2_smoke_launch.json"
    assert [run["seed"] for run in launched["runs"]] == [0, 1]
    wait_for_exit([run["pid"] for run in launched["runs"]])
    for seed in (0, 1):
        out = tmp_path / f"terrain_v2_smoke_seed{seed}"
        assert "saved to" in (tmp_path / f"terrain_v2_smoke_seed{seed}.log").read_text()
        assert load_policy(out).obs_version == 2
    report = ln.status(tmp_path)
    assert "terrain_v2_smoke_seed0" in report and "terrain_v2_smoke_seed1" in report
    assert report.count("done") == 2 and "512/512" in report


def test_stop_ends_a_run_with_a_checkpoint(tmp_path: Path) -> None:
    record = ln.launch(SMOKE, [0], 1, steps=2_000_000, runs=tmp_path, tensorboard=False)
    pid = json.loads(record.read_text())["runs"][0]["pid"]
    out = tmp_path / "terrain_v2_smoke_seed0"
    try:
        deadline = time.monotonic() + 120.0
        while not (out / "progress.json").exists():
            assert time.monotonic() < deadline and ln.is_training(pid)
            time.sleep(0.2)
        assert "running" in ln.status(tmp_path)
        assert ln.stop(tmp_path) == [pid]
        wait_for_exit([pid], 60.0)
    finally:
        if ln.is_training(pid):
            os.kill(pid, 9)
    progress = json.loads((out / "progress.json").read_text())
    assert progress["done"] is False
    assert (out / "checkpoints" / f"{progress['last_checkpoint']}.zip").exists()
    assert "stopped" in ln.status(tmp_path)
    assert ln.stop(tmp_path) == []  # nothing is left to stop


def test_stop_never_signals_a_process_that_is_not_a_training_run(tmp_path: Path) -> None:
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        record = {"runs": [{"seed": 0, "pid": sleeper.pid, "out": "x", "log": "x.log"}]}
        (tmp_path / "x_launch.json").write_text(json.dumps(record))
        assert ln.stop(tmp_path) == []
        assert sleeper.poll() is None
    finally:
        sleeper.kill()


@pytest.mark.parametrize("script", ["launch.sh", "status.sh", "stop.sh"])
def test_cloud_scripts_parse_and_call_the_launcher(script: str) -> None:
    path = ROOT / "scripts" / "cloud" / script
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    text = path.read_text()
    assert "set -euo pipefail" in text and "scripts/launch_seeds.py" in text


def test_status_script_reports_no_runs_in_an_empty_folder(tmp_path: Path) -> None:
    done = subprocess.run(
        ["bash", str(ROOT / "scripts" / "cloud" / "status.sh"), "--runs", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "no runs" in done.stdout


def test_status_exit_code_tells_whether_a_run_is_still_going(tmp_path: Path) -> None:
    assert ln.main(["status", "--runs", str(tmp_path), "--exit-code"]) == 0
    record = ln.launch(SMOKE, [0], 1, steps=2_000_000, runs=tmp_path, tensorboard=False)
    pid = json.loads(record.read_text())["runs"][0]["pid"]
    try:
        assert ln.main(["status", "--runs", str(tmp_path), "--exit-code"]) == ln.RUNNING
        assert ln.main(["status", "--runs", str(tmp_path)]) == 0
    finally:
        os.kill(pid, 9)
    wait_for_exit([pid], 30.0)
    assert ln.main(["status", "--runs", str(tmp_path), "--exit-code"]) == 0


def test_runs_get_one_math_thread_per_process_unless_the_caller_sets_it() -> None:
    env = ln.run_environment({"PATH": "x"})
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1" and env["PATH"] == "x"
    assert ln.run_environment({"OMP_NUM_THREADS": "4"})["OMP_NUM_THREADS"] == "4"


def test_a_second_launch_keeps_the_runs_of_the_first_in_the_record(tmp_path: Path) -> None:
    first = ln.launch(SMOKE, [0], 1, steps=2_000_000, runs=tmp_path, tensorboard=False)
    pid0 = json.loads(first.read_text())["runs"][0]["pid"]
    pids = [pid0]
    try:
        second = ln.launch(SMOKE, [1], 1, steps=2_000_000, runs=tmp_path, tensorboard=False)
        runs = json.loads(second.read_text())["runs"]
        pids = [run["pid"] for run in runs]
        assert second == first and [run["seed"] for run in runs] == [0, 1] and pids[0] == pid0
        assert ln.any_running(tmp_path)
        assert sorted(ln.stop(tmp_path)) == sorted(pids)
        wait_for_exit(pids, 60.0)
    finally:
        for pid in pids:
            if ln.is_training(pid):
                os.kill(pid, 9)


def test_relaunching_a_seed_replaces_its_entry_in_the_record(tmp_path: Path) -> None:
    old = {"mode": "nohup", "runs": [{"seed": 0, "pid": 1, "out": "a"}, {"seed": 1, "pid": 2}]}
    new = [{"seed": 1, "pid": 9}]
    merged = ln.merge_runs(old["runs"], new)
    assert merged == [{"seed": 0, "pid": 1, "out": "a"}, {"seed": 1, "pid": 9}]


def test_a_training_run_is_recognized_in_a_narrow_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ps cuts its output to COLUMNS. A long interpreter path then hides the script name.
    monkeypatch.setenv("COLUMNS", "20")
    record = ln.launch(SMOKE, [0], 1, steps=2_000_000, runs=tmp_path, tensorboard=False)
    pid = json.loads(record.read_text())["runs"][0]["pid"]
    try:
        assert ln.is_training(pid)
    finally:
        os.kill(pid, 9)

"""Start, watch and stop detached training runs, one per seed."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worm.training import PROGRESS_FILE, load_config, read_progress

ROOT = Path(__file__).parent.parent  # the commands run from here
TRAIN_SCRIPT = "scripts/train.py"
TENSORBOARD_PORT = 6006
RUNNING = 3  # exit code of `status --exit-code` while a run is still going


@dataclass(frozen=True)
class Launch:
    seed: int
    command: tuple[str, ...]
    out: Path
    log: Path

    @property
    def name(self) -> str:
        return self.out.name


def build_commands(
    config: Path,
    seeds: Sequence[int],
    n_envs: int,
    resume: bool = False,
    steps: int | None = None,
    runs: Path = Path("runs"),
    python: str = sys.executable,
) -> list[Launch]:
    """Return one training command per seed, each with its own out folder and log."""
    name = load_config(config).run.name
    launches = []
    for seed in seeds:
        out = runs / f"{name}_seed{seed}"
        command = [python, TRAIN_SCRIPT, "--config", str(config), "--seed", str(seed)]
        command += ["--n-envs", str(n_envs), "--out", str(out)]
        if steps is not None:
            command += ["--steps", str(steps)]
        if resume:
            command.append("--resume")
        launches.append(Launch(seed, tuple(command), out, runs / f"{out.name}.log"))
    return launches


def tmux_command(run: Launch) -> list[str]:
    """Return the tmux call that runs `run` in its own detached session."""
    shell = f"exec {shlex.join(run.command)} > {shlex.quote(str(run.log))} 2>&1"
    return ["tmux", "new-session", "-d", "-s", run.name, shell]


def _command_of(pid: int) -> str:
    """Return the command line of a live process, or "" when it has ended."""
    done = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat=,command="], capture_output=True, text=True
    )
    fields = done.stdout.strip().split(None, 1)
    if done.returncode != 0 or len(fields) < 2 or fields[0].startswith("Z"):
        return ""
    return fields[1]


def is_training(pid: int) -> bool:
    """Return True when `pid` is a live training run."""
    return TRAIN_SCRIPT in _command_of(pid)


def run_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Return the environment of a run: `base` plus one math thread per process.

    Each env worker is its own process. Without the cap, every process starts one
    BLAS and OpenMP thread per core on Linux, and the threads fight for the cores.
    On a 16-core VM the cap raised the speed of a run from 450 to 1,140 steps per second.
    """
    env = {**base, "PYTHONUNBUFFERED": "1"}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def _detached(command: Sequence[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as sink:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # the run outlives this process and its terminal
            env=run_environment(os.environ),
        )
    return process.pid


def _tmux(run: Launch) -> int:
    subprocess.run(tmux_command(run), cwd=ROOT, check=True)
    pane = subprocess.run(
        ["tmux", "list-panes", "-t", run.name, "-F", "#{pane_pid}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(pane.stdout.split()[0])


def launch(
    config: Path,
    seeds: Sequence[int],
    n_envs: int,
    resume: bool = False,
    mode: str = "nohup",
    steps: int | None = None,
    runs: Path = Path("runs"),
    tensorboard: bool = True,
) -> Path:
    """Start one detached run per seed. Return the path of the launch record.

    A seed whose out folder already holds a run starts only with `resume`.
    """
    launches = build_commands(config, seeds, n_envs, resume, steps, runs)
    for run in launches:
        if (ROOT / run.out / PROGRESS_FILE).exists() and not resume:
            raise FileExistsError(f"{run.out} already holds a run. Pass --resume to continue it.")
    (ROOT / runs).mkdir(parents=True, exist_ok=True)
    started: list[dict[str, Any]] = []
    for run in launches:
        pid = _tmux(run) if mode == "tmux" else _detached(run.command, ROOT / run.log)
        started.append(
            {
                "seed": run.seed,
                "pid": pid,
                "command": list(run.command),
                "out": str(run.out),
                "log": str(run.log),
            }
        )
    record: dict[str, Any] = {"mode": mode, "runs": started}
    if tensorboard:
        board = [sys.executable, "-m", "tensorboard.main", "--logdir", str(runs)]
        board += ["--port", str(TENSORBOARD_PORT)]  # it binds to localhost only
        record["tensorboard_pid"] = _detached(board, ROOT / runs / "tensorboard.log")
    path = ROOT / runs / f"{load_config(config).run.name}_launch.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def _records(runs: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((ROOT / runs).glob("*_launch.json"))]


def _eta(progress: dict[str, Any]) -> str:
    fps = float(progress.get("fps", 0))
    left = int(progress["target"]) - int(progress["steps"])
    if fps <= 0 or left <= 0:
        return ""
    minutes = round(left / fps / 60)
    return f", about {minutes // 60} h {minutes % 60} min left"


def status(runs: Path = Path("runs"), tail: int = 0) -> str:
    """Return one line per run: steps, level, speed and whether it still runs."""
    pids = {Path(run["out"]).name: int(run["pid"]) for rec in _records(runs) for run in rec["runs"]}
    lines = []
    for folder in sorted(path.parent for path in (ROOT / runs).glob(f"*/{PROGRESS_FILE}")):
        progress = read_progress(folder)
        if progress.get("done"):
            state = "done"
        elif folder.name in pids and is_training(pids[folder.name]):
            state = "running" + _eta(progress)
        else:
            state = "stopped (continue it with --resume)"
        steps, target = int(progress["steps"]), int(progress["target"])
        lines.append(
            f"{folder.name}: {steps}/{target} steps ({100 * steps / max(target, 1):.0f} %), "
            f"level {progress.get('level', 0)}, {progress.get('fps', 0)} fps, {state}"
        )
        log = folder.with_name(f"{folder.name}.log")
        if tail > 0 and log.exists():
            lines += ["    " + line for line in log.read_text().splitlines()[-tail:]]
    starting = [name for name, pid in pids.items() if is_training(pid)]
    for name in starting:
        if not (ROOT / runs / name / PROGRESS_FILE).exists():
            lines.append(f"{name}: running, no checkpoint yet")
    return "\n".join(lines) if lines else f"no runs in {runs}"


def any_running(runs: Path = Path("runs")) -> bool:
    """Return True while a run of the launch records is still going."""
    return any(is_training(int(run["pid"])) for rec in _records(runs) for run in rec["runs"])


def stop(runs: Path = Path("runs")) -> list[int]:
    """Send SIGTERM to every live run of the launch records. Each saves a checkpoint.

    Return the pids signalled. A pid that is no longer a training run is left alone.
    """
    stopped = []
    for record in _records(runs):
        for run in record["runs"]:
            pid = int(run["pid"])
            if is_training(pid):
                os.kill(pid, signal.SIGTERM)
                stopped.append(pid)
        board = record.get("tensorboard_pid")
        if board is not None and "tensorboard" in _command_of(int(board)):
            os.kill(int(board), signal.SIGTERM)
    return stopped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("launch", help="start one detached run per seed")
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--seeds", default="0", help="comma-separated, for example 0,1,2")
    start.add_argument("--n-envs", type=int, required=True)
    start.add_argument("--steps", type=int, default=None, help="replace the steps of the config")
    start.add_argument("--resume", action="store_true")
    how = start.add_mutually_exclusive_group()
    how.add_argument("--nohup", dest="mode", action="store_const", const="nohup", default="nohup")
    how.add_argument("--tmux", dest="mode", action="store_const", const="tmux")
    start.add_argument("--no-tensorboard", action="store_true")
    watch = commands.add_parser("status", help="print the progress of every run")
    watch.add_argument("--tail", type=int, default=3, help="log lines to print per run")
    watch.add_argument(
        "--exit-code", action="store_true", help=f"exit with {RUNNING} while a run is still going"
    )
    end = commands.add_parser("stop", help="checkpoint and end every run")
    for command in (start, watch, end):
        command.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    if args.action == "launch":
        seeds = [int(seed) for seed in args.seeds.split(",")]
        try:
            record = launch(
                args.config,
                seeds,
                args.n_envs,
                args.resume,
                args.mode,
                args.steps,
                args.runs,
                tensorboard=not args.no_tensorboard,
            )
        except FileExistsError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"started seeds {args.seeds}. Record: {record.name}. Watch: scripts/cloud/status.sh")
    elif args.action == "status":
        print(status(args.runs, args.tail))
        if args.exit_code and any_running(args.runs):
            return RUNNING
    else:
        print(f"sent SIGTERM to {len(stop(args.runs))} runs")
    return 0

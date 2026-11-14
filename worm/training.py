"""PPO training from a config, with checkpoints and resume, and policy loading."""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import time
import tomllib
import zipfile
from collections import deque
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from worm.curriculum import N_LEVELS, WorldCurriculum
from worm.env import WormGoalEnv
from worm.randomize import DomainRandomizer
from worm.spec import WormSpec, action_spec, obs_layout

POLICY_FILE = "policy.zip"
NORM_FILE = "vecnormalize.pkl"
META_FILE = "meta.json"
CONFIG_FILE = "config.toml"
PROGRESS_FILE = "progress.json"
CHECKPOINT_DIR = "checkpoints"


@dataclass(frozen=True)
class RunConfig:
    name: str = "run"
    steps: int = 3_000_000
    n_envs: int = 8
    seed: int = 0


@dataclass(frozen=True)
class EnvConfig:
    obs_version: int = 1
    world: str = "flat"  # "flat": the plane. "curriculum": worlds from `WorldCurriculum`.
    world_size: float = 8.0  # m
    cell: float = 0.05  # m
    randomizer_level: float = 1.0


@dataclass(frozen=True)
class CurriculumConfig:
    mode: str = "timesteps"  # "timesteps" or "adaptive"
    # timesteps: level L starts at this fraction of the run.
    boundaries: tuple[float, ...] = (0.0, 0.15, 0.35, 0.6)
    replay_lower: float = 0.3
    # adaptive: advance when the success rate over `window` episodes reaches `advance_at`.
    advance_at: float = 0.8
    window: int = 200


@dataclass(frozen=True)
class PPOConfig:
    n_steps: int = 1024
    batch_size: int = 1024
    n_epochs: int = 10
    gamma: float = 0.99
    learning_rate: float = 3e-4
    net_arch: tuple[int, ...] = (64, 64)
    log_std_init: float = -1.0


@dataclass(frozen=True)
class CheckpointConfig:
    every_steps: int = 0  # 0 saves no checkpoint during the run
    keep: int = 3  # 0 keeps every checkpoint


@dataclass(frozen=True)
class EvalConfig:
    every_steps: int = 0  # 0 runs no navigation eval during the run
    maps: int = 5
    map_size: float = 10.0  # m


@dataclass(frozen=True)
class TrainConfig:
    run: RunConfig = RunConfig()
    env: EnvConfig = EnvConfig()
    curriculum: CurriculumConfig = CurriculumConfig()
    ppo: PPOConfig = PPOConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    eval: EvalConfig = EvalConfig()


def _section(cls: type, name: str, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"unknown key {unknown[0]} in [{name}]")
    coerced: dict[str, Any] = {}
    for key, value in values.items():
        default = known[key].default
        if isinstance(default, tuple):
            kind = type(default[0])
            coerced[key] = tuple(kind(item) for item in value)
        elif isinstance(default, float):
            coerced[key] = float(value)
        elif type(value) is not type(default):
            raise ValueError(f"[{name}] {key} must be {type(default).__name__}")
        else:
            coerced[key] = value
    return cls(**coerced)


def load_config(path: Path) -> TrainConfig:
    """Read a TOML training config. An unknown section or key is an error."""
    with path.open("rb") as file:
        data = tomllib.load(file)
    sections = {f.name: f.default.__class__ for f in fields(TrainConfig)}
    unknown = sorted(set(data) - set(sections))
    if unknown:
        raise ValueError(f"unknown section [{unknown[0]}]")
    cfg = TrainConfig(**{name: _section(sections[name], name, data[name]) for name in data})
    if cfg.env.world not in ("flat", "curriculum"):
        raise ValueError(f"[env] world must be flat or curriculum, not {cfg.env.world}")
    if cfg.curriculum.mode not in ("timesteps", "adaptive"):
        raise ValueError(
            f"[curriculum] mode must be timesteps or adaptive, not {cfg.curriculum.mode}"
        )
    if len(cfg.curriculum.boundaries) != N_LEVELS:
        raise ValueError(f"[curriculum] boundaries needs {N_LEVELS} values")
    return cfg


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return repr(value)


def dump_config(cfg: TrainConfig) -> str:
    """Return `cfg` as TOML that `load_config` reads back."""
    lines: list[str] = []
    for name, section in asdict(cfg).items():
        lines.append(f"[{name}]")
        lines += [f"{key} = {_toml(value)}" for key, value in section.items()]
        lines.append("")
    return "\n".join(lines)


def level_at(boundaries: tuple[float, ...], fraction: float) -> int:
    """Return the curriculum level at `fraction` of the run."""
    return max(0, sum(fraction >= b for b in boundaries) - 1)


class AdaptiveLevel:
    """Decide when to advance: the success rate over a full window reaches the threshold."""

    def __init__(self, threshold: float, window: int) -> None:
        self._threshold = threshold
        self._results: deque[bool] = deque(maxlen=window)

    def record(self, success: bool) -> bool:
        """Add one episode. Return True when the level should advance."""
        self._results.append(success)
        full = len(self._results) == self._results.maxlen
        if full and sum(self._results) / len(self._results) >= self._threshold:
            self._results.clear()
            return True
        return False


def make_env(cfg: TrainConfig) -> Monitor:
    """Return one monitored training env of `cfg`."""
    world = None
    if cfg.env.world == "curriculum":
        world = WorldCurriculum(
            size=cfg.env.world_size, cell=cfg.env.cell, replay_lower=cfg.curriculum.replay_lower
        )
    env = WormGoalEnv(
        randomizer=DomainRandomizer(cfg.env.randomizer_level),
        world=world,
        obs_version=cfg.env.obs_version,
    )
    return Monitor(env, info_keywords=("is_success",))


def _git_commit() -> str:
    """Return the short commit hash of this checkout, or "unknown"."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "unknown"


def make_meta(obs_version: int, steps: int, config: str) -> dict[str, Any]:
    """Return the metadata of a saved run. It holds nothing about the training machine."""
    spec = WormSpec()
    return {
        "obs_version": obs_version,
        "obs_dim": obs_layout(spec, obs_version).dim,
        "action_dim": action_spec(spec).dim,
        "steps": steps,
        "config": config,
        "git_commit": _git_commit(),
    }


def write_meta(out: Path, meta: dict[str, Any]) -> None:
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")


def _strip_system_info(archive: Path) -> None:
    """Remove the file in which SB3 records the OS and hardware of the training machine."""
    clean = archive.with_suffix(".tmp")
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(clean, "w") as target:
        for item in source.infolist():
            if item.filename != "system_info.txt":
                target.writestr(item, source.read(item.filename))
    shutil.move(clean, archive)


def _save_model(model: PPO, archive: Path) -> None:
    """Save `model` without the log folder path and without the machine description."""
    log_dir, model.tensorboard_log = model.tensorboard_log, None
    try:
        model.save(archive)
    finally:
        model.tensorboard_log = log_dir
    _strip_system_info(archive)


def _checkpoint_step(archive: Path) -> int:
    return int(archive.stem.removeprefix("step_"))


def checkpoints(out: Path) -> list[Path]:
    """Return the checkpoint archives of the run in `out`, oldest first."""
    return sorted((out / CHECKPOINT_DIR).glob("step_*.zip"), key=_checkpoint_step)


def _norm_of(archive: Path) -> Path:
    return archive.with_name(f"{archive.stem}_vecnormalize.pkl")


def _save_checkpoint(model: PPO, out: Path, keep: int) -> str:
    """Save the model and its normalizer. Keep the newest `keep`. Return the name."""
    folder = out / CHECKPOINT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"step_{model.num_timesteps}.zip"
    _save_model(model, archive)
    model.get_vec_normalize_env().save(str(_norm_of(archive)))
    if keep > 0:
        for old in checkpoints(out)[:-keep]:
            old.unlink()
            _norm_of(old).unlink(missing_ok=True)
    return archive.stem


def read_progress(out: Path) -> dict[str, Any]:
    """Return the progress record of the run in `out`, or {} when there is none."""
    file = out / PROGRESS_FILE
    progress: dict[str, Any] = json.loads(file.read_text()) if file.exists() else {}
    return progress


class _Stop:
    """Set by SIGTERM or SIGINT. The run then saves a checkpoint and ends."""

    def __init__(self) -> None:
        self.requested = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        self.requested = True


class _StopCallback(BaseCallback):  # type: ignore[misc]
    def __init__(self, stop: _Stop) -> None:
        super().__init__()
        self._stop = stop

    def _on_step(self) -> bool:
        return not self._stop.requested


class CurriculumCallback(BaseCallback):  # type: ignore[misc]
    """Set the curriculum level of every env, by timesteps or by success rate."""

    def __init__(self, cfg: CurriculumConfig, total_steps: int, level: int = 0) -> None:
        super().__init__()
        self._cfg, self._total = cfg, total_steps
        self._adaptive = AdaptiveLevel(cfg.advance_at, cfg.window)
        self.level = level

    def _apply(self, level: int) -> None:
        self.level = level
        self.training_env.env_method("set_level", level)

    def _on_training_start(self) -> None:
        if self._cfg.mode == "timesteps":
            self.level = level_at(self._cfg.boundaries, self.num_timesteps / self._total)
        self._apply(self.level)

    def _on_step(self) -> bool:
        level = self.level
        if self._cfg.mode == "timesteps":
            level = level_at(self._cfg.boundaries, self.num_timesteps / self._total)
        else:
            for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
                if done and self._adaptive.record(bool(info.get("is_success", False))):
                    level = min(level + 1, N_LEVELS - 1)
        if level != self.level:
            self._apply(level)
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("curriculum/level", self.level)


class CheckpointCallback(BaseCallback):  # type: ignore[misc]
    """Save a checkpoint every `every_steps` and keep `progress.json` current."""

    def __init__(self, run: _Run) -> None:
        super().__init__()
        self._run = run
        self._next = 0

    def _on_training_start(self) -> None:
        every = self._run.cfg.checkpoint.every_steps
        self._next = (self.num_timesteps // every + 1) * every if every > 0 else 0

    def _on_step(self) -> bool:
        if self._next and self.num_timesteps >= self._next:
            self._next += self._run.cfg.checkpoint.every_steps
            self._run.checkpoint(done=False)
        return True


@dataclass
class _Run:
    """The files of one run and the state that `progress.json` reports."""

    cfg: TrainConfig
    out: Path
    model: PPO
    curriculum: CurriculumCallback | None = None
    started: float = field(default_factory=time.monotonic)
    first_step: int = 0

    def checkpoint(self, done: bool) -> None:
        name = _save_checkpoint(self.model, self.out, self.cfg.checkpoint.keep)
        self.write_progress(name, done)

    def write_progress(self, last_checkpoint: str | None, done: bool) -> None:
        steps = int(self.model.num_timesteps)
        elapsed = max(time.monotonic() - self.started, 1e-9)
        progress = {
            "steps": steps,
            "target": self.cfg.run.steps,
            "last_checkpoint": last_checkpoint,
            "level": 0 if self.curriculum is None else self.curriculum.level,
            "fps": round((steps - self.first_step) / elapsed),
            "done": done,
        }
        (self.out / PROGRESS_FILE).write_text(json.dumps(progress, indent=2) + "\n")


def _train_config(cfg: TrainConfig, out: Path, resume: bool = False, verbose: int = 0) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    newest: Path | None = None
    if resume:
        saved = checkpoints(out)
        if not saved:
            raise FileNotFoundError(f"no checkpoint to resume from in {out.name}")
        newest = saved[-1]
    (out / CONFIG_FILE).write_text(dump_config(cfg))
    n_envs = cfg.run.n_envs
    venv = (SubprocVecEnv if n_envs > 1 else DummyVecEnv)([partial(make_env, cfg)] * n_envs)
    log_dir = str(out / "tb")
    if newest is not None:
        env = VecNormalize.load(str(_norm_of(newest)), venv)
        model = PPO.load(newest, env=env, device="cpu", tensorboard_log=log_dir, verbose=verbose)
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=cfg.ppo.n_steps,
            batch_size=cfg.ppo.batch_size,
            n_epochs=cfg.ppo.n_epochs,
            gamma=cfg.ppo.gamma,
            learning_rate=cfg.ppo.learning_rate,
            policy_kwargs={
                "net_arch": list(cfg.ppo.net_arch),
                "log_std_init": cfg.ppo.log_std_init,
            },
            seed=cfg.run.seed,
            device="cpu",
            verbose=verbose,
            tensorboard_log=log_dir,
        )

    run = _Run(cfg, out, model, first_step=int(model.num_timesteps))
    stop = _Stop()
    callbacks: list[BaseCallback] = [_StopCallback(stop)]
    if cfg.env.world == "curriculum":
        level = int(read_progress(out).get("level", 0)) if resume else 0
        run.curriculum = CurriculumCallback(cfg.curriculum, cfg.run.steps, level)
        callbacks.append(run.curriculum)
    callbacks.append(CheckpointCallback(run))

    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, stop)
        except ValueError:  # not the main thread: the caller owns the signals
            pass
    try:
        remaining = cfg.run.steps - int(model.num_timesteps)
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=CallbackList(callbacks),
                reset_num_timesteps=not resume,
                tb_log_name="ppo",
            )
        if stop.requested:
            run.checkpoint(done=False)
            return out
        last = None
        if cfg.checkpoint.every_steps > 0:
            last = _save_checkpoint(model, out, cfg.checkpoint.keep)
        _save_model(model, out / POLICY_FILE)
        env.save(str(out / NORM_FILE))
        write_meta(out, make_meta(cfg.env.obs_version, int(model.num_timesteps), cfg.run.name))
        run.write_progress(last, done=True)
        return out
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        env.close()


def _train_steps(
    steps: int, n_envs: int, seed: int, out: Path, verbose: int = 0, obs_version: int = 1
) -> Path:
    """Train on the plane with the M1 settings, sized to `steps`."""
    rollout = min(1024, max(64, steps // n_envs))
    cfg = TrainConfig(
        run=RunConfig(name="default", steps=steps, n_envs=n_envs, seed=seed),
        env=EnvConfig(obs_version=obs_version),
        ppo=PPOConfig(n_steps=rollout, batch_size=min(1024, rollout * n_envs)),
    )
    return _train_config(cfg, out, verbose=verbose)


@overload
def train(cfg: TrainConfig, out: Path, resume: bool = ..., verbose: int = ...) -> Path: ...
@overload
def train(
    steps: int, n_envs: int, seed: int, out: Path, verbose: int = ..., obs_version: int = ...
) -> Path: ...
def train(*args: Any, **kwargs: Any) -> Path:
    """Train PPO. Save the policy, its normalizer and `meta.json` to `out`.

    `train(cfg, out, resume=False)` runs a `TrainConfig`. It writes TensorBoard logs,
    checkpoints, `config.toml` and `progress.json`. `resume=True` continues from the
    newest checkpoint in `out`. SIGTERM or SIGINT saves a checkpoint and ends the run.

    `train(steps, n_envs, seed, out)` is the M1 flat-ground run.
    """
    first = args[0] if args else kwargs.get("cfg")
    if isinstance(first, TrainConfig):
        return _train_config(*args, **kwargs)
    return _train_steps(*args, **kwargs)


def with_overrides(
    cfg: TrainConfig, steps: int | None = None, n_envs: int | None = None, seed: int | None = None
) -> TrainConfig:
    """Return `cfg` with the given run values replaced."""
    run = cfg.run
    run = replace(
        run,
        steps=run.steps if steps is None else steps,
        n_envs=run.n_envs if n_envs is None else n_envs,
        seed=run.seed if seed is None else seed,
    )
    return replace(cfg, run=run)


@dataclass(frozen=True, eq=False)
class LoadedPolicy:
    """A saved policy as a deterministic `obs -> action` function, plus its metadata."""

    model: PPO
    norm: VecNormalize
    meta: dict[str, Any]

    @property
    def obs_version(self) -> int:
        return int(self.meta["obs_version"])

    def __call__(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        action, _ = self.model.predict(self.norm.normalize_obs(obs), deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def read_meta(path: Path) -> dict[str, Any]:
    """Return the metadata of the run in `path`. A run without the file is obs v1."""
    file = path / META_FILE
    if not file.exists():
        return make_meta(1, 0, "unknown") | {"git_commit": "unknown"}
    meta: dict[str, Any] = json.loads(file.read_text())
    return meta


def load_policy(path: Path) -> LoadedPolicy:
    """Load the policy in `path` with the observation version that its metadata names."""
    meta = read_meta(path)
    version, dim = int(meta["obs_version"]), int(meta["obs_dim"])
    if obs_layout(WormSpec(), version).dim != dim:
        raise ValueError(f"meta.json: obs v{version} does not have {dim} dims")
    dummy = DummyVecEnv([partial(WormGoalEnv, obs_version=version)])
    try:
        norm = VecNormalize.load(str(path / NORM_FILE), dummy)
    except (AssertionError, ValueError) as error:
        raise ValueError(f"the normalizer in {path.name} does not fit obs v{version}") from error
    if norm.obs_rms.mean.shape != (dim,):
        raise ValueError(f"the normalizer in {path.name} does not fit obs v{version}")
    norm.training = False
    model = PPO.load(path / POLICY_FILE, device="cpu")
    if model.observation_space.shape != (dim,):
        raise ValueError(f"the policy in {path.name} does not take obs v{version}")
    return LoadedPolicy(model, norm, meta)

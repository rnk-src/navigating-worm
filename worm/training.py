"""PPO training and policy loading."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from worm.env import WormGoalEnv
from worm.spec import WormSpec, action_spec, obs_layout

POLICY_FILE = "policy.zip"
NORM_FILE = "vecnormalize.pkl"
META_FILE = "meta.json"


def _make_env(obs_version: int = 1) -> Monitor:
    return Monitor(WormGoalEnv(obs_version=obs_version), info_keywords=("is_success",))


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


def train(
    steps: int, n_envs: int, seed: int, out: Path, verbose: int = 0, obs_version: int = 1
) -> Path:
    """Train PPO for `steps` env steps. Save the policy and its normalizer to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    make = partial(_make_env, obs_version)
    venv = (SubprocVecEnv if n_envs > 1 else DummyVecEnv)([make] * n_envs)
    env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    rollout = min(1024, max(64, steps // n_envs))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=rollout,
        batch_size=min(1024, rollout * n_envs),
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.0},
        seed=seed,
        device="cpu",
        verbose=verbose,
    )
    model.learn(total_timesteps=steps)
    model.save(out / POLICY_FILE)
    _strip_system_info(out / POLICY_FILE)
    env.save(str(out / NORM_FILE))
    env.close()
    write_meta(out, make_meta(obs_version, steps, "default"))
    return out


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

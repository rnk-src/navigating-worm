"""PPO training and policy loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from worm.env import WormGoalEnv
from worm.route import Policy

POLICY_FILE = "policy.zip"
NORM_FILE = "vecnormalize.pkl"


def _make_env() -> Monitor:
    return Monitor(WormGoalEnv(), info_keywords=("is_success",))


def train(steps: int, n_envs: int, seed: int, out: Path, verbose: int = 0) -> Path:
    """Train PPO for `steps` env steps. Save the policy and its normalizer to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    venv = (SubprocVecEnv if n_envs > 1 else DummyVecEnv)([_make_env] * n_envs)
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
    env.save(str(out / NORM_FILE))
    env.close()
    return out


def load_policy(path: Path) -> Policy:
    """Load a saved policy as a deterministic `obs -> action` function."""
    norm = VecNormalize.load(str(path / NORM_FILE), DummyVecEnv([WormGoalEnv]))
    norm.training = False
    model = PPO.load(path / POLICY_FILE, device="cpu")

    def policy(obs: NDArray[np.float32]) -> NDArray[np.float32]:
        action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    return policy

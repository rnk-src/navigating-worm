from pathlib import Path

import numpy as np
import pytest

from worm.env import WormGoalEnv
from worm.training import load_policy, train


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train(steps=512, n_envs=1, seed=0, out=tmp_path_factory.mktemp("run"))


def test_training_saves_the_policy_and_its_normalizer(run_dir: Path) -> None:
    assert (run_dir / "policy.zip").stat().st_size > 0
    assert (run_dir / "vecnormalize.pkl").stat().st_size > 0


def test_loaded_policy_maps_an_observation_to_a_bounded_deterministic_action(
    run_dir: Path,
) -> None:
    policy = load_policy(run_dir)
    obs, _ = WormGoalEnv().reset(seed=0)
    action = policy(obs)
    assert action.shape == (7,) and action.dtype == np.float32
    assert np.all(np.abs(action) <= 1.0)
    assert np.array_equal(action, policy(obs))

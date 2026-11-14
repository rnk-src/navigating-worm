import json
import shutil
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


def test_saved_policy_archive_does_not_record_the_training_machine(run_dir: Path) -> None:
    import zipfile

    with zipfile.ZipFile(run_dir / "policy.zip") as archive:
        names = archive.namelist()
    assert "system_info.txt" not in names
    assert "policy.pth" in names  # the weights are still there


def test_saved_run_has_meta_json_with_obs_version(run_dir: Path) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["obs_version"] == 1 and meta["obs_dim"] == 24 and meta["action_dim"] == 7
    assert meta["steps"] == 512
    assert set(meta) == {"obs_version", "obs_dim", "action_dim", "steps", "config", "git_commit"}


def test_meta_contains_no_machine_info(run_dir: Path) -> None:
    text = (run_dir / "meta.json").read_text()
    assert "/" not in text and "\\" not in text
    assert not {"host", "hostname", "user", "path", "platform"} & set(json.loads(text))


def test_load_policy_without_meta_defaults_to_v1() -> None:
    committed = Path(__file__).parent.parent / "results" / "policy"
    assert not (committed / "meta.json").exists()
    policy = load_policy(committed)
    assert policy.meta["obs_version"] == 1 and policy.obs_version == 1
    assert policy(np.zeros(24, dtype=np.float32)).shape == (7,)


def test_load_policy_v2_builds_a_31d_normalizer(tmp_path: Path) -> None:
    run = train(steps=128, n_envs=1, seed=0, out=tmp_path, obs_version=2)
    policy = load_policy(run)
    assert policy.obs_version == 2 and policy.meta["obs_dim"] == 31
    obs, _ = WormGoalEnv(obs_version=2).reset(seed=0)
    assert policy(obs).shape == (7,)


def test_load_policy_rejects_a_meta_that_does_not_match_the_normalizer(run_dir: Path) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    broken = run_dir.parent / "broken"
    shutil.copytree(run_dir, broken)
    (broken / "meta.json").write_text(json.dumps({**meta, "obs_version": 2, "obs_dim": 31}))
    with pytest.raises(ValueError, match="obs"):
        load_policy(broken)


def test_committed_terrain_policy_loads_as_obs_v2() -> None:
    policy = load_policy(Path(__file__).parent.parent / "results" / "policy_terrain")
    assert policy.obs_version == 2 and policy.meta["config"] == "terrain_v2"
    obs, _ = WormGoalEnv(obs_version=2).reset(seed=0)
    assert policy(obs).shape == (7,)

"""The numpy policy must act exactly like the trained policy, and it must not need torch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from worm import npolicy
from worm.training import load_policy

POLICY = Path("results/policy_terrain")


@pytest.fixture(scope="module")
def trained():  # type: ignore[no-untyped-def]
    return load_policy(POLICY)


def test_the_exported_policy_matches_the_trained_policy(trained) -> None:  # type: ignore[no-untyped-def]
    exported = npolicy.NumpyPolicy.from_dict(npolicy.export(trained))
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = rng.normal(0.0, 1.5, trained.meta["obs_dim"]).astype(np.float32)
        assert np.allclose(exported(obs), trained(obs), atol=1e-4)


def test_the_export_survives_a_json_round_trip(trained, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(npolicy.export(trained)))
    again = npolicy.NumpyPolicy.from_dict(json.loads(path.read_text()))
    obs = np.linspace(-1, 1, trained.meta["obs_dim"], dtype=np.float32)
    assert np.allclose(again(obs), trained(obs), atol=1e-4)


def test_the_committed_export_matches_the_committed_policy(trained) -> None:  # type: ignore[no-untyped-def]
    committed = npolicy.load(POLICY / "policy.json")
    obs = np.full(trained.meta["obs_dim"], 0.3, dtype=np.float32)
    assert committed.obs_version == trained.obs_version == 2
    assert np.allclose(committed(obs), trained(obs), atol=1e-4)


def test_the_policy_module_needs_only_numpy() -> None:
    code = "import sys, worm.npolicy; print('torch' in sys.modules, 'stable_baselines3' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_actions_stay_in_range() -> None:
    policy = npolicy.load(POLICY / "policy.json")
    rng = np.random.default_rng(1)
    for _ in range(50):
        action = policy(rng.normal(0.0, 5.0, 31).astype(np.float32))
        assert action.shape == (7,) and np.all(np.abs(action) <= 1.0)

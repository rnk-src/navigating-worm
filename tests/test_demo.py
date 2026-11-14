"""The click-to-go session: map requests, controller loading and the headless session."""

import json
from pathlib import Path

import pytest

from worm import demo
from worm.demo import MapRequest


def test_map_request_from_json_clamps_and_normalizes_mix() -> None:
    req = MapRequest.from_json(
        {"seed": 7, "size": 99, "mix": [2, 1, 1], "slope": 80, "rocks": 500, "ditches": -3}
    )
    assert req.seed == 7 and req.size == 15.0  # the largest map
    assert req.mix == pytest.approx((0.5, 0.25, 0.25))
    assert req.max_slope_deg == 20.0 and req.n_rocks == 40 and req.n_ditches == 0
    cfg = req.to_config()
    assert cfg.size == (15.0, 15.0) and cfg.n_rocks == 40 and cfg.max_slope_deg == 20.0


def test_map_request_defaults_fill_missing_keys() -> None:
    assert MapRequest.from_json({}) == MapRequest()
    assert MapRequest.from_json({"seed": 3}).seed == 3


@pytest.mark.parametrize(
    "garbage",
    [
        [1, 2, 3],
        "map",
        {"seed": "abc"},
        {"size": None},
        {"mix": [1, 2]},
        {"mix": [0, 0, 0]},
        {"mix": [1, -1, 1]},
        {"mix": "half"},
        {"rocks": 2.5},
        {"slope": float("nan")},
    ],
)
def test_map_request_rejects_garbage(garbage: object) -> None:
    with pytest.raises(ValueError):
        MapRequest.from_json(garbage)  # type: ignore[arg-type]


def test_load_controller_falls_back_to_scripted_when_no_policy_exists(tmp_path: Path) -> None:
    controller = demo.load_controller(
        None, search=(tmp_path / "policy_terrain", tmp_path / "policy")
    )
    assert controller.name == "scripted" and controller.obs_version == 1
    assert demo.load_controller("scripted").name == "scripted"
    with pytest.raises(FileNotFoundError):
        demo.load_controller(tmp_path / "missing")


def test_load_controller_prefers_the_first_policy_that_exists(tmp_path: Path) -> None:
    from worm.training import train

    train(steps=64, n_envs=1, seed=0, out=tmp_path / "policy")
    search = (tmp_path / "policy_terrain", tmp_path / "policy")
    controller = demo.load_controller(None, search=search)
    assert controller.name == "policy" and controller.obs_version == 1
    assert controller.policy is not None
    explicit = demo.load_controller(tmp_path / "policy")
    assert explicit.name == "policy"


def test_load_controller_reads_obs_version_from_meta_and_defaults_to_1(tmp_path: Path) -> None:
    assert demo.read_obs_version(tmp_path) == 1
    (tmp_path / "meta.json").write_text(json.dumps({"obs_version": 2, "obs_dim": 31}))
    assert demo.read_obs_version(tmp_path) == 2

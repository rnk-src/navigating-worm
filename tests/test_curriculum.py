import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from worm.curriculum import N_LEVELS, WorldCurriculum
from worm.env import WormGoalEnv
from worm.planner import CostConfig
from worm.world import World

START = np.zeros(2)
TARGET = np.array([1.2, 0.9])
SMALL = WorldCurriculum(size=4.0, cell=0.1)


def sample(level: int, seed: int, curriculum: WorldCurriculum = SMALL) -> World:
    return curriculum.sampler(level)(np.random.default_rng(seed), START, TARGET)


def distance_to_leg(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    leg = b - a
    t = np.clip((points - a) @ leg / (leg @ leg), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * leg), axis=-1)


def test_there_are_four_levels() -> None:
    assert N_LEVELS == 4
    with pytest.raises(ValueError, match="level"):
        SMALL.sampler(4)


def test_level0_is_flat_single_class() -> None:
    worlds = [sample(0, seed) for seed in range(12)]
    for world in worlds:
        assert np.all(world.heights == 0.0) and not world.obstacle.any()
        assert len(np.unique(world.surface)) == 1
    assert {int(world.surface[0, 0]) for world in worlds} == {0, 1, 2}


def test_level1_is_flat_mixed() -> None:
    for seed in range(5):
        world = sample(1, seed)
        assert np.all(world.heights == 0.0) and not world.obstacle.any()
        assert len(np.unique(world.surface)) >= 2


def test_level2_slopes_are_bounded() -> None:
    steepest = []
    for seed in range(5):
        world = sample(2, seed)
        assert not world.obstacle.any()
        steepest.append(float(world.slope_deg().max()))
    # The hills stay at or below 10 degrees. The fine roughness adds a little on top.
    assert max(steepest) < 14.0
    assert max(steepest) > 2.0


def test_level3_has_rocks_and_steeper_ground_than_level2() -> None:
    worlds = [sample(3, seed) for seed in range(5)]
    assert all(world.obstacle.any() for world in worlds)
    level2 = [sample(2, seed) for seed in range(5)]
    mean_slope = [np.mean([np.median(w.slope_deg()) for w in ws]) for ws in (level2, worlds)]
    assert mean_slope[1] > mean_slope[0]


def test_level3_ditches_are_shallower_than_the_planner_step_limit() -> None:
    curriculum = WorldCurriculum(size=4.0, cell=0.1)
    assert curriculum.ditch_depth[1] < CostConfig().max_step


def test_corridor_to_target_is_obstacle_free() -> None:
    for seed in range(20):
        world = sample(3, seed)
        near = distance_to_leg(world.cell_xy(), START, TARGET) <= SMALL.corridor
        assert not world.obstacle[near].any()


def test_replay_lower_mixes_levels() -> None:
    curriculum = WorldCurriculum(size=4.0, cell=0.1, replay_lower=0.3)
    curriculum.set_level(2)
    rng = np.random.default_rng(0)
    levels = [curriculum.draw_level(rng) for _ in range(2000)]
    share = {level: levels.count(level) / len(levels) for level in range(4)}
    assert 0.65 < share[2] < 0.75
    assert 0.11 < share[0] < 0.19 and 0.11 < share[1] < 0.19
    assert share[3] == 0.0
    curriculum.set_level(0)
    assert {curriculum.draw_level(rng) for _ in range(50)} == {0}


def test_sampler_keeps_the_world_shape() -> None:
    shapes = {sample(level, seed).shape for level in range(4) for seed in range(3)}
    assert shapes == {(41, 41)}
    assert sample(3, 0, WorldCurriculum()).shape == (161, 161)


def test_same_seed_gives_the_same_world() -> None:
    assert np.array_equal(sample(3, 7).heights, sample(3, 7).heights)


def test_env_samples_the_target_before_the_world() -> None:
    env = WormGoalEnv(world=WorldCurriculum(size=8.0, cell=0.1, level=3, replay_lower=0.0))
    for seed in range(5):
        env.reset(seed=seed)
        world = env.world
        assert world is not None
        near = distance_to_leg(world.cell_xy(), START, env.target) <= 0.4
        assert not world.obstacle[near].any()
        assert env.level == 3


def test_set_level_via_env_method() -> None:
    venv = DummyVecEnv([lambda: WormGoalEnv(world=WorldCurriculum(size=4.0, cell=0.1))] * 2)
    assert venv.env_method("get_level") == [0, 0]
    venv.env_method("set_level", 1)
    assert venv.env_method("get_level") == [1, 1]
    venv.env_method("set_level", 1)
    curricula = [WorldCurriculum(size=4.0, cell=0.1, replay_lower=0.0) for _ in range(2)]
    venv = DummyVecEnv([lambda c=c: WormGoalEnv(world=c) for c in curricula])
    venv.env_method("set_level", 1)
    venv.reset()
    assert all(len(np.unique(world.surface)) >= 2 for world in venv.get_attr("world"))
    with pytest.raises(ValueError, match="curriculum"):
        WormGoalEnv().set_level(1)

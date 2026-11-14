import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from worm import world as wd
from worm.env import WormGoalEnv
from worm.sensors import SensorParams, read_terrain
from worm.terrain import TerrainClass
from worm.world import World

EXACT = SensorParams(class_error=0.0, tof_noise=0.0)
ORIGIN = np.zeros(2)


def step_world(edge: float, height: float = 0.05, cls: TerrainClass = TerrainClass.SAND) -> World:
    """Return a 2 m level world. Where x >= `edge` the ground is raised and of class `cls`."""
    world = wd.flat((2.0, 2.0))
    raised = world.cell_xy()[..., 0] >= edge - 1e-9
    world.heights[raised] = height
    world.surface[raised] = int(cls)
    return world


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_height_ahead_is_zero_on_flat_ground() -> None:
    reading = read_terrain(wd.flat((2.0, 2.0)), ORIGIN, 0.3, EXACT, rng())
    assert np.array_equal(reading["height_ahead"], np.zeros(4))


def test_height_ahead_sees_a_step_in_front() -> None:
    reading = read_terrain(step_world(0.2), ORIGIN, 0.0, EXACT, rng())
    near, far = reading["height_ahead"][:2], reading["height_ahead"][2:]
    assert np.allclose(near, 0.0)
    assert np.allclose(far, 0.5)  # 5 cm over the 10 cm scale


def test_height_ahead_is_in_the_head_frame() -> None:
    # The head points along +y, so the raised ground at x >= 0.05 is on its right.
    reading = read_terrain(step_world(0.05), ORIGIN, math.pi / 2, EXACT, rng())
    right, left = reading["height_ahead"][0::2], reading["height_ahead"][1::2]
    assert np.allclose(right, 0.5)
    assert np.allclose(left, 0.0)


def test_height_ahead_is_clipped_noisy_and_quantized() -> None:
    tall = read_terrain(step_world(0.2, height=0.5), ORIGIN, 0.0, EXACT, rng())
    assert np.allclose(tall["height_ahead"][2:], 1.0)
    noisy = read_terrain(wd.flat((2.0, 2.0)), ORIGIN, 0.0, SensorParams(), rng())["height_ahead"]
    assert np.any(noisy != 0.0) and np.all(np.abs(noisy) < 0.2)
    millimetres = noisy * 0.1 / 0.001
    assert np.allclose(millimetres, np.rint(millimetres))


def test_terrain_ahead_reports_the_class_25cm_ahead() -> None:
    world = step_world(0.2)
    ahead = read_terrain(world, ORIGIN, 0.0, EXACT, rng())["terrain_ahead"]
    assert ahead.tolist() == [0.0, 0.0, 1.0]
    behind = read_terrain(world, ORIGIN, math.pi, EXACT, rng())["terrain_ahead"]
    assert behind.tolist() == [1.0, 0.0, 0.0]


def test_class_error_rate_is_about_5_percent() -> None:
    world, generator = wd.flat((2.0, 2.0), cls=TerrainClass.GRASS), rng()
    reads = [
        int(np.argmax(read_terrain(world, ORIGIN, 0.0, SensorParams(), generator)["terrain_ahead"]))
        for _ in range(2000)
    ]
    wrong = sum(cls != TerrainClass.GRASS for cls in reads)
    assert 0.03 * 2000 <= wrong <= 0.07 * 2000
    assert set(reads) == {0, 1, 2}


def test_no_world_reads_the_default_class_and_level_ground() -> None:
    reading = read_terrain(None, ORIGIN, 0.0, EXACT, rng(), default=TerrainClass.GRASS)
    assert reading["terrain_ahead"].tolist() == [0.0, 1.0, 0.0]
    assert np.array_equal(reading["height_ahead"], np.zeros(4))


def test_v2_env_passes_the_checker() -> None:
    check_env(WormGoalEnv(world=step_world(0.2), obs_version=2), skip_render_check=True)
    check_env(WormGoalEnv(obs_version=2), skip_render_check=True)


def test_v2_env_observes_the_ground_ahead() -> None:
    env = WormGoalEnv(world=step_world(0.3, height=0.03), obs_version=2)
    obs, _ = env.reset(seed=0, options={"yaw": 0.0})
    assert obs.shape == (31,)
    slices = env.layout.slices
    assert obs[slices["terrain_ahead"]].sum() == 1.0
    heights = obs[slices["height_ahead"]]
    assert np.all(np.abs(heights[:2]) < 0.12)  # level ground 15 cm ahead, sensor noise only
    assert np.all(np.abs(heights[2:] - 0.3) < 0.12)  # the 3 cm step 30 cm ahead


def test_v2_flat_env_reports_level_ground() -> None:
    env = WormGoalEnv(obs_version=2, terrain=TerrainClass.SAND)
    obs, _ = env.reset(seed=0)
    assert np.array_equal(obs[env.layout.slices["height_ahead"]], np.zeros(4))
    assert obs[env.layout.slices["terrain_ahead"]].sum() == 1.0


def test_v1_env_still_24d_in_a_world() -> None:
    env = WormGoalEnv(world=step_world(0.2))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (24,) and env.observation_space.shape == (24,)

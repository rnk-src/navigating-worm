import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from worm import terrain
from worm import world as wd
from worm.env import WormGoalEnv
from worm.randomize import DomainRandomizer
from worm.terrain import TerrainClass

IDLE = np.array([-1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # zero amplitude
GO = np.array([0.8, 0.2, 0, 0, 0, 0, 0], dtype=np.float32)


def nominal_env(**kwargs) -> WormGoalEnv:  # type: ignore[no-untyped-def]
    return WormGoalEnv(randomizer=DomainRandomizer(0.0), **kwargs)


def test_spaces_match_the_spec() -> None:
    env = WormGoalEnv()
    assert env.observation_space.shape == (24,)
    assert env.action_space.shape == (7,)
    assert (env.action_space.low, env.action_space.high) == (pytest.approx(-1), pytest.approx(1))


def test_env_passes_gymnasium_checker() -> None:
    check_env(WormGoalEnv(), skip_render_check=True)


def test_same_seed_gives_the_same_first_observation() -> None:
    env = WormGoalEnv()
    first, _ = env.reset(seed=3)
    second, _ = env.reset(seed=3)
    other, _ = env.reset(seed=4)
    assert np.array_equal(first, second) and not np.array_equal(first, other)


def test_default_targets_are_1_to_3_m_away() -> None:
    env = WormGoalEnv()
    distances = [env.reset(seed=s)[1]["distance"] for s in range(30)]
    assert 1.0 <= min(distances) < 1.5 and 2.5 < max(distances) <= 3.0


def test_options_inject_the_target() -> None:
    env = nominal_env()
    _, info = env.reset(seed=0, options={"target": (3.0, 4.0)})
    assert info["distance"] == pytest.approx(5.0, abs=1e-6)


def test_zero_amplitude_action_keeps_the_worm_still() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    for _ in range(40):
        _, _, _, _, info = env.step(IDLE)
    assert np.linalg.norm(info["head_xy"]) < 0.005


def test_gait_action_moves_the_head_more_than_10cm_in_5s() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    for _ in range(100):
        _, _, _, _, info = env.step(GO)
    assert np.linalg.norm(info["head_xy"]) > 0.10


def test_episode_terminates_with_success_inside_15cm() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (0.14, 0.0)})
    _, value, terminated, truncated, info = env.step(IDLE)
    assert terminated and not truncated
    assert info["reached"] and info["is_success"]
    assert value > 19.0  # the success bonus


def test_episode_continues_outside_the_tolerance() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (0.17, 0.0)})
    _, _, terminated, truncated, info = env.step(IDLE)
    assert not terminated and not truncated and not info["reached"]


def test_episode_truncates_after_the_time_limit() -> None:
    env = nominal_env(episode_seconds=0.5)
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    flags = [env.step(IDLE)[2:4] for _ in range(10)]
    assert flags[:9] == [(False, False)] * 9
    assert flags[9] == (False, True)


def test_set_target_moves_the_goal_and_restarts_the_clock_without_moving_the_body() -> None:
    env = nominal_env(episode_seconds=0.5)
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    for _ in range(9):
        env.step(GO)
    before = env.data.qpos.copy()
    env.set_target(np.array([0.0, 3.0]))
    assert np.array_equal(env.data.qpos, before)
    _, _, _, truncated, info = env.step(IDLE)
    assert not truncated  # step 10 would have truncated without the clock restart
    assert info["distance"] == pytest.approx(3.0, abs=0.1)


def test_rolling_onto_its_side_terminates_with_a_penalty() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    env.data.qpos[2] = 0.3
    env.data.qpos[3:7] = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)  # 90 deg roll, in the air
    _, value, terminated, _, info = env.step(IDLE)
    assert terminated and not info["reached"]
    assert value < -10.0


def obs_part(env: WormGoalEnv, obs: np.ndarray, name: str) -> np.ndarray:
    return obs[env.layout.slices[name]]


def test_observation_reports_the_previous_action() -> None:
    env = nominal_env()
    obs, _ = env.reset(seed=0, options={"target": (5.0, 0.0)})
    assert np.array_equal(obs_part(env, obs, "prev_action"), np.zeros(7))
    obs, *_ = env.step(GO)
    assert obs_part(env, obs, "prev_action") == pytest.approx(GO)


def test_observation_reports_joint_motion_since_the_last_step() -> None:
    env = nominal_env()
    obs, _ = env.reset(seed=0, options={"target": (5.0, 0.0)})
    for _ in range(5):
        previous = obs
        obs, *_ = env.step(GO)
    delta = obs_part(env, obs, "joint_pos") - obs_part(env, previous, "joint_pos")
    assert np.abs(delta).max() > 0.01
    assert obs_part(env, obs, "joint_delta") == pytest.approx(delta, abs=1e-6)


@pytest.mark.parametrize(
    ("terrain", "one_hot"),
    [
        (TerrainClass.CONCRETE, [1, 0, 0]),
        (TerrainClass.GRASS, [0, 1, 0]),
        (TerrainClass.SAND, [0, 0, 1]),
    ],
)
def test_observation_carries_the_terrain_one_hot(terrain: TerrainClass, one_hot: list[int]) -> None:
    env = WormGoalEnv(terrain=terrain)
    obs, _ = env.reset(seed=0)
    assert obs_part(env, obs, "terrain").tolist() == one_hot


def test_terrain_class_sets_the_friction_range() -> None:
    env = WormGoalEnv(terrain=TerrainClass.SAND)
    forward = []
    for seed in range(20):
        env.reset(seed=seed)
        forward.append(env.params.friction.mu_forward)
    assert 0.4 <= min(forward) and max(forward) <= 0.8


def test_randomized_mass_and_servo_strength_reach_the_physics_model() -> None:
    env = WormGoalEnv()
    env.reset(seed=5)
    p = env.params
    assert p.mass_scale != 1.0 and p.servo.strength != 1.0
    assert env.model.body_mass[1:] == pytest.approx(0.04 * p.mass_scale)
    assert env.model.actuator_forcerange[:, 1] == pytest.approx(0.225 * p.servo.strength)
    assert env.model.actuator_gainprm[:, 0] == pytest.approx(3.0 * p.servo.strength)
    assert env.model.actuator_biasprm[:, 1] == pytest.approx(-3.0 * p.servo.strength)


def test_mass_scale_does_not_compound_across_resets() -> None:
    env = WormGoalEnv()
    for seed in range(5):
        env.reset(seed=seed)
    assert env.model.body_mass[1] == pytest.approx(0.04 * env.params.mass_scale)


def test_more_forward_friction_slows_the_worm() -> None:
    def travel(terrain: TerrainClass) -> float:
        env = WormGoalEnv(terrain=terrain)
        env.reset(seed=1, options={"target": (50.0, 0.0)})
        start = env.data.xpos[1, :2].copy()
        for _ in range(100):
            env.step(GO)
        return float(np.linalg.norm(env.data.xpos[1, :2] - start))

    assert travel(TerrainClass.CONCRETE) > travel(TerrainClass.SAND)


def test_render_returns_nothing_without_a_render_mode() -> None:
    assert nominal_env().render() is None


def test_render_returns_an_rgb_frame_in_rgb_array_mode(rendering: None) -> None:
    env = nominal_env(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None and frame.shape == (480, 640, 3)
    assert frame.std() > 0  # not a blank image
    env.close()


def green_centroid(frame: np.ndarray) -> np.ndarray:
    """Pixel (row, col) centre of the green worm body."""
    r, g, b = (frame[..., i].astype(int) for i in range(3))
    rows, cols = np.nonzero((g > r + 40) & (g > b + 40))
    assert len(rows) > 0, "the worm is not visible"
    return np.array([rows.mean(), cols.mean()])


def frames_before_and_after_moving(env: WormGoalEnv) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    before = env.render()
    for _ in range(100):
        env.step(GO)
    after = env.render()
    assert before is not None and after is not None
    return before, after


def test_follow_camera_keeps_the_worm_in_place_on_screen(rendering: None) -> None:
    before, after = frames_before_and_after_moving(nominal_env(render_mode="rgb_array"))
    assert np.linalg.norm(green_centroid(after) - green_centroid(before)) < 40


def test_overview_camera_stays_fixed_so_the_worm_moves_across_the_screen(rendering: None) -> None:
    env = nominal_env(render_mode="rgb_array")
    env.set_overview(center=(0.0, 0.0), half_extent=1.0)
    before, after = frames_before_and_after_moving(env)
    assert np.linalg.norm(green_centroid(after) - green_centroid(before)) > 40


def test_markers_are_drawn_in_their_own_colour(rendering: None) -> None:
    def blue_pixels(frame: np.ndarray) -> int:
        r, g, b = (frame[..., i].astype(int) for i in range(3))
        return int(((b > r + 60) & (b > g + 60)).sum())

    env = nominal_env(render_mode="rgb_array")
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    plain = env.render()
    env.markers = [((0.2, 0.2), (0.0, 0.0, 1.0, 1.0), 0.04)]
    marked = env.render()
    assert plain is not None and marked is not None
    assert blue_pixels(plain) == 0 and blue_pixels(marked) > 50


def test_target_marker_is_at_the_target_immediately_after_reset_and_set_target() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (2.0, -1.0)})
    assert env.data.site("target").xpos[:2] == pytest.approx([2.0, -1.0])
    env.set_target(np.array([0.5, 0.7]))
    assert env.data.site("target").xpos[:2] == pytest.approx([0.5, 0.7])


def test_set_target_returns_an_observation_of_the_new_target() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (0.3, 0.0)})
    obs = env.set_target(np.array([0.0, 2.4]))
    assert obs_part(env, obs, "distance")[0] * 3.0 == pytest.approx(2.4, abs=0.15)


def test_set_target_does_not_disturb_the_joint_motion_observation() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    for _ in range(5):
        obs, *_ = env.step(GO)
    delta = obs_part(env, obs, "joint_delta").copy()
    assert np.abs(delta).max() > 0.01
    switched = env.set_target(np.array([0.0, 2.0]))
    assert obs_part(env, switched, "joint_delta") == pytest.approx(delta)


def test_head_xy_matches_the_step_info() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"target": (5.0, 0.0)})
    for _ in range(20):
        _, _, _, _, info = env.step(GO)
    assert np.linalg.norm(env.head_xy) > 0.01
    assert np.array_equal(env.head_xy, info["head_xy"])


def test_render_fps_follows_the_control_rate() -> None:
    from worm.spec import WormSpec

    assert WormGoalEnv().metadata["render_fps"] == 20
    assert WormGoalEnv(spec=WormSpec(control_dt=0.1)).metadata["render_fps"] == 10


def test_observed_terrain_matches_the_sampled_class_when_none_is_forced() -> None:
    env = WormGoalEnv()
    seen = set()
    for seed in range(12):
        obs, _ = env.reset(seed=seed)
        hot = obs_part(env, obs, "terrain")
        assert hot.sum() == 1
        assert int(hot.argmax()) == int(env.params.terrain)
        seen.add(int(hot.argmax()))
    assert seen == {0, 1, 2}


def test_mass_scale_also_scales_inertia() -> None:
    env = WormGoalEnv()
    nominal = env.model.body_inertia.copy()
    env.reset(seed=1)
    scale = env.params.mass_scale
    assert scale != 1.0
    assert env.model.body_inertia[1:] == pytest.approx(nominal[1:] * scale)


# Recorded on the flat-plane env before the terrain refactor. Each row holds the first
# observation's sensor block (yaw rate, roll, pitch, bearing sin and cos, distance),
# the terrain class, and the head position after 40 GO steps.
_GOLDEN = [
    ([-0.00369, -0.012459, -0.007323, 0.387525, 0.921859, 0.901773], 2, [-0.007708, 0.007095]),
    ([0.000807, -0.004821, 0.005988, -0.068435, -0.997656, 0.55023], 1, [-0.041699, -0.014192]),
    ([-0.002044, 0.00455, -0.000992, 0.019046, -0.999819, 0.615682], 2, [0.022056, 0.086243]),
]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flat_env_matches_the_m1_golden_trajectory(seed: int) -> None:
    sensors, terrain_class, head = _GOLDEN[seed]
    env = WormGoalEnv()
    obs, _ = env.reset(seed=seed)
    assert obs[:8] == pytest.approx(np.zeros(8), abs=1e-6)
    assert obs[8:14] == pytest.approx(sensors, abs=2e-6)
    assert int(np.argmax(obs[21:24])) == terrain_class
    for _ in range(40):
        _, _, _, _, info = env.step(GO)
    assert info["head_xy"] == pytest.approx(head, abs=2e-6)


# --- heightfield worlds ---


def split_world(size: float = 4.0) -> wd.World:
    """Return a flat world: concrete at -x, sand at +x."""
    base = wd.flat((size, size))
    surface = base.surface.copy()
    surface[:, base.shape[1] // 2 :] = TerrainClass.SAND
    return wd.World(base.heights, surface, base.obstacle, base.cell)


def hilly_world(seed: int = 0) -> wd.World:
    return wd.generate(wd.WorldConfig(size=(4.0, 4.0), n_rocks=0, n_ditches=0), seed)


def test_world_env_passes_the_gymnasium_checker() -> None:
    check_env(WormGoalEnv(world=hilly_world()), skip_render_check=True)


def test_world_env_spawns_on_the_terrain_surface() -> None:
    world = hilly_world()
    env = WormGoalEnv(world=world, randomizer=DomainRandomizer(0.0))
    for start in [(0.0, 0.0), (1.2, -0.9), (-1.0, 1.1)]:
        env.reset(seed=0, options={"start": start, "yaw": 0.3})
        head = env.data.xpos[1]
        assert head[:2] == pytest.approx(start, abs=0.03)
        assert head[2] == pytest.approx(float(world.height_at(head[:2])) + 0.0125, abs=0.005)
        assert np.abs(env.data.qvel[:3]).max() < 0.05  # it is not falling or sliding


def test_reset_start_option_places_the_head() -> None:
    for env in (nominal_env(), nominal_env(world=wd.flat((4.0, 4.0)))):
        _, info = env.reset(seed=0, options={"start": (1.0, -0.5), "yaw": math.pi / 2})
        assert info["head_xy"] == pytest.approx((1.0, -0.5), abs=0.01)
        assert env.data.xpos[2, :2] == pytest.approx((1.0, -0.56), abs=0.01)  # tail toward -y


def test_default_target_is_1_to_3_m_from_the_start() -> None:
    env = nominal_env()
    distances = [env.reset(seed=s, options={"start": (7.0, 7.0)})[1]["distance"] for s in range(20)]
    assert 1.0 <= min(distances) and max(distances) <= 3.0
    assert env.head_xy == pytest.approx((7.0, 7.0), abs=1e-6)


def test_world_targets_lie_inside_the_world_and_off_the_rocks() -> None:
    world = wd.generate(wd.WorldConfig(size=(4.0, 4.0), n_rocks=8, n_ditches=0), 1)
    env = WormGoalEnv(world=world)
    for seed in range(30):
        env.reset(seed=seed)
        assert world.contains(env.target)
        row, col = world.to_cell(env.target)
        assert not world.obstacle[row, col]


def test_terrain_one_hot_is_the_class_under_the_head_in_a_world() -> None:
    env = WormGoalEnv(world=split_world(), randomizer=DomainRandomizer(0.0))
    obs, _ = env.reset(seed=0, options={"start": (-1.0, 0.0)})
    assert obs_part(env, obs, "terrain").tolist() == [1, 0, 0]
    assert env.surface_under_head == TerrainClass.CONCRETE
    obs, _ = env.reset(seed=0, options={"start": (1.0, 0.0)})
    assert obs_part(env, obs, "terrain").tolist() == [0, 0, 1]
    assert env.surface_under_head == TerrainClass.SAND


def test_flat_env_reports_the_episode_class_under_the_head() -> None:
    env = WormGoalEnv(terrain=TerrainClass.GRASS)
    env.reset(seed=0)
    assert env.surface_under_head == TerrainClass.GRASS and env.world is None


def test_world_sampler_gives_a_new_world_each_reset_without_recompiling() -> None:
    def sampler(rng: np.random.Generator) -> wd.World:
        return hilly_world(int(rng.integers(1_000_000)))

    env = WormGoalEnv(world=sampler)
    compiled = env.model
    env.reset(seed=0)
    first_world, first = env.world, env.model.hfield_data.copy()
    env.reset()
    assert env.model is compiled
    assert env.world is not first_world
    assert not np.array_equal(env.model.hfield_data, first)
    env.reset(seed=0)
    assert np.array_equal(env.model.hfield_data, first)  # the same seed gives the same world


def test_reset_world_option_replaces_the_world() -> None:
    env = WormGoalEnv(world=wd.flat((4.0, 4.0)), randomizer=DomainRandomizer(0.0))
    obs, _ = env.reset(seed=0, options={"world": split_world(), "start": (1.0, 0.0)})
    assert obs_part(env, obs, "terrain").tolist() == [0, 0, 1]


def test_per_episode_surface_table_is_sampled_per_class() -> None:
    env = WormGoalEnv(world=split_world(), randomizer=DomainRandomizer(0.0))
    env.reset(seed=0)
    assert env.params.surfaces is not None
    assert env.params.surfaces.props == tuple(terrain.nominal_surface(c) for c in TerrainClass)

    env = WormGoalEnv(world=split_world())
    env.reset(seed=0)
    first = env.params.surfaces
    env.reset(seed=1)
    assert first is not None and env.params.surfaces is not None
    assert first != env.params.surfaces
    sand = [p.friction.mu_forward for p in (first.props[2], env.params.surfaces.props[2])]
    assert all(0.4 <= mu <= 0.8 for mu in sand)


def test_sand_under_the_worm_slows_it_in_a_world() -> None:
    def travel(start_x: float) -> float:
        env = WormGoalEnv(world=split_world(), randomizer=DomainRandomizer(0.0))
        env.reset(seed=0, options={"start": (start_x, 0.0), "yaw": math.pi / 2, "target": (0, 9)})
        start = env.head_xy
        for _ in range(100):
            env.step(GO)
        return float(np.linalg.norm(env.head_xy - start))

    assert travel(-1.0) > 1.2 * travel(1.0) > 0.05


def test_leaving_the_world_truncates_the_episode() -> None:
    env = WormGoalEnv(world=wd.flat((4.0, 4.0)), randomizer=DomainRandomizer(0.0))
    env.reset(seed=0, options={"start": (1.6, 0.0), "yaw": 0.0, "target": (1.9, 0.0)})
    flags = []
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(GO)
        flags.append((terminated, truncated, info["out_of_world"]))
        if terminated or truncated:
            break
    assert flags[-1] == (False, True, True)
    assert env.head_xy[0] > 1.7  # within 0.3 m of the edge at x = 2


def test_world_env_renders_the_terrain(rendering: None) -> None:
    env = WormGoalEnv(world=hilly_world(), render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None and frame.std() > 0
    env.close()

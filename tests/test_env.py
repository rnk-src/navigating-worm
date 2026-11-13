import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

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
    [(None, [1, 0, 0]), (TerrainClass.GRASS, [0, 1, 0]), (TerrainClass.SAND, [0, 0, 1])],
)
def test_observation_carries_the_terrain_one_hot(
    terrain: TerrainClass | None, one_hot: list[int]
) -> None:
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

import numpy as np
import pytest
from numpy.typing import NDArray

from worm.env import WormGoalEnv
from worm.randomize import DomainRandomizer
from worm.route import make_waypoints, run_route


def steer(obs: NDArray[np.float32]) -> NDArray[np.float32]:
    """Hand-written controller: steady gait, turn bias proportional to the bearing."""
    sin_b, cos_b = obs[11:13]
    action = np.zeros(7, dtype=np.float32)
    action[0], action[1] = 0.8, 0.2
    action[2] = np.clip(1.0 * np.arctan2(sin_b, cos_b), -1.0, 1.0)
    return action


def idle(obs: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([-1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def nominal_env(seconds: float) -> WormGoalEnv:
    return WormGoalEnv(randomizer=DomainRandomizer(0.0), episode_seconds=seconds)


def test_waypoints_are_evenly_spaced_and_cover_the_length() -> None:
    points = make_waypoints(10.0, 2.0, np.random.default_rng(0))
    assert points.shape == (5, 2)
    gaps = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], points]), axis=0), axis=1)
    assert gaps == pytest.approx([2.0] * 5)


def test_waypoints_never_turn_more_than_30_degrees() -> None:
    points = np.vstack([[0.0, 0.0], make_waypoints(100.0, 2.0, np.random.default_rng(1))])
    legs = np.diff(points, axis=0)
    headings = np.unwrap(np.arctan2(legs[:, 1], legs[:, 0]))
    turns = np.abs(np.diff(headings))
    assert 0.0 < turns.max() <= np.radians(30.0) + 1e-9


def test_route_visits_every_waypoint_in_order_without_resetting_the_body() -> None:
    env = nominal_env(60.0)
    waypoints = np.array([[0.5, 0.0], [1.0, 0.3], [1.5, 0.3]])
    result = run_route(steer, env, waypoints, seed=0)
    assert result.success and (result.reached, result.total) == (3, 3)
    # The head ends at the last waypoint, so the body was never put back at the origin.
    assert np.linalg.norm(result.trajectory[-1] - waypoints[-1]) < 0.15
    steps_between = np.linalg.norm(np.diff(result.trajectory, axis=0), axis=1)
    assert steps_between.max() < 0.05  # no teleport between waypoints
    assert result.sim_seconds == pytest.approx(len(result.trajectory) * 0.05)


def test_route_fails_when_a_waypoint_times_out() -> None:
    result = run_route(idle, nominal_env(0.5), np.array([[5.0, 0.0], [6.0, 0.0]]), seed=0)
    assert not result.success
    assert (result.reached, result.total) == (0, 2)
    assert result.sim_seconds == pytest.approx(0.5)


def test_each_waypoint_gets_its_own_time_budget() -> None:
    # Three legs need more than one 20 s budget in total, but each leg fits inside 20 s.
    env = nominal_env(20.0)
    waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    result = run_route(steer, env, waypoints, seed=0)
    assert result.success and result.sim_seconds > 20.0


def test_on_step_callback_runs_once_per_control_step() -> None:
    calls = []
    result = run_route(
        idle, nominal_env(0.5), np.array([[5.0, 0.0]]), seed=0, on_step=lambda: calls.append(1)
    )
    assert len(calls) == len(result.trajectory) == 10


def test_showcase_route_heads_in_all_four_compass_directions_with_both_turn_senses() -> None:
    from worm.route import showcase_waypoints

    points = np.vstack([[0.0, 0.0], showcase_waypoints()])
    legs = np.diff(points, axis=0)
    headings = np.degrees(np.arctan2(legs[:, 1], legs[:, 0]))
    for compass in (0.0, 90.0, 180.0, -90.0):
        gap = np.abs((headings - compass + 180.0) % 360.0 - 180.0)
        assert gap.min() <= 45.0, f"no leg heads within 45 deg of {compass}"
    turns = (np.diff(headings) + 180.0) % 360.0 - 180.0
    assert turns.max() >= 90.0 and turns.min() <= -90.0  # a sharp left and a sharp right
    lengths = np.linalg.norm(legs, axis=1)
    assert lengths.min() >= 0.8 and lengths.max() <= 2.0  # inside the trained target range

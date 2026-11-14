"""Step-wise route following and navigation with replanning."""

import numpy as np
import pytest
from numpy.typing import NDArray

from worm import world as wd
from worm.controllers import bearing_steer
from worm.env import WormGoalEnv
from worm.navigate import RouteFollower
from worm.randomize import DomainRandomizer
from worm.spec import WormSpec, obs_layout

STEER = bearing_steer(obs_layout(WormSpec()))


def idle(obs: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([-1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def nominal_env(seconds: float = 60.0, **kwargs) -> WormGoalEnv:  # type: ignore[no-untyped-def]
    return WormGoalEnv(randomizer=DomainRandomizer(0.0), episode_seconds=seconds, **kwargs)


def drive(follower: RouteFollower, env: WormGoalEnv, policy=STEER) -> list:  # type: ignore[no-untyped-def]
    """Step until the follower is done. Return the reports that carry an event."""
    events = []
    for _ in range(5000):
        report = follower.step(env, policy)
        if report.event:
            events.append(report)
        if follower.done:
            return events
    raise AssertionError("the follower never finished")


def test_follower_uses_pass_tolerance_for_intermediate_and_env_tolerance_for_goal() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"yaw": 0.0})
    waypoints = np.array([[0.6, 0.0], [1.2, 0.0]])
    follower = RouteFollower(waypoints, pass_tolerance=0.3)
    follower.start(env)
    passed, arrived = drive(follower, env)
    assert (passed.event, arrived.event) == ("waypoint", "goal")
    assert 0.15 < np.linalg.norm(passed.head_xy - waypoints[0]) < 0.3
    assert np.linalg.norm(arrived.head_xy - waypoints[1]) < 0.15
    assert follower.reached == 2 and follower.done


def test_follower_without_pass_tolerance_uses_the_env_tolerance_everywhere() -> None:
    env = nominal_env()
    env.reset(seed=0, options={"yaw": 0.0})
    waypoints = np.array([[0.6, 0.0], [1.2, 0.0]])
    follower = RouteFollower(waypoints, pass_tolerance=None)
    follower.start(env)
    passed, _ = drive(follower, env)
    assert np.linalg.norm(passed.head_xy - waypoints[0]) < 0.15


def test_follower_reports_events() -> None:
    env = nominal_env(seconds=0.5)
    env.reset(seed=0)
    follower = RouteFollower(np.array([[5.0, 0.0]]))
    follower.start(env)
    (timeout,) = drive(follower, env, idle)
    assert timeout.event == "timeout" and follower.reached == 0
    assert timeout.waypoint_index == 0 and timeout.obs.shape == (24,)

    env = nominal_env()
    env.reset(seed=0)
    follower = RouteFollower(np.array([[5.0, 0.0]]))
    follower.start(env)
    env.data.qpos[2] = 0.3
    env.data.qpos[3:7] = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)  # on its side, in the air
    assert follower.step(env, idle).event == "flipped" and follower.done

    env = nominal_env(world=wd.flat((4.0, 4.0)))
    env.reset(seed=0, options={"start": (1.6, 0.0), "yaw": 0.0})
    follower = RouteFollower(np.array([[1.95, 0.0]]))
    follower.start(env)
    assert drive(follower, env)[-1].event == "out_of_world"


def test_follower_does_not_move_the_body_when_it_starts() -> None:
    env = nominal_env()
    env.reset(seed=0)
    before = env.data.qpos.copy()
    follower = RouteFollower(np.array([[1.0, 1.0]]))
    obs = follower.start(env)
    assert np.array_equal(env.data.qpos, before)
    assert env.target == pytest.approx((1.0, 1.0)) and obs.shape == (24,)

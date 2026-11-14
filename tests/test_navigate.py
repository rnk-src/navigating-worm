"""Step-wise route following and navigation with replanning."""

import numpy as np
import pytest
from numpy.typing import NDArray

from worm import planner
from worm import world as wd
from worm.controllers import Policy, bearing_steer
from worm.env import WormGoalEnv
from worm.navigate import NavConfig, Navigator, NavResult, RouteFollower, navigate
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


# --- Navigator: plan, drive, replan ---


def walled_world() -> wd.World:
    """Return a flat 4 m world with a wall at x = 0.4..0.9 that leaves a gap at y > 0.6."""
    base = wd.flat((4.0, 4.0))
    xy = base.cell_xy()
    wall = (xy[..., 0] >= 0.4) & (xy[..., 0] <= 0.9) & (xy[..., 1] <= 0.6)
    heights = np.where(wall, 0.1, 0.0).astype(np.float32)
    return wd.World(heights, base.surface, wall, base.cell)


START, BEHIND_WALL = (-0.6, -0.6), (1.4, -0.4)


def walled_navigator(
    policy: Policy = STEER, cfg: NavConfig | None = None
) -> tuple[WormGoalEnv, Navigator]:
    world = walled_world()
    env = nominal_env(world=world)
    env.reset(seed=0, options={"start": START, "yaw": 1.2})
    return env, Navigator(env, policy, planner.build_costmap(world), cfg or NavConfig())


def run(nav: Navigator, limit: int = 6000) -> None:
    for _ in range(limit):
        if nav.status != "driving":
            return
        nav.step()
    raise AssertionError("the navigator never finished")


def test_navigator_reaches_a_goal_behind_a_wall_via_the_gap() -> None:
    env, nav = walled_navigator()
    assert nav.status == "idle" and nav.plan is None
    result = nav.go(BEHIND_WALL)
    assert isinstance(result, planner.Plan) and nav.status == "driving"
    run(nav)
    assert nav.status == "reached" and nav.failure is None
    assert np.linalg.norm(env.head_xy - BEHIND_WALL) < 0.15
    crossing = nav.trail[(nav.trail[:, 0] > 0.4) & (nav.trail[:, 0] < 0.9)]
    assert len(crossing) > 0 and crossing[:, 1].min() > 0.6  # through the gap, not the wall


def test_navigator_does_not_reset_the_body_between_goals() -> None:
    env, nav = walled_navigator()
    nav.go((-0.6, 0.6))
    run(nav)
    first_leg = len(nav.trail)
    assert nav.status == "reached"
    nav.go((-1.2, 0.0))
    run(nav)
    assert nav.status == "reached" and len(nav.trail) > first_leg
    assert np.linalg.norm(np.diff(nav.trail, axis=0), axis=1).max() < 0.05  # no teleport


def test_navigator_replans_when_stuck() -> None:
    steps = {"n": 0}

    def sleepy(obs: NDArray[np.float32]) -> NDArray[np.float32]:
        steps["n"] += 1
        return idle(obs) if steps["n"] <= 100 else STEER(obs)  # idle for 5 s

    _, nav = walled_navigator(sleepy, NavConfig(stuck_seconds=3.0))
    nav.go((-0.6, 0.6))
    run(nav)
    assert nav.replans >= 1 and nav.status == "reached"


def test_navigator_replans_when_pushed_off_path() -> None:
    env, nav = walled_navigator()
    nav.go((-0.6, 1.2))  # straight along +y
    for _ in range(100):
        nav.step()
    env.data.qpos[0] -= 0.8  # push the worm 0.8 m sideways
    run(nav)
    assert nav.replans >= 1 and nav.status == "reached"


def test_navigator_gives_up_after_max_replans() -> None:
    _, nav = walled_navigator(idle, NavConfig(stuck_seconds=1.0, max_replans=2))
    nav.go((-0.6, 0.6))
    run(nav)
    assert nav.status == "failed" and nav.failure == "stuck" and nav.replans == 2


def test_navigator_keeps_driving_when_a_new_goal_is_infeasible() -> None:
    _, nav = walled_navigator()
    plan = nav.go((-0.6, 0.6))
    blocked = nav.go((0.65, -0.5))  # inside the wall
    assert isinstance(blocked, planner.Infeasible)
    assert blocked.reason == planner.Reason.GOAL_BLOCKED
    assert nav.status == "driving" and nav.plan is plan


def test_navigate_returns_infeasible_for_a_goal_inside_a_rock() -> None:
    world = walled_world()
    result = navigate(
        nominal_env(world=world), STEER, planner.build_costmap(world), START, (0.65, -0.5)
    )
    assert isinstance(result, planner.Infeasible)


def test_nav_result_accounts_time_per_terrain_class() -> None:
    base = wd.flat((4.0, 4.0))
    surface = base.surface.copy()
    surface[:, 41:] = 2  # sand at +x
    world = wd.World(base.heights, surface, base.obstacle, base.cell)
    result = navigate(
        nominal_env(world=world),
        STEER,
        planner.build_costmap(world),
        (-0.5, 0.0),
        (0.6, 0.0),
        seed=0,
    )
    assert isinstance(result, NavResult) and result.success and result.reason == "reached"
    concrete, grass, sand = result.class_seconds
    assert concrete > 1.0 and sand > 1.0 and grass == 0.0
    assert sum(result.class_seconds) == pytest.approx(result.sim_seconds)
    assert result.sim_seconds == pytest.approx(len(result.trajectory) * 0.05)
    assert result.replans == 0

"""Drive a route one control step at a time, and replan when the worm is stuck."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from worm import planner
from worm.controllers import Policy
from worm.env import WormGoalEnv
from worm.planner import CostMap, Infeasible, Plan
from worm.terrain import TerrainClass

Array = NDArray[np.float64]
Event = Literal["", "waypoint", "goal", "timeout", "flipped", "out_of_world"]


@dataclass(frozen=True, eq=False)
class StepReport:
    obs: NDArray[np.float32]
    head_xy: Array
    waypoint_index: int  # the waypoint that is the target after this step
    event: Event  # "" on an ordinary step


@dataclass
class RouteFollower:
    """Follow waypoints in order without resetting the body. Call `start`, then `step`.

    An intermediate waypoint counts as passed within `pass_tolerance`. The last one
    needs the env's own tolerance. `pass_tolerance=None` uses the env's everywhere.
    The per-waypoint timeout is the env's episode length.
    """

    waypoints: Array
    pass_tolerance: float | None = 0.3  # m
    _index: int = field(default=0, init=False)
    _done: bool = field(default=False, init=False)
    _obs: NDArray[np.float32] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=np.float64).reshape(-1, 2)
        if len(self.waypoints) == 0:
            raise ValueError("the route has no waypoints")

    def start(
        self, env: WormGoalEnv, obs: NDArray[np.float32] | None = None
    ) -> NDArray[np.float32]:
        """Aim the env at the first waypoint and return the observation.

        Pass `obs` when the env already targets the first waypoint, straight after a
        reset with that target. The follower then uses it and leaves the env alone.
        """
        self._index, self._done = 0, False
        self._obs = env.set_target(self.waypoints[0]) if obs is None else obs
        return self._obs

    @property
    def done(self) -> bool:
        return self._done

    @property
    def reached(self) -> int:
        """Return the number of waypoints reached so far."""
        return self._index

    def step(self, env: WormGoalEnv, policy: Policy) -> StepReport:
        """Advance one control step."""
        if self._obs is None or self._done:
            raise RuntimeError("call start() before step(), and stop when done")
        obs, _, terminated, truncated, info = env.step(policy(self._obs))
        last = self._index == len(self.waypoints) - 1
        arrived = bool(info["reached"])
        if not last and self.pass_tolerance is not None:
            arrived = arrived or info["distance"] < self.pass_tolerance
        event: Event = ""
        if arrived:
            self._index += 1
            if last:
                event, self._done = "goal", True
            else:
                event = "waypoint"
                obs = env.set_target(self.waypoints[self._index])
        elif terminated:
            event, self._done = "flipped", True
        elif truncated:
            event, self._done = ("out_of_world" if info["out_of_world"] else "timeout"), True
        self._obs = obs
        index = min(self._index, len(self.waypoints) - 1)
        return StepReport(obs, info["head_xy"], index, event)


@dataclass(frozen=True)
class NavConfig:
    """When the navigator replans and when it gives up. Times in sim seconds, lengths in m."""

    stuck_seconds: float = 15.0  # replan after this long without progress
    stuck_progress: float = 0.05  # progress means getting this much closer to the waypoint
    off_path: float = 0.6  # replan when the head is this far from the current leg
    waypoint_timeout: float = 60.0  # replan when one waypoint takes longer
    max_replans: int = 5  # per goal; one more failure ends the drive as "stuck"
    max_seconds: float = 1200.0  # per goal


Status = Literal["idle", "driving", "reached", "failed"]


def _distance_to_leg(point: Array, a: Array, b: Array) -> float:
    """Return the distance from `point` to the segment a-b."""
    leg = b - a
    length2 = float(leg @ leg)
    t = 0.0 if length2 == 0.0 else float(np.clip((point - a) @ leg / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * leg)))


class Navigator:
    """Plan a route to a goal and drive it, replanning from where the worm is.

    `go` plans from the current head position and never resets the body. `step`
    advances one control step. It replans when the worm makes no progress, strays
    from the path or takes too long over a waypoint.
    """

    def __init__(
        self, env: WormGoalEnv, policy: Policy, costmap: CostMap, cfg: NavConfig = NavConfig()
    ) -> None:
        self._env, self._policy, self._costmap, self._cfg = env, policy, costmap, cfg
        self._status: Status = "idle"
        self._plan: Plan | None = None
        self._follower: RouteFollower | None = None
        self._goal = np.zeros(2)
        self._replans = 0
        self._failure: str | None = None
        self._trail: list[Array] = []
        self._leg_start = np.zeros(2)
        self._best = np.inf
        self._since_progress = 0  # control steps
        self._since_waypoint = 0
        self._since_goal = 0

    @property
    def status(self) -> Status:
        return self._status

    @property
    def plan(self) -> Plan | None:
        return self._plan

    @property
    def replans(self) -> int:
        return self._replans

    @property
    def failure(self) -> str | None:
        """Return why the drive failed, or None."""
        return self._failure

    @property
    def trail(self) -> Array:
        """Return the (n, 2) head positions of every step driven so far."""
        return np.array(self._trail).reshape(-1, 2)

    @property
    def reached(self) -> int:
        """Return the number of waypoints of the current plan reached so far."""
        return 0 if self._follower is None else self._follower.reached

    def go(self, goal: ArrayLike) -> Plan | Infeasible:
        """Plan from the current head position to `goal` and start driving.

        An infeasible goal changes nothing: a drive in progress continues.
        """
        target = np.asarray(goal, dtype=np.float64)
        result = planner.plan(self._costmap, self._env.head_xy, target)
        if isinstance(result, Infeasible):
            return result
        self._goal = target
        self._replans, self._failure, self._since_goal = 0, None, 0
        self._follow(result)
        return result

    def _follow(self, plan: Plan) -> None:
        self._plan = plan
        self._follower = RouteFollower(plan.waypoints)
        self._follower.start(self._env)
        self._status = "driving"
        self._new_leg()

    def _new_leg(self) -> None:
        self._leg_start = self._env.head_xy
        self._best = np.inf
        self._since_progress = self._since_waypoint = 0

    def _fail(self, reason: str) -> None:
        self._status, self._failure = "failed", reason

    def _replan(self) -> None:
        if self._replans >= self._cfg.max_replans:
            self._fail("stuck")
            return
        self._replans += 1
        result = planner.plan(self._costmap, self._env.head_xy, self._goal)
        if isinstance(result, Infeasible):
            self._fail(result.reason.value)
        else:
            self._follow(result)

    def step(self) -> StepReport:
        """Advance one control step. Replan or finish as needed."""
        if self._status != "driving" or self._follower is None:
            raise RuntimeError("nothing to drive: call go() first")
        cfg, dt = self._cfg, self._env.spec_.control_dt
        report = self._follower.step(self._env, self._policy)
        self._trail.append(report.head_xy)
        self._since_goal += 1
        if report.event == "goal":
            self._status = "reached"
        elif report.event in ("flipped", "out_of_world"):
            self._fail(report.event)
        elif self._since_goal * dt >= cfg.max_seconds:
            self._fail("time limit")
        elif report.event == "timeout":
            self._replan()
        elif report.event == "waypoint":
            self._new_leg()
        else:
            waypoint = self._follower.waypoints[report.waypoint_index]
            distance = float(np.linalg.norm(waypoint - report.head_xy))
            self._since_progress += 1
            self._since_waypoint += 1
            if distance < self._best - cfg.stuck_progress:
                self._best, self._since_progress = distance, 0
            if (
                self._since_progress * dt >= cfg.stuck_seconds
                or self._since_waypoint * dt >= cfg.waypoint_timeout
                or _distance_to_leg(report.head_xy, self._leg_start, waypoint) > cfg.off_path
            ):
                self._replan()
        return report


@dataclass(frozen=True, eq=False)
class NavResult:
    success: bool
    reason: str  # "reached", or why it failed
    sim_seconds: float
    replans: int
    trajectory: Array  # (steps, 2) head positions
    class_seconds: tuple[float, float, float]  # time spent on each TerrainClass


def navigate(
    env: WormGoalEnv,
    policy: Policy,
    costmap: CostMap,
    start: ArrayLike,
    goal: ArrayLike,
    seed: int | None = None,
    cfg: NavConfig = NavConfig(),
    on_step: Callable[[], None] | None = None,
) -> NavResult | Infeasible:
    """Reset the env at `start`, then plan and drive to `goal`."""
    env.reset(seed=seed, options={"start": start, "target": goal})
    nav = Navigator(env, policy, costmap, cfg)
    result = nav.go(goal)
    if isinstance(result, Infeasible):
        return result
    dt = env.spec_.control_dt
    seconds = [0.0] * len(TerrainClass)
    while nav.status == "driving":
        seconds[int(env.surface_under_head)] += dt
        nav.step()
        if on_step is not None:
            on_step()
    trail = nav.trail
    return NavResult(
        success=nav.status == "reached",
        reason="reached" if nav.status == "reached" else str(nav.failure),
        sim_seconds=len(trail) * dt,
        replans=nav.replans,
        trajectory=trail,
        class_seconds=(seconds[0], seconds[1], seconds[2]),
    )

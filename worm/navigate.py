"""Drive a route one control step at a time, and replan when the worm is stuck."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from worm.controllers import Policy
from worm.env import WormGoalEnv

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

"""Follow a list of waypoints with one policy, without resetting the body."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from worm.env import WormGoalEnv

Array = NDArray[np.float64]
Policy = Callable[[NDArray[np.float32]], NDArray[np.float32]]


@dataclass(frozen=True)
class RouteResult:
    success: bool
    reached: int  # waypoints reached
    total: int  # waypoints in the route
    sim_seconds: float
    trajectory: Array  # (steps, 2) head positions


def make_waypoints(
    length: float, spacing: float, rng: np.random.Generator, max_turn: float = math.radians(30.0)
) -> Array:
    """Return a wandering route of `length` metres from the origin, as (n, 2) points."""
    heading = rng.uniform(-math.pi, math.pi)
    point = np.zeros(2)
    points = []
    for _ in range(round(length / spacing)):
        point = point + spacing * np.array([math.cos(heading), math.sin(heading)])
        points.append(point)
        heading += rng.uniform(-max_turn, max_turn)
    return np.array(points)


def run_route(
    policy: Policy,
    env: WormGoalEnv,
    waypoints: Array,
    seed: int | None = None,
    on_step: Callable[[], None] | None = None,
) -> RouteResult:
    """Drive through `waypoints` in order. Fail on a flip or a per-waypoint timeout.

    The per-waypoint timeout is the env's episode length. The tolerance is the env's.
    """
    obs, _ = env.reset(seed=seed, options={"target": waypoints[0]})
    trajectory = []
    reached = 0
    while reached < len(waypoints):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        trajectory.append(info["head_xy"])
        if on_step is not None:
            on_step()
        if info["reached"]:
            reached += 1
            if reached < len(waypoints):
                env.set_target(waypoints[reached])
        elif terminated or truncated:
            break
    return RouteResult(
        success=reached == len(waypoints),
        reached=reached,
        total=len(waypoints),
        sim_seconds=len(trajectory) * env.spec_.control_dt,
        trajectory=np.array(trajectory),
    )

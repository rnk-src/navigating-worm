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
    if round(length / spacing) < 1:
        raise ValueError(f"route length {length} m is shorter than half the spacing {spacing} m")
    heading = rng.uniform(-math.pi, math.pi)
    point = np.zeros(2)
    points = []
    for _ in range(round(length / spacing)):
        point = point + spacing * np.array([math.cos(heading), math.sin(heading)])
        points.append(point)
        heading += rng.uniform(-max_turn, max_turn)
    return np.array(points)


def split_seed(seed: int) -> tuple[int, int]:
    """Return independent (route seed, env seed) so route shape and physics do not correlate."""
    route_seed, env_seed = np.random.SeedSequence(seed).generate_state(2)
    return int(route_seed), int(env_seed)


def showcase_waypoints() -> Array:
    """Return a short demo route that heads in every compass direction, with sharp turns."""
    return np.array(
        [
            [1.2, 0.0],  # east
            [1.2, 1.2],  # north (left turn)
            [0.0, 1.2],  # west (left turn)
            [0.0, 2.4],  # north (right turn)
            [1.2, 2.4],  # east (right turn)
            [2.4, 1.2],  # south-east
            [2.4, 0.0],  # south
            [1.2, -1.0],  # south-west
            [0.0, 0.0],  # north-west, back to the start
        ]
    )


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
    if len(waypoints) == 0:
        raise ValueError("the route has no waypoints")
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
                obs = env.set_target(waypoints[reached])
        elif terminated or truncated:
            break
    return RouteResult(
        success=reached == len(waypoints),
        reached=reached,
        total=len(waypoints),
        sim_seconds=len(trajectory) * env.spec_.control_dt,
        trajectory=np.array(trajectory),
    )

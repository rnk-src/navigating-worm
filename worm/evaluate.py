"""Evaluate a policy: planned routes on generated worlds, and long routes on the plane."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from worm import planner
from worm import world as wd
from worm.controllers import Policy
from worm.env import WormGoalEnv
from worm.navigate import NavConfig, NavResult, navigate
from worm.planner import CostMap, Infeasible
from worm.route import make_waypoints, run_route, split_seed
from worm.terrain import TerrainClass
from worm.world import World, WorldConfig

Array = NDArray[np.float64]
EVAL_LEVELS = (1, 2, 3)
_CLASS_NAMES = tuple(cls.name.lower() for cls in TerrainClass)
_ROUTE_TRIES = 50  # planned pairs per map before the map is skipped
_SPEED_WINDOW = 1.0  # s, speed is the head displacement over windows of this length


def eval_world(level: int, size: float, seed: int) -> World:
    """Return the eval world of `level` on a square of `size` metres.

    Level 1 is level ground with mixed surfaces. Level 2 adds hills up to 10 degrees and
    roughness. Level 3 is the full terrain: hills up to 18 degrees, ditches and rocks.
    """
    if level == 1:
        cfg = WorldConfig(size=(size, size), hill_height=0.0, roughness=0.0, n_rocks=0, n_ditches=0)
    elif level == 2:
        cfg = WorldConfig(size=(size, size), max_slope_deg=10.0, n_rocks=0, n_ditches=0)
    elif level == 3:
        cfg = WorldConfig(
            size=(size, size),
            hill_height=0.45,
            max_slope_deg=18.0,
            n_rocks=round(0.1 * size**2),
            n_ditches=2,
        )
    else:
        raise ValueError(f"no eval world for level {level}")
    return wd.generate(cfg, seed)


def pick_route(
    costmap: CostMap, rng: np.random.Generator, min_route: float
) -> tuple[Array, Array, int] | None:
    """Draw a start and a goal at least `min_route` apart that the planner can connect.

    Return (start, goal, infeasible pairs skipped), or None when no pair is found.
    """
    free = np.argwhere(np.isfinite(costmap.cost))
    if len(free) < 2:
        return None
    skipped = 0
    for _ in range(20 * _ROUTE_TRIES):
        if skipped >= _ROUTE_TRIES:
            break
        start, goal = costmap.to_xy(free[rng.integers(len(free), size=2)])
        if np.linalg.norm(goal - start) < min_route:
            continue
        if isinstance(planner.plan(costmap, start, goal), Infeasible):
            skipped += 1
            continue
        return start, goal, skipped
    return None


def _class_metres(world: World, trail: Array, dt: float) -> list[float]:
    """Return the distance covered on each class, from the head position once per window."""
    stride = max(1, round(_SPEED_WINDOW / dt))
    points = trail[::stride]
    metres = [0.0] * len(TerrainClass)
    if len(points) < 2:
        return metres
    classes = world.class_at(points[:-1])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    for cls, length in zip(classes, lengths, strict=True):
        metres[int(cls)] += float(length)
    return metres


def _summary(routes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(routes)
    successes = sum(route["success"] for route in routes)
    seconds = np.array([route["class_seconds"] for route in routes]).reshape(-1, 3).sum(axis=0)
    metres = np.array([route["class_metres"] for route in routes]).reshape(-1, 3).sum(axis=0)
    total = float(seconds.sum())
    per_class = {
        name: {
            "seconds": round(float(seconds[k]), 2),
            "time_share": float(seconds[k]) / total if total > 0.0 else 0.0,
            "mean_speed": float(metres[k] / seconds[k]) if seconds[k] > 0.0 else 0.0,
        }
        for k, name in enumerate(_CLASS_NAMES)
    }
    failures: dict[str, int] = {}
    for route in routes:
        if not route["success"]:
            failures[route["failure_class"]] = failures.get(route["failure_class"], 0) + 1
    return {
        "routes": n,
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "replans_per_route": sum(route["replans"] for route in routes) / n if n else 0.0,
        "mean_sim_seconds": sum(route["sim_seconds"] for route in routes) / n if n else 0.0,
        "per_class": per_class,
        "failure_class": failures,
    }


def evaluate_terrain(
    policy: Policy,
    obs_version: int,
    maps: int,
    size: float,
    levels: tuple[int, ...] = EVAL_LEVELS,
    seed: int = 2000,
    min_route: float | None = None,
    nav_cfg: NavConfig | None = None,
) -> dict[str, Any]:
    """Drive one planned route on each of `maps` worlds per level. Return plain data.

    Map k of every level uses world seed `seed + k`. The start and the goal are random
    free points at least `min_route` apart (default: half the world size) that the
    planner can connect. The result holds the success rate overall and per level, and
    per terrain class the time share, the mean speed and the failures.
    """
    min_route = 0.5 * size if min_route is None else min_route
    nav_cfg = nav_cfg or NavConfig()
    routes: list[dict[str, Any]] = []
    infeasible = {level: 0 for level in levels}
    skipped_maps = {level: 0 for level in levels}
    for level in levels:
        for k in range(maps):
            world = eval_world(level, size, seed + k)
            costmap = planner.build_costmap(world)
            route_seed, env_seed = split_seed(seed + k)
            picked = pick_route(costmap, np.random.default_rng(route_seed), min_route)
            if picked is None:
                skipped_maps[level] += 1
                continue
            start, goal, skipped = picked
            infeasible[level] += skipped
            env = WormGoalEnv(
                world=world, episode_seconds=nav_cfg.waypoint_timeout, obs_version=obs_version
            )
            result = navigate(env, policy, costmap, start, goal, seed=env_seed, cfg=nav_cfg)
            assert isinstance(result, NavResult)  # pick_route planned this pair
            dt = env.spec_.control_dt
            routes.append(
                {
                    "level": level,
                    "map_seed": seed + k,
                    "success": result.success,
                    "reason": result.reason,
                    "distance": round(float(np.linalg.norm(goal - start)), 2),
                    "sim_seconds": round(result.sim_seconds, 2),
                    "replans": result.replans,
                    "class_seconds": [round(t, 2) for t in result.class_seconds],
                    "class_metres": [
                        round(m, 3) for m in _class_metres(world, result.trajectory, dt)
                    ],
                    "failure_class": None
                    if result.success
                    else _CLASS_NAMES[int(env.surface_under_head)],
                }
            )
            env.close()
    per_level = {}
    for level in levels:
        summary = _summary([route for route in routes if route["level"] == level])
        summary["infeasible_pairs"] = infeasible[level]
        summary["skipped_maps"] = skipped_maps[level]
        per_level[str(level)] = summary
    return {
        "maps": maps,
        "size": float(size),
        "seed": seed,
        "min_route": float(min_route),
        "overall": _summary(routes),
        "levels": per_level,
        "routes": routes,
    }


def evaluate_flat_routes(
    policy: Policy,
    obs_version: int,
    runs: int,
    length: float = 100.0,
    spacing: float = 2.0,
    timeout: float = 60.0,
    seed: int = 1000,
) -> dict[str, Any]:
    """Drive `runs` random waypoint routes of `length` metres on the plane."""
    rows = []
    for run in range(runs):
        route_seed, env_seed = split_seed(seed + run)
        env = WormGoalEnv(episode_seconds=timeout, obs_version=obs_version)
        waypoints = make_waypoints(length, spacing, np.random.default_rng(route_seed))
        result = run_route(policy, env, waypoints, seed=env_seed)
        rows.append(
            {
                "run": run,
                "success": result.success,
                "reached": result.reached,
                "total": result.total,
                "sim_seconds": round(result.sim_seconds, 1),
                "mu_forward": round(env.params.friction.mu_forward, 3),
                "mu_lateral": round(env.params.friction.mu_lateral, 3),
            }
        )
        env.close()
    successes = sum(row["success"] for row in rows)
    return {"successes": successes, "runs_total": len(rows), "runs": rows}

"""Record a video of the policy on a route that turns in every direction.

The left half follows the worm. The right half is a fixed top-down view of the route.
Red spheres are waypoints still to reach. Grey spheres are reached. Blue dots are the trail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from worm.env import WormGoalEnv
from worm.randomize import DomainRandomizer
from worm.route import run_route, showcase_waypoints
from worm.training import load_policy
from worm.video import FrameRecorder

RED, GREY, BLUE = (0.9, 0.1, 0.1, 1.0), (0.6, 0.6, 0.6, 1.0), (0.1, 0.3, 0.9, 1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("results/policy"))
    parser.add_argument("--out", type=Path, default=Path("results/showcase.mp4"))
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per waypoint")
    parser.add_argument("--speed", type=int, default=2, help="keep every Nth frame")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    waypoints = showcase_waypoints()
    corners = np.vstack([[0.0, 0.0], waypoints])
    low, high = corners.min(axis=0), corners.max(axis=0)
    env = WormGoalEnv(
        randomizer=DomainRandomizer(0.0), episode_seconds=args.timeout, render_mode="rgb_array"
    )
    middle = 0.5 * (low + high)
    center = (float(middle[0]), float(middle[1]))
    half_extent = 0.5 * float((high - low).max()) + 0.3

    trail: list[tuple[float, float]] = []

    def follow() -> None:
        head = env.head_xy
        trail.append((float(head[0]), float(head[1])))

    def compose() -> np.ndarray | None:
        current = int(np.argmin(np.linalg.norm(waypoints - env.target, axis=1)))
        env.markers = [(xy, BLUE, 0.012) for xy in trail[::4]]
        env.markers += [
            ((float(x), float(y)), GREY if i < current else RED, 0.05)
            for i, (x, y) in enumerate(waypoints)
        ]
        env.set_overview(None)
        close_up = env.render()
        env.set_overview(center, half_extent)
        overview = env.render()
        if close_up is None or overview is None:
            return None
        return np.hstack([close_up, overview])

    recorder = FrameRecorder(compose, every=args.speed)

    def record() -> None:
        follow()
        recorder()

    result = run_route(load_policy(args.policy), env, waypoints, seed=args.seed, on_step=record)
    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, recorder.frames, fps=env.metadata["render_fps"])
    print(f"reached {result.reached}/{result.total} in {result.sim_seconds:.0f} s of sim time")
    print(f"wrote {args.out} ({len(recorder.frames)} frames, {args.speed}x speed)")

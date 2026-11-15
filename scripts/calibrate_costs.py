"""Measure how fast a policy travels on each surface class, for the planner's cost table.

Each class gets a level world of that class alone. The worm drives to several targets
with randomized physics. The cost of a class is the concrete speed over its speed.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from worm import world as wd
from worm.cli import policy_from_arg
from worm.env import WormGoalEnv
from worm.terrain import TerrainClass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="results/policy", help='a policy folder or "scripted"')
    parser.add_argument("--targets", type=int, default=5, help="targets per class")
    parser.add_argument("--distance", type=float, default=2.0, help="target distance in metres")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    speeds = {}
    policy, obs_version = policy_from_arg(args.policy)
    for cls in TerrainClass:
        env = WormGoalEnv(
            world=wd.flat((6.0, 6.0), cls=cls), episode_seconds=90.0, obs_version=obs_version
        )
        covered, seconds = 0.0, 0.0
        for k in range(args.targets):
            angle = 2.0 * math.pi * k / args.targets
            target = args.distance * np.array([math.cos(angle), math.sin(angle)])
            obs, info = env.reset(seed=args.seed + k, options={"target": target})
            begin = info["distance"]
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(policy(obs))
                seconds += env.spec_.control_dt
                done = terminated or truncated
            covered += begin - info["distance"]
        speeds[cls] = covered / seconds
        print(f"{cls.name.lower()}: {speeds[cls]:.3f} m/s over {args.targets} targets")
    base = speeds[TerrainClass.CONCRETE]
    costs = ", ".join(f"{base / max(speeds[c], 1e-6):.2f}" for c in TerrainClass)
    print(f"surface_cost = ({costs})")

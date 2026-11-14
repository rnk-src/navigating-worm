"""Measure env throughput in control steps per second, on one core.

`--world flat` is the plane. `--world 10m` is a generated world of that side length.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from worm.env import WormGoalEnv
from worm.world import WorldConfig, generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="flat", help='"flat" or a side length such as "10m"')
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.world == "flat":
        env = WormGoalEnv()
    else:
        side = float(args.world.removesuffix("m"))
        env = WormGoalEnv(world=generate(WorldConfig(size=(side, side)), args.seed))
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    resets = 0
    begin = time.perf_counter()
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
            resets += 1
    elapsed = time.perf_counter() - begin
    print(f"{args.world}: {args.steps / elapsed:.0f} control steps/s ({resets} resets)")

"""Train the goal-reaching policy with PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from worm.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/ppo"))
    args = parser.parse_args()
    print("saved to", train(args.steps, args.n_envs, args.seed, args.out, verbose=1))

"""Train the goal-reaching policy with PPO.

Without `--config` this is the M1 flat-ground run. With `--config` the run comes from a
TOML file in `configs/`, and `--steps`, `--n-envs` and `--seed` replace its values.
SIGTERM saves a checkpoint and ends the run. `--resume` continues it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from worm.training import load_config, train, with_overrides

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="a TOML file from configs/")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.config is None:
        if args.resume:
            parser.error("--resume needs --config")
        steps = 3_000_000 if args.steps is None else args.steps
        n_envs = 8 if args.n_envs is None else args.n_envs
        out = train(steps, n_envs, args.seed or 0, args.out or Path("runs/ppo"), verbose=1)
    else:
        cfg = with_overrides(load_config(args.config), args.steps, args.n_envs, args.seed)
        target = args.out or Path("runs") / f"{cfg.run.name}_seed{cfg.run.seed}"
        out = train(cfg, target, resume=args.resume, verbose=1)
    print("saved to", out)

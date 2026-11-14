"""Command-line helpers shared by the scripts: world arguments and policy loading."""

from __future__ import annotations

import argparse
from pathlib import Path

from worm.controllers import Policy, bearing_steer
from worm.spec import WormSpec, obs_layout
from worm.world import World, WorldConfig, generate


def add_world_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="world seed")
    parser.add_argument("--size", type=float, default=10.0, help="side length in metres")
    parser.add_argument(
        "--mix", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("CONCRETE", "GRASS", "SAND"), help="area shares; they are normalized",
    )  # fmt: skip
    parser.add_argument("--max-slope", type=float, default=12.0, help="degrees")
    parser.add_argument("--rocks", type=int, default=10)
    parser.add_argument("--ditches", type=int, default=2)


def world_from_args(args: argparse.Namespace) -> World:
    cfg = WorldConfig(
        size=(args.size, args.size),
        mix=(args.mix[0], args.mix[1], args.mix[2]),
        max_slope_deg=args.max_slope,
        n_rocks=args.rocks,
        n_ditches=args.ditches,
    )
    return generate(cfg, args.seed)


def policy_from_arg(name: str) -> tuple[Policy, int]:
    """Return (policy, observation version) for a policy folder or "scripted".

    Build the env with this observation version, or the policy gets the wrong input.
    """
    if name == "scripted":
        return bearing_steer(obs_layout(WormSpec())), 1
    from worm.training import load_policy  # imports torch; the scripted path does not need it

    loaded = load_policy(Path(name))
    return loaded, loaded.obs_version

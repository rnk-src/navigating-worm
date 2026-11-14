"""Generate a seeded terrain world and write its top-down map as a PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from worm import mapviz
from worm.terrain import TerrainClass
from worm.world import WorldConfig, generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=float, default=10.0, help="side length in metres")
    parser.add_argument(
        "--mix", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("CONCRETE", "GRASS", "SAND"), help="area shares; they are normalized",
    )  # fmt: skip
    parser.add_argument("--max-slope", type=float, default=15.0, help="degrees")
    parser.add_argument("--rocks", type=int, default=10)
    parser.add_argument("--ditches", type=int, default=2)
    parser.add_argument("--px-per-m", type=float, default=40.0)
    parser.add_argument("--out", type=Path, default=Path("map.png"))
    args = parser.parse_args()

    cfg = WorldConfig(
        size=(args.size, args.size),
        mix=(args.mix[0], args.mix[1], args.mix[2]),
        max_slope_deg=args.max_slope,
        n_rocks=args.rocks,
        n_ditches=args.ditches,
    )
    world = generate(cfg, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, mapviz.world_image(world, args.px_per_m))
    shares = np.bincount(world.surface.ravel(), minlength=len(TerrainClass)) / world.surface.size
    mix = ", ".join(
        f"{c.name.lower()} {100 * s:.0f} %" for c, s in zip(TerrainClass, shares, strict=True)
    )
    print(f"seed {args.seed}: {args.size:g} m world, {len(world.rocks)} rocks, {mix}")
    steepest = world.slope_deg().max()
    print(f"height range {world.heights.max():.2f} m, steepest slope {steepest:.0f} deg")
    print(f"wrote {args.out}")

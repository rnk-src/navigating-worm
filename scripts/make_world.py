"""Generate a seeded terrain world and write its top-down map as a PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from worm import mapviz
from worm.cli import add_world_arguments, world_from_args
from worm.terrain import TerrainClass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_world_arguments(parser)
    parser.add_argument("--px-per-m", type=float, default=40.0)
    parser.add_argument("--out", type=Path, default=Path("map.png"))
    args = parser.parse_args()

    world = world_from_args(args)
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

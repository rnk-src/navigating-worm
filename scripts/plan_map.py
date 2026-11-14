"""Plan a route on a generated world and write a PNG of the map, blocked cells and path.

Exit code 2 means that no route exists; the reason is printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio

from worm import mapviz, planner
from worm.cli import add_world_arguments, world_from_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_world_arguments(parser)
    parser.add_argument("--start", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--px-per-m", type=float, default=40.0)
    parser.add_argument("--out", type=Path, default=Path("plan.png"))
    args = parser.parse_args()

    world = world_from_args(args)
    costmap = planner.build_costmap(world)
    result = planner.plan(costmap, args.start, args.goal)
    if isinstance(result, planner.Infeasible):
        print(f"infeasible: {result.reason.name}: {result.detail}")
        sys.exit(2)

    image = mapviz.world_image(world, args.px_per_m)
    mapviz.draw_blocked(image, world, costmap, args.px_per_m)
    mapviz.draw_plan(image, world, result.path, result.waypoints, args.px_per_m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.out, image)
    snapped = " (goal moved to free ground)" if result.goal_snapped else ""
    print(
        f"plan: {len(result.waypoints)} waypoints, cost {result.cost:.1f}, "
        f"{result.expanded} cells searched{snapped}"
    )
    print(f"wrote {args.out}")

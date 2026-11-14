"""Evaluate a policy on planned routes across generated terrain worlds.

Each level drives one route of at least half the world size on each of `--maps` worlds.
Level 1 is level ground with mixed surfaces, level 2 adds hills, level 3 is the full
terrain with rocks and ditches. The report gives the success rate overall and per level,
and per terrain class the time share, the mean speed and the failures.
`--flat-routes N` also drives N long routes on the plane, as the regression check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from worm import evaluate as ev
from worm.controllers import Policy, bearing_steer
from worm.spec import WormSpec, obs_layout


def report(name: str, summary: dict[str, Any]) -> None:
    print(
        f"{name}: success {summary['successes']}/{summary['routes']}, "
        f"{summary['replans_per_route']:.2f} replans per route"
    )
    for cls, row in summary["per_class"].items():
        print(f"    {cls}: {100 * row['time_share']:.0f} % of time, {row['mean_speed']:.3f} m/s")
    if summary["failure_class"]:
        print(f"    failures by class: {summary['failure_class']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="results/policy", help='a policy folder or "scripted"')
    parser.add_argument("--maps", type=int, default=50, help="worlds per level")
    parser.add_argument("--size", type=float, default=10.0, help="world side in metres")
    parser.add_argument("--levels", default="1,2,3", help="comma-separated levels")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--flat-routes", type=int, default=0, help="long routes on the plane")
    parser.add_argument("--flat-length", type=float, default=100.0, help="metres per flat route")
    parser.add_argument("--json", type=Path, default=None, help="write the result here")
    args = parser.parse_args()

    policy: Policy
    if args.policy == "scripted":
        policy, obs_version = bearing_steer(obs_layout(WormSpec())), 1
    else:
        from worm.training import load_policy  # imports torch

        loaded = load_policy(Path(args.policy))
        policy, obs_version = loaded, loaded.obs_version
    levels = tuple(int(level) for level in args.levels.split(","))
    result = ev.evaluate_terrain(policy, obs_version, args.maps, args.size, levels, args.seed)
    result = {"policy": Path(args.policy).name, "obs_version": obs_version, **result}
    for level in levels:
        report(f"level {level}", result["levels"][str(level)])
    report("overall", result["overall"])
    if args.flat_routes > 0:
        flat = ev.evaluate_flat_routes(
            policy, obs_version, args.flat_routes, length=args.flat_length
        )
        result["flat_routes"] = flat
        print(f"flat {args.flat_length:.0f} m routes: {flat['successes']}/{flat['runs_total']}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")

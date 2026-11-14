"""Plan a route on a generated world and drive it with a policy, headless.

The worm starts at `--start` and drives to `--goal`. The navigator replans when the
worm is stuck or off the path. `--video` writes an MP4: the follow camera on the left,
the map with the plan and the trail on the right.

Exit code 0: reached. 1: failed. 2: no route exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from worm import mapviz, planner
from worm.cli import add_world_arguments, world_from_args
from worm.controllers import bearing_steer
from worm.env import WormGoalEnv
from worm.navigate import NavConfig, Navigator
from worm.terrain import TerrainClass
from worm.training import load_policy
from worm.video import FrameRecorder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="results/policy", help='a policy folder or "scripted"')
    add_world_arguments(parser)
    parser.add_argument("--start", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    parser.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--env-seed", type=int, default=0, help="physics and sensor noise seed")
    parser.add_argument("--video", type=Path, default=None, help="write an MP4 here")
    parser.add_argument("--speed", type=int, default=4, help="video: keep every Nth frame")
    parser.add_argument("--json", type=Path, default=None, help="write the result here")
    args = parser.parse_args()

    world = world_from_args(args)
    costmap = planner.build_costmap(world)
    cfg = NavConfig()
    env = WormGoalEnv(
        world=world,
        episode_seconds=cfg.waypoint_timeout,
        render_mode="rgb_array" if args.video else None,
    )
    policy = (
        bearing_steer(env.layout) if args.policy == "scripted" else load_policy(Path(args.policy))
    )
    env.reset(seed=args.env_seed, options={"start": args.start, "target": args.goal})
    nav = Navigator(env, policy, costmap, cfg)
    result = nav.go(args.goal)
    if isinstance(result, planner.Infeasible):
        print(f"infeasible: {result.reason.name}: {result.detail}")
        sys.exit(2)

    px_per_m = 480.0 / args.size
    base_map = mapviz.world_image(world, px_per_m)
    mapviz.draw_blocked(base_map, world, costmap, px_per_m)

    def compose() -> np.ndarray | None:
        camera = env.render()
        if camera is None or nav.plan is None:
            return None
        overview = base_map.copy()
        mapviz.draw_plan(overview, world, nav.plan.path, nav.plan.waypoints, px_per_m, nav.reached)
        if len(nav.trail) > 1:
            mapviz.draw_polyline(overview, world, nav.trail, (255, 255, 255), px_per_m, width=1)
        mapviz.draw_points(overview, world, env.head_xy, (0, 0, 0), px_per_m, radius=4)
        return np.hstack([camera, overview[:480, :480]])

    recorder = FrameRecorder(compose, every=args.speed)
    dt = env.spec_.control_dt
    seconds = dict.fromkeys((c.name.lower() for c in TerrainClass), 0.0)
    while nav.status == "driving":
        seconds[env.surface_under_head.name.lower()] += dt
        nav.step()
        if args.video:
            recorder()
    env.close()

    sim_seconds = len(nav.trail) * dt
    reason = "reached" if nav.status == "reached" else str(nav.failure)
    print(f"{reason}: {sim_seconds:.0f} s of sim time, {nav.replans} replans")
    print("time per class: " + ", ".join(f"{name} {t:.0f} s" for name, t in seconds.items()))
    if args.video and recorder.frames:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, recorder.frames, fps=env.metadata["render_fps"])
        print(f"wrote {args.video} ({len(recorder.frames)} frames, {args.speed}x speed)")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "success": nav.status == "reached",
            "reason": reason,
            "policy": str(args.policy),
            "world_seed": args.seed,
            "start": list(args.start),
            "goal": list(args.goal),
            "sim_seconds": round(sim_seconds, 2),
            "replans": nav.replans,
            "class_seconds": {name: round(t, 2) for name, t in seconds.items()},
        }
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    sys.exit(0 if nav.status == "reached" else 1)

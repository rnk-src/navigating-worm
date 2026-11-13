"""Evaluate a policy on long waypoint routes with random friction per run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worm.env import WormGoalEnv
from worm.route import make_waypoints, run_route, split_seed
from worm.training import load_policy
from worm.video import FrameRecorder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--length", type=float, default=100.0, help="route length in metres")
    parser.add_argument("--spacing", type=float, default=2.0, help="waypoint spacing in metres")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per waypoint")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video", type=Path, default=None, help="write an MP4 of the first run")
    parser.add_argument("--video-seconds", type=float, default=30.0)
    parser.add_argument("--json", type=Path, default=None, help="write the result table here")
    args = parser.parse_args()

    policy = load_policy(args.policy)
    rows = []
    for run in range(args.runs):
        route_seed, env_seed = split_seed(args.seed + run)
        record = args.video is not None and run == 0
        env = WormGoalEnv(episode_seconds=args.timeout, render_mode="rgb_array" if record else None)
        fps = env.metadata["render_fps"]
        recorder = FrameRecorder(env.render, max_frames=round(args.video_seconds * fps))
        waypoints = make_waypoints(args.length, args.spacing, np.random.default_rng(route_seed))
        result = run_route(
            policy, env, waypoints, seed=env_seed, on_step=recorder if record else None
        )
        if recorder.frames:
            import imageio.v2 as imageio

            imageio.mimsave(args.video, recorder.frames, fps=fps)
        rows.append(
            {
                "run": run,
                "success": result.success,
                "reached": result.reached,
                "total": result.total,
                "sim_seconds": round(result.sim_seconds, 1),
                "mu_forward": round(env.params.friction.mu_forward, 3),
                "mu_lateral": round(env.params.friction.mu_lateral, 3),
            }
        )
        print(rows[-1])
        env.close()

    successes = sum(row["success"] for row in rows)
    print(f"success: {successes}/{len(rows)}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        summary = {"successes": successes, "runs_total": len(rows), "runs": rows}
        args.json.write_text(json.dumps(summary, indent=2) + "\n")

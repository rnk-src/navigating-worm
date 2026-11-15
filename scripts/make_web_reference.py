"""Record a native run for the browser engine to match.

The browser runs the same Python on MuJoCo compiled to WebAssembly. This script runs the
same map, goal and policy natively and saves where the worm is after fixed numbers of
control steps. `web/tests/engine.test.ts` compares the two.

Usage: uv run python scripts/make_web_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

from worm import npolicy
from worm.demo import Controller, DemoSession, MapRequest

POLICY = Path("results/policy_terrain/policy.json")
OUT = Path("web/tests/fixtures/native-run.json")
REQUEST = {"seed": 0, "size": 10, "mix": [0.4, 0.3, 0.3], "slope": 12, "rocks": 12, "ditches": 2}
CANDIDATE_GOALS = [(2.0, 1.0), (1.5, -1.5), (-1.5, 1.5), (2.5, 2.0), (-2.0, -1.0), (0.5, 2.5)]
CHECKPOINTS = (20, 60, 120, 240)


def main() -> None:
    policy = npolicy.load(POLICY)
    session = DemoSession(Controller("policy_terrain", policy, policy.obs_version))
    session.new_map(MapRequest.from_json(REQUEST))
    for goal in CANDIDATE_GOALS:
        answer = session.goal(goal)
        if answer["ok"]:
            break
    else:
        raise SystemExit("no candidate goal is free on this map")
    plan = answer["plan"]
    run = []
    done = 0
    for checkpoint in CHECKPOINTS:
        session.tick(checkpoint - done)
        done = checkpoint
        snap = session.snapshot()
        run.append(
            {
                "step": checkpoint,
                "head": snap["head"],
                "yaw": snap["yaw"],
                "status": snap["status"],
                "reached": snap["reached"],
                "sim_time": snap["sim_time"],
            }
        )
    OUT.write_text(
        json.dumps(
            {
                "request": REQUEST,
                "goal": list(goal),
                "n_waypoints": len(plan["waypoints"]),
                "path_start": plan["path"][0],
                "path_end": plan["path"][-1],
                "run": run,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT}")
    for row in run:
        print(row["step"], row["head"], row["status"], row["reached"])


if __name__ == "__main__":
    main()

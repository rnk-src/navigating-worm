import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from worm import evaluate as ev
from worm import planner
from worm.controllers import bearing_steer
from worm.spec import WormSpec, obs_layout
from worm.world import World

ROOT = Path(__file__).parent.parent
SCRIPTED = bearing_steer(obs_layout(WormSpec()))


def test_eval_worlds_get_harder_with_the_level() -> None:
    flat, hills, full = (ev.eval_world(level, 4.0, seed=3) for level in (1, 2, 3))
    assert np.all(flat.heights == 0.0) and len(np.unique(flat.surface)) >= 2
    assert hills.heights.max() > 0.0 and not hills.obstacle.any()
    assert full.obstacle.any()
    assert flat.shape == hills.shape == full.shape


def test_eval_terrain_writes_overall_level_and_per_class_breakdown() -> None:
    result = ev.evaluate_terrain(SCRIPTED, obs_version=1, maps=2, size=4.0, levels=(1, 2), seed=5)
    assert set(result["levels"]) == {"1", "2"}
    for level in result["levels"].values():
        assert level["routes"] == 2 and 0 <= level["successes"] <= 2
        assert level["success_rate"] == level["successes"] / 2
        assert level["replans_per_route"] >= 0.0
    overall = result["overall"]
    assert overall["routes"] == 4
    assert overall["successes"] == sum(lv["successes"] for lv in result["levels"].values())
    assert overall["successes"] >= 3  # the scripted controller drives short easy routes
    classes = overall["per_class"]
    assert set(classes) == {"concrete", "grass", "sand"}
    assert abs(sum(c["time_share"] for c in classes.values()) - 1.0) < 1e-6
    driven = [c for c in classes.values() if c["seconds"] > 5.0]
    assert driven and all(0.02 < c["mean_speed"] < 0.5 for c in driven)
    assert set(overall["failure_class"]) <= {"concrete", "grass", "sand"}
    assert sum(overall["failure_class"].values()) == overall["routes"] - overall["successes"]
    assert len(result["routes"]) == 4
    route = result["routes"][0]
    assert route["distance"] >= 2.0 and route["level"] == 1 and route["map_seed"] == 5
    json.dumps(result)  # the result is plain data


def test_eval_terrain_is_deterministic() -> None:
    first = ev.evaluate_terrain(SCRIPTED, 1, maps=1, size=4.0, levels=(2,), seed=9)
    again = ev.evaluate_terrain(SCRIPTED, 1, maps=1, size=4.0, levels=(2,), seed=9)
    assert first == again


def test_eval_terrain_skips_infeasible_pairs_and_counts_them(
    make_world: Callable[..., World],
) -> None:
    # A wall of rock along x = 0 splits a 6 m world. Only pairs on one side have a route.
    obstacle = np.zeros((121, 121), dtype=bool)
    obstacle[:, 58:63] = True
    walled = make_world(np.where(obstacle, 0.2, 0.0), obstacle=obstacle)
    costmap = planner.build_costmap(walled)
    infeasible = 0
    for seed in range(8):
        route = ev.pick_route(costmap, np.random.default_rng(seed), min_route=1.5)
        assert route is not None
        start, goal, skipped = route
        assert start[0] * goal[0] > 0.0  # both ends lie on the same side of the wall
        assert np.linalg.norm(goal - start) >= 1.5
        infeasible += skipped
    assert infeasible > 0

    blocked = make_world(np.full((41, 41), 0.2), obstacle=np.ones((41, 41), dtype=bool))
    assert ev.pick_route(planner.build_costmap(blocked), np.random.default_rng(0), 1.0) is None


def test_flat_route_eval_counts_successes() -> None:
    result = ev.evaluate_flat_routes(SCRIPTED, obs_version=1, runs=2, length=2.0, seed=1000)
    assert result["runs_total"] == 2 and 0 <= result["successes"] <= 2
    assert len(result["runs"]) == 2 and {"success", "reached", "total"} <= set(result["runs"][0])


def test_eval_terrain_script_writes_the_json(tmp_path: Path) -> None:
    out = tmp_path / "eval.json"
    done = subprocess.run(
        [sys.executable, "scripts/eval_terrain.py", "--policy", "scripted", "--maps", "1"]
        + ["--size", "4", "--levels", "1", "--seed", "5", "--flat-routes", "1"]
        + ["--flat-length", "2", "--json", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    saved = json.loads(out.read_text())
    assert saved["policy"] == "scripted" and saved["maps"] == 1 and saved["size"] == 4.0
    assert saved["levels"]["1"]["routes"] == 1 and saved["flat_routes"]["runs_total"] == 1
    assert "success" in done.stdout and "concrete" in done.stdout

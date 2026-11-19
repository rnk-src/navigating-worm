import json

import numpy as np
from numpy.typing import NDArray

from worm import demo
from worm.demo import Controller, MapRequest, RaceSession
from worm.navigate import NavConfig

SMALL_MAP = MapRequest(seed=3, size=4.0, max_slope_deg=5.0, n_rocks=0, n_ditches=0)
GOAL = (1.0, 0.5)


def idle(obs: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([-1.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def race(*controllers: Controller, cfg: NavConfig | None = None) -> RaceSession:
    made = RaceSession(list(controllers), cfg)
    made.new_map(SMALL_MAP)
    return made


def scripted(name: str) -> Controller:
    return Controller(name, demo.load_controller("scripted").policy, 1)


def test_racers_share_one_world_and_one_route() -> None:
    made = race(scripted("a"), scripted("b"))
    answer = made.goal(GOAL)
    assert answer["ok"]
    bodies = made.snapshot()["bodies"]
    assert [b["label"] for b in bodies] == ["a", "b"]
    sessions = made.sessions
    assert sessions[0].world is sessions[1].world
    assert (
        sessions[0].snapshot()["path"] == sessions[1].snapshot()["path"] == answer["plan"]["path"]
    )


def test_tick_steps_every_racer_in_lockstep() -> None:
    made = race(scripted("a"), scripted("b"), scripted("c"))
    made.goal(GOAL)
    made.tick(25)
    times = {s.snapshot()["sim_time"] for s in made.sessions}
    assert len(times) == 1 and times.pop() > 0


def test_a_finisher_gets_its_time_and_a_stalled_racer_is_dnf() -> None:
    made = race(scripted("fast"), Controller("idle", idle, 1), cfg=NavConfig(max_seconds=30.0))
    made.goal(GOAL)
    made.tick(2000)
    fast, stalled = made.snapshot()["bodies"]
    assert fast["done"] and fast["t_finish"] is not None and 0 < fast["t_finish"] < 30.0
    assert stalled["done"] and stalled["t_finish"] is None
    assert made.snapshot()["race_over"]


def test_a_new_goal_restarts_every_racer_at_the_spawn() -> None:
    made = race(scripted("a"), scripted("b"))
    made.goal(GOAL)
    made.tick(100)
    made.goal((-1.0, 0.5))
    starts = [s.snapshot() for s in made.sessions]
    assert all(s["sim_time"] == 0 for s in starts)
    assert starts[0]["body"] == starts[1]["body"]
    assert np.linalg.norm(np.array(starts[0]["head"])) < 0.3


def test_the_snapshot_keeps_the_leader_fields_and_is_json() -> None:
    made = race(scripted("lead"), scripted("ghost"))
    made.goal(GOAL)
    snap = made.snapshot()
    json.dumps(snap)
    assert snap["policy"] == "lead" and "trail" in snap and "body" in snap
    keys = {"label", "segs", "yaw", "trail", "done", "t_finish"}
    assert all(set(b) == keys for b in snap["bodies"])


def test_with_racing_off_only_the_leader_moves_and_shows() -> None:
    made = race(scripted("lead"), scripted("ghost"))
    made.set_racing(False)
    made.goal(GOAL)
    made.tick(25)
    lead, ghost = made.sessions
    assert lead.snapshot()["sim_time"] > 0 and ghost.snapshot()["sim_time"] == 0
    assert [b["label"] for b in made.snapshot()["bodies"]] == ["lead"]


def test_turning_racing_on_restarts_every_racer_on_the_current_goal() -> None:
    made = race(scripted("lead"), scripted("ghost"))
    made.set_racing(False)
    made.goal(GOAL)
    made.tick(25)
    made.set_racing(True)
    snaps = [s.snapshot() for s in made.sessions]
    assert all(s["sim_time"] == 0 and s["status"] == "driving" for s in snaps)
    assert snaps[0]["path"] == snaps[1]["path"] != []
    made.tick(5)
    assert len(made.snapshot()["bodies"]) == 2


def test_a_new_map_forgets_the_goal_so_racing_on_stays_idle() -> None:
    made = race(scripted("lead"), scripted("ghost"))
    made.set_racing(False)
    made.goal(GOAL)
    made.new_map(SMALL_MAP)
    made.set_racing(True)
    assert all(s.snapshot()["status"] == "idle" for s in made.sessions)

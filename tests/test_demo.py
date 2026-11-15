"""The click-to-go session: map requests, controller loading and the headless session."""

import json
import math
from pathlib import Path

import pytest

from worm import demo
from worm.demo import MapRequest


def test_map_request_from_json_clamps_and_normalizes_mix() -> None:
    req = MapRequest.from_json(
        {"seed": 7, "size": 99, "mix": [2, 1, 1], "slope": 80, "rocks": 500, "ditches": -3}
    )
    assert req.seed == 7 and req.size == 15.0  # the largest map
    assert req.mix == pytest.approx((0.5, 0.25, 0.25))
    assert req.max_slope_deg == 20.0 and req.n_rocks == 40 and req.n_ditches == 0
    cfg = req.to_config()
    assert cfg.size == (15.0, 15.0) and cfg.n_rocks == 40 and cfg.max_slope_deg == 20.0


def test_map_request_defaults_fill_missing_keys() -> None:
    assert MapRequest.from_json({}) == MapRequest()
    assert MapRequest.from_json({"seed": 3}).seed == 3


@pytest.mark.parametrize(
    "garbage",
    [
        [1, 2, 3],
        "map",
        {"seed": "abc"},
        {"size": None},
        {"mix": [1, 2]},
        {"mix": [0, 0, 0]},
        {"mix": [1, -1, 1]},
        {"mix": "half"},
        {"rocks": 2.5},
        {"slope": float("nan")},
    ],
)
def test_map_request_rejects_garbage(garbage: object) -> None:
    with pytest.raises(ValueError):
        MapRequest.from_json(garbage)  # type: ignore[arg-type]


def test_load_controller_falls_back_to_scripted_when_no_policy_exists(tmp_path: Path) -> None:
    controller = demo.load_controller(
        None, search=(tmp_path / "policy_terrain", tmp_path / "policy")
    )
    assert controller.name == "scripted" and controller.obs_version == 1
    assert demo.load_controller("scripted").name == "scripted"
    with pytest.raises(FileNotFoundError):
        demo.load_controller(tmp_path / "missing")


def test_load_controller_prefers_the_first_policy_that_exists(tmp_path: Path) -> None:
    from worm.training import train

    train(steps=64, n_envs=1, seed=0, out=tmp_path / "policy")
    search = (tmp_path / "policy_terrain", tmp_path / "policy")
    controller = demo.load_controller(None, search=search)
    assert controller.name == "policy" and controller.obs_version == 1
    assert controller.policy is not None
    explicit = demo.load_controller(tmp_path / "policy")
    assert explicit.name == "policy"


def test_load_controller_reads_obs_version_from_meta_and_defaults_to_1(tmp_path: Path) -> None:
    assert demo.read_obs_version(tmp_path) == 1
    (tmp_path / "meta.json").write_text(json.dumps({"obs_version": 2, "obs_dim": 31}))
    assert demo.read_obs_version(tmp_path) == 2


# --- DemoSession ---

SMALL_MAP = MapRequest(seed=3, size=4.0, max_slope_deg=5.0, n_rocks=0, n_ditches=0)
SNAPSHOT_KEYS = {
    "map_id", "policy", "status", "failure", "head", "yaw", "body", "trail", "path", "waypoints",
    "reached", "goal", "replans", "surface", "sim_time",
}  # fmt: skip


def session(req: MapRequest = SMALL_MAP) -> demo.DemoSession:
    made = demo.DemoSession(demo.load_controller("scripted"))
    made.new_map(req)
    return made


def rock_at(made: demo.DemoSession) -> tuple[float, float]:
    world = made.world
    x, y, _ = world.rocks[0]
    return float(x), float(y)


def test_new_map_spawns_the_worm_at_the_centre() -> None:
    made = demo.DemoSession(demo.load_controller("scripted"))
    assert made.map_id == 0
    made.new_map(SMALL_MAP)
    snap = made.snapshot()
    assert made.map_id == 1 and snap["map_id"] == 1
    assert snap["head"] == pytest.approx([0.0, 0.0], abs=0.05)
    assert snap["status"] == "idle" and snap["policy"] == "scripted"
    made.new_map(MapRequest(seed=4, size=5.0))
    assert made.map_id == 2 and made.world.seed == 4


def test_goal_returns_a_plan_with_waypoints() -> None:
    made = session()
    answer = made.goal((1.0, 0.5))
    assert answer["ok"] is True
    plan = answer["plan"]
    assert plan["waypoints"][-1] == pytest.approx([1.0, 0.5])
    assert plan["path"][0] == pytest.approx([0.0, 0.0], abs=0.05)
    assert plan["cost"] > 0 and plan["goal_snapped"] is False
    assert made.snapshot()["status"] == "driving"
    assert made.snapshot()["goal"] == pytest.approx([1.0, 0.5])


def test_goal_inside_a_rock_returns_the_reason() -> None:
    made = session(MapRequest(seed=1, size=5.0, n_rocks=6, n_ditches=0))
    answer = made.goal(rock_at(made))
    assert answer == {"ok": False, "reason": "GOAL_BLOCKED", "detail": answer["detail"]}
    assert "obstacle" in answer["detail"]
    assert made.snapshot()["status"] == "idle"
    assert made.goal((99.0, 0.0))["reason"] == "OUT_OF_BOUNDS"


def test_tick_moves_the_worm_toward_the_goal() -> None:
    made = session()
    made.goal((1.0, 0.5))
    before = made.snapshot()
    made.tick(200)
    after = made.snapshot()

    def distance(snap: dict) -> float:  # type: ignore[type-arg]
        return math.dist(snap["head"], (1.0, 0.5))

    assert distance(after) < distance(before) - 0.3
    assert after["sim_time"] == pytest.approx(10.0) and len(after["trail"]) > 10


def test_tick_does_nothing_while_idle_and_stops_at_the_goal() -> None:
    made = session()
    made.tick(20)
    assert made.snapshot()["sim_time"] == 0.0
    made.goal((0.4, 0.0))
    made.tick(2000)
    snap = made.snapshot()
    assert snap["status"] == "reached" and snap["sim_time"] < 60.0
    assert math.dist(snap["head"], (0.4, 0.0)) < 0.15


def test_snapshot_is_json_serializable_and_has_the_documented_keys() -> None:
    made = session()
    assert set(made.snapshot()) == SNAPSHOT_KEYS
    made.goal((1.0, 0.5))
    made.tick(30)
    snap = json.loads(json.dumps(made.snapshot()))
    assert set(snap) == SNAPSHOT_KEYS
    assert len(snap["body"]) == 5 and len(snap["body"][0]) == 2
    assert snap["surface"] in {"concrete", "grass", "sand"}
    assert snap["reached"] == 0 and snap["replans"] == 0 and snap["failure"] is None


def test_a_new_goal_while_driving_replans_from_the_current_position() -> None:
    made = session()
    made.goal((1.0, 0.5))
    made.tick(100)
    head = made.snapshot()["head"]
    answer = made.goal((-1.0, -0.5))
    assert answer["ok"] is True
    assert answer["plan"]["path"][0] == pytest.approx(head, abs=1e-6)
    assert made.snapshot()["goal"] == pytest.approx([-1.0, -0.5])
    assert math.dist(head, (0.0, 0.0)) > 0.2  # it did not go back to the centre


def test_map_png_is_a_png_of_the_world_size() -> None:
    import io

    import imageio.v2 as imageio

    made = session()
    data = made.map_png()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    image = imageio.imread(io.BytesIO(data))
    info = made.map_info()
    assert image.shape[:2] == (info["height"], info["width"]) == (600, 600)
    assert info["extent"] == pytest.approx([-2.0, 2.0, -2.0, 2.0])
    assert info["px_per_m"] == pytest.approx(150.0) and info["map_id"] == 1
    assert made.map_png(blocked=True) != data  # the blocked overlay changes pixels


def test_frame_jpeg_is_none_without_rendering_and_a_jpeg_with_it(rendering: None) -> None:
    assert session().frame_jpeg() is None
    made = demo.DemoSession(demo.load_controller("scripted"), render=True)
    made.new_map(SMALL_MAP)
    frame = made.frame_jpeg()
    made.close()
    assert frame is not None and frame[:2] == b"\xff\xd8"


def test_a_failed_new_map_keeps_the_current_map(monkeypatch: pytest.MonkeyPatch) -> None:
    made = session()

    def broken(_world: object) -> None:
        raise RuntimeError("costmap failed")

    monkeypatch.setattr(demo.planner, "build_costmap", broken)
    with pytest.raises(RuntimeError, match="costmap failed"):
        made.new_map(MapRequest(seed=9, size=5.0))
    assert made.map_id == 1 and made.snapshot()["map_id"] == 1
    made.tick(1)  # the old map still works


def test_a_huge_integer_is_rejected_as_not_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        MapRequest.from_json({"size": 10**400})


def test_sampled_trail_matches_a_slice_of_the_full_trail() -> None:
    made = session()
    made.goal((1.0, 0.5))
    made.tick(30)
    nav = made._nav
    assert nav is not None
    assert nav.trail_length == len(nav.trail) > 10
    assert (nav.sampled_trail(4, 5) == nav.trail[::4][-5:]).all()


# --- observation v2 policies ---

REPO = Path(__file__).parent.parent
TERRAIN_POLICY = REPO / "results" / "policy_terrain"


def test_default_search_prefers_the_terrain_policy() -> None:
    search = [REPO / path for path in demo.POLICY_SEARCH]
    controller = demo.load_controller(None, search=search)
    assert controller.name == "policy_terrain" and controller.obs_version == 2


def test_session_drives_an_obs_v2_policy_to_the_goal() -> None:
    made = demo.DemoSession(demo.load_controller(TERRAIN_POLICY))
    made.new_map(SMALL_MAP)
    made.goal((1.0, 0.5))
    made.tick(400)
    snap = made.snapshot()
    assert snap["policy"] == "policy_terrain"
    assert snap["status"] == "reached", snap["failure"]


def test_session_rejects_an_unknown_observation_version() -> None:
    scripted = demo.load_controller("scripted")
    future = demo.Controller("future", scripted.policy, obs_version=99)
    with pytest.raises(ValueError, match="observation version 99"):
        demo.DemoSession(future)


def test_the_demo_module_does_not_need_imageio_to_load() -> None:
    """The browser has no imageio. It is only needed to encode a PNG or a JPEG."""
    import subprocess
    import sys

    code = "import sys, worm.demo; print('imageio' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"

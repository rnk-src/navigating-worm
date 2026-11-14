"""The click-to-go HTTP server, tested over real sockets on a free port."""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from worm import demo, server

SMALL_MAP = {"seed": 3, "size": 4, "slope": 5, "rocks": 0, "ditches": 0}


class Client:
    def __init__(self, loop: server.SimLoop, httpd: Any) -> None:
        self.loop, self.httpd = loop, httpd
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(self, path: str) -> tuple[int, str, bytes]:
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as reply:
                return reply.status, reply.headers.get("Content-Type", ""), reply.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get("Content-Type", ""), error.read()

    def post(self, path: str, body: Any, raw: bytes | None = None) -> tuple[int, Any]:
        data = raw if raw is not None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as reply:
                return reply.status, json.loads(reply.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def state(self) -> dict[str, Any]:
        return json.loads(self.get("/api/state")[2])  # type: ignore[no-any-return]

    def wait_for(self, check: Any, seconds: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            state = self.state()
            if check(state):
                return state
            time.sleep(0.02)
        raise AssertionError(f"the state never matched: {state}")


def start(render: bool = False) -> Client:
    session = demo.DemoSession(demo.load_controller("scripted"), render=render)
    session.new_map(demo.MapRequest.from_json(SMALL_MAP))
    loop = server.SimLoop(session)
    httpd = server.make_server(loop, port=0)
    loop.start()
    server.serve_in_thread(httpd)
    return Client(loop, httpd)


def stop(made: Client) -> None:
    made.httpd.shutdown()
    made.httpd.server_close()
    made.loop.stop()
    made.loop.join(timeout=10)


@pytest.fixture
def client() -> Iterator[Client]:
    made = start()
    yield made
    stop(made)


def test_get_root_serves_the_page_with_a_canvas(client: Client) -> None:
    status, content_type, body = client.get("/")
    assert status == 200 and content_type.startswith("text/html")
    assert b"<canvas" in body


def test_state_and_map_endpoints_describe_the_world(client: Client) -> None:
    state = client.state()
    assert state["status"] == "idle" and state["map_id"] == 1
    assert state["paused"] is False and state["speed"] == 1.0
    status, content_type, body = client.get("/api/map.png")
    assert status == 200 and content_type == "image/png" and body[:4] == b"\x89PNG"
    assert client.get("/api/map.png?blocked=1")[2] != body
    info = json.loads(client.get("/api/map.json")[2])
    assert info["extent"] == [-2.0, 2.0, -2.0, 2.0] and info["map_id"] == 1


def test_post_map_then_goal_then_state_shows_driving(client: Client) -> None:
    status, answer = client.post("/api/map", {**SMALL_MAP, "seed": 5})
    assert status == 200 and answer == {"ok": True, "map_id": 2}
    status, answer = client.post("/api/goal", {"x": 1.0, "y": 0.5})
    assert status == 200 and answer["ok"] is True and answer["plan"]["waypoints"]
    state = client.wait_for(lambda s: s["sim_time"] > 0.5)
    assert state["status"] == "driving" and state["map_id"] == 2
    assert state["goal"] == [1.0, 0.5]


def test_an_infeasible_goal_answers_with_the_reason(client: Client) -> None:
    status, answer = client.post("/api/goal", {"x": 50.0, "y": 0.0})
    assert status == 200
    assert answer["ok"] is False and answer["reason"] == "OUT_OF_BOUNDS" and answer["detail"]


def test_bad_json_returns_400(client: Client) -> None:
    assert client.post("/api/goal", None, raw=b"{not json")[0] == 400
    assert client.post("/api/goal", {"x": "left", "y": 0})[0] == 400
    assert client.post("/api/goal", [1, 2])[0] == 400
    assert client.post("/api/map", {"mix": [0, 0, 0]})[0] == 400
    status, answer = client.post("/api/control", {"speed": "fast"})
    assert status == 400 and answer["ok"] is False and answer["error"]
    assert client.state()["status"] == "idle"  # the server is still alive


def test_unknown_path_returns_404(client: Client) -> None:
    assert client.get("/api/nothing")[0] == 404
    assert client.get("/../pyproject.toml")[0] == 404
    assert client.post("/api/nothing", {})[0] == 404


def test_server_binds_localhost_by_default() -> None:
    session = demo.DemoSession(demo.load_controller("scripted"))
    session.new_map(demo.MapRequest.from_json(SMALL_MAP))
    httpd = server.make_server(server.SimLoop(session), port=0)
    try:
        assert httpd.server_address[0] == "127.0.0.1"
    finally:
        httpd.server_close()


def test_frame_endpoint_returns_204_without_rendering(client: Client) -> None:
    status, _, body = client.get("/api/frame.jpg")
    assert status == 204 and body == b""


def test_frame_endpoint_returns_a_jpeg_with_rendering(rendering: None) -> None:
    made = start(render=True)
    try:
        deadline = time.monotonic() + 20
        status, content_type, body = 204, "", b""
        while status == 204 and time.monotonic() < deadline:
            status, content_type, body = made.get("/api/frame.jpg")
            time.sleep(0.05)
        assert status == 200 and content_type == "image/jpeg" and body[:2] == b"\xff\xd8"
    finally:
        stop(made)


def test_pause_stops_the_sim_and_resume_continues(client: Client) -> None:
    client.post("/api/goal", {"x": 1.5, "y": 0.5})
    client.wait_for(lambda s: s["sim_time"] > 0.2)
    assert client.post("/api/control", {"pause": True}) == (200, {"ok": True})
    time.sleep(0.1)
    frozen = client.state()["sim_time"]
    time.sleep(0.3)
    assert client.state()["sim_time"] == frozen and client.state()["paused"] is True
    client.post("/api/control", {"pause": False})
    client.wait_for(lambda s: s["sim_time"] > frozen)


def test_loop_speed_control_changes_steps_per_second(client: Client) -> None:
    def sim_seconds_per_second(speed: float) -> float:
        client.post("/api/control", {"speed": speed})
        time.sleep(0.1)
        begin, wall = client.state()["sim_time"], time.monotonic()
        time.sleep(0.6)
        return (client.state()["sim_time"] - begin) / (time.monotonic() - wall)

    client.post("/api/goal", {"x": 1.6, "y": 1.6})
    slow, fast = sim_seconds_per_second(1.0), sim_seconds_per_second(8.0)
    assert 0.5 < slow < 1.6  # real time
    assert fast > 3.0 * slow
    assert client.post("/api/control", {"speed": 500})[0] == 200
    assert client.state()["speed"] == 20.0  # clamped


def test_a_huge_integer_goal_returns_400_not_500(client: Client) -> None:
    status, answer = client.post("/api/goal", None, raw=b'{"x": 1' + b"0" * 400 + b', "y": 0}')
    assert status == 400 and answer["ok"] is False


def test_a_cross_origin_post_is_refused(client: Client) -> None:
    def post(headers: dict[str, str]) -> int:
        request = urllib.request.Request(
            client.base + "/api/control",
            data=b'{"pause": false}',
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as reply:
                return int(reply.status)
        except urllib.error.HTTPError as error:
            return error.code

    assert post({"Origin": "http://evil.example"}) == 403
    assert post({"Host": "evil.example:8765"}) == 403  # DNS rebinding
    assert post({"Origin": client.base}) == 200
    assert post({}) == 200

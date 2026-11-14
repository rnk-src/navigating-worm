"""The click-to-go HTTP server: one sim thread owns the session, HTTP threads ask it."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from worm.demo import DemoSession, MapRequest

STATIC_DIR = Path(__file__).parent / "static"
SPEED_RANGE = (1.0, 20.0)  # sim seconds per wall second
_MAX_BODY = 64 * 1024  # bytes
_CALL_TIMEOUT = 30.0  # s


class SimLoop(threading.Thread):
    """Run the session at `speed` x real time. Only this thread touches the session."""

    def __init__(self, session: DemoSession, hz: float = 20.0) -> None:
        super().__init__(name="sim-loop", daemon=True)
        self._session = session
        self._period = 1.0 / hz
        self._commands: queue.Queue[tuple[Callable[[], Any], Future[Any]] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._paused = False
        self._speed = 1.0
        self._state: dict[str, Any] = {}
        self._frame: bytes | None = None
        self._stopping = threading.Event()

    @property
    def session(self) -> DemoSession:
        return self._session

    def submit(self, call: Callable[[], Any]) -> Any:
        """Run `call` on the sim thread and return its result or raise its error."""
        future: Future[Any] = Future()
        self._commands.put((call, future))
        return future.result(timeout=_CALL_TIMEOUT)

    def latest(self) -> dict[str, Any]:
        """Return the newest state, with `paused` and `speed` added."""
        with self._lock:
            return {**self._state, "paused": self._paused, "speed": self._speed}

    def latest_frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    def set_speed(self, speed: float) -> None:
        with self._lock:
            self._speed = min(max(speed, SPEED_RANGE[0]), SPEED_RANGE[1])

    def stop(self) -> None:
        self._stopping.set()
        self._commands.put(None)

    def _publish(self, with_frame: bool) -> None:
        try:
            state = self._session.snapshot()
        except RuntimeError:  # no map yet
            return
        frame = self._session.frame_jpeg() if with_frame else None
        with self._lock:
            self._state = state
            if frame is not None:
                self._frame = frame

    def _run_command(self, item: tuple[Callable[[], Any], Future[Any]]) -> None:
        call, future = item
        try:
            future.set_result(call())
        except Exception as error:  # hand every error to the caller
            future.set_exception(error)

    def run(self) -> None:
        budget = 0.0  # control steps owed
        last = time.monotonic()
        self._publish(with_frame=True)
        try:
            while not self._stopping.is_set():
                try:
                    item = self._commands.get(timeout=self._period)
                except queue.Empty:
                    item = None
                if item is not None:
                    self._run_command(item)
                now = time.monotonic()
                elapsed, last = now - last, now
                with self._lock:
                    paused, speed = self._paused, self._speed
                stepped = False
                if not paused and self._state:
                    dt = self._session.control_dt
                    budget = min(budget + elapsed * speed / dt, speed / dt)
                    steps = int(budget)
                    budget -= steps
                    if steps:
                        self._session.tick(steps)
                        stepped = True
                self._publish(with_frame=stepped or item is not None or self._frame is None)
        finally:
            self._session.close()


def _need_object(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    return body


def _finite(body: dict[str, Any], key: str) -> float:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep the console quiet."""

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        if status != 204:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if status != 204:
            self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, "application/json", json.dumps(data).encode())

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    def do_GET(self) -> None:
        url = urlparse(self.path)
        loop, session = self.server.loop, self.server.session
        try:
            if url.path == "/":
                page = (STATIC_DIR / "index.html").read_bytes()
                self._send(200, "text/html; charset=utf-8", page)
            elif url.path == "/api/state":
                self._json(200, loop.latest())
            elif url.path == "/api/map.png":
                blocked = parse_qs(url.query).get("blocked", ["0"])[0] == "1"
                self._send(200, "image/png", loop.submit(lambda: session.map_png(blocked)))
            elif url.path == "/api/map.json":
                self._json(200, loop.submit(session.map_info))
            elif url.path == "/api/frame.jpg":
                frame = loop.latest_frame()
                if frame is None:
                    self._send(204, "", b"")
                else:
                    self._send(200, "image/jpeg", frame)
            else:
                self._error(404, "not found")
        except RuntimeError as error:
            self._error(409, str(error))
        except Exception as error:
            self._error(500, f"{type(error).__name__}: {error}")

    def _read_json(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("bad Content-Length") from error
        if not 0 <= length <= _MAX_BODY:
            raise ValueError("the body is too large")
        try:
            return json.loads(self.rfile.read(length) or b"null")
        except json.JSONDecodeError as error:
            raise ValueError(f"bad JSON: {error}") from error

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/map", "/api/goal", "/api/control"):
            self._error(404, "not found")
            return
        loop, session = self.server.loop, self.server.session
        try:
            body = _need_object(self._read_json())
            if path == "/api/map":
                request = MapRequest.from_json(body)

                def build() -> int:
                    session.new_map(request)
                    return session.map_id

                self._json(200, {"ok": True, "map_id": loop.submit(build)})
            elif path == "/api/goal":
                xy = (_finite(body, "x"), _finite(body, "y"))
                self._json(200, loop.submit(lambda: session.goal(xy)))
            else:
                self._control(body)
        except ValueError as error:
            self._error(400, str(error))
        except RuntimeError as error:
            self._error(409, str(error))
        except Exception as error:
            self._error(500, f"{type(error).__name__}: {error}")

    def _control(self, body: dict[str, Any]) -> None:
        loop = self.server.loop
        if not ({"pause", "speed"} & body.keys()):
            raise ValueError("give pause or speed")
        if "pause" in body and not isinstance(body["pause"], bool):
            raise ValueError("pause must be true or false")
        speed = _finite(body, "speed") if "speed" in body else None
        if "pause" in body:
            loop.set_paused(body["pause"])
        if speed is not None:
            loop.set_speed(speed)
        self._json(200, {"ok": True})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    loop: SimLoop
    session: DemoSession


def make_server(loop: SimLoop, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Build the HTTP server. Port 0 picks a free port."""
    httpd = _Server((host, port), _Handler)
    httpd.loop = loop
    httpd.session = loop.session
    return httpd


def serve_in_thread(httpd: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=httpd.serve_forever, name="http", daemon=True)
    thread.start()
    return thread

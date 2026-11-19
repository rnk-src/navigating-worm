"""Glue between the page and the worm's own Python. It runs inside Pyodide.

`make_demo(policy_json)` returns a `BrowserDemo`. The page calls its methods and gets JSON
text back. The worm code runs unchanged: `mujoco` is the shim in mujoco.py and `gymnasium`
is the stub. Only the substep loop is replaced, so it runs in JavaScript.
"""

from __future__ import annotations

import json

import js
import numpy as np
from pyodide.ffi import to_js

from worm import friction, mapviz, npolicy
from worm.demo import Controller, DemoSession, MapRequest, RaceSession, load_controller
from worm.env import WormGoalEnv
from worm.spec import WormSpec

_python_advance = WormGoalEnv._advance


def _friction_key(f: friction.TerrainResistive) -> tuple:  # type: ignore[type-arg]
    return (
        id(f),
        id(f._normal),
        f._mu.tobytes(),
        f._viscous.tobytes(),
        None if f._timeconst is None else f._timeconst.tobytes(),
        f._weight.tobytes(),
        f._hybrid,
        f._stiction,
    )


def _friction_config(f: friction.TerrainResistive) -> dict:  # type: ignore[type-arg]
    rows, cols = f._shape
    return {
        "rows": rows,
        "cols": cols,
        "half": [float(f._half[0]), float(f._half[1])],
        "cell": float(f._cell),
        "normal": np.ascontiguousarray(f._normal, dtype=np.float64).ravel(),
        "surface": np.ascontiguousarray(f._surface, dtype=np.uint8).ravel(),
        "bodyIds": np.asarray(f._ids, dtype=np.int32),
        "pairIds": np.asarray(f._pairs, dtype=np.int32),
        "weight": np.asarray(f._weight, dtype=np.float64),
        "mu": np.ascontiguousarray(f._mu, dtype=np.float64).ravel(),
        "viscous": np.asarray(f._viscous, dtype=np.float64),
        "timeconst": None if f._timeconst is None else np.asarray(f._timeconst, dtype=np.float64),
        "hybrid": bool(f._hybrid),
        "stiction": float(f._stiction),
        "v0": float(f._v0),
    }


def _browser_advance(self: WormGoalEnv, start: np.ndarray, applied: np.ndarray) -> None:  # type: ignore[type-arg]
    """Run the substeps of one control step in JavaScript."""
    f = self._friction
    if not isinstance(f, friction.TerrainResistive):
        _python_advance(self, start, applied)
        return
    handle = self.model._h  # type: ignore[attr-defined]
    key = _friction_key(f)
    if getattr(self, "_js_friction_key", None) != key:
        handle.setFriction(to_js(_friction_config(f), dict_converter=js.Object.fromEntries))
        self._js_friction_key = key  # type: ignore[attr-defined]
    handle.advance(to_js(start), to_js(applied), self.spec_.substeps)
    self.data._pull()  # type: ignore[attr-defined]


WormGoalEnv._advance = _browser_advance  # type: ignore[method-assign]


class BrowserDemo:
    """One `DemoSession` or `RaceSession` with JSON in and JSON out."""

    def __init__(self, session: DemoSession | RaceSession) -> None:
        self._session = session

    def new_map(self, request_json: str) -> str:
        try:
            self._session.new_map(MapRequest.from_json(json.loads(request_json)))
        except ValueError as error:
            return json.dumps({"ok": False, "error": str(error)})
        return json.dumps({"ok": True, "map_id": self._session.map_id})

    def map_info(self) -> str:
        return json.dumps(self._session.map_info())

    def map_rgb(self) -> object:
        """The top-down map as raw RGB bytes, height x width x 3."""
        info = self._session.map_info()
        image = mapviz.world_image(self._session.world, info["px_per_m"])
        return to_js(np.ascontiguousarray(image, dtype=np.uint8).ravel())

    def goal(self, x: float, y: float) -> str:
        return json.dumps(self._session.goal((float(x), float(y))))

    def tick(self, n_steps: int) -> None:
        self._session.tick(int(n_steps))

    def snapshot(self) -> str:
        return json.dumps(self._session.snapshot())

    def set_racing(self, on: bool) -> None:
        if isinstance(self._session, RaceSession):
            self._session.set_racing(bool(on))

    @property
    def control_dt(self) -> float:
        return WormSpec().control_dt


def _controller(label: str, policy: object) -> Controller:
    if policy == "scripted":
        return Controller(label, load_controller("scripted").policy, 1)
    loaded = npolicy.NumpyPolicy.from_dict(policy)  # type: ignore[arg-type]
    return Controller(label, loaded, loaded.obs_version)


def make_demo(policy_json: str) -> BrowserDemo:
    return BrowserDemo(DemoSession(_controller("policy_terrain", json.loads(policy_json))))


def make_race(racers_json: str) -> BrowserDemo:
    """Build a race from `[{label, policy}]`. `policy` is a policy dict or "scripted".

    The first racer leads: the snapshot keeps its single-body fields.
    """
    racers = [_controller(r["label"], r["policy"]) for r in json.loads(racers_json)]
    return BrowserDemo(RaceSession(racers))

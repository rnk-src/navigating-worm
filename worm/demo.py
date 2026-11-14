"""The click-to-go demo without HTTP or threads: one world, one worm, goals on request."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from worm import mapviz, planner
from worm.controllers import Policy, bearing_steer
from worm.env import WormGoalEnv
from worm.navigate import NavConfig, Navigator
from worm.sensors import head_pose
from worm.spec import WormSpec, obs_layout
from worm.world import World, WorldConfig, generate

META_FILE = "meta.json"
POLICY_SEARCH = (Path("results/policy_terrain"), Path("results/policy"))
_SIZE_RANGE = (4.0, 15.0)  # m
_MAX_SLOPE = 20.0  # degrees
_MAX_ROCKS = 40
_MAX_DITCHES = 6
_MAP_PIXELS = 600  # the map image is this wide
_TRAIL_EVERY = 4  # keep every Nth head position in a snapshot
_TRAIL_POINTS = 1500


def _number(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _integer(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a whole number")
    return value


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


@dataclass(frozen=True)
class MapRequest:
    """What the user chooses for a new map. Lengths in metres, angles in degrees."""

    seed: int = 0
    size: float = 10.0
    mix: tuple[float, float, float] = (0.4, 0.3, 0.3)  # stone, grass, sand
    max_slope_deg: float = 12.0
    n_rocks: int = 12
    n_ditches: int = 2

    def to_config(self) -> WorldConfig:
        return WorldConfig(
            size=(self.size, self.size),
            mix=self.mix,
            max_slope_deg=self.max_slope_deg,
            n_rocks=self.n_rocks,
            n_ditches=self.n_ditches,
        )

    @staticmethod
    def from_json(data: dict[str, Any]) -> MapRequest:
        """Build a request from `{seed, size, mix: [c, g, s], slope, rocks, ditches}`.

        Out-of-range numbers are clamped. Anything that is not a number raises ValueError.
        """
        if not isinstance(data, dict):
            raise ValueError("the map request must be an object")
        default = MapRequest()
        mix = data.get("mix", list(default.mix))
        if not isinstance(mix, list | tuple) or len(mix) != 3:
            raise ValueError("mix must be three numbers")
        shares = [_number({"mix": m}, "mix", 0.0) for m in mix]
        if min(shares) < 0.0 or sum(shares) <= 0.0:
            raise ValueError("mix needs non-negative shares and one positive share")
        total = sum(shares)
        return MapRequest(
            seed=int(_clamp(_integer(data, "seed", default.seed), 0, 2**31 - 1)),
            size=_clamp(_number(data, "size", default.size), *_SIZE_RANGE),
            mix=(shares[0] / total, shares[1] / total, shares[2] / total),
            max_slope_deg=_clamp(_number(data, "slope", default.max_slope_deg), 0.0, _MAX_SLOPE),
            n_rocks=int(_clamp(_integer(data, "rocks", default.n_rocks), 0, _MAX_ROCKS)),
            n_ditches=int(_clamp(_integer(data, "ditches", default.n_ditches), 0, _MAX_DITCHES)),
        )


@dataclass(frozen=True)
class Controller:
    name: str  # shown in the page: "scripted" or the policy folder name
    policy: Policy
    obs_version: int


def read_obs_version(path: Path) -> int:
    """Return the observation version in the policy's `meta.json`, or 1 without one."""
    meta = path / META_FILE
    if not meta.exists():
        return 1
    return int(json.loads(meta.read_text()).get("obs_version", 1))


def _scripted() -> Controller:
    return Controller("scripted", bearing_steer(obs_layout(WormSpec())), 1)


def _trained(path: Path) -> Controller:
    from worm.training import load_policy  # imports torch; the scripted demo does not need it

    return Controller(path.name, load_policy(path), read_obs_version(path))


def load_controller(
    choice: str | Path | None, search: Sequence[Path] = POLICY_SEARCH
) -> Controller:
    """Load the controller for the demo.

    `choice` is a policy folder or "scripted". Without a choice, take the first folder
    in `search` that holds a policy, and fall back to the scripted controller.
    """
    if choice is not None:
        if str(choice) == "scripted":
            return _scripted()
        path = Path(choice)
        if not (path / "policy.zip").exists():
            raise FileNotFoundError(f"no policy in {path}")
        return _trained(path)
    for path in search:
        if (path / "policy.zip").exists():
            return _trained(path)
    return _scripted()


def _points(array: Any) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(array).reshape(-1, 2)]


class DemoSession:
    """One world, one worm and its navigator. Not thread-safe: one thread owns it.

    `new_map` builds a world and puts the worm at its centre. `goal` plans a route from
    where the worm is. `tick` drives. `snapshot` returns everything a page needs to draw.
    """

    def __init__(self, controller: Controller, render: bool = False) -> None:
        if controller.obs_version != 1:
            raise ValueError(f"observation version {controller.obs_version} is not supported")
        self._controller = controller
        self._render = render
        self._map_id = 0
        self._env: WormGoalEnv | None = None
        self._nav: Navigator | None = None
        self._world: World | None = None
        self._costmap: planner.CostMap | None = None
        self._goal: list[float] | None = None
        self._png: dict[bool, bytes] = {}

    @property
    def map_id(self) -> int:
        """Return a counter that grows with every new map."""
        return self._map_id

    @property
    def control_dt(self) -> float:
        """Return the seconds of sim time in one control step."""
        return self._parts()[0].spec_.control_dt

    @property
    def world(self) -> World:
        if self._world is None:
            raise RuntimeError("no map yet: call new_map() first")
        return self._world

    def _parts(self) -> tuple[WormGoalEnv, Navigator, World, planner.CostMap]:
        if self._env is None or self._nav is None or self._costmap is None:
            raise RuntimeError("no map yet: call new_map() first")
        return self._env, self._nav, self.world, self._costmap

    def new_map(self, req: MapRequest) -> None:
        """Generate a world and spawn the worm at its centre."""
        self.close()
        world = generate(req.to_config(), req.seed)
        env = WormGoalEnv(
            world=world,
            episode_seconds=NavConfig().waypoint_timeout,
            render_mode="rgb_array" if self._render else None,
        )
        env.reset(seed=req.seed, options={"start": (0.0, 0.0)})
        costmap = planner.build_costmap(world)
        self._env, self._world, self._costmap = env, world, costmap
        self._nav = Navigator(env, self._controller.policy, costmap)
        self._goal = None
        self._png = {}
        self._map_id += 1

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def goal(self, xy: tuple[float, float]) -> dict[str, Any]:
        """Plan a route to `xy` from where the worm is. Return a JSON-able answer."""
        _, nav, _, _ = self._parts()
        result = nav.go(xy)
        if isinstance(result, planner.Infeasible):
            return {"ok": False, "reason": result.reason.name, "detail": result.detail}
        self._goal = [float(result.path[-1][0]), float(result.path[-1][1])]
        return {
            "ok": True,
            "plan": {
                "path": _points(result.path),
                "waypoints": _points(result.waypoints),
                "cost": round(result.cost, 2),
                "goal_snapped": result.goal_snapped,
            },
        }

    def tick(self, n_steps: int = 1) -> None:
        """Drive up to `n_steps` control steps. Do nothing when there is no goal."""
        _, nav, _, _ = self._parts()
        for _ in range(n_steps):
            if nav.status != "driving":
                break
            nav.step()

    def snapshot(self) -> dict[str, Any]:
        """Return the state of the demo as plain JSON types."""
        env, nav, _, _ = self._parts()
        _, yaw, _, _ = head_pose(env.data, 1)
        trail = nav.trail
        plan = nav.plan
        return {
            "map_id": self._map_id,
            "policy": self._controller.name,
            "status": nav.status,
            "failure": nav.failure,
            "head": _points(env.head_xy)[0],
            "yaw": round(float(yaw), 3),
            "body": _points(env.data.xpos[1 : 1 + env.spec_.n_segments, :2]),
            "trail": _points(trail[::_TRAIL_EVERY][-_TRAIL_POINTS:]),
            "path": [] if plan is None else _points(plan.path),
            "waypoints": [] if plan is None else _points(plan.waypoints),
            "reached": nav.reached,
            "goal": self._goal,
            "replans": nav.replans,
            "surface": env.surface_under_head.name.lower(),
            "sim_time": round(len(trail) * env.spec_.control_dt, 2),
        }

    def map_info(self) -> dict[str, Any]:
        """Return what a page needs to convert between map pixels and world metres."""
        world = self.world
        xmin, xmax, ymin, ymax = world.extent
        px_per_m = _MAP_PIXELS / (xmax - xmin)
        height, width = mapviz.image_size(world, px_per_m)
        return {
            "map_id": self._map_id,
            "extent": [xmin, xmax, ymin, ymax],
            "px_per_m": px_per_m,
            "width": width,
            "height": height,
        }

    def map_png(self, blocked: bool = False) -> bytes:
        """Return the top-down map as PNG bytes. `blocked` tints the cells no route may use."""
        if blocked not in self._png:
            _, _, world, costmap = self._parts()
            px_per_m = self.map_info()["px_per_m"]
            image = mapviz.world_image(world, px_per_m)
            if blocked:
                mapviz.draw_blocked(image, world, costmap, px_per_m)
            self._png[blocked] = imageio.imwrite("<bytes>", image, format="png")
        return self._png[blocked]

    def frame_jpeg(self) -> bytes | None:
        """Return the follow-camera view as JPEG bytes, or None without rendering."""
        if not self._render or self._env is None:
            return None
        frame = self._env.render()
        if frame is None:
            return None
        encoded: bytes = imageio.imwrite("<bytes>", frame, format="jpg")
        return encoded

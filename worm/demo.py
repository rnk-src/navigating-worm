"""The click-to-go demo without HTTP or threads: one world, one worm, goals on request."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worm.controllers import Policy, bearing_steer
from worm.spec import WormSpec, obs_layout
from worm.world import WorldConfig

META_FILE = "meta.json"
POLICY_SEARCH = (Path("results/policy_terrain"), Path("results/policy"))
_SIZE_RANGE = (4.0, 15.0)  # m
_MAX_SLOPE = 20.0  # degrees
_MAX_ROCKS = 40
_MAX_DITCHES = 6


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

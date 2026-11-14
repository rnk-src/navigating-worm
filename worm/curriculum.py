"""Training curriculum: worlds that get harder level by level. Pure numpy, no MuJoCo.

Level 0 is level ground of one class. Level 1 adds mixed surface patches. Level 2 adds
hills and fine roughness. Level 3 adds steeper hills, shallow ditches and rocks.
The band around the line from the start to the target never holds a rock: the planner
owns obstacle avoidance and the policy owns the local terrain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from worm import world as wd
from worm.terrain import TerrainClass
from worm.world import World, WorldConfig

Array = NDArray[np.float64]
N_LEVELS = 4
# Returns the world of one episode from (rng, start, target). Every world has one shape.
WorldSampler = Callable[[np.random.Generator, Array, Array], World]


@dataclass
class WorldCurriculum:
    """A world sampler with a level. Call it as `curriculum(rng, start, target)`.

    Each call draws the level of the world: the current level with probability
    `1 - replay_lower`, otherwise a lower level, uniformly. The replay keeps the
    policy from forgetting the easier ground.
    """

    size: float = 8.0  # m, side length; every level has the same grid shape
    cell: float = 0.05  # m
    replay_lower: float = 0.3
    level: int = 0
    corridor: float = 0.4  # m, half-width of the rock-free band from the start to the target
    patch_scale: tuple[float, float] = (0.5, 2.0)  # m
    slope_l2: tuple[float, float] = (4.0, 10.0)  # degrees, cap on the hills at level 2
    slope_l3: tuple[float, float] = (10.0, 18.0)  # degrees, cap on the hills at level 3
    hill_height_l3: tuple[float, float] = (0.3, 0.6)  # m
    # The deepest ditch stays below the height step that the planner blocks.
    ditch_depth: tuple[float, float] = (0.004, 0.010)  # m
    ditch_width: tuple[float, float] = (0.3, 0.5)  # m
    n_ditches: int = 2
    rock_density: float = 0.1  # rocks per square metre at level 3
    last_level: int = field(default=0, init=False)  # the level of the newest world

    def __post_init__(self) -> None:
        self.set_level(self.level)

    def set_level(self, level: int) -> None:
        if not 0 <= level < N_LEVELS:
            raise ValueError(f"level {level} is outside 0..{N_LEVELS - 1}")
        self.level = int(level)

    def draw_level(self, rng: np.random.Generator) -> int:
        if self.level > 0 and rng.random() < self.replay_lower:
            return int(rng.integers(self.level))
        return self.level

    def __call__(self, rng: np.random.Generator, start: Array, target: Array) -> World:
        self.last_level = self.draw_level(rng)
        return self.sampler(self.last_level)(rng, start, target)

    def sampler(self, level: int) -> WorldSampler:
        """Return the sampler of one fixed level."""
        if not 0 <= level < N_LEVELS:
            raise ValueError(f"level {level} is outside 0..{N_LEVELS - 1}")

        def sample(rng: np.random.Generator, start: Array, target: Array) -> World:
            return self._world(level, rng, start, target)

        return sample

    def _world(self, level: int, rng: np.random.Generator, start: Array, target: Array) -> World:
        side = (self.size, self.size)
        if level == 0:
            cls = TerrainClass(int(rng.integers(len(TerrainClass))))
            return wd.flat(side, self.cell, cls)
        cfg = WorldConfig(
            size=side,
            cell=self.cell,
            hill_height=0.0,
            roughness=0.0,
            n_rocks=0,
            n_ditches=0,
            patch_scale=float(rng.uniform(*self.patch_scale)),
        )
        if level == 2:
            slope = float(rng.uniform(*self.slope_l2))
            cfg = _with(cfg, hill_height=0.3, roughness=1.0, max_slope_deg=slope)
        if level == 3:
            cfg = _with(
                cfg,
                hill_height=float(rng.uniform(*self.hill_height_l3)),
                roughness=1.0,
                max_slope_deg=float(rng.uniform(*self.slope_l3)),
                n_rocks=round(self.rock_density * self.size**2),
                rock_radius=(0.08, 0.25),
                n_ditches=self.n_ditches,
                ditch_depth=self.ditch_depth,
                ditch_width=self.ditch_width,
            )
        seed = int(rng.integers(2**31))
        return wd.generate(cfg, seed, corridor=(start, target, self.corridor))


def _with(cfg: WorldConfig, **changes: Any) -> WorldConfig:
    return replace(cfg, **changes)

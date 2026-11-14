"""Global planner: turn a world and a goal into a waypoint route, or say why not.

Pure Python and numpy. No simulation, so it runs headless and fast.
Cells are (row, column): the row runs along +y and the column along +x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from worm.world import World

Cell = tuple[int, int]


@dataclass(frozen=True)
class CostConfig:
    """How terrain turns into travel cost. Lengths in metres, angles in degrees."""

    resolution: float = 0.1  # planning grid; the world grid is pooled down to this
    surface_cost: tuple[float, float, float] = (1.0, 1.4, 2.0)  # per TerrainClass, 1 / speed
    slope_weight: float = 1.0
    max_slope_deg: float = 20.0  # steeper cells are blocked
    max_step: float = 0.012  # a larger height step to a neighbour cell is blocked
    inflate_radius: float = 0.15  # blocked cells grow by this: half a body width plus the sway
    edge_margin: float = 0.4  # blocked band along the world edge; the env stops at 0.3 m


@dataclass(frozen=True, eq=False)
class CostMap:
    cost: NDArray[np.float64]  # (rows, cols), np.inf where blocked
    resolution: float
    origin: tuple[float, float]  # xy of the centre of cell (0, 0)

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.cost.shape
        return int(rows), int(cols)

    def to_cell(self, xy: ArrayLike) -> Cell:
        """Return the nearest cell. It may lie outside the grid."""
        x, y = np.asarray(xy, dtype=np.float64)
        return (
            round((float(y) - self.origin[1]) / self.resolution),
            round((float(x) - self.origin[0]) / self.resolution),
        )

    def to_xy(self, rc: ArrayLike) -> NDArray[np.float64]:
        cell = np.asarray(rc, dtype=np.float64)
        return np.stack(
            [
                self.origin[0] + cell[..., 1] * self.resolution,
                self.origin[1] + cell[..., 0] * self.resolution,
            ],
            axis=-1,
        )

    def inside(self, rc: Cell) -> bool:
        rows, cols = self.shape
        return 0 <= rc[0] < rows and 0 <= rc[1] < cols

    def free(self, rc: Cell) -> bool:
        """Return True when the cell is inside the grid and not blocked."""
        return self.inside(rc) and bool(np.isfinite(self.cost[rc]))


def _max_step(heights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the largest height difference from each cell to its 8 neighbours."""
    padded = np.pad(heights, 1, mode="edge")
    rows, cols = heights.shape
    step = np.zeros_like(heights)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                shifted = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
                step = np.maximum(step, np.abs(shifted - heights))
    return step


def _pool(grid: NDArray[np.float64], factor: int, reduce: str) -> NDArray[np.float64]:
    """Pool `factor` x `factor` blocks with max or mean. Edge blocks may be partial."""
    rows, cols = grid.shape
    out_rows, out_cols = math.ceil(rows / factor), math.ceil(cols / factor)
    padded = np.full((out_rows * factor, out_cols * factor), np.nan)
    padded[:rows, :cols] = grid
    blocks = padded.reshape(out_rows, factor, out_cols, factor)
    pooled: NDArray[np.float64] = (
        np.nanmax(blocks, axis=(1, 3)) if reduce == "max" else np.nanmean(blocks, axis=(1, 3))
    )
    return pooled


def _dilate(mask: NDArray[np.bool_], radius_cells: float) -> NDArray[np.bool_]:
    """Grow the True cells by a disc of `radius_cells`."""
    reach = int(math.floor(radius_cells))
    rows, cols = mask.shape
    padded = np.pad(mask, reach, mode="constant")
    grown = mask.copy()
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if dr * dr + dc * dc <= radius_cells * radius_cells:
                grown |= padded[reach + dr : reach + dr + rows, reach + dc : reach + dc + cols]
    return grown


def build_costmap(world: World, cfg: CostConfig = CostConfig()) -> CostMap:
    """Return the travel cost of every planning cell. Blocked cells cost infinity.

    A world cell is blocked when it is too steep, when the height step to a neighbour
    is too large (rock flanks, ditch walls), or when it is a rock. Blocked cells then
    grow by `inflate_radius`, and a band along the world edge is blocked.
    """
    heights = world.heights.astype(np.float64)
    slope = world.slope_deg().astype(np.float64)
    blocked = (slope > cfg.max_slope_deg) | (_max_step(heights) > cfg.max_step) | world.obstacle
    cost = np.asarray(cfg.surface_cost)[world.surface] * (
        1.0 + cfg.slope_weight * (slope / cfg.max_slope_deg) ** 2
    )

    factor = max(1, round(cfg.resolution / world.cell))
    resolution = factor * world.cell
    coarse_cost = _pool(cost, factor, "mean")
    coarse_blocked = _pool(blocked.astype(np.float64), factor, "max") > 0.0
    if cfg.inflate_radius > 0.0:
        coarse_blocked = _dilate(coarse_blocked, cfg.inflate_radius / resolution)

    first = world.to_xy((0.5 * (factor - 1), 0.5 * (factor - 1)))
    origin = (float(first[0]), float(first[1]))
    if cfg.edge_margin > 0.0:
        rx, ry = world.half_extent
        rows, cols = coarse_cost.shape
        xs = origin[0] + np.arange(cols) * resolution
        ys = origin[1] + np.arange(rows) * resolution
        coarse_blocked |= (rx - np.abs(xs) < cfg.edge_margin)[None, :]
        coarse_blocked |= (ry - np.abs(ys) < cfg.edge_margin)[:, None]
    coarse_cost[coarse_blocked] = np.inf
    return CostMap(coarse_cost, resolution, origin)

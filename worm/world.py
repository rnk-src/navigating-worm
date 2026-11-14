"""A terrain world: a grid of heights and surface classes. Pure numpy, no MuJoCo.

Grid points sit on cell centres. Row index runs along +y and column index along +x.
The world is centred on the origin, so the first and last grid points lie on the extent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from worm.terrain import TerrainClass

_MIN_ZMAX = 1e-3  # m, a MuJoCo hfield needs a positive height range


@dataclass(frozen=True, eq=False)
class World:
    heights: NDArray[np.float32]  # (rows, cols), metres, min 0
    surface: NDArray[np.uint8]  # (rows, cols), TerrainClass values
    obstacle: NDArray[np.bool_]  # (rows, cols), rocks (ground truth, for viz and tests)
    cell: float
    seed: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.heights.shape
        return int(rows), int(cols)

    @property
    def half_extent(self) -> tuple[float, float]:
        """Return (x, y) distance from the origin to the outermost grid points."""
        rows, cols = self.shape
        return 0.5 * (cols - 1) * self.cell, 0.5 * (rows - 1) * self.cell

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax)."""
        rx, ry = self.half_extent
        return -rx, rx, -ry, ry

    def _grid(self, xy: ArrayLike) -> NDArray[np.float64]:
        """Return fractional (..., [row, col]) grid coordinates, not clipped."""
        point = np.asarray(xy, dtype=np.float64)
        rx, ry = self.half_extent
        return np.stack([(point[..., 1] + ry), (point[..., 0] + rx)], axis=-1) / self.cell

    def to_cell(self, xy: ArrayLike) -> NDArray[np.int_]:
        """Return the nearest (..., [row, col]) cell, clipped to the grid."""
        rows, cols = self.shape
        cell = np.rint(self._grid(xy)).astype(np.int_)
        return np.clip(cell, 0, [rows - 1, cols - 1])

    def to_xy(self, rc: ArrayLike) -> NDArray[np.float64]:
        cell = np.asarray(rc, dtype=np.float64)
        rx, ry = self.half_extent
        return np.stack([cell[..., 1] * self.cell - rx, cell[..., 0] * self.cell - ry], axis=-1)

    def contains(self, xy: ArrayLike) -> NDArray[np.bool_]:
        point = np.asarray(xy, dtype=np.float64)
        rx, ry = self.half_extent
        inside: NDArray[np.bool_] = (np.abs(point[..., 0]) <= rx) & (np.abs(point[..., 1]) <= ry)
        return inside

    def _bilinear(self, grid: NDArray[np.floating], xy: ArrayLike) -> NDArray[np.float64]:
        """Interpolate `grid` (rows, cols, ...) at `xy`. Points outside clamp to the edge."""
        rows, cols = self.shape
        frac = np.clip(self._grid(xy), 0.0, [rows - 1, cols - 1])
        low = np.minimum(np.floor(frac).astype(np.int_), [max(rows - 2, 0), max(cols - 2, 0)])
        high = np.minimum(low + 1, [rows - 1, cols - 1])
        t = frac - low
        tr, tc = t[..., 0], t[..., 1]
        if grid.ndim == 3:
            tr, tc = tr[..., None], tc[..., None]
        r0, c0, r1, c1 = low[..., 0], low[..., 1], high[..., 0], high[..., 1]
        top = grid[r0, c0] * (1 - tc) + grid[r0, c1] * tc
        bottom = grid[r1, c0] * (1 - tc) + grid[r1, c1] * tc
        value: NDArray[np.float64] = np.asarray(top * (1 - tr) + bottom * tr, dtype=np.float64)
        return value

    def height_at(self, xy: ArrayLike) -> NDArray[np.float64]:
        """Return the bilinear height at `xy`."""
        return self._bilinear(self.heights, xy)

    def class_at(self, xy: ArrayLike) -> NDArray[np.uint8]:
        """Return the class of the nearest cell."""
        cell = self.to_cell(xy)
        classes: NDArray[np.uint8] = self.surface[cell[..., 0], cell[..., 1]]
        return classes

    @cached_property
    def gradient(self) -> NDArray[np.float64]:
        """Return (rows, cols, 2) central-difference slopes [dh/dx, dh/dy]."""
        heights = self.heights.astype(np.float64)
        rows, cols = self.shape
        d_row = np.gradient(heights, self.cell, axis=0) if rows > 1 else np.zeros_like(heights)
        d_col = np.gradient(heights, self.cell, axis=1) if cols > 1 else np.zeros_like(heights)
        return np.stack([d_col, d_row], axis=-1)

    def normal_at(self, xy: ArrayLike) -> NDArray[np.float64]:
        """Return the (..., 3) unit surface normal at `xy`."""
        slope = self._bilinear(self.gradient, xy)
        normal = np.concatenate([-slope, np.ones((*slope.shape[:-1], 1))], axis=-1)
        unit: NDArray[np.float64] = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        return unit

    def slope_deg(self) -> NDArray[np.float32]:
        """Return the slope of each cell in degrees."""
        steepness = np.hypot(self.gradient[..., 0], self.gradient[..., 1])
        degrees: NDArray[np.float32] = np.degrees(np.arctan(steepness)).astype(np.float32)
        return degrees

    def to_hfield_data(self) -> tuple[NDArray[np.float32], float]:
        """Return (normalized row-major heights, zmax) for a MuJoCo hfield.

        This is the only place that maps the grid to MuJoCo's hfield layout.
        MuJoCo puts row 0 at -y and column 0 at -x, which matches this grid.
        """
        zmax = max(float(self.heights.max()), _MIN_ZMAX)
        data: NDArray[np.float32] = (self.heights / zmax).astype(np.float32).ravel()
        return data, zmax


def flat(
    size: tuple[float, float], cell: float = 0.05, cls: TerrainClass = TerrainClass.CONCRETE
) -> World:
    """Return a level world of `size` (x, y) metres with one surface class."""
    shape = (round(size[1] / cell) + 1, round(size[0] / cell) + 1)
    return World(
        heights=np.zeros(shape, dtype=np.float32),
        surface=np.full(shape, int(cls), dtype=np.uint8),
        obstacle=np.zeros(shape, dtype=np.bool_),
        cell=cell,
    )

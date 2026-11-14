"""A terrain world: a grid of heights and surface classes. Pure numpy, no MuJoCo.

Grid points sit on cell centres. Row index runs along +y and column index along +x.
The world is centred on the origin, so the first and last grid points lie on the extent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from worm.terrain import ROUGHNESS, TerrainClass

_MIN_ZMAX = 1e-3  # m, a MuJoCo hfield needs a positive height range
_ROCK_GAP = 0.4  # m, free space kept between two rocks so that the worm can pass
_SPAWN_BLEND = 0.5  # m, width of the band where the relief fades into the flat spawn area
_ROUGHNESS_SCALE = 0.1  # m, correlation length of the fine relief
_PLACE_TRIES = 40


@dataclass(frozen=True)
class WorldConfig:
    """Parameters of a generated world. Lengths are in metres, angles in degrees."""

    size: tuple[float, float] = (10.0, 10.0)
    cell: float = 0.05
    hill_height: float = 0.3  # peak-to-peak of the low-frequency relief
    hill_scale: float = 2.0  # correlation length of the hills
    max_slope_deg: float = 15.0  # the relief is scaled down until its steepest slope fits
    roughness: float = 1.0  # multiplier on the per-class roughness amplitude
    n_rocks: int = 10
    rock_radius: tuple[float, float] = (0.08, 0.3)
    rock_height: tuple[float, float] = (0.05, 0.2)
    n_ditches: int = 2
    ditch_width: tuple[float, float] = (0.15, 0.5)
    ditch_depth: tuple[float, float] = (0.04, 0.15)
    ditch_length: tuple[float, float] = (1.0, 4.0)
    mix: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)  # area share CONCRETE, GRASS, SAND
    patch_scale: float = 1.5  # correlation length of the surface patches
    clear_radius: float = 0.5  # no obstacle and no slope this close to the spawn point (0, 0)


@dataclass(frozen=True, eq=False)
class World:
    heights: NDArray[np.float32]  # (rows, cols), metres, min 0
    surface: NDArray[np.uint8]  # (rows, cols), TerrainClass values
    obstacle: NDArray[np.bool_]  # (rows, cols), rocks (ground truth, for viz and tests)
    cell: float
    seed: int | None = None
    # (n, 3) rock centres and radii [x, y, radius], metres.
    rocks: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))

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


def smooth_noise(
    shape: tuple[int, int], scale_cells: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return Gaussian-filtered white noise with zero mean and unit standard deviation.

    `scale_cells` is the correlation length in cells. The field wraps at the edges.
    """
    white = rng.standard_normal(shape)
    f_row = np.fft.fftfreq(shape[0])[:, None]
    f_col = np.fft.rfftfreq(shape[1])[None, :]
    kernel = np.exp(-2.0 * (math.pi * scale_cells) ** 2 * (f_row**2 + f_col**2))
    field_: NDArray[np.float64] = np.fft.irfft2(np.fft.rfft2(white) * kernel, s=shape)
    field_ -= field_.mean()
    std = float(field_.std())
    return field_ / std if std > 1e-12 else field_


def _smoothstep(x: NDArray[np.float64]) -> NDArray[np.float64]:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _relief(
    cfg: WorldConfig, shape: tuple[int, int], spawn: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return hills that fade to level ground at the spawn point and respect the slope cap."""
    if cfg.hill_height <= 0.0:
        return np.zeros(shape)
    hills = smooth_noise(shape, cfg.hill_scale / cfg.cell, rng)
    span = float(np.ptp(hills))
    if span < 1e-12:
        return np.zeros(shape)
    hills *= cfg.hill_height / span
    centre = hills[shape[0] // 2, shape[1] // 2]
    hills = (hills - centre) * spawn
    d_row, d_col = np.gradient(hills, cfg.cell)
    steepest = float(np.hypot(d_row, d_col).max())
    limit = math.tan(math.radians(cfg.max_slope_deg))
    return hills * min(1.0, limit / steepest) if steepest > 0.0 else hills


def _surface(
    cfg: WorldConfig, shape: tuple[int, int], rng: np.random.Generator
) -> NDArray[np.uint8]:
    """Return patches whose area shares match `cfg.mix` exactly, up to one cell."""
    mix = np.clip(np.asarray(cfg.mix, dtype=np.float64), 0.0, None)
    if mix.sum() <= 0.0:
        raise ValueError("the surface mix needs one positive share")
    mix /= mix.sum()
    surface = np.full(shape, len(mix) - 1, dtype=np.uint8)
    free = np.ones(shape, dtype=np.bool_)
    remaining = 1.0
    for cls in range(len(mix) - 1):
        noise = smooth_noise(shape, cfg.patch_scale / cfg.cell, rng)
        share = mix[cls] / remaining if remaining > 1e-12 else 0.0
        if share >= 1.0 - 1e-12:
            chosen = free.copy()
        elif share <= 0.0:
            chosen = np.zeros(shape, dtype=np.bool_)
        else:
            chosen = free & (noise >= np.quantile(noise[free], 1.0 - share))
        surface[chosen] = cls
        free &= ~chosen
        remaining -= mix[cls]
    return surface


def _ditches(
    cfg: WorldConfig, xy: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return the depth map of the ditches: flat-bottomed trenches with cosine walls."""
    depth = np.zeros(xy.shape[:2])
    half = 0.5 * np.array(cfg.size)
    for _ in range(cfg.n_ditches):
        for _ in range(_PLACE_TRIES):
            width = float(rng.uniform(*cfg.ditch_width))
            length = float(rng.uniform(*cfg.ditch_length))
            deep = float(rng.uniform(*cfg.ditch_depth))
            centre = rng.uniform(-half, half)
            angle = float(rng.uniform(0.0, math.pi))
            along = np.array([math.cos(angle), math.sin(angle)])
            across = np.array([-along[1], along[0]])
            # Distance from the spawn point to the ditch centre line.
            s = float(np.clip(-centre @ along, -0.5 * length, 0.5 * length))
            if np.linalg.norm(centre + s * along) < cfg.clear_radius + 0.5 * width + cfg.cell:
                continue
            offset = xy - centre
            u = np.abs(offset @ across) / (0.5 * width)  # 0 at the centre line, 1 at the rim
            v = (np.abs(offset @ along) - 0.5 * length) / (0.5 * width) + 1.0  # 1 at the end
            edge = np.maximum(u, v)
            wall = 0.5 * (1.0 + np.cos(math.pi * np.clip((edge - 0.5) / 0.5, 0.0, 1.0)))
            depth = np.maximum(depth, deep * wall)
            break
    return depth


def _rocks(
    cfg: WorldConfig, world_xy: NDArray[np.float64], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Return (height added, obstacle mask, rocks as [x, y, radius]). Rocks are smooth caps."""
    added = np.zeros(world_xy.shape[:2])
    placed: list[tuple[float, float, float]] = []
    half = 0.5 * np.array(cfg.size)
    for _ in range(cfg.n_rocks):
        for _ in range(_PLACE_TRIES):
            radius = float(rng.uniform(*cfg.rock_radius))
            height = float(rng.uniform(*cfg.rock_height))
            # Snap the centre to a grid point so that the peak reaches the full height.
            centre = np.rint((rng.uniform(-half, half) + half) / cfg.cell) * cfg.cell - half
            if np.linalg.norm(centre) < cfg.clear_radius + radius + cfg.cell:
                continue
            if any(
                math.hypot(centre[0] - x, centre[1] - y) < radius + r + _ROCK_GAP
                for x, y, r in placed
            ):
                continue
            ratio = np.linalg.norm(world_xy - centre, axis=-1) / radius
            cap = height * np.sqrt(np.clip(1.0 - ratio**2, 0.0, None))
            added = np.maximum(added, cap)
            placed.append((float(centre[0]), float(centre[1]), radius))
            break
    rocks = np.array(placed, dtype=np.float64).reshape(-1, 3)
    return added, added > 0.0, rocks


def generate(cfg: WorldConfig, seed: int) -> World:
    """Return a seeded world: hills, fine relief, surface patches, ditches and rocks.

    Each feature draws from its own random stream. Changing the rock count therefore
    leaves the hills, the patches and the ditches of a seed unchanged.
    """
    base = flat(cfg.size, cfg.cell)
    shape = base.shape
    relief_rng, rough_rng, surface_rng, ditch_rng, rock_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)
    )
    xy = base.to_xy(np.stack(np.indices(shape), axis=-1))
    spawn = _smoothstep((np.linalg.norm(xy, axis=-1) - cfg.clear_radius) / _SPAWN_BLEND)

    surface = _surface(cfg, shape, surface_rng)
    heights = _relief(cfg, shape, spawn, relief_rng)
    amplitude = cfg.roughness * np.array([ROUGHNESS[c] for c in TerrainClass])[surface]
    fine = smooth_noise(shape, _ROUGHNESS_SCALE / cfg.cell, rough_rng)
    heights += np.clip(0.5 * fine, -1.0, 1.0) * amplitude * spawn
    heights -= _ditches(cfg, xy, ditch_rng)
    added, obstacle, rocks = _rocks(cfg, xy, rock_rng)
    heights += added
    heights -= heights.min()
    return World(heights.astype(np.float32), surface, obstacle, cfg.cell, seed, rocks)

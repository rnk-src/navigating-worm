"""The terrain world: a height grid and a surface class grid, with no MuJoCo dependency."""

import math

import numpy as np
import pytest

from worm import world as wd
from worm.terrain import TerrainClass


def ramp(deg: float = 10.0, size: float = 2.0, cell: float = 0.05) -> wd.World:
    """Return a world that rises along +x at `deg` degrees."""
    base = wd.flat((size, size), cell)
    xs = base.to_xy(np.stack(np.indices(base.shape), axis=-1))[..., 0]
    heights = (xs - xs.min()) * math.tan(math.radians(deg))
    return wd.World(heights.astype(np.float32), base.surface, base.obstacle, cell)


def test_flat_world_has_zero_height_and_one_class() -> None:
    world = wd.flat((2.0, 1.0), cell=0.05, cls=TerrainClass.SAND)
    assert world.shape == (21, 41)  # rows run along y, columns along x
    assert not world.heights.any() and not world.obstacle.any()
    assert np.all(world.surface == TerrainClass.SAND)


def test_to_cell_and_to_xy_round_trip_on_cell_centres() -> None:
    world = wd.flat((2.0, 1.0))
    cells = np.array([[0, 0], [20, 40], [7, 13]])
    assert np.array_equal(world.to_cell(world.to_xy(cells)), cells)
    assert world.to_xy((0, 0)) == pytest.approx((-1.0, -0.5))
    assert world.to_xy((20, 40)) == pytest.approx((1.0, 0.5))


def test_to_cell_clips_points_outside() -> None:
    world = wd.flat((2.0, 1.0))
    assert world.to_cell((9.0, -9.0)).tolist() == [0, 40]


def test_world_is_centred_on_the_origin() -> None:
    xmin, xmax, ymin, ymax = wd.flat((2.0, 1.0)).extent
    assert (xmin, ymin) == pytest.approx((-xmax, -ymax))
    assert (xmax, ymax) == pytest.approx((1.0, 0.5))


def test_height_at_is_bilinear_between_cells() -> None:
    base = wd.flat((0.1, 0.1), cell=0.1)  # 2 x 2 grid
    heights = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    world = wd.World(heights, base.surface, base.obstacle, 0.1)
    assert world.height_at((0.0, 0.0)) == pytest.approx(0.15)
    assert world.height_at((-0.05, 0.05)) == pytest.approx(0.2)
    assert world.height_at([(0.05, -0.05), (0.0, -0.05)]) == pytest.approx([0.1, 0.05])


def test_normal_on_a_tilted_plane_matches_the_analytic_normal() -> None:
    angle = math.radians(10.0)
    normal = ramp(10.0).normal_at([(0.0, 0.0), (0.33, -0.71)])
    assert normal == pytest.approx(
        np.tile([-math.sin(angle), 0.0, math.cos(angle)], (2, 1)), abs=1e-5
    )


def test_slope_deg_of_a_10_degree_ramp() -> None:
    assert ramp(10.0).slope_deg() == pytest.approx(10.0, abs=1e-3)


def test_contains_rejects_points_outside() -> None:
    world = wd.flat((2.0, 1.0))
    inside = world.contains([(0.0, 0.0), (0.99, 0.49), (1.01, 0.0), (0.0, -0.51)])
    assert inside.tolist() == [True, True, False, False]


def test_class_at_returns_the_nearest_cell_class() -> None:
    base = wd.flat((1.0, 1.0))
    surface = base.surface.copy()
    surface[:, 11:] = TerrainClass.GRASS  # the +x half
    world = wd.World(base.heights, surface, base.obstacle, base.cell)
    assert world.class_at([(-0.2, 0.0), (0.2, 0.3)]).tolist() == [0, 1]


def test_hfield_data_is_normalized_and_row_major() -> None:
    world = ramp(10.0, size=1.0)
    data, zmax = world.to_hfield_data()
    assert zmax == pytest.approx(float(world.heights.max()))
    assert data.shape == (world.shape[0] * world.shape[1],)
    assert data.max() == pytest.approx(1.0) and data.min() == pytest.approx(0.0)
    assert data[: world.shape[1]] == pytest.approx(world.heights[0] / zmax)

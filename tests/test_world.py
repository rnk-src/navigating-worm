"""The terrain world: a height grid and a surface class grid, with no MuJoCo dependency."""

import math

import numpy as np
import pytest

from worm import world as wd
from worm.terrain import TerrainClass


def ramp(deg: float = 10.0, size: float = 2.0, cell: float = 0.05) -> wd.World:
    """Return a world that rises along +x at `deg` degrees."""
    base = wd.flat((size, size), cell)
    xs = base.cell_xy()[..., 0]
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


# --- generator ---

SMALL = wd.WorldConfig(size=(5.0, 5.0), n_rocks=6, n_ditches=2)


def test_generate_is_deterministic_for_a_seed_and_differs_across_seeds() -> None:
    first, again, other = wd.generate(SMALL, 1), wd.generate(SMALL, 1), wd.generate(SMALL, 2)
    assert np.array_equal(first.heights, again.heights)
    assert np.array_equal(first.surface, again.surface)
    assert not np.array_equal(first.heights, other.heights)
    assert first.seed == 1 and first.shape == (101, 101)


def test_heights_are_nonnegative_and_slope_respects_max_slope() -> None:
    cfg = wd.WorldConfig(size=(5.0, 5.0), n_rocks=0, n_ditches=0, roughness=0.0, max_slope_deg=8.0)
    world = wd.generate(cfg, 0)
    assert world.heights.min() >= 0.0
    assert 4.0 < world.slope_deg().max() <= 8.0 + 1e-3


def test_roughness_adds_millimetre_relief_per_class() -> None:
    cfg = wd.WorldConfig(
        size=(3.0, 3.0), n_rocks=0, n_ditches=0, hill_height=0.0, mix=(1.0, 0.0, 0.0)
    )
    rough = wd.generate(cfg, 0)
    assert 0.002 <= rough.heights.max() - rough.heights.min() <= 0.02


def test_surface_mix_matches_the_requested_fractions_within_5_percent() -> None:
    cfg = wd.WorldConfig(size=(4.95, 4.95), mix=(0.6, 0.2, 0.2), n_rocks=0, n_ditches=0)
    world = wd.generate(cfg, 3)
    assert world.shape == (100, 100)
    fractions = np.bincount(world.surface.ravel(), minlength=3) / world.surface.size
    assert fractions == pytest.approx([0.6, 0.2, 0.2], abs=0.05)


def test_mix_of_one_class_gives_a_single_class_world() -> None:
    cfg = wd.WorldConfig(size=(2.0, 2.0), mix=(0.0, 1.0, 0.0), n_rocks=0, n_ditches=0)
    assert np.all(wd.generate(cfg, 0).surface == TerrainClass.GRASS)


def test_rocks_are_marked_as_obstacles_and_raised_by_at_least_min_height() -> None:
    cfg = wd.WorldConfig(size=(5.0, 5.0), n_rocks=5, n_ditches=0)
    bare = wd.generate(wd.WorldConfig(size=(5.0, 5.0), n_rocks=0, n_ditches=0), 4)
    world = wd.generate(cfg, 4)
    assert world.obstacle.any()
    raised = world.heights - bare.heights
    assert raised.max() >= cfg.rock_height[0] * 0.9
    assert np.all(raised[~world.obstacle] == pytest.approx(0.0, abs=1e-6))
    assert np.all(raised[world.obstacle] > 0.0)


def test_rocks_keep_a_gap_that_the_worm_can_pass() -> None:
    world = wd.generate(wd.WorldConfig(size=(5.0, 5.0), n_rocks=12, n_ditches=0), 5)
    centres = world.rocks[:, :2]
    radii = world.rocks[:, 2]
    assert len(centres) >= 2
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            gap = np.linalg.norm(centres[i] - centres[j]) - radii[i] - radii[j]
            assert gap >= 0.4


def test_ditches_are_depressions_of_at_least_min_depth() -> None:
    cfg = wd.WorldConfig(size=(5.0, 5.0), n_rocks=0, n_ditches=2)
    bare = wd.generate(wd.WorldConfig(size=(5.0, 5.0), n_rocks=0, n_ditches=0), 6)
    world = wd.generate(cfg, 6)
    # Every world is shifted so that its lowest point is zero. Compare the relief only.
    drop = bare.heights - world.heights
    drop -= drop.min()
    assert drop.max() >= cfg.ditch_depth[0] * 0.9
    assert np.count_nonzero(drop > 1e-6) < 0.3 * drop.size


def test_spawn_area_is_clear_and_flat() -> None:
    for seed in range(5):
        world = wd.generate(wd.WorldConfig(size=(4.0, 4.0), n_rocks=20, n_ditches=4), seed)
        near = np.linalg.norm(world.cell_xy(), axis=-1) <= 0.5
        assert not world.obstacle[near].any()
        assert world.slope_deg()[near].max() < 5.0


def test_smooth_noise_is_zero_mean_unit_std_and_correlated() -> None:
    noise = wd.smooth_noise((64, 64), 8.0, np.random.default_rng(0))
    assert abs(noise.mean()) < 1e-9 and noise.std() == pytest.approx(1.0)
    neighbour = np.corrcoef(noise[:, :-1].ravel(), noise[:, 1:].ravel())[0, 1]
    assert neighbour > 0.9

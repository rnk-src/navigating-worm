"""Top-down map images of a world, drawn with numpy only."""

import numpy as np

from worm import mapviz
from worm import world as wd
from worm.terrain import TerrainClass


def half_and_half() -> wd.World:
    """Return a 2 m x 1 m flat world: concrete at -x, sand at +x."""
    base = wd.flat((2.0, 1.0))
    surface = base.surface.copy()
    surface[:, 21:] = TerrainClass.SAND
    return wd.World(base.heights, surface, base.obstacle, base.cell)


def test_world_image_has_the_expected_size_and_class_colours() -> None:
    image = mapviz.world_image(half_and_half(), px_per_m=40.0, shade=False)
    assert image.shape == (40, 80, 3) and image.dtype == np.uint8
    assert tuple(image[20, 10]) == mapviz.CLASS_COLORS[TerrainClass.CONCRETE]
    assert tuple(image[20, 70]) == mapviz.CLASS_COLORS[TerrainClass.SAND]


def test_world_image_draws_rocks_dark() -> None:
    base = wd.flat((1.0, 1.0))
    obstacle = base.obstacle.copy()
    obstacle[8:13, 8:13] = True
    world = wd.World(base.heights, base.surface, obstacle, base.cell)
    image = mapviz.world_image(world, shade=False)
    assert image[20, 20].sum() < 0.5 * image[2, 2].sum()


def test_hillshade_makes_opposite_slopes_differ() -> None:
    base = wd.flat((2.0, 1.0))
    xs = base.cell_xy()[..., 0]
    tent = (0.3 * (1.0 - np.abs(xs))).astype(np.float32)
    image = mapviz.world_image(wd.World(tent, base.surface, base.obstacle, base.cell))
    assert abs(int(image[20, 20, 0]) - int(image[20, 60, 0])) > 10


def test_world_to_pixel_puts_plus_y_at_the_top() -> None:
    world = wd.flat((2.0, 1.0))
    top_left = mapviz.world_to_pixel(world, (-1.0, 0.5), 40.0)
    bottom_right = mapviz.world_to_pixel(world, (1.0, -0.5), 40.0)
    assert top_left.tolist() == [0, 0]
    assert bottom_right.tolist() == [79, 39]  # [x pixel, y pixel], clipped to the image


def test_draw_polyline_changes_pixels_along_the_line() -> None:
    world = wd.flat((2.0, 1.0))
    image = mapviz.world_image(world, shade=False)
    before = image.copy()
    mapviz.draw_polyline(image, world, [(-0.5, 0.0), (0.5, 0.0)], (255, 0, 0), 40.0)
    changed = np.any(image != before, axis=-1)
    assert changed[20, 21:59].all()  # the row of y = 0, between the two ends
    assert not changed[5].any() and not changed[:, 5].any()
    assert tuple(image[20, 40]) == (255, 0, 0)


def test_draw_points_draws_a_disc_at_each_point() -> None:
    world = wd.flat((2.0, 1.0))
    image = mapviz.world_image(world, shade=False)
    mapviz.draw_points(image, world, [(0.5, 0.25)], (0, 0, 255), 40.0, radius=3)
    blue = np.all(image == (0, 0, 255), axis=-1)
    assert 20 <= blue.sum() <= 40 and blue[10, 60]


def test_draw_blocked_tints_only_the_blocked_cells() -> None:
    from worm import planner

    base = wd.flat((2.0, 2.0))
    obstacle = base.obstacle.copy()
    obstacle[18:23, 28:33] = True  # around (0.5, 0.0)
    world = wd.World(base.heights, base.surface, obstacle, base.cell)
    costmap = planner.build_costmap(world, planner.CostConfig(edge_margin=0.0))
    image = mapviz.world_image(world, shade=False)
    before = image.copy()
    mapviz.draw_blocked(image, world, costmap, 40.0)
    changed = np.any(image != before, axis=-1)
    ring = mapviz.world_to_pixel(world, (0.5, 0.2), 40.0)  # inside the inflation ring
    away = mapviz.world_to_pixel(world, (-0.5, 0.0), 40.0)
    assert changed[ring[1], ring[0]] and not changed[away[1], away[0]]

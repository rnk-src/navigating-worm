"""Top-down map images of a world. Pure numpy: no matplotlib and no MuJoCo.

Image convention, shared with the click-to-go page: pixel x grows with world +x,
and pixel y grows downward, so world +y is at the top of the image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from worm.terrain import TerrainClass
from worm.world import World

if TYPE_CHECKING:
    from worm.planner import CostMap

Image = NDArray[np.uint8]
Color = tuple[int, int, int]

CLASS_COLORS: dict[TerrainClass, Color] = {
    TerrainClass.CONCRETE: (150, 150, 155),  # stone grey
    TerrainClass.GRASS: (96, 160, 80),  # grass green
    TerrainClass.SAND: (214, 190, 140),  # sand tan
}
ROCK_COLOR: Color = (60, 56, 54)
PATH_COLOR: Color = (30, 90, 255)
WAYPOINT_COLOR: Color = (230, 40, 40)
REACHED_COLOR: Color = (130, 130, 130)
GOAL_COLOR: Color = (255, 220, 0)
_LIGHT = np.array([-0.5, 0.5, 0.7]) / np.linalg.norm([-0.5, 0.5, 0.7])  # from the top left


def image_size(world: World, px_per_m: float) -> tuple[int, int]:
    """Return (height, width) of the map image in pixels."""
    xmin, xmax, ymin, ymax = world.extent
    return max(1, round((ymax - ymin) * px_per_m)), max(1, round((xmax - xmin) * px_per_m))


def world_to_pixel(world: World, xy: ArrayLike, px_per_m: float) -> NDArray[np.int_]:
    """Return the (..., [x pixel, y pixel]) of world points, clipped to the image."""
    point = np.asarray(xy, dtype=np.float64)
    xmin, _, _, ymax = world.extent
    height, width = image_size(world, px_per_m)
    pixel = np.stack([(point[..., 0] - xmin) * px_per_m, (ymax - point[..., 1]) * px_per_m], -1)
    return np.clip(np.floor(pixel).astype(np.int_), 0, [width - 1, height - 1])


def cell_colours(world: World) -> Image:
    """Return the (rows, cols, 3) class colour of every cell, with rocks drawn dark."""
    palette = np.array([CLASS_COLORS[c] for c in TerrainClass], dtype=np.uint8)
    colours: Image = palette[world.surface]
    colours[world.obstacle] = ROCK_COLOR
    return colours


def world_image(world: World, px_per_m: float = 40.0, shade: bool = True) -> Image:
    """Return an RGB map: class colour times hillshade, with rocks drawn dark."""
    colour = cell_colours(world).astype(np.float64)
    if shade:
        slope = world.gradient
        normal = np.concatenate([-slope, np.ones((*slope.shape[:2], 1))], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        light = normal @ _LIGHT
        # Level ground keeps its class colour. Lower ground is a little darker.
        depth = world.heights.max() - world.heights
        factor = (light / _LIGHT[2]) ** 2 * (1.0 - np.clip(depth, 0.0, 0.5))
        colour = colour * np.clip(factor, 0.3, 1.4)[..., None]
        colour[world.obstacle] = ROCK_COLOR
    cells = np.clip(colour, 0, 255).astype(np.uint8)

    height, width = image_size(world, px_per_m)
    xmin, _, _, ymax = world.extent
    xs = xmin + (np.arange(width) + 0.5) / px_per_m
    ys = ymax - (np.arange(height) + 0.5) / px_per_m
    centres = np.stack(np.broadcast_arrays(xs[None, :], ys[:, None]), axis=-1)
    index = world.to_cell(centres)
    image: Image = cells[index[..., 0], index[..., 1]]
    return image


def _disc(image: Image, centre: NDArray[np.int_], radius: float, rgb: Color) -> None:
    height, width = image.shape[:2]
    reach = int(np.ceil(radius))
    x0, x1 = max(centre[0] - reach, 0), min(centre[0] + reach + 1, width)
    y0, y1 = max(centre[1] - reach, 0), min(centre[1] + reach + 1, height)
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 <= radius**2
    image[y0:y1, x0:x1][mask] = rgb


def draw_points(
    image: Image, world: World, points: ArrayLike, rgb: Color, px_per_m: float, radius: int = 3
) -> None:
    """Draw a filled disc at each world point, in place."""
    for pixel in world_to_pixel(world, np.atleast_2d(points), px_per_m):
        _disc(image, pixel, radius, rgb)


def draw_polyline(
    image: Image, world: World, points: ArrayLike, rgb: Color, px_per_m: float, width: int = 2
) -> None:
    """Draw connected line segments through the world points, in place."""
    pixels = world_to_pixel(world, np.atleast_2d(points), px_per_m)
    for start, end in zip(pixels[:-1], pixels[1:], strict=True):
        steps = int(np.abs(end - start).max()) + 1
        for pixel in np.rint(np.linspace(start, end, steps)).astype(np.int_):
            _disc(image, pixel, 0.5 * width, rgb)


def draw_blocked(image: Image, world: World, costmap: CostMap, px_per_m: float) -> None:
    """Tint the pixels of blocked planning cells red, in place."""
    height, width = image.shape[:2]
    xmin, _, _, ymax = world.extent
    xs = xmin + (np.arange(width) + 0.5) / px_per_m
    ys = ymax - (np.arange(height) + 0.5) / px_per_m
    rows, cols = costmap.shape
    col = np.clip(
        np.rint((xs - costmap.origin[0]) / costmap.resolution).astype(np.int_), 0, cols - 1
    )
    row = np.clip(
        np.rint((ys - costmap.origin[1]) / costmap.resolution).astype(np.int_), 0, rows - 1
    )
    blocked = ~np.isfinite(costmap.cost[row[:, None], col[None, :]])
    tinted = 0.6 * image[blocked] + 0.4 * np.array([200, 40, 40])
    image[blocked] = tinted.astype(np.uint8)


def draw_plan(
    image: Image,
    world: World,
    path: ArrayLike,
    waypoints: ArrayLike,
    px_per_m: float,
    reached: int = 0,
) -> None:
    """Draw a planned route in place: path, waypoints (grey once reached) and the goal."""
    points = np.atleast_2d(waypoints)
    draw_polyline(image, world, path, PATH_COLOR, px_per_m)
    if reached:
        draw_points(image, world, points[:reached], REACHED_COLOR, px_per_m)
    if reached < len(points) - 1:
        draw_points(image, world, points[reached:-1], WAYPOINT_COLOR, px_per_m)
    draw_points(image, world, points[-1:], GOAL_COLOR, px_per_m, radius=5)

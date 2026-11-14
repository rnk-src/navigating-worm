"""Cost map, A* search, smoothing and waypoints. Pure Python, no simulation."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from worm import planner
from worm import world as wd
from worm.planner import CostConfig
from worm.terrain import TerrainClass

MakeWorld = Callable[..., wd.World]
# Most tests want the raw cost of a cell, without the safety ring or the edge band.
BARE = CostConfig(inflate_radius=0.0, edge_margin=0.0)


def level(side: float = 4.0) -> np.ndarray:
    n = round(side / 0.05) + 1
    return np.zeros((n, n))


def test_flat_uniform_world_costs_one_everywhere_inside(make_world: MakeWorld) -> None:
    costmap = planner.build_costmap(make_world(level()), BARE)
    assert costmap.cost.shape == (41, 41)  # 81 cells of 5 cm, pooled in pairs
    assert costmap.resolution == pytest.approx(0.1)
    assert np.all(costmap.cost == 1.0)


def test_costmap_cells_map_to_world_coordinates(make_world: MakeWorld) -> None:
    costmap = planner.build_costmap(make_world(level()), BARE)
    assert costmap.to_xy((0, 0)) == pytest.approx((-1.975, -1.975))
    assert costmap.to_cell((0.03, -1.0)) == (10, 20)  # (row along y, column along x)
    assert costmap.to_xy(costmap.to_cell((1.0, 0.5))) == pytest.approx((1.0, 0.5), abs=0.05)
    assert costmap.free((10, 20)) and not costmap.free((-1, 0)) and not costmap.free((0, 41))


def test_surface_cost_follows_the_class(make_world: MakeWorld) -> None:
    surface = np.zeros((81, 81))
    surface[:, 27:54] = TerrainClass.GRASS
    surface[:, 54:] = TerrainClass.SAND
    costmap = planner.build_costmap(make_world(level(), surface), BARE)
    cfg = CostConfig()
    row = costmap.cost[20]
    assert row[5] == cfg.surface_cost[0] and row[20] == cfg.surface_cost[1]
    assert row[35] == cfg.surface_cost[2]
    assert cfg.surface_cost[0] < cfg.surface_cost[1] < cfg.surface_cost[2]


def test_gentle_slope_costs_more_than_level_ground(make_world: MakeWorld) -> None:
    heights = np.tile(np.arange(81) * 0.05 * math.tan(math.radians(8.0)), (81, 1))
    costmap = planner.build_costmap(make_world(heights), BARE)
    assert np.all(np.isfinite(costmap.cost))
    assert costmap.cost[20, 20] == pytest.approx(1.0 + (8.0 / 20.0) ** 2, rel=0.02)


def test_steep_slope_is_blocked(make_world: MakeWorld) -> None:
    heights = np.tile(np.arange(81) * 0.05 * math.tan(math.radians(25.0)), (81, 1))
    costmap = planner.build_costmap(make_world(heights), BARE)
    assert np.all(np.isinf(costmap.cost))


def test_rock_and_its_inflation_ring_are_blocked(make_world: MakeWorld) -> None:
    obstacle = np.zeros((81, 81))
    obstacle[38:43, 38:43] = 1  # 0.2 m square at the centre
    world = make_world(level(), obstacle=obstacle)
    bare = planner.build_costmap(world, BARE)
    ringed = planner.build_costmap(world, CostConfig(edge_margin=0.0))
    centre = bare.to_cell((0.0, 0.0))
    assert not bare.free(centre) and not ringed.free(centre)
    near = bare.to_cell((0.25, 0.0))  # 15 cm from the rock edge
    assert bare.free(near) and not ringed.free(near)
    assert ringed.free(ringed.to_cell((0.45, 0.0)))


def test_shallow_ditch_is_passable_and_deep_ditch_is_blocked(make_world: MakeWorld) -> None:
    def ditch(depth: float) -> planner.CostMap:
        heights = np.full((81, 81), depth)
        # A trench along y at x = 0, with 20 cm wide cosine walls on each side.
        x = np.abs(np.arange(81) - 40) * 0.05
        profile = np.where(
            x < 0.1, 0.0, np.where(x < 0.3, 0.5 * (1 - np.cos(np.pi * (x - 0.1) / 0.2)), 1.0)
        )
        return planner.build_costmap(make_world(heights * profile[None, :]), BARE)

    assert np.all(np.isfinite(ditch(0.02).cost))
    deep = ditch(0.12).cost
    assert np.isinf(deep[20, 18]) and np.isinf(deep[20, 22])  # the walls
    assert np.isfinite(deep[20, 5])  # level ground away from the ditch


def test_downsampling_keeps_a_one_cell_obstacle(make_world: MakeWorld) -> None:
    obstacle = np.zeros((81, 81))
    obstacle[41, 33] = 1
    costmap = planner.build_costmap(make_world(level(), obstacle=obstacle), BARE)
    assert np.isinf(costmap.cost[20, 16])
    assert np.isinf(costmap.cost).sum() == 1


def test_a_band_along_the_world_edge_is_blocked(make_world: MakeWorld) -> None:
    costmap = planner.build_costmap(make_world(level()), CostConfig(inflate_radius=0.0))
    assert not costmap.free(costmap.to_cell((1.7, 0.0)))  # 0.3 m from the edge
    assert costmap.free(costmap.to_cell((1.5, 0.0)))

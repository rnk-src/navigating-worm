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


# --- search, smoothing, waypoints ---

INF = np.inf


def grid(cost: np.ndarray) -> planner.CostMap:
    """Return a cost map with 0.1 m cells whose cell (0, 0) is at the origin."""
    return planner.CostMap(cost.astype(np.float64), 0.1, (0.0, 0.0))


def path_cost(costmap: planner.CostMap, cells: list[tuple[int, int]]) -> float:
    total = 0.0
    for a, b in zip(cells[:-1], cells[1:], strict=True):
        step = math.hypot(a[0] - b[0], a[1] - b[1]) * costmap.resolution
        total += 0.5 * (costmap.cost[a] + costmap.cost[b]) * step
    return total


def crosses_blocked(costmap: planner.CostMap, points: np.ndarray) -> bool:
    """Sample the polyline every centimetre and report any blocked cell."""
    for a, b in zip(points[:-1], points[1:], strict=True):
        for t in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(b - a) / 0.01))):
            if not costmap.free(costmap.to_cell(a + t * (b - a))):
                return True
    return False


def test_astar_straight_line_on_empty_map_is_optimal() -> None:
    costmap = grid(np.ones((20, 20)))
    cells = planner.astar(costmap, (2, 2), (2, 17))
    assert cells is not None and cells[0] == (2, 2) and cells[-1] == (2, 17)
    assert path_cost(costmap, cells) == pytest.approx(1.5)
    diagonal = planner.astar(costmap, (0, 0), (10, 15))
    assert diagonal is not None
    assert path_cost(costmap, diagonal) == pytest.approx(0.1 * (10 * math.sqrt(2) + 5))


def test_astar_goes_around_a_wall_through_the_gap() -> None:
    cost = np.ones((20, 20))
    cost[:, 10] = INF
    cost[15, 10] = 1.0  # the only gap
    cells = planner.astar(grid(cost), (2, 2), (2, 17))
    assert cells is not None and (15, 10) in cells


def test_astar_returns_none_when_walled_off() -> None:
    cost = np.ones((20, 20))
    cost[:, 10] = INF
    assert planner.astar(grid(cost), (2, 2), (2, 17)) is None


def test_astar_prefers_a_longer_concrete_detour_over_a_short_sand_crossing_when_cheaper() -> None:
    cost = np.ones((20, 30))
    cost[0:15, 10:20] = 5.0  # a sand field; rows 15..19 stay cheap
    costmap = grid(cost)
    cells = planner.astar(costmap, (5, 5), (5, 25))
    assert cells is not None
    assert all(costmap.cost[c] == 1.0 for c in cells)  # it walks around
    cost[0:15, 10:20] = 1.1  # now the sand is barely slower
    direct = planner.astar(grid(cost), (5, 5), (5, 25))
    assert direct is not None and all(c[0] == 5 for c in direct)  # it goes straight through


def test_astar_does_not_cut_blocked_corners() -> None:
    cost = np.ones((6, 6))
    cost[2, 3] = INF
    cost[3, 2] = INF
    cells = planner.astar(grid(cost), (2, 2), (3, 3))
    assert cells is not None and len(cells) > 2  # the diagonal step squeezes between two blocks


def test_smooth_reduces_vertices_and_never_enters_blocked_cells() -> None:
    cost = np.ones((30, 30))
    cost[5:25, 14:16] = INF  # a wall with free rows above and below
    costmap = grid(cost)
    cells = planner.astar(costmap, (15, 3), (15, 26))
    assert cells is not None
    path = planner.smooth(costmap, cells)
    assert 3 <= len(path) <= 6 < len(cells)
    assert path[0] == pytest.approx(costmap.to_xy((15, 3)))
    assert path[-1] == pytest.approx(costmap.to_xy((15, 26)))
    assert not crosses_blocked(costmap, path)


def test_smooth_keeps_a_cheap_detour_instead_of_a_costly_shortcut() -> None:
    cost = np.ones((20, 30))
    cost[0:15, 10:20] = 5.0
    costmap = grid(cost)
    cells = planner.astar(costmap, (5, 5), (5, 25))
    assert cells is not None
    path = planner.smooth(costmap, cells)
    rows = (path[:, 1] / 0.1).round()
    assert rows.max() >= 15  # the smoothed path still goes around the sand


def test_waypoints_are_at_most_2m_apart_and_end_at_the_goal() -> None:
    path = np.array([[0.0, 0.0], [7.0, 0.0]])
    waypoints = planner.to_waypoints(path)
    assert waypoints[-1] == pytest.approx((7.0, 0.0))
    legs = np.linalg.norm(np.diff(np.vstack([path[:1], waypoints]), axis=0), axis=1)
    assert len(waypoints) == 4 and legs.max() <= 2.0 + 1e-9
    assert legs == pytest.approx(1.75)  # even legs, not 2 + 2 + 2 + 1


def test_waypoints_keep_corners() -> None:
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.5], [3.0, 1.5]])
    waypoints = planner.to_waypoints(path)
    assert waypoints == pytest.approx(np.array([[1.0, 0.0], [1.0, 1.5], [3.0, 1.5]]))


def test_waypoints_drop_slight_bends_and_merge_close_points() -> None:
    slight = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.1], [3.0, 0.1]])  # bends of 6 degrees
    assert len(planner.to_waypoints(slight)) == 2  # 3 m split into two legs, bends dropped
    close = np.array([[0.0, 0.0], [1.0, 0.0], [1.1, 0.1], [1.1, 1.0]])  # two corners 14 cm apart
    merged = planner.to_waypoints(close)
    assert len(merged) == 2 and merged[-1] == pytest.approx((1.1, 1.0))


def walled_costmap() -> planner.CostMap:
    """Return a 4 m x 4 m map with a rock at (2, 2) and a sealed room at the top right."""
    cost = np.ones((41, 41))
    cost[16:25, 16:25] = INF  # the rock: 1.6 m to 2.4 m on both axes
    cost[30, 30:] = INF
    cost[30:, 30] = INF  # the room walls
    return grid(cost)


@pytest.mark.parametrize(
    ("start", "goal", "reason"),
    [
        ((0.5, 0.5), (9.0, 0.5), planner.Reason.OUT_OF_BOUNDS),
        ((-1.0, 0.5), (1.0, 0.5), planner.Reason.OUT_OF_BOUNDS),
        ((2.0, 2.0), (0.5, 0.5), planner.Reason.START_BLOCKED),
        ((0.5, 0.5), (2.0, 2.0), planner.Reason.GOAL_BLOCKED),
        ((0.5, 0.5), (3.5, 3.5), planner.Reason.NO_PATH),
    ],
)
def test_plan_reports_each_infeasible_reason(
    start: tuple[float, float], goal: tuple[float, float], reason: planner.Reason
) -> None:
    result = planner.plan(walled_costmap(), start, goal)
    assert isinstance(result, planner.Infeasible)
    assert result.reason == reason and result.detail


def test_plan_returns_a_route_with_exact_end_points() -> None:
    result = planner.plan(walled_costmap(), (0.52, 0.47), (3.03, 1.51))
    assert isinstance(result, planner.Plan)
    assert result.path[0] == pytest.approx((0.52, 0.47))
    assert result.path[-1] == pytest.approx((3.03, 1.51))
    assert result.waypoints[-1] == pytest.approx((3.03, 1.51))
    assert not result.goal_snapped and result.cost > 2.5 and result.expanded > 0


def test_plan_snaps_a_start_next_to_a_rock() -> None:
    # (1.56, 2.0) rounds to column 16, the rim of the rock. Column 15 is free.
    result = planner.plan(walled_costmap(), (1.56, 2.0), (0.5, 0.5))
    assert isinstance(result, planner.Plan)
    assert result.path[0] == pytest.approx((1.56, 2.0))  # the route still starts at the worm
    assert result.path[1] == pytest.approx((1.5, 2.0))  # then it steps to the nearest free cell


def test_plan_snaps_a_goal_just_inside_a_blocked_cell_and_reports_it() -> None:
    result = planner.plan(walled_costmap(), (0.5, 0.5), (1.56, 2.0))
    assert isinstance(result, planner.Plan) and result.goal_snapped
    assert result.path[-1] == pytest.approx((1.5, 2.0))  # the nearest free cell centre


def test_plan_on_generated_worlds_never_crosses_blocked_cells() -> None:
    rng = np.random.default_rng(0)
    plans = 0
    for seed in range(20):
        world = wd.generate(wd.WorldConfig(size=(6.0, 6.0), n_rocks=8, n_ditches=1), seed)
        costmap = planner.build_costmap(world)
        free = np.argwhere(np.isfinite(costmap.cost))
        start, goal = (costmap.to_xy(free[i]) for i in rng.choice(len(free), 2, replace=False))
        result = planner.plan(costmap, start, goal)
        if isinstance(result, planner.Infeasible):
            assert result.reason == planner.Reason.NO_PATH
            continue
        plans += 1
        assert not crosses_blocked(costmap, result.path)
        route = np.vstack([result.path[:1], result.waypoints])
        assert not crosses_blocked(costmap, route)
        assert np.linalg.norm(np.diff(route, axis=0), axis=1).max() <= 2.0 + 1e-9
    assert plans >= 12


def test_plan_10m_world_runs_under_1s() -> None:
    import time

    world = wd.generate(wd.WorldConfig(), 0)
    costmap = planner.build_costmap(world)
    free = np.argwhere(np.isfinite(costmap.cost))
    corners = [free[np.argmin(free @ d)] for d in ([1, 1], [-1, -1])]
    begin = time.perf_counter()
    result = planner.plan(costmap, costmap.to_xy(corners[0]), costmap.to_xy(corners[1]))
    assert time.perf_counter() - begin < 1.0
    assert isinstance(result, planner.Plan | planner.Infeasible)


def test_cost_defaults_are_ordered_concrete_grass_sand() -> None:
    # Measured with scripts/calibrate_costs.py on the committed policy: 1.00, 1.22, 2.03.
    assert CostConfig().surface_cost == (1.0, 1.2, 2.0)

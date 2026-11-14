"""Global planner: turn a world and a goal into a waypoint route, or say why not.

Pure Python and numpy. No simulation, so it runs headless and fast.
Cells are (row, column): the row runs along +y and the column along +x.
"""

from __future__ import annotations

import enum
import heapq
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
    surface_cost: tuple[float, float, float] = (1.0, 1.2, 2.0)  # per TerrainClass; measured
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


class Reason(enum.Enum):
    OUT_OF_BOUNDS = "out of bounds"
    START_BLOCKED = "start blocked"
    GOAL_BLOCKED = "goal blocked"
    NO_PATH = "no path"


@dataclass(frozen=True)
class Infeasible:
    reason: Reason
    detail: str


@dataclass(frozen=True, eq=False)
class Plan:
    path: NDArray[np.float64]  # (n, 2) smoothed polyline, start .. goal
    waypoints: NDArray[np.float64]  # (m, 2) route for the follower; excludes the start
    cost: float
    expanded: int  # cells that the search expanded
    goal_snapped: bool  # the goal moved to the nearest free cell


_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_START_SNAP = 0.3  # m, the worm may sit beside a rock, inside its inflation ring
_GOAL_SNAP = 0.2  # m


def _search(costmap: CostMap, start: Cell, goal: Cell) -> tuple[list[Cell] | None, float, int]:
    """A* with 8 neighbours. Return (cells or None, path cost, cells expanded).

    An edge costs the mean of its two cell costs times its length. A diagonal step
    needs both cells beside it free, so the path never squeezes past a blocked corner.
    """
    if not (costmap.free(start) and costmap.free(goal)):
        return None, math.inf, 0
    cost = costmap.cost
    rows, cols = costmap.shape
    res = costmap.resolution
    cheapest = float(cost[np.isfinite(cost)].min()) * res
    diagonal = math.sqrt(2.0) - 1.0

    def heuristic(cell: Cell) -> float:
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return cheapest * (max(dr, dc) + diagonal * min(dr, dc))

    best = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    done: set[Cell] = set()
    queue = [(heuristic(start), 0.0, start)]
    while queue:
        _, so_far, cell = heapq.heappop(queue)
        if cell in done:
            continue
        done.add(cell)
        if cell == goal:
            cells = [cell]
            while cells[-1] != start:
                cells.append(parent[cells[-1]])
            return cells[::-1], so_far, len(done)
        r, c = cell
        here = cost[r, c]
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            there = cost[nr, nc]
            if there == math.inf:
                continue
            if dr and dc and (cost[r, nc] == math.inf or cost[nr, c] == math.inf):
                continue
            step = res * (math.sqrt(2.0) if dr and dc else 1.0)
            total = so_far + 0.5 * (here + there) * step
            neighbour = (nr, nc)
            if total < best.get(neighbour, math.inf):
                best[neighbour] = total
                parent[neighbour] = cell
                heapq.heappush(queue, (total + heuristic(neighbour), total, neighbour))
    return None, math.inf, len(done)


def astar(costmap: CostMap, start: Cell, goal: Cell) -> list[Cell] | None:
    """Return the cheapest chain of cells from `start` to `goal`, or None."""
    return _search(costmap, start, goal)[0]


def _line(costmap: CostMap, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return the travel cost of the straight line a -> b, or infinity if it is blocked.

    The line is sampled every quarter cell. A diagonal move between two samples also
    needs both cells beside it free.
    """
    length = float(np.linalg.norm(b - a))
    count = max(2, math.ceil(length / (0.25 * costmap.resolution)) + 1)
    points = a + np.linspace(0.0, 1.0, count)[:, None] * (b - a)
    origin = np.array(costmap.origin)
    index = np.rint((points - origin) / costmap.resolution).astype(np.int_)
    rows, cols = costmap.shape
    r, c = index[:, 1], index[:, 0]
    if r.min() < 0 or c.min() < 0 or r.max() >= rows or c.max() >= cols:
        return math.inf
    cost = costmap.cost
    along = cost[r, c]
    if not np.all(np.isfinite(along)):
        return math.inf
    corner = (r[1:] != r[:-1]) & (c[1:] != c[:-1])
    if corner.any():
        beside = np.concatenate(
            [cost[r[1:][corner], c[:-1][corner]], cost[r[:-1][corner], c[1:][corner]]]
        )
        if not np.all(np.isfinite(beside)):
            return math.inf
    return float(0.5 * (along[1:] + along[:-1]).sum() * length / (count - 1))


def smooth(costmap: CostMap, cells: list[Cell], eps: float = 0.05) -> NDArray[np.float64]:
    """Pull the cell chain tight. Return the (n, 2) polyline through the kept cell centres.

    From each kept vertex, jump to the farthest later cell whose straight line is free
    and costs at most (1 + eps) times the stretch of the chain that it replaces.
    """
    points = costmap.to_xy(np.array(cells))
    edge = [
        0.5 * (costmap.cost[a] + costmap.cost[b]) * float(np.linalg.norm(q - p))
        for a, b, p, q in zip(cells[:-1], cells[1:], points[:-1], points[1:], strict=True)
    ]
    prefix = np.concatenate([[0.0], np.cumsum(edge)])
    kept = [0]
    last = len(cells) - 1
    while kept[-1] < last:
        i = kept[-1]
        target = i + 1
        for j in range(last, i + 1, -1):
            if _line(costmap, points[i], points[j]) <= (1.0 + eps) * (prefix[j] - prefix[i]):
                target = j
                break
        kept.append(target)
    smoothed: NDArray[np.float64] = points[kept]
    return smoothed


def to_waypoints(
    path: NDArray[np.float64], spacing: float = 2.0, corner_deg: float = 20.0, merge: float = 0.3
) -> NDArray[np.float64]:
    """Turn a polyline into follower waypoints. The start is left out; the goal is last.

    Vertices that turn more than `corner_deg` stay. A corner closer than `merge` to
    the one before it is dropped. Stretches longer than `spacing` are split evenly.
    """
    anchors = [path[0]]
    for k in range(1, len(path) - 1):
        before, after = path[k] - path[k - 1], path[k + 1] - path[k]
        turn = abs(math.atan2(before[0] * after[1] - before[1] * after[0], float(before @ after)))
        if math.degrees(turn) > corner_deg and np.linalg.norm(path[k] - anchors[-1]) >= merge:
            anchors.append(path[k])
    if len(anchors) > 1 and np.linalg.norm(path[-1] - anchors[-1]) < merge:
        anchors.pop()
    anchors.append(path[-1])
    waypoints = []
    for a, b in zip(anchors[:-1], anchors[1:], strict=True):
        pieces = max(1, math.ceil(float(np.linalg.norm(b - a)) / spacing - 1e-9))
        waypoints += [a + (b - a) * (k + 1) / pieces for k in range(pieces)]
    return np.array(waypoints).reshape(-1, 2)


def _snap(costmap: CostMap, xy: NDArray[np.float64], radius: float) -> Cell | None:
    """Return the cell of `xy` if it is free, else the nearest free cell within `radius`."""
    cell = costmap.to_cell(xy)
    if costmap.free(cell):
        return cell
    reach = math.ceil(radius / costmap.resolution) + 1
    best: tuple[float, Cell] | None = None
    for r in range(cell[0] - reach, cell[0] + reach + 1):
        for c in range(cell[1] - reach, cell[1] + reach + 1):
            if costmap.free((r, c)):
                distance = float(np.linalg.norm(costmap.to_xy((r, c)) - xy))
                if distance <= radius and (best is None or distance < best[0]):
                    best = (distance, (r, c))
    return None if best is None else best[1]


def _route_is_free(costmap: CostMap, points: NDArray[np.float64]) -> bool:
    return all(
        math.isfinite(_line(costmap, a, b)) for a, b in zip(points[:-1], points[1:], strict=True)
    )


def plan(
    costmap: CostMap, start: ArrayLike, goal: ArrayLike, spacing: float = 2.0
) -> Plan | Infeasible:
    """Plan a waypoint route from `start` to `goal`, or return the reason there is none."""
    start_xy = np.asarray(start, dtype=np.float64)
    goal_xy = np.asarray(goal, dtype=np.float64)
    for name, xy in (("start", start_xy), ("goal", goal_xy)):
        if not costmap.inside(costmap.to_cell(xy)):
            return Infeasible(
                Reason.OUT_OF_BOUNDS, f"the {name} ({xy[0]:.2f}, {xy[1]:.2f}) is outside the map"
            )
    start_cell = _snap(costmap, start_xy, _START_SNAP)
    if start_cell is None:
        return Infeasible(
            Reason.START_BLOCKED, f"no free ground within {_START_SNAP} m of the start"
        )
    goal_cell = _snap(costmap, goal_xy, _GOAL_SNAP)
    if goal_cell is None:
        return Infeasible(
            Reason.GOAL_BLOCKED,
            f"the goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) is on an obstacle, a steep slope "
            "or the map edge",
        )
    cells, cost, expanded = _search(costmap, start_cell, goal_cell)
    if cells is None:
        return Infeasible(
            Reason.NO_PATH,
            f"obstacles separate the start from the goal ({expanded} cells searched)",
        )

    goal_snapped = goal_cell != costmap.to_cell(goal_xy)
    vertices = [start_xy, *smooth(costmap, cells)]
    if not goal_snapped:
        vertices.append(goal_xy)
    path = np.array([v for k, v in enumerate(vertices) if k == 0 or np.any(v != vertices[k - 1])])
    if len(path) == 1:  # the start is the goal
        path = np.vstack([path, path])
    waypoints = to_waypoints(path, spacing)
    if not _route_is_free(costmap, np.vstack([path[:1], waypoints])):
        # Dropping a slight bend or merging two corners cut through a blocked cell.
        waypoints = to_waypoints(path, spacing, corner_deg=0.0, merge=0.0)
    return Plan(path, waypoints, cost, expanded, goal_snapped)

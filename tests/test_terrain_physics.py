"""Slope behaviour of the worm on a heightfield world. Kept as a regression spike.

A smooth `tanh` drag cannot hold a body still on a slope: it always creeps. The hybrid
mode of `TerrainResistive` adds real contact friction, which can.
"""

import math

import mujoco
import numpy as np
import pytest

from worm import friction, gait, model, terrain
from worm import world as wd
from worm.spec import WormSpec


def ramp(deg: float, size: float = 2.0) -> wd.World:
    """Return a world that rises along +x at `deg` degrees."""
    base = wd.flat((size, size))
    heights = (base.cell_xy()[..., 0] + 0.5 * size) * math.tan(math.radians(deg))
    return wd.World(heights.astype(np.float32), base.surface, base.obstacle, base.cell)


def displacement(slope_deg: float, yaw_deg: float, amp_deg: float, hybrid: bool = True) -> float:
    """Drop the worm on a ramp, let it settle, then run the gait for 10 s.

    Return the head displacement along +x (uphill) in metres. Zero amplitude means idle.
    """
    spec, world = WormSpec(), ramp(slope_deg)
    m = model.build(spec, world)
    d = mujoco.MjData(m)
    adapter = friction.TerrainResistive(
        model.segment_body_ids(m, spec), model.pair_ids(m, spec), world, hybrid=hybrid
    )
    adapter.configure(m, terrain.NOMINAL)
    yaw = math.radians(yaw_deg)
    d.qpos[:3] = (0.0, 0.0, float(world.height_at((0.0, 0.0))) + 0.05)
    d.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    for _ in range(250):
        friction.step(m, d, adapter)
    start = float(d.xpos[1, 0])
    phase = 0.0
    for _ in range(200):
        phase += 2.0 * math.pi * 0.05
        d.ctrl[:] = gait.serpenoid(phase, math.radians(amp_deg), 1.26, 0.0, 4)
        for _ in range(25):
            friction.step(m, d, adapter)
    return float(d.xpos[1, 0]) - start


@pytest.mark.parametrize("yaw_deg", [0, 90, 180])
def test_idle_worm_on_a_10_degree_slope_creeps_less_than_5cm_in_10s(yaw_deg: int) -> None:
    assert abs(displacement(10.0, yaw_deg, amp_deg=0.0)) < 0.05


def test_pure_drag_creeps_down_a_10_degree_slope() -> None:
    # Characterization: this is why worlds use the hybrid mode.
    assert displacement(10.0, 0, amp_deg=0.0, hybrid=False) < -0.20


def test_gait_climbs_a_5_degree_slope() -> None:
    assert displacement(5.0, 0, amp_deg=45.0) > 0.05


def test_gait_on_a_level_world_is_at_least_as_fast_as_on_the_plane() -> None:
    assert displacement(0.0, 0, amp_deg=45.0) > 0.20

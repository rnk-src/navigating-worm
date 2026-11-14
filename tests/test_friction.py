"""Friction behaviour, measured by coasting one segment and comparing distances."""

import math
from typing import Any

import mujoco
import numpy as np
import pytest

from worm import friction, model, terrain
from worm import world as wd
from worm.spec import WormSpec
from worm.terrain import FrictionParams

PARAMS = FrictionParams(mu_forward=0.3, mu_lateral=1.2)


def coast(adapter_name: str, yaw_deg: float, lateral: bool) -> float:
    """Launch one segment at 0.3 m/s along its own forward or lateral axis. Return metres."""
    spec = WormSpec(n_segments=1)
    m = model.build(spec)
    d = mujoco.MjData(m)
    adapter: friction.FrictionModel = (
        friction.ResistiveForce(model.segment_body_ids(m, spec))
        if adapter_name == "resistive"
        else friction.ContactAnisotropic()
    )
    adapter.configure(m, PARAMS)
    yaw = math.radians(yaw_deg)
    d.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    angle = yaw + (math.pi / 2 if lateral else 0.0)
    d.qvel[:2] = 0.3 * np.array([math.cos(angle), math.sin(angle)])
    for _ in range(1500):
        friction.step(m, d, adapter)
    return float(np.linalg.norm(d.qpos[:2]))


@pytest.mark.parametrize("yaw_deg", [0, 45, 90, 200])
def test_resistive_force_resists_sideways_3x_more_at_every_yaw(yaw_deg: int) -> None:
    forward = coast("resistive", yaw_deg, lateral=False)
    sideways = coast("resistive", yaw_deg, lateral=True)
    assert forward / sideways >= 3.0
    # Coulomb stopping distance v^2 / (2 mu g) = 0.09 / (2 * 0.3 * 9.81) = 15.3 mm.
    # The smooth tanh region near zero speed adds a little.
    assert forward == pytest.approx(0.0153, rel=0.25)


def test_contact_adapter_anisotropy_depends_on_yaw() -> None:
    # Characterization: MuJoCo fixes the contact tangent axes to the world, not the body.
    # This is why ContactAnisotropic is not the default adapter.
    ratios = [
        coast("contact", yaw, lateral=False) / coast("contact", yaw, lateral=True)
        for yaw in (0, 90)
    ]
    assert min(ratios) < 1.0 < max(ratios)


def test_heavier_segment_coasts_the_same_distance() -> None:
    # Coulomb-like drag scales with the load, so the mass cancels out.
    spec = WormSpec(n_segments=1, seg_mass=0.12)
    m = model.build(spec)
    d = mujoco.MjData(m)
    adapter = friction.ResistiveForce(model.segment_body_ids(m, spec))
    adapter.configure(m, PARAMS)
    d.qvel[0] = 0.3
    for _ in range(1500):
        friction.step(m, d, adapter)
    assert d.qpos[0] == pytest.approx(coast("resistive", 0, lateral=False), rel=0.02)


# --- TerrainResistive: per-cell surfaces on a heightfield world ---


def on_world(
    world: wd.World, n_segments: int = 1, **kwargs: Any
) -> tuple[mujoco.MjModel, mujoco.MjData, friction.TerrainResistive]:
    spec = WormSpec(n_segments=n_segments)
    m = model.build(spec, world)
    adapter = friction.TerrainResistive(
        model.segment_body_ids(m, spec), model.pair_ids(m, spec), world, **kwargs
    )
    return m, mujoco.MjData(m), adapter


def uniform_table(props: terrain.SurfaceProps) -> terrain.SurfaceTable:
    return terrain.SurfaceTable((props, props, props))


def nominal_table() -> terrain.SurfaceTable:
    concrete, grass, sand = (terrain.nominal_surface(c) for c in terrain.TerrainClass)
    return terrain.SurfaceTable((concrete, grass, sand))


def split_world(left: terrain.TerrainClass, right: terrain.TerrainClass) -> wd.World:
    """Return a flat 2 m world with one class at -x and another at +x."""
    base = wd.flat((2.0, 2.0))
    surface = base.surface.copy()
    surface[:, :20] = left
    surface[:, 20:] = right
    return wd.World(base.heights, surface, base.obstacle, base.cell)


@pytest.mark.parametrize("n_segments", [1, 5])
def test_terrain_resistive_on_flat_uniform_world_equals_resistive_force(n_segments: int) -> None:
    m, d, adapter = on_world(wd.flat((2.0, 2.0)), n_segments)
    reference = friction.ResistiveForce(model.segment_body_ids(m, WormSpec(n_segments=n_segments)))
    reference.configure(m, PARAMS)
    adapter.configure(m, PARAMS)
    d.qvel[:2] = (0.2, 0.1)
    d.qvel[5] = 1.0
    d.ctrl[:] = 0.4
    largest = 0.0
    for _ in range(200):
        mujoco.mj_step1(m, d)
        reference.apply(m, d)
        expected = d.xfrc_applied.copy()
        d.xfrc_applied[:] = 0.0
        adapter.apply(m, d)
        assert np.abs(d.xfrc_applied - expected).max() <= 1e-12
        largest = max(largest, float(np.abs(expected).max()))
        mujoco.mj_step2(m, d)
    assert largest > 0.05  # the comparison saw real forces


def coast_on(world: wd.World, start_x: float) -> tuple[float, int]:
    """Launch one segment forward at 0.3 m/s. Return (metres travelled, class under it)."""
    m, d, adapter = on_world(world)
    adapter.configure_table(m, nominal_table())
    d.qpos[0] = start_x
    d.qvel[0] = 0.3
    for _ in range(1500):
        friction.step(m, d, adapter)
    return float(d.qpos[0] - start_x), int(adapter.classes[0])


def test_terrain_resistive_uses_the_class_under_each_segment() -> None:
    world = split_world(terrain.TerrainClass.CONCRETE, terrain.TerrainClass.SAND)
    on_concrete, concrete = coast_on(world, -0.6)
    on_sand, sand = coast_on(world, 0.4)
    assert (concrete, sand) == (terrain.TerrainClass.CONCRETE, terrain.TerrainClass.SAND)
    assert on_sand < 0.6 * on_concrete


def settled_height(cls: terrain.TerrainClass) -> float:
    m, d, adapter = on_world(wd.flat((1.0, 1.0), cls=cls))
    adapter.configure_table(m, nominal_table())
    for _ in range(1500):
        friction.step(m, d, adapter)
    return float(d.qpos[2])


def test_sand_segment_sinks_deeper_than_concrete() -> None:
    concrete = settled_height(terrain.TerrainClass.CONCRETE)
    sand = settled_height(terrain.TerrainClass.SAND)
    # The plan asked for 1 mm. A contact soft enough for that jitters on the hfield, and
    # a softer one falls through it. See docs/decisions/0002. Sand is mostly viscous drag.
    assert concrete - sand >= 0.0001
    assert concrete == pytest.approx(0.0125, abs=0.0005)


def test_viscous_term_scales_with_speed() -> None:
    props = terrain.SurfaceProps(FrictionParams(0.0, 0.0), 0.2, 0.005, 0.0)
    forces = []
    for speed in (0.1, 0.2):
        m, d, adapter = on_world(wd.flat((1.0, 1.0)))
        adapter.configure_table(m, uniform_table(props))
        d.qvel[1] = speed
        mujoco.mj_step1(m, d)
        adapter.apply(m, d)
        forces.append(d.xfrc_applied[1, :3].copy())
    assert forces[0] == pytest.approx([0.0, -0.02, 0.0], abs=1e-9)  # -c v
    assert forces[1] == pytest.approx(2.0 * forces[0], abs=1e-9)


def test_drag_directions_lie_in_the_terrain_tangent_plane_on_a_slope() -> None:
    angle = math.radians(10.0)
    base = wd.flat((2.0, 2.0))
    heights = ((base.cell_xy()[..., 0] + 1.0) * math.tan(angle)).astype(np.float32)
    world = wd.World(heights, base.surface, base.obstacle, base.cell)
    m, d, adapter = on_world(world)
    adapter.configure_table(m, nominal_table())
    d.qpos[2] = float(world.height_at((0.0, 0.0))) + 0.02
    d.qvel[:3] = (0.2, 0.1, 0.0)
    mujoco.mj_step1(m, d)
    adapter.apply(m, d)
    force = d.xfrc_applied[1, :3]
    normal = np.array([-math.sin(angle), 0.0, math.cos(angle)])
    assert np.linalg.norm(force) > 0.05
    assert abs(force @ normal) < 1e-6 * np.linalg.norm(force)


def test_set_world_swaps_the_surface_map() -> None:
    world = split_world(terrain.TerrainClass.CONCRETE, terrain.TerrainClass.SAND)
    m, d, adapter = on_world(world)
    adapter.configure_table(m, nominal_table())
    d.qpos[0] = 0.5
    mujoco.mj_step1(m, d)
    adapter.apply(m, d)
    assert adapter.classes[0] == terrain.TerrainClass.SAND
    adapter.set_world(split_world(terrain.TerrainClass.GRASS, terrain.TerrainClass.GRASS))
    adapter.apply(m, d)
    assert adapter.classes[0] == terrain.TerrainClass.GRASS
    with pytest.raises(ValueError, match="shape"):
        adapter.set_world(wd.flat((1.0, 1.0)))

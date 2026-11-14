"""Build the MuJoCo model of the worm from a WormSpec."""

from __future__ import annotations

import mujoco
import numpy as np

from worm.spec import WormSpec
from worm.world import World

FLOOR = "floor"
HFIELD = "terrain"
# The compiled height range of every hfield, in metres. It is fixed so that a model
# can load any world of the same shape without a recompile.
HFIELD_ZMAX = 2.0
_HFIELD_BASE = 0.1  # m, thickness below the lowest point


def _segment_xml(spec: WormSpec, index: int) -> str:
    """Return the nested body XML for segment `index` and everything behind it."""
    half = (0.45 * spec.seg_len, 0.5 * spec.seg_width, 0.5 * spec.seg_height)
    geom = (
        f'<geom name="seg{index}" type="box" size="{half[0]} {half[1]} {half[2]}" '
        f'mass="{spec.seg_mass}" rgba="0.2 0.6 0.3 1"/>'
    )
    child = _segment_xml(spec, index + 1) if index + 1 < spec.n_segments else ""
    if index == 0:
        return f'<body name="seg0" pos="0 0 {half[2]}"><freejoint name="root"/>{geom}{child}</body>'
    joint = (
        f'<joint name="yaw{index}" type="hinge" axis="0 0 1" pos="{0.5 * spec.seg_len} 0 0" '
        f'range="{-spec.joint_limit} {spec.joint_limit}" damping="0.02" armature="0.0002"/>'
    )
    return f'<body name="seg{index}" pos="{-spec.seg_len} 0 0">{joint}{geom}{child}</body>'


def to_xml(spec: WormSpec, world: World | None = None) -> str:
    """Return the MJCF string. The head is segment 0 and faces +x.

    Without a world the floor is an infinite plane. With a world the floor is an
    hfield of the world's shape; `build` fills in its heights.
    """
    if world is None:
        terrain_asset = ""
        floor = f'<geom name="{FLOOR}" type="plane" size="0 0 0.1" material="grid"/>'
    else:
        rows, cols = world.shape
        rx, ry = world.half_extent
        terrain_asset = (
            f'<hfield name="{HFIELD}" nrow="{rows}" ncol="{cols}" '
            f'size="{rx} {ry} {HFIELD_ZMAX} {_HFIELD_BASE}"/>'
        )
        floor = f'<geom name="{FLOOR}" type="hfield" hfield="{HFIELD}" rgba="0.6 0.6 0.6 1"/>'
    pairs = "".join(
        f'<pair geom1="{FLOOR}" geom2="seg{i}" condim="3" friction="1 1 0.005 0.0001 0.0001"/>'
        for i in range(spec.n_segments)
    )
    actuators = "".join(
        f'<position name="servo{i}" joint="yaw{i}" kp="{spec.servo_kp}" '
        f'ctrlrange="{-spec.joint_limit} {spec.joint_limit}" '
        f'forcerange="{-spec.servo_torque} {spec.servo_torque}"/>'
        for i in range(1, spec.n_segments)
    )
    return f"""
<mujoco model="worm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{spec.physics_dt}" integrator="implicitfast" cone="elliptic"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85"
             rgb2="0.7 0.7 0.7" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="4 4" texuniform="true"
              specular="0" shininess="0"/>{terrain_asset}
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    {floor}
    <site name="target" pos="0 0 0.01" size="0.03" rgba="0.9 0.2 0.2 1"/>
    {_segment_xml(spec, 0)}
  </worldbody>
  <contact>{pairs}</contact>
  <actuator>{actuators}</actuator>
</mujoco>
"""


def build(spec: WormSpec, world: World | None = None) -> mujoco.MjModel:
    """Compile the worm model. Only floor-segment pairs collide."""
    m = mujoco.MjModel.from_xml_string(to_xml(spec, world))
    if world is not None:
        load_world(m, world)
    return m


def load_world(m: mujoco.MjModel, world: World) -> None:
    """Overwrite the hfield heights in place. The world must have the compiled shape.

    A renderer that already exists must upload the hfield again to show the change.
    """
    if m.nhfield != 1:
        raise ValueError("the model has no hfield; build it with a world")
    shape = (int(m.hfield_nrow[0]), int(m.hfield_ncol[0]))
    if world.shape != shape:
        raise ValueError(f"world shape {world.shape} does not match the hfield shape {shape}")
    data, zmax = world.to_hfield_data()
    if zmax > HFIELD_ZMAX:
        raise ValueError(f"world height {zmax:.2f} m exceeds the hfield range {HFIELD_ZMAX} m")
    m.hfield_data[:] = data * np.float32(zmax / HFIELD_ZMAX)


def segment_body_ids(model: mujoco.MjModel, spec: WormSpec) -> list[int]:
    return [model.body(f"seg{i}").id for i in range(spec.n_segments)]


def pair_ids(model: mujoco.MjModel, spec: WormSpec) -> list[int]:
    """Return the id of the floor contact pair of each segment, in segment order."""
    geom2 = [int(g) for g in model.pair_geom2]
    return [geom2.index(model.geom(f"seg{i}").id) for i in range(spec.n_segments)]

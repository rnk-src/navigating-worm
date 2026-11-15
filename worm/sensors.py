"""Sensor models. They read only what the real robot can measure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from worm.spec import N_TERRAIN_CLASSES, ObsLayout, WormSpec
from worm.terrain import TerrainClass
from worm.world import World

Array = NDArray[np.float64]
CLASS_AHEAD = 0.25  # m, the camera classifies the ground this far ahead of the head
HEIGHT_SCALE = 0.1  # m, a height difference is divided by this, then clipped to [-1, 1]
# Height sample points in the head frame, [forward, left] in metres: a 2 x 2 downsample
# of the footprint of a downward-tilted ToF array. Order: near right, near left, far right,
# far left.
HEIGHT_POINTS: Array = np.array([[0.15, -0.06], [0.15, 0.06], [0.30, -0.06], [0.30, 0.06]])


@dataclass(frozen=True)
class SensorParams:
    gyro_noise: float = 0.02  # rad/s std
    gyro_bias: float = 0.02  # rad/s std of the per-episode bias
    tilt_noise: float = 0.01  # rad std
    bearing_noise: float = 0.05  # rad std (EKF heading + GPS error)
    distance_noise: float = 0.03  # m std
    class_error: float = 0.05  # probability that the terrain classifier reports a wrong class
    tof_noise: float = 0.003  # m std of one height reading
    tof_quant: float = 0.001  # m, resolution of one height reading


def head_pose(data: mujoco.MjData, head_id: int) -> tuple[Array, float, float, float]:
    """Return head (xy, yaw, roll, pitch) from the ground-truth state."""
    rot = data.xmat[head_id].reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    pitch = -math.asin(float(np.clip(rot[2, 0], -1.0, 1.0)))
    roll = math.atan2(rot[2, 1], rot[2, 2])
    return data.xpos[head_id, :2].copy(), yaw, roll, pitch


def read_terrain(
    world: World | None,
    xy: Array,
    yaw: float,
    params: SensorParams,
    rng: np.random.Generator,
    default: TerrainClass = TerrainClass.CONCRETE,
) -> dict[str, Array]:
    """Return the obs v2 terrain measurements ahead of a head at `xy` with heading `yaw`.

    `terrain_ahead` is the one-hot class `CLASS_AHEAD` in front, with classifier errors.
    `height_ahead` is the scaled ground height at `HEIGHT_POINTS` relative to the ground
    under the head. Without a world the ground is level and of class `default`.
    """
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-forward[1], forward[0]])
    cls = int(default) if world is None else int(world.class_at(xy + CLASS_AHEAD * forward))
    if params.class_error > 0.0 and rng.random() < params.class_error:
        cls = (cls + int(rng.integers(1, N_TERRAIN_CLASSES))) % N_TERRAIN_CLASSES
    one_hot = np.zeros(N_TERRAIN_CLASSES)
    one_hot[cls] = 1.0
    heights = np.zeros(len(HEIGHT_POINTS))
    if world is not None:
        points = xy + HEIGHT_POINTS[:, :1] * forward + HEIGHT_POINTS[:, 1:] * left
        heights = world.height_at(points) - float(world.height_at(xy))
        if params.tof_noise > 0.0:
            heights = heights + rng.normal(0.0, params.tof_noise, len(heights))
        if params.tof_quant > 0.0:
            heights = np.round(heights / params.tof_quant) * params.tof_quant
        heights = np.clip(heights / HEIGHT_SCALE, -1.0, 1.0)
    return {"terrain_ahead": one_hot, "height_ahead": heights}


class SensorSuite:
    def __init__(self, model: mujoco.MjModel, spec: WormSpec, layout: ObsLayout) -> None:
        self._spec = spec
        self._layout = layout
        self._head = model.body("seg0").id
        self._params = SensorParams()
        self._gyro_bias = 0.0

    def reset(self, params: SensorParams, rng: np.random.Generator) -> None:
        self._params = params
        self._gyro_bias = float(rng.normal(0.0, params.gyro_bias))

    def read(
        self, data: mujoco.MjData, target: Array, rng: np.random.Generator
    ) -> dict[str, Array]:
        """Return scaled, noisy measurements keyed by `ObsLayout` slice name."""
        spec, layout, p = self._spec, self._layout, self._params
        xy, yaw, roll, pitch = head_pose(data, self._head)
        res = spec.servo_resolution
        joint_pos = np.round(data.qpos[7:] / res) * res
        yaw_rate = data.qvel[5] + self._gyro_bias + rng.normal(0.0, p.gyro_noise)
        tilt = np.array([roll, pitch]) + rng.normal(0.0, p.tilt_noise, 2)
        delta = target - xy
        bearing = math.atan2(delta[1], delta[0]) - yaw + rng.normal(0.0, p.bearing_noise)
        distance = float(np.linalg.norm(delta)) + rng.normal(0.0, p.distance_noise)
        return {
            "joint_pos": joint_pos / spec.joint_limit,
            "yaw_rate": np.array([yaw_rate / layout.yaw_rate_scale]),
            "roll_pitch": tilt,
            "bearing": np.array([math.sin(bearing), math.cos(bearing)]),
            "distance": np.array([np.clip(distance, 0.0, layout.max_distance)])
            / layout.max_distance,
        }

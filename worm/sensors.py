"""Sensor models. They read only what the real robot can measure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from worm.spec import ObsLayout, WormSpec

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SensorParams:
    gyro_noise: float = 0.02  # rad/s std
    gyro_bias: float = 0.02  # rad/s std of the per-episode bias
    tilt_noise: float = 0.01  # rad std
    bearing_noise: float = 0.05  # rad std (EKF heading + GPS error)
    distance_noise: float = 0.03  # m std


def head_pose(data: mujoco.MjData, head_id: int) -> tuple[Array, float, float, float]:
    """Return head (xy, yaw, roll, pitch) from the ground-truth state."""
    rot = data.xmat[head_id].reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    pitch = -math.asin(float(np.clip(rot[2, 0], -1.0, 1.0)))
    roll = math.atan2(rot[2, 1], rot[2, 2])
    return data.xpos[head_id, :2].copy(), yaw, roll, pitch


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

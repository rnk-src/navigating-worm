"""Single source of truth for morphology, action layout and observation layout."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

N_TERRAIN_CLASSES = 3


@dataclass(frozen=True)
class WormSpec:
    """Body and servo constants, matched to the Feetech SCS0009 datasheet."""

    n_segments: int = 5
    seg_len: float = 0.06  # m, joint-to-joint pitch
    seg_width: float = 0.03  # m
    seg_height: float = 0.025  # m
    seg_mass: float = 0.04  # kg, servo + printed shell
    joint_limit: float = math.radians(60.0)  # rad, mechanical stop
    servo_torque: float = 0.225  # N m (2.3 kg cm stall)
    servo_speed: float = 8.0  # rad/s, loaded speed cap
    servo_kp: float = 3.0  # N m / rad
    servo_resolution: float = math.radians(300.0) / 1024  # rad per feedback count
    physics_dt: float = 0.002  # s
    control_dt: float = 0.05  # s (20 Hz)

    @property
    def n_joints(self) -> int:
        return self.n_segments - 1

    @property
    def substeps(self) -> int:
        return round(self.control_dt / self.physics_dt)


@dataclass(frozen=True)
class ActionSpec:
    """Policy action: 3 CPG parameters plus one residual per joint, all in [-1, 1]."""

    n_joints: int = 4
    amp_max: float = math.radians(50.0)  # rad
    freq_min: float = 0.3  # Hz
    freq_max: float = 2.0  # Hz
    bias_max: float = math.radians(20.0)  # rad
    residual_max: float = math.radians(15.0)  # rad
    wave: float = 2.0 * math.pi / 5.0  # rad, phase lag between adjacent joints

    @property
    def dim(self) -> int:
        return 3 + self.n_joints


@dataclass(frozen=True)
class ObsLayout:
    """Named slices of the flat observation vector."""

    n_joints: int = 4
    action_dim: int = 7
    max_distance: float = 3.0  # m, the distance observation is clipped here, then divided by it
    yaw_rate_scale: float = 5.0  # rad/s, the yaw rate observation is divided by this
    slices: dict[str, slice] = field(init=False)

    def __post_init__(self) -> None:
        sizes = {
            "joint_pos": self.n_joints,
            "joint_delta": self.n_joints,
            "yaw_rate": 1,
            "roll_pitch": 2,
            "bearing": 2,  # sin, cos of the target bearing in the head frame
            "distance": 1,
            "prev_action": self.action_dim,
            "terrain": N_TERRAIN_CLASSES,
        }
        slices: dict[str, slice] = {}
        start = 0
        for name, size in sizes.items():
            slices[name] = slice(start, start + size)
            start += size
        object.__setattr__(self, "slices", slices)

    @property
    def dim(self) -> int:
        return int(max(s.stop for s in self.slices.values()))


def action_spec(spec: WormSpec) -> ActionSpec:
    return ActionSpec(n_joints=spec.n_joints)


def obs_layout(spec: WormSpec) -> ObsLayout:
    return ObsLayout(n_joints=spec.n_joints, action_dim=action_spec(spec).dim)

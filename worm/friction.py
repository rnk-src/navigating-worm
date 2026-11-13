"""Friction seam: make the ground resist sideways motion more than forward motion."""

from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np

from worm.terrain import FrictionParams

_GRAVITY = 9.81


class FrictionModel(Protocol):
    def configure(self, model: mujoco.MjModel, params: FrictionParams) -> None:
        """Set per-episode model constants. Call once after a reset."""

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Set per-step forces. Call between `mj_step1` and `mj_step2`."""


class ContactAnisotropic:
    """MuJoCo per-axis contact friction.

    The contact tangent axes follow the world frame for a flat floor, not the segment.
    The anisotropy is therefore wrong as soon as a segment yaws. Kept for comparison only.
    """

    def configure(self, model: mujoco.MjModel, params: FrictionParams) -> None:
        model.pair_friction[:, 0] = params.mu_forward
        model.pair_friction[:, 1] = params.mu_lateral

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        return None


class ResistiveForce:
    """Body-frame drag on each segment, applied through `xfrc_applied`.

    The contact itself is made near-frictionless. Each segment then gets a force
    `-m g mu tanh(v / v0)` along its own forward and lateral axes. The `tanh` keeps
    the force smooth near zero speed, which the explicit integration needs.
    """

    def __init__(self, body_ids: list[int], v0: float = 0.05) -> None:
        self._ids = np.asarray(body_ids)
        self._v0 = v0
        self._mu = np.zeros(2)
        self._load = np.zeros(len(body_ids))

    def configure(self, model: mujoco.MjModel, params: FrictionParams) -> None:
        model.pair_friction[:, :2] = 1e-4
        self._mu[:] = (params.mu_forward, params.mu_lateral)
        self._load = model.body_mass[self._ids] * _GRAVITY

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        ids = self._ids
        # cvel is [angular, linear] at the subtree COM of the root body; move it to each body.
        offset = data.xpos[ids] - data.subtree_com[model.body_rootid[ids]]
        vel = data.cvel[ids, 3:] + np.cross(data.cvel[ids, :3], offset)
        rot = data.xmat[ids].reshape(-1, 3, 3)
        force = np.zeros((len(ids), 3))
        for axis in (0, 1):
            direction = rot[:, :, axis].copy()
            direction[:, 2] = 0.0
            direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
            speed = np.einsum("ij,ij->i", vel, direction)
            magnitude = self._load * self._mu[axis] * np.tanh(speed / self._v0)
            force -= magnitude[:, None] * direction
        data.xfrc_applied[ids, :3] = force


def step(model: mujoco.MjModel, data: mujoco.MjData, friction: FrictionModel) -> None:
    """Advance physics one step with the friction model applied."""
    mujoco.mj_step1(model, data)
    friction.apply(model, data)
    mujoco.mj_step2(model, data)

"""Friction seam: make the ground resist sideways motion more than forward motion."""

from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np
from numpy.typing import NDArray

from worm.terrain import FrictionParams, SurfaceTable
from worm.world import World

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


class TerrainResistive:
    """`ResistiveForce` on a heightfield world, with the surface looked up per segment.

    Every physics step, each segment takes its friction, viscous drag and contact
    softness from the cell under it. The drag acts in the terrain tangent plane, and
    the load is the weight component along the surface normal.

    On a flat world with one surface this gives exactly the forces of `ResistiveForce`.

    `hybrid=True` moves a share of the friction into the MuJoCo contact as isotropic
    Coulomb friction, `stiction_fraction * mu_forward`. A real contact can hold a
    segment still on a slope; the smooth `tanh` drag alone always creeps. The drag
    then supplies only the rest, so the totals at speed stay the same.

    The default share is 0.7. At the nominal forward friction of 0.3 the contact then
    holds on slopes up to atan(0.21) = 11.9 degrees. A share of 0.5 creeps 9 cm in 10 s
    on a 10 degree slope.
    """

    def __init__(
        self,
        body_ids: list[int],
        pair_ids: list[int],
        world: World,
        v0: float = 0.05,
        hybrid: bool = False,
        stiction_fraction: float = 0.7,
    ) -> None:
        self._ids = np.asarray(body_ids)
        self._pairs = np.asarray(pair_ids)
        self._v0 = v0
        self._hybrid = hybrid
        self._stiction = stiction_fraction
        self._mu = np.zeros((3, 2))
        self._viscous = np.zeros(3)
        self._timeconst: NDArray[np.float64] | None = None
        self._weight = np.zeros(len(body_ids))
        self._classes = np.zeros(len(body_ids), dtype=np.uint8)
        self._shape = world.shape
        self.set_world(world)

    def set_world(self, world: World) -> None:
        """Use another world of the same shape."""
        if world.shape != self._shape:
            raise ValueError(f"world shape {world.shape} does not match {self._shape}")
        slope = world.gradient
        normal = np.concatenate([-slope, np.ones((*slope.shape[:2], 1))], axis=-1)
        self._normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        self._surface = world.surface
        self._cell = world.cell
        self._half = np.array(world.half_extent)
        self._last = np.array([world.shape[1] - 1, world.shape[0] - 1])

    def configure(self, model: mujoco.MjModel, params: FrictionParams) -> None:
        """Use one friction for every class, with no viscous drag. Contact softness is kept."""
        self._weight = model.body_mass[self._ids] * _GRAVITY
        self._mu[:] = (params.mu_forward, params.mu_lateral)
        self._viscous[:] = 0.0
        self._timeconst = None
        if not self._hybrid:
            model.pair_friction[:, :2] = 1e-4

    def configure_table(self, model: mujoco.MjModel, table: SurfaceTable) -> None:
        """Use the per-class surface properties of this episode."""
        self._weight = model.body_mass[self._ids] * _GRAVITY
        self._mu, self._viscous, self._timeconst = table.arrays()
        if not self._hybrid:
            model.pair_friction[:, :2] = 1e-4

    @property
    def classes(self) -> NDArray[np.uint8]:
        """Return the class under each segment at the last `apply`."""
        return self._classes

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        ids = self._ids
        position = data.xpos[ids]
        # Nearest grid point: [column, row] from [x, y].
        index = np.rint((position[:, :2] + self._half) / self._cell).astype(np.int_)
        index = np.clip(index, 0, self._last)
        classes = self._surface[index[:, 1], index[:, 0]]
        normal = self._normal[index[:, 1], index[:, 0]]
        self._classes = classes
        mu = self._mu[classes]
        if self._hybrid:
            contact = self._stiction * mu[:, 0]
            model.pair_friction[self._pairs, 0] = contact
            model.pair_friction[self._pairs, 1] = contact
            mu = mu - contact[:, None]
        if self._timeconst is not None:
            # This takes effect at the next collision pass, one physics step later.
            model.pair_solref[self._pairs, 0] = self._timeconst[classes]

        offset = position - data.subtree_com[model.body_rootid[ids]]
        vel = data.cvel[ids, 3:] + np.cross(data.cvel[ids, :3], offset)
        rot = data.xmat[ids].reshape(-1, 3, 3)
        load = self._weight * normal[:, 2]
        force = np.zeros((len(ids), 3))
        for axis in (0, 1):
            direction = rot[:, :, axis]
            direction = direction - np.einsum("ij,ij->i", direction, normal)[:, None] * normal
            direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
            speed = np.einsum("ij,ij->i", vel, direction)
            magnitude = load * mu[:, axis] * np.tanh(speed / self._v0)
            force -= magnitude[:, None] * direction
        tangent = vel - np.einsum("ij,ij->i", vel, normal)[:, None] * normal
        force -= self._viscous[classes][:, None] * tangent
        data.xfrc_applied[ids, :3] = force


def step(model: mujoco.MjModel, data: mujoco.MjData, friction: FrictionModel) -> None:
    """Advance physics one step with the friction model applied."""
    mujoco.mj_step1(model, data)
    friction.apply(model, data)
    mujoco.mj_step2(model, data)

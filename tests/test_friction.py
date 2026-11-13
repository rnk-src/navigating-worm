"""Friction behaviour, measured by coasting one segment and comparing distances."""

import math

import mujoco
import numpy as np
import pytest

from worm import friction, model
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

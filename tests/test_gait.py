import math

import mujoco
import numpy as np
import pytest

from worm import friction, gait, model
from worm.spec import WormSpec
from worm.terrain import FrictionParams


def rollout(seconds: float, amp_deg: float, bias_deg: float) -> tuple[float, float]:
    """Run the open-loop gait at 1 Hz. Return (head x in metres, total head yaw in degrees)."""
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    fric = friction.ResistiveForce(model.segment_body_ids(m, spec))
    fric.configure(m, FrictionParams(0.3, 1.2))
    phase, yaw_total = 0.0, 0.0
    for _ in range(round(seconds / 0.05)):
        phase += 2.0 * math.pi * 1.0 * 0.05
        d.ctrl[:] = gait.serpenoid(phase, math.radians(amp_deg), 1.26, math.radians(bias_deg), 4)
        for _ in range(25):
            friction.step(m, d, fric)
            yaw_total += d.qvel[5] * 0.002
    return float(d.qpos[0]), math.degrees(yaw_total)


def test_wave_lags_by_the_wave_number_per_joint() -> None:
    targets = gait.serpenoid(math.pi / 2, 1.0, math.pi / 2, 0.0, 3)
    assert targets == pytest.approx([1.0, 0.0, -1.0], abs=1e-12)


def test_zero_amplitude_gives_a_constant_bend() -> None:
    targets = gait.serpenoid(1.7, 0.0, 1.0, 0.1, 4)
    assert np.ptp(targets) == 0.0 and abs(targets[0]) == pytest.approx(0.1)


def test_open_loop_gait_advances_more_than_20cm_in_10s() -> None:
    x, _ = rollout(10.0, amp_deg=45.0, bias_deg=0.0)
    assert x > 0.20


def test_positive_bias_turns_left_and_negative_turns_right() -> None:
    _, left = rollout(5.0, amp_deg=40.0, bias_deg=10.0)
    _, right = rollout(5.0, amp_deg=40.0, bias_deg=-10.0)
    assert left > 20.0
    assert right < -20.0

import mujoco
import numpy as np

from worm import model
from worm.spec import WormSpec


def test_model_has_one_servo_per_joint() -> None:
    m = model.build(WormSpec())
    assert (m.nu, m.nq, m.nv) == (4, 11, 10)  # free joint (7/6) + 4 hinges


def test_worm_rests_flat_on_the_floor() -> None:
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)
    ids = model.segment_body_ids(m, spec)
    assert np.allclose(d.xpos[ids, 2], 0.0125, atol=0.002)  # half the 25 mm height


def test_head_faces_plus_x_and_segments_trail_behind() -> None:
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    xs = d.xpos[model.segment_body_ids(m, spec), 0]
    assert np.allclose(xs, [0.0, -0.06, -0.12, -0.18, -0.24])


def test_servo_holds_a_commanded_angle_within_torque_limit() -> None:
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    d.ctrl[:] = [0.5, 0.0, 0.0, 0.0]
    for _ in range(500):
        mujoco.mj_step(m, d)
        assert np.all(np.abs(d.actuator_force) <= 0.225 + 1e-9)
    assert abs(d.qpos[7] - 0.5) < 0.05

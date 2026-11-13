import math

import mujoco
import numpy as np
import pytest

from worm import model
from worm.sensors import SensorParams, SensorSuite
from worm.spec import WormSpec, obs_layout

NOISELESS = SensorParams(0.0, 0.0, 0.0, 0.0, 0.0)


def read(
    target: tuple[float, float],
    yaw_deg: float = 0.0,
    joints: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    yaw_rate: float = 0.0,
    params: SensorParams = NOISELESS,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Place the head at the origin with the given yaw and read the sensors once."""
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    yaw = math.radians(yaw_deg)
    d.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    d.qpos[7:] = joints
    d.qvel[5] = yaw_rate
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(seed)
    suite = SensorSuite(m, spec, obs_layout(spec))
    suite.reset(params, rng)
    return suite.read(d, np.array(target), rng)


@pytest.mark.parametrize(
    ("target", "yaw_deg", "sin_cos"),
    [
        ((1.0, 0.0), 0.0, (0.0, 1.0)),  # straight ahead
        ((0.0, 1.0), 0.0, (1.0, 0.0)),  # to the left
        ((0.0, 1.0), 90.0, (0.0, 1.0)),  # ahead again once the head has turned left
        ((-1.0, 0.0), 90.0, (1.0, 0.0)),  # behind in the world, left of the head
    ],
)
def test_bearing_is_measured_in_the_head_frame(
    target: tuple[float, float], yaw_deg: float, sin_cos: tuple[float, float]
) -> None:
    assert read(target, yaw_deg)["bearing"] == pytest.approx(sin_cos, abs=1e-9)


def test_distance_is_scaled_by_3m_and_clipped() -> None:
    assert read((1.5, 0.0))["distance"] == pytest.approx([0.5])
    assert read((0.0, -40.0))["distance"] == pytest.approx([1.0])


def test_joint_feedback_is_quantized_to_servo_counts() -> None:
    # One count is 300 deg / 1024 = 0.005113 rad. 0.2 rad is 39.11 counts -> 39 counts.
    # The reading is scaled by the 60 deg joint limit.
    reading = read((1.0, 0.0), joints=(0.2, 0.0, 0.0, -0.2))["joint_pos"]
    count = math.radians(300.0) / 1024
    assert reading == pytest.approx(
        [39 * count / (math.pi / 3), 0.0, 0.0, -39 * count / (math.pi / 3)]
    )


def test_yaw_rate_is_scaled_by_5_rad_per_s() -> None:
    assert read((1.0, 0.0), yaw_rate=2.5)["yaw_rate"] == pytest.approx([0.5])


def test_flat_worm_reads_zero_roll_and_pitch() -> None:
    assert read((1.0, 0.0), yaw_deg=70.0)["roll_pitch"] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_gyro_bias_is_constant_within_an_episode_and_differs_between_episodes() -> None:
    params = SensorParams(0.0, 0.1, 0.0, 0.0, 0.0)
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    suite = SensorSuite(m, spec, obs_layout(spec))
    rng = np.random.default_rng(1)
    target = np.array([1.0, 0.0])
    suite.reset(params, rng)
    first = [float(suite.read(d, target, rng)["yaw_rate"][0]) for _ in range(3)]
    suite.reset(params, rng)
    second = float(suite.read(d, target, rng)["yaw_rate"][0])
    assert first[0] != 0.0 and first == [first[0]] * 3
    assert second != first[0]


def test_noise_changes_bearing_and_distance_readings() -> None:
    noisy = SensorParams(0.0, 0.0, 0.0, 0.05, 0.03)
    a, b = read((1.5, 0.0), params=noisy, seed=1), read((1.5, 0.0), params=noisy, seed=2)
    assert a["bearing"][0] != b["bearing"][0]
    assert a["distance"][0] != b["distance"][0]
    assert abs(a["distance"][0] - 0.5) < 0.1  # 0.03 m noise on 1.5 m, scaled by 3 m

import numpy as np
import pytest

from worm.servo import ServoModel, ServoParams
from worm.spec import WormSpec

IDEAL = ServoParams(delay_steps=0, lag_tau=0.0, deadband=0.0)


def run(params: ServoParams, commands: list[float]) -> list[float]:
    """Send the same command to every joint. Return joint 0 of each applied target."""
    servo = ServoModel(WormSpec())
    servo.reset(params)
    return [float(servo.command(np.full(4, c))[0]) for c in commands]


def test_command_takes_effect_after_the_delay() -> None:
    params = ServoParams(delay_steps=2, lag_tau=0.0, deadband=0.0)
    assert run(params, [0.2, 0.2, 0.2, 0.2]) == pytest.approx([0.0, 0.0, 0.2, 0.2])


def test_speed_cap_limits_the_move_per_step() -> None:
    # 8 rad/s * 0.05 s = 0.4 rad per control step.
    assert run(IDEAL, [1.0, 1.0, 1.0]) == pytest.approx([0.4, 0.8, 1.0])


def test_voltage_sag_scales_the_speed_cap() -> None:
    params = ServoParams(delay_steps=0, lag_tau=0.0, deadband=0.0, speed_scale=0.5)
    assert run(params, [1.0, 1.0]) == pytest.approx([0.2, 0.4])


def test_target_is_clamped_to_the_joint_limit() -> None:
    assert run(IDEAL, [-5.0] * 4)[-1] == pytest.approx(-np.pi / 3)


def test_changes_smaller_than_the_deadband_are_ignored() -> None:
    params = ServoParams(delay_steps=0, lag_tau=0.0, deadband=0.01)
    assert run(params, [0.3, 0.305, 0.32]) == pytest.approx([0.3, 0.3, 0.32])


def test_first_order_lag_closes_half_the_error_per_step() -> None:
    # alpha = 1 - exp(-dt / tau) = 0.5 when tau = dt / ln 2.
    params = ServoParams(delay_steps=0, lag_tau=0.05 / np.log(2.0), deadband=0.0)
    assert run(params, [0.2, 0.2, 0.2]) == pytest.approx([0.1, 0.15, 0.175])


def test_reset_forgets_the_previous_episode() -> None:
    servo = ServoModel(WormSpec())
    servo.reset(ServoParams(delay_steps=1, lag_tau=0.0, deadband=0.0))
    servo.command(np.full(4, 0.3))
    servo.command(np.full(4, 0.3))
    servo.reset(ServoParams(delay_steps=1, lag_tau=0.0, deadband=0.0))
    assert servo.command(np.full(4, 0.3)) == pytest.approx(np.zeros(4))

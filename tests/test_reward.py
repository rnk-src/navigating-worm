import numpy as np
import pytest

from worm.reward import StepState, reward

ZERO = np.zeros(7)
STILL = StepState(distance=1.0, bearing=0.0)


def total(prev: StepState, cur: StepState, action=ZERO, prev_action=ZERO) -> float:  # type: ignore[no-untyped-def]
    return reward(prev, cur, action, prev_action)[0]


def test_moving_1cm_closer_earns_one_point_more_than_standing_still() -> None:
    closer = total(STILL, StepState(0.99, 0.0))
    assert closer - total(STILL, STILL) == pytest.approx(1.0)


def test_moving_away_is_penalized() -> None:
    assert total(STILL, StepState(1.01, 0.0)) < total(STILL, STILL)


def test_reaching_the_target_adds_a_bonus_larger_than_any_step_progress() -> None:
    bonus = total(STILL, StepState(1.0, 0.0, reached=True)) - total(STILL, STILL)
    assert bonus == pytest.approx(20.0)


def test_facing_the_target_beats_facing_away() -> None:
    assert total(STILL, STILL) > total(STILL, StepState(1.0, np.pi))


def test_changing_the_action_is_penalized() -> None:
    assert total(STILL, STILL, np.ones(7), ZERO) < total(STILL, STILL, np.ones(7), np.ones(7))


def test_joint_residuals_cost_energy_but_cpg_parameters_do_not() -> None:
    cpg_only = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    residual = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert total(STILL, STILL, cpg_only, cpg_only) == pytest.approx(total(STILL, STILL))
    assert total(STILL, STILL, residual, residual) < total(STILL, STILL)


def test_flipping_is_penalized() -> None:
    assert total(STILL, StepState(1.0, 0.0, flipped=True)) < total(STILL, STILL) - 10.0


def test_terms_sum_to_the_total() -> None:
    value, terms = reward(STILL, StepState(0.9, 0.3, reached=True), np.ones(7), ZERO)
    assert value == pytest.approx(sum(terms.values()))
    assert terms["progress"] == pytest.approx(10.0)

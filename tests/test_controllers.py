"""Hand-written controllers."""

import math

import numpy as np

from worm.controllers import bearing_steer
from worm.spec import WormSpec, obs_layout


def observe(bearing_rad: float) -> np.ndarray:
    layout = obs_layout(WormSpec())
    obs = np.zeros(layout.dim, dtype=np.float32)
    obs[layout.slices["bearing"]] = (math.sin(bearing_rad), math.cos(bearing_rad))
    return obs


def test_bearing_steer_turns_toward_the_target_side() -> None:
    steer = bearing_steer(obs_layout(WormSpec()))
    left, ahead, right = (steer(observe(b)) for b in (0.5, 0.0, -0.5))
    # A positive turn bias turns the worm left.
    assert left[2] > 0.3 and right[2] < -0.3 and ahead[2] == 0.0
    assert left.shape == (7,) and left.dtype == np.float32
    assert left[0] > 0.0  # the gait runs


def test_bearing_steer_saturates_and_scales_with_the_gain() -> None:
    layout = obs_layout(WormSpec())
    assert bearing_steer(layout)(observe(3.0))[2] == 1.0
    assert bearing_steer(layout, gain=0.5)(observe(0.5))[2] == np.float32(0.25)

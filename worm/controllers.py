"""Hand-written controllers. They share the `Policy` signature with trained policies."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from worm.spec import ObsLayout

Policy = Callable[[NDArray[np.float32]], NDArray[np.float32]]


def bearing_steer(layout: ObsLayout, gain: float = 1.0) -> Policy:
    """Return a controller that runs a steady gait and steers toward the target.

    The turn bias is `gain` times the target bearing in radians, clipped to [-1, 1].
    A positive bias turns the worm left. This is the fallback when no policy is trained.
    """
    bearing = layout.slices["bearing"]
    dim = layout.action_dim

    def policy(obs: NDArray[np.float32]) -> NDArray[np.float32]:
        sin_b, cos_b = obs[bearing]
        action = np.zeros(dim, dtype=np.float32)
        action[0], action[1] = 0.8, 0.2  # amplitude and frequency
        action[2] = np.clip(gain * np.arctan2(sin_b, cos_b), -1.0, 1.0)
        return action

    return policy

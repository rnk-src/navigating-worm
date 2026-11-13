"""Reward for reaching a target. Pure functions only."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StepState:
    distance: float  # m, head to target
    bearing: float  # rad, target direction in the head frame
    reached: bool = False
    flipped: bool = False


def reward(
    prev: StepState,
    cur: StepState,
    action: NDArray[np.float64],
    prev_action: NDArray[np.float64],
) -> tuple[float, dict[str, float]]:
    """Return the step reward and its named terms."""
    terms = {
        "progress": 100.0 * (prev.distance - cur.distance),
        "alignment": 0.05 * math.cos(cur.bearing),
        "action_rate": -0.05 * float(np.mean((action - prev_action) ** 2)),
        "energy": -0.01 * float(np.mean(action[3:] ** 2)),
        "success": 20.0 if cur.reached else 0.0,
        "flip": -20.0 if cur.flipped else 0.0,
    }
    return sum(terms.values()), terms

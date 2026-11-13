"""Servo command model: what target the servo actually tracks after its own delays."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from worm.spec import WormSpec

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ServoParams:
    delay_steps: int = 1  # control steps between command and motion
    lag_tau: float = 0.03  # s, first-order lag time constant
    deadband: float = 0.01  # rad, smaller target changes are ignored
    speed_scale: float = 1.0  # multiplies the speed cap (voltage sag lowers it)
    strength: float = 1.0  # multiplies stall torque and stiffness


class ServoModel:
    """Turn commanded joint targets into the targets the position actuators receive."""

    def __init__(self, spec: WormSpec) -> None:
        self._spec = spec
        self._params = ServoParams()
        self._queue: deque[Array] = deque()
        self._applied = np.zeros(spec.n_joints)

    def reset(self, params: ServoParams) -> None:
        self._params = params
        self._applied = np.zeros(self._spec.n_joints)
        self._queue = deque([self._applied.copy() for _ in range(params.delay_steps)])

    def command(self, target: Array) -> Array:
        """Accept one command per control step. Return the applied target."""
        spec, p = self._spec, self._params
        self._queue.append(np.clip(target, -spec.joint_limit, spec.joint_limit))
        error = self._queue.popleft() - self._applied
        error[np.abs(error) < p.deadband] = 0.0
        alpha = 1.0 - np.exp(-spec.control_dt / p.lag_tau) if p.lag_tau > 0 else 1.0
        max_move = spec.servo_speed * p.speed_scale * spec.control_dt
        self._applied = self._applied + np.clip(alpha * error, -max_move, max_move)
        return self._applied.copy()

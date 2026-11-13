"""Domain randomization: sample the physics of one episode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worm import terrain
from worm.sensors import SensorParams
from worm.servo import ServoParams
from worm.terrain import FrictionParams, TerrainClass


@dataclass(frozen=True)
class PhysicsParams:
    friction: FrictionParams = terrain.NOMINAL
    servo: ServoParams = ServoParams()
    sensor: SensorParams = SensorParams()
    mass_scale: float = 1.0


# M1 trains one policy on a single wide friction range that covers every class.
_WIDE_FORWARD = (0.15, 0.8)
_WIDE_RATIO = (2.0, 6.0)
_NOMINAL_RATIO = terrain.NOMINAL.mu_lateral / terrain.NOMINAL.mu_forward


class DomainRandomizer:
    """`level` in [0, 1] scales every range. Level 0 returns the nominal physics."""

    def __init__(self, level: float = 1.0) -> None:
        self.level = float(np.clip(level, 0.0, 1.0))

    def _span(self, rng: np.random.Generator, nominal: float, low: float, high: float) -> float:
        return nominal + self.level * (float(rng.uniform(low, high)) - nominal)

    def sample(self, rng: np.random.Generator, cls: TerrainClass | None = None) -> PhysicsParams:
        """Sample one episode. `cls=None` uses the wide M1 friction range."""
        if self.level == 0.0:
            return PhysicsParams()
        if cls is None:
            mu_forward = self._span(rng, terrain.NOMINAL.mu_forward, *_WIDE_FORWARD)
            ratio = self._span(rng, _NOMINAL_RATIO, *_WIDE_RATIO)
            fric = FrictionParams(mu_forward, mu_forward * ratio)
        else:
            fric = terrain.params(cls, rng)
        servo = ServoParams(
            delay_steps=int(rng.integers(1, 2 + round(2 * self.level))),
            lag_tau=self._span(rng, 0.03, 0.01, 0.06),
            deadband=self._span(rng, 0.01, 0.0, 0.02),
            speed_scale=self._span(rng, 1.0, 0.7, 1.1),
            strength=self._span(rng, 1.0, 0.7, 1.1),
        )
        return PhysicsParams(fric, servo, SensorParams(), self._span(rng, 1.0, 0.8, 1.3))

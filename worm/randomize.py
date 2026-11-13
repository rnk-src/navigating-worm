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
    terrain: TerrainClass = TerrainClass.CONCRETE


_NOMINAL_RATIO = terrain.NOMINAL.mu_lateral / terrain.NOMINAL.mu_forward


class DomainRandomizer:
    """`level` in [0, 1] scales every range. Level 0 returns the nominal physics."""

    def __init__(self, level: float = 1.0) -> None:
        self.level = float(np.clip(level, 0.0, 1.0))

    def _span(self, rng: np.random.Generator, nominal: float, low: float, high: float) -> float:
        return nominal + self.level * (float(rng.uniform(low, high)) - nominal)

    def _shrink(self, nominal: float, value: float) -> float:
        return value if self.level == 1.0 else nominal + self.level * (value - nominal)

    def sample(self, rng: np.random.Generator, cls: TerrainClass | None = None) -> PhysicsParams:
        """Sample one episode. `cls=None` draws a terrain class uniformly."""
        if cls is None:
            cls = TerrainClass(int(rng.integers(len(TerrainClass))))
        if self.level == 0.0:
            return PhysicsParams(terrain=cls)
        raw = terrain.params(cls, rng)
        mu_forward = self._shrink(terrain.NOMINAL.mu_forward, raw.mu_forward)
        ratio = self._shrink(_NOMINAL_RATIO, raw.mu_lateral / raw.mu_forward)
        fric = FrictionParams(mu_forward, mu_forward * ratio)
        servo = ServoParams(
            delay_steps=int(rng.integers(1, 2 + round(2 * self.level))),
            lag_tau=self._span(rng, 0.03, 0.01, 0.06),
            deadband=self._span(rng, 0.01, 0.0, 0.02),
            speed_scale=self._span(rng, 1.0, 0.7, 1.1),
            strength=self._span(rng, 1.0, 0.7, 1.1),
        )
        mass = self._span(rng, 1.0, 0.8, 1.3)
        return PhysicsParams(fric, servo, SensorParams(), mass, cls)

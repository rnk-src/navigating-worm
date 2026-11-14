"""Domain randomization: sample the physics of one episode."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from worm import terrain
from worm.sensors import SensorParams
from worm.servo import ServoParams
from worm.terrain import FrictionParams, SurfaceProps, SurfaceTable, TerrainClass


@dataclass(frozen=True)
class PhysicsParams:
    friction: FrictionParams = terrain.NOMINAL
    servo: ServoParams = ServoParams()
    sensor: SensorParams = SensorParams()
    mass_scale: float = 1.0
    terrain: TerrainClass = TerrainClass.CONCRETE
    surfaces: SurfaceTable | None = None  # per-class surfaces, set only in a world


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

    def sample_surfaces(self, rng: np.random.Generator) -> SurfaceTable:
        """Sample the surface of every terrain class for one episode in a world.

        Level 0 returns the nominal surface of each class. This is a separate draw
        from `sample`, so the flat-plane random stream is unchanged.
        """
        props: list[SurfaceProps] = []
        for cls in TerrainClass:
            nominal = terrain.nominal_surface(cls)
            if self.level == 0.0:
                props.append(nominal)
                continue
            raw = terrain.surface_props(cls, rng)
            mu_forward = self._shrink(nominal.friction.mu_forward, raw.friction.mu_forward)
            ratio = self._shrink(
                nominal.friction.mu_lateral / nominal.friction.mu_forward,
                raw.friction.mu_lateral / raw.friction.mu_forward,
            )
            props.append(
                replace(
                    raw,
                    friction=FrictionParams(mu_forward, mu_forward * ratio),
                    viscous=self._shrink(nominal.viscous, raw.viscous),
                    solref_timeconst=self._shrink(nominal.solref_timeconst, raw.solref_timeconst),
                )
            )
        return SurfaceTable((props[0], props[1], props[2]))

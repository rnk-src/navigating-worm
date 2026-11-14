"""Terrain classes and the friction that each class implies."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrictionParams:
    """Ground resistance of one segment, as Coulomb-like coefficients in the body frame."""

    mu_forward: float
    mu_lateral: float


class TerrainClass(enum.IntEnum):
    CONCRETE = 0
    GRASS = 1
    SAND = 2


NOMINAL = FrictionParams(mu_forward=0.3, mu_lateral=1.2)

# Per class: (mu_forward range, lateral/forward ratio range).
_RANGES: dict[TerrainClass, tuple[tuple[float, float], tuple[float, float]]] = {
    TerrainClass.CONCRETE: ((0.15, 0.45), (3.0, 6.0)),
    TerrainClass.GRASS: ((0.3, 0.6), (2.5, 4.0)),
    TerrainClass.SAND: ((0.4, 0.8), (2.0, 3.0)),
}


def params(cls: TerrainClass, rng: np.random.Generator) -> FrictionParams:
    """Sample friction for one episode on the given terrain class."""
    forward_range, ratio_range = _RANGES[cls]
    mu_forward = float(rng.uniform(*forward_range))
    return FrictionParams(mu_forward, mu_forward * float(rng.uniform(*ratio_range)))


# Per class: amplitude of the fine surface relief, in metres. Sand is smooth.
ROUGHNESS: dict[TerrainClass, float] = {
    TerrainClass.CONCRETE: 0.004,
    TerrainClass.GRASS: 0.002,
    TerrainClass.SAND: 0.0,
}

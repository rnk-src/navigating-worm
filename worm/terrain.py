"""Terrain classes and the friction that each class implies."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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


@dataclass(frozen=True)
class SurfaceProps:
    """Everything the physics needs to know about one surface class in one episode."""

    friction: FrictionParams
    viscous: float  # N s / m per segment, speed-proportional drag
    solref_timeconst: float  # s, contact softness; a larger value lets the segment sink
    roughness: float  # m, amplitude of the fine relief that the generator adds


@dataclass(frozen=True)
class SurfaceTable:
    """The surface properties of one episode, indexed by `TerrainClass`."""

    props: tuple[SurfaceProps, SurfaceProps, SurfaceProps]

    def arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (mu [forward, lateral] (3, 2), viscous (3,), solref timeconst (3,))."""
        mu = np.array([[p.friction.mu_forward, p.friction.mu_lateral] for p in self.props])
        viscous = np.array([p.viscous for p in self.props])
        timeconst = np.array([p.solref_timeconst for p in self.props])
        return mu, viscous, timeconst


# Per class: (viscous range, solref timeconst range, roughness range).
_Range = tuple[float, float]
_SURFACE_RANGES: dict[TerrainClass, tuple[_Range, _Range, _Range]] = {
    TerrainClass.CONCRETE: ((0.0, 0.02), (0.005, 0.005), (0.002, 0.006)),
    TerrainClass.GRASS: ((0.05, 0.15), (0.01, 0.01), (0.001, 0.003)),
    TerrainClass.SAND: ((0.15, 0.4), (0.02, 0.04), (0.0, 0.0)),
}


def surface_props(cls: TerrainClass, rng: np.random.Generator) -> SurfaceProps:
    """Sample the surface properties of one class for one episode."""
    viscous, timeconst, roughness = _SURFACE_RANGES[cls]
    return SurfaceProps(
        friction=params(cls, rng),
        viscous=float(rng.uniform(*viscous)),
        solref_timeconst=float(rng.uniform(*timeconst)),
        roughness=float(rng.uniform(*roughness)),
    )


def nominal_surface(cls: TerrainClass) -> SurfaceProps:
    """Return the midpoint of every range of the class."""
    (f_low, f_high), (r_low, r_high) = _RANGES[cls]
    mu_forward = 0.5 * (f_low + f_high)
    viscous, timeconst, roughness = _SURFACE_RANGES[cls]
    return SurfaceProps(
        friction=FrictionParams(mu_forward, mu_forward * 0.5 * (r_low + r_high)),
        viscous=0.5 * sum(viscous),
        solref_timeconst=0.5 * sum(timeconst),
        roughness=0.5 * sum(roughness),
    )


# Per class: nominal amplitude of the fine surface relief, in metres. Sand is smooth.
ROUGHNESS: dict[TerrainClass, float] = {cls: nominal_surface(cls).roughness for cls in TerrainClass}

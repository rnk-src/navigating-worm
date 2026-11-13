"""Terrain classes and the friction that each class implies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrictionParams:
    """Ground resistance of one segment, as Coulomb-like coefficients in the body frame."""

    mu_forward: float
    mu_lateral: float

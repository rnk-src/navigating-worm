"""Per-class surface properties."""

import numpy as np
import pytest

from worm import terrain
from worm.terrain import TerrainClass


@pytest.mark.parametrize(
    ("cls", "viscous", "timeconst", "roughness"),
    [
        (TerrainClass.CONCRETE, (0.0, 0.02), (0.005, 0.005), (0.002, 0.006)),
        (TerrainClass.GRASS, (0.05, 0.15), (0.01, 0.01), (0.001, 0.003)),
        (TerrainClass.SAND, (0.15, 0.4), (0.02, 0.04), (0.0, 0.0)),
    ],
)
def test_surface_props_ranges_per_class(
    cls: TerrainClass,
    viscous: tuple[float, float],
    timeconst: tuple[float, float],
    roughness: tuple[float, float],
) -> None:
    rng = np.random.default_rng(0)
    samples = [terrain.surface_props(cls, rng) for _ in range(200)]
    for name, (low, high) in [
        ("viscous", viscous), ("solref_timeconst", timeconst), ("roughness", roughness)
    ]:  # fmt: skip
        values = [getattr(s, name) for s in samples]
        assert low <= min(values) and max(values) <= high
        if high > low:
            assert max(values) - min(values) > 0.8 * (high - low)  # the range is used
    forward = [s.friction.mu_forward for s in samples]
    low, high = terrain._RANGES[cls][0]
    assert low <= min(forward) and max(forward) <= high


@pytest.mark.parametrize("cls", list(TerrainClass))
def test_nominal_surface_is_inside_the_range(cls: TerrainClass) -> None:
    nominal = terrain.nominal_surface(cls)
    (f_low, f_high), (r_low, r_high) = terrain._RANGES[cls]
    assert f_low < nominal.friction.mu_forward < f_high
    assert r_low < nominal.friction.mu_lateral / nominal.friction.mu_forward < r_high
    rng = np.random.default_rng(1)
    samples = [terrain.surface_props(cls, rng) for _ in range(50)]
    for name in ("viscous", "solref_timeconst", "roughness"):
        values = [getattr(s, name) for s in samples]
        assert min(values) <= getattr(nominal, name) <= max(values)


def test_surface_table_arrays_are_indexed_by_class() -> None:
    table = terrain.SurfaceTable(
        (
            terrain.nominal_surface(TerrainClass.CONCRETE),
            terrain.nominal_surface(TerrainClass.GRASS),
            terrain.nominal_surface(TerrainClass.SAND),
        )
    )
    mu, viscous, timeconst = table.arrays()
    assert mu.shape == (3, 2) and viscous.shape == (3,) and timeconst.shape == (3,)
    assert mu[2, 0] == pytest.approx(0.6) and mu[0, 0] == pytest.approx(0.3)
    assert viscous[0] < viscous[1] < viscous[2]
    assert timeconst[0] < timeconst[1] < timeconst[2]


def test_sand_is_smooth_and_concrete_is_the_roughest() -> None:
    assert terrain.ROUGHNESS[TerrainClass.SAND] == 0.0
    assert terrain.ROUGHNESS[TerrainClass.CONCRETE] > terrain.ROUGHNESS[TerrainClass.GRASS] > 0.0

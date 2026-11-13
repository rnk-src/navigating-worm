import numpy as np

from worm.randomize import DomainRandomizer, PhysicsParams
from worm.terrain import TerrainClass


def samples(level: float, cls: TerrainClass | None = None, n: int = 200) -> list[PhysicsParams]:
    rng = np.random.default_rng(0)
    randomizer = DomainRandomizer(level)
    return [randomizer.sample(rng, cls) for _ in range(n)]


def test_level_zero_always_returns_nominal_physics() -> None:
    assert all(p == PhysicsParams() for p in samples(0.0, n=20))
    assert PhysicsParams().friction.mu_lateral / PhysicsParams().friction.mu_forward == 4.0


def test_full_level_keeps_lateral_friction_2_to_6_times_forward() -> None:
    ratios = [p.friction.mu_lateral / p.friction.mu_forward for p in samples(1.0)]
    assert 2.0 <= min(ratios) < 2.5 and 5.5 < max(ratios) <= 6.0


def test_full_level_spans_the_wide_forward_friction_range() -> None:
    forward = [p.friction.mu_forward for p in samples(1.0)]
    assert 0.15 <= min(forward) < 0.2 and 0.75 < max(forward) <= 0.8


def test_half_level_halves_the_spread_around_nominal() -> None:
    # Nominal mu_forward is 0.3, so the range (0.15, 0.8) shrinks to (0.225, 0.55).
    forward = [p.friction.mu_forward for p in samples(0.5)]
    assert 0.225 <= min(forward) and max(forward) <= 0.55


def test_servo_delay_is_one_to_three_steps_at_full_level() -> None:
    assert {p.servo.delay_steps for p in samples(1.0)} == {1, 2, 3}


def test_sand_has_more_forward_friction_than_concrete() -> None:
    def mean(cls: TerrainClass) -> float:
        return float(np.mean([p.friction.mu_forward for p in samples(1.0, cls)]))

    assert mean(TerrainClass.SAND) > mean(TerrainClass.GRASS) > mean(TerrainClass.CONCRETE)


def test_full_level_varies_servo_dynamics_and_mass_within_bounds() -> None:
    drawn = samples(1.0)
    bounds = {
        "lag_tau": (0.01, 0.06),
        "deadband": (0.0, 0.02),
        "speed_scale": (0.7, 1.1),
        "strength": (0.7, 1.1),
    }
    for name, (low, high) in bounds.items():
        values = [getattr(p.servo, name) for p in drawn]
        assert low <= min(values) < max(values) <= high, name
    masses = [p.mass_scale for p in drawn]
    assert 0.8 <= min(masses) < max(masses) <= 1.3

import hashlib

import mujoco
import numpy as np
import pytest

from worm import model
from worm import world as wd
from worm.spec import WormSpec


def test_model_has_one_servo_per_joint() -> None:
    m = model.build(WormSpec())
    assert (m.nu, m.nq, m.nv) == (4, 11, 10)  # free joint (7/6) + 4 hinges


def test_worm_rests_flat_on_the_floor() -> None:
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)
    ids = model.segment_body_ids(m, spec)
    assert np.allclose(d.xpos[ids, 2], 0.0125, atol=0.002)  # half the 25 mm height


def test_head_faces_plus_x_and_segments_trail_behind() -> None:
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    xs = d.xpos[model.segment_body_ids(m, spec), 0]
    assert np.allclose(xs, [0.0, -0.06, -0.12, -0.18, -0.24])


def test_servo_holds_a_commanded_angle_within_torque_limit() -> None:
    spec = WormSpec()
    m = model.build(spec)
    d = mujoco.MjData(m)
    d.ctrl[:] = [0.5, 0.0, 0.0, 0.0]
    for _ in range(500):
        mujoco.mj_step(m, d)
        assert np.all(np.abs(d.actuator_force) <= 0.225 + 1e-9)
    assert abs(d.qpos[7] - 0.5) < 0.05


# --- heightfield worlds ---


def raised_block() -> wd.World:
    """Return a 2 m world with a 10 cm high plateau in the -x, +y quarter."""
    base = wd.flat((2.0, 2.0))
    heights = base.heights.copy()
    heights[28:37, 4:13] = 0.1  # rows 28..36 are y = 0.4..0.8; columns 4..12 are x = -0.8..-0.4
    return wd.World(heights, base.surface, base.obstacle, base.cell)


def ground_height(m: mujoco.MjModel, xy: tuple[float, float]) -> float:
    """Return the terrain height under `xy`, measured with a ray from 1 m above."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    geom = np.zeros(1, dtype=np.int32)
    start = np.array([xy[0], xy[1], 1.0])
    distance = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), None, 1, -1, geom)
    assert m.geom(int(geom[0])).name == model.FLOOR
    return 1.0 - float(distance)


def test_build_without_world_is_unchanged() -> None:
    # Characterization: the digest was recorded before the hfield option was added.
    digest = hashlib.sha256(model.to_xml(WormSpec()).encode()).hexdigest()
    assert digest == "602cf367b4fa9a6b69d00c1065b6ad9ec976e38ba3c1beca592e12c2fa51a247"


def test_build_with_world_has_one_hfield_matching_the_world_shape() -> None:
    world = wd.flat((2.0, 1.0))
    m = model.build(WormSpec(), world)
    assert m.nhfield == 1
    assert (m.hfield_nrow[0], m.hfield_ncol[0]) == world.shape == (21, 41)
    assert m.hfield_size[0, :2] == pytest.approx([1.0, 0.5])
    assert m.geom(model.FLOOR).type == mujoco.mjtGeom.mjGEOM_HFIELD


def test_hfield_rows_run_along_plus_y() -> None:
    world = raised_block()
    m = model.build(WormSpec(), world)
    centre = world.to_xy((32, 8))
    assert centre == pytest.approx((-0.6, 0.6))
    assert ground_height(m, (-0.6, 0.6)) == pytest.approx(0.1, abs=1e-4)
    for elsewhere in [(0.6, -0.6), (0.6, 0.6), (-0.6, -0.6)]:
        assert ground_height(m, elsewhere) == pytest.approx(0.0, abs=1e-4)


def test_hfield_height_matches_the_world_between_cells() -> None:
    world = wd.generate(wd.WorldConfig(size=(2.0, 2.0), n_rocks=0, n_ditches=0), 0)
    m = model.build(WormSpec(), world)
    for xy in [(0.31, -0.62), (-0.77, 0.13), (0.05, 0.9)]:
        assert ground_height(m, xy) == pytest.approx(float(world.height_at(xy)), abs=2e-3)


def test_a_segment_dropped_on_the_plateau_rests_on_it() -> None:
    spec = WormSpec(n_segments=1)
    m = model.build(spec, raised_block())
    for xy, expected in [((-0.6, 0.6), 0.1125), ((0.6, -0.6), 0.0125)]:
        d = mujoco.MjData(m)
        d.qpos[:3] = (xy[0], xy[1], 0.15)
        for _ in range(500):
            mujoco.mj_step(m, d)
        assert d.qpos[2] == pytest.approx(expected, abs=0.002)


def test_worm_rests_on_a_flat_hfield_at_the_same_height_as_on_the_plane() -> None:
    spec = WormSpec()
    m = model.build(spec, wd.flat((2.0, 2.0)))
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.allclose(d.xpos[model.segment_body_ids(m, spec), 2], 0.0125, atol=0.002)


def test_load_world_overwrites_heights_in_place() -> None:
    m = model.build(WormSpec(), wd.flat((2.0, 2.0)))
    assert ground_height(m, (-0.6, 0.6)) == pytest.approx(0.0, abs=1e-4)
    model.load_world(m, raised_block())
    assert ground_height(m, (-0.6, 0.6)) == pytest.approx(0.1, abs=1e-4)


def test_load_world_rejects_another_shape_and_a_plane_model() -> None:
    m = model.build(WormSpec(), wd.flat((2.0, 2.0)))
    with pytest.raises(ValueError, match="shape"):
        model.load_world(m, wd.flat((1.0, 2.0)))
    with pytest.raises(ValueError, match="hfield"):
        model.load_world(model.build(WormSpec()), wd.flat((2.0, 2.0)))


def test_pair_ids_follow_the_segment_order() -> None:
    spec = WormSpec()
    m = model.build(spec)
    ids = model.pair_ids(m, spec)
    assert [m.geom(int(m.pair_geom2[i])).name for i in ids] == [f"seg{i}" for i in range(5)]

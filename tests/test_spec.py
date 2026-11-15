import pytest

from worm.spec import LATEST_OBS_VERSION, WormSpec, action_spec, obs_layout


def test_observation_slices_are_contiguous_and_total_24() -> None:
    layout = obs_layout(WormSpec())
    names = list(layout.slices)
    assert names == [
        "joint_pos",
        "joint_delta",
        "yaw_rate",
        "roll_pitch",
        "bearing",
        "distance",
        "prev_action",
        "terrain",
    ]
    bounds = [(s.start, s.stop) for s in layout.slices.values()]
    assert bounds == [(0, 4), (4, 8), (8, 9), (9, 11), (11, 13), (13, 14), (14, 21), (21, 24)]
    assert layout.dim == 24


def test_layout_follows_the_segment_count() -> None:
    spec = WormSpec(n_segments=7)
    assert action_spec(spec).dim == 9  # 3 CPG parameters + 6 joints
    assert obs_layout(spec).dim == 6 + 6 + 1 + 2 + 2 + 1 + 9 + 3


def test_control_runs_at_20_hz_with_whole_substeps() -> None:
    spec = WormSpec()
    assert spec.substeps * spec.physics_dt == 0.05


def test_obs_v2_appends_terrain_ahead_and_height_ahead_and_totals_31() -> None:
    v1, v2 = obs_layout(WormSpec()), obs_layout(WormSpec(), version=2)
    assert list(v2.slices)[: len(v1.slices)] == list(v1.slices)
    assert all(v2.slices[name] == v1.slices[name] for name in v1.slices)
    assert v2.slices["terrain_ahead"] == slice(24, 27)
    assert v2.slices["height_ahead"] == slice(27, 31)
    assert v2.dim == 31 and v2.version == 2 and v1.version == 1
    assert LATEST_OBS_VERSION == 2


def test_unknown_obs_version_raises() -> None:
    with pytest.raises(ValueError, match="observation version"):
        obs_layout(WormSpec(), version=3)

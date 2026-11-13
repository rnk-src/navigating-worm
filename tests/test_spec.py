from worm.spec import WormSpec, action_spec, obs_layout


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

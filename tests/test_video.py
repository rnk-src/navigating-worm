import numpy as np

from worm.video import FrameRecorder


class CountingCamera:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> np.ndarray:
        self.calls += 1
        return np.full((2, 2, 3), self.calls, dtype=np.uint8)


def test_recorder_keeps_one_frame_per_call() -> None:
    camera = CountingCamera()
    recorder = FrameRecorder(camera)
    for _ in range(5):
        recorder()
    assert [int(f[0, 0, 0]) for f in recorder.frames] == [1, 2, 3, 4, 5]


def test_recorder_stops_rendering_once_the_cap_is_reached() -> None:
    camera = CountingCamera()
    recorder = FrameRecorder(camera, max_frames=3)
    for _ in range(50):
        recorder()
    assert len(recorder.frames) == 3
    assert camera.calls == 3  # no renders are thrown away


def test_recorder_keeps_every_nth_step_and_renders_only_those() -> None:
    camera = CountingCamera()
    recorder = FrameRecorder(camera, every=4)
    for _ in range(10):
        recorder()
    assert len(recorder.frames) == 2 and camera.calls == 2  # steps 4 and 8


def test_recorder_skips_a_camera_that_returns_nothing() -> None:
    recorder = FrameRecorder(lambda: None)
    recorder()
    assert recorder.frames == []

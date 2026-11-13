"""Collect rendered frames during a rollout."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class FrameRecorder:
    """Call once per control step. Keeps every `every`-th frame, up to `max_frames`.

    It renders only the frames that it keeps.
    """

    def __init__(
        self, render: Callable[[], Frame | None], max_frames: int | None = None, every: int = 1
    ) -> None:
        self._render = render
        self._max_frames = max_frames
        self._every = every
        self._steps = 0
        self.frames: list[Frame] = []

    def __call__(self) -> None:
        self._steps += 1
        if self._steps % self._every:
            return
        if self._max_frames is not None and len(self.frames) >= self._max_frames:
            return
        frame = self._render()
        if frame is not None:
            self.frames.append(frame)

import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

from worm.world import World

_PROBE = """
from worm.env import WormGoalEnv
env = WormGoalEnv(render_mode="rgb_array")
env.reset(seed=0)
assert env.render() is not None
"""


@pytest.fixture(scope="session")
def rendering() -> None:
    """Skip the test when this machine cannot render.

    A missing GL backend aborts the interpreter, so the probe runs in a subprocess.
    """
    probe = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, timeout=120)
    if probe.returncode != 0:
        pytest.skip("rendering unavailable on this machine")


@pytest.fixture
def make_world() -> Callable[..., World]:
    """Return a builder of hand-made worlds: `make_world(heights, surface=None, obstacle=None)`."""

    def build(
        heights: np.ndarray,
        surface: np.ndarray | None = None,
        obstacle: np.ndarray | None = None,
        cell: float = 0.05,
    ) -> World:
        shape = heights.shape
        return World(
            heights=heights.astype(np.float32),
            surface=np.zeros(shape, np.uint8) if surface is None else surface.astype(np.uint8),
            obstacle=np.zeros(shape, np.bool_) if obstacle is None else obstacle.astype(np.bool_),
            cell=cell,
        )

    return build

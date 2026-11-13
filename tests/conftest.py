import subprocess
import sys

import pytest

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

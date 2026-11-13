"""Open-loop serpenoid gait. Baseline, test fixture and later sysID excitation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def serpenoid(phase: float, amp: float, wave: float, bias: float, n: int) -> NDArray[np.float64]:
    """Return `n` joint targets (rad). Joint 0 is nearest the head.

    The wave travels from head to tail as `phase` grows, which pushes the body forward.
    A positive `bias` bends the body so that it turns left (counter-clockwise).
    """
    index = np.arange(n, dtype=np.float64)
    return amp * np.sin(phase - index * wave) - bias

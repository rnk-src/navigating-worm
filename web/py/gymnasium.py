"""Just enough of gymnasium for `worm.env.WormGoalEnv` to run in the browser.

The environment only needs the base class, a seeded `np_random` and the `Box` space.
The real package is not needed to drive a worm.
"""

from __future__ import annotations

import types

import numpy as np


class Env:
    _np_random: np.random.Generator | None = None

    def __class_getitem__(cls, _item):  # type: ignore[no-untyped-def]
        return cls

    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[no-untyped-def]
        if seed is not None or self._np_random is None:
            self._np_random = np.random.default_rng(seed)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random


class Box:
    def __init__(self, low, high, shape, dtype=np.float32) -> None:  # type: ignore[no-untyped-def]
        self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype


spaces = types.SimpleNamespace(Box=Box)

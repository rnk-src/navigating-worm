"""A trained policy as plain numpy, so the browser demo needs no torch.

`export` reads a policy loaded by `worm.training.load_policy` and returns a JSON-able dict.
`NumpyPolicy` runs that dict. Only `export` needs torch, and it takes the loaded policy as an
argument, so this module imports numpy alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FORMAT = 1


class NumpyPolicy:
    """A deterministic `obs -> action` function: normalise, tanh layers, linear head, clip."""

    def __init__(self, data: dict[str, Any]) -> None:
        if data.get("format") != FORMAT:
            raise ValueError(f"policy format {data.get('format')} is not supported")
        self.obs_version = int(data["obs_version"])
        self._mean = np.asarray(data["obs_mean"], dtype=np.float64)
        self._std = np.sqrt(np.asarray(data["obs_var"], dtype=np.float64) + float(data["epsilon"]))
        self._clip = float(data["clip_obs"])
        self._hidden = [
            (np.asarray(layer["w"], dtype=np.float64), np.asarray(layer["b"], dtype=np.float64))
            for layer in data["hidden"]
        ]
        self._head = (
            np.asarray(data["head"]["w"], dtype=np.float64),
            np.asarray(data["head"]["b"], dtype=np.float64),
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> NumpyPolicy:
        return NumpyPolicy(data)

    def __call__(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        x = np.clip((np.asarray(obs, dtype=np.float64) - self._mean) / self._std, -self._clip, self._clip)
        for weight, bias in self._hidden:
            x = np.tanh(weight @ x + bias)
        weight, bias = self._head
        return np.clip(weight @ x + bias, -1.0, 1.0).astype(np.float32)


def load(path: Path | str) -> NumpyPolicy:
    """Read a policy written by `export`."""
    return NumpyPolicy(json.loads(Path(path).read_text()))


def export(loaded: Any) -> dict[str, Any]:
    """Return the weights of a `LoadedPolicy` (a tanh MLP with a linear head) as plain lists."""
    net = loaded.model.policy

    def layer(module: Any) -> dict[str, list[Any]]:
        return {
            "w": module.weight.detach().cpu().numpy().astype(np.float64).tolist(),
            "b": module.bias.detach().cpu().numpy().astype(np.float64).tolist(),
        }

    linear = [m for m in net.mlp_extractor.policy_net if hasattr(m, "weight")]
    norm = loaded.norm
    return {
        "format": FORMAT,
        "obs_version": int(loaded.meta["obs_version"]),
        "obs_mean": norm.obs_rms.mean.astype(np.float64).tolist(),
        "obs_var": norm.obs_rms.var.astype(np.float64).tolist(),
        "epsilon": float(norm.epsilon),
        "clip_obs": float(norm.clip_obs),
        "hidden": [layer(m) for m in linear],
        "head": layer(net.action_net),
    }

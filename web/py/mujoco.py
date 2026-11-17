"""A `mujoco` module for Pyodide, backed by MuJoCo compiled to WebAssembly.

The worm's Python code runs unchanged on top of it. The module keeps a numpy mirror of each
model field the code touches. Python edits the mirror in place. The mirror goes to JavaScript
when a `mj_*` call needs it, and the results come back after the call. The JavaScript `Sim`
object (web/src/physics.ts) owns the real model and data.
"""

from __future__ import annotations

import numpy as np
from pyodide.ffi import to_js

_make = None  # JavaScript function: xml string -> Sim. Set by `_bind`.


def _bind(make) -> None:  # type: ignore[no-untyped-def]
    global _make
    _make = make


# Model fields the worm code uses: name -> (columns, dtype). One column means a 1-D array.
_FIELDS = {
    "body_mass": (1, np.float64),
    "body_inertia": (3, np.float64),
    "body_rootid": (1, np.int32),
    "pair_geom2": (1, np.int32),
    "pair_friction": (5, np.float64),
    "pair_solref": (2, np.float64),
    "hfield_data": (1, np.float32),
    "hfield_nrow": (1, np.int32),
    "hfield_ncol": (1, np.int32),
    "actuator_forcerange": (2, np.float64),
    "actuator_gainprm": (10, np.float64),
    "actuator_biasprm": (10, np.float64),
    "site_pos": (3, np.float64),
}
# Fields that never change the physics after the model is built, so a push skips them.
_STATIC = {"hfield_nrow", "hfield_ncol", "body_rootid", "pair_geom2"}


class _Named:
    def __init__(self, id: int, pos: np.ndarray | None = None) -> None:  # noqa: A002
        self.id = id
        self.pos = pos


class MjModel:
    def __init__(self, handle) -> None:  # type: ignore[no-untyped-def]
        self._h = handle
        self._cache: dict[str, np.ndarray] = {}
        self._dims = {k: int(v) for k, v in handle.dims().to_py().items()}
        self._dummy: dict[str, np.ndarray] = {}

    @staticmethod
    def from_xml_string(xml: str) -> MjModel:
        if _make is None:
            raise RuntimeError("the mujoco module is not bound to the physics engine yet")
        return MjModel(_make(xml))

    @property
    def nhfield(self) -> int:
        return self._dims["nhfield"]

    def _array(self, name: str) -> np.ndarray:
        if name not in self._cache:
            cols, dtype = _FIELDS[name]
            raw = np.array(self._h.modelGet(name).to_py(), dtype=dtype)
            self._cache[name] = raw if cols == 1 else raw.reshape(-1, cols)
        return self._cache[name]

    def __getattr__(self, name: str) -> np.ndarray:
        if name in _FIELDS:
            return self._array(name)
        # The hfield texture only colours the 3D view, which the browser draws itself.
        if name == "tex_adr":
            return self._dummy.setdefault(name, np.zeros(1, dtype=np.int32))
        if name == "tex_data":
            cells = int(self._array("hfield_nrow")[0] * self._array("hfield_ncol")[0])
            return self._dummy.setdefault(name, np.zeros(3 * cells, dtype=np.uint8))
        raise AttributeError(name)

    def body(self, name: str) -> _Named:
        return _Named(int(self._h.bodyId(name)))

    def geom(self, name: str) -> _Named:
        return _Named(int(self._h.geomId(name)))

    def site(self, name: str) -> _Named:
        index = int(self._h.siteId(name))
        return _Named(index, self._array("site_pos")[index])

    def texture(self, name: str) -> _Named:
        return _Named(0)

    def _push(self, names: set[str] | None = None, full: bool = False) -> None:
        for name, array in self._cache.items():
            if names is not None and name not in names:
                continue
            if names is None and not full and (name == "hfield_data" or name in _STATIC):
                continue
            self._h.modelSet(name, to_js(np.ascontiguousarray(array).ravel()))

    def __del__(self) -> None:
        try:
            self._h.free()
        except Exception:  # noqa: BLE001 - best effort at interpreter shutdown
            pass


class MjData:
    def __init__(self, model: MjModel) -> None:
        self._h = model._h
        d = model._dims
        nq, nv, nu, nb = d["nq"], d["nv"], d["nu"], d["nbody"]
        self._sizes = [
            ("qpos", (nq,)),
            ("qvel", (nv,)),
            ("ctrl", (nu,)),
            ("xpos", (nb, 3)),
            ("xmat", (nb, 9)),
            ("cvel", (nb, 6)),
            ("subtree_com", (nb, 3)),
        ]
        for name, shape in self._sizes:
            setattr(self, name, np.zeros(shape))
        self.xfrc_applied = np.zeros((nb, 6))
        self._pull()

    def _push(self) -> None:
        buf = np.concatenate([self.qpos, self.qvel, self.ctrl, self.xfrc_applied.ravel()])
        self._h.push(to_js(buf))

    def _pull(self) -> None:
        buf = np.array(self._h.pull().to_py(), dtype=np.float64)
        at = 0
        for name, shape in self._sizes:
            size = int(np.prod(shape))
            getattr(self, name)[...] = buf[at : at + size].reshape(shape)
            at += size


def mj_resetData(model: MjModel, data: MjData) -> None:
    model._h.reset()
    data._pull()


def mj_setConst(model: MjModel, data: MjData) -> None:
    model._push(full=True)
    data._push()
    model._h.setConst()
    data._pull()


def mj_forward(model: MjModel, data: MjData) -> None:
    model._push()
    data._push()
    model._h.forward()
    data._pull()


def mj_kinematics(model: MjModel, data: MjData) -> None:
    model._push()
    data._push()
    model._h.kinematics()
    data._pull()


def mj_step1(model: MjModel, data: MjData) -> None:
    data._push()
    model._h.step1()
    data._pull()


def mj_step2(model: MjModel, data: MjData) -> None:
    # Python may have changed the contact pair friction and softness between the two halves.
    model._push({"pair_friction", "pair_solref"})
    data._push()
    model._h.step2()
    data._pull()

"""Gymnasium environment: drive the worm head to a target point."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from numpy.typing import NDArray

from worm import friction, gait, model
from worm.randomize import DomainRandomizer, PhysicsParams
from worm.reward import StepState, reward
from worm.sensors import SensorSuite, head_pose
from worm.servo import ServoModel
from worm.spec import WormSpec, action_spec, obs_layout
from worm.terrain import TerrainClass
from worm.world import World

Array = NDArray[np.float64]
_FLIP_ANGLE = math.radians(60.0)
_EDGE_MARGIN = 0.3  # m, an episode in a world ends when the head is this close to the edge
_SPAWN_CLEARANCE = 0.003  # m, drop height above the highest ground under the body
_SETTLE_STEPS = 25  # physics steps after a spawn in a world
_TARGET_TRIES = 20

# Returns the world of one episode. Every world must have the same shape.
WorldSampler = Callable[[np.random.Generator], World]


class WormGoalEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Reach a target within `tolerance`. The action is 3 CPG parameters plus joint residuals.

    `reset(options={"target": (x, y)})` injects a target. `set_target` changes the
    target mid-episode without moving the body, which route following uses.
    More reset options: `"start": (x, y)` and `"yaw": rad` place the head, and
    `"world": World` replaces the world with another of the same shape.

    Without `world` the ground is a flat plane with one terrain class per episode.
    With a `World`, or a `WorldSampler` that returns a new world per reset, the ground
    is a heightfield. Each segment then takes its surface from the cell under it, and
    the terrain one-hot is the class under the head.
    """

    def __init__(
        self,
        spec: WormSpec | None = None,
        randomizer: DomainRandomizer | None = None,
        terrain: TerrainClass | None = None,
        episode_seconds: float = 30.0,
        tolerance: float = 0.15,
        target_range: tuple[float, float] = (1.0, 3.0),
        render_mode: str | None = None,
        world: World | WorldSampler | None = None,
    ) -> None:
        self.spec_ = spec or WormSpec()
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": round(1.0 / self.spec_.control_dt),
        }
        self.actions = action_spec(self.spec_)
        self.layout = obs_layout(self.spec_)
        self.randomizer = randomizer or DomainRandomizer()
        self.terrain = terrain
        self.max_steps = round(episode_seconds / self.spec_.control_dt)
        self.tolerance = tolerance
        self.target_range = target_range
        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None
        self._overview: tuple[tuple[float, float], float] | None = None
        # Extra spheres to draw: (xy, rgba, radius). Rendering only.
        self.markers: list[tuple[tuple[float, float], tuple[float, float, float, float], float]]
        self.markers = []

        self._sampler: WorldSampler | None = None
        self._world: World | None = None
        if isinstance(world, World):
            self._world = world
        elif world is not None:
            self._sampler = world
            self._world = world(np.random.default_rng(0))  # fixes the compiled shape
        self.model = model.build(self.spec_, self._world)
        self.data = mujoco.MjData(self.model)
        self._nominal_mass = self.model.body_mass.copy()
        self._nominal_inertia = self.model.body_inertia.copy()
        self._head = self.model.body("seg0").id
        body_ids = model.segment_body_ids(self.model, self.spec_)
        self._body_ids = np.asarray(body_ids)
        self._friction: friction.ResistiveForce | friction.TerrainResistive
        if self._world is None:
            self._friction = friction.ResistiveForce(body_ids)
        else:
            self._friction = friction.TerrainResistive(
                body_ids, model.pair_ids(self.model, self.spec_), self._world, hybrid=True
            )
        self._servo = ServoModel(self.spec_)
        self._sensors = SensorSuite(self.model, self.spec_, self.layout)

        self.observation_space = gym.spaces.Box(-10.0, 10.0, (self.layout.dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.actions.dim,), np.float32)

        self.params = PhysicsParams()
        self._target = np.zeros(2)
        self._phase = 0.0
        self._steps = 0
        self._prev_action = np.zeros(self.actions.dim)
        self._prev_joint_pos = np.zeros(self.spec_.n_joints)
        self._joint_delta = np.zeros(self.spec_.n_joints)
        self._state = StepState(0.0, 0.0)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        self.params = self.randomizer.sample(rng, self.terrain)
        if self._world is not None:
            if "world" in options:
                self._set_world(options["world"])
            elif self._sampler is not None:
                self._set_world(self._sampler(rng))
            self.params = replace(self.params, surfaces=self.randomizer.sample_surfaces(rng))
        self._apply_params(self.params)

        mujoco.mj_resetData(self.model, self.data)
        yaw = rng.uniform(-math.pi, math.pi)
        if "yaw" in options:
            yaw = float(options["yaw"])
        start = np.asarray(options.get("start", (0.0, 0.0)), dtype=np.float64)
        self.data.qpos[:2] = start
        self.data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)
        if self._world is not None:
            self._drop_onto_terrain()

        self._phase = float(rng.uniform(0.0, 2.0 * math.pi))
        self._prev_action = np.zeros(self.actions.dim)
        self._prev_joint_pos = np.zeros(self.spec_.n_joints)
        self._joint_delta = np.zeros(self.spec_.n_joints)
        if "target" in options:
            target = np.asarray(options["target"], dtype=np.float64)
        else:
            target = self._sample_target(start)
        return self.set_target(target), self._info({})

    def _sample_target(self, start: Array) -> Array:
        """Draw a target at `target_range` from the start. In a world, keep it reachable."""
        rng, world = self.np_random, self._world
        target = start
        for _ in range(_TARGET_TRIES if world is not None else 1):
            angle = rng.uniform(-math.pi, math.pi)
            offset = rng.uniform(*self.target_range) * np.array([math.cos(angle), math.sin(angle)])
            target = start + offset
            if world is None:
                break
            row, col = world.to_cell(target)
            if world.contains(target) and not world.obstacle[row, col]:
                break
        else:
            assert world is not None
            xmin, xmax, ymin, ymax = world.extent
            target = np.clip(target, [xmin, ymin], [xmax, ymax])
        return target

    def _set_world(self, world: World) -> None:
        model.load_world(self.model, world)
        assert isinstance(self._friction, friction.TerrainResistive)
        self._friction.set_world(world)
        self._world = world
        if self._renderer is not None:  # the GPU copy of the hfield is stale
            self._renderer.close()
            self._renderer = None

    def _drop_onto_terrain(self) -> None:
        """Lift the level body just clear of the ground under it, then let it settle."""
        assert self._world is not None
        ground = self._world.height_at(self.data.xpos[self._body_ids, :2])
        self.data.qpos[2] = float(ground.max()) + 0.5 * self.spec_.seg_height + _SPAWN_CLEARANCE
        mujoco.mj_forward(self.model, self.data)
        for _ in range(_SETTLE_STEPS):
            friction.step(self.model, self.data, self._friction)

    @property
    def world(self) -> World | None:
        return self._world

    @property
    def surface_under_head(self) -> TerrainClass:
        """Return the terrain class that the head is on."""
        if self._world is None:
            return self.params.terrain
        return TerrainClass(int(self._world.class_at(self.head_xy)))

    @property
    def target(self) -> Array:
        return self._target.copy()

    @property
    def head_xy(self) -> Array:
        xy: Array = self.data.xpos[self._head, :2].copy()
        return xy

    def set_target(self, target: Array) -> NDArray[np.float32]:
        """Move the target and restart the episode clock. The body keeps its state.

        Return the observation of the new target.
        """
        self._target = np.asarray(target, dtype=np.float64).copy()
        self.model.site("target").pos[:2] = self._target
        if self._world is not None:
            self.model.site("target").pos[2] = float(self._world.height_at(self._target)) + 0.01
        mujoco.mj_kinematics(self.model, self.data)  # refresh the target marker position
        self._steps = 0
        self._state = self._step_state()
        return self._observe(advance=False)

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        act = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        a, spec = self.actions, self.spec_
        amp = 0.5 * (act[0] + 1.0) * a.amp_max
        freq = a.freq_min + 0.5 * (act[1] + 1.0) * (a.freq_max - a.freq_min)
        self._phase = (self._phase + 2.0 * math.pi * freq * spec.control_dt) % (2.0 * math.pi)
        targets = gait.serpenoid(self._phase, amp, a.wave, act[2] * a.bias_max, spec.n_joints)
        targets += act[3:] * a.residual_max

        start = self.data.ctrl.copy()
        applied = self._servo.command(targets)
        for k in range(spec.substeps):
            self.data.ctrl[:] = start + (applied - start) * (k + 1) / spec.substeps
            friction.step(self.model, self.data, self._friction)

        prev, self._state = self._state, self._step_state()
        value, terms = reward(prev, self._state, act, self._prev_action)
        self._prev_action = act
        self._steps += 1
        terminated = self._state.reached or self._state.flipped
        outside = self._out_of_world() and not terminated
        truncated = (self._steps >= self.max_steps or outside) and not terminated
        return self._observe(), value, terminated, truncated, self._info(terms, outside)

    def _out_of_world(self) -> bool:
        if self._world is None:
            return False
        rx, ry = self._world.half_extent
        x, y = self.head_xy
        return bool(abs(x) > rx - _EDGE_MARGIN or abs(y) > ry - _EDGE_MARGIN)

    def set_overview(self, center: tuple[float, float] | None, half_extent: float = 1.0) -> None:
        """Render from a fixed top-down camera that frames `center` +- `half_extent` metres.

        `center=None` returns to the camera that follows the head.
        """
        self._overview = None if center is None else (center, half_extent)

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        if self._overview is None:
            camera.lookat[:2] = self.data.xpos[self._head, :2]
            if self._world is not None:
                camera.lookat[2] = self.data.xpos[self._head, 2]
            camera.distance, camera.elevation, camera.azimuth = 1.5, -60.0, 90.0
        else:
            center, half_extent = self._overview
            camera.lookat[:2] = center
            # The vertical field of view is 45 degrees. Leave a 20 % margin.
            camera.distance = 1.2 * half_extent / math.tan(math.radians(22.5))
            camera.elevation, camera.azimuth = -90.0, 90.0
        self._renderer.update_scene(self.data, camera)
        scene = self._renderer.scene
        for xy, rgba, radius in self.markers[: scene.maxgeom - scene.ngeom]:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, radius),
                np.array([xy[0], xy[1], radius + self._ground(xy)]),
                np.eye(3).flatten(),
                np.asarray(rgba, dtype=np.float32),
            )
            scene.ngeom += 1
        frame: NDArray[np.uint8] = self._renderer.render()
        return frame

    def _ground(self, xy: tuple[float, float]) -> float:
        return 0.0 if self._world is None else float(self._world.height_at(xy))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _apply_params(self, params: PhysicsParams) -> None:
        m, spec = self.model, self.spec_
        m.body_mass[:] = self._nominal_mass * params.mass_scale
        m.body_inertia[:] = self._nominal_inertia * params.mass_scale
        mujoco.mj_setConst(m, self.data)
        torque = spec.servo_torque * params.servo.strength
        m.actuator_forcerange[:] = (-torque, torque)
        m.actuator_gainprm[:, 0] = spec.servo_kp * params.servo.strength
        m.actuator_biasprm[:, 1] = -spec.servo_kp * params.servo.strength
        if isinstance(self._friction, friction.TerrainResistive) and params.surfaces is not None:
            self._friction.configure_table(m, params.surfaces)
        else:
            self._friction.configure(m, params.friction)
        self._servo.reset(params.servo)
        self._sensors.reset(params.sensor, self.np_random)

    def _step_state(self) -> StepState:
        xy, yaw, roll, pitch = head_pose(self.data, self._head)
        delta = self._target - xy
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(delta[1], delta[0]) - yaw
        flipped = abs(roll) > _FLIP_ANGLE or abs(pitch) > _FLIP_ANGLE
        return StepState(distance, bearing, distance < self.tolerance, flipped)

    def _observe(self, advance: bool = True) -> NDArray[np.float32]:
        """Build the observation. `advance=False` keeps the joint-motion reference unchanged."""
        obs = np.zeros(self.layout.dim, dtype=np.float32)
        slices = self.layout.slices
        for name, value in self._sensors.read(self.data, self._target, self.np_random).items():
            obs[slices[name]] = value
        if advance:
            joint_pos = obs[slices["joint_pos"]].astype(np.float64)
            self._joint_delta = joint_pos - self._prev_joint_pos
            self._prev_joint_pos = joint_pos
        obs[slices["joint_delta"]] = self._joint_delta
        obs[slices["prev_action"]] = self._prev_action
        obs[slices["terrain"].start + int(self.surface_under_head)] = 1.0
        return np.clip(obs, -10.0, 10.0)

    def _info(self, terms: dict[str, float], out_of_world: bool = False) -> dict[str, Any]:
        return {
            "distance": self._state.distance,
            "reached": self._state.reached,
            "is_success": self._state.reached,
            "head_xy": self.head_xy,
            "reward_terms": terms,
            "out_of_world": out_of_world,
        }

# Plan: terrain world, planner, click-to-go UI, terrain-aware policy

The cloud requirements are folded in: a uv bootstrap script, a parallel-seed launcher with checkpoint and resume, and a `docs/CLOUD.md` runbook. Fable prepares these and smoke-tests them locally. It never provisions anything.

---

## 0. What I found

**Branch state**
- `main` is at `6eaa249` (PR #2 merged). PRs #1 and #2 are both merged.
- HEAD is `fix/mass-terrain-level` at `25e90e2`, "fix: scale inertia with mass and sample the terrain class with its one-hot".
  - It is one commit ahead of `main`. It is **local only**: not pushed, no PR.
  - It changes `worm/env.py`, `worm/randomize.py`, `tests/test_env.py` and `tests/test_randomize.py`.
  - `DomainRandomizer.sample(rng, cls=None)` now draws a `TerrainClass` uniformly. `PhysicsParams.terrain` carries it. `env._observe` writes the one-hot. Inertia now scales with mass.
- The README is stale against this commit:
  - "Known limits" still says inertia is not scaled and the policy always sees "concrete".
  - The v2 results are not reported.

**What already exists of M2**
- Three "v2" policies in `runs/v2_seed{0,1,2}`. `runs/` is git-ignored, so they are uncommitted.
  - They were trained on the fix branch, 3M steps each.
  - Each scores **50/50** on the 100 m route eval (`runs/v2_eval_seed*_50.json`).
  - The M1 policies re-evaluated on the fixed code score 49/50 (`runs/fixed_seed*_50.json`).
  - `runs/v2_seed0/policy.zip` has no `system_info.txt`, so it is safe to commit.
- `results/policy` is still the M1 policy. It was trained with a constant CONCRETE one-hot, so GRASS and SAND inputs are out of distribution for it.
- So M2 so far means "one terrain class per episode, friction only, flat plane".

**Measured training throughput** (12-core laptop CPU, CPU torch, `n_envs=8`, PPO 64x64)

| Run | Steps | Time | fps |
|---|---|---|---|
| M1 `train.log` | 3.0M | 852 s | ~3,520 |
| `train_seed1` / `train_seed2` | 3.0M | 865 / 871 s | ~3,450 |
| `train_v2_seed0` (alone) | 3.0M | 820 s | ~3,660 |
| `train_v2_seed1` and `seed2`, run at the same time | 3.0M each | 1,428 s | ~2,100 each, ~4,200 combined |

Two runs in parallel get about 1.15x the combined throughput of one run. Per env that is about 450 steps/s, or about 22x real time at 20 Hz. Flat routes run at about 0.1 m/s (100 m takes about 850–970 sim-seconds).

**Environment facts**
- mujoco 4.0.1, SB3 2.9.2, torch 2.15, Python 3.11.
- `mjpython` is in `.venv/bin`, and `mujoco.viewer` imports.
- An hfield compiles. The test used `nrow=5 ncol=5 size="1 1 0.2 0.05"` and got `hfield_data` with 25 entries.
- Pillow and websockets are installed as transitive dependencies. scipy, matplotlib and any web framework are not.
- Python 3.11 has `tomllib`, so configs can be TOML with no new dependency.
- CI is `.github/workflows/check.yml` (uv + `make check`). The PR template has sections What / Why / How tested / AI assistance.
- `PROGRESS.md` is already in `.git/info/exclude`.

**Code facts that drive the design**
- `ResistiveForce` already computes drag per segment (`body_ids` vectorized). It flattens the body axes to horizontal and uses load = m·g.
  - Per-cell coefficients are a small change: make `_mu` shape (n, 2) per step instead of (2,).
- `ContactAnisotropic` writes `model.pair_friction`. Every segment has its **own** `<pair>` with the floor.
  - So per-segment contact friction and `pair_solref` (softness) can also be changed each step, before `mj_step1`.
  - This is how sinkage on sand and isotropic stiction become possible.
- `run_route` is a closed loop. The UI needs a stepwise follower, so `run_route` will become a thin wrapper.
- `load_policy` builds `DummyVecEnv([WormGoalEnv])` for VecNormalize. It must build an env with the **matching obs version** once the layout changes.
- The hand-written `steer` controller exists only in `tests/test_route.py`. It scores 8/10 and is a useful last-resort fallback for the demo.

---

## 1. Order and stop points

**Order: Step 0 housekeeping, then A, then C, then D, then B.** This is the demo path first, as the user proposed. Reasons:
1. The demo works with existing policies. A v1 policy sees the class under its head through the existing one-hot, and slope through `roll_pitch`.
2. B is compute-bound and changes the obs layout. It needs A's world generator, A's per-cell physics and C's planner. B's eval harness ("route success on mixed-terrain maps") is C's navigator. Doing B first would mean building A and C inside B anyway.
3. C1 (the pure planner) needs only the `World` data type from A1. Placing it right after A means every stop point is a shippable, demoable state.
4. B's long training can then be prepared on a stable base and run on the cloud VM while the rest of the work is reviewed.

| Stop | Reached after | What ships |
|---|---|---|
| **SP0** | Step 0 | PR: inertia and terrain-sampling fix, v2 policy committed, README honest |
| **SP1** | A1 + A2 | Seeded procedural worlds plus a PNG map renderer (`scripts/make_world.py`) |
| **SP2** | A3–A5 | The worm drives on hfield worlds with per-cell surfaces using the current policy. Flat world unchanged (golden test) |
| **SP3** | C1–C2 | Headless planner. `scripts/plan_map.py` writes a PNG of map, cost and path, or prints `Infeasible(reason)` |
| **SP4** | C3–C4 | `scripts/navigate.py`: plan, drive, replan, MP4, headless end to end |
| **SP5 (MVP)** | D | Click-to-go web app works with the committed policy or the scripted fallback |
| **SP6** | B1–B5 | Obs v2 plus versioning, curriculum, config-driven training, cloud scripts, smoke runs, eval harness. Long training ready to launch |
| **SP7** | B6 (compute) | Terrain-aware policy trained and evaluated. Per-terrain breakdown in README. Demo defaults to it |

If the budget runs out, any stop point is a valid place to end. Each one has its PR open, `make check` green and `PROGRESS.md` current.

---

## 2. Rules for the executing agent (Fable)

1. **Settled decisions are closed.** Do not re-litigate the "Settled decisions" in `docs/PLAN.md`:
   - MuJoCo + Gymnasium + SB3 PPO; lateral undulation.
   - 7-D CPG + residual action; 20 Hz control.
   - Real-sensor-only observations; one policy conditioned on terrain class.
   - Train on 1–3 m targets; 2 m waypoint spacing; 15 cm tolerance.
   - `ResistiveForce` as the default friction on flat ground.
   - Also keep `N_TERRAIN_CLASSES = 3`. Keep the enum name `CONCRETE`. The UI labels it "stone / concrete".
2. **Decisions made in this plan are recorded.** Fable writes them as short ADRs in `docs/decisions/NNNN-title.md` (Context / Decision / Consequences, 10–30 lines each). A new decision gets a new ADR. Fable updates the Milestones section of `docs/PLAN.md` once, in Step 0. The old "M2" becomes M2a–M2d; M3–M5 hardware stays.
3. **TDD on every task:**
   - Write the test.
   - Run `uv run pytest tests/test_x.py::test_name -q` and **paste that it fails** (in your working notes or the commit body, not in pushed files).
   - Implement, then run it again and watch it pass.
   - Characterization (golden) tests that pass at once are allowed only where this plan marks them as such.
4. **Never claim a task or milestone is done without showing the full `make check` output** (the last lines of all four stages) in your reply or `PROGRESS.md`.
5. **Git workflow:**
   - Small Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`). One logical change per commit, usually a test plus its code.
   - Every commit message ends with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
   - Never commit to `main`. Never merge. Never force-push a branch that has an open PR, except `--force-with-lease` after a rebase onto a merged base.
   - Open PRs with `gh pr create --base <base> --title ... --body-file <tmpfile in scratch dir>`, filling every section of `.github/PULL_REQUEST_TEMPLATE.md`. "AI assistance" says what Claude Fable wrote and what was verified by running.
6. **Branch chaining (unattended):**
   - Before starting milestone X, run `gh pr view <prev-PR> --json state,mergedAt`.
   - If the previous PR is **merged**: `git fetch origin && git switch -c feat/X origin/main`, and use `--base main`.
   - If it is **not merged**: `git switch -c feat/X feat/prev`, and use `--base feat/prev`. The PR body's first line is `Stacked on #N — merge that first.` GitHub retargets to `main` by itself when the base is merged and its branch deleted.
   - If a base PR gets review changes, rebase the stack in order with `git rebase --update-refs`, then `--force-with-lease`.
   - Chain: `fix/mass-terrain-level` (PR #3) → `feat/terrain-world` (A) → `feat/path-planner` (C) → `feat/click-to-go` (D) → `feat/terrain-policy` (B).
7. **Blockers:** after two serious attempts or about 45 minutes on one blocker:
   - Write it in `PROGRESS.md` under Blockers (symptom, what was tried, the next idea).
   - Mark the test `pytest.mark.xfail(strict=True, reason=...)` only if the code is committed behind a flag.
   - Move to the next **independent** task. The dependency notes on each task say which ones are independent.
   - Never weaken an acceptance threshold silently. Write any change of threshold in the ADR.
8. **`PROGRESS.md` after every task** (git-excluded): `## State (date)` with branch, last commit and PR links; `## Done`; `## Next` (the exact next task ID from this plan); `## Blockers`; `## Measurements` (fps, eval numbers). Use the `handoff` skill format if it is available.
9. **Public repo rules:**
   - No real names, emails, usernames, absolute paths (`/Users/...`), hostnames, IPs, cloud account IDs, bucket names or SSH key names in any committed file, PR body or commit message.
   - Cloud scripts take everything from environment variables or arguments.
   - Run `git grep -nE '/Users/|/home/[a-z]|@gmail|ssh-rsa'` before each push.
10. **No hardware.** No firmware flashing or actuation. **No cloud provisioning.** Prepare scripts and smoke-test them locally only, with a fake `hcloud` and `--dry-run`.
11. **Long jobs:**
    - Any command expected to run more than 10 minutes runs in the background (`run_in_background`) with output to `runs/<name>.log`.
    - Never block a session on full training. Smoke runs only, unless the task says "launch".
12. **Keep tests fast.** New tests use tiny worlds (≤ 2 m × 2 m, ≤ 64×64 cells) and short rollouts. The whole suite should stay under about 3 minutes. Training tests use ≤ 512 steps.

---

## 3. Step 0: housekeeping (branch `fix/mass-terrain-level`, about 45 min)

Order:
1. `make check` on the branch and confirm it is green.
2. Copy `runs/v2_seed0/{policy.zip,vecnormalize.pkl}` to `results/policy/`, replacing M1. Copy `runs/v2_eval_seed{0,1,2}_50.json` to `results/route_eval_50_v2_seed{0,1,2}.json`. Check with `unzip -l` that there is no `system_info.txt`, and check the JSONs for paths.
   - Commit: `feat: ship the terrain-sampled policy (50/50 routes on three seeds)`.
3. README:
   - Fix "Known limits" (inertia now scales; the one-hot now carries the sampled class).
   - Add a v2 row to the table (3M steps, 50/50 × 3 seeds) and keep the M1 rows as history.
   - Add a "Roadmap" pointer to the new milestones.
   - Commit: `docs: report terrain-sampled training results`.
4. `docs/PLAN.md`: replace M2 with M2a World / M2b Planner / M2c Click-to-go / M2d Terrain-aware policy, each with a one-paragraph summary. Add `docs/decisions/0001-milestone-order.md`. Commit `docs: plan terrain world, planner, click-to-go UI and terrain policy`.
5. `git push -u origin fix/mass-terrain-level`, then `gh pr create --base main` (PR #3).

**Acceptance:** PR open; `make check` green; README states 50/50 honestly; `results/policy` loads with `load_policy` (the existing `test_cli` covers loading).

---

## 4. Milestone A: heightfield world (`feat/terrain-world`, about 8–12 h of agent time, little compute)

### Goal
A seeded procedural world with:
- height: hills and slopes, roughness, rocks, ditches;
- a per-cell surface class.

The world is compiled into a MuJoCo `hfield`. Per-segment surface physics are looked up every physics step. Flat `WormGoalEnv()` behaves exactly as before.

### Design decisions (ADR `0002-terrain-world-physics.md`)
1. **Representation.** `World` is a pure numpy dataclass with no MuJoCo dependency, so the planner and UI can use it headless.
   - Heights are in metres, ≥ 0, shape (rows, cols). Row index runs along +y, column index along +x.
   - Cell size defaults to 0.05 m. A 6 cm segment covers about one cell.
   - The world is centred on the origin.
2. **MuJoCo mapping.**
   - `<hfield nrow=rows ncol=cols size="rx ry zmax base">` with `rx = (cols-1)*cell/2` and `ry = (rows-1)*cell/2`.
   - `hfield_data = heights/zmax`, row-major.
   - A **verification test** confirms row 0 ↔ y = −ry: drop a ball on a single raised cell and check where it lands. If MuJoCo turns out to use the other orientation, flip it in exactly one function, `World.to_hfield_data()`.
3. **Friction: generalize the resistive seam rather than use `ContactAnisotropic`.**
   - `ContactAnisotropic` cannot follow the body frame (proven by `test_contact_adapter_anisotropy_depends_on_yaw`). So the per-cell anisotropy must come from `xfrc_applied`.
   - New adapter `TerrainResistive(FrictionModel)`. Per step it:
     - looks up each segment's cell class;
     - takes `(mu_fwd, mu_lat, viscous)` from a per-episode `SurfaceTable`;
     - projects the body axes onto the **terrain tangent plane**, with the normal from the heightfield gradient;
     - uses load = m·g·n_z;
     - adds a viscous term `-c·v` (sand and grass).
   - With a flat surface and one class, this is **bit-identical** to `ResistiveForce`. A test proves it.
4. **Sinkage and compliance.** Each segment has its own `<pair>`, so `apply()` writes `model.pair_solref[i]` per segment from its class: sand is softer (larger timeconst, so a few mm of sinkage), concrete is stiff. `apply()` runs between `mj_step1` and `mj_step2`, so the new solref takes effect at the **next** step's collision. This 2 ms lag is accepted and documented.
5. **Slope stiction.** This is the spike in A4.
   - Pure tanh drag creeps downhill. On a 10° slope with μ_f = 0.3 and v0 = 0.05, creep is about 3 cm/s.
   - If the creep test fails, switch `TerrainResistive` to **hybrid mode**:
     - isotropic MuJoCo contact friction `mu_c = k·mu_fwd` (k = 0.5) per pair, which gives real Coulomb stiction and has no orientation issue because it is isotropic;
     - plus xfrc anisotropic drag `(mu_fwd − mu_c, mu_lat − mu_c)`.
   - Flat `ResistiveForce` is never changed.
6. **Surface classes** (keep 3; extend `terrain.py`). `SurfaceProps(mu_forward, mu_lateral, viscous, solref_timeconst, roughness)`. The existing `_RANGES` stay as the friction source.
   - CONCRETE ("stone"): low viscous; stiff (timeconst 0.005); high-frequency roughness of 2–6 mm comes from the generator.
   - GRASS: viscous 0.05–0.15 N·s/m per segment; timeconst 0.01; roughness 1–3 mm.
   - SAND: viscous 0.15–0.4; timeconst 0.02–0.04 (sinkage); smooth.
   - The ranges are constants in `terrain.py` and get a unit test each.
7. **Rendering.** Use the default hfield material in MuJoCo frames. The UI draws the class map itself in 2D. Class-coloured hfield textures via `MjSpec` are a **nice-to-have** (A6). If they fail, drop them and note it in the ADR.
8. **Back-compat.**
   - `model.build(spec)` and `WormGoalEnv()` with `world=None` produce the identical plane model.
   - Golden characterization test: the first observation and the head position after 40 GO steps for seeds 0–2. Record the values on the branch before any env change.
9. **Training-size worlds reuse one compiled model.** `hfield_data` is overwritten in place on reset when the shape matches. In rgb mode the renderer needs `mjr_uploadHField`. Training never renders.

### Interfaces

```python
# worm/world.py  (pure numpy; imports only terrain.TerrainClass)
@dataclass(frozen=True)
class WorldConfig:
    size: tuple[float, float] = (10.0, 10.0)   # m
    cell: float = 0.05                          # m
    hill_height: float = 0.3                    # m, peak-to-peak of low-frequency relief
    hill_scale: float = 2.0                     # m, correlation length
    max_slope_deg: float = 15.0                 # relief rescaled so the max slope <= this
    roughness: float = 1.0                      # x the per-class roughness amplitude
    n_rocks: int = 10
    rock_radius: tuple[float, float] = (0.08, 0.3)
    rock_height: tuple[float, float] = (0.05, 0.2)
    n_ditches: int = 2
    ditch_width: tuple[float, float] = (0.15, 0.5)
    ditch_depth: tuple[float, float] = (0.04, 0.15)
    ditch_length: tuple[float, float] = (1.0, 4.0)
    mix: tuple[float, float, float] = (1/3, 1/3, 1/3)  # area fraction CONCRETE, GRASS, SAND
    patch_scale: float = 1.5                    # m, correlation length of surface patches
    clear_radius: float = 0.5                   # m, no obstacles around the spawn point (0, 0)

@dataclass(frozen=True)
class World:
    heights: NDArray[np.float32]                # (rows, cols), metres, min 0
    surface: NDArray[np.uint8]                  # (rows, cols), TerrainClass values
    obstacle: NDArray[np.bool_]                 # (rows, cols), rocks (ground truth, for viz and tests)
    cell: float
    seed: int | None = None
    @property def shape(self) -> tuple[int, int]
    @property def extent(self) -> tuple[float, float, float, float]   # xmin, xmax, ymin, ymax
    def to_cell(self, xy: ArrayLike) -> NDArray[np.int_]              # (..., 2) -> (..., [row, col]), clipped
    def to_xy(self, rc: ArrayLike) -> NDArray[np.float64]
    def contains(self, xy: ArrayLike) -> NDArray[np.bool_]
    def height_at(self, xy: ArrayLike) -> NDArray[np.float64]         # bilinear
    def class_at(self, xy: ArrayLike) -> NDArray[np.uint8]            # nearest cell
    def normal_at(self, xy: ArrayLike) -> NDArray[np.float64]         # (..., 3) unit, from the central-difference gradient
    def slope_deg(self) -> NDArray[np.float32]                        # per cell
    def to_hfield_data(self) -> tuple[NDArray[np.float32], float]     # (normalized flat data, zmax)

def flat(size: tuple[float, float], cell: float = 0.05,
         cls: TerrainClass = TerrainClass.CONCRETE) -> World
def generate(cfg: WorldConfig, seed: int) -> World
def smooth_noise(shape: tuple[int, int], scale_cells: float, rng: np.random.Generator) -> NDArray[np.float64]
    # FFT-filtered white noise (Gaussian kernel), zero mean, unit std; no scipy
```

```python
# worm/terrain.py  (additions)
@dataclass(frozen=True)
class SurfaceProps:
    friction: FrictionParams
    viscous: float          # N s / m per segment
    solref_timeconst: float # s
    roughness: float        # m, generator amplitude
@dataclass(frozen=True)
class SurfaceTable:
    props: tuple[SurfaceProps, SurfaceProps, SurfaceProps]   # indexed by TerrainClass
    def arrays(self) -> tuple[NDArray, NDArray, NDArray]     # mu (3,2), viscous (3,), timeconst (3,)
def surface_props(cls: TerrainClass, rng: np.random.Generator) -> SurfaceProps
def nominal_surface(cls: TerrainClass) -> SurfaceProps       # range midpoints, used at randomizer level 0
```

```python
# worm/friction.py  (additions; Protocol unchanged)
class TerrainResistive:
    def __init__(self, body_ids: list[int], pair_ids: list[int], world: World,
                 v0: float = 0.05, hybrid: bool = False, stiction_fraction: float = 0.5) -> None
    def configure(self, model, params: FrictionParams) -> None   # uniform table from params (tests)
    def configure_table(self, model, table: SurfaceTable) -> None
    def set_world(self, world: World) -> None                     # same shape, new data
    def apply(self, model, data) -> None
    @property
    def classes(self) -> NDArray[np.uint8]                        # class under each segment at the last apply
```

```python
# worm/model.py
def to_xml(spec: WormSpec, world: World | None = None) -> str   # world -> <hfield> + type="hfield" floor geom
def build(spec: WormSpec, world: World | None = None) -> mujoco.MjModel
def load_world(m: mujoco.MjModel, world: World) -> None          # overwrite hfield_data in place (shape must match)
def pair_ids(m: mujoco.MjModel, spec: WormSpec) -> list[int]
```

```python
# worm/env.py  (additions; all defaults keep M1 behaviour)
class WormGoalEnv:
    def __init__(..., world: World | WorldSampler | None = None, obs_version: int = 1) -> None
    # reset options: "target": (x, y); "start": (x, y); "yaw": float; "world": World (replace, same shape)
    @property def world(self) -> World | None
    @property def surface_under_head(self) -> TerrainClass
# WorldSampler = Callable[[np.random.Generator], World]   (A5, used for training: a new world per reset)
```

```python
# worm/mapviz.py  (pure numpy -> uint8 RGB, nearest-neighbour upscaling, no matplotlib)
CLASS_COLORS: dict[TerrainClass, tuple[int, int, int]]   # stone grey, grass green, sand tan
def world_image(world: World, px_per_m: float = 40.0, shade: bool = True) -> NDArray[np.uint8]
    # class colour x hillshade; rocks dark; ditches darker by depth
def draw_polyline(img, world, points, rgb, px_per_m, width=2) -> None
def draw_points(img, world, points, rgb, px_per_m, radius=3) -> None
def world_to_pixel(world, xy, px_per_m) -> NDArray[np.int_]  # +y is up in the image
```

### Files
- **Create:** `worm/world.py`, `worm/mapviz.py`, `scripts/make_world.py`, `scripts/bench_env.py`, `tests/test_world.py`, `tests/test_mapviz.py`, `tests/test_terrain_physics.py`, `docs/decisions/0002-terrain-world-physics.md`.
- **Modify:** `worm/terrain.py`, `worm/friction.py`, `worm/model.py`, `worm/env.py`, `worm/randomize.py` (`PhysicsParams.surfaces: SurfaceTable | None`), `tests/test_env.py`, `tests/test_model.py`, `tests/test_friction.py`, README (short "Terrain worlds" section).

### Tasks (each one is test first, watch it fail, implement, watch it pass, commit)

**A0. Golden back-compat test (characterization; passes at once).**
- `tests/test_env.py::test_flat_env_matches_the_m1_golden_trajectory`. For seeds 0, 1, 2 it stores `obs0[:8]` and the head xy after 40 `GO` steps, rounded to 1e-6.
- Values are generated once on the branch tip before any change and pasted as literals.
- Commit `test: pin the flat-world trajectory before the terrain refactor`.

**A1. `World` basics and coordinates.**
- `test_flat_world_has_zero_height_and_one_class`
- `test_to_cell_and_to_xy_round_trip_on_cell_centres`
- `test_world_is_centred_on_the_origin` (extent symmetric)
- `test_height_at_is_bilinear_between_cells`
- `test_normal_on_a_tilted_plane_matches_the_analytic_normal` (heights = x·tan 10°)
- `test_slope_deg_of_a_10_degree_ramp`
- `test_contains_rejects_points_outside`

**A2. Generator (pure).**
- `test_generate_is_deterministic_for_a_seed_and_differs_across_seeds`
- `test_heights_are_nonnegative_and_slope_respects_max_slope` (excluding rock and ditch cells)
- `test_surface_mix_matches_the_requested_fractions_within_5_percent` (100×100 cells, mix (0.6, 0.2, 0.2))
- `test_rocks_are_marked_as_obstacles_and_raised_by_at_least_min_height`
- `test_ditches_are_depressions_of_at_least_min_depth`
- `test_spawn_area_is_clear_and_flat` (radius 0.5 m, slope < 5°, no obstacle)
- `test_mix_of_one_class_gives_a_single_class_world`
- Implement `smooth_noise` with numpy FFT. Pick the class with argmax of three weighted noise fields, with weights chosen by quantile so the area fractions match `mix`.
- Rocks are smooth caps `h·(1 − (r/R)²)^0.5`; ditches are oriented rectangles with cosine-profile walls.
- **Then** add `worm/mapviz.py` with tests:
  - `test_world_image_has_the_expected_size_and_class_colours`
  - `test_world_to_pixel_puts_+y_at_the_top`
  - `test_draw_polyline_changes_pixels_along_the_line`
- Then `scripts/make_world.py --seed --size --mix --out map.png`, covered by a smoke test in `tests/test_cli.py`.
- → **SP1** (push the branch; no PR yet).

**A3. MuJoCo hfield model.**
- `test_model.py::test_build_without_world_is_unchanged` (XML string equals the old output; compare `to_xml(spec)` before and after with the golden)
- `test_build_with_world_has_one_hfield_matching_the_world_shape`
- `test_hfield_rows_run_along_plus_y` (raise one cell at (row r, col c); drop a 1 cm sphere above `world.to_xy((r, c))`; assert it rests ≥ the raised height; a sphere 3 cells away rests at ground)
- `test_worm_rests_on_a_flat_hfield_at_the_same_height_as_on_the_plane` (±2 mm)
- `test_load_world_overwrites_heights_in_place`
- Implement via XML (`<asset><hfield name="terrain" nrow ncol size/>`); fill `m.hfield_data` after compile. The floor geom keeps the name `"floor"` so the pairs are unchanged.

**A4. `TerrainResistive` physics.**
- `test_friction.py`:
  - `test_terrain_resistive_on_flat_uniform_world_equals_resistive_force` (same `xfrc_applied` to 1e-12 over 200 steps on a 1-segment and a 5-segment model)
  - `test_terrain_resistive_uses_the_class_under_each_segment` (concrete left half, sand right half: a segment coasting on sand stops sooner)
  - `test_sand_segment_sinks_deeper_than_concrete` (≥ 1 mm after settling)
  - `test_viscous_term_scales_with_speed`
  - `test_drag_directions_lie_in_the_terrain_tangent_plane_on_a_slope` (force · normal ≈ 0)
- `tests/test_terrain_physics.py` (the **spike**, always kept as a regression test):
  - `test_idle_worm_on_a_10_degree_slope_creeps_less_than_5cm_in_10s`
  - `test_gait_climbs_a_5_degree_slope` (open-loop GO across 10 s gains > 5 cm of uphill displacement; if the gait cannot climb at all, record that and set the threshold to "not slower than 50 % of flat")
- If the creep test fails with pure tanh, implement `hybrid=True` and make it the default **for worlds only**. Write the numbers in ADR 0002.
- Also add `terrain.py` tests: `test_surface_props_ranges_per_class` and `test_nominal_surface_is_inside_the_range`.

**A5. Env integration.**
- `test_env.py`:
  - `test_world_env_passes_the_gymnasium_checker`
  - `test_world_env_spawns_on_the_terrain_surface` (head z ≈ height_at + half height ± 5 mm after reset settle)
  - `test_reset_start_option_places_the_head` (world and flat)
  - `test_terrain_one_hot_is_the_class_under_the_head_in_a_world` (half-and-half world; start on each half)
  - `test_world_sampler_gives_a_new_world_each_reset_without_recompiling` (same `env.model` object; different `hfield_data`)
  - `test_per_episode_surface_table_is_sampled_per_class` (level 0 gives nominal)
  - A0's golden test still passes.
- Reset in world mode:
  - sample the `SurfaceTable`;
  - `model.load_world` if a sampler is used;
  - place the root at `start` (default (0, 0)) and z = max height under the footprint + half height + 3 mm;
  - `mj_forward`, then 25 settle physics steps with zero ctrl.
- Targets must lie inside the world: `target_range` targets are re-sampled up to 20 times until `world.contains`, and must not be in an obstacle.

**A6. Benchmark plus optional texture.**
- `scripts/bench_env.py --world {flat,10m} --steps 2000` prints control steps/s. Write the numbers in the ADR and `PROGRESS.md`.
- Then try class-coloured hfield rendering (MjSpec texture from `mapviz.world_image`, with a rendering-gated test `test_world_render_shows_class_colours`). Timebox: 1 h. Drop it if it does not work.

**A7.** README section, ADR, `make check`, push, PR (`--base` per rule 6). → **SP2**.

### Acceptance criteria
- `make check` green, and A0's golden test is unchanged.
- The flat 100 m eval of `results/policy` is not rerun in CI. Fable runs `eval_route.py --runs 10` once locally and gets ≥ 9/10 (it was 50/50).
- **Mixed-world sanity check with the existing policy:** a 10 m mixed-surface world with no obstacles and max slope 10°. Run 10 straight 3 m targets via `scripts/navigate.py` after C, or an ad-hoc loop now, and record the success count. This is informational, not a gate.
- World env throughput ≥ 40 % of flat on a 10 m × 10 m world (flat is about 450 control steps/s per env).

### Risks and fallbacks
| Risk | Fallback |
|---|---|
| hfield-box collision is slow or jittery (many contacts) | Raise the `pair` `margin` a little. Lower world-mode cell resolution to 0.08 m. Swap the segment collision geoms for 2 spheres per box in world mode only (`model.to_xml(..., collision="spheres")`). Last resort: coarser training worlds (8 m at 0.1 m) |
| The `pair_solref` change has no visible sinkage | Use `pair_margin` / `pair_gap` per pair instead, or model sand as viscous only. Document it |
| Slope creep | Hybrid stiction (A4) |
| Hfield orientation confusion | One function plus a ball-drop test |
| The worm falls off the world edge | Env truncates with `info["out_of_world"]=True` when the head leaves the extent minus 0.3 m |

---

## 5. Milestone C: planner and navigator (`feat/path-planner`, about 6–9 h, no compute)

### Goal
Turn a `World` and a goal into a feasible waypoint route, or `Infeasible(reason)`. Then drive it with any policy, with stuck and off-path replanning. The planner is pure Python and fully headless.

### Design decisions (ADR `0003-planner.md`)
- **Grid.** Plan on a 0.1 m grid. That is 2× downsampling of the 5 cm world: max-pool for obstacles and step heights, mean for cost. A 10 m map is 100×100 = 10k nodes; 20 m is 40k. Pure-Python A* with `heapq` takes well under 1 s.
- **Cost per cell:** `cost = surface_cost[class] · (1 + slope_weight · (slope/max_slope)²)`.
  - Defaults are 1 / relative speed: CONCRETE 1.0, GRASS 1.4, SAND 2.0, then overwritten by calibration (C5).
  - Infinite when:
    - slope > `max_slope_deg` (20°);
    - the max height step to any 8-neighbour (at 5 cm resolution) > `max_step` (0.012 m, about half the 2.5 cm body height). This catches rock flanks and ditch walls; shallow ditches stay passable;
    - the cell is a ground-truth obstacle (redundant with the step test, kept as a safety net);
    - the cell is out of bounds.
  - Then **inflate** blocked cells by `inflate_radius` (0.15 m: half-width plus the undulation amplitude of about 8 cm, plus margin) using a disk dilation (numpy, separable passes or a precomputed disk offset loop).
- **Search.** A* with 8-connectivity. Edge cost = mean of the two cell costs × step length. Heuristic = octile distance × min cell cost (admissible). No corner cutting past blocked cells.
- **Smoothing.** Greedy line-of-sight string pulling. A shortcut from i to j is accepted if every cell on the supercover line is finite and the line's integrated cost is ≤ (1 + ε) × the replaced section's cost (ε = 0.05). This is Theta*-like quality without Theta* complexity. Theta* is not built (YAGNI).
- **Waypoints.** Resample the smoothed polyline so that:
  - every polyline corner with turn > 20° is a waypoint;
  - straight stretches are split into ≤ 2.0 m legs;
  - points closer than 0.3 m are merged;
  - the goal is always last.
- **Tolerance.** Intermediate waypoints use a pass-through tolerance of 0.3 m. The final goal uses the env's 0.15 m. This keeps the settled 15 cm for arrival while avoiding stalls at corners.
- **Infeasible reasons** (an enum): `OUT_OF_BOUNDS`, `START_BLOCKED`, `GOAL_BLOCKED`, `NO_PATH`.
  - A start inside an inflated cell is snapped to the nearest free cell within 0.3 m (the worm itself may sit next to a rock). Otherwise `START_BLOCKED`.
  - The goal is snapped within 0.2 m, and the snap is reported in `Plan.goal_snapped`.
- **Stepwise route following.** A new `RouteFollower` is a step-driven state machine. `run_route` becomes a wrapper with identical behaviour; the existing `test_route.py` passes unchanged and proves it.
- **Replanning** in `Navigator`:
  - **Stuck:** the best distance to the current waypoint has not improved by 0.05 m in `stuck_seconds` (15 s).
  - **Off path:** cross-track distance to the current leg > 0.6 m.
  - **Waypoint timeout:** 60 s.
  - Replanning starts from the current head position. At most `max_replans` (5) replans, then fail `"stuck"`.
  - A flip or leaving the world fails at once.

### Interfaces

```python
# worm/planner.py
@dataclass(frozen=True)
class CostConfig:
    resolution: float = 0.1
    surface_cost: tuple[float, float, float] = (1.0, 1.4, 2.0)
    slope_weight: float = 1.0
    max_slope_deg: float = 20.0
    max_step: float = 0.012
    inflate_radius: float = 0.15
@dataclass(frozen=True)
class CostMap:
    cost: NDArray[np.float64]       # (r, c), np.inf where blocked
    resolution: float
    origin: tuple[float, float]     # xy of cell (0, 0) centre
    def to_cell(self, xy) -> tuple[int, int]; def to_xy(self, rc) -> NDArray; def free(self, rc) -> bool
class Reason(enum.Enum): OUT_OF_BOUNDS; START_BLOCKED; GOAL_BLOCKED; NO_PATH
@dataclass(frozen=True)
class Infeasible:
    reason: Reason
    detail: str
@dataclass(frozen=True)
class Plan:
    path: NDArray[np.float64]       # (n, 2) smoothed polyline, start .. goal
    waypoints: NDArray[np.float64]  # (m, 2) route for the follower (excludes the start)
    cost: float
    expanded: int
    goal_snapped: bool
def build_costmap(world: World, cfg: CostConfig = CostConfig()) -> CostMap
def astar(costmap: CostMap, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None
def smooth(costmap: CostMap, cells: list[tuple[int, int]], eps: float = 0.05) -> NDArray[np.float64]
def to_waypoints(path: NDArray[np.float64], spacing: float = 2.0, corner_deg: float = 20.0,
                 merge: float = 0.3) -> NDArray[np.float64]
def plan(costmap: CostMap, start: ArrayLike, goal: ArrayLike, spacing: float = 2.0) -> Plan | Infeasible
```

```python
# worm/navigate.py
@dataclass
class RouteFollower:
    waypoints: NDArray[np.float64]
    pass_tolerance: float = 0.3          # intermediate
    def start(self, env: WormGoalEnv, obs) -> NDArray[np.float32]   # sets the first target, returns obs
    def step(self, env, policy) -> StepReport                       # one control step
    @property def done(self) -> bool; @property def reached(self) -> int
@dataclass(frozen=True)
class StepReport:
    obs: NDArray[np.float32]; head_xy: NDArray[np.float64]; waypoint_index: int
    event: Literal["", "waypoint", "goal", "timeout", "flipped", "out_of_world"]
@dataclass(frozen=True)
class NavConfig:
    stuck_seconds: float = 15.0; stuck_progress: float = 0.05; off_path: float = 0.6
    waypoint_timeout: float = 60.0; max_replans: int = 5; max_seconds: float = 1200.0
class Navigator:
    def __init__(self, env: WormGoalEnv, policy: Policy, costmap: CostMap, cfg: NavConfig = NavConfig()) -> None
    def go(self, goal: ArrayLike) -> Plan | Infeasible      # plans from the current head xy; does not reset the body
    def step(self) -> StepReport                             # advance one control step; replans if needed
    @property def status(self) -> Literal["idle", "driving", "reached", "failed"]
    @property def plan(self) -> Plan | None; @property def replans(self) -> int; @property def failure(self) -> str | None
    @property def trail(self) -> NDArray[np.float64]
@dataclass(frozen=True)
class NavResult:
    success: bool; reason: str; sim_seconds: float; replans: int
    trajectory: NDArray; class_seconds: tuple[float, float, float]   # time spent on each class
def navigate(env, policy, costmap, start, goal, seed: int | None = None,
             cfg: NavConfig = NavConfig(), on_step: Callable[[], None] | None = None) -> NavResult | Infeasible
```

```python
# worm/controllers.py
def bearing_steer(layout: ObsLayout, gain: float = 1.0) -> Policy   # moved from tests/test_route.py; the demo fallback
```

### Files
- **Create:** `worm/planner.py`, `worm/navigate.py`, `worm/controllers.py`, `scripts/plan_map.py`, `scripts/navigate.py`, `scripts/calibrate_costs.py`, `tests/test_planner.py`, `tests/test_navigate.py`, `tests/test_controllers.py`, `docs/decisions/0003-planner.md`.
- **Modify:** `worm/route.py` (`run_route` delegates to `RouteFollower`), `tests/test_route.py` (import `bearing_steer`), `tests/test_cli.py`.

### Tasks
**C1. Cost map.** Hand-built `World` fixtures in a `tests/conftest.py` helper `make_world(heights, surface)`.
- `test_flat_uniform_world_costs_one_everywhere_inside`
- `test_surface_cost_follows_the_class`
- `test_steep_slope_is_blocked`
- `test_rock_and_its_inflation_ring_are_blocked`
- `test_shallow_ditch_is_passable_and_deep_ditch_is_blocked`
- `test_downsampling_keeps_a_one_cell_obstacle` (max-pool)

**C2. A*, smoothing, waypoints, plan.**
- `test_astar_straight_line_on_empty_map_is_optimal`
- `test_astar_goes_around_a_wall_through_the_gap`
- `test_astar_returns_none_when_walled_off`
- `test_astar_prefers_a_longer_concrete_detour_over_a_short_sand_crossing_when_cheaper`
- `test_astar_does_not_cut_blocked_corners`
- `test_smooth_reduces_vertices_and_never_enters_blocked_cells`
- `test_waypoints_are_at_most_2m_apart_and_end_at_the_goal`
- `test_waypoints_keep_corners`
- `test_plan_reports_each_infeasible_reason` (parametrized over the 4 reasons)
- `test_plan_snaps_a_start_next_to_a_rock`
- `test_plan_on_generated_worlds_never_crosses_blocked_cells` (property-style: seeds 0–19, random free start and goal)
- `test_plan_10m_world_runs_under_1s`
- Then `scripts/plan_map.py --seed --size --mix --start x y --goal x y --out plan.png` (PNG of world, path and waypoints, or prints `infeasible: NO_PATH ...` and exits 2), with a CLI smoke test. → **SP3**.

**C3. `RouteFollower` refactor and `bearing_steer`.**
- First move `steer` into `worm/controllers.py` with `test_bearing_steer_turns_toward_the_target_side`.
- Then the follower tests: `test_follower_uses_pass_tolerance_for_intermediate_and_env_tolerance_for_goal` and `test_follower_reports_events`.
- Then refactor `run_route` onto it. **All existing `test_route.py` tests must pass unmodified, apart from the import.**

**C4. Navigator.** Tests use `bearing_steer` on small nominal worlds (4×4 m), 0.5 m-wide wall with a gap.
- `test_navigator_reaches_a_goal_behind_a_wall_via_the_gap`
- `test_navigator_does_not_reset_the_body_between_goals` (two `go()` calls; no teleport > 5 cm per step)
- `test_navigator_replans_when_stuck` (a policy wrapper that idles for 20 s, then steers: `replans ≥ 1` and success)
- `test_navigator_replans_when_pushed_off_path` (teleport the root 1 m sideways mid-route)
- `test_navigator_gives_up_after_max_replans` (an idle policy gives `failed` / "stuck")
- `test_navigate_returns_infeasible_for_a_goal_inside_a_rock`
- `test_nav_result_accounts_time_per_terrain_class`
- Then `scripts/navigate.py --policy results/policy --seed 3 --size 10 --goal 4 4 [--video out.mp4] [--json out.json]`. The video reuses `FrameRecorder` and composes the follow cam with the `mapviz` overview (the same layout as `showcase.py`). CLI smoke test with `--policy scripted`. → **SP4**.

**C5. Cost calibration (small compute, about 5 min).** `scripts/calibrate_costs.py --policy results/policy` measures the mean speed per class on flat single-class worlds (5 targets × 3 classes) and prints `surface_cost` = v_concrete / v_class. Fable runs it once, writes the numbers into `CostConfig` defaults with the measurement in ADR 0003, and adds a unit test `test_cost_defaults_are_ordered_concrete_grass_sand` (only if the measurement confirms that order; otherwise use the measured order and note it).

**C6.** `make check`, ADR, README "Planner" section with a committed PNG `results/plan_example.png` (small, < 200 KB), PR.

### Acceptance criteria
- The planner tests pass headless.
- On 20 generated 10 m worlds (default config), every non-infeasible plan has zero blocked cells and all legs ≤ 2 m.
- `scripts/navigate.py` with `results/policy` on 10 generated 10 m worlds with random feasible goals ≥ 4 m away: success recorded, target ≥ 7/10. Below that it is a **finding**, recorded in the PR and in `PROGRESS.md` as motivation for B, not a blocker.
- `test_route.py` passes unchanged.

### Risks and fallbacks
| Risk | Fallback |
|---|---|
| A* is too slow on 20 m maps | Use 0.15 m resolution for maps > 15 m, or cap the UI map size at 15 m |
| The policy stalls at corners | Increase the pass tolerance to 0.4 m, or add a lookahead (target the point 1.5 m ahead along the path, "carrot" mode, in `RouteFollower(carrot=1.5)`) |
| Inflation closes the narrow gaps the generator makes | Ensure the generator keeps at least 0.4 m between rocks (test in A2), or lower `inflate_radius` to 0.12 |

---

## 6. Milestone D: click-to-go UI (`feat/click-to-go`, about 6–9 h, no compute)

### Decision (ADR `0004-click-to-go-ui.md`): a local web UI on the stdlib HTTP server
Rejected: the MuJoCo passive viewer. Reasons:
- On macOS `launch_passive` needs the `mjpython` launcher, which is awkward for users and CI.
- The viewer has `key_callback` but **no mouse-click callback**. Double-click selection works on bodies, but the world body (which holds the hfield geom) is not selectable, so "click a ground point" would need hacks on private viewer internals (`perturb.select`, ray picks).
- No sliders or forms for seed and mix without a custom GUI.
- Not testable headless.

Chosen: `http.server.ThreadingHTTPServer` (stdlib, **no new dependencies**) serving one static `index.html` (vanilla JS plus a `<canvas>`). Reasons:
- A top-down 2D map you click on is the natural interface for "click somewhere".
- The core logic stays in a pure `DemoSession`, which is unit-testable. The HTTP layer is thin and tested with `urllib` against a server on port 0.
- It works on macOS, Linux and inside the cloud VM over SSH port-forwarding.
- The MuJoCo 3D follow-cam is optional: a `/frame.jpg` endpoint rendered **on the sim thread**, because GL contexts are thread-bound. It is skipped when rendering is unavailable.

### Architecture
- **Sim thread:** owns `env`, `Navigator` and the optional `mujoco.Renderer`. It loops at `speed` × 20 Hz with wall-clock pacing and checks a command queue (new map, goal, pause, speed). After each tick it publishes an immutable snapshot under a lock: head xy, yaw, trail (decimated), plan path and waypoints, status, replans, class under head, sim time, and the latest JPEG bytes if rendering is on.
- **HTTP threads:** only enqueue commands and read snapshots.

Endpoints (JSON):

| Endpoint | Purpose |
|---|---|
| `GET /` | `index.html` |
| `GET /api/state` | snapshot |
| `GET /api/map.png` | `mapviz.world_image` of the current world, cached per map |
| `GET /api/map.json` | extent, px_per_m, map id |
| `POST /api/map {seed, size, mix:[c,g,s], slope, rocks, ditches}` | new world: rebuilds the env, spawns at the centre |
| `POST /api/goal {x, y}` | returns `{"ok": true, "plan": {...}}` or `{"ok": false, "reason": "GOAL_BLOCKED", "detail": ...}` |
| `POST /api/control {pause: bool, speed: 1..20}` | pause and speed |
| `GET /api/frame.jpg` | 204 if rendering is off |

**Front end:**
- Map image on a canvas. A click converts pixel to world xy using `map.json` and POSTs a goal.
- Draws the path (blue), waypoints (red, grey once reached), trail, the worm as an oriented triangle, and the goal flag. Infeasible goals show a red X with the reason text.
- Sidebar: seed input, size select (5/10/15 m), three mix sliders (normalized), slope, rocks and ditches sliders, "New map", speed slider, pause, status line (policy name, status, replans, class under head), and an optional 3D frame.
- Polls `/api/state` every 100 ms. No build step and no external CDN, so it works offline.

**Policy fallback chain** in `scripts/demo.py` and `worm.demo.load_controller(spec)`:
1. `--policy PATH`, if given;
2. `results/policy_terrain` (B's output, which carries `meta.json` with `obs_version`);
3. `results/policy` (v2 terrain-sampled, obs v1);
4. `scripted` (`bearing_steer`).

The status line shows which one loaded. The env is built with the controller's `obs_version`.

### Interfaces

```python
# worm/demo.py   (no HTTP, no threads)
@dataclass(frozen=True)
class MapRequest:
    seed: int = 0; size: float = 10.0; mix: tuple[float, float, float] = (0.4, 0.3, 0.3)
    max_slope_deg: float = 12.0; n_rocks: int = 12; n_ditches: int = 2
    def to_config(self) -> WorldConfig
    @staticmethod
    def from_json(d: dict[str, Any]) -> MapRequest     # validates and clamps; raises ValueError
@dataclass(frozen=True)
class Controller:
    name: str
    policy: Policy
    obs_version: int
def load_controller(choice: str | Path | None) -> Controller   # fallback chain above
class DemoSession:
    def __init__(self, controller: Controller, render: bool = False) -> None
    def new_map(self, req: MapRequest) -> None
    def goal(self, xy: tuple[float, float]) -> dict[str, Any]   # JSON-able plan or infeasible
    def tick(self, n_steps: int = 1) -> None
    def snapshot(self) -> dict[str, Any]
    def map_png(self) -> bytes
    def frame_jpeg(self) -> bytes | None
    @property def map_id(self) -> int

# worm/server.py
class SimLoop(threading.Thread):     # owns the session; command queue; speed and pause; publishes snapshots
    def __init__(self, session: DemoSession, hz: float = 20.0) -> None
    def submit(self, cmd: Callable[[DemoSession], Any]) -> Future[Any]
    def latest(self) -> dict[str, Any]
    def stop(self) -> None
def make_server(loop: SimLoop, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer
# worm/static/index.html   (packaged; loaded via importlib.resources)
# scripts/demo.py  --policy --port --host 127.0.0.1 --render/--no-render --open (webbrowser.open)
```

### Files
- **Create:** `worm/demo.py`, `worm/server.py`, `worm/static/index.html`, `scripts/demo.py`, `tests/test_demo.py`, `tests/test_server.py`, `docs/decisions/0004-click-to-go-ui.md`.
- **Modify:** README (a "Click to go" section with the command, plus a screenshot `results/demo.png` captured with the `run` or browser skill if available; otherwise skip the screenshot), `docs/PLAN.md` architecture table (add the planner, navigator and demo rows). Hatch includes `worm/static/` (it is in the package dir). Check that `uv build` is unnecessary; tests use `importlib.resources.files("worm") / "static" / "index.html"`.

### Tasks
**D1. `MapRequest` and `load_controller`.**
- `test_map_request_from_json_clamps_and_normalizes_mix`
- `test_map_request_rejects_garbage`
- `test_load_controller_falls_back_to_scripted_when_no_policy_exists` (tmp path)
- `test_load_controller_reads_obs_version_from_meta_and_defaults_to_1`

**D2. `DemoSession`** (headless; scripted controller, 4 m map).
- `test_new_map_spawns_the_worm_at_the_centre`
- `test_goal_returns_a_plan_with_waypoints`
- `test_goal_inside_a_rock_returns_the_reason`
- `test_tick_moves_the_worm_toward_the_goal` (200 ticks shorten the distance)
- `test_snapshot_is_json_serializable_and_has_the_documented_keys`
- `test_a_new_goal_while_driving_replans_from_the_current_position`
- `test_map_png_is_a_png_of_the_world_size`

**D3. `SimLoop` and the server** (a thread; port 0; `urllib.request`).
- `test_get_root_serves_the_page_with_a_canvas`
- `test_post_map_then_goal_then_state_shows_driving`
- `test_bad_json_returns_400`
- `test_unknown_path_returns_404`
- `test_server_binds_localhost_by_default`
- `test_frame_endpoint_returns_204_without_rendering`
- `test_frame_endpoint_returns_a_jpeg_with_rendering` (uses the `rendering` fixture)
- `test_loop_speed_control_changes_steps_per_second` (loose bounds)
- Every test shuts the server and loop down in a fixture finalizer.

**D4. Front end.** Write `index.html`.
- Test only the structure: `test_index_html_has_no_external_urls` (no `http` in `src`/`href`) and `test_index_html_references_every_api_endpoint`.
- Manually launch `scripts/demo.py` and drive it with the `run` skill or the browser tool if available: click 3 goals, one of them on a rock. Record in `PROGRESS.md` what was observed.

**D5.** CLI smoke test: `scripts/demo.py --port 0 --smoke` starts, serves `/api/state` once, and exits 0. `make check`, ADR, README, PR. → **SP5 (MVP)**.

### Acceptance criteria
- `uv run python scripts/demo.py --open`:
  - generates a map;
  - a click plans and shows a path in under 1 s on a 10 m map;
  - the worm visibly drives it at the chosen speed;
  - a click on a rock shows "GOAL_BLOCKED";
  - a click while driving re-routes.
- It works with `results/policy` and with `--policy scripted`.
- The UI is localhost only. It writes no files and makes no network calls outside localhost.

### Risks and fallbacks
| Risk | Fallback |
|---|---|
| macOS GL renderer fails off the main thread | Run `SimLoop` on the main thread and `serve_forever` in a daemon thread (`scripts/demo.py --main-thread-sim`), or ship with `--no-render` as the default |
| The sim is too slow at high speed on big maps | Cap the speed, or show "sim time / wall time" in the UI |
| The browser canvas shows y flipped | One `world_to_pixel` convention, shared by the Python tests (mapviz) and a JS function, with a comment in both |

---

## 7. Milestone B: terrain-aware policy (`feat/terrain-policy`, about 8–12 h agent time, plus compute)

### Goal
A policy that sees real-sensor-plausible terrain cues and is trained on a curriculum of mixed-terrain worlds. It must beat the current policy on mixed-terrain routes and must not regress on the flat 100 m eval. Old policies keep loading and evaluating.

### Design decisions (ADR `0005-obs-v2-and-versioning.md`, ADR `0006-terrain-curriculum.md`)
**Obs v2 = v1 (24) + 7 = 31 dims.** All new inputs are real-sensor-plausible:
- `terrain` (existing 3): the class **under the head**. On the robot this is the camera classifier's latest classification of the patch the head is now on.
- `terrain_ahead` (new 3): the one-hot class at 0.25 m ahead along the head axis (camera view). Per step, with probability `class_error` (0.05) a random wrong class is reported (classifier error).
- `height_ahead` (new 4): a 2×2 downsample of the VL53L5CX 8×8 footprint.
  - Sample points in the head frame: forward {0.15, 0.30} m × lateral {−0.06, +0.06} m.
  - Value = `(h(point) − h(head)) / 0.1`, clipped to [−1, 1].
  - Gaussian noise σ = 3 mm, quantized to 1 mm.
  - Computed by bilinear lookup in `World.height_at`, not `mj_ray`. This is deterministic and cheap. ADR 0005 notes it as equivalent to a downward-tilted ToF on a 2.5-D world.
  - It is zero in flat mode.
- Slope is **already** sensed as IMU `roll_pitch`. No new slope channel; this avoids duplication.
- Sensor noise parameters go in `SensorParams` (`class_error`, `tof_noise`, `tof_quant`).

**Versioning:**
- `ObsLayout(version: int = 1)`. `obs_layout(spec, version=1)` builds `sizes` from the registry `_OBS_FIELDS: dict[int, tuple[tuple[str, int|callable], ...]]`. v1 is exactly today's list, and `test_spec.py` stays unchanged.
- `LATEST_OBS_VERSION = 2`. The env takes `obs_version` (default **1** until B ships a trained policy, then the CLI default moves to 2 while the env default stays 1 for back-compat).
- Each saved run gets `meta.json`: `{"obs_version": 2, "obs_dim": 31, "action_dim": 7, "steps": ..., "config": "<name>", "git_commit": "<short sha>"}`. No machine info.
- `load_policy(path) -> LoadedPolicy` (a callable plus `.meta`) reads `meta.json`, with v1 as the default when it is missing. It builds the VecNormalize dummy env with the matching version and asserts `obs_dim`.
- `LoadedPolicy.__call__` keeps the `Policy` signature, so `run_route` and the existing tests still work.

**Curriculum** (`worm/curriculum.py`): `WorldCurriculum.sampler(level) -> WorldSampler` on fixed-shape 8 m × 8 m worlds (160×160 cells at 5 cm, one compiled model per env).

| Level | World |
|---|---|
| L0 | flat, single random class (today's v2 task) |
| L1 | flat, mixed patches (`patch_scale` 0.5–2 m) |
| L2 | L1 + hills (max slope ≤ 10°) + per-class roughness |
| L3 | slope ≤ 18°, shallow ditches (< max_step), small rocks off-corridor |

- Rocks and blocking ditches are kept out of a 0.4 m corridor around start → target. The planner owns avoidance; the policy owns local terrain (ADR 0006).
- Level schedule from config: either by timesteps (`[0, 0.15, 0.35, 0.6]` fractions) or adaptive (advance when rolling `success_rate` ≥ 0.8 over 200 episodes).
- Each reset samples the level as a mixture: 70 % current level, 30 % uniform over lower levels. This prevents forgetting flat ground.
- The callback calls `venv.env_method("set_level", L)`.

**Reward** is unchanged (settled). Episode length stays 30 s, targets 1–3 m.

**Network:** keep `[64, 64]` by default (it fits the ESP32 MLP export later). The config allows `[128, 128]`.

### Training infrastructure (built in B, used locally and on the cloud)
**`configs/*.toml`:**
- `m1_flat.toml`: reproduces the current 3M run.
- `terrain_v2.toml`: obs v2 + curriculum, 20M steps.
- `terrain_v2_smoke.toml`: 20k steps, 2 envs, used in tests.

Keys:
```toml
[run]  name = "terrain_v2"; steps = 20_000_000; n_envs = 16; seed = 0
[env]  obs_version = 2; world = "curriculum"; world_size = 8.0; cell = 0.05; randomizer_level = 1.0
[curriculum] mode = "timesteps"; boundaries = [0.0, 0.15, 0.35, 0.6]; replay_lower = 0.3
[ppo]  n_steps = 1024; batch_size = 4096; n_epochs = 10; gamma = 0.99; learning_rate = 3e-4; net_arch = [64, 64]; log_std_init = -1.0
[checkpoint] every_steps = 1_000_000; keep = 3
[eval] every_steps = 2_000_000; maps = 5; map_size = 10.0   # in-training nav eval (optional; off in smoke)
```

**`worm/training.py`:**
- `TrainConfig` dataclass plus `load_config(path) -> TrainConfig` (tomllib).
- `train(cfg, out, resume=False) -> Path`. The old `train(steps, n_envs, seed, out)` signature is kept as a wrapper so `test_training.py` and `test_cli.py` pass unchanged.
- Features:
  - TensorBoard to `out/tb`.
  - `CheckpointCallback` to `out/checkpoints/` with `save_vecnormalize=True`, keeping the last `keep`.
  - `CurriculumCallback`.
  - `NavEvalCallback`, which runs the C navigator on `maps` seeded worlds and logs `eval/nav_success` and per-class speed.
  - Writes `out/config.toml` (a copy), `out/meta.json` and `out/progress.json` (steps done, last checkpoint).
- `resume=True`: find the newest checkpoint, `PPO.load(ckpt, env=venv)` with `VecNormalize.load`, then `learn(remaining, reset_num_timesteps=False)`.
- SIGTERM/SIGINT handler: save a checkpoint, then exit 0. This makes spot preemption and `tmux kill` safe.
- Final save strips `system_info.txt` (existing helper) on every saved zip, including checkpoints.

**`scripts/train.py`:** adds `--config PATH`, `--resume`, and overrides `--steps --n-envs --seed --out`. Without `--config`, the behaviour is exactly as today.

**`scripts/eval_terrain.py --policy P --maps 50 --size 10 --levels 1,2,3 --seed 2000 --json out.json`:**
- Per map: generate a world, pick a random feasible start and goal ≥ 5 m apart (resample when infeasible), `navigate`.
- Report the overall success rate, per level, and the **per-terrain breakdown**:
  - time share per class;
  - mean speed per class;
  - the failure class (class under the head at failure);
  - replans per route.
- Also `--flat-routes 50` runs the existing 100 m route eval for the regression check.

**Cloud** (coordinator's requirements; Fable never provisions):
- **`scripts/cloud/bootstrap.sh`** (bash, `set -euo pipefail`, idempotent, fresh Ubuntu 22.04/24.04):
  1. `apt-get install -y git build-essential libgl1 libegl1 libosmesa6 tmux rsync`, using sudo only if it is not root;
  2. install uv via the official installer if it is missing;
  3. `uv sync --frozen`;
  4. `export MUJOCO_GL=egl`, falling back to osmesa (render tests skip when neither works);
  5. `make check`;
  6. print `nproc` and a 20 s `scripts/bench_env.py` result, so the operator can choose `n_envs`.
  - No credentials and no hard-coded repo URL: it runs inside an already cloned checkout, and `REPO_URL` is optional.
- **`scripts/cloud/launch.sh`** (and a Python core `scripts/launch_seeds.py` so it is testable): `launch.sh --config configs/terrain_v2.toml --seeds 0,1,2 --n-envs 15 [--tmux|--nohup] [--resume]`.
  - Per seed: `uv run python scripts/train.py --config ... --seed S --n-envs N --out runs/<name>_seed<S> [--resume]`, logging to `runs/<name>_seed<S>.log`.
  - It also starts `tensorboard --logdir runs --bind_all=false --port 6006` in its own tmux window or with nohup.
  - It writes `runs/<name>_launch.json` (pids, commands) and refuses to start a seed whose out dir has `progress.json` unless `--resume` is given.
  - **`scripts/cloud/status.sh`** tails the logs and prints `progress.json` steps.
  - **`scripts/cloud/stop.sh`** sends SIGTERM, which checkpoints and exits.
- **`docs/CLOUD.md`** runbook:
  - **VM spec:** Hetzner CCX63 (48 dedicated vCPU, 192 GB) or similar; any ≥ 32 dedicated-vCPU x86 box; no GPU needed. Fallback: AWS c7a.12xlarge spot (48 vCPU) with the SIGTERM checkpoint plus `--resume`.
  - **Steps:**
    1. Create the VM in the provider console (by hand, not by script);
    2. `ssh` in and clone the repo;
    3. `bash scripts/cloud/bootstrap.sh`;
    4. `bash scripts/cloud/launch.sh --config configs/terrain_v2.toml --seeds 0,1,2 --n-envs 15 --tmux`;
    5. monitor with `ssh -L 6006:localhost:6006` and TensorBoard;
    6. `status.sh`;
    7. after it finishes, `rsync -avz --include='*/' --include='policy.zip' --include='vecnormalize.pkl' --include='meta.json' --include='*.json' --include='*.log' --include='tb/***' --exclude='checkpoints/***' --exclude='*' <user>@<vm>:<repo>/runs/ runs/`, with placeholders only;
    8. run `eval_terrain.py` locally or on the VM;
    9. `stop.sh`;
    10. **delete the VM in the console. Billing continues until it is deleted, and a stopped VM still bills on Hetzner.**
  - Also: a cost estimate table, how to resume after preemption, and a checklist so no credentials go into the repo.
- **Auto-delete (user decision: laptop orchestrator plus a VM failsafe).** Billing stops only on **delete**, not on power-off. Add ADR `0007-cloud-auto-delete.md`.
  - **`scripts/cloud/run_remote.sh`** runs on the laptop and uses the `hcloud` CLI. Inputs come only from env vars and arguments: `HCLOUD_TOKEN` (from the env, never echoed), `--type ccx63`, `--location`, `--ssh-key NAME`, `--config`, `--seeds`, `--n-envs`, `--max-hours 12`.
    - Steps:
      1. `hcloud server create` with a unique name `worm-train-<timestamp>`, labelled `purpose=worm-train`;
      2. wait for SSH;
      3. rsync the working tree (`git ls-files`) to the VM;
      4. run `bootstrap.sh`;
      5. install the failsafe;
      6. run `launch.sh`;
      7. poll `status.sh` every 5 min until every seed's `progress.json` reaches its step target or the process has exited;
      8. rsync the results back (the include list in CLOUD.md);
      9. **verify locally** that each seed has `policy.zip`, `vecnormalize.pkl` and `meta.json`;
      10. only then `hcloud server delete`;
      11. confirm the delete with `hcloud server list -l purpose=worm-train`.
    - Failure handling:
      - If the rsync or the verify fails, retry 3 times, then **do not delete**. Print a loud warning. The failsafe still caps the cost.
      - A `trap` on EXIT/INT tells the user the VM name and whether it was deleted.
    - Flags: `--dry-run` prints every command. `--keep` skips the delete.
  - **Failsafe on the VM** (`scripts/cloud/failsafe.sh`, installed by `run_remote.sh`):
    - A systemd timer (or `nohup sleep N && ...`) fires after `--max-hours`.
    - It reads its own server ID from the Hetzner metadata service (`http://169.254.169.254/hetzner/v1/metadata/instance-id`).
    - It calls `DELETE https://api.hetzner.cloud/v1/servers/<id>` with the token from `/root/.hcloud_token`. That file is written by `run_remote.sh` over SSH with mode 600, never into the repo or the logs.
    - Before deleting, it SIGTERMs training so checkpoints save. The docs warn that results not rsynced by then are lost.
  - **Tests** (headless, no network):
    - a fake `hcloud` on `PATH` (a shell stub that logs its arguments);
    - `test_run_remote_dry_run_orders_create_launch_sync_verify_delete`;
    - `test_run_remote_skips_delete_when_verify_fails`;
    - `test_failsafe_builds_the_delete_request_from_metadata` (run with a fake `curl`);
    - `test_no_token_appears_in_any_log_or_committed_file`.
  - **CLOUD.md:**
    - Use a **separate Hetzner project** just for this, because a token reaches only its own project.
    - Create a read-write token and `export HCLOUD_TOKEN=...` in the shell only.
    - Run one command: `bash scripts/cloud/run_remote.sh --config configs/terrain_v2.toml --seeds 0,1,2 --n-envs 15`.
    - Keep the Mac awake with `caffeinate -i`.
    - Check the console afterwards that no server is left.
  - Fable never runs `run_remote.sh` without `--dry-run`. Rule 10 still holds: the user runs the real command.
- **`Dockerfile`** (optional, B5 if time allows): `python:3.11-slim` + apt GL libs + uv, `COPY` the repo, `uv sync --frozen`, `ENV MUJOCO_GL=egl`, entrypoint `scripts/train.py`. Test: none (the build needs Docker). Document it as optional.

### Files
- **Create:** `worm/curriculum.py`, `configs/{m1_flat,terrain_v2,terrain_v2_smoke}.toml`, `scripts/eval_terrain.py`, `scripts/launch_seeds.py`, `scripts/cloud/{bootstrap,launch,status,stop}.sh`, `docs/CLOUD.md`, `tests/test_curriculum.py`, `tests/test_obs_v2.py`, `tests/test_train_config.py`, `tests/test_launch.py`, ADRs 0005 and 0006, optional `Dockerfile` + `.dockerignore`.
- **Modify:** `worm/spec.py`, `worm/sensors.py` (`read_terrain(world, head pose, rng)`), `worm/env.py`, `worm/training.py`, `scripts/train.py`, `scripts/eval_route.py` (`--obs-version` from meta), `README.md`, `docs/PLAN.md` (obs section: add v2).

### Tasks
**B1. Spec versioning.**
- `test_spec.py::test_obs_v1_is_unchanged` (the existing test kept)
- `test_obs_v2_appends_terrain_ahead_and_height_ahead_and_totals_31`
- `test_unknown_obs_version_raises`

**B2. Obs v2 in sensors and env.**
- `test_obs_v2.py`:
  - `test_height_ahead_is_zero_on_flat_ground`
  - `test_height_ahead_sees_a_step_in_front` (a step 0.2 m ahead raises the far pair, not the near pair)
  - `test_height_ahead_is_in_the_head_frame` (rotate the worm 90°; the step is to the side, so the lateral asymmetry shows)
  - `test_terrain_ahead_reports_the_class_25cm_ahead`
  - `test_class_error_rate_is_about_5_percent` (2,000 reads, 3–7 %)
  - `test_v2_env_passes_the_checker`
  - `test_v1_env_still_24d_in_a_world`

**B3. Policy metadata and loading.**
- `test_training.py` additions:
  - `test_saved_run_has_meta_json_with_obs_version`
  - `test_load_policy_without_meta_defaults_to_v1` (load the committed `results/policy`)
  - `test_load_policy_v2_builds_a_31d_normalizer`
  - `test_meta_contains_no_machine_info` (no `/`, no hostname-like keys)

**B4. Curriculum.**
- `test_curriculum.py`:
  - `test_level0_is_flat_single_class`
  - `test_level1_is_flat_mixed`
  - `test_level2_slopes_are_bounded`
  - `test_corridor_to_target_is_obstacle_free` (sample target-aware: `WorldSampler` gets `(rng, start, target)`; env samples the target first, then the world)
  - `test_replay_lower_mixes_levels`
  - `test_sampler_keeps_the_world_shape`
  - `test_set_level_via_env_method` (DummyVecEnv)

**B5. Config-driven training, checkpoint and resume, TensorBoard.**
- `test_train_config.py`:
  - `test_load_config_parses_every_section_and_rejects_unknown_keys`
  - `test_smoke_config_trains_and_writes_tb_checkpoints_and_meta` (≤ 4k steps, `every_steps = 1024`)
  - `test_resume_continues_the_step_count` (train 2,048 steps; resume to 4,096; `progress.json` = 4,096, and `model.num_timesteps` continues)
  - `test_sigterm_saves_a_checkpoint` (subprocess, send SIGTERM after the first checkpoint, exit 0, newest checkpoint present)
  - `test_old_train_signature_still_works` (the existing test)
- Then `scripts/train.py --config`.
- Then the cloud scripts. `test_launch.py`:
  - `test_launch_seeds_builds_one_command_per_seed_with_its_own_out_and_log` (pure function `build_commands(cfg, seeds, n_envs, resume)`)
  - `test_launch_refuses_existing_run_without_resume`
  - `test_launch_nohup_mode_runs_the_smoke_config_for_two_seeds` (real subprocesses with the smoke config, 2 seeds × 1 env, waits, checks both logs and policies)
- `bash -n` on every `.sh` and `shellcheck` if installed. `bootstrap.sh` gets a `--dry-run` flag that prints its steps; the test runs it with `--dry-run` and asserts that `uv sync --frozen` and `make check` are in the output.
- Smoke-test `bootstrap.sh` for real only locally in the existing checkout, skipping apt on macOS (the script detects a non-Debian OS and skips apt with a message).
- `docs/CLOUD.md`.

**B6. Eval harness, smoke train, launch prep.**
- `eval_terrain.py` tests:
  - `test_eval_terrain_writes_overall_level_and_per_class_breakdown` (scripted controller, 2 maps, 4 m)
  - `test_eval_terrain_skips_infeasible_pairs_and_counts_them`
- Then compute (§8). Commit trained artifacts only from a full run: `results/policy_terrain/` (zip + pkl + meta) and `results/terrain_eval_*.json`.
- README results table with the per-terrain breakdown and a TensorBoard curve PNG exported via tensorboard's event reader into `mapviz`-style plain numpy drawing (or skip the PNG).
- Switch the demo fallback chain to prefer it (already coded in D1).
- PR. → **SP6 / SP7**.

### Success metrics (the B gate, recorded in the README even if missed)
All evaluated with `eval_terrain.py --maps 50 --size 10 --seed 2000` (fixed seeds), routes ≥ 5 m via the planner:
1. **Mixed flat (L1):** ≥ 90 % success.
2. **Full terrain (L3 config at 10 m with obstacles):** ≥ 80 % success.
3. **Beats the baseline:** improves on `results/policy` (v1) by ≥ 10 pp at L3, **or** has ≥ 15 % higher mean speed on SAND at equal success.
4. **Per-terrain breakdown reported:** success, mean speed per class, failure class histogram, replans per route.
5. **No regression:** flat 100 m route eval ≥ 48/50 (v2 baseline 50/50). Seeds 0, 1, 2 reported, as for M1.

### Risks and fallbacks
| Risk | Fallback |
|---|---|
| Training does not converge at L3 | Report L2 results. Try `[128, 128]`, 40M steps, or the adaptive curriculum |
| v2 is no better than v1 | Ship v2 anyway only if it meets gate 5. The README states the finding honestly. The demo keeps v1 |
| The world env is too slow to train | Training worlds of 6 m at 0.1 m cells; drop roughness at L1 |

---

## 8. Long training: estimates and procedure

**Grounding:** flat training runs at about 3,500–3,660 fps (8 envs, laptop CPU). Two concurrent runs total about 4,200 fps. The world env is expected at 40–70 % of flat, to be measured in A6 and B6. Planning figure: **about 2,000 fps per run locally.**

| Run | Steps | Laptop (1 run, 8 envs) | Laptop (3 seeds at once, about 1.4k fps each) | 48-vCPU VM (3 seeds × 15 envs, about 4–6k fps each, est.) |
|---|---|---|---|---|
| Smoke (`terrain_v2_smoke`) | 20k | < 1 min | – | – |
| Short sanity check (L0–L1 only) | 2M | about 17 min | – | – |
| `terrain_v2` (curriculum) | 20M | about 2.8 h | about 4 h | about 1–1.5 h |
| Extended (if L3 is not converged) | 40M | about 5.5 h | about 8 h | about 2–3 h |

VM cost: a 48-vCPU dedicated VM costs roughly a few dollars or less per hour. 3 seeds × 20M plus eval should cost a few dollars of rental. `docs/CLOUD.md` tells the operator to check current pricing, and the repo commits no prices beyond "order of magnitude".

**What Fable does:**
1. Always: the smoke config (in tests) plus one **2M-step local sanity run** in the background (`runs/terrain_v2_sanity_seed0.log`). Check that the TensorBoard `rollout/success_rate` rises above 0.5 at L0–L1. Record the fps in `PROGRESS.md` and ADR 0006 and recompute the table above with measured numbers.
2. If the session clearly has more than about 4 hours left and the laptop is idle: launch `scripts/cloud/launch.sh --config configs/terrain_v2.toml --seeds 0 --n-envs 8 --nohup` locally in the background. Checkpoints every 1M steps make it interruptible and resumable with `--resume`. Keep working on docs and eval while it runs.
3. Otherwise: **do not run full training.** Leave it ready to run:
   - the exact commands in the README "Train the terrain policy" section and in `docs/CLOUD.md`;
   - a `PROGRESS.md` Next item: "Run `terrain_v2` on cloud per `docs/CLOUD.md`, then `eval_terrain.py`, then commit `results/policy_terrain`".
   - The B PR is opened as "infrastructure + smoke results; full training pending", with the sanity-run curve numbers in the body.
4. Never commit checkpoints or `runs/` (already git-ignored). Commit only the final `policy.zip`, `vecnormalize.pkl` and `meta.json` (about 200 KB) and the eval JSON.

---

## 9. Time and compute summary
| Milestone | Agent time | Compute |
|---|---|---|
| Step 0 | 0.5–1 h | none |
| A | 8–12 h | about 10 min of benchmarks |
| C | 6–9 h | about 10 min (calibration, navigate checks) |
| D | 6–9 h | none |
| B infrastructure | 8–12 h | 2M sanity run (about 20–30 min, background) |
| B full training | – | about 4 h laptop for 3 seeds, or about 1.5 h on a 48-vCPU VM |

---

## 10. Per-PR checklist (Fable copies it into each PR body under "How tested")
- `make check` output: the ruff format, ruff check, mypy and pytest summary lines. Paste the last lines.
- New tests listed by name. The failing-first run is noted in the commits.
- Measurements (fps, eval numbers) with the commands that produced them.
- The privacy grep is clean.
- `Stacked on #N` line if applicable.
- "AI assistance": code and tests written by Claude Fable 5.2; what was run and checked.

---

### Critical Files for Implementation
- worm/env.py
- worm/friction.py
- worm/model.py
- worm/spec.py
- worm/training.py

Other files the plan builds on:
- worm/route.py
- worm/terrain.py
- worm/randomize.py
- worm/sensors.py
- scripts/train.py
- scripts/eval_route.py
- scripts/showcase.py
- tests/test_env.py
- tests/test_route.py
- docs/PLAN.md
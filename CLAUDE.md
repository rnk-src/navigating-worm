# Navigating Worm

Tier: portfolio
Last updated: 2026-11-14

## What this is
A servo-driven worm robot that learns a terrain-adaptive gait with RL in MuJoCo, then transfers it to hardware. M1, the simulation MVP, is complete. M2 is in progress: terrain worlds (M2a), the planner (M2b) and the click-to-go demo (M2c) are built; the terrain-aware policy (M2d) is trained and drives the demo by default.

## Commands
- Check (format + lint + types + tests): `make check`
- Train: `uv run python scripts/train.py --out runs/ppo`
- Evaluate on flat routes: `uv run python scripts/eval_route.py --policy results/policy`
- Draw a world map: `uv run python scripts/make_world.py --seed 1 --out map.png`
- Plan a route: `uv run python scripts/plan_map.py --seed 1 --goal 4 3.5 --out plan.png`
- Plan and drive: `uv run python scripts/navigate.py --policy results/policy --seed 1 --goal 4 3.5`
- Train from a config: `uv run python scripts/train.py --config configs/terrain_v2.toml --n-envs 8`
- Launch, watch, stop a detached run: `scripts/cloud/launch.sh --config configs/terrain_v2.toml --seeds 0 --n-envs 8`, `scripts/cloud/status.sh`, `scripts/cloud/stop.sh`
- Evaluate on terrain: `uv run python scripts/eval_terrain.py --policy results/policy --maps 50`

## Layout
- `docs/` is local only and listed in `.gitignore`. It is not in the public repo.
- `docs/PLAN.md` holds the architecture, module interfaces and milestones. Read it before you change a module.
- `docs/MVP_PLAN.md` is the execution plan for M2. Follow its task order and its rules.
- `docs/decisions/` holds one short ADR per decision. Add a new ADR for a new decision.
- `worm/spec.py` is the single source of truth for the observation and action layout.
- `worm/world.py`, `worm/planner.py` and `worm/mapviz.py` are pure numpy. Keep MuJoCo out of them.

## Gotchas
- This repo is public. Put no real name, email, absolute path or personal setup reference in any file.
- Never commit to `main`. Use a `feat/` or `fix/` branch and open a PR.
- Never flash firmware or actuate hardware without asking first.
- Write the failing test first. Watch it fail before you write the code.
- A positive gait bias turns the worm left. Joint signs are easy to get backwards.
- Tests must run headless. Video rendering is optional.
- The flat-plane env must stay bit-identical. `test_flat_env_matches_the_m1_golden_trajectory` guards it.
- World rows run along +y and columns along +x. Only `World.to_hfield_data` maps the grid to MuJoCo.
- Worlds use hybrid contact stiction, so the worm is faster on a world than on the plane. See ADR 0002.
- Build the env with the obs version of the policy. `worm.cli.policy_from_arg` returns both.
- A new obs version only appends fields. Never reorder the fields of v1.
- A world sampler gets `(rng, start, target)`. The env samples the target before the world.
- Never commit `runs/` or checkpoints. Commit only a final `policy.zip`, `vecnormalize.pkl` and `meta.json`.
- Soft contacts are unstable on an hfield. Keep the contact time constant at or below 0.04 s.

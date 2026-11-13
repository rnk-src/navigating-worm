# Navigating Worm

Tier: portfolio
Last updated: 2026-11-13

## What this is
A servo-driven worm robot that learns a terrain-adaptive gait with RL in MuJoCo, then transfers it to hardware. The current milestone is M1, the simulation MVP.

## Commands
- Check (format + lint + types + tests): `make check`

## Layout
- `docs/PLAN.md` holds the architecture, module interfaces and milestones. Read it before you change a module.
- `worm/spec.py` is the single source of truth for the observation and action layout.

## Gotchas
- This repo is public. Put no real name, email, absolute path or personal setup reference in any file.
- Never commit to `main`. Use a `feat/` or `fix/` branch and open a PR.
- Never flash firmware or actuate hardware without asking first.
- Tests must run headless. Video rendering is optional.

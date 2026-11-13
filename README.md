# Navigating Worm

A small, camera-equipped, servo-driven worm robot. It learns its gait with reinforcement learning in MuJoCo, transfers the gait to real hardware, and navigates toward a GPS coordinate.

The main idea is a **terrain-conditioned gait**: the camera classifies the surface and the policy adapts to it.

## Status
Milestone 1 (simulation MVP) is in progress. No results exist yet.

## Develop
Install [uv](https://docs.astral.sh/uv/), then run:

```
make check
```

This runs the format check, lint, type check and tests.

## Plan
See [docs/PLAN.md](docs/PLAN.md) for the architecture, milestones and bill of materials.

## License
MIT

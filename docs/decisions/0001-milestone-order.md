# 0001: Milestone order — world, planner, click-to-go, then the terrain policy

## Context
M1 ships a policy that follows waypoints on a flat plane. One commit after M1 samples a terrain class per episode. The old "M2" was one line: three friction regimes and a terrain one-hot. That is now done on flat ground.

The remaining work has four parts: a terrain world, a path planner, a click-to-go demo and a terrain-aware policy. The policy work is bound by compute and changes the observation layout. The other three parts need little compute.

## Decision
Split M2 into M2a World, M2b Planner, M2c Click-to-go and M2d Terrain-aware policy. Build them in that order.

- The demo path comes first. The existing policy already senses the class under its head and the slope through roll and pitch.
- The policy work needs the world generator, the per-cell physics and the planner. Its evaluation is "route success on mixed-terrain maps", which is the navigator.
- Every part ends at a state that can ship: a pull request, a green check and a runnable script.

Each later decision gets its own record in this folder.

## Consequences
- The demo may run with a policy that was never trained on slopes or mixed surfaces. Its success rate there is a finding, not a gate.
- The observation layout stays at version 1 until M2d. M2d adds versioning so old policies keep loading.
- The hardware milestones M3 and M4 are unchanged. The surface-photo CNN moves from M2 to M4.

# 0003: Planner and navigator

## Context
M2b turns a world and a goal into a route that the worm can drive, or a reason why no route exists. The policy follows waypoints but does not avoid obstacles. The planner must run headless and fast enough for a click in the demo.

## Decision
- **Grid.** Plan on a 0.1 m grid, the 5 cm world pooled in pairs: max for blocked, mean for cost.
- **Cost.** `surface_cost[class] * (1 + (slope / 20 deg)^2)`. A cell is blocked when its slope exceeds 20 degrees, when the height step to any of its 8 neighbours exceeds 12 mm, or when it is a rock.
- **Safety bands.** Blocked cells grow by 0.15 m. A 0.4 m band along the world edge is blocked, because the env ends an episode 0.3 m from the edge. The edge band is an addition to the plan.
- **Search.** A* with 8 neighbours and an octile heuristic. An edge costs the mean of its two cells times its length. A diagonal step needs both side cells free.
- **Smoothing.** Greedy string pulling. A shortcut is taken when its line is free and costs at most 1.05 times the stretch it replaces. No Theta*.
- **Waypoints.** Corners that turn more than 20 degrees stay. Corners closer than 0.3 m merge. Stretches split into even legs of at most 2 m. If dropping or merging a vertex would cross a blocked cell, the planner keeps every vertex.
- **Snapping.** A start in a blocked cell snaps to free ground within 0.3 m. A goal snaps within 0.2 m, and the plan reports it.
- **Reasons.** `OUT_OF_BOUNDS`, `START_BLOCKED`, `GOAL_BLOCKED`, `NO_PATH`.
- **Following.** `RouteFollower` advances one control step per call. Intermediate waypoints pass within 0.3 m; the goal needs the env's 0.15 m. `run_route` delegates to it with the env tolerance everywhere, and its trajectories are byte-identical to the old loop.
- **Replanning.** The `Navigator` replans from the head position when the best distance to the waypoint has not improved by 5 cm in 15 s, when the head is more than 0.6 m from the current leg, or when a waypoint takes more than 60 s. After 5 replans it fails as "stuck". A flip or leaving the world fails at once.

## Measurements
- Cost calibration, `scripts/calibrate_costs.py --policy results/policy` (5 targets of 2 m per class, level single-class worlds, randomized physics): concrete 0.172 m/s, grass 0.140 m/s, sand 0.085 m/s. Costs 1.00, 1.22, 2.03. The defaults are (1.0, 1.2, 2.0).
- A plan across a 10 m world searches about 3,000 cells and takes well under 1 s (test).
- Navigation with the committed policy on 10 default 10 m worlds (15 degree slopes, 10 rocks, 2 ditches), one random feasible goal at least 4 m away each: **10/10 reached, 0 replans**. The target was 7/10. The scripted `bearing_steer` controller also reached 10/10, about twice as slowly.

## Consequences
- The 12 mm step rule also blocks smooth slopes above about 10 to 13 degrees, because a 5 cm cell on such a slope rises more than 12 mm. The 20 degree slope rule therefore rarely decides. This is conservative and suits a policy that never trained on slopes. M2d may relax it.
- The walls of every generated ditch are blocked. Ditches act as obstacles unless they are very shallow.
- The planner uses ground truth: the world grid and the rock mask. The real robot will need a map source; that is the stretch goal in `docs/PLAN.md`.
- The navigator does not detect a blocked route in advance. It replans from the same map, so a worm that is physically stuck fails after about 90 s.

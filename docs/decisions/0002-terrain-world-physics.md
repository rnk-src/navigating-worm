# 0002: Terrain world physics

## Context
M2a needs ground with hills, rocks, ditches and a surface class per cell. The M1 friction model is a body-frame drag on a flat plane. MuJoCo's own anisotropic contact friction cannot follow the body frame (see `test_contact_adapter_anisotropy_depends_on_yaw`). The flat-plane env must stay bit-identical, because the committed policy and the M1 results depend on it.

## Decision
- **World.** `worm/world.py` holds a pure numpy `World`: heights, surface classes and a rock mask on a 5 cm grid, centred on the origin. Row index runs along +y, column index along +x. Only `World.to_hfield_data` maps the grid to MuJoCo. A ray test and a drop test confirm the orientation.
- **Model.** `model.build(spec, world)` compiles an `hfield` floor. Every hfield has a fixed 2 m height range, so `model.load_world` can swap in any world of the same shape without a recompile. `build(spec)` without a world returns the same XML as before (digest test).
- **Friction.** `TerrainResistive` generalizes `ResistiveForce`. Each physics step, each segment takes friction, viscous drag and contact softness from the cell under it. The drag acts in the terrain tangent plane, with load `m g n_z`. On a flat world with one surface its forces equal `ResistiveForce` to 1e-12 (test).
- **Slope stiction: hybrid mode, share 0.7.** A `tanh` drag cannot hold a body still. Worlds therefore put a share of the friction into the MuJoCo contact as isotropic Coulomb friction, `0.7 * mu_forward`. The drag supplies the rest, so the totals at speed are unchanged. The plan proposed a share of 0.5; the spike below shows that 0.5 is not enough.
- **Softness.** `apply` writes `pair_solref` per segment. It takes effect one physics step (2 ms) later. This lag is accepted.
- **Rendering.** The hfield carries one texel per cell with the class colour. `load_world` repaints it.

## Measurements
Idle worm on a 10 degree slope for 10 s, nominal friction (0.3 forward, 1.2 lateral). Creep in cm, facing uphill / across / downhill:

| Mode | Creep | Flat gait, cm in 10 s | Climb at 5 degrees |
|---|---|---|---|
| Pure drag | 32 / 7 / 32 | 52 | 22 |
| Hybrid, share 0.5 | 8.6 / 0.1 / 8.6 | 64 | 33 |
| Hybrid, share 0.7 | 0.1 / 0.1 / 0.1 | 72 | 38 |
| Hybrid, share 1.0 | 0.1 / 0.1 / 0.1 | 76 | 46 |

Sinkage of one segment by contact time constant: 0.03 s gives 0.17 mm; 0.085 s gives 1.0 mm with visible jitter at rest; 0.12 s falls through the hfield.

Throughput on one core, random actions: flat 806 control steps/s; 10 m world 651 (81 % of flat).

The committed policy, never trained on a world, on 10 m mixed-surface worlds without obstacles, ten 3 m targets each: 10/10 on level ground, 10/10 with slopes up to 10 degrees, 9/10 with slopes up to 15 degrees.

## Consequences
- **Threshold change.** The plan asked that sand sinks at least 1 mm deeper than concrete. The test asserts 0.1 mm. A contact soft enough for 1 mm is unstable on an hfield. Sand keeps the planned time constant range (0.02 to 0.04 s) and differs mainly through friction and viscous drag.
- The hybrid mode makes the open-loop gait about 40 % faster on a level world than on the plane, because Coulomb contact friction replaces part of the smooth drag at low speed. Worlds and the plane are therefore not the same dynamics. The policy still works (numbers above). M2d trains on worlds directly.
- With forward friction below `tan(slope) / 0.7`, a segment slides downhill. That is physical: the surface is too slippery to hold.
- The env ends an episode when the head comes within 0.3 m of the world edge (`info["out_of_world"]`).
- A renderer is rebuilt after a world swap, because the GPU copy of the hfield is stale. Training never renders.

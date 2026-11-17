// MuJoCo (WebAssembly) for one worm world, and the terrain-friction model that runs inside
// its substep loop. The friction port mirrors `TerrainResistive.apply` in worm/friction.py;
// tests/parity.test.ts keeps the two in step.
/* eslint-disable @typescript-eslint/no-explicit-any */

/** Everything the friction model needs. Python fills it from its `TerrainResistive`. */
export interface FrictionConfig {
  rows: number;
  cols: number;
  half: [number, number]; // half extent of the world in x and y, metres
  cell: number; // metres per grid cell
  normal: Float64Array; // rows * cols * 3 unit surface normals
  surface: Uint8Array; // rows * cols terrain classes
  bodyIds: Int32Array;
  pairIds: Int32Array;
  weight: Float64Array; // per segment, N
  mu: Float64Array; // 3 classes * [forward, lateral]
  viscous: Float64Array; // 3 classes
  timeconst: Float64Array | null; // 3 classes, or null to leave the contact softness alone
  hybrid: boolean;
  stiction: number;
  v0: number;
}

/** Round half to even, like numpy's `rint`. */
export function rint(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

export interface Dims {
  nq: number;
  nv: number;
  nu: number;
  nbody: number;
  nhfield: number;
  npair: number;
}

export class Sim {
  readonly model: any;
  readonly data: any;
  private friction: FrictionConfig | null = null;

  constructor(
    private readonly mj: any,
    xml: string,
  ) {
    this.model = mj.MjModel.from_xml_string(xml);
    this.data = new mj.MjData(this.model);
  }

  dims(): Dims {
    const m = this.model;
    return { nq: m.nq, nv: m.nv, nu: m.nu, nbody: m.nbody, nhfield: m.nhfield, npair: m.npair };
  }

  // ----- model and data fields, by name -----
  modelGet(name: string): ArrayLike<number> {
    return this.model[name];
  }
  modelSet(name: string, values: ArrayLike<number>): void {
    this.model[name].set(values);
  }
  bodyId(name: string): number {
    return this.model.body(name).id;
  }
  geomId(name: string): number {
    return this.model.geom(name).id;
  }
  siteId(name: string): number {
    return this.model.site(name).id;
  }

  // ----- the calls the Python mujoco module makes -----
  reset(): void {
    this.mj.mj_resetData(this.model, this.data);
  }
  forward(): void {
    this.mj.mj_forward(this.model, this.data);
  }
  kinematics(): void {
    this.mj.mj_kinematics(this.model, this.data);
  }
  setConst(): void {
    this.mj.mj_setConst(this.model, this.data);
  }
  step1(): void {
    this.mj.mj_step1(this.model, this.data);
  }
  step2(): void {
    this.mj.mj_step2(this.model, this.data);
  }

  /** Python writes these before a call: qpos, qvel, ctrl, xfrc_applied, packed in this order. */
  push(buf: ArrayLike<number>): void {
    const d = this.data;
    const { nq, nv, nu, nbody } = this.dims();
    let at = 0;
    d.qpos.set((buf as Float64Array).subarray(at, (at += nq)));
    d.qvel.set((buf as Float64Array).subarray(at, (at += nv)));
    d.ctrl.set((buf as Float64Array).subarray(at, (at += nu)));
    d.xfrc_applied.set((buf as Float64Array).subarray(at, at + 6 * nbody));
  }

  /** Python reads these after a call: qpos, qvel, ctrl, xpos, xmat, cvel, subtree_com. */
  pull(): Float64Array {
    const d = this.data;
    const parts = [d.qpos, d.qvel, d.ctrl, d.xpos, d.xmat, d.cvel, d.subtree_com];
    const out = new Float64Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }

  // ----- the fast path: a whole control step of substeps without leaving JavaScript -----
  setFriction(config: FrictionConfig | null): void {
    this.friction = config;
  }

  /** Move the servo targets from `start` to `applied` in `substeps` equal steps. */
  advance(start: Float64Array, applied: Float64Array, substeps: number): void {
    const f = this.friction;
    if (!f) throw new Error('advance needs a friction config');
    const ctrl = this.data.ctrl;
    for (let k = 0; k < substeps; k++) {
      const t = (k + 1) / substeps;
      for (let j = 0; j < ctrl.length; j++) ctrl[j] = start[j] + (applied[j] - start[j]) * t;
      this.mj.mj_step1(this.model, this.data);
      this.applyFriction(f);
      this.mj.mj_step2(this.model, this.data);
    }
  }

  private applyFriction(f: FrictionConfig): void {
    const d = this.data;
    const m = this.model;
    const xpos = d.xpos as Float64Array;
    const xmat = d.xmat as Float64Array;
    const cvel = d.cvel as Float64Array;
    const com = d.subtree_com as Float64Array;
    const rootid = m.body_rootid as Int32Array;
    const pairFriction = m.pair_friction as Float64Array;
    const pairSolref = m.pair_solref as Float64Array;
    const xfrc = d.xfrc_applied as Float64Array;
    const lastX = f.cols - 1;
    const lastY = f.rows - 1;

    for (let i = 0; i < f.bodyIds.length; i++) {
      const body = f.bodyIds[i];
      const px = xpos[3 * body];
      const py = xpos[3 * body + 1];
      const pz = xpos[3 * body + 2];
      const ix = Math.min(Math.max(rint((px + f.half[0]) / f.cell), 0), lastX);
      const iy = Math.min(Math.max(rint((py + f.half[1]) / f.cell), 0), lastY);
      const cellIndex = iy * f.cols + ix;
      const cls = f.surface[cellIndex];
      const nx = f.normal[3 * cellIndex];
      const ny = f.normal[3 * cellIndex + 1];
      const nz = f.normal[3 * cellIndex + 2];
      let mu0 = f.mu[2 * cls];
      let mu1 = f.mu[2 * cls + 1];
      const pair = f.pairIds[i];
      if (f.hybrid) {
        const contact = f.stiction * mu0;
        pairFriction[5 * pair] = contact;
        pairFriction[5 * pair + 1] = contact;
        mu0 -= contact;
        mu1 -= contact;
      }
      if (f.timeconst) pairSolref[2 * pair] = f.timeconst[cls];

      // Velocity of the body origin: cvel is [angular, linear] at the subtree centre of mass.
      const root = rootid[body];
      const ox = px - com[3 * root];
      const oy = py - com[3 * root + 1];
      const oz = pz - com[3 * root + 2];
      const wx = cvel[6 * body];
      const wy = cvel[6 * body + 1];
      const wz = cvel[6 * body + 2];
      const vx = cvel[6 * body + 3] + (wy * oz - wz * oy);
      const vy = cvel[6 * body + 4] + (wz * ox - wx * oz);
      const vz = cvel[6 * body + 5] + (wx * oy - wy * ox);

      const load = f.weight[i] * nz; // the weight arrives already multiplied by g
      let fx = 0;
      let fy = 0;
      let fz = 0;
      for (let axis = 0; axis < 2; axis++) {
        // Column `axis` of the body's rotation matrix, projected onto the surface plane.
        let dx = xmat[9 * body + axis];
        let dy = xmat[9 * body + 3 + axis];
        let dz = xmat[9 * body + 6 + axis];
        const dn = dx * nx + dy * ny + dz * nz;
        dx -= dn * nx;
        dy -= dn * ny;
        dz -= dn * nz;
        const norm = Math.hypot(dx, dy, dz) + 1e-9;
        dx /= norm;
        dy /= norm;
        dz /= norm;
        const speed = vx * dx + vy * dy + vz * dz;
        const magnitude = load * (axis === 0 ? mu0 : mu1) * Math.tanh(speed / f.v0);
        fx -= magnitude * dx;
        fy -= magnitude * dy;
        fz -= magnitude * dz;
      }
      const vn = vx * nx + vy * ny + vz * nz;
      const visc = f.viscous[cls];
      fx -= visc * (vx - vn * nx);
      fy -= visc * (vy - vn * ny);
      fz -= visc * (vz - vn * nz);
      xfrc[6 * body] = fx;
      xfrc[6 * body + 1] = fy;
      xfrc[6 * body + 2] = fz;
    }
  }

  free(): void {
    this.data.delete();
    this.model.delete();
  }
}

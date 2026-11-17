// Boots the worm's Python (Pyodide) on top of MuJoCo (WebAssembly) and wraps it in typed calls.
/* eslint-disable @typescript-eslint/no-explicit-any */
import { Sim } from './physics';

export interface BootOptions {
  mj: any; // the loaded @mujoco/mujoco module
  pyodide: any; // a loaded Pyodide
  wheel: Uint8Array; // the built `worm` wheel
  mujocoPy: string; // text of py/mujoco.py
  gymnasiumPy: string; // text of py/gymnasium.py
  browserPy: string; // text of py/browser.py
  policy: string; // text of policy.json
  racers?: string; // JSON [{label, policy}]; set, it boots a race and `policy` is unused
  onProgress?: (message: string) => void;
}

export interface MapRequest {
  seed: number;
  size: number;
  mix: [number, number, number];
  slope: number;
  rocks: number;
  ditches: number;
}

export interface MapInfo {
  map_id: number;
  extent: [number, number, number, number];
  px_per_m: number;
  width: number;
  height: number;
}

export type GoalAnswer =
  | { ok: true; plan: { path: number[][]; waypoints: number[][]; cost: number; goal_snapped: boolean } }
  | { ok: false; reason?: string; detail?: string; error?: string };

export interface Snapshot {
  map_id: number;
  policy: string;
  status: 'idle' | 'driving' | 'reached' | 'failed';
  failure: string | null;
  head: number[];
  yaw: number;
  body: number[][];
  trail: number[][];
  path: number[][];
  waypoints: number[][];
  reached: number;
  goal: number[] | null;
  replans: number;
  surface: string;
  sim_time: number;
  bodies?: RaceBody[]; // a race only; the first is the leader above
  race_over?: boolean;
}

export interface RaceBody {
  label: string;
  segs: number[][];
  yaw: number;
  trail: number[][];
  done: boolean;
  t_finish: number | null; // sim seconds; null until it arrives, and for a DNF
}

export class Engine {
  /** The physics of the current map, for a 3D view to read. */
  sims: Sim[] = [];

  private constructor(
    private readonly demo: any,
    readonly controlDt: number,
  ) {}

  static async create(o: BootOptions): Promise<Engine> {
    const say = o.onProgress ?? (() => {});
    const py = o.pyodide;
    say('Loading numpy');
    await py.loadPackage('numpy');
    say('Loading the worm code');
    py.unpackArchive(o.wheel, 'wheel');
    py.FS.mkdirTree('/shim');
    py.FS.writeFile('/shim/mujoco.py', o.mujocoPy);
    py.FS.writeFile('/shim/gymnasium.py', o.gymnasiumPy);
    py.runPython("import sys\nif '/shim' not in sys.path: sys.path.insert(0, '/shim')");

    const engine: { current?: Engine } = {};
    const makeSim = (xml: string): Sim => {
      const sim = new Sim(o.mj, xml);
      engine.current?.remember(sim);
      return sim;
    };
    const mujoco = py.pyimport('mujoco');
    mujoco._bind(makeSim);
    mujoco.destroy();

    say('Starting the simulation');
    py.runPython(o.browserPy);
    const make = py.globals.get(o.racers ? 'make_race' : 'make_demo');
    const demo = make(o.racers ?? o.policy);
    make.destroy();
    const dt = demo.control_dt as number;
    const created = new Engine(demo, dt);
    engine.current = created;
    return created;
  }

  private remember(sim: Sim): void {
    this.sims.push(sim);
  }

  /** Id of the current map, kept here so callers need no Python call per frame. -1 before the first map. */
  mapId = -1;

  newMap(request: MapRequest): { ok: boolean; map_id?: number; error?: string } {
    const previous = this.sims;
    this.sims = []; // the new map's sims register themselves during the call
    const result = JSON.parse(this.demo.new_map(JSON.stringify(request)));
    if (result.ok) this.mapId = result.map_id;
    else this.sims = previous; // a refused map leaves the old one running
    return result;
  }
  mapInfo(): MapInfo {
    return JSON.parse(this.demo.map_info());
  }
  mapRgb(): Uint8Array {
    return this.demo.map_rgb() as Uint8Array;
  }
  goal(x: number, y: number): GoalAnswer {
    return JSON.parse(this.demo.goal(x, y));
  }
  tick(n: number): void {
    this.demo.tick(n);
  }
  snapshot(): Snapshot {
    return JSON.parse(this.demo.snapshot());
  }
}

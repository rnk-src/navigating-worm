// The data source the page talks to (window.wormApi). It owns the pacing loop that turns wall
// time into control steps, and it reports start-up progress to the start-up panel.
import type { Engine, GoalAnswer, MapInfo, MapRequest, Snapshot } from './engine';

export type EngineLike = Pick<Engine, 'controlDt' | 'newMap' | 'mapInfo' | 'mapRgb' | 'goal' | 'tick' | 'snapshot'>;

export const DEFAULT_MAP: MapRequest = { seed: 0, size: 10, mix: [0.4, 0.3, 0.3], slope: 12, rocks: 12, ditches: 2 };

const MAX_STEPS_PER_CHUNK = 8;
const FRAME_BUDGET_MS = 10;

export interface BootState {
  state: 'loading' | 'ready' | 'error';
  message: string;
}

export interface ApiOptions {
  now?: () => number; // milliseconds
  makeImageUrl?: (rgb: Uint8Array, info: MapInfo) => string;
}

/** A blob URL for the map picture, drawn on a canvas. */
function canvasImageUrl(rgb: Uint8Array, info: MapInfo): string {
  const canvas = document.createElement('canvas');
  canvas.width = info.width;
  canvas.height = info.height;
  const ctx = canvas.getContext('2d')!;
  const image = ctx.createImageData(info.width, info.height);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    image.data[j] = rgb[i];
    image.data[j + 1] = rgb[i + 1];
    image.data[j + 2] = rgb[i + 2];
    image.data[j + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL('image/png'); // a data URL needs no cleanup
}

export class BrowserApi {
  boot = {
    state: 'loading' as BootState['state'],
    message: 'Loading',
    subscribe: (fn: (b: BootState) => void): void => {
      this.listeners.push(fn);
      fn(this.boot);
    },
  };
  readonly ready: Promise<void>;
  engine: EngineLike | null = null;
  camera?: { mount(container: HTMLElement): void; unmount(): void };

  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private listeners: Array<(b: BootState) => void> = [];
  private userPaused = false;
  private hostPaused = false;
  private speed = 1;
  private acc = 0;
  private readonly now: () => number;
  private readonly makeImageUrl: (rgb: Uint8Array, info: MapInfo) => string;

  constructor(options: ApiOptions = {}) {
    this.now = options.now ?? (() => performance.now());
    this.makeImageUrl = options.makeImageUrl ?? canvasImageUrl;
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.ready.catch(() => {}); // the start-up panel reports the error; no unhandled rejection
  }

  private setBoot(state: BootState['state'], message: string): void {
    this.boot.state = state;
    this.boot.message = message;
    for (const fn of this.listeners) fn(this.boot);
  }
  progress(message: string): void {
    this.setBoot('loading', message);
  }
  fail(message: string): void {
    this.setBoot('error', message);
    this.rejectReady(new Error(message));
  }

  /** True while the host or the user has paused the simulation. */
  get paused(): boolean {
    return this.userPaused || this.hostPaused;
  }

  /** Hands over the booted engine, starts on the default map and opens the page. */
  attach(engine: EngineLike): void {
    this.engine = engine;
    engine.newMap(DEFAULT_MAP);
    this.setBoot('ready', 'Ready');
    this.resolveReady();
  }

  /** Advance by `dt` seconds of wall time. Drops any backlog the budget could not cover. */
  frame(dt: number): void {
    const engine = this.engine;
    if (!engine || this.userPaused || this.hostPaused) return;
    this.acc += (dt * this.speed) / engine.controlDt;
    const start = this.now();
    while (this.acc >= 1) {
      const n = Math.min(Math.floor(this.acc), MAX_STEPS_PER_CHUNK);
      engine.tick(n);
      this.acc -= n;
      if (this.now() - start > FRAME_BUDGET_MS) break;
    }
    if (this.acc >= 1) this.acc %= 1; // drop the backlog, keep the fraction
  }

  /** Start the requestAnimationFrame loop. */
  run(): void {
    let last = this.now();
    const loop = (t: number): void => {
      const dt = Math.min((t - last) / 1000, 0.25);
      last = t;
      this.frame(dt);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  /** The portfolio sends pause and resume when the frame scrolls off screen or back. */
  hostMessage(data: unknown): void {
    const kind = typeof data === 'string' ? data : (data as { type?: string } | null)?.type;
    if (kind === 'pause') this.hostPaused = true;
    else if (kind === 'resume') this.hostPaused = false;
  }

  private need(): EngineLike {
    if (!this.engine) throw new Error('The simulation is not ready');
    return this.engine;
  }

  async mapInfo(): Promise<MapInfo> {
    return this.need().mapInfo();
  }
  mapImageUrl(_id: number): string {
    const engine = this.need();
    return this.makeImageUrl(engine.mapRgb(), engine.mapInfo());
  }
  async state(): Promise<Snapshot & { paused: boolean; speed: number }> {
    return { ...this.need().snapshot(), paused: this.userPaused, speed: this.speed };
  }
  async goal(x: number, y: number): Promise<GoalAnswer> {
    return this.need().goal(x, y);
  }
  async newMap(request: MapRequest): Promise<{ ok: boolean; map_id?: number; error?: string }> {
    const engine = this.need();
    this.acc = 0;
    return engine.newMap(request);
  }
  async control(body: { pause?: boolean; speed?: number }): Promise<{ ok: true }> {
    if (body.pause !== undefined) this.userPaused = body.pause;
    if (body.speed !== undefined) this.speed = body.speed;
    return { ok: true };
  }
}

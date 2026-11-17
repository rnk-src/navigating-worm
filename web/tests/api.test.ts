import { expect, test } from 'vitest';
import { BrowserApi, DEFAULT_MAP, type EngineLike } from '../src/api';

function fakeEngine(controlDt = 0.05) {
  const calls: number[] = [];
  const maps: unknown[] = [];
  const engine: EngineLike = {
    controlDt,
    newMap: (r) => (maps.push(r), { ok: true, map_id: maps.length }),
    mapInfo: () => ({ map_id: maps.length, extent: [0, 0, 1, 1], px_per_m: 1, width: 1, height: 1 }),
    mapRgb: () => new Uint8Array(3),
    goal: () => ({ ok: false, reason: 'blocked' }),
    tick: (n) => void calls.push(n),
    snapshot: () => ({ map_id: maps.length, status: 'idle', sim_time: 0 }) as ReturnType<EngineLike['snapshot']>,
  };
  return { engine, calls, maps };
}

function makeApi(opts: { step?: number } = {}) {
  let t = 0;
  const api = new BrowserApi({ now: () => (t += opts.step ?? 0), makeImageUrl: () => 'blob:map' });
  return api;
}

test('attach starts on the default map and resolves ready', async () => {
  const { engine, maps } = fakeEngine();
  const api = makeApi();
  api.attach(engine);
  await api.ready;
  expect(maps[0]).toEqual(DEFAULT_MAP);
  expect(api.boot.state).toBe('ready');
});

test('one frame at 1x runs one control step per control period', () => {
  const { engine, calls } = fakeEngine(0.05);
  const api = makeApi();
  api.attach(engine);
  api.frame(0.1); // 0.1 s of wall time at 1x = 2 control steps
  expect(calls.reduce((a, b) => a + b, 0)).toBe(2);
});

test('speed multiplies the steps and a frame never runs more than the budget allows', () => {
  const { engine, calls } = fakeEngine(0.05);
  const api = makeApi({ step: 6 }); // each clock read costs 6 ms, so the 10 ms budget allows two chunks
  api.attach(engine);
  void api.control({ speed: 16 });
  api.frame(0.5); // 160 steps wanted
  expect(Math.max(...calls)).toBeLessThanOrEqual(8);
  expect(calls.length).toBeLessThan(20);
});

test('backlog is dropped, not carried to the next frame', () => {
  const { engine, calls } = fakeEngine(0.05);
  const api = makeApi({ step: 6 });
  api.attach(engine);
  void api.control({ speed: 16 });
  api.frame(0.5);
  calls.length = 0;
  api.frame(0.001); // 0.016 steps
  expect(calls).toEqual([]);
});

test('a paused api runs no steps', () => {
  const { engine, calls } = fakeEngine();
  const api = makeApi();
  api.attach(engine);
  void api.control({ pause: true });
  api.frame(1);
  expect(calls).toEqual([]);
});

test('state adds paused and speed to the snapshot', async () => {
  const { engine } = fakeEngine();
  const api = makeApi();
  api.attach(engine);
  await api.control({ pause: true, speed: 4 });
  expect(await api.state()).toMatchObject({ status: 'idle', paused: true, speed: 4 });
});

test('the host pause and resume messages pause and resume, even before attach', () => {
  const { engine, calls } = fakeEngine();
  const api = makeApi();
  api.hostMessage('pause');
  api.attach(engine);
  api.frame(1);
  expect(calls).toEqual([]);
  api.hostMessage({ type: 'resume' });
  api.frame(0.1);
  expect(calls.length).toBeGreaterThan(0);
});

test('a user pause survives a host resume', () => {
  const { engine, calls } = fakeEngine();
  const api = makeApi();
  api.attach(engine);
  void api.control({ pause: true });
  api.hostMessage('pause');
  api.hostMessage('resume');
  api.frame(1);
  expect(calls).toEqual([]);
});

test('boot reports progress to subscribers and then the error', () => {
  const api = makeApi();
  const seen: string[] = [];
  api.boot.subscribe((b) => seen.push(`${b.state}:${b.message}`));
  api.progress('Loading numpy');
  api.fail('no network');
  expect(seen.at(-2)).toBe('loading:Loading numpy');
  expect(seen.at(-1)).toBe('error:no network');
  expect(api.boot.state).toBe('error');
});

test('newMap passes the request through and mapImageUrl returns the blob url', async () => {
  const { engine, maps } = fakeEngine();
  const api = makeApi();
  api.attach(engine);
  const req = { ...DEFAULT_MAP, seed: 7 };
  expect(await api.newMap(req)).toMatchObject({ ok: true });
  expect(maps.at(-1)).toEqual(req);
  expect(api.mapImageUrl(1)).toBe('blob:map');
});

test('a failed start rejects ready', async () => {
  const api = makeApi();
  api.fail('CDN blocked');
  await expect(api.ready).rejects.toThrow('CDN blocked');
});

test('paused reports a pause from the host or the user', () => {
  const { engine } = fakeEngine();
  const api = makeApi();
  api.attach(engine);
  expect(api.paused).toBe(false);
  api.hostMessage('pause');
  expect(api.paused).toBe(true);
  api.hostMessage('resume');
  expect(api.paused).toBe(false);
});

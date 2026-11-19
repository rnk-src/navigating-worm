import fs from 'node:fs';
import path from 'node:path';
import { beforeAll, expect, test } from 'vitest';
import type { Engine } from '../src/engine';
import { bootEngine, buildWheel, cdnReachable, root } from './helpers';

const here = path.dirname(new URL(import.meta.url).pathname);
const native = JSON.parse(fs.readFileSync(path.join(here, 'fixtures/native-run.json'), 'utf8'));
let engine: Engine;
let wheel: Uint8Array;
let online = false;

beforeAll(async () => {
  online = await cdnReachable();
  if (!online) return;
  wheel = buildWheel();
  engine = await bootEngine(wheel);
}, 240_000);

test('the browser engine drives the same map and goal as the native run', { timeout: 240_000 }, async (ctx) => {
  if (!online) return ctx.skip();
  expect(engine.newMap(native.request)).toMatchObject({ ok: true });
  const answer = engine.goal(native.goal[0], native.goal[1]);
  expect(answer.ok).toBe(true);
  if (!answer.ok) return;
  expect(answer.plan.waypoints.length).toBe(native.n_waypoints);
  expect(answer.plan.path[0]).toEqual(native.path_start);
  expect(answer.plan.path[answer.plan.path.length - 1]).toEqual(native.path_end);

  let done = 0;
  for (const row of native.run) {
    engine.tick(row.step - done);
    done = row.step;
    const snap = engine.snapshot();
    const miss = Math.hypot(snap.head[0] - row.head[0], snap.head[1] - row.head[1]);
    console.log(`step ${row.step}: browser ${snap.head} native ${row.head} miss ${miss.toFixed(3)} m, ${snap.status}/${row.status}, reached ${snap.reached}/${row.reached}`);
    expect(miss).toBeLessThan(0.1);
  }
});

test('a refused map keeps the current map, its sims and its id', { timeout: 240_000 }, async (ctx) => {
  if (!online) return ctx.skip();
  const first = engine.newMap(native.request);
  expect(first.ok).toBe(true);
  const sims = engine.sims.length;
  expect(engine.mapId).toBe(first.map_id);
  const refused = engine.newMap({ ...native.request, mix: [0, 0, 0] });
  expect(refused.ok).toBe(false);
  expect(engine.sims.length).toBe(sims);
  expect(engine.mapId).toBe(first.map_id);
});

test('a race engine steps every racer on one shared route', { timeout: 240_000 }, async (ctx) => {
  if (!online) return ctx.skip();
  const policy = (dir: string) => JSON.parse(fs.readFileSync(path.join(root, dir, 'policy.json'), 'utf8'));
  const race = await bootEngine(wheel, [
    { label: 'final', policy: policy('results/policy_terrain') },
    { label: 'flat', policy: policy('results/policy') },
    { label: 'scripted', policy: 'scripted' },
  ]);
  expect(race.newMap(native.request)).toMatchObject({ ok: true });
  expect(race.sims.length).toBe(3);
  expect(race.goal(native.goal[0], native.goal[1]).ok).toBe(true);
  race.tick(40);
  const snap = race.snapshot();
  expect(snap.policy).toBe('final');
  expect(snap.bodies?.map((b) => b.label)).toEqual(['final', 'flat', 'scripted']);
  expect(snap.bodies?.every((b) => b.segs.length === snap.body.length && !b.done)).toBe(true);
  race.setRacing(false);
  race.tick(5);
  expect(race.snapshot().bodies?.map((b) => b.label)).toEqual(['final']);
});

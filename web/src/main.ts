// Browser entry. The page reads window.wormApi, so set it first and synchronously. The heavy
// libraries load afterwards, and the start-up panel shows the progress.
import { BrowserApi } from './api';
import { createFollowCam } from './camera';
import { Engine } from './engine';

declare const __PYODIDE_VERSION__: string;
declare global {
  interface Window {
    wormApi: BrowserApi;
  }
}

const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${__PYODIDE_VERSION__}/full/`;

const api = new BrowserApi();
let engine: Engine | null = null;
// The page checks api.camera once at start-up, so set it now. three.js loads when the cam opens.
api.camera = createFollowCam(api, () => engine);
window.wormApi = api;
// Only the page that frames this demo may pause it.
window.addEventListener('message', (e) => {
  if (e.source === window.parent) api.hostMessage(e.data);
});

async function bytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}
async function text(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.text();
}

// The race, leader first. The leader is the final policy, which is also the step 20M checkpoint.
// The page styles each racer by its label.
const GHOSTS: Array<[label: string, file: string]> = [
  ['9M steps', 'py/race/step_9000000.json'],
  ['4M steps', 'py/race/step_4000000.json'],
  ['1M steps', 'py/race/step_1000000.json'],
  ['Untrained', 'py/race/step_0.json'],
  ['Flat policy', 'py/race/flat.json'],
];

async function racers(final: string): Promise<string> {
  const ghosts = await Promise.all(GHOSTS.map(async ([label, file]) => ({ label, policy: JSON.parse(await text(file)) })));
  return JSON.stringify([
    { label: 'Final', policy: JSON.parse(final) },
    ...ghosts,
    { label: 'Scripted gait', policy: 'scripted' },
  ]);
}

async function boot(): Promise<void> {
  api.progress('Loading the physics engine and Python');
  const [{ default: loadMujoco }, { loadPyodide }] = await Promise.all([import('@mujoco/mujoco'), import('pyodide')]);
  const [mj, pyodide, wheel, mujocoPy, gymnasiumPy, browserPy, policy] = await Promise.all([
    loadMujoco(),
    loadPyodide({ indexURL: PYODIDE_URL }),
    bytes('py/worm.whl'),
    text('py/mujoco.py'),
    text('py/gymnasium.py'),
    text('py/browser.py'),
    text('py/policy.json').then(async (final) => ({ final, race: await racers(final) })),
  ]);
  engine = await Engine.create({
    mj,
    pyodide,
    wheel,
    mujocoPy,
    gymnasiumPy,
    browserPy,
    policy: policy.final,
    racers: policy.race,
    onProgress: (m) => api.progress(m),
  });
  api.attach(engine);
  api.run();
}

boot().catch((e: unknown) => api.fail(e instanceof Error ? e.message : String(e)));

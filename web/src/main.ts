// Browser entry. The page reads window.wormApi, so set it first and synchronously. The heavy
// libraries load afterwards, and the start-up panel shows the progress.
import { BrowserApi } from './api';
import { Engine } from './engine';

declare const __PYODIDE_VERSION__: string;
declare global {
  interface Window {
    wormApi: BrowserApi;
  }
}

const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${__PYODIDE_VERSION__}/full/`;

const api = new BrowserApi();
window.wormApi = api;
window.addEventListener('message', (e) => api.hostMessage(e.data));

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
    text('py/policy.json'),
  ]);
  const engine = await Engine.create({
    mj,
    pyodide,
    wheel,
    mujocoPy,
    gymnasiumPy,
    browserPy,
    policy,
    onProgress: (m) => api.progress(m),
  });
  api.attach(engine);
  api.run();
}

boot().catch((e: unknown) => api.fail(e instanceof Error ? e.message : String(e)));

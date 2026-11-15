import { execSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import load_mujoco from '@mujoco/mujoco';
import { loadPyodide } from 'pyodide';
import { Engine } from '../src/engine';

const here = path.dirname(new URL(import.meta.url).pathname);
export const root = path.join(here, '../..');
const pyodideVersion = JSON.parse(fs.readFileSync(path.join(here, '../node_modules/pyodide/package.json'), 'utf8')).version;
export const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/`;

/** Pyodide fetches numpy from its CDN, so the engine tests need the network. */
export async function cdnReachable(): Promise<boolean> {
  try {
    const r = await fetch(`${PYODIDE_CDN}pyodide-lock.json`, { method: 'HEAD', signal: AbortSignal.timeout(5000) });
    return r.ok;
  } catch {
    return false;
  }
}

export function buildWheel(): Uint8Array {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'worm-wheel-'));
  execSync(`uv build --wheel --out-dir ${out}`, { cwd: root, stdio: 'pipe' });
  const file = fs.readdirSync(out).find((f) => f.endsWith('.whl'))!;
  return new Uint8Array(fs.readFileSync(path.join(out, file)));
}

export async function bootEngine(wheel: Uint8Array): Promise<Engine> {
  const mj = await load_mujoco();
  // The core comes from the npm package. Only the numpy wheel comes from the CDN, at the same version.
  const pyodide = await loadPyodide({ indexURL: path.join(here, '../node_modules/pyodide') });
  const lock = await (await fetch(`${PYODIDE_CDN}pyodide-lock.json`)).json();
  await pyodide.loadPackage(PYODIDE_CDN + lock.packages.numpy.file_name);
  const py = (f: string) => fs.readFileSync(path.join(here, '../py', f), 'utf8');
  return Engine.create({
    mj,
    pyodide,
    wheel,
    mujocoPy: py('mujoco.py'),
    gymnasiumPy: py('gymnasium.py'),
    browserPy: py('browser.py'),
    policy: fs.readFileSync(path.join(root, 'results/policy_terrain/policy.json'), 'utf8'),
  });
}

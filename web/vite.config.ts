import fs from 'node:fs';
import { defineConfig } from 'vite';

// The Pyodide files come from jsDelivr. The URL must match the installed npm version.
const pyodideVersion = JSON.parse(fs.readFileSync('node_modules/pyodide/package.json', 'utf8')).version;

// base './' lets the built app live in any subfolder, e.g. /demos/navigating-worm/.
export default defineConfig({
  base: './',
  define: { __PYODIDE_VERSION__: JSON.stringify(pyodideVersion) },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
});

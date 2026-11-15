import { defineConfig } from 'vite';

// base './' lets the built app live in any subfolder, e.g. /demos/navigating-worm/.
export default defineConfig({
  base: './',
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
  optimizeDeps: { exclude: ['@mujoco/mujoco'] },
});

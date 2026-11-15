// The follow cam: a three.js view of the heightfield, the worm's boxes and the goal. three.js
// loads only when the cam is first switched on.
/* eslint-disable @typescript-eslint/no-explicit-any */
import type { BrowserApi } from './api';
import { followPose, terrainGrid } from './camera-math';
import type { Engine } from './engine';

const BACK_M = 0.45;
const UP_M = 0.3;
const GOAL_REFRESH_MS = 250;

export function createFollowCam(api: BrowserApi, getEngine: () => Engine | null) {
  let stop: (() => void) | null = null;

  async function start(container: HTMLElement): Promise<() => void> {
    const THREE = await import('three');
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.domElement.style.cssText = 'display:block;width:100%;aspect-ratio:4/3;background:#07060f';
    renderer.domElement.setAttribute('role', 'img');
    renderer.domElement.setAttribute('aria-label', 'Follow camera view of the worm');
    container.append(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07060f);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(-1, -2, 3);
    scene.add(sun);
    const camera = new THREE.PerspectiveCamera(60, 4 / 3, 0.02, 60);
    camera.up.set(0, 0, 1);

    const goal = new THREE.Mesh(new THREE.SphereGeometry(0.05), new THREE.MeshBasicMaterial({ color: 0xe5484d }));
    goal.visible = false;
    scene.add(goal);

    let terrain: any = null;
    let segs: any[] = [];
    let mapId = -1;
    let sim: any = null;

    function disposeWorld(): void {
      for (const o of [terrain, ...segs]) {
        if (!o) continue;
        scene.remove(o);
        o.geometry.dispose();
        o.material.map?.dispose();
        o.material.dispose();
      }
      terrain = null;
      segs = [];
    }

    async function rebuild(engine: Engine): Promise<void> {
      disposeWorld();
      sim = [...engine.sims].reverse().find((s) => s.dims().nhfield === 1) ?? null;
      if (!sim) return;
      const info = engine.mapInfo();
      const m = sim.model;
      const rows = m.hfield_nrow[0];
      const cols = m.hfield_ncol[0];
      const grid = terrainGrid(m.hfield_data.slice(0, rows * cols), rows, cols, info.extent, m.hfield_size[2]);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(grid.positions, 3));
      geometry.setAttribute('uv', new THREE.BufferAttribute(grid.uvs, 2));
      geometry.setIndex(new THREE.BufferAttribute(grid.indices, 1));
      geometry.computeVertexNormals();
      const texture = await new THREE.TextureLoader().loadAsync(api.mapImageUrl(info.map_id));
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.magFilter = THREE.NearestFilter;
      terrain = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ map: texture }));
      scene.add(terrain);
      for (let i = 0; ; i++) {
        let id: number;
        try {
          id = sim.geomId(`seg${i}`);
        } catch {
          break;
        }
        const s = m.geom_size.slice(id * 3, id * 3 + 3);
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]),
          new THREE.MeshLambertMaterial({ color: i === 0 ? 0xb8a2ff : 0x8a72e0 }),
        );
        mesh.userData.body = sim.bodyId(`seg${i}`);
        segs.push(mesh);
        scene.add(mesh);
      }
      mapId = info.map_id;
    }

    let lastGoal = 0;
    let raf = 0;
    let alive = true;
    function frame(t: number): void {
      if (!alive) return;
      raf = requestAnimationFrame(frame);
      const engine = getEngine();
      if (!engine) return;
      const current = engine.mapInfo().map_id;
      if (current !== mapId) {
        mapId = current; // block repeats while the async rebuild runs
        void rebuild(engine);
        return;
      }
      if (!sim || !segs.length) return;
      const xpos = sim.data.xpos;
      const xmat = sim.data.xmat;
      const q = new THREE.Matrix4();
      for (const mesh of segs) {
        const b = mesh.userData.body as number;
        const r = xmat.slice(b * 9, b * 9 + 9);
        q.set(r[0], r[1], r[2], xpos[b * 3], r[3], r[4], r[5], xpos[b * 3 + 1], r[6], r[7], r[8], xpos[b * 3 + 2], 0, 0, 0, 1);
        mesh.matrixAutoUpdate = false;
        mesh.matrix.copy(q);
      }
      const b0 = segs[0].userData.body as number;
      const yaw = Math.atan2(xmat[b0 * 9 + 3], xmat[b0 * 9]);
      const pose = followPose([xpos[b0 * 3], xpos[b0 * 3 + 1], xpos[b0 * 3 + 2]], yaw, BACK_M, UP_M);
      camera.position.set(...pose.position);
      camera.lookAt(...pose.target);
      if (t - lastGoal > GOAL_REFRESH_MS) {
        lastGoal = t;
        const g = engine.snapshot().goal;
        goal.visible = !!g;
        if (g) goal.position.set(g[0], g[1], 0.05);
      }
      const w = container.clientWidth || 160;
      renderer.setSize(w, Math.round((w * 3) / 4), false);
      renderer.render(scene, camera);
    }
    raf = requestAnimationFrame(frame);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      disposeWorld();
      goal.geometry.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }

  return {
    mount(container: HTMLElement): void {
      stop?.();
      let cancelled = false;
      let release: (() => void) | null = null;
      stop = () => {
        cancelled = true;
        release?.();
      };
      void start(container).then((r) => (cancelled ? r() : (release = r)));
    },
    unmount(): void {
      stop?.();
      stop = null;
    },
  };
}

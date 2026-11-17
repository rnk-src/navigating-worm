// Pure geometry for the follow cam, kept apart from three.js so it can be tested.

export interface Grid {
  positions: Float32Array; // x, y, z per vertex
  uvs: Float32Array; // u, v per vertex; the map image's top edge is v = 1
  indices: Uint32Array;
}

/** A triangle grid for a MuJoCo heightfield. `heights` is row-major, normalised to 0..1. */
export function terrainGrid(
  heights: ArrayLike<number>,
  rows: number,
  cols: number,
  extent: [number, number, number, number], // xmin, xmax, ymin, ymax
  zmax: number,
): Grid {
  const [xmin, xmax, ymin, ymax] = extent;
  const positions = new Float32Array(rows * cols * 3);
  const uvs = new Float32Array(rows * cols * 2);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const u = c / (cols - 1);
      const v = r / (rows - 1);
      positions.set([xmin + u * (xmax - xmin), ymin + v * (ymax - ymin), heights[i] * zmax], i * 3);
      uvs.set([u, v], i * 2);
    }
  }
  const indices = new Uint32Array((rows - 1) * (cols - 1) * 6);
  let k = 0;
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c;
      indices.set([a, a + 1, a + cols, a + 1, a + cols + 1, a + cols], k);
      k += 6;
    }
  }
  return { positions, uvs, indices };
}

/** Camera behind and above the head, looking at it. World z is up. */
export function followPose(
  head: ArrayLike<number>,
  yaw: number,
  back: number,
  up: number,
): { position: [number, number, number]; target: [number, number, number] } {
  return {
    position: [head[0] - back * Math.cos(yaw), head[1] - back * Math.sin(yaw), head[2] + up],
    target: [head[0], head[1], head[2]],
  };
}

const TWO_PI = Math.PI * 2;

/** Ease `current` toward `target` with time constant `tau` seconds, turning the short way round. */
export function smoothAngle(current: number, target: number, dt: number, tau: number): number {
  const k = 1 - Math.exp(-dt / tau);
  const gap = ((((target - current) % TWO_PI) + 3 * Math.PI) % TWO_PI) - Math.PI;
  return current + gap * k;
}

export function smoothPoint(
  current: [number, number, number],
  target: ArrayLike<number>,
  dt: number,
  tau: number,
): [number, number, number] {
  const k = 1 - Math.exp(-dt / tau);
  return [
    current[0] + (target[0] - current[0]) * k,
    current[1] + (target[1] - current[1]) * k,
    current[2] + (target[2] - current[2]) * k,
  ];
}

/** Direction of travel along the whole body, from the tail to the head. */
export function heading(tail: ArrayLike<number>, head: ArrayLike<number>): number {
  return Math.atan2(head[1] - tail[1], head[0] - tail[0]);
}

/** Numbers overlapping async runs, so a slow older run can tell that a newer one replaced it. */
export function latestRun() {
  let newest = 0;
  return {
    start: (): number => ++newest,
    isCurrent: (run: number): boolean => run === newest,
  };
}

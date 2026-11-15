import { expect, test } from 'vitest';
import { followPose, terrainGrid } from '../src/camera-math';

test('terrainGrid places each vertex by row, column and height', () => {
  const g = terrainGrid(new Float32Array([0, 0.5, 1, 0, 0, 0.25]), 2, 3, [-1, 1, -2, 2], 0.4);
  expect(g.positions.length).toBe(6 * 3);
  // row 0, column 1: x midway, y at ymin, z = 0.5 * zmax
  expect([...g.positions.slice(3, 6)]).toEqual([0, -2, expect.closeTo(0.2, 6)]);
  // last vertex: row 1, column 2
  expect([...g.positions.slice(15, 18)]).toEqual([1, 2, expect.closeTo(0.1, 6)]);
});

test('terrainGrid uvs run from the map corners and the grid has two triangles per cell', () => {
  const g = terrainGrid(new Float32Array(6), 2, 3, [-1, 1, -2, 2], 1);
  expect([...g.uvs.slice(0, 2)]).toEqual([0, 0]);
  expect([...g.uvs.slice(10, 12)]).toEqual([1, 1]);
  expect(g.indices.length).toBe((2 - 1) * (3 - 1) * 6);
  expect(Math.max(...g.indices)).toBe(5);
});

test('followPose sits behind and above the head, looking at it', () => {
  const pose = followPose([1, 2, 0.1], Math.PI / 2, 0.5, 0.3);
  expect(pose.position[0]).toBeCloseTo(1, 6);
  expect(pose.position[1]).toBeCloseTo(1.5, 6);
  expect(pose.position[2]).toBeCloseTo(0.4, 6);
  expect(pose.target).toEqual([1, 2, 0.1]);
});

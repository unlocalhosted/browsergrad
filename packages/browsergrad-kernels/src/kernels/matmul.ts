/**
 * Matmul kernel: A [M, K] × B [K, N] → C [M, N]
 *
 * Naive triple-loop in WGSL. Workgroup size 8×8; each thread computes one
 * element of C. Correct for all M/K/N; not optimized.
 *
 * Optimized tiled matmul is planned but additive — same surface, different
 * cache key suffix, swapped via a future `mode` option.
 */

import { dispatch, type KernelDescriptor } from "../runner";
import { KernelError, type KernelDevice, type Tensor } from "../types";

const WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Params {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.M || col >= params.N) { return; }
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + A[row * params.K + k] * B[k * params.N + col];
  }
  C[row * params.N + col] = sum;
}
`;

const DESCRIPTOR: KernelDescriptor = {
  name: "matmul",
  wgsl: WGSL,
  workgroupSize: [8, 8, 1],
};

export async function matmul(
  device: KernelDevice,
  A: Tensor,
  B: Tensor,
): Promise<Tensor> {
  if (A.shape.length !== 2 || B.shape.length !== 2) {
    throw new KernelError(
      `matmul: expected 2D tensors, got ${A.shape.length}D and ${B.shape.length}D`,
    );
  }
  const [M, KA] = A.shape as [number, number];
  const [KB, N] = B.shape as [number, number];
  if (KA !== KB) {
    throw new KernelError(`matmul: inner dimensions don't match (${KA} vs ${KB})`);
  }

  const params = new Uint32Array([M, KA, N, 0]);
  const data = await dispatch(device, DESCRIPTOR, {
    inputs: [A.data, B.data],
    outputLength: M * N,
    params,
    dispatchCount: [M, N, 1],
    cacheKeySuffix: "f32",
  });
  return { shape: [M, N], data };
}

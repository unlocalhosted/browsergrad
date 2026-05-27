/**
 * Tiled matmul kernel: A [M, K] × B [K, N] → C [M, N]
 *
 * 16×16 tile, workgroup_size(16, 16, 1). Each workgroup computes a 16×16
 * tile of C. Workgroup-shared memory holds A_tile and B_tile (256 f32 each
 * = 1 KB each, 2 KB total — well under the WebGPU minimum 16 KB workgroup
 * memory limit).
 *
 * Algorithm (standard textbook tiled GEMM):
 *
 *   for tile_k in 0..ceil(K/16):
 *     A_tile[lx][ly] = A[block_row*16 + lx, tile_k*16 + ly]
 *     B_tile[lx][ly] = B[tile_k*16 + lx, block_col*16 + ly]
 *     workgroupBarrier()
 *     for k in 0..16:
 *       acc += A_tile[lx][k] * B_tile[k][ly]
 *     workgroupBarrier()
 *   C[row][col] = acc
 *
 * Why this is the load-bearing matmul kernel:
 *   - Naive matmul: M*N*K loads from A + M*N*K loads from B = 2*M*N*K reads.
 *   - Tiled: each A_tile loaded ceil(N/16) times by ceil(M/16) different
 *     workgroups → ~M*K*(N/16) reads of A and ~K*N*(M/16) reads of B,
 *     so ~M*N*K/16 + M*N*K/16 = M*N*K/8 total reads vs 2*M*N*K naive.
 *     That's 16× reduction in DRAM pressure → ~5–8× speedup in practice
 *     on a memory-bound naive baseline.
 *
 * Why we keep the naive matmul.ts:
 *   - The reference test against `referenceMatmul` is easier to debug when
 *     a regression hits — bisect between naive and tiled to localize.
 *   - PRD-012a explicitly wanted the tiled variant as additive, not as a
 *     replacement, so callers can opt out via `mode: 'naive'`.
 *
 * What this is NOT:
 *   - WMMA / tensor-core variant. WebGPU has no portable subgroup or WMMA
 *     extension exposed across drivers as of 2026-05. PRD-012b's call.
 *   - Mixed-precision variant. f16 path is PRD-012a's f16 follow-up.
 *   - Batched matmul. 2-D only in v0 (matches the realizer surface).
 */

import {
  runDirect,
  type KernelDescriptor,
  type DirectDispatchResult,
} from "../runner.js";
import { KernelError, type KernelDevice, type Tensor } from "../types.js";

const TILE = 16;

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

const TILE_SIZE: u32 = ${TILE}u;

var<workgroup> A_tile: array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> B_tile: array<array<f32, ${TILE}>, ${TILE}>;

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let lx = lid.x;
  let ly = lid.y;
  let row = wgid.x * TILE_SIZE + lx;
  let col = wgid.y * TILE_SIZE + ly;

  let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;
  var acc: f32 = 0.0;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let a_col = t * TILE_SIZE + ly;
    let b_row = t * TILE_SIZE + lx;

    let a_in = (row < params.M) && (a_col < params.K);
    let b_in = (b_row < params.K) && (col < params.N);

    A_tile[lx][ly] = select(0.0, A[row * params.K + a_col], a_in);
    B_tile[lx][ly] = select(0.0, B[b_row * params.N + col], b_in);
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
      acc = acc + A_tile[lx][k] * B_tile[k][ly];
    }
    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    C[row * params.N + col] = acc;
  }
}
`;

const DESCRIPTOR: KernelDescriptor = {
  name: "matmul_tiled",
  wgsl: WGSL,
  workgroupSize: [TILE, TILE, 1],
};

/**
 * Direct-dispatch tiled matmul. Inputs and output are GPUBuffers.
 * Caller owns the output buffer's lifetime.
 */
export function matmulTiledDirect(
  device: KernelDevice,
  A: GPUBuffer,
  B: GPUBuffer,
  M: number,
  K: number,
  N: number,
): DirectDispatchResult {
  const params = new Uint32Array([M, K, N, 0]);
  return runDirect(device, DESCRIPTOR, {
    inputBuffers: [A, B],
    outputLength: M * N,
    params,
    // Dispatch in TILE-aligned blocks; runDirect divides by workgroupSize.
    dispatchCount: [
      Math.ceil(M / TILE) * TILE,
      Math.ceil(N / TILE) * TILE,
      1,
    ],
    cacheKeySuffix: "f32-tiled16",
  });
}

/** Host-tensor variant: re-uses dispatch path with per-call host transfers.
 *  Mirrors `matmul.ts`'s `matmul` surface so callers can swap by import. */
export async function matmulTiled(
  device: KernelDevice,
  A: Tensor,
  B: Tensor,
): Promise<Tensor> {
  if (A.shape.length !== 2 || B.shape.length !== 2) {
    throw new KernelError(
      `matmul_tiled: expected 2D tensors, got ${A.shape.length}D and ${B.shape.length}D`,
    );
  }
  const [M, KA] = A.shape as [number, number];
  const [KB, N] = B.shape as [number, number];
  if (KA !== KB) {
    throw new KernelError(`matmul_tiled: inner dims don't match (${KA} vs ${KB})`);
  }
  const { dispatch } = await import("../runner.js");
  const params = new Uint32Array([M, KA, N, 0]);
  const data = await dispatch(device, DESCRIPTOR, {
    inputs: [A.data, B.data],
    outputLength: M * N,
    params,
    dispatchCount: [
      Math.ceil(M / TILE) * TILE,
      Math.ceil(N / TILE) * TILE,
      1,
    ],
    cacheKeySuffix: "f32-tiled16",
  });
  return { shape: [M, N], data };
}

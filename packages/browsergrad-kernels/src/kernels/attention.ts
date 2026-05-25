/**
 * Scaled dot-product attention (Vaswani et al. 2017).
 *
 *   scores  = Q @ K^T / sqrt(d_k)
 *   weights = softmax(scores)            (along last axis)
 *   out     = weights @ V
 *
 * v0 shapes: Q, K, V all [S, D]. Single head, no batching, no mask.
 *
 * v0 implementation: composes matmul + softmax in JS-orchestrating-GPU mode —
 * three separate kernel dispatches with intermediate buffers passed back through
 * the host. Correct, simple, slower than a fused kernel.
 *
 * A fused scaled-dot-product-attention kernel is on the roadmap (similar to
 * FlashAttention's tiling but much simpler) — additive, same surface.
 */

import { dispatch, type KernelDescriptor } from "../runner";
import { KernelError, type KernelDevice, type Tensor } from "../types";
import { matmul } from "./matmul";
import { softmax } from "./softmax";

const TRANSPOSE_2D_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
struct Params { rows: u32, cols: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  if (i >= params.rows || j >= params.cols) { return; }
  // X is [rows, cols]; Y is [cols, rows]
  Y[j * params.rows + i] = X[i * params.cols + j];
}
`;

const SCALE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
struct Params { n: u32, scale_bits: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let s = bitcast<f32>(params.scale_bits);
  Y[i] = X[i] * s;
}
`;

const TRANSPOSE_DESC: KernelDescriptor = {
  name: "transpose2d",
  wgsl: TRANSPOSE_2D_WGSL,
  workgroupSize: [8, 8, 1],
};

const SCALE_DESC: KernelDescriptor = {
  name: "scale",
  wgsl: SCALE_WGSL,
  workgroupSize: [64, 1, 1],
};

async function transpose2d(device: KernelDevice, x: Tensor): Promise<Tensor> {
  if (x.shape.length !== 2) {
    throw new KernelError(`transpose2d: expected 2D, got ${x.shape.length}D`);
  }
  const [R, C] = x.shape as [number, number];
  const params = new Uint32Array([R, C, 0, 0]);
  const data = await dispatch(device, TRANSPOSE_DESC, {
    inputs: [x.data],
    outputLength: R * C,
    params,
    dispatchCount: [R, C, 1],
    cacheKeySuffix: "f32",
  });
  return { shape: [C, R], data };
}

async function scale(device: KernelDevice, x: Tensor, factor: number): Promise<Tensor> {
  const n = x.data.length;
  const scaleBuf = new Float32Array([factor]);
  const scaleBits = new Uint32Array(scaleBuf.buffer, scaleBuf.byteOffset, 1)[0]!;
  const params = new Uint32Array([n, scaleBits, 0, 0]);
  const data = await dispatch(device, SCALE_DESC, {
    inputs: [x.data],
    outputLength: n,
    params,
    dispatchCount: [n, 1, 1],
    cacheKeySuffix: "f32",
  });
  return { shape: x.shape, data };
}

export async function attention(
  device: KernelDevice,
  Q: Tensor,
  K: Tensor,
  V: Tensor,
): Promise<Tensor> {
  if (Q.shape.length !== 2 || K.shape.length !== 2 || V.shape.length !== 2) {
    throw new KernelError(
      `attention: expected 2D Q, K, V, got ranks ${Q.shape.length}/${K.shape.length}/${V.shape.length}`,
    );
  }
  const [S, DQ] = Q.shape as [number, number];
  const [SK, DK] = K.shape as [number, number];
  const [SV, _DV] = V.shape as [number, number];
  if (DQ !== DK) throw new KernelError(`attention: Q dim ${DQ} ≠ K dim ${DK}`);
  if (S !== SK || S !== SV) {
    throw new KernelError(
      `attention: sequence lengths must match (Q: ${S}, K: ${SK}, V: ${SV})`,
    );
  }

  const KT = await transpose2d(device, K);
  const scores = await matmul(device, Q, KT);
  const scaled = await scale(device, scores, 1 / Math.sqrt(DK));
  const weights = await softmax(device, scaled);
  return matmul(device, weights, V);
}

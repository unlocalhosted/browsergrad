/**
 * Softmax along the last axis. Stable variant (subtract max → exp → divide by sum).
 *
 * Each workgroup row corresponds to one independent softmax computation:
 *   - One thread per row (workgroup_size = 1×1×1) — the row loop runs serially.
 *   - rows = numel / lastDim
 *
 * The 1-thread-per-row design is the simplest correct WGSL softmax. Per-thread
 * cooperative reduction (shared memory, two-pass) is faster but harder to read;
 * planned for v0.2 as a separate kernel variant.
 */

import { dispatch, numel, type KernelDescriptor } from "../runner.js";
import { KernelError, type KernelDevice, type Tensor } from "../types.js";

const WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= params.rows) { return; }
  let base = r * params.cols;

  // Pass 1: find max for numerical stability.
  //
  // Note: initializing maxVal with the first element rather than -f32_max.
  // A negative literal at the magnitude of -f32_max ( -3.4028235e38 ) parsed
  // as 0 on Chromium's SwiftShader + Metal-driver path (confirmed by a series
  // of incremental probe kernels — literal init alone is the failure point).
  // Using X[base] sidesteps the literal-parse issue entirely and is also a
  // touch more numerically robust: subtracting an actual element instead of
  // an arbitrarily-tiny floor doesn't change correctness but avoids hitting
  // exp(very_large_finite_positive) on cols where every input is far above
  // -f32_max.
  var maxVal: f32 = X[base];
  for (var i: u32 = 1u; i < params.cols; i = i + 1u) {
    let v = X[base + i];
    if (v > maxVal) { maxVal = v; }
  }

  // Pass 2: accumulate sum of exp(x - max) in a register. Do not write Y.
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    sum = sum + exp(X[base + i] - maxVal);
  }

  // Pass 3: pure-write the final normalized values.
  let inv = 1.0 / sum;
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    Y[base + i] = exp(X[base + i] - maxVal) * inv;
  }
}
`;

const DESCRIPTOR: KernelDescriptor = {
  name: "softmax",
  wgsl: WGSL,
  workgroupSize: [64, 1, 1],
};

export async function softmax(
  device: KernelDevice,
  x: Tensor,
): Promise<Tensor> {
  if (x.shape.length === 0) {
    throw new KernelError("softmax: scalar tensor not supported");
  }
  const D = x.shape[x.shape.length - 1]!;
  const total = numel(x.shape);
  const rows = total / D;
  if (!Number.isInteger(rows)) {
    throw new KernelError(`softmax: shape [${x.shape.join(", ")}] is not consistent`);
  }

  const params = new Uint32Array([rows, D, 0, 0]);
  const data = await dispatch(device, DESCRIPTOR, {
    inputs: [x.data],
    outputLength: total,
    params,
    dispatchCount: [rows, 1, 1],
    cacheKeySuffix: "f32",
  });
  return { shape: x.shape, data };
}

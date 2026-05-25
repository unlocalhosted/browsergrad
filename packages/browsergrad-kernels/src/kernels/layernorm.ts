/**
 * Layer normalization along the last axis.
 *
 *   y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * gamma and beta are optional. When omitted we pass shape-D vectors of 1s and 0s
 * respectively, so the WGSL kernel always sees four storage bindings — keeps the
 * pipeline cache key small and the bind-group layout uniform.
 *
 * One thread per row. Each thread does a 3-pass scan (mean, variance, normalize).
 */

import { dispatch, numel, type KernelDescriptor } from "../runner.js";
import { KernelError, type KernelDevice, type Tensor } from "../types.js";

const WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gamma: array<f32>;
@group(0) @binding(2) var<storage, read> Beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  eps_bits: u32, // bitcast of f32 eps; WGSL has no f32 in uniforms with bool packing
  _pad: u32,
};
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= params.rows) { return; }
  let base = r * params.cols;
  let n = f32(params.cols);
  let eps = bitcast<f32>(params.eps_bits);

  // Pass 1: mean
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    sum = sum + X[base + i];
  }
  let mean = sum / n;

  // Pass 2: variance
  var varSum: f32 = 0.0;
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    let d = X[base + i] - mean;
    varSum = varSum + d * d;
  }
  let variance = varSum / n;
  let invStd = 1.0 / sqrt(variance + eps);

  // Pass 3: normalize + affine
  for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
    let normed = (X[base + i] - mean) * invStd;
    Y[base + i] = normed * Gamma[i] + Beta[i];
  }
}
`;

const DESCRIPTOR: KernelDescriptor = {
  name: "layernorm",
  wgsl: WGSL,
  workgroupSize: [64, 1, 1],
};

export interface LayerNormOptions {
  gamma?: Tensor;
  beta?: Tensor;
  eps?: number;
}

export async function layernorm(
  device: KernelDevice,
  x: Tensor,
  opts: LayerNormOptions = {},
): Promise<Tensor> {
  if (x.shape.length === 0) {
    throw new KernelError("layernorm: scalar tensor not supported");
  }
  const D = x.shape[x.shape.length - 1]!;
  const total = numel(x.shape);
  const rows = total / D;
  if (!Number.isInteger(rows)) {
    throw new KernelError(`layernorm: shape [${x.shape.join(", ")}] is not consistent`);
  }

  // Default gamma=1, beta=0 if not provided.
  const gammaData = opts.gamma?.data ?? defaultVec(D, 1);
  const betaData = opts.beta?.data ?? defaultVec(D, 0);
  if (gammaData.length !== D) {
    throw new KernelError(`layernorm: gamma length ${gammaData.length} ≠ last dim ${D}`);
  }
  if (betaData.length !== D) {
    throw new KernelError(`layernorm: beta length ${betaData.length} ≠ last dim ${D}`);
  }

  const eps = opts.eps ?? 1e-5;
  // Pack eps via bitcast through a Float32Array view.
  const epsBuf = new Float32Array([eps]);
  const epsBits = new Uint32Array(epsBuf.buffer, epsBuf.byteOffset, 1)[0]!;
  const params = new Uint32Array([rows, D, epsBits, 0]);

  const data = await dispatch(device, DESCRIPTOR, {
    inputs: [x.data, gammaData, betaData],
    outputLength: total,
    params,
    dispatchCount: [rows, 1, 1],
    cacheKeySuffix: "f32",
  });
  return { shape: x.shape, data };
}

function defaultVec(length: number, fill: number): Float32Array {
  const out = new Float32Array(length);
  if (fill !== 0) out.fill(fill);
  return out;
}

/**
 * Elementwise activations: relu, gelu.
 *
 * One thread per element. Workgroup size 64 (1D dispatch).
 *
 * GELU uses the tanh approximation:
 *   gelu(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
 * This is the variant in GPT-2/BERT and what the JS reference uses.
 */

import { dispatch, numel, type KernelDescriptor } from "../runner.js";
import { type KernelDevice, type Tensor } from "../types.js";

const RELU_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
struct Params { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let v = X[i];
  Y[i] = select(0.0, v, v > 0.0);
}
`;

const GELU_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
struct Params { n: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(2) var<uniform> params: Params;

// sqrt(2/pi) ≈ 0.7978845608
const GELU_C: f32 = 0.7978845608;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let v = X[i];
  let inner = GELU_C * (v + 0.044715 * v * v * v);
  Y[i] = 0.5 * v * (1.0 + tanh(inner));
}
`;

const RELU_DESCRIPTOR: KernelDescriptor = {
  name: "relu",
  wgsl: RELU_WGSL,
  workgroupSize: [64, 1, 1],
};

const GELU_DESCRIPTOR: KernelDescriptor = {
  name: "gelu",
  wgsl: GELU_WGSL,
  workgroupSize: [64, 1, 1],
};

async function runElementwise(
  device: KernelDevice,
  desc: KernelDescriptor,
  x: Tensor,
): Promise<Tensor> {
  const n = numel(x.shape);
  const params = new Uint32Array([n, 0, 0, 0]);
  const data = await dispatch(device, desc, {
    inputs: [x.data],
    outputLength: n,
    params,
    dispatchCount: [n, 1, 1],
    cacheKeySuffix: "f32",
  });
  return { shape: x.shape, data };
}

export function relu(device: KernelDevice, x: Tensor): Promise<Tensor> {
  return runElementwise(device, RELU_DESCRIPTOR, x);
}

export function gelu(device: KernelDevice, x: Tensor): Promise<Tensor> {
  return runElementwise(device, GELU_DESCRIPTOR, x);
}

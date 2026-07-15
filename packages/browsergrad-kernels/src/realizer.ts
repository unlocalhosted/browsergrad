/**
 * createWebGpuRealizerBridge — the production WebGPU bridge for the
 * browsergrad-jit Python realizer (PRD-011.5).
 *
 * The bridge satisfies the `WebGpuBridge` Protocol declared on the
 * Python side (see `browsergrad_jit/_bridge.py`). Each method:
 *
 *   - Crosses the Pyodide JS boundary as a synchronous-from-Python call
 *     via JSPI (Pyodide 0.27+ on Chrome 144+). Caller asynchrony is
 *     hidden behind Python's normal call stack.
 *   - Operates on opaque integer handles. Python never sees a GPUBuffer
 *     directly; the bridge owns the lifetime via its internal handle
 *     map.
 *   - Returns either a handle (compute ops) or a Uint8Array (materialise),
 *     which Pyodide marshals back to Python.
 *
 * v0 scope per the DL/GPU review:
 *   - f32 only. f16/AMP path is PRD-012a.
 *   - 2-D matmul only. Batched matmul is PRD-012a.
 *   - Fused row-wise online-softmax attention forward. This is not a
 *     block-tiled FlashAttention schedule. Backward is separate.
 *   - Naive (un-tiled) matmul kernel — the existing `kernels/matmul.ts`.
 *     Replacing with a tiled GEMM is the first deliverable of PRD-012a.
 *
 * What this is NOT:
 *   - A pattern matcher. The Python `bg.kernels.flash_attention(...)`
 *     wrapper constructs the CUSTOM UOp explicitly; the bridge has no
 *     "is this an attention block?" logic.
 *   - A cost model. The realizer just dispatches what it gets.
 *   - An OPFS pipeline cache. The device's in-memory pipeline cache
 *     covers session-scope reuse; cross-session caching is PRD-008.2.
 */

import {
  materializeFloat32,
  uploadFloat32,
  runDirect,
  type DirectDispatchProfile,
  type DirectDispatchProfileOptions,
  type DirectDispatchResult,
  type KernelDescriptor,
} from "./runner.js";
import { matmulTiledDirect } from "./kernels/matmul_tiled.js";
import {
  fusedElementwiseDirect,
  type FusedOp,
} from "./kernels/fused_elementwise.js";
import { rowWiseOnlineAttentionDirect } from "./kernels/flash_attention.js";
import { runTensorGpuPlan, runTensorGpuPlanResident, type TensorPlanInput } from "./tensor_plan.js";
import { KernelError, type KernelDevice } from "./types.js";

type Handle = number;

interface BufferRecord {
  buffer: GPUBuffer;
  byteLength: number;
  shape: readonly number[];
  dtype: string;
}

export interface WebGpuRealizerBridge {
  upload(data: Uint8Array, shape: readonly number[], dtype: string): Handle;
  /**
   * Read a handle's contents back to a Uint8Array of f32 byte content.
   *
   * Returns a Promise because GPU readback is asynchronous (mapAsync). Pyodide
   * JSPI consumers (Python's `bridge.materialize(...)` call) hide the await
   * transparently — the Promise is unwrapped at the JS↔Python boundary, so
   * Python sees a synchronous Uint8Array return as the Protocol declares.
   *
   * JS/TS consumers must `await` this. (Previous releases declared the return
   * type as `Uint8Array` directly to mirror the Python Protocol; that was a
   * type contract violation — runtime always returned a Promise.)
   */
  materialize(handle: Handle, shape: readonly number[], dtype: string): Promise<Uint8Array>;
  release(handle: Handle): void;
  matmul(a: Handle, b: Handle, m: number, k: number, n: number, dtype: string): Handle;
  fused_elementwise(
    inputs: readonly Handle[],
    ops: readonly (readonly [string, number, number])[],
    shape: readonly number[],
    dtype: string,
  ): Handle;
  cast(handle: Handle, srcDtype: string, dstDtype: string, shape: readonly number[]): Handle;
  flash_attention(
    q: Handle,
    k: Handle,
    v: Handle,
    mask: Handle | null,
    b: number,
    h: number,
    sq: number,
    sk: number,
    d: number,
    scale: number,
    dtype: string,
  ): Promise<Handle>;
  conv1d(
    input: Handle,
    weight: Handle,
    bias: Handle | null,
    n: number,
    cIn: number,
    lIn: number,
    cOut: number,
    k: number,
    stride: number,
    padding: number,
    dilation: number,
    groups: number,
    lOut: number,
    dtype: string,
  ): Handle;
  conv1d_backward_input(
    dy: Handle,
    weight: Handle,
    n: number,
    cIn: number,
    lIn: number,
    cOut: number,
    k: number,
    stride: number,
    padding: number,
    dilation: number,
    groups: number,
    lOut: number,
    dtype: string,
  ): Handle;
  conv1d_backward_weight(
    dy: Handle,
    input: Handle,
    n: number,
    cIn: number,
    lIn: number,
    cOut: number,
    k: number,
    stride: number,
    padding: number,
    dilation: number,
    groups: number,
    lOut: number,
    dtype: string,
  ): Handle;
  conv1d_backward_bias(
    dy: Handle,
    n: number,
    cOut: number,
    lOut: number,
    dtype: string,
  ): Handle;
  conv2d(
    input: Handle,
    weight: Handle,
    bias: Handle | null,
    n: number,
    cIn: number,
    h: number,
    w: number,
    cOut: number,
    kh: number,
    kw: number,
    strideH: number,
    strideW: number,
    padH: number,
    padW: number,
    dilationH: number,
    dilationW: number,
    groups: number,
    outH: number,
    outW: number,
    dtype: string,
  ): Handle;
  conv2d_backward_input(
    dy: Handle,
    weight: Handle,
    n: number,
    cIn: number,
    h: number,
    w: number,
    cOut: number,
    kh: number,
    kw: number,
    strideH: number,
    strideW: number,
    padH: number,
    padW: number,
    dilationH: number,
    dilationW: number,
    groups: number,
    outH: number,
    outW: number,
    dtype: string,
  ): Handle;
  conv2d_backward_weight(
    dy: Handle,
    input: Handle,
    n: number,
    cIn: number,
    h: number,
    w: number,
    cOut: number,
    kh: number,
    kw: number,
    strideH: number,
    strideW: number,
    padH: number,
    padW: number,
    dilationH: number,
    dilationW: number,
    groups: number,
    outH: number,
    outW: number,
    dtype: string,
  ): Handle;
  conv2d_backward_bias(
    dy: Handle,
    n: number,
    cOut: number,
    outH: number,
    outW: number,
    dtype: string,
  ): Handle;
  run_user_kernel(
    inputs: readonly Handle[],
    wgsl: string,
    name: string,
    hash: string,
    workgroupSize: readonly [number, number, number],
    dispatchShape: readonly [number, number, number],
    outputLength: number,
    outputShape: readonly number[],
    dtype: string,
  ): Handle;
  run_tensor_plan(
    plan: unknown,
    inputs: readonly unknown[],
    dtype: string,
  ): Promise<Uint8Array>;
  run_tensor_plan_resident(
    plan: unknown,
    inputs: readonly unknown[],
    dtype: string,
  ): Handle;
  /** Diagnostic — number of GPU buffers currently alive. */
  aliveHandleCount(): number;
  /** Correctness-labeled BrowserGrad-owned WebGPU resource snapshot. */
  resourceSnapshot(): WebGpuResourceSnapshot;
  /** Wait for submitted WebGPU profile readbacks, then return a fresh snapshot. */
  flushProfiles(): Promise<WebGpuResourceSnapshot>;
}

export type WebGpuTimingMode =
  | "timestamp-query"
  | "queue-completion"
  | "unavailable";

export interface WebGpuResourceSnapshot {
  readonly timingMode: WebGpuTimingMode;
  readonly timestampQueryAvailable: boolean;
  readonly currentOwnedGpuBytes: number;
  readonly peakOwnedGpuBytes: number;
  readonly totalAllocatedGpuBytes: number;
  readonly totalReleasedGpuBytes: number;
  readonly aliveHandleCount: number;
  readonly logicalTensorPlanPeakBytes?: number;
  readonly pendingProfileCount: number;
  readonly passProfiles: readonly DirectDispatchProfile[];
}

function assertF32(dtype: string, op: string): void {
  if (dtype !== "float32") {
    throw new KernelError(
      `WebGPU bridge: ${op} only supports float32 in v0 (got ${dtype}). ` +
        `f16/AMP path is PRD-012a.`,
    );
  }
}

function assertPositiveInt(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new KernelError(`WebGPU bridge: ${name} must be a positive integer`);
  }
}

function conv1dWgsl(hasBias: boolean): string {
  const biasBinding = hasBias ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : "";
  const outputBinding = hasBias ? 3 : 2;
  const paramsBinding = hasBias ? 4 : 3;
  const biasLine = hasBias ? " + B[co]" : "";
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
${biasBinding}
@group(0) @binding(${outputBinding}) var<storage, read_write> Y: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  l_in: u32,
  c_out: u32,
  k: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  groups: u32,
  l_out: u32,
};
@group(0) @binding(${paramsBinding}) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = P.n * P.c_out * P.l_out;
  if (idx >= total) { return; }
  let pos = idx % P.l_out;
  let co = (idx / P.l_out) % P.c_out;
  let nn = idx / (P.l_out * P.c_out);
  let c_per_group = P.c_in / P.groups;
  let out_per_group = P.c_out / P.groups;
  let group = co / out_per_group;
  let c0 = group * c_per_group;
  var acc = 0.0;
  for (var ci_local = 0u; ci_local < c_per_group; ci_local = ci_local + 1u) {
    let ci = c0 + ci_local;
    for (var r = 0u; r < P.k; r = r + 1u) {
      let li_unpadded = i32(pos * P.stride + r * P.dilation) - i32(P.padding);
      if (li_unpadded < 0 || li_unpadded >= i32(P.l_in)) { continue; }
      let x_index = (nn * P.c_in + ci) * P.l_in + u32(li_unpadded);
      let w_index = (co * c_per_group + ci_local) * P.k + r;
      acc = acc + X[x_index] * W[w_index];
    }
  }
  Y[idx] = acc${biasLine};
}
`;
}

const CONV1D_BACKWARD_INPUT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> DX: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  l_in: u32,
  c_out: u32,
  k: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  groups: u32,
  l_out: u32,
};
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = P.n * P.c_in * P.l_in;
  if (idx >= total) { return; }
  let li = idx % P.l_in;
  let ci = (idx / P.l_in) % P.c_in;
  let nn = idx / (P.l_in * P.c_in);
  let c_per_group = P.c_in / P.groups;
  let out_per_group = P.c_out / P.groups;
  let group = ci / c_per_group;
  let ci_local = ci - group * c_per_group;
  let o0 = group * out_per_group;
  var acc = 0.0;
  for (var co_local = 0u; co_local < out_per_group; co_local = co_local + 1u) {
    let co = o0 + co_local;
    for (var r = 0u; r < P.k; r = r + 1u) {
      let pos_num = i32(li) + i32(P.padding) - i32(r * P.dilation);
      if (pos_num < 0 || (pos_num % i32(P.stride)) != 0) { continue; }
      let pos_i = pos_num / i32(P.stride);
      if (pos_i < 0 || pos_i >= i32(P.l_out)) { continue; }
      let pos = u32(pos_i);
      let dy_index = (nn * P.c_out + co) * P.l_out + pos;
      let w_index = (co * c_per_group + ci_local) * P.k + r;
      acc = acc + DY[dy_index] * W[w_index];
    }
  }
  DX[idx] = acc;
}
`;

const CONV1D_BACKWARD_WEIGHT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> DW: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  l_in: u32,
  c_out: u32,
  k: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  groups: u32,
  l_out: u32,
};
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let c_per_group = P.c_in / P.groups;
  let total = P.c_out * c_per_group * P.k;
  if (idx >= total) { return; }
  let r = idx % P.k;
  let ci_local = (idx / P.k) % c_per_group;
  let co = idx / (P.k * c_per_group);
  let out_per_group = P.c_out / P.groups;
  let group = co / out_per_group;
  let ci = group * c_per_group + ci_local;
  var acc = 0.0;
  for (var nn = 0u; nn < P.n; nn = nn + 1u) {
    for (var pos = 0u; pos < P.l_out; pos = pos + 1u) {
      let li_i = i32(pos * P.stride + r * P.dilation) - i32(P.padding);
      if (li_i < 0 || li_i >= i32(P.l_in)) { continue; }
      let li = u32(li_i);
      let dy_index = (nn * P.c_out + co) * P.l_out + pos;
      let x_index = (nn * P.c_in + ci) * P.l_in + li;
      acc = acc + DY[dy_index] * X[x_index];
    }
  }
  DW[idx] = acc;
}
`;

const CONV1D_BACKWARD_BIAS_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read_write> DB: array<f32>;

struct Params {
  n: u32,
  c_out: u32,
  l_out: u32,
  pad: u32,
};
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let co = gid.x;
  if (co >= P.c_out) { return; }
  var acc = 0.0;
  for (var nn = 0u; nn < P.n; nn = nn + 1u) {
    for (var pos = 0u; pos < P.l_out; pos = pos + 1u) {
      let dy_index = (nn * P.c_out + co) * P.l_out + pos;
      acc = acc + DY[dy_index];
    }
  }
  DB[co] = acc;
}
`;

function conv2dWgsl(hasBias: boolean): string {
  const biasBinding = hasBias ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : "";
  const outputBinding = hasBias ? 3 : 2;
  const paramsBinding = hasBias ? 4 : 3;
  const biasLine = hasBias ? " + B[co]" : "";
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
${biasBinding}
@group(0) @binding(${outputBinding}) var<storage, read_write> Y: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  h: u32,
  w: u32,
  c_out: u32,
  kh: u32,
  kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_h: u32,
  out_w: u32,
};
@group(0) @binding(${paramsBinding}) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = P.n * P.c_out * P.out_h * P.out_w;
  if (idx >= total) {
    return;
  }
  let ow = idx % P.out_w;
  let oh = (idx / P.out_w) % P.out_h;
  let co = (idx / (P.out_w * P.out_h)) % P.c_out;
  let nn = idx / (P.out_w * P.out_h * P.c_out);
  let c_per_group = P.c_in / P.groups;
  let out_per_group = P.c_out / P.groups;
  let group = co / out_per_group;
  let c0 = group * c_per_group;
  var acc = 0.0;
  for (var ci_local = 0u; ci_local < c_per_group; ci_local = ci_local + 1u) {
    let ci = c0 + ci_local;
    for (var r = 0u; r < P.kh; r = r + 1u) {
      let ih_unpadded = i32(oh * P.stride_h + r * P.dilation_h) - i32(P.pad_h);
      if (ih_unpadded < 0 || ih_unpadded >= i32(P.h)) {
        continue;
      }
      for (var c = 0u; c < P.kw; c = c + 1u) {
        let iw_unpadded = i32(ow * P.stride_w + c * P.dilation_w) - i32(P.pad_w);
        if (iw_unpadded < 0 || iw_unpadded >= i32(P.w)) {
          continue;
        }
        let x_index = ((nn * P.c_in + ci) * P.h + u32(ih_unpadded)) * P.w + u32(iw_unpadded);
        let w_index = ((co * c_per_group + ci_local) * P.kh + r) * P.kw + c;
        acc = acc + X[x_index] * W[w_index];
      }
    }
  }
  Y[idx] = acc${biasLine};
}
`;
}

const CONV2D_BACKWARD_INPUT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> DX: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  h: u32,
  w: u32,
  c_out: u32,
  kh: u32,
  kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_h: u32,
  out_w: u32,
};
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = P.n * P.c_in * P.h * P.w;
  if (idx >= total) { return; }
  let iw = idx % P.w;
  let ih = (idx / P.w) % P.h;
  let ci = (idx / (P.w * P.h)) % P.c_in;
  let nn = idx / (P.w * P.h * P.c_in);
  let c_per_group = P.c_in / P.groups;
  let out_per_group = P.c_out / P.groups;
  let group = ci / c_per_group;
  let ci_local = ci - group * c_per_group;
  let o0 = group * out_per_group;
  var acc = 0.0;
  for (var co_local = 0u; co_local < out_per_group; co_local = co_local + 1u) {
    let co = o0 + co_local;
    for (var r = 0u; r < P.kh; r = r + 1u) {
      let h_num = i32(ih) + i32(P.pad_h) - i32(r * P.dilation_h);
      if (h_num < 0 || (h_num % i32(P.stride_h)) != 0) { continue; }
      let oh_i = h_num / i32(P.stride_h);
      if (oh_i < 0 || oh_i >= i32(P.out_h)) { continue; }
      let oh = u32(oh_i);
      for (var c = 0u; c < P.kw; c = c + 1u) {
        let w_num = i32(iw) + i32(P.pad_w) - i32(c * P.dilation_w);
        if (w_num < 0 || (w_num % i32(P.stride_w)) != 0) { continue; }
        let ow_i = w_num / i32(P.stride_w);
        if (ow_i < 0 || ow_i >= i32(P.out_w)) { continue; }
        let ow = u32(ow_i);
        let dy_index = ((nn * P.c_out + co) * P.out_h + oh) * P.out_w + ow;
        let w_index = ((co * c_per_group + ci_local) * P.kh + r) * P.kw + c;
        acc = acc + DY[dy_index] * W[w_index];
      }
    }
  }
  DX[idx] = acc;
}
`;

const CONV2D_BACKWARD_WEIGHT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> DW: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  h: u32,
  w: u32,
  c_out: u32,
  kh: u32,
  kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_h: u32,
  out_w: u32,
};
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let c_per_group = P.c_in / P.groups;
  let total = P.c_out * c_per_group * P.kh * P.kw;
  if (idx >= total) { return; }
  let kc = idx % P.kw;
  let kr = (idx / P.kw) % P.kh;
  let ci_local = (idx / (P.kw * P.kh)) % c_per_group;
  let co = idx / (P.kw * P.kh * c_per_group);
  let out_per_group = P.c_out / P.groups;
  let group = co / out_per_group;
  let ci = group * c_per_group + ci_local;
  var acc = 0.0;
  for (var nn = 0u; nn < P.n; nn = nn + 1u) {
    for (var oh = 0u; oh < P.out_h; oh = oh + 1u) {
      let ih_i = i32(oh * P.stride_h + kr * P.dilation_h) - i32(P.pad_h);
      if (ih_i < 0 || ih_i >= i32(P.h)) { continue; }
      let ih = u32(ih_i);
      for (var ow = 0u; ow < P.out_w; ow = ow + 1u) {
        let iw_i = i32(ow * P.stride_w + kc * P.dilation_w) - i32(P.pad_w);
        if (iw_i < 0 || iw_i >= i32(P.w)) { continue; }
        let iw = u32(iw_i);
        let dy_index = ((nn * P.c_out + co) * P.out_h + oh) * P.out_w + ow;
        let x_index = ((nn * P.c_in + ci) * P.h + ih) * P.w + iw;
        acc = acc + DY[dy_index] * X[x_index];
      }
    }
  }
  DW[idx] = acc;
}
`;

const CONV2D_BACKWARD_BIAS_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read_write> DB: array<f32>;

struct Params {
  n: u32,
  c_out: u32,
  out_h: u32,
  out_w: u32,
};
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let co = gid.x;
  if (co >= P.c_out) { return; }
  var acc = 0.0;
  for (var nn = 0u; nn < P.n; nn = nn + 1u) {
    for (var oh = 0u; oh < P.out_h; oh = oh + 1u) {
      for (var ow = 0u; ow < P.out_w; ow = ow + 1u) {
        let dy_index = ((nn * P.c_out + co) * P.out_h + oh) * P.out_w + ow;
        acc = acc + DY[dy_index];
      }
    }
  }
  DB[co] = acc;
}
`;

function validateConv2dShape(
  op: string,
  n: number,
  cIn: number,
  h: number,
  w: number,
  cOut: number,
  kh: number,
  kw: number,
  strideH: number,
  strideW: number,
  padH: number,
  padW: number,
  dilationH: number,
  dilationW: number,
  groups: number,
  outH: number,
  outW: number,
): void {
  for (const [value, name] of [
    [n, "n"],
    [cIn, "cIn"],
    [h, "h"],
    [w, "w"],
    [cOut, "cOut"],
    [kh, "kh"],
    [kw, "kw"],
    [strideH, "strideH"],
    [strideW, "strideW"],
    [dilationH, "dilationH"],
    [dilationW, "dilationW"],
    [groups, "groups"],
    [outH, "outH"],
    [outW, "outW"],
  ] as const) {
    assertPositiveInt(value, `${op}.${name}`);
  }
  if (!Number.isInteger(padH) || padH < 0 || !Number.isInteger(padW) || padW < 0) {
    throw new KernelError(`WebGPU bridge: ${op} padding must be non-negative integers`);
  }
  if (cIn % groups !== 0 || cOut % groups !== 0) {
    throw new KernelError(`WebGPU bridge: ${op} channels must be divisible by groups`);
  }
}

function validateConv1dShape(
  op: string,
  n: number,
  cIn: number,
  lIn: number,
  cOut: number,
  k: number,
  stride: number,
  padding: number,
  dilation: number,
  groups: number,
  lOut: number,
): void {
  for (const [value, name] of [
    [n, "n"],
    [cIn, "cIn"],
    [lIn, "lIn"],
    [cOut, "cOut"],
    [k, "k"],
    [stride, "stride"],
    [dilation, "dilation"],
    [groups, "groups"],
    [lOut, "lOut"],
  ] as const) {
    assertPositiveInt(value, `${op}.${name}`);
  }
  if (!Number.isInteger(padding) || padding < 0) {
    throw new KernelError(`WebGPU bridge: ${op} padding must be a non-negative integer`);
  }
  if (cIn % groups !== 0 || cOut % groups !== 0) {
    throw new KernelError(`WebGPU bridge: ${op} channels must be divisible by groups`);
  }
}

function conv1dParams(
  n: number,
  cIn: number,
  lIn: number,
  cOut: number,
  k: number,
  stride: number,
  padding: number,
  dilation: number,
  groups: number,
  lOut: number,
): Uint32Array {
  return new Uint32Array([
    n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut, 0, 0,
  ]);
}

function conv2dParams(
  n: number,
  cIn: number,
  h: number,
  w: number,
  cOut: number,
  kh: number,
  kw: number,
  strideH: number,
  strideW: number,
  padH: number,
  padW: number,
  dilationH: number,
  dilationW: number,
  groups: number,
  outH: number,
  outW: number,
): Uint32Array {
  return new Uint32Array([
    n, cIn, h, w, cOut, kh, kw, strideH, strideW, padH, padW,
    dilationH, dilationW, groups, outH, outW,
  ]);
}

function bytesToFloat32(data: unknown, name: string): Float32Array {
  if (!(data instanceof Uint8Array)) {
    throw new KernelError(`WebGPU bridge: ${name}.data must be Uint8Array bytes`);
  }
  if (data.byteLength % 4 !== 0) {
    throw new KernelError(`WebGPU bridge: ${name}.data byte length must be divisible by 4`);
  }
  if ((data.byteOffset & 3) === 0) {
    return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  }
  const copy = new Uint8Array(data.byteLength);
  copy.set(data);
  return new Float32Array(copy.buffer);
}

function normalizeTensorPlanInputs(
  inputs: readonly unknown[],
  getHandle?: (handle: Handle, op: string) => BufferRecord,
): TensorPlanInput[] {
  return inputs.map((input, i) => {
    if (input === null || typeof input !== "object" || Array.isArray(input)) {
      throw new KernelError(`WebGPU bridge: tensor plan input ${i} must be an object`);
    }
    const record = input as Record<string, unknown>;
    const rawId = record.valueId ?? record.value_id;
    if (typeof rawId !== "number" || !Number.isInteger(rawId)) {
      throw new KernelError(`WebGPU bridge: tensor plan input ${i}.valueId must be an integer`);
    }
    if (record.handle !== undefined) {
      if (typeof record.handle !== "number" || !Number.isInteger(record.handle)) {
        throw new KernelError(`WebGPU bridge: tensor plan input ${i}.handle must be an integer`);
      }
      if (!getHandle) {
        throw new KernelError(`WebGPU bridge: tensor plan input ${i}.handle is unsupported here`);
      }
      const rec = getHandle(record.handle, `tensor_plan[input${i}]`);
      assertF32(rec.dtype, `tensor_plan[input${i}]`);
      return {
        valueId: rawId,
        resident: {
          buffer: rec.buffer,
          byteLength: rec.byteLength,
        },
      };
    }
    return {
      valueId: rawId,
      data: bytesToFloat32(record.data, `tensor plan input ${i}`),
    };
  });
}

/**
 * Construct a WebGpuRealizerBridge bound to the given device. Holds a
 * mutable handle map internally; never thread-shared.
 */
export interface WebGpuRealizerBridgeOptions {
  readonly profiling?: boolean;
}

export function createWebGpuRealizerBridge(
  device: KernelDevice,
  options: WebGpuRealizerBridgeOptions = {},
): WebGpuRealizerBridge {
  const handles = new Map<Handle, BufferRecord>();
  let nextId = 1;
  let currentOwnedGpuBytes = 0;
  let peakOwnedGpuBytes = 0;
  let totalAllocatedGpuBytes = 0;
  let totalReleasedGpuBytes = 0;
  let logicalTensorPlanPeakBytes: number | undefined;
  const timestampQueryAvailable = device.gpu.features.has("timestamp-query");
  const profilingEnabled = options.profiling !== false;
  const passProfiles: DirectDispatchProfile[] = [];
  const pendingProfiles = new Set<Promise<DirectDispatchProfile>>();

  const mint = (
    buffer: GPUBuffer,
    byteLength: number,
    shape: readonly number[],
    dtype: string,
  ): Handle => {
    const id = nextId++;
    handles.set(id, { buffer, byteLength, shape, dtype });
    currentOwnedGpuBytes += byteLength;
    totalAllocatedGpuBytes += byteLength;
    peakOwnedGpuBytes = Math.max(peakOwnedGpuBytes, currentOwnedGpuBytes);
    return id;
  };

  const releaseRecord = (rec: BufferRecord): void => {
    rec.buffer.destroy();
    currentOwnedGpuBytes = Math.max(0, currentOwnedGpuBytes - rec.byteLength);
    totalReleasedGpuBytes += rec.byteLength;
  };

  const profileFor = (label: string): DirectDispatchProfileOptions =>
    profilingEnabled ? { enabled: true, label } : { enabled: false, label };

  const trackProfilePromise = (profilePromise: Promise<DirectDispatchProfile>): void => {
    const tracked = profilePromise.then((profile) => {
      passProfiles.push(profile);
      return profile;
    });
    pendingProfiles.add(tracked);
    tracked.finally(() => {
      pendingProfiles.delete(tracked);
    }).catch(() => {});
  };

  const trackProfile = (result: DirectDispatchResult): void => {
    if (!result.profile) return;
    trackProfilePromise(result.profile);
  };

  const mintProfiled = (
    result: DirectDispatchResult,
    shape: readonly number[],
    dtype: string,
  ): Handle => {
    trackProfile(result);
    return mint(result.buffer, result.byteLength, shape, dtype);
  };

  const snapshot = (): WebGpuResourceSnapshot => {
    const latestProfile = passProfiles[passProfiles.length - 1];
    const timingMode = latestProfile?.timingMode
      ?? (pendingProfiles.size > 0
        ? (timestampQueryAvailable ? "timestamp-query" : "queue-completion")
        : "unavailable");
    return {
      timingMode,
      timestampQueryAvailable,
      currentOwnedGpuBytes,
      peakOwnedGpuBytes,
      totalAllocatedGpuBytes,
      totalReleasedGpuBytes,
      aliveHandleCount: handles.size,
      pendingProfileCount: pendingProfiles.size,
      passProfiles,
      ...(logicalTensorPlanPeakBytes !== undefined
        ? { logicalTensorPlanPeakBytes }
        : {}),
    };
  };

  const get = (handle: Handle, op: string): BufferRecord => {
    const rec = handles.get(handle);
    if (rec === undefined) {
      throw new KernelError(
        `WebGPU bridge: ${op} called with unknown handle ${handle}. ` +
          `It may have been released already.`,
      );
    }
    return rec;
  };

  return {
    upload(data: Uint8Array, shape: readonly number[], dtype: string): Handle {
      assertF32(dtype, "upload");
      // Reinterpret the input bytes as f32. Tolerate misaligned byteOffset
      // by copying into a fresh aligned ArrayBuffer when necessary.
      let f32: Float32Array;
      if ((data.byteOffset & 3) === 0 && data.byteLength % 4 === 0) {
        f32 = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
      } else {
        const copy = new Uint8Array(data.byteLength);
        copy.set(data);
        f32 = new Float32Array(copy.buffer);
      }
      const buf = uploadFloat32(device, f32);
      return mint(buf, data.byteLength, shape, dtype);
    },

    async materialize(handle: Handle, shape: readonly number[], dtype: string): Promise<Uint8Array> {
      assertF32(dtype, "materialize");
      // Shape is informational here; the buffer carries its own byte length.
      void shape;
      const rec = get(handle, "materialize");
      // materializeFloat32 returns Float32Array; reinterpret the bytes as
      // a Uint8Array view so the Python side reads raw f32 bytes. Pyodide
      // JSPI unwraps this Promise transparently — the Python Protocol
      // declares a sync return, JS callers must await.
      const f32 = await materializeFloat32(device, rec.buffer, rec.byteLength);
      return new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    },

    release(handle: Handle): void {
      const rec = handles.get(handle);
      if (rec === undefined) return; // idempotent
      releaseRecord(rec);
      handles.delete(handle);
    },

    matmul(a: Handle, b: Handle, m: number, k: number, n: number, dtype: string): Handle {
      assertF32(dtype, "matmul");
      const aRec = get(a, "matmul[A]");
      const bRec = get(b, "matmul[B]");
      // PRD-012a: tiled GEMM (16×16 workgroup-shared tiles) replaces the
      // naive triple-loop. Reduces DRAM reads from 2*M*N*K to ~M*N*K/8 —
      // the load-bearing perf win the megakernel-PRD was claiming.
      const result = matmulTiledDirect(
        device,
        aRec.buffer,
        bRec.buffer,
        m,
        k,
        n,
        profileFor("matmul_tiled"),
      );
      return mintProfiled(result, [m, n], dtype);
    },

    fused_elementwise(
      inputs: readonly Handle[],
      ops: readonly (readonly [string, number, number])[],
      shape: readonly number[],
      dtype: string,
    ): Handle {
      assertF32(dtype, "fused_elementwise");
      // PRD-012a: WGSL codegen for arbitrary elementwise chains. The ops
      // list is the same shape the Python fusion pass produces; we walk
      // it, emit a single compute shader, and pipeline-cache by hash.
      const inputBufs = inputs.map((h, i) => get(h, `fused_elementwise[in${i}]`).buffer);
      let total = 1;
      for (const d of shape) total *= d;
      if (total === 0) total = 1;
      const fusedOps: FusedOp[] = ops.map((o) => [o[0], o[1], o[2]] as FusedOp);
      const result = fusedElementwiseDirect(
        device,
        inputBufs,
        fusedOps,
        total,
        profileFor("fused_elementwise"),
      );
      return mintProfiled(result, shape, dtype);
    },

    cast(
      handle: Handle,
      srcDtype: string,
      dstDtype: string,
      shape: readonly number[],
    ): Handle {
      // v0: f32→f32 is a no-op (returns same buffer alias). Any other
      // cast is unsupported until f16/AMP lands in PRD-012a.
      if (srcDtype === "float32" && dstDtype === "float32") {
        // GPU-only copy via CopyBufferToBuffer (PRD-012a). The source
        // buffer's lifetime stays with the caller; the new handle owns
        // a freshly-allocated GPUBuffer with STORAGE | COPY_SRC usage so
        // downstream ops can read it and a future materialize() can
        // copy from it.
        const src = get(handle, "cast[f32→f32]");
        const impl = device.gpu;
        let n = 1;
        for (const d of shape) n *= d;
        const byteLength = (n || 1) * 4;
        const aligned = Math.ceil(byteLength / 4) * 4;
        const dst = impl.createBuffer({
          size: aligned,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        const encoder = impl.createCommandEncoder({ label: "bg-cast-f32-f32" });
        encoder.copyBufferToBuffer(src.buffer, 0, dst, 0, aligned);
        impl.queue.submit([encoder.finish()]);
        return mint(dst, byteLength, shape, dstDtype);
      }
      throw new KernelError(
        `WebGPU bridge: cast ${srcDtype}→${dstDtype} not supported in v0. ` +
          `f16/bf16 cast kernels land in PRD-012b.`,
      );
    },

    async flash_attention(
      q: Handle,
      k: Handle,
      v: Handle,
      mask: Handle | null,
      b: number,
      h: number,
      sq: number,
      sk: number,
      d: number,
      scale: number,
      dtype: string,
    ): Promise<Handle> {
      assertF32(dtype, "flash_attention");
      const qRec = get(q, "flash_attention[Q]");
      const kRec = get(k, "flash_attention[K]");
      const vRec = get(v, "flash_attention[V]");
      const maskRec = mask !== null ? get(mask, "flash_attention[mask]") : null;
      const result = rowWiseOnlineAttentionDirect(
        device,
        qRec.buffer,
        kRec.buffer,
        vRec.buffer,
        maskRec?.buffer ?? null,
        { B: b, H: h, Sq: sq, Sk: sk, D: d },
        scale,
        profileFor("flash_attention"),
      );
      return mintProfiled(result, [b, h, sq, d], dtype);
    },

    conv1d(
      input: Handle,
      weight: Handle,
      bias: Handle | null,
      n: number,
      cIn: number,
      lIn: number,
      cOut: number,
      k: number,
      stride: number,
      padding: number,
      dilation: number,
      groups: number,
      lOut: number,
      dtype: string,
    ): Handle {
      assertF32(dtype, "conv1d");
      validateConv1dShape(
        "conv1d",
        n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut,
      );
      const inputRec = get(input, "conv1d[input]");
      const weightRec = get(weight, "conv1d[weight]");
      const inputBuffers = [inputRec.buffer, weightRec.buffer];
      if (bias !== null) {
        inputBuffers.push(get(bias, "conv1d[bias]").buffer);
      }
      const outputLength = n * cOut * lOut;
      const params = conv1dParams(
        n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut,
      );
      const result = runDirect(device, {
        name: bias === null ? "conv1d_nobias" : "conv1d_bias",
        wgsl: conv1dWgsl(bias !== null),
        workgroupSize: [64, 1, 1],
      }, {
        inputBuffers,
        outputLength,
        params,
        dispatchCount: [outputLength, 1, 1],
        cacheKeySuffix: [
          bias === null ? "nobias" : "bias",
          n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut,
        ].join("_"),
        profile: profileFor(bias === null ? "conv1d_nobias" : "conv1d_bias"),
      });
      return mintProfiled(result, [n, cOut, lOut], dtype);
    },

    conv1d_backward_input(
      dy: Handle,
      weight: Handle,
      n: number,
      cIn: number,
      lIn: number,
      cOut: number,
      k: number,
      stride: number,
      padding: number,
      dilation: number,
      groups: number,
      lOut: number,
      dtype: string,
    ): Handle {
      assertF32(dtype, "conv1d_backward_input");
      validateConv1dShape(
        "conv1d_backward_input",
        n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut,
      );
      const dyRec = get(dy, "conv1d_backward_input[dy]");
      const weightRec = get(weight, "conv1d_backward_input[weight]");
      const outputLength = n * cIn * lIn;
      const params = conv1dParams(
        n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut,
      );
      const result = runDirect(device, {
        name: "conv1d_backward_input",
        wgsl: CONV1D_BACKWARD_INPUT_WGSL,
        workgroupSize: [64, 1, 1],
      }, {
        inputBuffers: [dyRec.buffer, weightRec.buffer],
        outputLength,
        params,
        dispatchCount: [outputLength, 1, 1],
        cacheKeySuffix: [n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut].join("_"),
        profile: profileFor("conv1d_backward_input"),
      });
      return mintProfiled(result, [n, cIn, lIn], dtype);
    },

    conv1d_backward_weight(
      dy: Handle,
      input: Handle,
      n: number,
      cIn: number,
      lIn: number,
      cOut: number,
      k: number,
      stride: number,
      padding: number,
      dilation: number,
      groups: number,
      lOut: number,
      dtype: string,
    ): Handle {
      assertF32(dtype, "conv1d_backward_weight");
      validateConv1dShape(
        "conv1d_backward_weight",
        n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut,
      );
      const dyRec = get(dy, "conv1d_backward_weight[dy]");
      const inputRec = get(input, "conv1d_backward_weight[input]");
      const cPerGroup = cIn / groups;
      const outputLength = cOut * cPerGroup * k;
      const params = conv1dParams(
        n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut,
      );
      const result = runDirect(device, {
        name: "conv1d_backward_weight",
        wgsl: CONV1D_BACKWARD_WEIGHT_WGSL,
        workgroupSize: [64, 1, 1],
      }, {
        inputBuffers: [dyRec.buffer, inputRec.buffer],
        outputLength,
        params,
        dispatchCount: [outputLength, 1, 1],
        cacheKeySuffix: [n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut].join("_"),
        profile: profileFor("conv1d_backward_weight"),
      });
      return mintProfiled(result, [cOut, cPerGroup, k], dtype);
    },

    conv1d_backward_bias(
      dy: Handle,
      n: number,
      cOut: number,
      lOut: number,
      dtype: string,
    ): Handle {
      assertF32(dtype, "conv1d_backward_bias");
      for (const [value, name] of [
        [n, "n"],
        [cOut, "cOut"],
        [lOut, "lOut"],
      ] as const) {
        assertPositiveInt(value, `conv1d_backward_bias.${name}`);
      }
      const dyRec = get(dy, "conv1d_backward_bias[dy]");
      const params = new Uint32Array([n, cOut, lOut, 0]);
      const result = runDirect(device, {
        name: "conv1d_backward_bias",
        wgsl: CONV1D_BACKWARD_BIAS_WGSL,
        workgroupSize: [64, 1, 1],
      }, {
        inputBuffers: [dyRec.buffer],
        outputLength: cOut,
        params,
        dispatchCount: [cOut, 1, 1],
        cacheKeySuffix: [n, cOut, lOut].join("_"),
        profile: profileFor("conv1d_backward_bias"),
      });
      return mintProfiled(result, [cOut], dtype);
    },

    conv2d(
      input: Handle,
      weight: Handle,
      bias: Handle | null,
      n: number,
      cIn: number,
      h: number,
      w: number,
      cOut: number,
      kh: number,
      kw: number,
      strideH: number,
      strideW: number,
      padH: number,
      padW: number,
      dilationH: number,
      dilationW: number,
      groups: number,
      outH: number,
      outW: number,
      dtype: string,
    ): Handle {
      assertF32(dtype, "conv2d");
      validateConv2dShape(
        "conv2d",
        n, cIn, h, w, cOut, kh, kw,
        strideH, strideW, padH, padW, dilationH, dilationW,
        groups, outH, outW,
      );
      const inputRec = get(input, "conv2d[input]");
      const weightRec = get(weight, "conv2d[weight]");
      const inputBuffers = [inputRec.buffer, weightRec.buffer];
      if (bias !== null) {
        inputBuffers.push(get(bias, "conv2d[bias]").buffer);
      }
      const outputLength = n * cOut * outH * outW;
      const params = conv2dParams(
        n, cIn, h, w, cOut, kh, kw,
        strideH, strideW, padH, padW, dilationH, dilationW,
        groups, outH, outW,
      );
      const result = runDirect(device, {
        name: bias === null ? "conv2d_nobias" : "conv2d_bias",
        wgsl: conv2dWgsl(bias !== null),
        workgroupSize: [64, 1, 1],
      }, {
        inputBuffers,
        outputLength,
        params,
        dispatchCount: [outputLength, 1, 1],
        cacheKeySuffix: [
          bias === null ? "nobias" : "bias",
          n, cIn, h, w, cOut, kh, kw,
          strideH, strideW, padH, padW, dilationH, dilationW,
          groups, outH, outW,
        ].join("_"),
        profile: profileFor(bias === null ? "conv2d_nobias" : "conv2d_bias"),
      });
      return mintProfiled(result, [n, cOut, outH, outW], dtype);
    },

    conv2d_backward_input(
      dy: Handle,
      weight: Handle,
      n: number,
      cIn: number,
      h: number,
      w: number,
      cOut: number,
      kh: number,
      kw: number,
      strideH: number,
      strideW: number,
      padH: number,
      padW: number,
      dilationH: number,
      dilationW: number,
      groups: number,
      outH: number,
      outW: number,
      dtype: string,
    ): Handle {
      assertF32(dtype, "conv2d_backward_input");
      validateConv2dShape(
        "conv2d_backward_input",
        n, cIn, h, w, cOut, kh, kw,
        strideH, strideW, padH, padW, dilationH, dilationW,
        groups, outH, outW,
      );
      const dyRec = get(dy, "conv2d_backward_input[dy]");
      const weightRec = get(weight, "conv2d_backward_input[weight]");
      const outputLength = n * cIn * h * w;
      const params = conv2dParams(
        n, cIn, h, w, cOut, kh, kw,
        strideH, strideW, padH, padW, dilationH, dilationW,
        groups, outH, outW,
      );
      const result = runDirect(device, {
        name: "conv2d_backward_input",
        wgsl: CONV2D_BACKWARD_INPUT_WGSL,
        workgroupSize: [64, 1, 1],
      }, {
        inputBuffers: [dyRec.buffer, weightRec.buffer],
        outputLength,
        params,
        dispatchCount: [outputLength, 1, 1],
        cacheKeySuffix: [
          n, cIn, h, w, cOut, kh, kw,
          strideH, strideW, padH, padW, dilationH, dilationW,
          groups, outH, outW,
        ].join("_"),
        profile: profileFor("conv2d_backward_input"),
      });
      return mintProfiled(result, [n, cIn, h, w], dtype);
    },

    conv2d_backward_weight(
      dy: Handle,
      input: Handle,
      n: number,
      cIn: number,
      h: number,
      w: number,
      cOut: number,
      kh: number,
      kw: number,
      strideH: number,
      strideW: number,
      padH: number,
      padW: number,
      dilationH: number,
      dilationW: number,
      groups: number,
      outH: number,
      outW: number,
      dtype: string,
    ): Handle {
      assertF32(dtype, "conv2d_backward_weight");
      validateConv2dShape(
        "conv2d_backward_weight",
        n, cIn, h, w, cOut, kh, kw,
        strideH, strideW, padH, padW, dilationH, dilationW,
        groups, outH, outW,
      );
      const dyRec = get(dy, "conv2d_backward_weight[dy]");
      const inputRec = get(input, "conv2d_backward_weight[input]");
      const cPerGroup = cIn / groups;
      const outputLength = cOut * cPerGroup * kh * kw;
      const params = conv2dParams(
        n, cIn, h, w, cOut, kh, kw,
        strideH, strideW, padH, padW, dilationH, dilationW,
        groups, outH, outW,
      );
      const result = runDirect(device, {
        name: "conv2d_backward_weight",
        wgsl: CONV2D_BACKWARD_WEIGHT_WGSL,
        workgroupSize: [64, 1, 1],
      }, {
        inputBuffers: [dyRec.buffer, inputRec.buffer],
        outputLength,
        params,
        dispatchCount: [outputLength, 1, 1],
        cacheKeySuffix: [
          n, cIn, h, w, cOut, kh, kw,
          strideH, strideW, padH, padW, dilationH, dilationW,
          groups, outH, outW,
        ].join("_"),
        profile: profileFor("conv2d_backward_weight"),
      });
      return mintProfiled(result, [cOut, cPerGroup, kh, kw], dtype);
    },

    conv2d_backward_bias(
      dy: Handle,
      n: number,
      cOut: number,
      outH: number,
      outW: number,
      dtype: string,
    ): Handle {
      assertF32(dtype, "conv2d_backward_bias");
      for (const [value, name] of [
        [n, "n"],
        [cOut, "cOut"],
        [outH, "outH"],
        [outW, "outW"],
      ] as const) {
        assertPositiveInt(value, `conv2d_backward_bias.${name}`);
      }
      const dyRec = get(dy, "conv2d_backward_bias[dy]");
      const params = new Uint32Array([n, cOut, outH, outW]);
      const result = runDirect(device, {
        name: "conv2d_backward_bias",
        wgsl: CONV2D_BACKWARD_BIAS_WGSL,
        workgroupSize: [64, 1, 1],
      }, {
        inputBuffers: [dyRec.buffer],
        outputLength: cOut,
        params,
        dispatchCount: [cOut, 1, 1],
        cacheKeySuffix: [n, cOut, outH, outW].join("_"),
        profile: profileFor("conv2d_backward_bias"),
      });
      return mintProfiled(result, [cOut], dtype);
    },

    run_user_kernel(
      inputs: readonly Handle[],
      wgsl: string,
      name: string,
      hash: string,
      workgroupSize: readonly [number, number, number],
      dispatchShape: readonly [number, number, number],
      outputLength: number,
      outputShape: readonly number[],
      dtype: string,
    ): Handle {
      assertF32(dtype, "run_user_kernel");
      const inputBufs = inputs.map(
        (h, i) => get(h, `run_user_kernel[in${i}]`).buffer,
      );
      const desc: KernelDescriptor = {
        // Cache key prefix carries the first 8 hash chars so kernels with
        // the same `name` but different WGSL get distinct cache entries.
        name: `user_${hash.slice(0, 8)}_${name}`,
        wgsl,
        workgroupSize: [workgroupSize[0], workgroupSize[1], workgroupSize[2]],
      };
      // v0: no uniform params. Users bake constants into WGSL.
      const result = runDirect(device, desc, {
        inputBuffers: inputBufs,
        outputLength,
        params: new Uint32Array(0),
        dispatchCount: [dispatchShape[0], dispatchShape[1], dispatchShape[2]],
        cacheKeySuffix: hash,
        profile: profileFor(`user_kernel:${name}`),
      });
      return mintProfiled(result, outputShape, dtype);
    },

    async run_tensor_plan(
      plan: unknown,
      inputs: readonly unknown[],
      dtype: string,
    ): Promise<Uint8Array> {
      assertF32(dtype, "run_tensor_plan");
      const result = await runTensorGpuPlan(
        device,
        plan,
        normalizeTensorPlanInputs(inputs, get),
      );
      if (profilingEnabled) {
        for (const profile of result.profiles) trackProfilePromise(profile);
      }
      return new Uint8Array(
        result.data.buffer,
        result.data.byteOffset,
        result.data.byteLength,
      );
    },

    run_tensor_plan_resident(
      plan: unknown,
      inputs: readonly unknown[],
      dtype: string,
    ): Handle {
      assertF32(dtype, "run_tensor_plan_resident");
      const result = runTensorGpuPlanResident(
        device,
        plan,
        normalizeTensorPlanInputs(inputs, get),
      );
      logicalTensorPlanPeakBytes = Math.max(
        logicalTensorPlanPeakBytes ?? 0,
        result.peakLiveBytes,
      );
      if (profilingEnabled) {
        for (const profile of result.profiles) trackProfilePromise(profile);
      }
      return mint(result.buffer, result.byteLength, result.shape, dtype);
    },

    aliveHandleCount(): number {
      return handles.size;
    },

    resourceSnapshot(): WebGpuResourceSnapshot {
      return snapshot();
    },

    async flushProfiles(): Promise<WebGpuResourceSnapshot> {
      await Promise.allSettled([...pendingProfiles]);
      return snapshot();
    },
  };
}

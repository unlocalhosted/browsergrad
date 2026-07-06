import { fusedElementwiseDirect, type FusedOp } from "./kernels/fused_elementwise.js";
import { matmulTiledDirect } from "./kernels/matmul_tiled.js";
import {
  materializeFloat32,
  runDirect,
  uploadFloat32,
  type DirectDispatchResult,
} from "./runner.js";
import { KernelError, type KernelDevice } from "./types.js";

export type TensorPlanOp =
  | "BUFFER"
  | "LOAD"
  | "CONST"
  | "CAST"
  | "ADD"
  | "MUL"
  | "DIV"
  | "NEG"
  | "EXP"
  | "LOG"
  | "MATMUL"
  | "REDUCE"
  | "RESHAPE"
  | "PERMUTE"
  | "BROADCAST_TO"
  | "CONV1D"
  | "CONV1D_BACKWARD_INPUT"
  | "CONV1D_BACKWARD_WEIGHT"
  | "CONV1D_BACKWARD_BIAS"
  | "CONV2D"
  | "CONV2D_BACKWARD_INPUT"
  | "CONV2D_BACKWARD_WEIGHT"
  | "CONV2D_BACKWARD_BIAS"
  | "CONV_TRANSPOSE2D"
  | "CONV_TRANSPOSE2D_BACKWARD_INPUT"
  | "CONV_TRANSPOSE2D_BACKWARD_WEIGHT"
  | "CONV_TRANSPOSE2D_BACKWARD_BIAS"
  | "CONV3D"
  | "CONV3D_BACKWARD_INPUT"
  | "CONV3D_BACKWARD_WEIGHT"
  | "CONV3D_BACKWARD_BIAS"
  | "LAYER_NORM"
  | "LAYER_NORM_BACKWARD_INPUT"
  | "LAYER_NORM_BACKWARD_WEIGHT"
  | "LAYER_NORM_BACKWARD_BIAS"
  | "SGD_UPDATE"
  | "ADAMW_UPDATE_M"
  | "ADAMW_UPDATE_V"
  | "ADAMW_UPDATE_PARAM"
  | "ADAM_UPDATE_M"
  | "ADAM_UPDATE_V"
  | "ADAM_UPDATE_PARAM";

export interface TensorPlanStep {
  readonly step: number;
  readonly valueId: number;
  readonly op: TensorPlanOp;
  readonly inputIds: readonly number[];
  readonly shape: readonly number[];
  readonly dtype: "float32";
  readonly arg?: unknown;
}

export interface TensorPlanBuffer {
  readonly valueId: number;
  readonly op: string;
  readonly shape: readonly number[];
  readonly dtype: "float32";
  readonly bytes: number;
  readonly firstStep: number;
  readonly lastStep: number;
  readonly materialize: boolean;
}

export interface TensorGpuPlan {
  readonly steps: readonly TensorPlanStep[];
  readonly buffers: readonly TensorPlanBuffer[];
  readonly rootId: number;
  readonly materializationBoundary: "root";
  readonly peakLiveBytes: number;
  readonly hasCustomOps: false;
}

export interface TensorPlanInput {
  readonly valueId: number;
  readonly data?: Float32Array;
  readonly resident?: {
    readonly buffer: GPUBuffer;
    readonly byteLength: number;
  };
}

export interface TensorPlanRunResult {
  readonly data: Float32Array;
  readonly shape: readonly number[];
  readonly peakLiveBytes: number;
  readonly materializedValueId: number;
}

export interface TensorPlanResidentResult {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
  readonly shape: readonly number[];
  readonly peakLiveBytes: number;
  readonly residentValueId: number;
}

type RawRecord = Record<string, unknown>;

interface ResidentValue {
  readonly buffer: GPUBuffer;
  readonly shape: readonly number[];
  readonly owns: boolean;
  readonly byteLength: number;
}

export function normalizeTensorGpuPlan(raw: unknown): TensorGpuPlan {
  const obj = expectRecord(raw, "tensor plan");
  const steps = expectArray(obj.steps, "tensor plan.steps").map((step, i) => {
    const s = expectRecord(step, `tensor plan.steps[${i}]`);
    return {
      step: expectNumber(s.step, `steps[${i}].step`),
      valueId: expectNumber(s.valueId ?? s.value_id, `steps[${i}].valueId`),
      op: expectOp(s.op, `steps[${i}].op`),
      inputIds: expectArray(s.inputIds ?? s.input_ids, `steps[${i}].inputIds`).map((v, j) =>
        expectNumber(v, `steps[${i}].inputIds[${j}]`),
      ),
      shape: expectArray(s.shape, `steps[${i}].shape`).map((v, j) =>
        expectNumber(v, `steps[${i}].shape[${j}]`),
      ),
      dtype: expectFloat32(s.dtype, `steps[${i}].dtype`),
      arg: s.arg,
    };
  });
  const buffers = expectArray(obj.buffers, "tensor plan.buffers").map((buffer, i) => {
    const b = expectRecord(buffer, `tensor plan.buffers[${i}]`);
    return {
      valueId: expectNumber(b.valueId ?? b.value_id, `buffers[${i}].valueId`),
      op: expectString(b.op, `buffers[${i}].op`),
      shape: expectArray(b.shape, `buffers[${i}].shape`).map((v, j) =>
        expectNumber(v, `buffers[${i}].shape[${j}]`),
      ),
      dtype: expectFloat32(b.dtype, `buffers[${i}].dtype`),
      bytes: expectNumber(b.bytes, `buffers[${i}].bytes`),
      firstStep: expectNumber(b.firstStep ?? b.first_step, `buffers[${i}].firstStep`),
      lastStep: expectNumber(b.lastStep ?? b.last_step, `buffers[${i}].lastStep`),
      materialize: expectBoolean(b.materialize, `buffers[${i}].materialize`),
    };
  });
  return {
    steps,
    buffers,
    rootId: expectNumber(obj.rootId ?? obj.root_id, "tensor plan.rootId"),
    materializationBoundary: expectRootBoundary(
      obj.materializationBoundary ?? obj.materialization_boundary,
    ),
    peakLiveBytes: expectNumber(
      obj.peakLiveBytes ?? obj.peak_live_bytes,
      "tensor plan.peakLiveBytes",
    ),
    hasCustomOps: expectFalse(obj.hasCustomOps ?? obj.has_custom_ops, "tensor plan.hasCustomOps"),
  };
}

export async function runTensorGpuPlan(
  device: KernelDevice,
  rawPlan: TensorGpuPlan | unknown,
  inputs: readonly TensorPlanInput[],
): Promise<TensorPlanRunResult> {
  const { root, rootMeta, owned, peakLiveBytes, rootId } = executeTensorGpuPlan(
    device,
    rawPlan,
    inputs,
  );
  try {
    const data = await materializeFloat32(device, root.buffer, root.byteLength);
    return {
      data,
      shape: rootMeta.shape,
      peakLiveBytes,
      materializedValueId: rootId,
    };
  } finally {
    for (const buffer of owned) buffer.destroy();
  }
}

export function runTensorGpuPlanResident(
  device: KernelDevice,
  rawPlan: TensorGpuPlan | unknown,
  inputs: readonly TensorPlanInput[],
): TensorPlanResidentResult {
  const { root, rootMeta, owned, peakLiveBytes, rootId } = executeTensorGpuPlan(
    device,
    rawPlan,
    inputs,
  );
  if (!owned.has(root.buffer)) {
    for (const buffer of owned) buffer.destroy();
    throw new KernelError(
      `tensor plan root ${rootId} aliases an input buffer; resident output requires an owned root`,
    );
  }
  owned.delete(root.buffer);
  for (const buffer of owned) buffer.destroy();
  return {
    buffer: root.buffer,
    byteLength: root.byteLength,
    shape: rootMeta.shape,
    peakLiveBytes,
    residentValueId: rootId,
  };
}

function executeTensorGpuPlan(
  device: KernelDevice,
  rawPlan: TensorGpuPlan | unknown,
  inputs: readonly TensorPlanInput[],
): {
  readonly root: ResidentValue;
  readonly rootMeta: TensorPlanBuffer;
  readonly owned: Set<GPUBuffer>;
  readonly peakLiveBytes: number;
  readonly rootId: number;
} {
  const plan = normalizeTensorGpuPlan(rawPlan);
  validatePlan(plan);
  const inputData = new Map<number, TensorPlanInput>();
  for (const input of inputs) {
    if (inputData.has(input.valueId)) {
      throw new KernelError(`tensor plan input ${input.valueId} provided twice`);
    }
    inputData.set(input.valueId, input);
  }

  const buffersByValue = new Map<number, TensorPlanBuffer>();
  for (const b of plan.buffers) buffersByValue.set(b.valueId, b);

  const values = new Map<number, ResidentValue>();
  const owned = new Set<GPUBuffer>();

  try {
    for (const step of plan.steps) {
      const value = executeStep(device, step, values, inputData);
      values.set(step.valueId, value);
      if (value.owns) owned.add(value.buffer);
    }
  } catch (err) {
    for (const buffer of owned) buffer.destroy();
    throw err;
  }

  const root = values.get(plan.rootId);
  const rootMeta = buffersByValue.get(plan.rootId);
  if (!root || !rootMeta) {
    for (const buffer of owned) buffer.destroy();
    throw new KernelError(`tensor plan root ${plan.rootId} was not produced`);
  }
  return {
    root,
    rootMeta,
    owned,
    peakLiveBytes: plan.peakLiveBytes,
    rootId: plan.rootId,
  };
}

function executeStep(
  device: KernelDevice,
  step: TensorPlanStep,
  values: Map<number, ResidentValue>,
  inputData: Map<number, TensorPlanInput>,
): ResidentValue {
  switch (step.op) {
    case "BUFFER": {
      const input = inputData.get(step.valueId);
      if (!input) throw new KernelError(`tensor plan missing BUFFER input ${step.valueId}`);
      if (input.resident) {
        validateNumel(input.resident.byteLength / 4, step.shape, `BUFFER ${step.valueId}`);
        return {
          buffer: input.resident.buffer,
          shape: step.shape,
          owns: false,
          byteLength: input.resident.byteLength,
        };
      }
      if (!input.data) {
        throw new KernelError(`tensor plan BUFFER ${step.valueId} input has neither data nor resident buffer`);
      }
      validateNumel(input.data.length, step.shape, `BUFFER ${step.valueId}`);
      const buffer = uploadFloat32(device, input.data);
      return { buffer, shape: step.shape, owns: true, byteLength: input.data.byteLength };
    }
    case "LOAD": {
      const src = requireValue(values, step.inputIds[0], step.op);
      return { ...src, shape: step.shape, owns: false };
    }
    case "MATMUL": {
      const a = requireValue(values, step.inputIds[0], step.op);
      const b = requireValue(values, step.inputIds[1], step.op);
      if (a.shape.length !== 2 || b.shape.length !== 2 || step.shape.length !== 2) {
        throw new KernelError("tensor plan MATMUL supports 2-D shapes in v0");
      }
      const m = a.shape[0]!;
      const k = a.shape[1]!;
      const k2 = b.shape[0]!;
      const n = b.shape[1]!;
      if (k !== k2 || step.shape[0] !== m || step.shape[1] !== n) {
        throw new KernelError(
          `tensor plan MATMUL shape mismatch: ${shapeStr(a.shape)} @ ${shapeStr(b.shape)} -> ${shapeStr(step.shape)}`,
        );
      }
      return fromDirect(step, matmulTiledDirect(device, a.buffer, b.buffer, m, k, n));
    }
    case "ADD":
    case "MUL":
    case "DIV": {
      return elementwise(step, values, [[step.op, -1, -2]], 2, device);
    }
    case "NEG":
    case "EXP":
    case "LOG": {
      return elementwise(step, values, [[step.op, -1, -1]], 1, device);
    }
    case "CAST": {
      const src = requireValue(values, step.inputIds[0], step.op);
      if (step.dtype !== "float32") {
        throw new KernelError(`tensor plan CAST only supports float32 in v0`);
      }
      return { ...src, shape: step.shape, owns: false };
    }
    case "RESHAPE": {
      const src = requireValue(values, step.inputIds[0], step.op);
      if (numel(src.shape) !== numel(step.shape)) {
        throw new KernelError(
          `tensor plan RESHAPE cannot change numel: ${shapeStr(src.shape)} -> ${shapeStr(step.shape)}`,
        );
      }
      return { ...src, shape: step.shape, owns: false };
    }
    case "PERMUTE": {
      const src = requireValue(values, step.inputIds[0], step.op);
      const axes = expectAxes(step.arg, src.shape.length, step.op);
      return fromDirect(step, permuteDirect(device, src, step.shape, axes));
    }
    case "BROADCAST_TO": {
      const src = requireValue(values, step.inputIds[0], step.op);
      return fromDirect(step, broadcastDirect(device, src, step.shape));
    }
    case "REDUCE": {
      const src = requireValue(values, step.inputIds[0], step.op);
      const spec = expectReduceSpec(step.arg, src.shape.length);
      return fromDirect(step, reduceDirect(device, src, step.shape, spec));
    }
    case "CONV1D": {
      const x = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      const bias = step.inputIds.length > 2
        ? requireValue(values, step.inputIds[2], step.op)
        : null;
      return fromDirect(step, conv1dDirect(device, x, weight, bias, step.shape, step.arg));
    }
    case "CONV1D_BACKWARD_INPUT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, conv1dBackwardInputDirect(device, dy, weight, step.shape, step.arg));
    }
    case "CONV1D_BACKWARD_WEIGHT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const x = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, conv1dBackwardWeightDirect(device, dy, x, step.shape, step.arg));
    }
    case "CONV1D_BACKWARD_BIAS": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      return fromDirect(step, conv1dBackwardBiasDirect(device, dy, step.shape, step.arg));
    }
    case "CONV2D": {
      const x = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      const bias = step.inputIds.length > 2
        ? requireValue(values, step.inputIds[2], step.op)
        : null;
      return fromDirect(step, conv2dDirect(device, x, weight, bias, step.shape, step.arg));
    }
    case "CONV2D_BACKWARD_INPUT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, conv2dBackwardInputDirect(device, dy, weight, step.shape, step.arg));
    }
    case "CONV2D_BACKWARD_WEIGHT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const x = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, conv2dBackwardWeightDirect(device, dy, x, step.shape, step.arg));
    }
    case "CONV2D_BACKWARD_BIAS": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      return fromDirect(step, conv2dBackwardBiasDirect(device, dy, step.shape, step.arg));
    }
    case "CONV_TRANSPOSE2D": {
      const x = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      const bias = step.inputIds.length > 2
        ? requireValue(values, step.inputIds[2], step.op)
        : null;
      return fromDirect(step, convTranspose2dDirect(device, x, weight, bias, step.shape, step.arg));
    }
    case "CONV_TRANSPOSE2D_BACKWARD_INPUT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, convTranspose2dBackwardInputDirect(device, dy, weight, step.shape, step.arg));
    }
    case "CONV_TRANSPOSE2D_BACKWARD_WEIGHT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const x = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, convTranspose2dBackwardWeightDirect(device, dy, x, step.shape, step.arg));
    }
    case "CONV_TRANSPOSE2D_BACKWARD_BIAS": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      return fromDirect(step, convTranspose2dBackwardBiasDirect(device, dy, step.shape, step.arg));
    }
    case "CONV3D": {
      const x = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      const bias = step.inputIds.length > 2
        ? requireValue(values, step.inputIds[2], step.op)
        : null;
      return fromDirect(step, conv3dDirect(device, x, weight, bias, step.shape, step.arg));
    }
    case "CONV3D_BACKWARD_INPUT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, conv3dBackwardInputDirect(device, dy, weight, step.shape, step.arg));
    }
    case "CONV3D_BACKWARD_WEIGHT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const x = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, conv3dBackwardWeightDirect(device, dy, x, step.shape, step.arg));
    }
    case "CONV3D_BACKWARD_BIAS": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      return fromDirect(step, conv3dBackwardBiasDirect(device, dy, step.shape, step.arg));
    }
    case "LAYER_NORM": {
      const x = requireValue(values, step.inputIds[0], step.op);
      const weight = requireValue(values, step.inputIds[1], step.op);
      const bias = requireValue(values, step.inputIds[2], step.op);
      return fromDirect(step, layerNormDirect(device, x, weight, bias, step.shape, step.arg));
    }
    case "LAYER_NORM_BACKWARD_INPUT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const x = requireValue(values, step.inputIds[1], step.op);
      const weight = requireValue(values, step.inputIds[2], step.op);
      return fromDirect(step, layerNormBackwardInputDirect(device, dy, x, weight, step.shape, step.arg));
    }
    case "LAYER_NORM_BACKWARD_WEIGHT": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      const x = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, layerNormBackwardWeightDirect(device, dy, x, step.shape, step.arg));
    }
    case "LAYER_NORM_BACKWARD_BIAS": {
      const dy = requireValue(values, step.inputIds[0], step.op);
      return fromDirect(step, layerNormBackwardBiasDirect(device, dy, step.shape, step.arg));
    }
    case "SGD_UPDATE": {
      const param = requireValue(values, step.inputIds[0], step.op);
      const grad = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, sgdUpdateDirect(device, param, grad, step.shape, step.arg));
    }
    case "ADAMW_UPDATE_M": {
      const m = requireValue(values, step.inputIds[0], step.op);
      const grad = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, adamwUpdateMDirect(device, m, grad, step.shape, step.arg));
    }
    case "ADAMW_UPDATE_V": {
      const v = requireValue(values, step.inputIds[0], step.op);
      const grad = requireValue(values, step.inputIds[1], step.op);
      return fromDirect(step, adamwUpdateVDirect(device, v, grad, step.shape, step.arg));
    }
    case "ADAMW_UPDATE_PARAM": {
      const param = requireValue(values, step.inputIds[0], step.op);
      const grad = requireValue(values, step.inputIds[1], step.op);
      const m = requireValue(values, step.inputIds[2], step.op);
      const v = requireValue(values, step.inputIds[3], step.op);
      return fromDirect(step, adamwUpdateParamDirect(device, param, grad, m, v, step.shape, step.arg));
    }
    case "ADAM_UPDATE_M": {
      const param = requireValue(values, step.inputIds[0], step.op);
      const grad = requireValue(values, step.inputIds[1], step.op);
      const m = requireValue(values, step.inputIds[2], step.op);
      return fromDirect(step, adamUpdateMDirect(device, param, grad, m, step.shape, step.arg));
    }
    case "ADAM_UPDATE_V": {
      const param = requireValue(values, step.inputIds[0], step.op);
      const grad = requireValue(values, step.inputIds[1], step.op);
      const v = requireValue(values, step.inputIds[2], step.op);
      return fromDirect(step, adamUpdateVDirect(device, param, grad, v, step.shape, step.arg));
    }
    case "ADAM_UPDATE_PARAM": {
      const param = requireValue(values, step.inputIds[0], step.op);
      const m = requireValue(values, step.inputIds[1], step.op);
      const v = requireValue(values, step.inputIds[2], step.op);
      return fromDirect(step, adamUpdateParamDirect(device, param, m, v, step.shape, step.arg));
    }
    case "CONST":
      throw new KernelError("tensor plan CONST lowering needs scalar fill kernel");
    default:
      assertNever(step.op);
  }
}

interface ReduceSpec {
  readonly op: "sum" | "mean";
  readonly axes: readonly number[];
}

interface Conv1dArg {
  readonly n: number;
  readonly cIn: number;
  readonly lIn: number;
  readonly cOut: number;
  readonly k: number;
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
  readonly groups: number;
  readonly lOut: number;
}

interface Conv2dArg {
  readonly n: number;
  readonly cIn: number;
  readonly h: number;
  readonly w: number;
  readonly cOut: number;
  readonly kh: number;
  readonly kw: number;
  readonly strideH: number;
  readonly strideW: number;
  readonly padH: number;
  readonly padW: number;
  readonly dilationH: number;
  readonly dilationW: number;
  readonly groups: number;
  readonly outH: number;
  readonly outW: number;
}

interface ConvTranspose2dArg {
  readonly n: number;
  readonly cIn: number;
  readonly h: number;
  readonly w: number;
  readonly cOut: number;
  readonly cOutPerGroup: number;
  readonly kh: number;
  readonly kw: number;
  readonly strideH: number;
  readonly strideW: number;
  readonly padH: number;
  readonly padW: number;
  readonly outputPadH: number;
  readonly outputPadW: number;
  readonly dilationH: number;
  readonly dilationW: number;
  readonly groups: number;
  readonly outH: number;
  readonly outW: number;
}

interface Conv3dArg {
  readonly n: number;
  readonly cIn: number;
  readonly d: number;
  readonly h: number;
  readonly w: number;
  readonly cOut: number;
  readonly kd: number;
  readonly kh: number;
  readonly kw: number;
  readonly strideD: number;
  readonly strideH: number;
  readonly strideW: number;
  readonly padD: number;
  readonly padH: number;
  readonly padW: number;
  readonly dilationD: number;
  readonly dilationH: number;
  readonly dilationW: number;
  readonly groups: number;
  readonly outD: number;
  readonly outH: number;
  readonly outW: number;
}

interface SgdUpdateArg {
  readonly lr: number;
  readonly weightDecay: number;
}

interface AdamwUpdateArg {
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  readonly step: number;
}

interface AdamUpdateArg {
  readonly lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  readonly step: number;
}

interface LayerNormArg {
  readonly normalizedShape: readonly number[];
  readonly rows: number;
  readonly cols: number;
  readonly eps: number;
}

const PERMUTE_WGSL = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;

struct Params {
  input_shape: vec4<u32>,
  output_shape: vec4<u32>,
  axes: vec4<u32>,
  output_total: u32,
  rank: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(2) var<uniform> P: Params;

fn coord_to_index(c: vec4<u32>, shape: vec4<u32>, rank: u32) -> u32 {
  var idx = 0u;
  for (var i = 0u; i < rank; i = i + 1u) {
    idx = idx * shape[i] + c[i];
  }
  return idx;
}

fn linear_to_coord(index: u32, shape: vec4<u32>, rank: u32) -> vec4<u32> {
  var rem = index;
  var coord = vec4<u32>(0u);
  for (var off = 0u; off < rank; off = off + 1u) {
    let i = rank - 1u - off;
    let dim = shape[i];
    coord[i] = rem % dim;
    rem = rem / dim;
  }
  return coord;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_idx = gid.x;
  if (out_idx >= P.output_total) { return; }
  let out_coord = linear_to_coord(out_idx, P.output_shape, P.rank);
  var in_coord = vec4<u32>(0u);
  for (var i = 0u; i < P.rank; i = i + 1u) {
    in_coord[P.axes[i]] = out_coord[i];
  }
  let in_idx = coord_to_index(in_coord, P.input_shape, P.rank);
  Y[out_idx] = X[in_idx];
}
`;

const BROADCAST_WGSL = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;

struct Params {
  input_shape: vec4<u32>,
  output_shape: vec4<u32>,
  input_rank: u32,
  output_rank: u32,
  output_total: u32,
  pad0: u32,
};
@group(0) @binding(2) var<uniform> P: Params;

fn coord_to_index(c: vec4<u32>, shape: vec4<u32>, rank: u32) -> u32 {
  var idx = 0u;
  for (var i = 0u; i < rank; i = i + 1u) {
    idx = idx * shape[i] + c[i];
  }
  return idx;
}

fn linear_to_coord(index: u32, shape: vec4<u32>, rank: u32) -> vec4<u32> {
  var rem = index;
  var coord = vec4<u32>(0u);
  for (var off = 0u; off < rank; off = off + 1u) {
    let i = rank - 1u - off;
    let dim = shape[i];
    coord[i] = rem % dim;
    rem = rem / dim;
  }
  return coord;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_idx = gid.x;
  if (out_idx >= P.output_total) { return; }
  let out_coord = linear_to_coord(out_idx, P.output_shape, P.output_rank);
  var in_coord = vec4<u32>(0u);
  let rank_delta = P.output_rank - P.input_rank;
  for (var i = 0u; i < P.input_rank; i = i + 1u) {
    let out_axis = i + rank_delta;
    in_coord[i] = select(out_coord[out_axis], 0u, P.input_shape[i] == 1u);
  }
  let in_idx = coord_to_index(in_coord, P.input_shape, P.input_rank);
  Y[out_idx] = X[in_idx];
}
`;

const REDUCE_WGSL = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;

struct Params {
  input_shape: vec4<u32>,
  output_shape: vec4<u32>,
  reduce_mask: vec4<u32>,
  out_to_input_axis: vec4<u32>,
  input_rank: u32,
  output_rank: u32,
  output_total: u32,
  reduce_total: u32,
  is_mean: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
@group(0) @binding(2) var<uniform> P: Params;

fn coord_to_index(c: vec4<u32>, shape: vec4<u32>, rank: u32) -> u32 {
  var idx = 0u;
  for (var i = 0u; i < rank; i = i + 1u) {
    idx = idx * shape[i] + c[i];
  }
  return idx;
}

fn linear_to_coord(index: u32, shape: vec4<u32>, rank: u32) -> vec4<u32> {
  var rem = index;
  var coord = vec4<u32>(0u);
  for (var off = 0u; off < rank; off = off + 1u) {
    let i = rank - 1u - off;
    let dim = shape[i];
    coord[i] = rem % dim;
    rem = rem / dim;
  }
  return coord;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_idx = gid.x;
  if (out_idx >= P.output_total) { return; }
  let out_coord = linear_to_coord(out_idx, P.output_shape, P.output_rank);
  var base = vec4<u32>(0u);
  for (var out_axis = 0u; out_axis < P.output_rank; out_axis = out_axis + 1u) {
    base[P.out_to_input_axis[out_axis]] = out_coord[out_axis];
  }

  let l0 = select(1u, P.input_shape[0], P.reduce_mask[0] == 1u);
  let l1 = select(1u, P.input_shape[1], P.reduce_mask[1] == 1u);
  let l2 = select(1u, P.input_shape[2], P.reduce_mask[2] == 1u);
  let l3 = select(1u, P.input_shape[3], P.reduce_mask[3] == 1u);
  var acc = 0.0;
  for (var r0 = 0u; r0 < l0; r0 = r0 + 1u) {
    for (var r1 = 0u; r1 < l1; r1 = r1 + 1u) {
      for (var r2 = 0u; r2 < l2; r2 = r2 + 1u) {
        for (var r3 = 0u; r3 < l3; r3 = r3 + 1u) {
          var c = base;
          if (P.reduce_mask[0] == 1u) { c[0] = r0; }
          if (P.reduce_mask[1] == 1u) { c[1] = r1; }
          if (P.reduce_mask[2] == 1u) { c[2] = r2; }
          if (P.reduce_mask[3] == 1u) { c[3] = r3; }
          acc = acc + X[coord_to_index(c, P.input_shape, P.input_rank)];
        }
      }
    }
  }
  if (P.is_mean == 1u) {
    acc = acc / f32(P.reduce_total);
  }
  Y[out_idx] = acc;
}
`;

function permuteDirect(
  device: KernelDevice,
  src: ResidentValue,
  shape: readonly number[],
  axes: readonly number[],
): DirectDispatchResult {
  if (src.shape.length > 4) throw new KernelError("tensor plan PERMUTE supports rank <= 4 in v0");
  if (shape.length !== src.shape.length) {
    throw new KernelError("tensor plan PERMUTE output rank must match input rank");
  }
  const params = new Uint32Array([
    ...pad4(src.shape),
    ...pad4(shape),
    ...pad4(axes),
    numel(shape),
    shape.length,
    0,
    0,
  ]);
  return runDirect(device, {
    name: "tensor_plan_permute",
    wgsl: PERMUTE_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [src.buffer],
    outputLength: numel(shape),
    params,
    dispatchCount: [numel(shape), 1, 1],
    cacheKeySuffix: `${shapeStr(src.shape)}_${shapeStr(shape)}_${axes.join("_")}`,
  });
}

function broadcastDirect(
  device: KernelDevice,
  src: ResidentValue,
  shape: readonly number[],
): DirectDispatchResult {
  if (src.shape.length > 4 || shape.length > 4) {
    throw new KernelError("tensor plan BROADCAST_TO supports rank <= 4 in v0");
  }
  if (src.shape.length > shape.length) {
    throw new KernelError("tensor plan BROADCAST_TO cannot reduce rank");
  }
  const delta = shape.length - src.shape.length;
  for (let i = 0; i < src.shape.length; i++) {
    const srcDim = src.shape[i]!;
    const outDim = shape[i + delta]!;
    if (srcDim !== 1 && srcDim !== outDim) {
      throw new KernelError(
        `tensor plan BROADCAST_TO incompatible dim ${i}: ${srcDim} -> ${outDim}`,
      );
    }
  }
  const params = new Uint32Array([
    ...pad4(src.shape),
    ...pad4(shape),
    src.shape.length,
    shape.length,
    numel(shape),
    0,
  ]);
  return runDirect(device, {
    name: "tensor_plan_broadcast",
    wgsl: BROADCAST_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [src.buffer],
    outputLength: numel(shape),
    params,
    dispatchCount: [numel(shape), 1, 1],
    cacheKeySuffix: `${shapeStr(src.shape)}_${shapeStr(shape)}`,
  });
}

function reduceDirect(
  device: KernelDevice,
  src: ResidentValue,
  shape: readonly number[],
  spec: ReduceSpec,
): DirectDispatchResult {
  if (src.shape.length > 4 || shape.length > 4) {
    throw new KernelError("tensor plan REDUCE supports rank <= 4 in v0");
  }
  const reduceMask = Array.from({ length: src.shape.length }, () => 0);
  for (const axis of spec.axes) reduceMask[axis] = 1;
  const outToInputAxis: number[] = [];
  for (let axis = 0; axis < src.shape.length; axis++) {
    if (reduceMask[axis] === 0) outToInputAxis.push(axis);
  }
  let reduceTotal = 1;
  for (const axis of spec.axes) reduceTotal *= src.shape[axis]!;
  const params = new Uint32Array([
    ...pad4(src.shape),
    ...pad4(shape),
    ...pad4(reduceMask),
    ...pad4(outToInputAxis),
    src.shape.length,
    shape.length,
    numel(shape),
    reduceTotal,
    spec.op === "mean" ? 1 : 0,
    0,
    0,
    0,
  ]);
  return runDirect(device, {
    name: "tensor_plan_reduce",
    wgsl: REDUCE_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [src.buffer],
    outputLength: numel(shape),
    params,
    dispatchCount: [numel(shape), 1, 1],
    cacheKeySuffix: `${spec.op}_${shapeStr(src.shape)}_${shapeStr(shape)}_${spec.axes.join("_")}`,
  });
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
  if (idx >= total) { return; }
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
      if (ih_unpadded < 0 || ih_unpadded >= i32(P.h)) { continue; }
      for (var c = 0u; c < P.kw; c = c + 1u) {
        let iw_unpadded = i32(ow * P.stride_w + c * P.dilation_w) - i32(P.pad_w);
        if (iw_unpadded < 0 || iw_unpadded >= i32(P.w)) { continue; }
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

function convTranspose2dWgsl(hasBias: boolean): string {
  const biasBinding = hasBias ? "@group(0) @binding(2) var<storage, read> B: array<f32>;" : "";
  const outputBinding = hasBias ? 3 : 2;
  const paramsBinding = hasBias ? 4 : 3;
  const biasInit = hasBias ? "B[co]" : "0.0";
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
  c_out_per_group: u32,
  kh: u32,
  kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  output_pad_h: u32,
  output_pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_h: u32,
  out_w: u32,
  pad0: u32,
};
@group(0) @binding(${paramsBinding}) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = P.n * P.c_out * P.out_h * P.out_w;
  if (idx >= total) { return; }
  let ow = idx % P.out_w;
  let oh = (idx / P.out_w) % P.out_h;
  let co = (idx / (P.out_w * P.out_h)) % P.c_out;
  let nn = idx / (P.out_w * P.out_h * P.c_out);
  let in_per_group = P.c_in / P.groups;
  let out_per_group = P.c_out_per_group;
  let group = co / out_per_group;
  let ci0 = group * in_per_group;
  let co_local = co - group * out_per_group;
  var acc = ${biasInit};
  for (var ci_local = 0u; ci_local < in_per_group; ci_local = ci_local + 1u) {
    let ci = ci0 + ci_local;
    for (var r = 0u; r < P.kh; r = r + 1u) {
      let h_num = i32(oh) + i32(P.pad_h) - i32(r * P.dilation_h);
      if (h_num < 0 || (h_num % i32(P.stride_h)) != 0) { continue; }
      let ih_i = h_num / i32(P.stride_h);
      if (ih_i < 0 || ih_i >= i32(P.h)) { continue; }
      let ih = u32(ih_i);
      for (var c = 0u; c < P.kw; c = c + 1u) {
        let w_num = i32(ow) + i32(P.pad_w) - i32(c * P.dilation_w);
        if (w_num < 0 || (w_num % i32(P.stride_w)) != 0) { continue; }
        let iw_i = w_num / i32(P.stride_w);
        if (iw_i < 0 || iw_i >= i32(P.w)) { continue; }
        let iw = u32(iw_i);
        let x_index = ((nn * P.c_in + ci) * P.h + ih) * P.w + iw;
        let w_index = ((ci * P.c_out_per_group + co_local) * P.kh + r) * P.kw + c;
        acc = acc + X[x_index] * W[w_index];
      }
    }
  }
  Y[idx] = acc;
}
`;
}

const CONV_TRANSPOSE2D_BACKWARD_INPUT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> DX: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  h: u32,
  w: u32,
  c_out: u32,
  c_out_per_group: u32,
  kh: u32,
  kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  output_pad_h: u32,
  output_pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_h: u32,
  out_w: u32,
  pad0: u32,
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
  let in_per_group = P.c_in / P.groups;
  let out_per_group = P.c_out_per_group;
  let group = ci / in_per_group;
  let co0 = group * out_per_group;
  var acc = 0.0;
  for (var co_local = 0u; co_local < out_per_group; co_local = co_local + 1u) {
    let co = co0 + co_local;
    for (var r = 0u; r < P.kh; r = r + 1u) {
      let oh_i = i32(ih * P.stride_h) - i32(P.pad_h) + i32(r * P.dilation_h);
      if (oh_i < 0 || oh_i >= i32(P.out_h)) { continue; }
      let oh = u32(oh_i);
      for (var c = 0u; c < P.kw; c = c + 1u) {
        let ow_i = i32(iw * P.stride_w) - i32(P.pad_w) + i32(c * P.dilation_w);
        if (ow_i < 0 || ow_i >= i32(P.out_w)) { continue; }
        let ow = u32(ow_i);
        let dy_index = ((nn * P.c_out + co) * P.out_h + oh) * P.out_w + ow;
        let w_index = ((ci * P.c_out_per_group + co_local) * P.kh + r) * P.kw + c;
        acc = acc + DY[dy_index] * W[w_index];
      }
    }
  }
  DX[idx] = acc;
}
`;

const CONV_TRANSPOSE2D_BACKWARD_WEIGHT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> DW: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  h: u32,
  w: u32,
  c_out: u32,
  c_out_per_group: u32,
  kh: u32,
  kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  output_pad_h: u32,
  output_pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_h: u32,
  out_w: u32,
  pad0: u32,
};
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = P.c_in * P.c_out_per_group * P.kh * P.kw;
  if (idx >= total) { return; }
  let kc = idx % P.kw;
  let kr = (idx / P.kw) % P.kh;
  let co_local = (idx / (P.kw * P.kh)) % P.c_out_per_group;
  let ci = idx / (P.kw * P.kh * P.c_out_per_group);
  let in_per_group = P.c_in / P.groups;
  let group = ci / in_per_group;
  let co = group * P.c_out_per_group + co_local;
  var acc = 0.0;
  for (var nn = 0u; nn < P.n; nn = nn + 1u) {
    for (var ih = 0u; ih < P.h; ih = ih + 1u) {
      let oh_i = i32(ih * P.stride_h) - i32(P.pad_h) + i32(kr * P.dilation_h);
      if (oh_i < 0 || oh_i >= i32(P.out_h)) { continue; }
      let oh = u32(oh_i);
      for (var iw = 0u; iw < P.w; iw = iw + 1u) {
        let ow_i = i32(iw * P.stride_w) - i32(P.pad_w) + i32(kc * P.dilation_w);
        if (ow_i < 0 || ow_i >= i32(P.out_w)) { continue; }
        let ow = u32(ow_i);
        let dy_index = ((nn * P.c_out + co) * P.out_h + oh) * P.out_w + ow;
        let x_index = ((nn * P.c_in + ci) * P.h + ih) * P.w + iw;
        acc = acc + DY[dy_index] * X[x_index];
      }
    }
  }
  DW[idx] = acc;
}
`;

const CONV_TRANSPOSE2D_BACKWARD_BIAS_WGSL = `
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

function conv3dWgsl(hasBias: boolean): string {
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
  d: u32,
  h: u32,
  w: u32,
  c_out: u32,
  kd: u32,
  kh: u32,
  kw: u32,
  stride_d: u32,
  stride_h: u32,
  stride_w: u32,
  pad_d: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_d: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_d: u32,
  out_h: u32,
  out_w: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(${paramsBinding}) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = P.n * P.c_out * P.out_d * P.out_h * P.out_w;
  if (idx >= total) { return; }
  let ow = idx % P.out_w;
  let oh = (idx / P.out_w) % P.out_h;
  let od = (idx / (P.out_w * P.out_h)) % P.out_d;
  let co = (idx / (P.out_w * P.out_h * P.out_d)) % P.c_out;
  let nn = idx / (P.out_w * P.out_h * P.out_d * P.c_out);
  let c_per_group = P.c_in / P.groups;
  let out_per_group = P.c_out / P.groups;
  let group = co / out_per_group;
  let c0 = group * c_per_group;
  var acc = 0.0;
  for (var ci_local = 0u; ci_local < c_per_group; ci_local = ci_local + 1u) {
    let ci = c0 + ci_local;
    for (var rd = 0u; rd < P.kd; rd = rd + 1u) {
      let id_unpadded = i32(od * P.stride_d + rd * P.dilation_d) - i32(P.pad_d);
      if (id_unpadded < 0 || id_unpadded >= i32(P.d)) { continue; }
      for (var rh = 0u; rh < P.kh; rh = rh + 1u) {
        let ih_unpadded = i32(oh * P.stride_h + rh * P.dilation_h) - i32(P.pad_h);
        if (ih_unpadded < 0 || ih_unpadded >= i32(P.h)) { continue; }
        for (var rw = 0u; rw < P.kw; rw = rw + 1u) {
          let iw_unpadded = i32(ow * P.stride_w + rw * P.dilation_w) - i32(P.pad_w);
          if (iw_unpadded < 0 || iw_unpadded >= i32(P.w)) { continue; }
          let x_index = (((nn * P.c_in + ci) * P.d + u32(id_unpadded)) * P.h + u32(ih_unpadded)) * P.w + u32(iw_unpadded);
          let w_index = (((co * c_per_group + ci_local) * P.kd + rd) * P.kh + rh) * P.kw + rw;
          acc = acc + X[x_index] * W[w_index];
        }
      }
    }
  }
  Y[idx] = acc${biasLine};
}
`;
}

const CONV3D_BACKWARD_INPUT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> DX: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  d: u32,
  h: u32,
  w: u32,
  c_out: u32,
  kd: u32,
  kh: u32,
  kw: u32,
  stride_d: u32,
  stride_h: u32,
  stride_w: u32,
  pad_d: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_d: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_d: u32,
  out_h: u32,
  out_w: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = P.n * P.c_in * P.d * P.h * P.w;
  if (idx >= total) { return; }
  let iw = idx % P.w;
  let ih = (idx / P.w) % P.h;
  let id = (idx / (P.w * P.h)) % P.d;
  let ci = (idx / (P.w * P.h * P.d)) % P.c_in;
  let nn = idx / (P.w * P.h * P.d * P.c_in);
  let c_per_group = P.c_in / P.groups;
  let out_per_group = P.c_out / P.groups;
  let group = ci / c_per_group;
  let ci_local = ci - group * c_per_group;
  let o0 = group * out_per_group;
  var acc = 0.0;
  for (var co_local = 0u; co_local < out_per_group; co_local = co_local + 1u) {
    let co = o0 + co_local;
    for (var rd = 0u; rd < P.kd; rd = rd + 1u) {
      let d_num = i32(id) + i32(P.pad_d) - i32(rd * P.dilation_d);
      if (d_num < 0 || (d_num % i32(P.stride_d)) != 0) { continue; }
      let od_i = d_num / i32(P.stride_d);
      if (od_i < 0 || od_i >= i32(P.out_d)) { continue; }
      let od = u32(od_i);
      for (var rh = 0u; rh < P.kh; rh = rh + 1u) {
        let h_num = i32(ih) + i32(P.pad_h) - i32(rh * P.dilation_h);
        if (h_num < 0 || (h_num % i32(P.stride_h)) != 0) { continue; }
        let oh_i = h_num / i32(P.stride_h);
        if (oh_i < 0 || oh_i >= i32(P.out_h)) { continue; }
        let oh = u32(oh_i);
        for (var rw = 0u; rw < P.kw; rw = rw + 1u) {
          let w_num = i32(iw) + i32(P.pad_w) - i32(rw * P.dilation_w);
          if (w_num < 0 || (w_num % i32(P.stride_w)) != 0) { continue; }
          let ow_i = w_num / i32(P.stride_w);
          if (ow_i < 0 || ow_i >= i32(P.out_w)) { continue; }
          let ow = u32(ow_i);
          let dy_index = (((nn * P.c_out + co) * P.out_d + od) * P.out_h + oh) * P.out_w + ow;
          let w_index = (((co * c_per_group + ci_local) * P.kd + rd) * P.kh + rh) * P.kw + rw;
          acc = acc + DY[dy_index] * W[w_index];
        }
      }
    }
  }
  DX[idx] = acc;
}
`;

const CONV3D_BACKWARD_WEIGHT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> DW: array<f32>;

struct Params {
  n: u32,
  c_in: u32,
  d: u32,
  h: u32,
  w: u32,
  c_out: u32,
  kd: u32,
  kh: u32,
  kw: u32,
  stride_d: u32,
  stride_h: u32,
  stride_w: u32,
  pad_d: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_d: u32,
  dilation_h: u32,
  dilation_w: u32,
  groups: u32,
  out_d: u32,
  out_h: u32,
  out_w: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let c_per_group = P.c_in / P.groups;
  let total = P.c_out * c_per_group * P.kd * P.kh * P.kw;
  if (idx >= total) { return; }
  let rw = idx % P.kw;
  let rh = (idx / P.kw) % P.kh;
  let rd = (idx / (P.kw * P.kh)) % P.kd;
  let ci_local = (idx / (P.kw * P.kh * P.kd)) % c_per_group;
  let co = idx / (P.kw * P.kh * P.kd * c_per_group);
  let out_per_group = P.c_out / P.groups;
  let group = co / out_per_group;
  let ci = group * c_per_group + ci_local;
  var acc = 0.0;
  for (var nn = 0u; nn < P.n; nn = nn + 1u) {
    for (var od = 0u; od < P.out_d; od = od + 1u) {
      let id_i = i32(od * P.stride_d + rd * P.dilation_d) - i32(P.pad_d);
      if (id_i < 0 || id_i >= i32(P.d)) { continue; }
      let id = u32(id_i);
      for (var oh = 0u; oh < P.out_h; oh = oh + 1u) {
        let ih_i = i32(oh * P.stride_h + rh * P.dilation_h) - i32(P.pad_h);
        if (ih_i < 0 || ih_i >= i32(P.h)) { continue; }
        let ih = u32(ih_i);
        for (var ow = 0u; ow < P.out_w; ow = ow + 1u) {
          let iw_i = i32(ow * P.stride_w + rw * P.dilation_w) - i32(P.pad_w);
          if (iw_i < 0 || iw_i >= i32(P.w)) { continue; }
          let iw = u32(iw_i);
          let dy_index = (((nn * P.c_out + co) * P.out_d + od) * P.out_h + oh) * P.out_w + ow;
          let x_index = (((nn * P.c_in + ci) * P.d + id) * P.h + ih) * P.w + iw;
          acc = acc + DY[dy_index] * X[x_index];
        }
      }
    }
  }
  DW[idx] = acc;
}
`;

const CONV3D_BACKWARD_BIAS_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read_write> DB: array<f32>;

struct Params {
  n: u32,
  c_out: u32,
  out_d: u32,
  out_h: u32,
  out_w: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let co = gid.x;
  if (co >= P.c_out) { return; }
  var acc = 0.0;
  for (var nn = 0u; nn < P.n; nn = nn + 1u) {
    for (var od = 0u; od < P.out_d; od = od + 1u) {
      for (var oh = 0u; oh < P.out_h; oh = oh + 1u) {
        for (var ow = 0u; ow < P.out_w; ow = ow + 1u) {
          let dy_index = (((nn * P.c_out + co) * P.out_d + od) * P.out_h + oh) * P.out_w + ow;
          acc = acc + DY[dy_index];
        }
      }
    }
  }
  DB[co] = acc;
}
`;

const SGD_UPDATE_WGSL = `
@group(0) @binding(0) var<storage, read> P0: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> P1: array<f32>;

struct Params {
  lr: f32,
  weight_decay: f32,
  total: u32,
  pad: u32,
};
@group(0) @binding(3) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.total) { return; }
  let grad = G[idx] + U.weight_decay * P0[idx];
  P1[idx] = P0[idx] - U.lr * grad;
}
`;

const ADAMW_UPDATE_M_WGSL = `
@group(0) @binding(0) var<storage, read> M0: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> M1: array<f32>;

struct Params {
  beta1: f32,
  total: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(3) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.total) { return; }
  M1[idx] = U.beta1 * M0[idx] + (1.0 - U.beta1) * G[idx];
}
`;

const ADAMW_UPDATE_V_WGSL = `
@group(0) @binding(0) var<storage, read> V0: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> V1: array<f32>;

struct Params {
  beta2: f32,
  total: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(3) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.total) { return; }
  V1[idx] = U.beta2 * V0[idx] + (1.0 - U.beta2) * G[idx] * G[idx];
}
`;

const ADAMW_UPDATE_PARAM_WGSL = `
@group(0) @binding(0) var<storage, read> P0: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> M1: array<f32>;
@group(0) @binding(3) var<storage, read> V1: array<f32>;
@group(0) @binding(4) var<storage, read_write> P1: array<f32>;

struct Params {
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  weight_decay: f32,
  step: u32,
  total: u32,
  pad: u32,
};
@group(0) @binding(5) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.total) { return; }
  let step_f = f32(U.step);
  let m_hat = M1[idx] / (1.0 - pow(U.beta1, step_f));
  let v_hat = V1[idx] / (1.0 - pow(U.beta2, step_f));
  let update = m_hat / (sqrt(v_hat) + U.eps);
  P1[idx] = P0[idx] - U.lr * update - U.lr * U.weight_decay * P0[idx];
}
`;

const ADAM_UPDATE_M_WGSL = `
@group(0) @binding(0) var<storage, read> P0: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> M0: array<f32>;
@group(0) @binding(3) var<storage, read_write> M1: array<f32>;

struct Params {
  beta1: f32,
  weight_decay: f32,
  total: u32,
  pad: u32,
};
@group(0) @binding(4) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.total) { return; }
  let grad = G[idx] + U.weight_decay * P0[idx];
  M1[idx] = U.beta1 * M0[idx] + (1.0 - U.beta1) * grad;
}
`;

const ADAM_UPDATE_V_WGSL = `
@group(0) @binding(0) var<storage, read> P0: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> V0: array<f32>;
@group(0) @binding(3) var<storage, read_write> V1: array<f32>;

struct Params {
  beta2: f32,
  weight_decay: f32,
  total: u32,
  pad: u32,
};
@group(0) @binding(4) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.total) { return; }
  let grad = G[idx] + U.weight_decay * P0[idx];
  V1[idx] = U.beta2 * V0[idx] + (1.0 - U.beta2) * grad * grad;
}
`;

const ADAM_UPDATE_PARAM_WGSL = `
@group(0) @binding(0) var<storage, read> P0: array<f32>;
@group(0) @binding(1) var<storage, read> M1: array<f32>;
@group(0) @binding(2) var<storage, read> V1: array<f32>;
@group(0) @binding(3) var<storage, read_write> P1: array<f32>;

struct Params {
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  step: u32,
  total: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(4) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= U.total) { return; }
  let step_f = f32(U.step);
  let m_hat = M1[idx] / (1.0 - pow(U.beta1, step_f));
  let v_hat = V1[idx] / (1.0 - pow(U.beta2, step_f));
  P1[idx] = P0[idx] - U.lr * (m_hat / (sqrt(v_hat) + U.eps));
}
`;

const LAYER_NORM_WGSL = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  eps_bits: u32,
  pad: u32,
};
@group(0) @binding(4) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= U.rows) { return; }
  let base = r * U.cols;
  let eps = bitcast<f32>(U.eps_bits);
  let n = f32(U.cols);
  var sum = 0.0;
  for (var i = 0u; i < U.cols; i = i + 1u) {
    sum = sum + X[base + i];
  }
  let mean = sum / n;
  var var_sum = 0.0;
  for (var i = 0u; i < U.cols; i = i + 1u) {
    let d = X[base + i] - mean;
    var_sum = var_sum + d * d;
  }
  let inv_std = 1.0 / sqrt(var_sum / n + eps);
  for (var i = 0u; i < U.cols; i = i + 1u) {
    let x_hat = (X[base + i] - mean) * inv_std;
    Y[base + i] = x_hat * W[i] + B[i];
  }
}
`;

const LAYER_NORM_BACKWARD_INPUT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> DX: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  eps_bits: u32,
  pad: u32,
};
@group(0) @binding(4) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= U.rows) { return; }
  let base = r * U.cols;
  let eps = bitcast<f32>(U.eps_bits);
  let n = f32(U.cols);
  var sum = 0.0;
  for (var i = 0u; i < U.cols; i = i + 1u) {
    sum = sum + X[base + i];
  }
  let mean = sum / n;
  var var_sum = 0.0;
  for (var i = 0u; i < U.cols; i = i + 1u) {
    let d = X[base + i] - mean;
    var_sum = var_sum + d * d;
  }
  let inv_std = 1.0 / sqrt(var_sum / n + eps);
  var sum_g = 0.0;
  var sum_g_xhat = 0.0;
  for (var i = 0u; i < U.cols; i = i + 1u) {
    let x_hat = (X[base + i] - mean) * inv_std;
    let g = DY[base + i] * W[i];
    sum_g = sum_g + g;
    sum_g_xhat = sum_g_xhat + g * x_hat;
  }
  for (var i = 0u; i < U.cols; i = i + 1u) {
    let x_hat = (X[base + i] - mean) * inv_std;
    let g = DY[base + i] * W[i];
    DX[base + i] = (inv_std / n) * (n * g - sum_g - x_hat * sum_g_xhat);
  }
}
`;

const LAYER_NORM_BACKWARD_WEIGHT_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> DW: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  eps_bits: u32,
  pad: u32,
};
@group(0) @binding(3) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= U.cols) { return; }
  let eps = bitcast<f32>(U.eps_bits);
  let n = f32(U.cols);
  var acc = 0.0;
  for (var r = 0u; r < U.rows; r = r + 1u) {
    let base = r * U.cols;
    var sum = 0.0;
    for (var i = 0u; i < U.cols; i = i + 1u) {
      sum = sum + X[base + i];
    }
    let mean = sum / n;
    var var_sum = 0.0;
    for (var i = 0u; i < U.cols; i = i + 1u) {
      let d = X[base + i] - mean;
      var_sum = var_sum + d * d;
    }
    let inv_std = 1.0 / sqrt(var_sum / n + eps);
    let x_hat = (X[base + c] - mean) * inv_std;
    acc = acc + DY[base + c] * x_hat;
  }
  DW[c] = acc;
}
`;

const LAYER_NORM_BACKWARD_BIAS_WGSL = `
@group(0) @binding(0) var<storage, read> DY: array<f32>;
@group(0) @binding(1) var<storage, read_write> DB: array<f32>;

struct Params {
  rows: u32,
  cols: u32,
  eps_bits: u32,
  pad: u32,
};
@group(0) @binding(2) var<uniform> U: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= U.cols) { return; }
  var acc = 0.0;
  for (var r = 0u; r < U.rows; r = r + 1u) {
    acc = acc + DY[r * U.cols + c];
  }
  DB[c] = acc;
}
`;

function conv1dDirect(
  device: KernelDevice,
  x: ResidentValue,
  weight: ResidentValue,
  bias: ResidentValue | null,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectRecord(rawArg, "tensor plan CONV1D.arg");
  const n = expectNumber(arg.n, "CONV1D.n");
  const cIn = expectNumber(arg.c_in, "CONV1D.c_in");
  const lIn = expectNumber(arg.l_in, "CONV1D.l_in");
  const cOut = expectNumber(arg.c_out, "CONV1D.c_out");
  const k = expectNumber(arg.k, "CONV1D.k");
  const stride = expectNumber(arg.stride, "CONV1D.stride");
  const padding = expectNumber(arg.padding, "CONV1D.padding");
  const dilation = expectNumber(arg.dilation, "CONV1D.dilation");
  const groups = expectNumber(arg.groups, "CONV1D.groups");
  const lOut = expectNumber(arg.l_out, "CONV1D.l_out");
  validateConv1dShape("CONV1D", n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut);
  expectShape(x.shape, [n, cIn, lIn], "CONV1D.input");
  expectShape(weight.shape, [cOut, cIn / groups, k], "CONV1D.weight");
  expectShape(shape, [n, cOut, lOut], "CONV1D.output");
  const inputBuffers = [x.buffer, weight.buffer];
  if (bias !== null) {
    expectShape(bias.shape, [cOut], "CONV1D.bias");
    inputBuffers.push(bias.buffer);
  }
  const outputLength = n * cOut * lOut;
  return runDirect(device, {
    name: bias === null ? "tensor_plan_conv1d_nobias" : "tensor_plan_conv1d_bias",
    wgsl: conv1dWgsl(bias !== null),
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers,
    outputLength,
    params: new Uint32Array([n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut, 0, 0]),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: [
      bias === null ? "nobias" : "bias",
      n, cIn, lIn, cOut, k, stride, padding, dilation, groups, lOut,
    ].join("_"),
  });
}

function conv1dBackwardInputDirect(
  device: KernelDevice,
  dy: ResidentValue,
  weight: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv1dArg(rawArg, "CONV1D_BACKWARD_INPUT");
  validateConv1dShape(
    "CONV1D_BACKWARD_INPUT",
    arg.n, arg.cIn, arg.lIn, arg.cOut, arg.k,
    arg.stride, arg.padding, arg.dilation, arg.groups, arg.lOut,
  );
  expectShape(dy.shape, [arg.n, arg.cOut, arg.lOut], "CONV1D_BACKWARD_INPUT.dy");
  expectShape(weight.shape, [arg.cOut, arg.cIn / arg.groups, arg.k], "CONV1D_BACKWARD_INPUT.weight");
  expectShape(shape, [arg.n, arg.cIn, arg.lIn], "CONV1D_BACKWARD_INPUT.output");
  const outputLength = arg.n * arg.cIn * arg.lIn;
  return runDirect(device, {
    name: "tensor_plan_conv1d_backward_input",
    wgsl: CONV1D_BACKWARD_INPUT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, weight.buffer],
    outputLength,
    params: conv1dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: conv1dCacheKey(arg),
  });
}

function conv1dBackwardWeightDirect(
  device: KernelDevice,
  dy: ResidentValue,
  x: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv1dArg(rawArg, "CONV1D_BACKWARD_WEIGHT");
  validateConv1dShape(
    "CONV1D_BACKWARD_WEIGHT",
    arg.n, arg.cIn, arg.lIn, arg.cOut, arg.k,
    arg.stride, arg.padding, arg.dilation, arg.groups, arg.lOut,
  );
  expectShape(dy.shape, [arg.n, arg.cOut, arg.lOut], "CONV1D_BACKWARD_WEIGHT.dy");
  expectShape(x.shape, [arg.n, arg.cIn, arg.lIn], "CONV1D_BACKWARD_WEIGHT.input");
  expectShape(shape, [arg.cOut, arg.cIn / arg.groups, arg.k], "CONV1D_BACKWARD_WEIGHT.output");
  const outputLength = arg.cOut * (arg.cIn / arg.groups) * arg.k;
  return runDirect(device, {
    name: "tensor_plan_conv1d_backward_weight",
    wgsl: CONV1D_BACKWARD_WEIGHT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, x.buffer],
    outputLength,
    params: conv1dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: conv1dCacheKey(arg),
  });
}

function conv1dBackwardBiasDirect(
  device: KernelDevice,
  dy: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv1dArg(rawArg, "CONV1D_BACKWARD_BIAS");
  validateConv1dShape(
    "CONV1D_BACKWARD_BIAS",
    arg.n, arg.cIn, arg.lIn, arg.cOut, arg.k,
    arg.stride, arg.padding, arg.dilation, arg.groups, arg.lOut,
  );
  expectShape(dy.shape, [arg.n, arg.cOut, arg.lOut], "CONV1D_BACKWARD_BIAS.dy");
  expectShape(shape, [arg.cOut], "CONV1D_BACKWARD_BIAS.output");
  return runDirect(device, {
    name: "tensor_plan_conv1d_backward_bias",
    wgsl: CONV1D_BACKWARD_BIAS_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer],
    outputLength: arg.cOut,
    params: new Uint32Array([arg.n, arg.cOut, arg.lOut, 0]),
    dispatchCount: [arg.cOut, 1, 1],
    cacheKeySuffix: `${arg.n}_${arg.cOut}_${arg.lOut}`,
  });
}

function conv2dDirect(
  device: KernelDevice,
  x: ResidentValue,
  weight: ResidentValue,
  bias: ResidentValue | null,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectRecord(rawArg, "tensor plan CONV2D.arg");
  const n = expectNumber(arg.n, "CONV2D.n");
  const cIn = expectNumber(arg.c_in, "CONV2D.c_in");
  const h = expectNumber(arg.h, "CONV2D.h");
  const w = expectNumber(arg.w, "CONV2D.w");
  const cOut = expectNumber(arg.c_out, "CONV2D.c_out");
  const kh = expectNumber(arg.kh, "CONV2D.kh");
  const kw = expectNumber(arg.kw, "CONV2D.kw");
  const strideH = expectNumber(arg.stride_h, "CONV2D.stride_h");
  const strideW = expectNumber(arg.stride_w, "CONV2D.stride_w");
  const padH = expectNumber(arg.pad_h, "CONV2D.pad_h");
  const padW = expectNumber(arg.pad_w, "CONV2D.pad_w");
  const dilationH = expectNumber(arg.dilation_h, "CONV2D.dilation_h");
  const dilationW = expectNumber(arg.dilation_w, "CONV2D.dilation_w");
  const groups = expectNumber(arg.groups, "CONV2D.groups");
  const outH = expectNumber(arg.out_h, "CONV2D.out_h");
  const outW = expectNumber(arg.out_w, "CONV2D.out_w");
  validateConv2dShape(
    "CONV2D",
    n, cIn, h, w, cOut, kh, kw,
    strideH, strideW, padH, padW, dilationH, dilationW,
    groups, outH, outW,
  );
  expectShape(x.shape, [n, cIn, h, w], "CONV2D.input");
  expectShape(weight.shape, [cOut, cIn / groups, kh, kw], "CONV2D.weight");
  expectShape(shape, [n, cOut, outH, outW], "CONV2D.output");
  const inputBuffers = [x.buffer, weight.buffer];
  if (bias !== null) {
    expectShape(bias.shape, [cOut], "CONV2D.bias");
    inputBuffers.push(bias.buffer);
  }
  const outputLength = n * cOut * outH * outW;
  return runDirect(device, {
    name: bias === null ? "tensor_plan_conv2d_nobias" : "tensor_plan_conv2d_bias",
    wgsl: conv2dWgsl(bias !== null),
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers,
    outputLength,
    params: new Uint32Array([
      n, cIn, h, w, cOut, kh, kw,
      strideH, strideW, padH, padW, dilationH, dilationW,
      groups, outH, outW,
    ]),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: [
      bias === null ? "nobias" : "bias",
      n, cIn, h, w, cOut, kh, kw,
      strideH, strideW, padH, padW, dilationH, dilationW,
      groups, outH, outW,
    ].join("_"),
  });
}

function conv2dBackwardInputDirect(
  device: KernelDevice,
  dy: ResidentValue,
  weight: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv2dArg(rawArg, "CONV2D_BACKWARD_INPUT");
  validateConv2dShapeFromArg("CONV2D_BACKWARD_INPUT", arg);
  expectShape(dy.shape, [arg.n, arg.cOut, arg.outH, arg.outW], "CONV2D_BACKWARD_INPUT.dy");
  expectShape(weight.shape, [arg.cOut, arg.cIn / arg.groups, arg.kh, arg.kw], "CONV2D_BACKWARD_INPUT.weight");
  expectShape(shape, [arg.n, arg.cIn, arg.h, arg.w], "CONV2D_BACKWARD_INPUT.output");
  const outputLength = arg.n * arg.cIn * arg.h * arg.w;
  return runDirect(device, {
    name: "tensor_plan_conv2d_backward_input",
    wgsl: CONV2D_BACKWARD_INPUT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, weight.buffer],
    outputLength,
    params: conv2dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: conv2dCacheKey(arg),
  });
}

function conv2dBackwardWeightDirect(
  device: KernelDevice,
  dy: ResidentValue,
  x: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv2dArg(rawArg, "CONV2D_BACKWARD_WEIGHT");
  validateConv2dShapeFromArg("CONV2D_BACKWARD_WEIGHT", arg);
  expectShape(dy.shape, [arg.n, arg.cOut, arg.outH, arg.outW], "CONV2D_BACKWARD_WEIGHT.dy");
  expectShape(x.shape, [arg.n, arg.cIn, arg.h, arg.w], "CONV2D_BACKWARD_WEIGHT.input");
  expectShape(shape, [arg.cOut, arg.cIn / arg.groups, arg.kh, arg.kw], "CONV2D_BACKWARD_WEIGHT.output");
  const outputLength = arg.cOut * (arg.cIn / arg.groups) * arg.kh * arg.kw;
  return runDirect(device, {
    name: "tensor_plan_conv2d_backward_weight",
    wgsl: CONV2D_BACKWARD_WEIGHT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, x.buffer],
    outputLength,
    params: conv2dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: conv2dCacheKey(arg),
  });
}

function conv2dBackwardBiasDirect(
  device: KernelDevice,
  dy: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv2dArg(rawArg, "CONV2D_BACKWARD_BIAS");
  validateConv2dShapeFromArg("CONV2D_BACKWARD_BIAS", arg);
  expectShape(dy.shape, [arg.n, arg.cOut, arg.outH, arg.outW], "CONV2D_BACKWARD_BIAS.dy");
  expectShape(shape, [arg.cOut], "CONV2D_BACKWARD_BIAS.output");
  return runDirect(device, {
    name: "tensor_plan_conv2d_backward_bias",
    wgsl: CONV2D_BACKWARD_BIAS_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer],
    outputLength: arg.cOut,
    params: new Uint32Array([arg.n, arg.cOut, arg.outH, arg.outW]),
    dispatchCount: [arg.cOut, 1, 1],
    cacheKeySuffix: `${arg.n}_${arg.cOut}_${arg.outH}_${arg.outW}`,
  });
}

function convTranspose2dDirect(
  device: KernelDevice,
  x: ResidentValue,
  weight: ResidentValue,
  bias: ResidentValue | null,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConvTranspose2dArg(rawArg, "CONV_TRANSPOSE2D");
  validateConvTranspose2dShapeFromArg("CONV_TRANSPOSE2D", arg);
  expectShape(x.shape, [arg.n, arg.cIn, arg.h, arg.w], "CONV_TRANSPOSE2D.input");
  expectShape(
    weight.shape,
    [arg.cIn, arg.cOutPerGroup, arg.kh, arg.kw],
    "CONV_TRANSPOSE2D.weight",
  );
  expectShape(shape, [arg.n, arg.cOut, arg.outH, arg.outW], "CONV_TRANSPOSE2D.output");
  const inputBuffers = [x.buffer, weight.buffer];
  if (bias !== null) {
    expectShape(bias.shape, [arg.cOut], "CONV_TRANSPOSE2D.bias");
    inputBuffers.push(bias.buffer);
  }
  const outputLength = arg.n * arg.cOut * arg.outH * arg.outW;
  return runDirect(device, {
    name: bias === null ? "tensor_plan_conv_transpose2d_nobias" : "tensor_plan_conv_transpose2d_bias",
    wgsl: convTranspose2dWgsl(bias !== null),
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers,
    outputLength,
    params: convTranspose2dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: `${bias === null ? "nobias" : "bias"}_${convTranspose2dCacheKey(arg)}`,
  });
}

function convTranspose2dBackwardInputDirect(
  device: KernelDevice,
  dy: ResidentValue,
  weight: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConvTranspose2dArg(rawArg, "CONV_TRANSPOSE2D_BACKWARD_INPUT");
  validateConvTranspose2dShapeFromArg("CONV_TRANSPOSE2D_BACKWARD_INPUT", arg);
  expectShape(
    dy.shape,
    [arg.n, arg.cOut, arg.outH, arg.outW],
    "CONV_TRANSPOSE2D_BACKWARD_INPUT.dy",
  );
  expectShape(
    weight.shape,
    [arg.cIn, arg.cOutPerGroup, arg.kh, arg.kw],
    "CONV_TRANSPOSE2D_BACKWARD_INPUT.weight",
  );
  expectShape(shape, [arg.n, arg.cIn, arg.h, arg.w], "CONV_TRANSPOSE2D_BACKWARD_INPUT.output");
  const outputLength = arg.n * arg.cIn * arg.h * arg.w;
  return runDirect(device, {
    name: "tensor_plan_conv_transpose2d_backward_input",
    wgsl: CONV_TRANSPOSE2D_BACKWARD_INPUT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, weight.buffer],
    outputLength,
    params: convTranspose2dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: convTranspose2dCacheKey(arg),
  });
}

function convTranspose2dBackwardWeightDirect(
  device: KernelDevice,
  dy: ResidentValue,
  x: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConvTranspose2dArg(rawArg, "CONV_TRANSPOSE2D_BACKWARD_WEIGHT");
  validateConvTranspose2dShapeFromArg("CONV_TRANSPOSE2D_BACKWARD_WEIGHT", arg);
  expectShape(
    dy.shape,
    [arg.n, arg.cOut, arg.outH, arg.outW],
    "CONV_TRANSPOSE2D_BACKWARD_WEIGHT.dy",
  );
  expectShape(x.shape, [arg.n, arg.cIn, arg.h, arg.w], "CONV_TRANSPOSE2D_BACKWARD_WEIGHT.input");
  expectShape(
    shape,
    [arg.cIn, arg.cOutPerGroup, arg.kh, arg.kw],
    "CONV_TRANSPOSE2D_BACKWARD_WEIGHT.output",
  );
  const outputLength = arg.cIn * arg.cOutPerGroup * arg.kh * arg.kw;
  return runDirect(device, {
    name: "tensor_plan_conv_transpose2d_backward_weight",
    wgsl: CONV_TRANSPOSE2D_BACKWARD_WEIGHT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, x.buffer],
    outputLength,
    params: convTranspose2dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: convTranspose2dCacheKey(arg),
  });
}

function convTranspose2dBackwardBiasDirect(
  device: KernelDevice,
  dy: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConvTranspose2dArg(rawArg, "CONV_TRANSPOSE2D_BACKWARD_BIAS");
  validateConvTranspose2dShapeFromArg("CONV_TRANSPOSE2D_BACKWARD_BIAS", arg);
  expectShape(
    dy.shape,
    [arg.n, arg.cOut, arg.outH, arg.outW],
    "CONV_TRANSPOSE2D_BACKWARD_BIAS.dy",
  );
  expectShape(shape, [arg.cOut], "CONV_TRANSPOSE2D_BACKWARD_BIAS.output");
  return runDirect(device, {
    name: "tensor_plan_conv_transpose2d_backward_bias",
    wgsl: CONV_TRANSPOSE2D_BACKWARD_BIAS_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer],
    outputLength: arg.cOut,
    params: new Uint32Array([arg.n, arg.cOut, arg.outH, arg.outW]),
    dispatchCount: [arg.cOut, 1, 1],
    cacheKeySuffix: `${arg.n}_${arg.cOut}_${arg.outH}_${arg.outW}`,
  });
}

function conv3dDirect(
  device: KernelDevice,
  x: ResidentValue,
  weight: ResidentValue,
  bias: ResidentValue | null,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv3dArg(rawArg, "CONV3D");
  validateConv3dShapeFromArg("CONV3D", arg);
  expectShape(x.shape, [arg.n, arg.cIn, arg.d, arg.h, arg.w], "CONV3D.input");
  expectShape(weight.shape, [arg.cOut, arg.cIn / arg.groups, arg.kd, arg.kh, arg.kw], "CONV3D.weight");
  expectShape(shape, [arg.n, arg.cOut, arg.outD, arg.outH, arg.outW], "CONV3D.output");
  const inputBuffers = [x.buffer, weight.buffer];
  if (bias !== null) {
    expectShape(bias.shape, [arg.cOut], "CONV3D.bias");
    inputBuffers.push(bias.buffer);
  }
  const outputLength = arg.n * arg.cOut * arg.outD * arg.outH * arg.outW;
  return runDirect(device, {
    name: bias === null ? "tensor_plan_conv3d_nobias" : "tensor_plan_conv3d_bias",
    wgsl: conv3dWgsl(bias !== null),
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers,
    outputLength,
    params: conv3dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: `${bias === null ? "nobias" : "bias"}_${conv3dCacheKey(arg)}`,
  });
}

function conv3dBackwardInputDirect(
  device: KernelDevice,
  dy: ResidentValue,
  weight: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv3dArg(rawArg, "CONV3D_BACKWARD_INPUT");
  validateConv3dShapeFromArg("CONV3D_BACKWARD_INPUT", arg);
  expectShape(dy.shape, [arg.n, arg.cOut, arg.outD, arg.outH, arg.outW], "CONV3D_BACKWARD_INPUT.dy");
  expectShape(weight.shape, [arg.cOut, arg.cIn / arg.groups, arg.kd, arg.kh, arg.kw], "CONV3D_BACKWARD_INPUT.weight");
  expectShape(shape, [arg.n, arg.cIn, arg.d, arg.h, arg.w], "CONV3D_BACKWARD_INPUT.output");
  const outputLength = arg.n * arg.cIn * arg.d * arg.h * arg.w;
  return runDirect(device, {
    name: "tensor_plan_conv3d_backward_input",
    wgsl: CONV3D_BACKWARD_INPUT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, weight.buffer],
    outputLength,
    params: conv3dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: conv3dCacheKey(arg),
  });
}

function conv3dBackwardWeightDirect(
  device: KernelDevice,
  dy: ResidentValue,
  x: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv3dArg(rawArg, "CONV3D_BACKWARD_WEIGHT");
  validateConv3dShapeFromArg("CONV3D_BACKWARD_WEIGHT", arg);
  expectShape(dy.shape, [arg.n, arg.cOut, arg.outD, arg.outH, arg.outW], "CONV3D_BACKWARD_WEIGHT.dy");
  expectShape(x.shape, [arg.n, arg.cIn, arg.d, arg.h, arg.w], "CONV3D_BACKWARD_WEIGHT.input");
  expectShape(shape, [arg.cOut, arg.cIn / arg.groups, arg.kd, arg.kh, arg.kw], "CONV3D_BACKWARD_WEIGHT.output");
  const outputLength = arg.cOut * (arg.cIn / arg.groups) * arg.kd * arg.kh * arg.kw;
  return runDirect(device, {
    name: "tensor_plan_conv3d_backward_weight",
    wgsl: CONV3D_BACKWARD_WEIGHT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, x.buffer],
    outputLength,
    params: conv3dParams(arg),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: conv3dCacheKey(arg),
  });
}

function conv3dBackwardBiasDirect(
  device: KernelDevice,
  dy: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectConv3dArg(rawArg, "CONV3D_BACKWARD_BIAS");
  validateConv3dShapeFromArg("CONV3D_BACKWARD_BIAS", arg);
  expectShape(dy.shape, [arg.n, arg.cOut, arg.outD, arg.outH, arg.outW], "CONV3D_BACKWARD_BIAS.dy");
  expectShape(shape, [arg.cOut], "CONV3D_BACKWARD_BIAS.output");
  return runDirect(device, {
    name: "tensor_plan_conv3d_backward_bias",
    wgsl: CONV3D_BACKWARD_BIAS_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer],
    outputLength: arg.cOut,
    params: new Uint32Array([arg.n, arg.cOut, arg.outD, arg.outH, arg.outW, 0, 0, 0]),
    dispatchCount: [arg.cOut, 1, 1],
    cacheKeySuffix: `${arg.n}_${arg.cOut}_${arg.outD}_${arg.outH}_${arg.outW}`,
  });
}

function layerNormDirect(
  device: KernelDevice,
  x: ResidentValue,
  weight: ResidentValue,
  bias: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectLayerNormArg(rawArg, "LAYER_NORM");
  expectShape(x.shape, shape, "LAYER_NORM.input");
  expectShape(weight.shape, arg.normalizedShape, "LAYER_NORM.weight");
  expectShape(bias.shape, arg.normalizedShape, "LAYER_NORM.bias");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_layer_norm",
    wgsl: LAYER_NORM_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [x.buffer, weight.buffer, bias.buffer],
    outputLength,
    params: layerNormParams(arg),
    dispatchCount: [arg.rows, 1, 1],
    cacheKeySuffix: `${shapeStr(shape)}_${arg.rows}_${arg.cols}_${arg.eps}`,
  });
}

function layerNormBackwardInputDirect(
  device: KernelDevice,
  dy: ResidentValue,
  x: ResidentValue,
  weight: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectLayerNormArg(rawArg, "LAYER_NORM_BACKWARD_INPUT");
  expectShape(dy.shape, shape, "LAYER_NORM_BACKWARD_INPUT.dy");
  expectShape(x.shape, shape, "LAYER_NORM_BACKWARD_INPUT.input");
  expectShape(weight.shape, arg.normalizedShape, "LAYER_NORM_BACKWARD_INPUT.weight");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_layer_norm_backward_input",
    wgsl: LAYER_NORM_BACKWARD_INPUT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, x.buffer, weight.buffer],
    outputLength,
    params: layerNormParams(arg),
    dispatchCount: [arg.rows, 1, 1],
    cacheKeySuffix: `${shapeStr(shape)}_${arg.rows}_${arg.cols}_${arg.eps}`,
  });
}

function layerNormBackwardWeightDirect(
  device: KernelDevice,
  dy: ResidentValue,
  x: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectLayerNormArg(rawArg, "LAYER_NORM_BACKWARD_WEIGHT");
  expectShape(dy.shape, x.shape, "LAYER_NORM_BACKWARD_WEIGHT.dy");
  expectShape(shape, arg.normalizedShape, "LAYER_NORM_BACKWARD_WEIGHT.output");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_layer_norm_backward_weight",
    wgsl: LAYER_NORM_BACKWARD_WEIGHT_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer, x.buffer],
    outputLength,
    params: layerNormParams(arg),
    dispatchCount: [arg.cols, 1, 1],
    cacheKeySuffix: `${shapeStr(x.shape)}_${arg.rows}_${arg.cols}_${arg.eps}`,
  });
}

function layerNormBackwardBiasDirect(
  device: KernelDevice,
  dy: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectLayerNormArg(rawArg, "LAYER_NORM_BACKWARD_BIAS");
  expectShape(shape, arg.normalizedShape, "LAYER_NORM_BACKWARD_BIAS.output");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_layer_norm_backward_bias",
    wgsl: LAYER_NORM_BACKWARD_BIAS_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [dy.buffer],
    outputLength,
    params: layerNormParams(arg),
    dispatchCount: [arg.cols, 1, 1],
    cacheKeySuffix: `${shapeStr(dy.shape)}_${arg.rows}_${arg.cols}`,
  });
}

function sgdUpdateDirect(
  device: KernelDevice,
  param: ResidentValue,
  grad: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectSgdUpdateArg(rawArg);
  expectShape(param.shape, shape, "SGD_UPDATE.param");
  expectShape(grad.shape, shape, "SGD_UPDATE.grad");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_sgd_update",
    wgsl: SGD_UPDATE_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [param.buffer, grad.buffer],
    outputLength,
    params: sgdUpdateParams(arg, outputLength),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: `${shapeStr(shape)}_${arg.lr}_${arg.weightDecay}`,
  });
}

function adamwUpdateMDirect(
  device: KernelDevice,
  m: ResidentValue,
  grad: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectAdamwUpdateArg(rawArg, "ADAMW_UPDATE_M");
  expectShape(m.shape, shape, "ADAMW_UPDATE_M.m");
  expectShape(grad.shape, shape, "ADAMW_UPDATE_M.grad");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_adamw_update_m",
    wgsl: ADAMW_UPDATE_M_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [m.buffer, grad.buffer],
    outputLength,
    params: adamwMvParams(arg.beta1, outputLength),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: `${shapeStr(shape)}_${arg.beta1}`,
  });
}

function adamwUpdateVDirect(
  device: KernelDevice,
  v: ResidentValue,
  grad: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectAdamwUpdateArg(rawArg, "ADAMW_UPDATE_V");
  expectShape(v.shape, shape, "ADAMW_UPDATE_V.v");
  expectShape(grad.shape, shape, "ADAMW_UPDATE_V.grad");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_adamw_update_v",
    wgsl: ADAMW_UPDATE_V_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [v.buffer, grad.buffer],
    outputLength,
    params: adamwMvParams(arg.beta2, outputLength),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: `${shapeStr(shape)}_${arg.beta2}`,
  });
}

function adamwUpdateParamDirect(
  device: KernelDevice,
  param: ResidentValue,
  grad: ResidentValue,
  m: ResidentValue,
  v: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectAdamwUpdateArg(rawArg, "ADAMW_UPDATE_PARAM");
  expectShape(param.shape, shape, "ADAMW_UPDATE_PARAM.param");
  expectShape(grad.shape, shape, "ADAMW_UPDATE_PARAM.grad");
  expectShape(m.shape, shape, "ADAMW_UPDATE_PARAM.m");
  expectShape(v.shape, shape, "ADAMW_UPDATE_PARAM.v");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_adamw_update_param",
    wgsl: ADAMW_UPDATE_PARAM_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [param.buffer, grad.buffer, m.buffer, v.buffer],
    outputLength,
    params: adamwParamParams(arg, outputLength),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: [
      shapeStr(shape),
      arg.lr,
      arg.beta1,
      arg.beta2,
      arg.eps,
      arg.weightDecay,
      arg.step,
    ].join("_"),
  });
}

function adamUpdateMDirect(
  device: KernelDevice,
  param: ResidentValue,
  grad: ResidentValue,
  m: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectAdamUpdateArg(rawArg, "ADAM_UPDATE_M");
  expectShape(param.shape, shape, "ADAM_UPDATE_M.param");
  expectShape(grad.shape, shape, "ADAM_UPDATE_M.grad");
  expectShape(m.shape, shape, "ADAM_UPDATE_M.m");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_adam_update_m",
    wgsl: ADAM_UPDATE_M_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [param.buffer, grad.buffer, m.buffer],
    outputLength,
    params: adamMomentParams(arg.beta1, arg.weightDecay, outputLength),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: `${shapeStr(shape)}_${arg.beta1}_${arg.weightDecay}`,
  });
}

function adamUpdateVDirect(
  device: KernelDevice,
  param: ResidentValue,
  grad: ResidentValue,
  v: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectAdamUpdateArg(rawArg, "ADAM_UPDATE_V");
  expectShape(param.shape, shape, "ADAM_UPDATE_V.param");
  expectShape(grad.shape, shape, "ADAM_UPDATE_V.grad");
  expectShape(v.shape, shape, "ADAM_UPDATE_V.v");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_adam_update_v",
    wgsl: ADAM_UPDATE_V_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [param.buffer, grad.buffer, v.buffer],
    outputLength,
    params: adamMomentParams(arg.beta2, arg.weightDecay, outputLength),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: `${shapeStr(shape)}_${arg.beta2}_${arg.weightDecay}`,
  });
}

function adamUpdateParamDirect(
  device: KernelDevice,
  param: ResidentValue,
  m: ResidentValue,
  v: ResidentValue,
  shape: readonly number[],
  rawArg: unknown,
): DirectDispatchResult {
  const arg = expectAdamUpdateArg(rawArg, "ADAM_UPDATE_PARAM");
  expectShape(param.shape, shape, "ADAM_UPDATE_PARAM.param");
  expectShape(m.shape, shape, "ADAM_UPDATE_PARAM.m");
  expectShape(v.shape, shape, "ADAM_UPDATE_PARAM.v");
  const outputLength = numel(shape);
  return runDirect(device, {
    name: "tensor_plan_adam_update_param",
    wgsl: ADAM_UPDATE_PARAM_WGSL,
    workgroupSize: [64, 1, 1],
  }, {
    inputBuffers: [param.buffer, m.buffer, v.buffer],
    outputLength,
    params: adamParamParams(arg, outputLength),
    dispatchCount: [outputLength, 1, 1],
    cacheKeySuffix: [
      shapeStr(shape),
      arg.lr,
      arg.beta1,
      arg.beta2,
      arg.eps,
      arg.step,
    ].join("_"),
  });
}

function elementwise(
  step: TensorPlanStep,
  values: Map<number, ResidentValue>,
  ops: readonly FusedOp[],
  arity: number,
  device: KernelDevice,
): ResidentValue {
  if (step.inputIds.length !== arity) {
    throw new KernelError(`tensor plan ${step.op} expected ${arity} inputs`);
  }
  const inputs = step.inputIds.map((id) => requireValue(values, id, step.op));
  for (const input of inputs) {
    if (numel(input.shape) !== numel(step.shape)) {
      throw new KernelError(
        `tensor plan ${step.op} needs equal numel inputs in v0: got ${shapeStr(input.shape)} vs ${shapeStr(step.shape)}`,
      );
    }
  }
  return fromDirect(
    step,
    fusedElementwiseDirect(
      device,
      inputs.map((input) => input.buffer),
      ops,
      numel(step.shape),
    ),
  );
}

function fromDirect(step: TensorPlanStep, result: DirectDispatchResult): ResidentValue {
  return {
    buffer: result.buffer,
    shape: step.shape,
    owns: true,
    byteLength: result.byteLength,
  };
}

function validatePlan(plan: TensorGpuPlan): void {
  if (plan.hasCustomOps) {
    throw new KernelError("tensor plan runtime refuses CUSTOM-backed plans");
  }
  if (plan.materializationBoundary !== "root") {
    throw new KernelError("tensor plan runtime only supports root materialization");
  }
  const seen = new Set<number>();
  for (const step of plan.steps) {
    if (seen.has(step.valueId)) {
      throw new KernelError(`tensor plan value ${step.valueId} produced twice`);
    }
    for (const inputId of step.inputIds) {
      if (!seen.has(inputId)) {
        throw new KernelError(`tensor plan step ${step.step} reads ${inputId} before production`);
      }
    }
    seen.add(step.valueId);
  }
  if (!seen.has(plan.rootId)) {
    throw new KernelError(`tensor plan root ${plan.rootId} missing from steps`);
  }
}

function requireValue(
  values: Map<number, ResidentValue>,
  valueId: number | undefined,
  op: string,
): ResidentValue {
  if (valueId === undefined) throw new KernelError(`tensor plan ${op} missing input`);
  const value = values.get(valueId);
  if (!value) throw new KernelError(`tensor plan ${op} input ${valueId} not resident`);
  return value;
}

function validateNumel(length: number, shape: readonly number[], name: string): void {
  const expected = numel(shape);
  if (length !== expected) {
    throw new KernelError(`${name} data length ${length} does not match shape ${shapeStr(shape)} (${expected})`);
  }
}

function expectShape(actual: readonly number[], expected: readonly number[], name: string): void {
  if (actual.length !== expected.length) {
    throw new KernelError(`${name} expected rank ${expected.length}, got ${shapeStr(actual)}`);
  }
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) {
      throw new KernelError(
        `${name} shape mismatch at dim ${i}: expected ${expected[i]}, got ${actual[i]}`,
      );
    }
  }
}

function assertPositiveInt(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new KernelError(`tensor plan ${name} must be a positive integer`);
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
    throw new KernelError(`tensor plan ${op}.padding must be a non-negative integer`);
  }
  if (cIn % groups !== 0 || cOut % groups !== 0) {
    throw new KernelError(`tensor plan ${op} channels must be divisible by groups`);
  }
}

function expectConv1dArg(rawArg: unknown, op: string): Conv1dArg {
  const arg = expectRecord(rawArg, `tensor plan ${op}.arg`);
  return {
    n: expectNumber(arg.n, `${op}.n`),
    cIn: expectNumber(arg.c_in, `${op}.c_in`),
    lIn: expectNumber(arg.l_in, `${op}.l_in`),
    cOut: expectNumber(arg.c_out, `${op}.c_out`),
    k: expectNumber(arg.k, `${op}.k`),
    stride: expectNumber(arg.stride, `${op}.stride`),
    padding: expectNumber(arg.padding, `${op}.padding`),
    dilation: expectNumber(arg.dilation, `${op}.dilation`),
    groups: expectNumber(arg.groups, `${op}.groups`),
    lOut: expectNumber(arg.l_out, `${op}.l_out`),
  };
}

function conv1dParams(arg: Conv1dArg): Uint32Array {
  return new Uint32Array([
    arg.n,
    arg.cIn,
    arg.lIn,
    arg.cOut,
    arg.k,
    arg.stride,
    arg.padding,
    arg.dilation,
    arg.groups,
    arg.lOut,
    0,
    0,
  ]);
}

function conv1dCacheKey(arg: Conv1dArg): string {
  return [
    arg.n,
    arg.cIn,
    arg.lIn,
    arg.cOut,
    arg.k,
    arg.stride,
    arg.padding,
    arg.dilation,
    arg.groups,
    arg.lOut,
  ].join("_");
}

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
    throw new KernelError(`tensor plan ${op} padding must be non-negative integers`);
  }
  if (cIn % groups !== 0 || cOut % groups !== 0) {
    throw new KernelError(`tensor plan ${op} channels must be divisible by groups`);
  }
}

function expectConv2dArg(rawArg: unknown, op: string): Conv2dArg {
  const arg = expectRecord(rawArg, `tensor plan ${op}.arg`);
  return {
    n: expectNumber(arg.n, `${op}.n`),
    cIn: expectNumber(arg.c_in, `${op}.c_in`),
    h: expectNumber(arg.h, `${op}.h`),
    w: expectNumber(arg.w, `${op}.w`),
    cOut: expectNumber(arg.c_out, `${op}.c_out`),
    kh: expectNumber(arg.kh, `${op}.kh`),
    kw: expectNumber(arg.kw, `${op}.kw`),
    strideH: expectNumber(arg.stride_h, `${op}.stride_h`),
    strideW: expectNumber(arg.stride_w, `${op}.stride_w`),
    padH: expectNumber(arg.pad_h, `${op}.pad_h`),
    padW: expectNumber(arg.pad_w, `${op}.pad_w`),
    dilationH: expectNumber(arg.dilation_h, `${op}.dilation_h`),
    dilationW: expectNumber(arg.dilation_w, `${op}.dilation_w`),
    groups: expectNumber(arg.groups, `${op}.groups`),
    outH: expectNumber(arg.out_h, `${op}.out_h`),
    outW: expectNumber(arg.out_w, `${op}.out_w`),
  };
}

function validateConv2dShapeFromArg(op: string, arg: Conv2dArg): void {
  validateConv2dShape(
    op,
    arg.n,
    arg.cIn,
    arg.h,
    arg.w,
    arg.cOut,
    arg.kh,
    arg.kw,
    arg.strideH,
    arg.strideW,
    arg.padH,
    arg.padW,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outH,
    arg.outW,
  );
}

function conv2dParams(arg: Conv2dArg): Uint32Array {
  return new Uint32Array([
    arg.n,
    arg.cIn,
    arg.h,
    arg.w,
    arg.cOut,
    arg.kh,
    arg.kw,
    arg.strideH,
    arg.strideW,
    arg.padH,
    arg.padW,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outH,
    arg.outW,
  ]);
}

function conv2dCacheKey(arg: Conv2dArg): string {
  return [
    arg.n,
    arg.cIn,
    arg.h,
    arg.w,
    arg.cOut,
    arg.kh,
    arg.kw,
    arg.strideH,
    arg.strideW,
    arg.padH,
    arg.padW,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outH,
    arg.outW,
  ].join("_");
}

function validateConvTranspose2dShape(
  op: string,
  n: number,
  cIn: number,
  h: number,
  w: number,
  cOut: number,
  cOutPerGroup: number,
  kh: number,
  kw: number,
  strideH: number,
  strideW: number,
  padH: number,
  padW: number,
  outputPadH: number,
  outputPadW: number,
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
    [cOutPerGroup, "cOutPerGroup"],
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
    throw new KernelError(`tensor plan ${op} padding must be non-negative integers`);
  }
  if (
    !Number.isInteger(outputPadH) || outputPadH < 0 ||
    !Number.isInteger(outputPadW) || outputPadW < 0
  ) {
    throw new KernelError(`tensor plan ${op} output_padding must be non-negative integers`);
  }
  if (cIn % groups !== 0 || cOut !== cOutPerGroup * groups) {
    throw new KernelError(`tensor plan ${op} channels must match groups`);
  }
  if (outputPadH >= strideH || outputPadW >= strideW) {
    throw new KernelError(`tensor plan ${op} output_padding must be smaller than stride`);
  }
}

function expectConvTranspose2dArg(rawArg: unknown, op: string): ConvTranspose2dArg {
  const arg = expectRecord(rawArg, `tensor plan ${op}.arg`);
  return {
    n: expectNumber(arg.n, `${op}.n`),
    cIn: expectNumber(arg.c_in, `${op}.c_in`),
    h: expectNumber(arg.h, `${op}.h`),
    w: expectNumber(arg.w, `${op}.w`),
    cOut: expectNumber(arg.c_out, `${op}.c_out`),
    cOutPerGroup: expectNumber(arg.c_out_per_group, `${op}.c_out_per_group`),
    kh: expectNumber(arg.kh, `${op}.kh`),
    kw: expectNumber(arg.kw, `${op}.kw`),
    strideH: expectNumber(arg.stride_h, `${op}.stride_h`),
    strideW: expectNumber(arg.stride_w, `${op}.stride_w`),
    padH: expectNumber(arg.pad_h, `${op}.pad_h`),
    padW: expectNumber(arg.pad_w, `${op}.pad_w`),
    outputPadH: expectNumber(arg.output_pad_h, `${op}.output_pad_h`),
    outputPadW: expectNumber(arg.output_pad_w, `${op}.output_pad_w`),
    dilationH: expectNumber(arg.dilation_h, `${op}.dilation_h`),
    dilationW: expectNumber(arg.dilation_w, `${op}.dilation_w`),
    groups: expectNumber(arg.groups, `${op}.groups`),
    outH: expectNumber(arg.out_h, `${op}.out_h`),
    outW: expectNumber(arg.out_w, `${op}.out_w`),
  };
}

function validateConvTranspose2dShapeFromArg(op: string, arg: ConvTranspose2dArg): void {
  validateConvTranspose2dShape(
    op,
    arg.n,
    arg.cIn,
    arg.h,
    arg.w,
    arg.cOut,
    arg.cOutPerGroup,
    arg.kh,
    arg.kw,
    arg.strideH,
    arg.strideW,
    arg.padH,
    arg.padW,
    arg.outputPadH,
    arg.outputPadW,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outH,
    arg.outW,
  );
}

function convTranspose2dParams(arg: ConvTranspose2dArg): Uint32Array {
  return new Uint32Array([
    arg.n,
    arg.cIn,
    arg.h,
    arg.w,
    arg.cOut,
    arg.cOutPerGroup,
    arg.kh,
    arg.kw,
    arg.strideH,
    arg.strideW,
    arg.padH,
    arg.padW,
    arg.outputPadH,
    arg.outputPadW,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outH,
    arg.outW,
    0,
  ]);
}

function convTranspose2dCacheKey(arg: ConvTranspose2dArg): string {
  return [
    arg.n,
    arg.cIn,
    arg.h,
    arg.w,
    arg.cOut,
    arg.cOutPerGroup,
    arg.kh,
    arg.kw,
    arg.strideH,
    arg.strideW,
    arg.padH,
    arg.padW,
    arg.outputPadH,
    arg.outputPadW,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outH,
    arg.outW,
  ].join("_");
}

function validateConv3dShape(
  op: string,
  n: number,
  cIn: number,
  d: number,
  h: number,
  w: number,
  cOut: number,
  kd: number,
  kh: number,
  kw: number,
  strideD: number,
  strideH: number,
  strideW: number,
  padD: number,
  padH: number,
  padW: number,
  dilationD: number,
  dilationH: number,
  dilationW: number,
  groups: number,
  outD: number,
  outH: number,
  outW: number,
): void {
  for (const [value, name] of [
    [n, "n"],
    [cIn, "cIn"],
    [d, "d"],
    [h, "h"],
    [w, "w"],
    [cOut, "cOut"],
    [kd, "kd"],
    [kh, "kh"],
    [kw, "kw"],
    [strideD, "strideD"],
    [strideH, "strideH"],
    [strideW, "strideW"],
    [dilationD, "dilationD"],
    [dilationH, "dilationH"],
    [dilationW, "dilationW"],
    [groups, "groups"],
    [outD, "outD"],
    [outH, "outH"],
    [outW, "outW"],
  ] as const) {
    assertPositiveInt(value, `${op}.${name}`);
  }
  if (
    !Number.isInteger(padD) || padD < 0 ||
    !Number.isInteger(padH) || padH < 0 ||
    !Number.isInteger(padW) || padW < 0
  ) {
    throw new KernelError(`tensor plan ${op} padding must be non-negative integers`);
  }
  if (cIn % groups !== 0 || cOut % groups !== 0) {
    throw new KernelError(`tensor plan ${op} channels must be divisible by groups`);
  }
}

function expectConv3dArg(rawArg: unknown, op: string): Conv3dArg {
  const arg = expectRecord(rawArg, `tensor plan ${op}.arg`);
  return {
    n: expectNumber(arg.n, `${op}.n`),
    cIn: expectNumber(arg.c_in, `${op}.c_in`),
    d: expectNumber(arg.d, `${op}.d`),
    h: expectNumber(arg.h, `${op}.h`),
    w: expectNumber(arg.w, `${op}.w`),
    cOut: expectNumber(arg.c_out, `${op}.c_out`),
    kd: expectNumber(arg.kd, `${op}.kd`),
    kh: expectNumber(arg.kh, `${op}.kh`),
    kw: expectNumber(arg.kw, `${op}.kw`),
    strideD: expectNumber(arg.stride_d, `${op}.stride_d`),
    strideH: expectNumber(arg.stride_h, `${op}.stride_h`),
    strideW: expectNumber(arg.stride_w, `${op}.stride_w`),
    padD: expectNumber(arg.pad_d, `${op}.pad_d`),
    padH: expectNumber(arg.pad_h, `${op}.pad_h`),
    padW: expectNumber(arg.pad_w, `${op}.pad_w`),
    dilationD: expectNumber(arg.dilation_d, `${op}.dilation_d`),
    dilationH: expectNumber(arg.dilation_h, `${op}.dilation_h`),
    dilationW: expectNumber(arg.dilation_w, `${op}.dilation_w`),
    groups: expectNumber(arg.groups, `${op}.groups`),
    outD: expectNumber(arg.out_d, `${op}.out_d`),
    outH: expectNumber(arg.out_h, `${op}.out_h`),
    outW: expectNumber(arg.out_w, `${op}.out_w`),
  };
}

function validateConv3dShapeFromArg(op: string, arg: Conv3dArg): void {
  validateConv3dShape(
    op,
    arg.n,
    arg.cIn,
    arg.d,
    arg.h,
    arg.w,
    arg.cOut,
    arg.kd,
    arg.kh,
    arg.kw,
    arg.strideD,
    arg.strideH,
    arg.strideW,
    arg.padD,
    arg.padH,
    arg.padW,
    arg.dilationD,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outD,
    arg.outH,
    arg.outW,
  );
}

function conv3dParams(arg: Conv3dArg): Uint32Array {
  return new Uint32Array([
    arg.n,
    arg.cIn,
    arg.d,
    arg.h,
    arg.w,
    arg.cOut,
    arg.kd,
    arg.kh,
    arg.kw,
    arg.strideD,
    arg.strideH,
    arg.strideW,
    arg.padD,
    arg.padH,
    arg.padW,
    arg.dilationD,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outD,
    arg.outH,
    arg.outW,
    0,
    0,
  ]);
}

function conv3dCacheKey(arg: Conv3dArg): string {
  return [
    arg.n,
    arg.cIn,
    arg.d,
    arg.h,
    arg.w,
    arg.cOut,
    arg.kd,
    arg.kh,
    arg.kw,
    arg.strideD,
    arg.strideH,
    arg.strideW,
    arg.padD,
    arg.padH,
    arg.padW,
    arg.dilationD,
    arg.dilationH,
    arg.dilationW,
    arg.groups,
    arg.outD,
    arg.outH,
    arg.outW,
  ].join("_");
}

function expectSgdUpdateArg(rawArg: unknown): SgdUpdateArg {
  const arg = expectRecord(rawArg, "tensor plan SGD_UPDATE.arg");
  const lr = expectFiniteNumber(arg.lr, "SGD_UPDATE.lr");
  const weightDecay = expectFiniteNumber(
    arg.weight_decay ?? arg.weightDecay ?? 0,
    "SGD_UPDATE.weight_decay",
  );
  if (lr < 0) throw new KernelError("tensor plan SGD_UPDATE.lr must be >= 0");
  return { lr, weightDecay };
}

function sgdUpdateParams(arg: SgdUpdateArg, total: number): Uint32Array {
  const buffer = new ArrayBuffer(16);
  const view = new DataView(buffer);
  view.setFloat32(0, arg.lr, true);
  view.setFloat32(4, arg.weightDecay, true);
  view.setUint32(8, total, true);
  view.setUint32(12, 0, true);
  return new Uint32Array(buffer);
}

function expectAdamwUpdateArg(rawArg: unknown, op: string): AdamwUpdateArg {
  const arg = expectRecord(rawArg, `tensor plan ${op}.arg`);
  const lr = expectFiniteNumber(arg.lr ?? 0.001, `${op}.lr`);
  const beta1 = expectFiniteNumber(arg.beta1, `${op}.beta1`);
  const beta2 = expectFiniteNumber(arg.beta2, `${op}.beta2`);
  const eps = expectFiniteNumber(arg.eps ?? 1e-8, `${op}.eps`);
  const weightDecay = expectFiniteNumber(
    arg.weight_decay ?? arg.weightDecay ?? 0,
    `${op}.weight_decay`,
  );
  const step = expectNumber(arg.step, `${op}.step`);
  if (lr < 0) throw new KernelError(`tensor plan ${op}.lr must be >= 0`);
  if (step <= 0) throw new KernelError(`tensor plan ${op}.step must be >= 1`);
  return { lr, beta1, beta2, eps, weightDecay, step };
}

function expectAdamUpdateArg(rawArg: unknown, op: string): AdamUpdateArg {
  const arg = expectRecord(rawArg, `tensor plan ${op}.arg`);
  const lr = expectFiniteNumber(arg.lr ?? 0.001, `${op}.lr`);
  const beta1 = expectFiniteNumber(arg.beta1 ?? 0.9, `${op}.beta1`);
  const beta2 = expectFiniteNumber(arg.beta2 ?? 0.999, `${op}.beta2`);
  const eps = expectFiniteNumber(arg.eps ?? 1e-8, `${op}.eps`);
  const weightDecay = expectFiniteNumber(
    arg.weight_decay ?? arg.weightDecay ?? 0,
    `${op}.weight_decay`,
  );
  const step = expectNumber(arg.step ?? 1, `${op}.step`);
  if (lr < 0) throw new KernelError(`tensor plan ${op}.lr must be >= 0`);
  if (step <= 0) throw new KernelError(`tensor plan ${op}.step must be >= 1`);
  return { lr, beta1, beta2, eps, weightDecay, step };
}

function expectLayerNormArg(rawArg: unknown, op: string): LayerNormArg {
  const arg = expectRecord(rawArg, `tensor plan ${op}.arg`);
  const normalizedShape = expectArray(
    arg.normalized_shape ?? arg.normalizedShape,
    `${op}.normalized_shape`,
  ).map((v, i) => expectNumber(v, `${op}.normalized_shape[${i}]`));
  const rows = expectNumber(arg.rows, `${op}.rows`);
  const cols = expectNumber(arg.cols, `${op}.cols`);
  const eps = expectFiniteNumber(arg.eps ?? 1e-5, `${op}.eps`);
  if (normalizedShape.length === 0) {
    throw new KernelError(`tensor plan ${op}.normalized_shape must be non-empty`);
  }
  if (rows <= 0 || cols <= 0) {
    throw new KernelError(`tensor plan ${op}.rows and cols must be positive`);
  }
  if (numel(normalizedShape) !== cols) {
    throw new KernelError(`tensor plan ${op}.cols must equal normalized_shape numel`);
  }
  if (eps <= 0) throw new KernelError(`tensor plan ${op}.eps must be > 0`);
  return { normalizedShape, rows, cols, eps };
}

function adamwMvParams(beta: number, total: number): Uint32Array {
  const buffer = new ArrayBuffer(16);
  const view = new DataView(buffer);
  view.setFloat32(0, beta, true);
  view.setUint32(4, total, true);
  view.setUint32(8, 0, true);
  view.setUint32(12, 0, true);
  return new Uint32Array(buffer);
}

function adamwParamParams(arg: AdamwUpdateArg, total: number): Uint32Array {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setFloat32(0, arg.lr, true);
  view.setFloat32(4, arg.beta1, true);
  view.setFloat32(8, arg.beta2, true);
  view.setFloat32(12, arg.eps, true);
  view.setFloat32(16, arg.weightDecay, true);
  view.setUint32(20, arg.step, true);
  view.setUint32(24, total, true);
  view.setUint32(28, 0, true);
  return new Uint32Array(buffer);
}

function adamMomentParams(beta: number, weightDecay: number, total: number): Uint32Array {
  const buffer = new ArrayBuffer(16);
  const view = new DataView(buffer);
  view.setFloat32(0, beta, true);
  view.setFloat32(4, weightDecay, true);
  view.setUint32(8, total, true);
  view.setUint32(12, 0, true);
  return new Uint32Array(buffer);
}

function adamParamParams(arg: AdamUpdateArg, total: number): Uint32Array {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setFloat32(0, arg.lr, true);
  view.setFloat32(4, arg.beta1, true);
  view.setFloat32(8, arg.beta2, true);
  view.setFloat32(12, arg.eps, true);
  view.setUint32(16, arg.step, true);
  view.setUint32(20, total, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 0, true);
  return new Uint32Array(buffer);
}

function layerNormParams(arg: LayerNormArg): Uint32Array {
  const epsBuf = new Float32Array([arg.eps]);
  const epsBits = new Uint32Array(epsBuf.buffer, epsBuf.byteOffset, 1)[0]!;
  return new Uint32Array([arg.rows, arg.cols, epsBits, 0]);
}

function pad4(values: readonly number[]): [number, number, number, number] {
  if (values.length > 4) throw new KernelError("tensor plan rank > 4 unsupported in v0");
  return [
    values[0] ?? 1,
    values[1] ?? 1,
    values[2] ?? 1,
    values[3] ?? 1,
  ];
}

function numel(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= Math.max(d, 1);
  return n;
}

function shapeStr(shape: readonly number[]): string {
  return `[${shape.join(",")}]`;
}

function assertNever(value: never): never {
  throw new KernelError(`unsupported tensor plan op ${String(value)}`);
}

function expectRecord(value: unknown, name: string): RawRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new KernelError(`${name} must be an object`);
  }
  return value as RawRecord;
}

function expectArray(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new KernelError(`${name} must be an array`);
  return value;
}

function expectNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value)) {
    throw new KernelError(`${name} must be an integer`);
  }
  return value;
}

function expectFiniteNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new KernelError(`${name} must be a finite number`);
  }
  return value;
}

function expectString(value: unknown, name: string): string {
  if (typeof value !== "string") throw new KernelError(`${name} must be a string`);
  return value;
}

function expectBoolean(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new KernelError(`${name} must be a boolean`);
  return value;
}

function expectAxes(arg: unknown, rank: number, op: string): number[] {
  const obj = expectRecord(arg, `tensor plan ${op}.arg`);
  const rawAxes = expectArray(obj.axes, `tensor plan ${op}.arg.axes`);
  if (rawAxes.length !== rank) {
    throw new KernelError(`tensor plan ${op} axes length must equal rank`);
  }
  const seen = new Set<number>();
  return rawAxes.map((axis, i) => {
    const n = expectNumber(axis, `tensor plan ${op}.arg.axes[${i}]`);
    if (n < 0 || n >= rank) {
      throw new KernelError(`tensor plan ${op} axis ${n} out of range for rank ${rank}`);
    }
    if (seen.has(n)) throw new KernelError(`tensor plan ${op} duplicate axis ${n}`);
    seen.add(n);
    return n;
  });
}

function expectReduceSpec(arg: unknown, rank: number): ReduceSpec {
  const obj = expectRecord(arg, "tensor plan REDUCE.arg");
  const op = expectString(obj.op, "tensor plan REDUCE.arg.op");
  if (op !== "sum" && op !== "mean") {
    throw new KernelError(`tensor plan REDUCE supports sum/mean in v0 (got ${op})`);
  }
  const rawAxis = obj.axis;
  let axes: number[];
  if (rawAxis === null || rawAxis === undefined) {
    axes = Array.from({ length: rank }, (_, i) => i);
  } else if (Array.isArray(rawAxis)) {
    axes = rawAxis.map((axis, i) => normalizeAxis(
      expectNumber(axis, `tensor plan REDUCE.arg.axis[${i}]`),
      rank,
    ));
  } else {
    axes = [normalizeAxis(expectNumber(rawAxis, "tensor plan REDUCE.arg.axis"), rank)];
  }
  const seen = new Set<number>();
  for (const axis of axes) {
    if (seen.has(axis)) throw new KernelError(`tensor plan REDUCE duplicate axis ${axis}`);
    seen.add(axis);
  }
  return { op, axes };
}

function normalizeAxis(axis: number, rank: number): number {
  const n = axis < 0 ? axis + rank : axis;
  if (n < 0 || n >= rank) {
    throw new KernelError(`tensor plan REDUCE axis ${axis} out of range for rank ${rank}`);
  }
  return n;
}

function expectFloat32(value: unknown, name: string): "float32" {
  if (value !== "float32") throw new KernelError(`${name} must be "float32"`);
  return "float32";
}

function expectRootBoundary(value: unknown): "root" {
  if (value !== "root") {
    throw new KernelError(`tensor plan.materializationBoundary must be "root"`);
  }
  return "root";
}

function expectFalse(value: unknown, name: string): false {
  if (value !== false) throw new KernelError(`${name} must be false`);
  return false;
}

function expectOp(value: unknown, name: string): TensorPlanOp {
  const op = expectString(value, name);
  switch (op) {
    case "BUFFER":
    case "LOAD":
    case "CONST":
    case "CAST":
    case "ADD":
    case "MUL":
    case "DIV":
    case "NEG":
    case "EXP":
    case "LOG":
    case "MATMUL":
    case "REDUCE":
    case "RESHAPE":
    case "PERMUTE":
    case "BROADCAST_TO":
    case "CONV1D":
    case "CONV1D_BACKWARD_INPUT":
    case "CONV1D_BACKWARD_WEIGHT":
    case "CONV1D_BACKWARD_BIAS":
    case "CONV2D":
    case "CONV2D_BACKWARD_INPUT":
    case "CONV2D_BACKWARD_WEIGHT":
    case "CONV2D_BACKWARD_BIAS":
    case "CONV_TRANSPOSE2D":
    case "CONV_TRANSPOSE2D_BACKWARD_INPUT":
    case "CONV_TRANSPOSE2D_BACKWARD_WEIGHT":
    case "CONV_TRANSPOSE2D_BACKWARD_BIAS":
    case "CONV3D":
    case "CONV3D_BACKWARD_INPUT":
    case "CONV3D_BACKWARD_WEIGHT":
    case "CONV3D_BACKWARD_BIAS":
    case "LAYER_NORM":
    case "LAYER_NORM_BACKWARD_INPUT":
    case "LAYER_NORM_BACKWARD_WEIGHT":
    case "LAYER_NORM_BACKWARD_BIAS":
    case "SGD_UPDATE":
    case "ADAMW_UPDATE_M":
    case "ADAMW_UPDATE_V":
    case "ADAMW_UPDATE_PARAM":
    case "ADAM_UPDATE_M":
    case "ADAM_UPDATE_V":
    case "ADAM_UPDATE_PARAM":
      return op;
    default:
      throw new KernelError(`${name} unsupported op ${JSON.stringify(op)}`);
  }
}

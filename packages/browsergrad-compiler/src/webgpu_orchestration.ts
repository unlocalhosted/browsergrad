import {
  defineWgslKernelProgram,
  type WgslKernelProgram,
  type WgslKernelRunInput,
  type WgslKernelSequenceStep,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames } from "./ast_queries.js";
import { createCudaHostDynamicLaunchPlan } from "./dynamic_launch.js";
import { createCudaPeerCopyPlan, type CudaPeerCopyOperation } from "./peer_copy.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import { createCudaGridSyncPhasePlan, createCudaRuntimePlan } from "./runtime_plan.js";
import { emitKernelIrWgsl } from "./wgsl.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
  type CudaLiteDiagnostic,
  type KernelLaunch,
} from "./types.js";

export type CudaWebGpuExecutionPlanKind =
  | "single-dispatch"
  | "grid-sync-phases"
  | "host-dynamic-launch"
  | "host-peer-copy";

export type CudaWebGpuExecutionPlan =
  | {
      readonly supported: true;
      readonly kind: CudaWebGpuExecutionPlanKind;
      readonly steps: readonly WgslKernelSequenceStep[];
      readonly input: WgslKernelRunInput;
    }
  | {
      readonly supported: false;
      readonly reason: string;
      readonly diagnostics: readonly CudaLiteDiagnostic[];
    };

export interface CudaWebGpuExecutionPlanOptions {
  readonly compileKernel?: (
    source: string,
    options?: CompileCudaLiteOptions,
  ) => CompiledCudaLiteKernel;
}

const peerCopyProgramCache = new Map<CudaPeerCopyOperation["valueType"], WgslKernelProgram>();

export function createCudaWebGpuExecutionPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: CudaWebGpuExecutionPlanOptions = {},
): CudaWebGpuExecutionPlan {
  const runtimePlan = createCudaRuntimePlan(compiled);
  const blockers: string[] = [];

  const gridSyncPhasePlan = createCudaGridSyncPhasePlan(compiled.ir);
  const gridSyncPlan = createGridSyncWebGpuPlan(compiled, input, launch, gridSyncPhasePlan);
  if (gridSyncPlan) return gridSyncPlan;
  if (runtimePlan.operations.some((operation) => operation.kind === "grid-sync") && !gridSyncPhasePlan.supported) {
    blockers.push(`grid-sync: ${gridSyncPhasePlan.reason}`);
  }

  const hostDynamicPlan = createCudaHostDynamicLaunchPlan(compiled, input, launch);
  const dynamicLaunchPlan = createHostLiftedDynamicWebGpuPlan(compiled, input, launch, options, hostDynamicPlan);
  if (dynamicLaunchPlan) return dynamicLaunchPlan;
  if (runtimePlan.operations.some((operation) => operation.kind === "device-launch") && !hostDynamicPlan.supported) {
    blockers.push(`device-launch: ${hostDynamicPlan.reason}`);
  }

  const peerCopyRuntimePlan = createCudaPeerCopyPlan(compiled, input, launch);
  const peerCopyPlan = createHostLiftedPeerCopyWebGpuPlan(compiled, input, launch, peerCopyRuntimePlan);
  if (peerCopyPlan) return peerCopyPlan;
  if (runtimePlan.operations.some((operation) => operation.kind === "peer-copy") && !peerCopyRuntimePlan.supported) {
    blockers.push(`peer-copy: ${peerCopyRuntimePlan.reason}`);
  }

  const unsupported = createReferenceOnlyRuntimePlan(compiled, blockers);
  if (unsupported) return unsupported;

  return createSingleDispatchWebGpuPlan(compiled, input, launch);
}

export function normalizeCudaWebGpuReadback(
  compiled: CompiledCudaLiteKernel,
  buffers: Readonly<Record<string, WgslTypedArray>>,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = { ...buffers };
  for (const pool of compiled.ir.params.filter(isDevicePoolParam)) {
    const data = buffers[poolDataName(pool.name)];
    if (data) out[pool.name] = data;
  }
  for (const poolName of collectExternalDevicePoolNames(compiled.ir.body)) {
    const data = buffers[poolDataName(poolName)];
    if (data) out[poolName] = data;
  }
  return out;
}

export function normalizeCudaWebGpuReadbackNames(
  compiled: CompiledCudaLiteKernel,
  names: readonly string[],
): readonly string[] {
  const aliases = new Map<string, string>();
  for (const pool of compiled.ir.params.filter(isDevicePoolParam)) {
    aliases.set(pool.name, poolDataName(pool.name));
  }
  for (const poolName of collectExternalDevicePoolNames(compiled.ir.body)) {
    aliases.set(poolName, poolDataName(poolName));
  }
  return [...new Set(names.map((name) => aliases.get(name) ?? name))];
}

function createGridSyncWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  gridSyncPhasePlan: ReturnType<typeof createCudaGridSyncPhasePlan>,
): CudaWebGpuExecutionPlan | undefined {
  if (!gridSyncPhasePlan.supported || gridSyncPhasePlan.modules.length <= 1) return undefined;
  const wgslInput = createWgslRunInput(compiled, input);
  const dispatchCount = dispatchCountForLaunch(launch);
  const steps = gridSyncPhasePlan.modules.map((module): WgslKernelSequenceStep => ({
    program: emitKernelIrWgsl(module, { features: featureOptionsFor(module.requiredFeatures) }).program,
    launch: { dispatchCount },
    ...(wgslInput.uniforms === undefined ? {} : { uniforms: wgslInput.uniforms }),
  }));
  return {
    supported: true,
    kind: "grid-sync-phases",
    steps,
    input: wgslInput,
  };
}

function createHostLiftedPeerCopyWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  plan: ReturnType<typeof createCudaPeerCopyPlan>,
): CudaWebGpuExecutionPlan | undefined {
  if (!plan.supported || plan.copies.length === 0) return undefined;
  const parentInput = createWgslRunInput(compiled, input);
  const steps: WgslKernelSequenceStep[] = [{
    program: compiled.wgslProgram,
    launch: { dispatchCount: dispatchCountForLaunch(launch) },
    ...(parentInput.uniforms === undefined ? {} : { uniforms: parentInput.uniforms }),
  }];

  appendPeerCopySteps(steps, plan.copies);

  return {
    supported: true,
    kind: "host-peer-copy",
    steps,
    input: {
      buffers: { ...parentInput.buffers },
      ...(parentInput.residentBuffers === undefined ? {} : { residentBuffers: parentInput.residentBuffers }),
      ...(parentInput.textures === undefined ? {} : { textures: parentInput.textures }),
      ...(parentInput.readback === undefined ? {} : { readback: parentInput.readback }),
    },
  };
}

function createHostLiftedDynamicWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: CudaWebGpuExecutionPlanOptions,
  plan: ReturnType<typeof createCudaHostDynamicLaunchPlan>,
): CudaWebGpuExecutionPlan | undefined {
  if (!plan.supported || plan.launches.length === 0) return undefined;
  if (!options.compileKernel) {
    return {
      supported: false,
      reason: "dynamic child compiler unavailable for WebGPU host orchestration",
      diagnostics: referenceRuntimeDiagnostics(
        compiled,
        "dynamic child compiler unavailable for WebGPU host orchestration",
      ),
    };
  }

  const parentInput = createWgslRunInput(compiled, input);
  const buffers: Record<string, WgslTypedArray> = { ...parentInput.buffers };
  const residentBuffers = { ...parentInput.residentBuffers };
  const steps: WgslKernelSequenceStep[] = [{
    program: compiled.wgslProgram,
    launch: { dispatchCount: dispatchCountForLaunch(launch) },
    ...(parentInput.uniforms === undefined ? {} : { uniforms: parentInput.uniforms }),
  }];
  const childCompileCache = new Map<string, CompiledCudaLiteKernel>();

  for (const item of plan.launches) {
    const childCompiled = getOrCompileDynamicChild(
      compiled,
      item,
      childCompileCache,
      options.compileKernel,
    );
    if (!childCompiled) return undefined;
    const childRuntime = createCudaRuntimePlan(childCompiled);
    if (!childRuntime.operations.every((operation) => operation.kind === "device-sync" || operation.kind === "peer-copy")) return undefined;
    const childWgslInput = createWgslRunInput(childCompiled, item.input);
    for (const [name, value] of Object.entries(childWgslInput.buffers)) {
      buffers[item.storageAliases[name] ?? name] = value;
    }
    for (const [name, value] of Object.entries(childWgslInput.residentBuffers ?? {})) {
      residentBuffers[item.storageAliases[name] ?? name] = value;
    }
    const childLaunch = { gridDim: item.gridDim, blockDim: item.blockDim };
    steps.push({
      program: childCompiled.wgslProgram,
      launch: { dispatchCount: dispatchCountForLaunch(childLaunch) },
      storageAliases: item.storageAliases,
      ...(childWgslInput.uniforms === undefined ? {} : { uniforms: childWgslInput.uniforms }),
    });
    if (childRuntime.operations.some((operation) => operation.kind === "peer-copy")) {
      const peerCopyPlan = createCudaPeerCopyPlan(childCompiled, item.input, childLaunch);
      if (!peerCopyPlan.supported) return undefined;
      appendPeerCopySteps(steps, peerCopyPlan.copies, item.storageAliases);
    }
  }

  return {
    supported: true,
    kind: "host-dynamic-launch",
    steps,
    input: {
      buffers,
      ...(Object.keys(residentBuffers).length === 0 ? {} : { residentBuffers }),
      ...(parentInput.textures === undefined ? {} : { textures: parentInput.textures }),
      ...(parentInput.readback === undefined ? {} : { readback: parentInput.readback }),
    },
  };
}

function createSingleDispatchWebGpuPlan(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): CudaWebGpuExecutionPlan {
  const wgslInput = createWgslRunInput(compiled, input);
  return {
    supported: true,
    kind: "single-dispatch",
    steps: [{
      program: compiled.wgslProgram,
      launch: { dispatchCount: dispatchCountForLaunch(launch) },
      ...(wgslInput.uniforms === undefined ? {} : { uniforms: wgslInput.uniforms }),
    }],
    input: wgslInput,
  };
}

function createReferenceOnlyRuntimePlan(
  compiled: CompiledCudaLiteKernel,
  blockers: readonly string[],
): CudaWebGpuExecutionPlan | undefined {
  const diagnostic = compiled.diagnostics.find((item) =>
    item.code === "unsupported-dynamic-parallelism" ||
    item.code === "unsupported-cuda-runtime" ||
    item.code === "unsupported-cooperative-groups"
  );
  if (!diagnostic) return undefined;
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (runtimePlan.operations.every((operation) => operation.kind === "device-sync")) return undefined;
  const labels = [...new Set(runtimePlan.operations.map((operation) => operation.kind))].join(", ");
  const reason = labels.length > 0
    ? `CUDA runtime orchestration is reference-only (${labels}); WebGPU host orchestration is not implemented yet`
    : "CUDA runtime orchestration is reference-only; WebGPU host orchestration is not implemented yet";
  const message = blockers.length > 0
    ? `${reason}: ${blockers.join("; ")}`
    : reason;
  return {
    supported: false,
    reason: message,
    diagnostics: [{
      ...diagnostic,
      severity: "error",
      message,
    }],
  };
}

function referenceRuntimeDiagnostics(
  compiled: CompiledCudaLiteKernel,
  message: string,
): readonly CudaLiteDiagnostic[] {
  const diagnostic = compiled.diagnostics.find((item) =>
    item.code === "unsupported-dynamic-parallelism" ||
    item.code === "unsupported-cuda-runtime" ||
    item.code === "unsupported-cooperative-groups"
  );
  if (!diagnostic) return [];
  return [{
    ...diagnostic,
    severity: "error",
    message,
  }];
}

function definePeerCopyProgram(copy: CudaPeerCopyOperation): WgslKernelProgram {
  const cached = peerCopyProgramCache.get(copy.valueType);
  if (cached) return cached;
  const valueType = copy.valueType === "float" ? "f32" : copy.valueType === "int" ? "i32" : "u32";
  const program = defineWgslKernelProgram({
    name: `bg_peer_copy_${copy.valueType}`,
    workgroupSize: [64, 1, 1],
    bindings: [
      { kind: "storage", name: "bg_peer_dst", valueType, access: "read_write", binding: 0 },
      { kind: "storage", name: "bg_peer_src", valueType, access: "read", binding: 1 },
      { kind: "uniform", name: "params", byteLength: 16, binding: 2 },
    ],
    wgsl: [
      "struct Params {",
      "  dst_base: u32,",
      "  src_base: u32,",
      "  count: u32,",
      "};",
      "@group(0) @binding(0) var<storage, read_write> bg_peer_dst: array<" + valueType + ">;",
      "@group(0) @binding(1) var<storage, read> bg_peer_src: array<" + valueType + ">;",
      "@group(0) @binding(2) var<uniform> params: Params;",
      "@compute @workgroup_size(64, 1, 1)",
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
      "  let index = gid.x;",
      "  if (index < params.count) {",
      "    bg_peer_dst[params.dst_base + index] = bg_peer_src[params.src_base + index];",
      "  }",
      "}",
    ].join("\n"),
  });
  peerCopyProgramCache.set(copy.valueType, program);
  return program;
}

function packPeerCopyParams(copy: CudaPeerCopyOperation): Uint8Array {
  const bytes = new Uint8Array(16);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, copy.dstOffset, true);
  view.setUint32(4, copy.srcOffset, true);
  view.setUint32(8, copy.elementCount, true);
  return bytes;
}

function appendPeerCopySteps(
  steps: WgslKernelSequenceStep[],
  copies: readonly CudaPeerCopyOperation[],
  storageAliases: Readonly<Record<string, string>> = {},
): void {
  for (const copy of copies) {
    steps.push({
      program: definePeerCopyProgram(copy),
      launch: { dispatchCount: [Math.max(copy.elementCount, 1), 1, 1] },
      storageAliases: {
        bg_peer_dst: storageAliases[copy.dstRoot] ?? copy.dstRoot,
        bg_peer_src: storageAliases[copy.srcRoot] ?? copy.srcRoot,
      },
      uniforms: { params: packPeerCopyParams(copy) },
    });
  }
}

function getOrCompileDynamicChild(
  parent: CompiledCudaLiteKernel,
  item: {
    readonly kernel: { readonly name: string };
    readonly blockDim: readonly [number, number, number];
    readonly pointerBaseOffsets: Readonly<Record<string, number>>;
  },
  cache: Map<string, CompiledCudaLiteKernel>,
  compileKernel: NonNullable<CudaWebGpuExecutionPlanOptions["compileKernel"]>,
): CompiledCudaLiteKernel | undefined {
  const key = JSON.stringify({
    kernelName: item.kernel.name,
    blockDim: item.blockDim,
    pointerBaseOffsets: item.pointerBaseOffsets,
  });
  const cached = cache.get(key);
  if (cached) return cached;
  try {
    const compiled = compileKernel(parent.ast.source, {
      kernelName: item.kernel.name,
      features: featureOptionsFor(parent.ir.requiredFeatures),
      referenceCudaRuntime: true,
      workgroupSize: item.blockDim,
      pointerBaseOffsets: item.pointerBaseOffsets,
    });
    cache.set(key, compiled);
    return compiled;
  } catch {
    return undefined;
  }
}

function createWgslRunInput(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): WgslKernelRunInput {
  const uniforms = packCudaWebGpuUniformParams(compiled, input);
  const buffers = {
    ...input.buffers,
    ...surfaceBufferInputs(compiled, input),
    ...memoryPoolBufferInputs(compiled, input),
    ...constantBufferInputs(compiled, input),
  };
  const readback = input.readback === undefined
    ? [
      ...compiled.ir.params
        .filter((param) => (param.pointer && !param.constant) || param.valueType === "surface2d")
        .map((param) => param.valueType === "devicepool" ? poolDataName(param.name) : param.name),
      ...collectExternalDevicePoolNames(compiled.ir.body).map(poolDataName),
    ]
    : normalizeCudaWebGpuReadbackNames(compiled, input.readback);
  return {
    buffers,
    ...(input.residentBuffers === undefined ? {} : { residentBuffers: input.residentBuffers }),
    ...(input.textures === undefined ? {} : { textures: input.textures }),
    ...(uniforms.byteLength === 0 ? {} : { uniforms: { params: uniforms } }),
    readback,
  };
}

function dispatchCountForLaunch(launch: KernelLaunch): readonly [number, number, number] {
  return [
    launch.gridDim[0] * launch.blockDim[0],
    launch.gridDim[1] * launch.blockDim[1],
    launch.gridDim[2] * launch.blockDim[2],
  ];
}

function featureOptionsFor(
  requiredFeatures: readonly string[],
): Partial<Record<"shader-f16" | "subgroups" | "compatibility", boolean>> {
  return {
    ...(requiredFeatures.includes("shader-f16") ? { "shader-f16": true } : {}),
    ...(requiredFeatures.includes("subgroups") ? { subgroups: true } : {}),
  };
}

export function packCudaWebGpuUniformParams(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Uint8Array {
  const scalarParams = [
    ...compiled.ir.params.filter((param) => !param.pointer && param.valueType !== "surface2d"),
    ...compiled.ir.constants.filter((constant) => constant.dimensions.length === 0),
    ...compiled.ir.params.filter((param) => param.valueType === "surface2d").flatMap((param) => [
      { name: `${param.name}_width`, valueType: "uint" as const, surface: param.name, span: param.span },
      { name: `${param.name}_height`, valueType: "uint" as const, surface: param.name, span: param.span },
    ]),
    ...compiled.ir.params
      .filter((param) => param.pointer && compiled.pointerBaseOffsets?.[param.name] !== undefined)
      .map((param) => ({
        name: pointerBaseOffsetUniformName(param.name),
        valueType: "uint" as const,
        pointerBase: param.name,
        span: param.span,
      })),
  ];
  if (scalarParams.length === 0) return new Uint8Array(0);
  const bytes = new Uint8Array(Math.max(16, scalarParams.length * 4));
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < scalarParams.length; i++) {
    const param = scalarParams[i]!;
    const value = "surface" in param
      ? (param.name.endsWith("_width") ? input.surfaces?.[param.surface]?.width : input.surfaces?.[param.surface]?.height)
      : "pointerBase" in param
      ? compiled.pointerBaseOffsets?.[param.pointerBase]
      : "pointer" in param
      ? input.scalars?.[param.name]
      : input.constants?.[param.name];
    if (value === undefined) {
      const kind = "surface" in param ? "surface input" : "pointer" in param ? "scalar input" : "constant input";
      throw new CudaLiteCompilerError(`missing ${kind} '${param.name}'`, [{
        code: "missing-scalar",
        severity: "error",
        message: `missing ${kind} '${param.name}'`,
        span: param.span,
      }]);
    }
    if (typeof value !== "number") {
      throw new CudaLiteCompilerError(`constant '${param.name}' must be a scalar number`, [{
        code: "invalid-constant-input",
        severity: "error",
        message: `constant '${param.name}' must be a scalar number`,
        span: param.span,
      }]);
    }
    const offset = i * 4;
    if (param.valueType === "int") view.setInt32(offset, Math.trunc(value), true);
    else if (param.valueType === "uint") view.setUint32(offset, Math.trunc(value), true);
    else if (param.valueType === "half") view.setUint16(offset, float16Bits(value), true);
    else if (param.valueType === "bool") view.setUint32(offset, value ? 1 : 0, true);
    else view.setFloat32(offset, value, true);
  }
  return bytes;
}

function surfaceBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const surface of compiled.ir.params.filter((param) => param.valueType === "surface2d")) {
    const value = input.surfaces?.[surface.name];
    if (!value) {
      throw new CudaLiteCompilerError(`missing surface input '${surface.name}'`, [{
        code: "missing-surface",
        severity: "error",
        message: `missing surface input '${surface.name}'`,
        span: surface.span,
      }]);
    }
    out[surface.name] = value.data;
  }
  return out;
}

function memoryPoolBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const pool of [
    ...compiled.ir.params.filter(isDevicePoolParam).map((param) => ({ name: param.name, span: param.span })),
    ...collectExternalDevicePoolNames(compiled.ir.body).map((name) => ({ name, span: compiled.ir.body[0]?.span ?? compiled.ir.params[0]?.span ?? { start: 0, end: 0, line: 1, column: 1 } })),
  ]) {
    const value = input.memoryPools?.[pool.name];
    if (!value) {
      throw new CudaLiteCompilerError(`missing memory pool input '${pool.name}'`, [{
        code: "missing-memory-pool",
        severity: "error",
        message: `missing memory pool input '${pool.name}'`,
        span: pool.span,
      }]);
    }
    if (!(value.data instanceof Uint32Array)) {
      throw new CudaLiteCompilerError(`memory pool '${pool.name}' expects Uint32Array data`, [{
        code: "invalid-memory-pool",
        severity: "error",
        message: `memory pool '${pool.name}' expects Uint32Array data`,
        span: pool.span,
      }]);
    }
    const offset = value.offset ?? new Uint32Array([0]);
    if (!(offset instanceof Uint32Array) || offset.length < 1) {
      throw new CudaLiteCompilerError(`memory pool '${pool.name}' offset expects Uint32Array length >= 1`, [{
        code: "invalid-memory-pool",
        severity: "error",
        message: `memory pool '${pool.name}' offset expects Uint32Array length >= 1`,
        span: pool.span,
      }]);
    }
    out[poolDataName(pool.name)] = value.data;
    out[poolOffsetName(pool.name)] = offset;
  }
  return out;
}

function constantBufferInputs(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): Record<string, WgslTypedArray> {
  const out: Record<string, WgslTypedArray> = {};
  for (const constant of compiled.ir.constants.filter((item) => item.dimensions.length > 0)) {
    const value = input.constants?.[constant.name];
    if (!value || typeof value === "number") {
      throw new CudaLiteCompilerError(`missing constant buffer '${constant.name}'`, [{
        code: "missing-constant",
        severity: "error",
        message: `missing constant buffer '${constant.name}'`,
        span: constant.span,
      }]);
    }
    out[constant.name] = value;
  }
  return out;
}

function isDevicePoolParam(param: { readonly pointer: boolean; readonly valueType: string }): boolean {
  return param.pointer && param.valueType === "devicepool";
}

function float16Bits(value: number): number {
  const half = new Float16Array([value]);
  return new Uint16Array(half.buffer)[0] ?? 0;
}

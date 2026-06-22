import {
  defineWgslKernelProgram,
  runWgslKernelProgram,
  runWgslKernelProgramSequence,
  type KernelDevice,
  type WgslKernelProgram,
  type WgslKernelSequenceStep,
  type WgslTypedArray,
} from "@unlocalhosted/browsergrad-kernels";
import { collectExternalDevicePoolNames } from "./ast_queries.js";
import { analyzeCudaLite, lowerAnalyzedCudaLiteToKernelIr } from "./analyzer.js";
import { createCudaLoweringPlan } from "./compatibility.js";
import { createCudaHostDynamicLaunchPlan } from "./dynamic_launch.js";
import { parseCudaLite } from "./parser.js";
import { createCudaPeerCopyPlan, type CudaPeerCopyOperation } from "./peer_copy.js";
import { pointerBaseOffsetUniformName } from "./pointer_offsets.js";
import { poolDataName, poolOffsetName } from "./pool_bindings.js";
import { runCompiledKernelReference } from "./reference.js";
import { createCudaGridSyncPhasePlan, createCudaRuntimePlan } from "./runtime_plan.js";
import { emitKernelIrWgsl } from "./wgsl.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
  type KernelLaunch,
  type ReferenceKernelResult,
} from "./types.js";
import { formatCudaLiteDiagnostics } from "./diagnostics.js";

export function compileCudaLiteKernel(
  source: string,
  options: CompileCudaLiteOptions = {},
): CompiledCudaLiteKernel {
  const ast = parseCudaLite(source);
  const analysis = analyzeCudaLite(ast, options);
  const errors = analysis.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
  if (errors.length > 0) {
    throw new CudaLiteCompilerError(
      `CUDA-lite compile failed\n${formatCudaLiteDiagnostics(source, errors)}`,
      errors,
    );
  }
  const ir = lowerAnalyzedCudaLiteToKernelIr(analysis, options);
  const emitted = emitKernelIrWgsl(
    ir,
    {
      ...(options.features === undefined ? {} : { features: options.features }),
      ...(options.pointerBaseOffsets === undefined ? {} : { pointerBaseOffsets: options.pointerBaseOffsets }),
    },
  );
  const loweringPlan = createCudaLoweringPlan(analysis.diagnostics);
  return {
    ast,
    analysis,
    ir,
    wgsl: emitted.wgsl,
    wgslProgram: emitted.program,
    diagnostics: analysis.diagnostics,
    loweringPlan,
    ...(options.pointerBaseOffsets === undefined ? {} : { pointerBaseOffsets: options.pointerBaseOffsets }),
  };
}

export { runCompiledKernelReference };

export async function runCompiledKernelWebGpu(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): Promise<ReferenceKernelResult> {
  validateLaunch(launch, compiled.ir.workgroupSize);
  const gridSyncPhasePlan = createCudaGridSyncPhasePlan(compiled.ir);
  if (gridSyncPhasePlan.supported && gridSyncPhasePlan.modules.length > 1) {
    const wgslInput = createWgslRunInput(compiled, input);
    const dispatchCount = dispatchCountForLaunch(launch);
    const programs = gridSyncPhasePlan.modules.map((module) =>
      emitKernelIrWgsl(module, { features: featureOptionsFor(module.requiredFeatures) }).program
    );
    const result = await runWgslKernelProgramSequence(
      device,
      programs.map((program) => ({ program, launch: { dispatchCount } })),
      wgslInput,
    );
    return { buffers: normalizePoolReadback(compiled, result.buffers), trace: [] };
  }
  const dynamicResult = await tryRunHostLiftedDynamicLaunch(device, compiled, input, launch);
  if (dynamicResult) return dynamicResult;
  const peerCopyResult = await tryRunHostLiftedPeerCopy(device, compiled, input, launch);
  if (peerCopyResult) return peerCopyResult;
  rejectReferenceOnlyRuntime(compiled);
  const wgslInput = createWgslRunInput(compiled, input);
  const dispatchCount = dispatchCountForLaunch(launch);
  const result = await runWgslKernelProgram(device, compiled.wgslProgram, wgslInput, { dispatchCount });
  return { buffers: normalizePoolReadback(compiled, result.buffers), trace: [] };
}

async function tryRunHostLiftedPeerCopy(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): Promise<ReferenceKernelResult | undefined> {
  const plan = createCudaPeerCopyPlan(compiled, input, launch);
  if (!plan.supported || plan.copies.length === 0) return undefined;
  const parentInput = createWgslRunInput(compiled, input);
  const buffers: Record<string, WgslTypedArray> = { ...parentInput.buffers };
  const steps: WgslKernelSequenceStep[] = [{
    program: compiled.wgslProgram,
    launch: { dispatchCount: dispatchCountForLaunch(launch) },
    ...(parentInput.uniforms === undefined ? {} : { uniforms: parentInput.uniforms }),
  }];

  appendPeerCopySteps(steps, plan.copies);

  const result = await runWgslKernelProgramSequence(
    device,
    steps,
    {
      buffers,
      ...(parentInput.textures === undefined ? {} : { textures: parentInput.textures }),
      readback: parentInput.readback,
    },
  );
  return { buffers: normalizePoolReadback(compiled, result.buffers), trace: [] };
}

function definePeerCopyProgram(copy: CudaPeerCopyOperation): WgslKernelProgram {
  const valueType = copy.valueType === "float" ? "f32" : copy.valueType === "int" ? "i32" : "u32";
  return defineWgslKernelProgram({
    name: `bg_peer_copy_${copy.valueType}`,
    workgroupSize: [64, 1, 1],
    bindings: [
      { kind: "storage", name: "bg_peer_dst", valueType: copy.valueType === "float" ? "f32" : copy.valueType === "int" ? "i32" : "u32", access: "read_write", binding: 0 },
      { kind: "storage", name: "bg_peer_src", valueType: copy.valueType === "float" ? "f32" : copy.valueType === "int" ? "i32" : "u32", access: "read", binding: 1 },
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

async function tryRunHostLiftedDynamicLaunch(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): Promise<ReferenceKernelResult | undefined> {
  const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);
  if (!plan.supported || plan.launches.length === 0) return undefined;
  const parentInput = createWgslRunInput(compiled, input);
  const buffers: Record<string, WgslTypedArray> = { ...parentInput.buffers };
  const steps: WgslKernelSequenceStep[] = [{
    program: compiled.wgslProgram,
    launch: { dispatchCount: dispatchCountForLaunch(launch) },
    ...(parentInput.uniforms === undefined ? {} : { uniforms: parentInput.uniforms }),
  }];

  for (const item of plan.launches) {
    let childCompiled: CompiledCudaLiteKernel;
    try {
      childCompiled = compileCudaLiteKernel(compiled.ast.source, {
        kernelName: item.kernel.name,
        features: featureOptionsFor(compiled.ir.requiredFeatures),
        referenceCudaRuntime: true,
        workgroupSize: item.blockDim,
        pointerBaseOffsets: item.pointerBaseOffsets,
      });
    } catch {
      return undefined;
    }
    const childRuntime = createCudaRuntimePlan(childCompiled);
    if (!childRuntime.operations.every((operation) => operation.kind === "device-sync" || operation.kind === "peer-copy")) return undefined;
    const childWgslInput = createWgslRunInput(childCompiled, item.input);
    for (const [name, value] of Object.entries(childWgslInput.buffers)) {
      buffers[item.storageAliases[name] ?? name] = value;
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

  const result = await runWgslKernelProgramSequence(
    device,
    steps,
    {
      buffers,
      ...(parentInput.textures === undefined ? {} : { textures: parentInput.textures }),
      readback: parentInput.readback,
    },
  );
  return { buffers: normalizePoolReadback(compiled, result.buffers), trace: [] };
}

function createWgslRunInput(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
): {
  readonly buffers: Record<string, WgslTypedArray>;
  readonly textures?: NonNullable<CompiledKernelInput["textures"]>;
  readonly uniforms?: { readonly params: Uint8Array };
  readonly readback: readonly string[];
} {
  const uniforms = packScalarParams(compiled, input);
  const buffers = {
    ...input.buffers,
    ...surfaceBufferInputs(compiled, input),
    ...memoryPoolBufferInputs(compiled, input),
    ...constantBufferInputs(compiled, input),
  };
  const readback = input.readback ??
    [
      ...compiled.ir.params
        .filter((param) => (param.pointer && !param.constant) || param.valueType === "surface2d")
        .map((param) => param.valueType === "devicepool" ? poolDataName(param.name) : param.name),
      ...collectExternalDevicePoolNames(compiled.ir.body).map(poolDataName),
    ];
  return {
    buffers,
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

function rejectReferenceOnlyRuntime(compiled: CompiledCudaLiteKernel): void {
  const diagnostic = compiled.diagnostics.find((item) =>
    item.code === "unsupported-dynamic-parallelism" ||
    item.code === "unsupported-cuda-runtime" ||
    item.code === "unsupported-cooperative-groups"
  );
  if (!diagnostic) return;
  const runtimePlan = createCudaRuntimePlan(compiled);
  if (runtimePlan.operations.every((operation) => operation.kind === "device-sync")) return;
  const labels = [...new Set(runtimePlan.operations.map((operation) => operation.kind))].join(", ");
  const message = labels.length > 0
    ? `CUDA runtime orchestration is reference-only (${labels}); WebGPU host orchestration is not implemented yet`
    : "CUDA runtime orchestration is reference-only; WebGPU host orchestration is not implemented yet";
  throw new CudaLiteCompilerError(message, [{
    ...diagnostic,
    severity: "error",
    message,
  }]);
}

function packScalarParams(
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

function normalizePoolReadback(
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

function validateLaunch(launch: KernelLaunch, workgroupSize: readonly [number, number, number]): void {
  for (let axis = 0; axis < 3; axis++) {
    if (launch.blockDim[axis] !== workgroupSize[axis]) {
      throw new CudaLiteCompilerError("launch.blockDim must match compiled workgroupSize", [{
        code: "launch-workgroup-mismatch",
        severity: "error",
        message: "launch.blockDim must match compiled workgroupSize",
        span: { start: 0, end: 0, line: 1, column: 1 },
      }]);
    }
  }
}

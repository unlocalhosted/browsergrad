import {
  prepareWgslKernelProgramSequence,
  runWgslKernelProgramSequence,
  type KernelDevice,
  type WgslTypedArray,
  type WgslPreparedKernelSequence,
  type WgslPreparedKernelSequenceRunOptions,
} from "@unlocalhosted/browsergrad-kernels";
import { analyzeCudaLite } from "./analyzer.js";
import { createCudaLoweringPlan } from "./compatibility.js";
import { createCudaLiteCompileCacheKey } from "./cache-key.js";
import { validateCudaKernelLaunch } from "./launch.js";
import { createCudaHostDynamicLaunchPlan } from "./dynamic_launch.js";
import { cloneReferenceTypedArray } from "./reference_inputs.js";
import { parseCudaLite } from "./parser.js";
import {
  canRunCompiledKernelSemanticReference,
  runCompiledKernelSemanticReference,
} from "./semantic_reference.js";
import {
  createCudaLiteSemanticModel,
  lowerSemanticModelToKernelIr,
} from "./semantic_ir.js";
import { validateSemanticKernelIr } from "./semantic_ir_verifier.js";
import { typeCheckSemanticKernelIr } from "./semantic_type_check.js";
import { legalizeSemanticKernelIrForWgsl } from "./wgsl_legalization.js";
import { lowerSemanticCudaRuntime } from "./semantic_runtime_lowering.js";
import {
  canEmitSemanticKernelIrWgsl,
  emitSemanticKernelIrWgsl,
} from "./semantic_wgsl.js";
import {
  type CudaWebGpuExecutionPlan,
  createCudaWebGpuExecutionPlan,
  normalizeCudaWebGpuReadback,
  normalizeCudaWebGpuReadbackNames,
  packCudaWebGpuUniformParams,
  type CudaWebGpuExecutionPlanKind,
  type CudaWebGpuExecutionPlanOptions,
} from "./webgpu_orchestration.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CompiledKernelInput,
  type CompileCudaLiteOptions,
  type CudaLiteDiagnostic,
  type KernelLaunch,
  type ReferenceKernelResult,
} from "./types.js";
import { formatCudaLiteDiagnostics } from "./diagnostics.js";

type SupportedCudaWebGpuExecutionPlan = Extract<CudaWebGpuExecutionPlan, { readonly supported: true }>;

export interface CompiledKernelWebGpuExecutionOptions {
  readonly compileKernel?: (
    source: string,
    options?: CompileCudaLiteOptions,
  ) => CompiledCudaLiteKernel;
  readonly childCompileCacheMaxEntries?: number;
  readonly maxHostExpandedParentInvocations?: number;
  readonly maxHostDynamicLaunchDepth?: number;
}

export interface PreparedCompiledKernelWebGpuRunOptions {
  readonly readback?: readonly string[];
  readonly awaitCompletion?: boolean;
  readonly scalars?: Readonly<Record<string, number>>;
}

export interface PreparedCompiledKernelWebGpu {
  readonly kind: CudaWebGpuExecutionPlanKind;
  readonly stepCount: number;
  run(options?: PreparedCompiledKernelWebGpuRunOptions): Promise<ReferenceKernelResult>;
  destroy(): void;
}

export type PrepareCompiledKernelWebGpuOptions = CompiledKernelWebGpuExecutionOptions;

export function compileCudaLiteKernel(
  source: string,
  options: CompileCudaLiteOptions = {},
): CompiledCudaLiteKernel {
  validateTextureDescriptorOptions(options);
  const ast = parseCudaLite(source);
  const analysis = analyzeCudaLite(ast, options);
  const errors = analysis.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
  if (errors.length > 0) {
    throw new CudaLiteCompilerError(
      `CUDA-lite compile failed\n${formatCudaLiteDiagnostics(source, errors)}`,
      errors,
    );
  }
  validateBindlessTextureOptions(options, analysis);
  const semantic = createCudaLiteSemanticModel(analysis);
  const kernelIr = lowerSemanticCudaRuntime(lowerSemanticModelToKernelIr(analysis, semantic, options));
  const verifiedKernelIr = validateSemanticKernelIr(kernelIr);
  const typeCheckedKernelIr = typeCheckSemanticKernelIr(verifiedKernelIr);
  const wgslLegalizedKernelIr = legalizeSemanticKernelIrForWgsl(typeCheckedKernelIr);
  const diagnostics = reconcileSemanticRuntimeDiagnostics(analysis.diagnostics, kernelIr.operations);
  const semanticWgslOptions = {
    ...(options.f16Mode === undefined ? {} : { f16Mode: options.f16Mode }),
    ...(options.pointerBaseOffsets === undefined ? {} : { pointerBaseOffsets: options.pointerBaseOffsets }),
    ...(options.textureDescriptors === undefined ? {} : { textureDescriptors: options.textureDescriptors }),
  };
  const emitted = canEmitSemanticKernelIrWgsl(kernelIr, semanticWgslOptions)
    ? emitSemanticKernelIrWgsl(wgslLegalizedKernelIr, semanticWgslOptions)
    : undefined;
  const loweringPlan = createCudaLoweringPlan(diagnostics);
  return {
    ast,
    semantic,
    kernelIr,
    analysis,
    ...(emitted === undefined ? {} : { wgsl: emitted.wgsl, wgslProgram: emitted.program }),
    diagnostics,
    loweringPlan,
    ...(options.pointerBaseOffsets === undefined ? {} : { pointerBaseOffsets: options.pointerBaseOffsets }),
    ...(options.dynamicSharedMemory === undefined ? {} : { dynamicSharedMemory: options.dynamicSharedMemory }),
    ...(options.textureDescriptors === undefined ? {} : { textureDescriptors: options.textureDescriptors }),
    ...(options.f16Mode === undefined ? {} : { f16Mode: options.f16Mode }),
    ...(options.subgroupMode === undefined ? {} : { subgroupMode: options.subgroupMode }),
  };
}

function reconcileSemanticRuntimeDiagnostics(
  diagnostics: readonly CudaLiteDiagnostic[],
  operations: CompiledCudaLiteKernel["kernelIr"]["operations"],
): readonly CudaLiteDiagnostic[] {
  const hasDeviceLaunch = semanticOperationsContainKind(operations, "device-launch");
  return diagnostics.filter((diagnostic) =>
    diagnostic.code !== "cuda-dynamic-launch-host-orchestration" || hasDeviceLaunch,
  );
}

function semanticOperationsContainKind(
  operations: CompiledCudaLiteKernel["kernelIr"]["operations"],
  kind: CompiledCudaLiteKernel["kernelIr"]["operations"][number]["kind"],
): boolean {
  return operations.some((operation) => {
    if (operation.kind === kind) return true;
    if (operation.kind === "block" || operation.kind === "loop") return semanticOperationsContainKind(operation.body, kind);
    if (operation.kind === "branch") {
      return semanticOperationsContainKind(operation.consequent, kind) || semanticOperationsContainKind(operation.alternate, kind);
    }
    return false;
  });
}

function validateTextureDescriptorOptions(options: CompileCudaLiteOptions): void {
  for (const [name, descriptor] of Object.entries(options.textureDescriptors ?? {})) {
    if (descriptor.filterMode !== undefined && descriptor.filterMode !== "point" && descriptor.filterMode !== "linear") {
      throw new RangeError(`texture descriptor '${name}' uses unsupported filterMode '${descriptor.filterMode}'`);
    }
  }
}

function validateBindlessTextureOptions(
  options: CompileCudaLiteOptions,
  analysis: ReturnType<typeof analyzeCudaLite>,
): void {
  const names = options.bindlessTextures ?? [];
  const occupied = new Set([
    ...analysis.kernel.params.map((param) => param.name),
    ...analysis.constants.map((constant) => constant.name),
    ...analysis.deviceGlobals.map((global) => global.name),
    ...analysis.textures.map((texture) => texture.name),
  ]);
  const seen = new Set<string>();
  for (const name of names) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/u.test(name)) throw new RangeError(`bindless texture name '${name}' is not a CUDA identifier`);
    if (seen.has(name)) throw new RangeError(`duplicate bindless texture name '${name}'`);
    if (occupied.has(name)) throw new RangeError(`bindless texture name '${name}' collides with source symbol`);
    seen.add(name);
  }
}

export function cudaLiteWebGpuCompileOptions(
  options: CompileCudaLiteOptions = {},
): CompileCudaLiteOptions {
  return {
    ...options,
    referenceDynamicParallelism: true,
    referenceGridSync: true,
    referenceCudaRuntime: true,
  };
}

export function compileCudaLiteKernelForWebGpu(
  source: string,
  options: CompileCudaLiteOptions = {},
): CompiledCudaLiteKernel {
  return compileCudaLiteKernel(source, cudaLiteWebGpuCompileOptions(options));
}

export {
  canEmitSemanticKernelIrWgsl,
  canRunCompiledKernelSemanticReference,
  emitSemanticKernelIrWgsl,
  runCompiledKernelSemanticReference,
};

export function runCompiledKernelReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
): ReferenceKernelResult {
  if (semanticOperationsContainKind(compiled.kernelIr.operations, "device-launch")) {
    return runCompiledKernelDynamicReference(compiled, input, launch, 0);
  }
  return runCompiledKernelSemanticReference(compiled, input, launch);
}

function runCompiledKernelDynamicReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  depth: number,
): ReferenceKernelResult {
  if (depth >= 32) throw new CudaLiteCompilerError("semantic dynamic launch depth exceeded 32", [{
    code: "semantic-reference-unsupported",
    severity: "error",
    message: "semantic dynamic launch depth exceeded 32",
    span: compiled.kernelIr.span,
  }]);
  const working = cloneDynamicReferenceInput(input);
  const plan = createCudaHostDynamicLaunchPlan(compiled, working, launch);
  if (!plan.supported) throw new CudaLiteCompilerError(plan.reason ?? "semantic dynamic launch planning failed", [{
    code: "semantic-reference-unsupported",
    severity: "error",
    message: plan.reason ?? "semantic dynamic launch planning failed",
    span: compiled.kernelIr.span,
  }]);
  const traces: ReferenceKernelResult["trace"][number][] = [];
  for (const item of plan.launches) {
    const child = compileCudaLiteKernel(compiled.ast.source, {
      kernelName: item.kernel.name,
      referenceDynamicParallelism: true,
      referenceGridSync: true,
      referenceCudaRuntime: true,
      workgroupSize: item.blockDim,
      pointerBaseOffsets: item.pointerBaseOffsets,
      ...(compiled.f16Mode === undefined ? {} : { f16Mode: compiled.f16Mode }),
      ...(compiled.subgroupMode === undefined ? {} : { subgroupMode: compiled.subgroupMode }),
      ...(compiled.textureDescriptors === undefined ? {} : { textureDescriptors: compiled.textureDescriptors }),
    });
    const childLaunch = { gridDim: item.gridDim, blockDim: item.blockDim };
    const result = semanticOperationsContainKind(child.kernelIr.operations, "device-launch")
      ? runCompiledKernelDynamicReference(child, item.input, childLaunch, depth + 1)
      : runCompiledKernelSemanticReference(child, item.input, childLaunch);
    traces.push(...result.trace);
    copyDynamicReferenceReadback(item.input, result.buffers);
  }
  for (const [poolName, offset] of Object.entries(plan.poolOffsetUpdates ?? {})) {
    const target = working.memoryPools?.[poolName]?.offset;
    if (target) target[0] = offset >>> 0;
  }
  const readbackNames = working.readback ?? [
    ...Object.keys(working.buffers),
    ...Object.keys(working.memoryPools ?? {}),
    ...Object.keys(working.deviceGlobals ?? {}),
  ];
  return {
    buffers: Object.fromEntries(readbackNames.map((name) => {
      const value = working.buffers[name] ?? working.memoryPools?.[name]?.data ?? working.deviceGlobals?.[name];
      if (!value) throw new Error(`missing dynamic reference readback '${name}'`);
      return [name, value];
    })),
    trace: traces,
  };
}

function cloneDynamicReferenceInput(input: CompiledKernelInput): CompiledKernelInput {
  return {
    ...input,
    buffers: Object.fromEntries(Object.entries(input.buffers).map(([name, value]) => [name, cloneReferenceTypedArray(value)])),
    ...(input.deviceGlobals === undefined ? {} : {
      deviceGlobals: Object.fromEntries(Object.entries(input.deviceGlobals).map(([name, value]) => [name, cloneReferenceTypedArray(value)])),
    }),
    ...(input.memoryPools === undefined ? {} : {
      memoryPools: Object.fromEntries(Object.entries(input.memoryPools).map(([name, pool]) => [name, {
        data: new Uint32Array(pool.data),
        ...(pool.offset === undefined ? {} : { offset: new Uint32Array(pool.offset) }),
      }])),
    }),
  };
}

function copyDynamicReferenceReadback(
  input: CompiledKernelInput,
  buffers: Readonly<Record<string, WgslTypedArray>>,
): void {
  for (const [name, value] of Object.entries(buffers)) {
    const target = input.buffers[name] ?? input.deviceGlobals?.[name] ?? input.memoryPools?.[name]?.data;
    if (target) {
      const length = Math.min(target.length, value.length);
      for (let index = 0; index < length; index++) target[index] = value[index] ?? 0;
    }
  }
}

function createCachedWebGpuChildCompiler(
  options: CompiledKernelWebGpuExecutionOptions,
): NonNullable<CompiledKernelWebGpuExecutionOptions["compileKernel"]> {
  const compile = options.compileKernel ?? compileCudaLiteKernelForWebGpu;
  const maxEntries = options.childCompileCacheMaxEntries ?? 64;
  if (!Number.isInteger(maxEntries) || maxEntries < 0) {
    throw new RangeError("childCompileCacheMaxEntries must be a non-negative integer");
  }
  if (maxEntries === 0) {
    return (source, compileOptions) => compile(source, cudaLiteWebGpuCompileOptions(compileOptions));
  }
  const cache = new Map<string, CompiledCudaLiteKernel>();
  return (source, compileOptions = {}) => {
    const webGpuOptions = cudaLiteWebGpuCompileOptions(compileOptions);
    const key = createCudaLiteCompileCacheKey(source, webGpuOptions);
    const cached = cache.get(key);
    if (cached) {
      cache.delete(key);
      cache.set(key, cached);
      return cached;
    }
    const compiled = compile(source, webGpuOptions);
    cache.set(key, compiled);
    while (cache.size > maxEntries) {
      const oldest = cache.keys().next().value;
      if (oldest === undefined) break;
      cache.delete(oldest);
    }
    return compiled;
  };
}

export async function runCompiledKernelWebGpu(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: CompiledKernelWebGpuExecutionOptions = {},
): Promise<ReferenceKernelResult> {
  validateCudaKernelLaunch(launch, compiled.kernelIr.workgroupSize);
  const compileKernel = createCachedWebGpuChildCompiler(options);
  const planOptions = webGpuExecutionPlanOptions(options, compileKernel);
  const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, planOptions);
  if (!executionPlan.supported) {
    throw new CudaLiteCompilerError(executionPlan.reason, executionPlan.diagnostics);
  }
  assertCompiledKernelWebGpuDeviceFeatures(device, compiled);
  const result = await runWgslKernelProgramSequence(
    device,
    executionPlan.steps,
    executionPlan.input,
  );
  return { buffers: normalizeCudaWebGpuReadback(compiled, result.buffers), trace: [] };
}

export async function prepareCompiledKernelWebGpu(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: PrepareCompiledKernelWebGpuOptions = {},
): Promise<PreparedCompiledKernelWebGpu> {
  validateCudaKernelLaunch(launch, compiled.kernelIr.workgroupSize);
  const compileKernel = createCachedWebGpuChildCompiler(options);
  const planOptions = webGpuExecutionPlanOptions(options, compileKernel);
  const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, planOptions);
  if (!executionPlan.supported) {
    throw new CudaLiteCompilerError(executionPlan.reason, executionPlan.diagnostics);
  }
  assertCompiledKernelWebGpuDeviceFeatures(device, compiled);
  const prepared = await prepareWgslKernelProgramSequence(
    device,
    executionPlan.steps,
    executionPlan.input,
  );
  return new PreparedCompiledKernelWebGpuImpl(compiled, executionPlan.kind, input, launch, executionPlan, prepared, planOptions);
}

class PreparedCompiledKernelWebGpuImpl implements PreparedCompiledKernelWebGpu {
  readonly stepCount: number;
  private destroyed = false;

  constructor(
    private readonly compiled: CompiledCudaLiteKernel,
    readonly kind: PreparedCompiledKernelWebGpu["kind"],
    private readonly input: CompiledKernelInput,
    private readonly launch: KernelLaunch,
    private readonly executionPlan: SupportedCudaWebGpuExecutionPlan,
    private readonly prepared: WgslPreparedKernelSequence,
    private readonly planOptions: CudaWebGpuExecutionPlanOptions,
  ) {
    this.stepCount = prepared.stepCount;
  }

  async run(options?: PreparedCompiledKernelWebGpuRunOptions): Promise<ReferenceKernelResult> {
    if (this.destroyed) {
      throw new CudaLiteCompilerError("prepared compiled WebGPU kernel has been destroyed", [{
        code: "prepared-webgpu-kernel-destroyed",
        severity: "error",
        message: "prepared compiled WebGPU kernel has been destroyed",
        span: { start: 0, end: 0, line: 1, column: 1 },
      }]);
    }
    const result = await this.prepared.run(normalizePreparedRunOptions(
      this.compiled,
      this.input,
      this.launch,
      this.executionPlan,
      this.planOptions,
      options,
    ));
    return { buffers: normalizeCudaWebGpuReadback(this.compiled, result.buffers), trace: [] };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.prepared.destroy();
  }
}

function normalizePreparedRunOptions(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  initialPlan: SupportedCudaWebGpuExecutionPlan,
  planOptions: CudaWebGpuExecutionPlanOptions,
  options: PreparedCompiledKernelWebGpuRunOptions | undefined,
): WgslPreparedKernelSequenceRunOptions | undefined {
  if (!options) return undefined;
  const out: {
    readback?: readonly string[];
    awaitCompletion?: boolean;
    uniforms?: Readonly<Record<string, ArrayBuffer | ArrayBufferView>>;
    stepUniforms?: Readonly<Record<number, Readonly<Record<string, ArrayBuffer | ArrayBufferView>>>>;
  } = {};
  if (options.readback !== undefined) {
    out.readback = normalizeCudaWebGpuReadbackNames(compiled, options.readback);
  }
  if (options.awaitCompletion !== undefined) out.awaitCompletion = options.awaitCompletion;
  if (options.scalars !== undefined) {
    const nextInput = {
      ...input,
      scalars: { ...input.scalars, ...options.scalars },
    };
    if (initialPlan.kind === "single-dispatch" || initialPlan.kind === "runtime-elided-single-dispatch" || initialPlan.kind === "grid-sync-phases" || initialPlan.kind === "host-retirement-reduction") {
      const uniforms = packCudaWebGpuUniformParams(compiled, nextInput);
      if (uniforms.byteLength > 0) out.uniforms = { bg_uniforms: uniforms };
      return out;
    }
    const nextPlan = createCudaWebGpuExecutionPlan(compiled, nextInput, launch, planOptions);
    if (!nextPlan.supported) throw new CudaLiteCompilerError(nextPlan.reason, nextPlan.diagnostics);
    validatePreparedPlanTopology(initialPlan, nextPlan);
    const stepUniforms = stepUniformUpdatesForPlan(nextPlan);
    if (Object.keys(stepUniforms).length > 0) out.stepUniforms = stepUniforms;
  }
  return out;
}

function webGpuExecutionPlanOptions(
  options: CompiledKernelWebGpuExecutionOptions,
  compileKernel: NonNullable<CompiledKernelWebGpuExecutionOptions["compileKernel"]>,
): CudaWebGpuExecutionPlanOptions {
  return {
    compileKernel,
    ...(options.maxHostExpandedParentInvocations === undefined ? {} : { maxHostExpandedParentInvocations: options.maxHostExpandedParentInvocations }),
    ...(options.maxHostDynamicLaunchDepth === undefined ? {} : { maxHostDynamicLaunchDepth: options.maxHostDynamicLaunchDepth }),
  };
}

function assertCompiledKernelWebGpuDeviceFeatures(
  device: KernelDevice,
  compiled: CompiledCudaLiteKernel,
): void {
  const required = compiled.kernelIr.requiredFeatures;
  if (required.length === 0) return;
  const featureOwner = device as { readonly gpu?: { readonly features?: GPUSupportedFeatures } };
  const features = featureOwner.gpu?.features;
  const missing = required.filter((feature) => features?.has(feature as GPUFeatureName) !== true);
  if (missing.length === 0) return;
  const message = `WebGPU device missing required feature(s): ${missing.join(", ")}`;
  throw new CudaLiteCompilerError(message, [{
    code: "missing-webgpu-device-feature",
    severity: "error",
    message,
    span: compiled.kernelIr.span,
  }]);
}

function validatePreparedPlanTopology(
  initialPlan: SupportedCudaWebGpuExecutionPlan,
  nextPlan: SupportedCudaWebGpuExecutionPlan,
): void {
  if (initialPlan.kind !== nextPlan.kind) {
    throwPreparedTopologyChanged("prepared scalar update changed WebGPU execution plan kind");
  }
  if (initialPlan.steps.length !== nextPlan.steps.length) {
    throwPreparedTopologyChanged("prepared scalar update changed WebGPU step count");
  }
  for (let i = 0; i < initialPlan.steps.length; i++) {
    const initial = initialPlan.steps[i]!;
    const next = nextPlan.steps[i]!;
    if (!sameTuple(initial.launch.dispatchCount, next.launch.dispatchCount)) {
      throwPreparedTopologyChanged(`prepared scalar update changed dispatch count for step ${i}`);
    }
    if (!sameRecord(initial.storageAliases, next.storageAliases)) {
      throwPreparedTopologyChanged(`prepared scalar update changed storage aliases for step ${i}`);
    }
    if (programTopologyKey(initial.program) !== programTopologyKey(next.program)) {
      throwPreparedTopologyChanged(`prepared scalar update changed WGSL program topology for step ${i}`);
    }
  }
}

function stepUniformUpdatesForPlan(
  plan: SupportedCudaWebGpuExecutionPlan,
): Record<number, Readonly<Record<string, ArrayBuffer | ArrayBufferView>>> {
  const updates: Record<number, Readonly<Record<string, ArrayBuffer | ArrayBufferView>>> = {};
  for (let i = 0; i < plan.steps.length; i++) {
    const uniforms = plan.steps[i]?.uniforms;
    if (uniforms && Object.keys(uniforms).length > 0) updates[i] = uniforms;
  }
  return updates;
}

function programTopologyKey(program: SupportedCudaWebGpuExecutionPlan["steps"][number]["program"]): string {
  return [
    program.name,
    hashString(program.wgsl),
    program.workgroupSize.join(","),
    program.bindings.map(bindingTopologyKey).join("|"),
  ].join("::");
}

function sameTuple(left: readonly [number, number, number], right: readonly [number, number, number]): boolean {
  return left[0] === right[0] && left[1] === right[1] && left[2] === right[2];
}

function sameRecord(
  left: Readonly<Record<string, string>> | undefined,
  right: Readonly<Record<string, string>> | undefined,
): boolean {
  const leftEntries = Object.entries(left ?? {});
  const rightEntries = Object.entries(right ?? {});
  if (leftEntries.length !== rightEntries.length) return false;
  for (const [key, value] of leftEntries) {
    if (right?.[key] !== value) return false;
  }
  return true;
}

function throwPreparedTopologyChanged(message: string): never {
  throw new CudaLiteCompilerError(message, [{
    code: "prepared-scalar-update-topology-changed",
    severity: "error",
    message,
    span: { start: 0, end: 0, line: 1, column: 1 },
  }]);
}

function bindingTopologyKey(
  binding: SupportedCudaWebGpuExecutionPlan["steps"][number]["program"]["bindings"][number],
): string {
  if (binding.kind === "storage") {
    return `s:${binding.binding}:${binding.name}:${binding.valueType}:${binding.access}`;
  }
  if (binding.kind === "uniform") {
    return `u:${binding.binding}:${binding.name}:${binding.byteLength ?? ""}`;
  }
  return `t:${binding.binding}:${binding.name}:${binding.valueType}`;
}

function hashString(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

import {
  prepareWgslKernelProgramSequence,
  runWgslKernelProgramSequence,
  type KernelDevice,
  type WgslPreparedKernelSequence,
  type WgslPreparedKernelSequenceRunOptions,
} from "@unlocalhosted/browsergrad-kernels";
import { analyzeCudaLite, lowerAnalyzedCudaLiteToKernelIr } from "./analyzer.js";
import { createCudaLoweringPlan } from "./compatibility.js";
import { createCudaLiteCompileCacheKey } from "./cache-key.js";
import { validateCudaKernelLaunch } from "./launch.js";
import { parseCudaLite } from "./parser.js";
import { runCompiledKernelReference } from "./reference.js";
import { emitKernelIrWgsl } from "./wgsl.js";
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
      ...(options.f16Mode === undefined ? {} : { f16Mode: options.f16Mode }),
      ...(options.subgroupMode === undefined ? {} : { subgroupMode: options.subgroupMode }),
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
    ...(options.f16Mode === undefined ? {} : { f16Mode: options.f16Mode }),
    ...(options.subgroupMode === undefined ? {} : { subgroupMode: options.subgroupMode }),
  };
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

export { runCompiledKernelReference };

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
  validateCudaKernelLaunch(launch, compiled.ir.workgroupSize);
  const compileKernel = createCachedWebGpuChildCompiler(options);
  const planOptions = webGpuExecutionPlanOptions(options, compileKernel);
  const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, planOptions);
  if (!executionPlan.supported) {
    throw new CudaLiteCompilerError(executionPlan.reason, executionPlan.diagnostics);
  }
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
  validateCudaKernelLaunch(launch, compiled.ir.workgroupSize);
  const compileKernel = createCachedWebGpuChildCompiler(options);
  const planOptions = webGpuExecutionPlanOptions(options, compileKernel);
  const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, planOptions);
  if (!executionPlan.supported) {
    throw new CudaLiteCompilerError(executionPlan.reason, executionPlan.diagnostics);
  }
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
    if (initialPlan.kind === "single-dispatch" || initialPlan.kind === "runtime-elided-single-dispatch" || initialPlan.kind === "grid-sync-phases") {
      const uniforms = packCudaWebGpuUniformParams(compiled, nextInput);
      if (uniforms.byteLength > 0) out.uniforms = { params: uniforms };
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

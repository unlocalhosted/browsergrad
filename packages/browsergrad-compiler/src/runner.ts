import {
  defineWgslKernelProgram,
  prepareWgslKernelProgramSequence,
  runWgslKernelProgramSequence,
  type KernelDevice,
  type WgslTypedArray,
  type WgslPreparedKernelSequence,
  type WgslPreparedKernelSequenceRunOptions,
} from "@unlocalhosted/browsergrad-kernels";
import { analyzeCudaLite } from "./analyzer.js";
import { createCudaLoweringPlan } from "./compatibility.js";
import {
  createCudaLiteCompileCacheKey,
  createCudaLiteLayoutBindingCompileCacheKey,
  createCudaLiteViewCopyBindingCompileCacheKey,
} from "./cache-key.js";
import { validateCudaKernelLaunch } from "./launch.js";
import { createCudaHostDynamicLaunchPlan } from "./dynamic_launch.js";
import { cloneReferenceTypedArray } from "./reference_inputs.js";
import { parseCudaLite } from "./parser.js";
import {
  canRunCompiledKernelSemanticReference,
  runCompiledKernelSemanticReference as runCompiledKernelSemanticReferenceUnchecked,
} from "./semantic_reference.js";
import { createCudaLiteSemanticModel, lowerSemanticModelToKernelIr } from "./semantic_ir.js";
import { validateSemanticKernelIr } from "./semantic_ir_verifier.js";
import { typeCheckSemanticKernelIr } from "./semantic_type_check.js";
import { legalizeSemanticKernelIrForWgsl } from "./wgsl_legalization.js";
import { lowerSemanticCudaRuntime } from "./semantic_runtime_lowering.js";
import { lowerCudaLiteLayoutBindings } from "./semantic_layout_lowering.js";
import {
  CudaLiteLayoutBindingError,
  unwrapPreparedCudaLiteLayoutBindings,
  type PreparedCudaLiteLayoutBindings,
} from "./semantic_layout_bindings.js";
import { lowerCudaLiteViewCopyBinding } from "./semantic_view_copy_lowering.js";
import {
  CudaLiteViewCopyBindingError,
  unwrapPreparedCudaLiteViewCopyBinding,
  type PreparedCudaLiteViewCopyBinding,
} from "./semantic_view_copy_bindings.js";
import {
  canEmitSemanticKernelIrWgsl,
  emitSemanticKernelIrWgsl,
  semanticKernelIrWgslPreflightFailure,
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
  type RunCompiledKernelReferenceOptions,
} from "./types.js";
import {
  createCudaLiteCompilerError,
  withCudaLiteDiagnosticSource,
} from "./diagnostics.js";

type SupportedCudaWebGpuExecutionPlan = Extract<CudaWebGpuExecutionPlan, { readonly supported: true }>;

interface CompiledLayoutBindingAuthority {
  readonly prepared: PreparedCudaLiteLayoutBindings;
  readonly compileCacheKey: string;
}

interface CompiledViewCopyBindingAuthority {
  readonly prepared: PreparedCudaLiteViewCopyBinding;
  readonly compileCacheKey: string;
}

const COMPILED_LAYOUT_BINDING_AUTHORITIES = new WeakMap<object, CompiledLayoutBindingAuthority>();
const COMPILED_VIEW_COPY_BINDING_AUTHORITIES = new WeakMap<object, CompiledViewCopyBindingAuthority>();
const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Float32Array.prototype) as object;
const TYPED_ARRAY_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, "byteLength")?.get;
const TYPED_ARRAY_BYTE_OFFSET_GETTER = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, "byteOffset")?.get;
const TYPED_ARRAY_BUFFER_GETTER = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, "buffer")?.get;
const TYPED_ARRAY_TAG_GETTER = Object.getOwnPropertyDescriptor(TYPED_ARRAY_PROTOTYPE, Symbol.toStringTag)?.get;
const ARRAY_BUFFER_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, "byteLength")?.get;
const SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER = typeof SharedArrayBuffer === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(SharedArrayBuffer.prototype, "byteLength")?.get;

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

export interface CompiledCudaLiteLayoutBoundKernel extends CompiledCudaLiteKernel {
  readonly preparedLayoutBindings: PreparedCudaLiteLayoutBindings;
  readonly layoutBindingCompileCacheKey: string;
}

export interface CompiledCudaLiteViewCopyBoundKernel extends CompiledCudaLiteKernel {
  readonly preparedViewCopyBinding: PreparedCudaLiteViewCopyBinding;
  readonly viewCopyBindingCompileCacheKey: string;
}

export type PrepareCompiledKernelWebGpuOptions = CompiledKernelWebGpuExecutionOptions;

export function compileCudaLiteKernel(
  source: string,
  options: CompileCudaLiteOptions = {},
): CompiledCudaLiteKernel {
  try {
    return compileCudaLiteKernelUnchecked(source, options);
  } catch (error) {
    if (error instanceof CudaLiteCompilerError) {
      throw withCudaLiteDiagnosticSource(error, source);
    }
    throw error;
  }
}

export function compileCudaLiteKernelWithLayoutBindings(
  source: string,
  prepared: PreparedCudaLiteLayoutBindings,
  options: CompileCudaLiteOptions = {},
): CompiledCudaLiteLayoutBoundKernel {
  try {
    const compiled = compileCudaLiteKernelUnchecked(source, options, prepared);
    const compileCacheKey = createCudaLiteLayoutBindingCompileCacheKey(source, prepared, options);
    const result = Object.freeze({
      ...compiled,
      preparedLayoutBindings: prepared,
      layoutBindingCompileCacheKey: compileCacheKey,
    });
    COMPILED_LAYOUT_BINDING_AUTHORITIES.set(result, { prepared, compileCacheKey });
    return result;
  } catch (error) {
    if (error instanceof CudaLiteCompilerError) {
      throw withCudaLiteDiagnosticSource(error, source);
    }
    if (error instanceof CudaLiteLayoutBindingError) {
      throw createCudaLiteCompilerError(error.message, [{
        code: error.code,
        severity: "error",
        message: error.message,
        span: error.span ?? { start: 0, end: source.length, line: 1, column: 1 },
      }], source);
    }
    throw error;
  }
}

export function compileCudaLiteKernelWithViewCopyBinding(
  source: string,
  prepared: PreparedCudaLiteViewCopyBinding,
  options: CompileCudaLiteOptions = {},
): CompiledCudaLiteViewCopyBoundKernel {
  try {
    const compiled = compileCudaLiteKernelUnchecked(source, options, undefined, prepared);
    const compileCacheKey = createCudaLiteViewCopyBindingCompileCacheKey(source, prepared, options);
    const result = Object.freeze({
      ...compiled,
      preparedViewCopyBinding: prepared,
      viewCopyBindingCompileCacheKey: compileCacheKey,
    });
    COMPILED_VIEW_COPY_BINDING_AUTHORITIES.set(result, { prepared, compileCacheKey });
    return result;
  } catch (error) {
    if (error instanceof CudaLiteCompilerError) {
      throw withCudaLiteDiagnosticSource(error, source);
    }
    if (error instanceof CudaLiteViewCopyBindingError) {
      throw createCudaLiteCompilerError(error.message, [{
        code: error.code,
        severity: "error",
        message: error.message,
        span: error.span ?? { start: 0, end: source.length, line: 1, column: 1 },
      }], source);
    }
    throw error;
  }
}

function compileCudaLiteKernelUnchecked(
  source: string,
  options: CompileCudaLiteOptions,
  preparedLayoutBindings?: PreparedCudaLiteLayoutBindings,
  preparedViewCopyBinding?: PreparedCudaLiteViewCopyBinding,
): CompiledCudaLiteKernel {
  validateTextureDescriptorOptions(options);
  const ast = parseCudaLite(source);
  const analysis = analyzeCudaLite(ast, options);
  const errors = analysis.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
  if (errors.length > 0) {
    throw createCudaLiteCompilerError(
      "CUDA-lite compile failed",
      errors,
      source,
    );
  }
  validateBindlessTextureOptions(options, analysis);
  const semantic = createCudaLiteSemanticModel(analysis);
  const runtimeKernelIr = lowerSemanticCudaRuntime(lowerSemanticModelToKernelIr(analysis, semantic, options));
  const kernelIr = preparedLayoutBindings !== undefined
    ? lowerCudaLiteLayoutBindings(runtimeKernelIr, preparedLayoutBindings, options)
    : preparedViewCopyBinding !== undefined
      ? lowerCudaLiteViewCopyBinding(runtimeKernelIr, preparedViewCopyBinding, options)
      : runtimeKernelIr;
  const verifiedKernelIr = validateSemanticKernelIr(kernelIr);
  const typeCheckedKernelIr = typeCheckSemanticKernelIr(verifiedKernelIr);
  const wgslLegalizedKernelIr = legalizeSemanticKernelIrForWgsl(typeCheckedKernelIr);
  const semanticDiagnostics = reconcileSemanticRuntimeDiagnostics(analysis.diagnostics, kernelIr.operations);
  const semanticWgslOptions = {
    ...(options.f16Mode === undefined ? {} : { f16Mode: options.f16Mode }),
    ...(options.pointerBaseOffsets === undefined ? {} : { pointerBaseOffsets: options.pointerBaseOffsets }),
    ...(options.textureDescriptors === undefined ? {} : { textureDescriptors: options.textureDescriptors }),
  };
  const preflightFailure = semanticKernelIrWgslPreflightFailure(wgslLegalizedKernelIr);
  const semanticPlan = createCudaLoweringPlan(semanticDiagnostics);
  const diagnostics: readonly CudaLiteDiagnostic[] = preflightFailure !== undefined && semanticPlan.canDirectLowerToWgsl
    ? [...semanticDiagnostics, {
        code: "semantic-wgsl-unsupported",
        severity: "error",
        message: preflightFailure.message,
        span: preflightFailure.span,
      }]
    : semanticDiagnostics;
  const baseEmitted = preflightFailure === undefined
    ? emitSemanticKernelIrWgsl(wgslLegalizedKernelIr, semanticWgslOptions)
    : undefined;
  const emitted = baseEmitted === undefined
    ? undefined
    : preparedLayoutBindings !== undefined ? {
        ...baseEmitted,
        program: defineWgslKernelProgram({
          ...baseEmitted.program,
          name: layoutBoundProgramName(baseEmitted.program.name, preparedLayoutBindings),
        }),
      }
      : preparedViewCopyBinding !== undefined ? {
          ...baseEmitted,
          program: defineWgslKernelProgram({
            ...baseEmitted.program,
            name: viewCopyBoundProgramName(baseEmitted.program.name, preparedViewCopyBinding),
          }),
        }
      : baseEmitted;
  const loweringPlan = createCudaLoweringPlan(diagnostics);
  return {
    ast,
    semantic,
    kernelIr,
    verifiedKernelIr,
    typeCheckedKernelIr,
    wgslLegalizedKernelIr,
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

function layoutBoundProgramName(
  baseName: string,
  prepared: PreparedCudaLiteLayoutBindings,
): string {
  return `__bg_layout_${prepared.layoutSemanticHash}_${prepared.bindingProjectionHash}_${baseName}`;
}

function isLayoutBoundProgramName(name: string | undefined): boolean {
  return name !== undefined && /^__bg_layout_[0-9a-f]{64}_[0-9a-f]{64}_/u.test(name);
}

function viewCopyBoundProgramName(
  baseName: string,
  prepared: PreparedCudaLiteViewCopyBinding,
): string {
  return `__bg_view_copy_${prepared.layoutSemanticHash}_${prepared.kernelSemanticHash}_${prepared.specializationHash}_${prepared.bindingProjectionHash}_${baseName}`;
}

function isViewCopyBoundProgramName(name: string | undefined): boolean {
  return name !== undefined && /^__bg_view_copy_(?:[0-9a-f]{64}_){4}/u.test(name);
}

function compiledLayoutBindingAuthority(
  compiled: CompiledCudaLiteKernel,
): CompiledLayoutBindingAuthority | undefined {
  const authority = COMPILED_LAYOUT_BINDING_AUTHORITIES.get(compiled);
  if (authority !== undefined) return authority;
  const candidate = compiled as Partial<CompiledCudaLiteLayoutBoundKernel>;
  if (
    candidate.preparedLayoutBindings !== undefined ||
    candidate.layoutBindingCompileCacheKey !== undefined ||
    isLayoutBoundProgramName(compiled.wgslProgram?.name)
  ) {
    throwLayoutRuntimeError(
      compiled,
      "BG-COMPILER-LAYOUT-BINDING-UNVERIFIED-COMPILED",
      "layout-bound compiled kernel is not authorized by this compiler instance",
      compiled.kernelIr.span,
    );
  }
  return undefined;
}

function compiledViewCopyBindingAuthority(
  compiled: CompiledCudaLiteKernel,
): CompiledViewCopyBindingAuthority | undefined {
  const authority = COMPILED_VIEW_COPY_BINDING_AUTHORITIES.get(compiled);
  if (authority !== undefined) return authority;
  const candidate = compiled as Partial<CompiledCudaLiteViewCopyBoundKernel>;
  if (
    candidate.preparedViewCopyBinding !== undefined ||
    candidate.viewCopyBindingCompileCacheKey !== undefined ||
    isViewCopyBoundProgramName(compiled.wgslProgram?.name)
  ) {
    throwViewCopyRuntimeError(
      compiled,
      "BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-COMPILED",
      "view-copy-bound compiled kernel is not authorized by this compiler instance",
      compiled.kernelIr.span,
    );
  }
  return undefined;
}

function validateLayoutBoundRuntimeInput(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  backend: "cpu" | "webgpu",
): void {
  const authority = compiledLayoutBindingAuthority(compiled);
  if (authority !== undefined) {
    const record = unwrapPreparedCudaLiteLayoutBindings(authority.prepared);
    for (const binding of record.bindings) {
      const name = binding.summary.parameter;
      const typed = input.buffers[name];
      const resident = input.residentBuffers?.[name];
      if (typed !== undefined && resident !== undefined) {
        throwLayoutRuntimeError(
          compiled,
          "BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER",
          `layout-bound buffer '${name}' cannot be both typed and resident`,
          bindingSourceSpan(compiled, name),
        );
      }
      const expectedBytes = BigInt(binding.summary.allocationByteLength);
      if (resident !== undefined) {
        if (backend === "cpu") {
          throwLayoutRuntimeError(
            compiled,
            "BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER",
            `CPU reference requires a typed Float32Array for layout-bound buffer '${name}'`,
            bindingSourceSpan(compiled, name),
          );
        }
        if (
          resident.valueType !== "f32" ||
          !Number.isSafeInteger(resident.byteLength) ||
          resident.byteLength < 0 ||
          BigInt(resident.byteLength) < expectedBytes
        ) {
          throwLayoutRuntimeError(
            compiled,
            "BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER",
            `resident layout-bound buffer '${name}' must be f32 with at least ${expectedBytes} bytes`,
            bindingSourceSpan(compiled, name),
          );
        }
        continue;
      }
      const facts = typedArrayFacts(typed);
      if (facts === undefined || facts.tag !== "Float32Array" || BigInt(facts.byteLength) < expectedBytes) {
        throwLayoutRuntimeError(
          compiled,
          "BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER",
          `layout-bound buffer '${name}' must be a native Float32Array with at least ${expectedBytes} bytes`,
          bindingSourceSpan(compiled, name),
        );
      }
    }
  }
  validateViewCopyBoundRuntimeInput(compiled, input, backend);
}

function validateViewCopyBoundRuntimeInput(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  backend: "cpu" | "webgpu",
): void {
  const authority = compiledViewCopyBindingAuthority(compiled);
  if (authority === undefined) return;
  const record = unwrapPreparedCudaLiteViewCopyBinding(authority.prepared);
  const bindings = [
    {
      name: authority.prepared.sourceParameter,
      expectedBytes: record.specialization.source.allocationByteLength,
    },
    {
      name: authority.prepared.destinationParameter,
      expectedBytes: record.specialization.destination.allocationByteLength,
    },
  ] as const;
  const typedFacts = new Map<string, ReturnType<typeof typedArrayFacts>>();
  for (const binding of bindings) {
    const typed = input.buffers[binding.name];
    const resident = input.residentBuffers?.[binding.name];
    if (typed !== undefined && resident !== undefined) {
      throwViewCopyRuntimeError(
        compiled,
        "BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER",
        `view-copy buffer '${binding.name}' cannot be both typed and resident`,
        bindingSourceSpan(compiled, binding.name),
      );
    }
    if (resident !== undefined) {
      if (backend === "cpu") {
        throwViewCopyRuntimeError(
          compiled,
          "BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER",
          `CPU reference requires a typed Uint32Array for view-copy buffer '${binding.name}'`,
          bindingSourceSpan(compiled, binding.name),
        );
      }
      if (
        resident.valueType !== "u32" ||
        !Number.isSafeInteger(resident.byteLength) ||
        BigInt(resident.byteLength) !== binding.expectedBytes
      ) {
        throwViewCopyRuntimeError(
          compiled,
          "BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER",
          `resident view-copy buffer '${binding.name}' must be u32 with exactly ${binding.expectedBytes} bytes`,
          bindingSourceSpan(compiled, binding.name),
        );
      }
      continue;
    }
    const facts = typedArrayFacts(typed);
    typedFacts.set(binding.name, facts);
    if (
      facts === undefined ||
      facts.tag !== "Uint32Array" ||
      facts.byteOffset !== 0 ||
      BigInt(facts.byteLength) !== binding.expectedBytes ||
      facts.shared ||
      BigInt(facts.bufferByteLength) !== binding.expectedBytes
    ) {
      throwViewCopyRuntimeError(
        compiled,
        "BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER",
        `view-copy buffer '${binding.name}' must be a native Uint32Array with exactly ${binding.expectedBytes} bytes`,
        bindingSourceSpan(compiled, binding.name),
      );
    }
  }
  const sourceResident = input.residentBuffers?.[authority.prepared.sourceParameter];
  const destinationResident = input.residentBuffers?.[authority.prepared.destinationParameter];
  if (sourceResident !== undefined && sourceResident.buffer === destinationResident?.buffer) {
    throwViewCopyRuntimeError(
      compiled,
      "BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER",
      "view-copy source and destination resident buffers must be distinct",
      compiled.kernelIr.span,
    );
  }
  const sourceFacts = typedFacts.get(authority.prepared.sourceParameter);
  const destinationFacts = typedFacts.get(authority.prepared.destinationParameter);
  if (sourceFacts !== undefined && destinationFacts !== undefined && sourceFacts.buffer === destinationFacts.buffer) {
    throwViewCopyRuntimeError(
      compiled,
      "BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER",
      "view-copy source and destination typed buffers must have distinct roots",
      compiled.kernelIr.span,
    );
  }
}

function typedArrayFacts(value: unknown): {
  readonly tag: string;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly buffer: ArrayBufferLike;
  readonly bufferByteLength: number;
  readonly shared: boolean;
} | undefined {
  if (
    value === undefined ||
    TYPED_ARRAY_BYTE_LENGTH_GETTER === undefined ||
    TYPED_ARRAY_BYTE_OFFSET_GETTER === undefined ||
    TYPED_ARRAY_BUFFER_GETTER === undefined ||
    TYPED_ARRAY_TAG_GETTER === undefined
  ) return undefined;
  try {
    const tag = Reflect.apply(TYPED_ARRAY_TAG_GETTER, value, []) as unknown;
    const byteLength = Reflect.apply(TYPED_ARRAY_BYTE_LENGTH_GETTER, value, []) as unknown;
    const byteOffset = Reflect.apply(TYPED_ARRAY_BYTE_OFFSET_GETTER, value, []) as unknown;
    const buffer = Reflect.apply(TYPED_ARRAY_BUFFER_GETTER, value, []) as unknown;
    const bufferFacts = nativeBufferFacts(buffer);
    return typeof tag === "string" && typeof byteLength === "number" && Number.isSafeInteger(byteLength) && byteLength >= 0 &&
      typeof byteOffset === "number" && Number.isSafeInteger(byteOffset) && byteOffset >= 0 &&
      bufferFacts !== undefined
      ? { tag, byteLength, byteOffset, buffer: buffer as ArrayBufferLike, ...bufferFacts }
      : undefined;
  } catch {
    return undefined;
  }
}

function nativeBufferFacts(value: unknown): { readonly bufferByteLength: number; readonly shared: boolean } | undefined {
  if (ARRAY_BUFFER_BYTE_LENGTH_GETTER !== undefined) {
    try {
      const byteLength = Reflect.apply(ARRAY_BUFFER_BYTE_LENGTH_GETTER, value, []) as unknown;
      if (typeof byteLength === "number" && Number.isSafeInteger(byteLength) && byteLength >= 0) {
        return { bufferByteLength: byteLength, shared: false };
      }
    } catch {
      // Try SharedArrayBuffer's distinct internal-slot brand below.
    }
  }
  if (SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER !== undefined) {
    try {
      const byteLength = Reflect.apply(SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER, value, []) as unknown;
      if (typeof byteLength === "number" && Number.isSafeInteger(byteLength) && byteLength >= 0) {
        return { bufferByteLength: byteLength, shared: true };
      }
    } catch {
      return undefined;
    }
  }
  return undefined;
}

function throwViewCopyRuntimeError(
  compiled: CompiledCudaLiteKernel,
  code: "BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER" | "BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-COMPILED",
  message: string,
  span: CompiledCudaLiteKernel["kernelIr"]["span"],
): never {
  throw createCudaLiteCompilerError(`${code}: ${message}`, [{
    code,
    severity: "error",
    message,
    span,
  }], compiled.ast.source);
}

function bindingSourceSpan(compiled: CompiledCudaLiteKernel, parameter: string) {
  return compiled.kernelIr.params.find((candidate) => candidate.name === parameter)?.span ?? compiled.kernelIr.span;
}

function throwLayoutRuntimeError(
  compiled: CompiledCudaLiteKernel,
  code: "BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER" | "BG-COMPILER-LAYOUT-BINDING-UNVERIFIED-COMPILED",
  message: string,
  span: CompiledCudaLiteKernel["kernelIr"]["span"],
): never {
  throw createCudaLiteCompilerError(`${code}: ${message}`, [{
    code,
    severity: "error",
    message,
    span,
  }], compiled.ast.source);
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
};

export function runCompiledKernelSemanticReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: RunCompiledKernelReferenceOptions = {},
): ReferenceKernelResult {
  validateLayoutBoundRuntimeInput(compiled, input, "cpu");
  return runCompiledKernelSemanticReferenceUnchecked(compiled, input, launch, options);
}

export function runCompiledKernelReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  options: RunCompiledKernelReferenceOptions = {},
): ReferenceKernelResult {
  validateLayoutBoundRuntimeInput(compiled, input, "cpu");
  if (semanticOperationsContainKind(compiled.kernelIr.operations, "device-launch")) {
    return runCompiledKernelDynamicReference(compiled, input, launch, 0, options);
  }
  return runCompiledKernelSemanticReferenceUnchecked(compiled, input, launch, options);
}

function runCompiledKernelDynamicReference(
  compiled: CompiledCudaLiteKernel,
  input: CompiledKernelInput,
  launch: KernelLaunch,
  depth: number,
  options: RunCompiledKernelReferenceOptions,
): ReferenceKernelResult {
  if (depth >= 32) throw createCudaLiteCompilerError("semantic dynamic launch depth exceeded 32", [{
    code: "semantic-reference-unsupported",
    severity: "error",
    message: "semantic dynamic launch depth exceeded 32",
    span: compiled.kernelIr.span,
  }], compiled.ast.source);
  const working = cloneDynamicReferenceInput(input);
  const plan = createCudaHostDynamicLaunchPlan(compiled, working, launch);
  if (!plan.supported) throw createCudaLiteCompilerError(plan.reason ?? "semantic dynamic launch planning failed", [{
    code: "semantic-reference-unsupported",
    severity: "error",
    message: plan.reason ?? "semantic dynamic launch planning failed",
    span: compiled.kernelIr.span,
  }], compiled.ast.source);
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
      ? runCompiledKernelDynamicReference(child, item.input, childLaunch, depth + 1, options)
      : runCompiledKernelSemanticReference(child, item.input, childLaunch, options);
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
  validateLayoutBoundRuntimeInput(compiled, input, "webgpu");
  validateCudaKernelLaunch(launch, compiled.kernelIr.workgroupSize);
  const compileKernel = createCachedWebGpuChildCompiler(options);
  const planOptions = webGpuExecutionPlanOptions(options, compileKernel);
  const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, planOptions);
  if (!executionPlan.supported) {
    throw createCudaLiteCompilerError(executionPlan.reason, executionPlan.diagnostics, compiled.ast.source);
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
  validateLayoutBoundRuntimeInput(compiled, input, "webgpu");
  validateCudaKernelLaunch(launch, compiled.kernelIr.workgroupSize);
  const compileKernel = createCachedWebGpuChildCompiler(options);
  const planOptions = webGpuExecutionPlanOptions(options, compileKernel);
  const executionPlan = createCudaWebGpuExecutionPlan(compiled, input, launch, planOptions);
  if (!executionPlan.supported) {
    throw createCudaLiteCompilerError(executionPlan.reason, executionPlan.diagnostics, compiled.ast.source);
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
      throw createCudaLiteCompilerError("prepared compiled WebGPU kernel has been destroyed", [{
        code: "prepared-webgpu-kernel-destroyed",
        severity: "error",
        message: "prepared compiled WebGPU kernel has been destroyed",
        span: { start: 0, end: 0, line: 1, column: 1 },
      }], this.compiled.ast.source);
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
    if (!nextPlan.supported) throw createCudaLiteCompilerError(nextPlan.reason, nextPlan.diagnostics, compiled.ast.source);
    validatePreparedPlanTopology(initialPlan, nextPlan, compiled.ast.source);
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
  throw createCudaLiteCompilerError(message, [{
    code: "missing-webgpu-device-feature",
    severity: "error",
    message,
    span: compiled.kernelIr.span,
  }], compiled.ast.source);
}

function validatePreparedPlanTopology(
  initialPlan: SupportedCudaWebGpuExecutionPlan,
  nextPlan: SupportedCudaWebGpuExecutionPlan,
  source: string,
): void {
  if (initialPlan.kind !== nextPlan.kind) {
    throwPreparedTopologyChanged("prepared scalar update changed WebGPU execution plan kind", source);
  }
  if (initialPlan.steps.length !== nextPlan.steps.length) {
    throwPreparedTopologyChanged("prepared scalar update changed WebGPU step count", source);
  }
  for (let i = 0; i < initialPlan.steps.length; i++) {
    const initial = initialPlan.steps[i]!;
    const next = nextPlan.steps[i]!;
    if (!sameTuple(initial.launch.dispatchCount, next.launch.dispatchCount)) {
      throwPreparedTopologyChanged(`prepared scalar update changed dispatch count for step ${i}`, source);
    }
    if (!sameRecord(initial.storageAliases, next.storageAliases)) {
      throwPreparedTopologyChanged(`prepared scalar update changed storage aliases for step ${i}`, source);
    }
    if (programTopologyKey(initial.program) !== programTopologyKey(next.program)) {
      throwPreparedTopologyChanged(`prepared scalar update changed WGSL program topology for step ${i}`, source);
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

function throwPreparedTopologyChanged(message: string, source: string): never {
  throw createCudaLiteCompilerError(message, [{
    code: "prepared-scalar-update-topology-changed",
    severity: "error",
    message,
    span: { start: 0, end: 0, line: 1, column: 1 },
  }], source);
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

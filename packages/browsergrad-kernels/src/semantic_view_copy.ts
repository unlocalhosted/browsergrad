import {
  layoutArtifactPayload,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  prepareViewCopySpecialization,
  type PrepareViewCopySpecializationRequest,
  type PreparedViewCopySpecialization,
  type VerifiedKernelArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  encodeWireU64,
  hashNamedComponents,
  type JsonObject,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import type { KernelDevice } from "./types.js";
import {
  clearWgslPipelineCache,
  defineWgslKernelProgram,
  prepareWgslKernelProgramSequence,
  WgslPipelineCreationError,
  WgslShaderCreationError,
  type WgslKernelLaunch,
  type WgslKernelProgram,
  type WgslKernelRunResult,
} from "./wgsl_program.js";
import {
  emitSemanticViewCopyWgsl,
  SemanticViewCopyWgslLoweringError,
  type IntegerRange,
} from "./semantic_view_copy_wgsl.js";

export const SEMANTIC_VIEW_COPY_WEBGPU_PROFILE = "browsergrad.webgpu.view-copy.i32@1";
export const SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION = "1.0.0";
const DEFAULT_WORKGROUP_SIZE = 64;
const MAX_CONFIGURABLE_WORKGROUP_SIZE = 256;
const DEFAULT_MAX_WGSL_BYTES = 64 * 1024;
const MAX_CONFIGURABLE_WGSL_BYTES = 1024 * 1024;
const DEFAULT_MAX_TRANSIENT_WORKING_SET_BYTES = 256 * 1024 * 1024;
const MAX_CONFIGURABLE_TRANSIENT_WORKING_SET_BYTES = 1024 * 1024 * 1024;
const DEFAULT_EXECUTION_TIMEOUT_MS = 30_000;
const MAX_EXECUTION_TIMEOUT_MS = 5 * 60_000;
const PREPARED_VIEW_COPIES = new WeakSet<object>();
const LOST_VIEW_COPY_DEVICES = new WeakSet<GPUDevice>();
const WATCHED_VIEW_COPY_DEVICES = new WeakSet<object>();
const ACTIVE_VIEW_COPY_DEVICES = new WeakSet<GPUDevice>();

export interface PrepareSemanticViewCopyWgslRequest extends Omit<
  PrepareViewCopySpecializationRequest,
  "cacheSourceByteOffsets"
> {
  readonly workgroupSize?: number;
  readonly maxWgslBytes?: number;
  /** Bounds owned host snapshots/results plus GPU upload/output/readback storage for one run. */
  readonly maxTransientWorkingSetBytes?: number;
}

export interface PreparedSemanticViewCopyWgsl {
  readonly semantic: PreparedViewCopySpecialization;
  readonly backendProfile: typeof SEMANTIC_VIEW_COPY_WEBGPU_PROFILE;
  readonly backendVersion: typeof SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION;
  readonly wgslModuleHash: string;
  readonly workgroupSize: number;
  readonly plannedTransientGpuBytes: WireU64;
  readonly plannedTransientHostBytes: WireU64;
  readonly plannedTransientWorkingSetBytes: WireU64;
  readonly maxTransientWorkingSetBytes: WireU64;
  readonly program: WgslKernelProgram;
  readonly launch: WgslKernelLaunch;
  readonly sourceLocationRange: IntegerRange;
  readonly destinationLocationRange: IntegerRange;
}

export interface SemanticViewCopyWebGpuBuffers {
  readonly sourceWords: Uint32Array;
  readonly destinationWords: Uint32Array;
}

export interface SemanticViewCopyWebGpuRunOptions {
  /** Stops a queued submission or suppresses a result from work already submitted. */
  readonly signal?: AbortSignal;
  /** Caller-visible wait budget. Submitted GPU work is allowed to clean up in the background. */
  readonly timeoutMs?: number;
}

export interface SemanticViewCopyWebGpuDeviceFacts {
  readonly features: readonly string[];
  readonly limits: Readonly<{
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupsPerDimension: number;
    maxComputeInvocationsPerWorkgroup: number;
    maxComputeWorkgroupSizeX: number;
    maxBindingsPerBindGroup: number;
    maxStorageBuffersPerShaderStage: number;
  }>;
}

export interface SemanticViewCopyWebGpuTrace {
  readonly operationId: string;
  readonly semanticSpecializationHash: string;
  readonly backendSpecializationHash: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly wgslModuleHash: string;
  readonly backendProfile: typeof SEMANTIC_VIEW_COPY_WEBGPU_PROFILE;
  readonly backendVersion: typeof SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION;
  readonly logicalShape: readonly WireU64[];
  readonly elementCount: WireU64;
  readonly bytesRead: WireU64;
  readonly bytesWritten: WireU64;
  readonly plannedTransientGpuBytes: WireU64;
  readonly plannedTransientHostBytes: WireU64;
  readonly plannedTransientWorkingSetBytes: WireU64;
  readonly maxTransientWorkingSetBytes: WireU64;
  readonly submitted: boolean;
  readonly device: SemanticViewCopyWebGpuDeviceFacts;
}

export interface SemanticViewCopyWebGpuResult {
  readonly destinationWords: Uint32Array;
  readonly trace: SemanticViewCopyWebGpuTrace;
}

export type SemanticViewCopyWebGpuErrorCode =
  | "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE"
  | "BG-WEBGPU-VIEW-COPY-RESOURCE-LIMIT"
  | "BG-WEBGPU-VIEW-COPY-INVALID-BINDING"
  | "BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT"
  | "BG-WEBGPU-VIEW-COPY-SHADER"
  | "BG-WEBGPU-VIEW-COPY-PIPELINE"
  | "BG-WEBGPU-VIEW-COPY-VALIDATION"
  | "BG-WEBGPU-VIEW-COPY-OUT-OF-MEMORY"
  | "BG-WEBGPU-VIEW-COPY-INTERNAL"
  | "BG-WEBGPU-VIEW-COPY-DEVICE-LOST"
  | "BG-WEBGPU-VIEW-COPY-CANCELLED"
  | "BG-WEBGPU-VIEW-COPY-TIMEOUT"
  | "BG-WEBGPU-VIEW-COPY-EXECUTION";

export class SemanticViewCopyWebGpuError extends Error {
  constructor(
    readonly code: SemanticViewCopyWebGpuErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SemanticViewCopyWebGpuError";
  }
}

export async function prepareSemanticViewCopyWgsl(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedKernelArtifact,
  request: PrepareSemanticViewCopyWgslRequest,
): Promise<PreparedSemanticViewCopyWgsl> {
  const workgroupSize = resolvePositiveInteger(
    request.workgroupSize,
    DEFAULT_WORKGROUP_SIZE,
    MAX_CONFIGURABLE_WORKGROUP_SIZE,
    "$.workgroupSize",
  );
  const maxWgslBytes = resolvePositiveInteger(
    request.maxWgslBytes,
    DEFAULT_MAX_WGSL_BYTES,
    MAX_CONFIGURABLE_WGSL_BYTES,
    "$.maxWgslBytes",
  );
  const maxTransientWorkingSetBytes = resolvePositiveInteger(
    request.maxTransientWorkingSetBytes,
    DEFAULT_MAX_TRANSIENT_WORKING_SET_BYTES,
    MAX_CONFIGURABLE_TRANSIENT_WORKING_SET_BYTES,
    "$.maxTransientWorkingSetBytes",
  );
  const semantic = await prepareViewCopySpecialization(layoutArtifact, kernelArtifact, request);
  verifyStaticWebGpuProfile(semantic);
  const plannedTransientGpuBytes = semantic.source.allocationByteLength
    + (semantic.destination.allocationByteLength * 2n);
  const plannedTransientHostBytes = semantic.destination.allocationByteLength;
  const plannedTransientWorkingSetBytes = plannedTransientGpuBytes + plannedTransientHostBytes;
  if (plannedTransientWorkingSetBytes > BigInt(maxTransientWorkingSetBytes)) {
    fail(
      "BG-WEBGPU-VIEW-COPY-RESOURCE-LIMIT",
      "$.maxTransientWorkingSetBytes",
      `planned owned host and GPU buffers require ${plannedTransientWorkingSetBytes} bytes; limit is ${maxTransientWorkingSetBytes}`,
    );
  }
  let emitted;
  try {
    emitted = emitSemanticViewCopyWgsl(layoutArtifactPayload(layoutArtifact), semantic, workgroupSize);
  } catch (error) {
    if (error instanceof SemanticViewCopyWgslLoweringError) {
      throw new SemanticViewCopyWebGpuError(error.code, error.path, error.message, { cause: error });
    }
    throw error;
  }
  const wgslBytes = new TextEncoder().encode(emitted.source).byteLength;
  if (wgslBytes > maxWgslBytes) {
    fail("BG-WEBGPU-VIEW-COPY-RESOURCE-LIMIT", "$.maxWgslBytes", `generated WGSL requires ${wgslBytes} bytes; limit is ${maxWgslBytes}`);
  }
  const wgslModuleHash = await hashNamedComponents({
    profile: SEMANTIC_VIEW_COPY_WEBGPU_PROFILE,
    backendVersion: SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
    semanticSpecialization: semantic.specializationHash,
    workgroupSize,
    source: emitted.source,
  });
  const program = freezeProgram(defineWgslKernelProgram({
    name: `bg_semantic_view_copy_${wgslModuleHash}`,
    wgsl: emitted.source,
    bindings: [
      { kind: "storage", name: "source_words", valueType: "u32", access: "read", binding: 0 },
      { kind: "storage", name: "destination_words", valueType: "u32", access: "read_write", binding: 1 },
    ],
    workgroupSize: [workgroupSize, 1, 1],
  }));
  const prepared = Object.freeze({
    semantic,
    backendProfile: SEMANTIC_VIEW_COPY_WEBGPU_PROFILE,
    backendVersion: SEMANTIC_VIEW_COPY_WEBGPU_BACKEND_VERSION,
    wgslModuleHash,
    workgroupSize,
    plannedTransientGpuBytes: encodeWireU64(plannedTransientGpuBytes),
    plannedTransientHostBytes: encodeWireU64(plannedTransientHostBytes),
    plannedTransientWorkingSetBytes: encodeWireU64(plannedTransientWorkingSetBytes),
    maxTransientWorkingSetBytes: encodeWireU64(BigInt(maxTransientWorkingSetBytes)),
    program,
    launch: Object.freeze({ dispatchCount: Object.freeze([Number(semantic.elementCount), 1, 1] as const) }),
    sourceLocationRange: emitted.sourceLocationRange,
    destinationLocationRange: emitted.destinationLocationRange,
  });
  PREPARED_VIEW_COPIES.add(prepared);
  return prepared;
}

export async function runSemanticViewCopyWebGpu(
  device: KernelDevice,
  prepared: PreparedSemanticViewCopyWgsl,
  buffers: SemanticViewCopyWebGpuBuffers,
  options: SemanticViewCopyWebGpuRunOptions = {},
): Promise<SemanticViewCopyWebGpuResult> {
  if (!PREPARED_VIEW_COPIES.has(prepared as object)) {
    fail("BG-WEBGPU-VIEW-COPY-INVALID-BINDING", "$.prepared", "prepared plan was not produced by prepareSemanticViewCopyWgsl in this module instance");
  }
  throwIfCancelled(options.signal);
  watchDeviceLoss(device);
  if (LOST_VIEW_COPY_DEVICES.has(device.gpu)) {
    fail("BG-WEBGPU-VIEW-COPY-DEVICE-LOST", "$.device", "WebGPU device was previously lost and cannot execute this prepared plan");
  }
  const timeoutMs = resolvePositiveInteger(
    options.timeoutMs,
    DEFAULT_EXECUTION_TIMEOUT_MS,
    MAX_EXECUTION_TIMEOUT_MS,
    "$.timeoutMs",
  );
  if (ACTIVE_VIEW_COPY_DEVICES.has(device.gpu)) {
    fail(
      "BG-WEBGPU-VIEW-COPY-RESOURCE-LIMIT",
      "$.device.inFlight",
      "only one semantic view-copy may be in flight per GPUDevice; retry after the active run settles",
    );
  }
  ACTIVE_VIEW_COPY_DEVICES.add(device.gpu);
  let execution: Promise<WgslKernelRunResult> | undefined;
  try {
    if (!HOST_IS_LITTLE_ENDIAN) {
      fail("BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE", "$.environment.byteOrder", "bit-exact u32 view-copy bindings require a little-endian JavaScript host");
    }
    const sourceSlots = validateWords(
      buffers.sourceWords,
      prepared.semantic.source.allocationByteLength,
      prepared.semantic.source.allocationAlignmentBytes,
      "$.buffers.sourceWords",
    );
    const destinationSlots = validateWords(
      buffers.destinationWords,
      prepared.semantic.destination.allocationByteLength,
      prepared.semantic.destination.allocationAlignmentBytes,
      "$.buffers.destinationWords",
    );
    if (byteRangesOverlap(sourceSlots, destinationSlots)) {
      fail("BG-WEBGPU-VIEW-COPY-INVALID-BINDING", "$.buffers", "forbid-overlap operation cannot bind overlapping source and destination words");
    }
    const nativeBuffers = {
      sourceWords: nativeWords(sourceSlots),
      destinationWords: nativeWords(destinationSlots),
    };
    const deviceFacts = readAndVerifyDeviceFacts(device.gpu, prepared);

    if (prepared.semantic.elementCount === 0n) {
      const destinationWords = copyWords(nativeBuffers.destinationWords);
      const backendSpecializationHash = await hashBackendSpecialization(prepared, deviceFacts);
      throwIfCancelled(options.signal);
      return Object.freeze({
        destinationWords,
        trace: createTrace(prepared, backendSpecializationHash, deviceFacts, false),
      });
    }

    throwIfCancelled(options.signal);
    execution = runWithDeviceDiagnostics(device, prepared, nativeBuffers);
    void execution.then(
      () => ACTIVE_VIEW_COPY_DEVICES.delete(device.gpu),
      () => ACTIVE_VIEW_COPY_DEVICES.delete(device.gpu),
    );
    const [output, backendSpecializationHash] = await Promise.all([
      awaitBoundedExecution(execution, options.signal, timeoutMs),
      hashBackendSpecialization(prepared, deviceFacts),
    ]);
    const destinationWords = output.buffers.destination_words;
    if (!(destinationWords instanceof Uint32Array)) {
      fail("BG-WEBGPU-VIEW-COPY-INTERNAL", "$.result", "WGSL runner returned a non-u32 destination allocation");
    }
    return Object.freeze({
      destinationWords,
      trace: createTrace(prepared, backendSpecializationHash, deviceFacts, true),
    });
  } finally {
    if (execution === undefined) ACTIVE_VIEW_COPY_DEVICES.delete(device.gpu);
  }
}

function verifyStaticWebGpuProfile(prepared: PreparedViewCopySpecialization): void {
  for (const [role, accessor] of [["source", prepared.source], ["destination", prepared.destination]] as const) {
    if (accessor.allocationByteLength % 4n !== 0n) {
      fail("BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE", `$.${role}.allocationByteLength`, "f32 root allocation length must be a multiple of four bytes");
    }
    if (prepared.elementCount > 0n && accessor.allocationByteLength === 0n) {
      fail("BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE", `$.${role}.allocationByteLength`, "nonempty WebGPU view-copy requires nonempty root allocations");
    }
    if (accessor.allocationByteLength > 2_147_483_647n) {
      fail("BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE", `$.${role}.allocationByteLength`, "initial signed-address WGSL profile is limited to i32-sized allocations");
    }
  }
}

function readAndVerifyDeviceFacts(
  gpu: GPUDevice,
  prepared: PreparedSemanticViewCopyWgsl,
): SemanticViewCopyWebGpuDeviceFacts {
  const limits = Object.freeze({
    maxBufferSize: gpu.limits.maxBufferSize,
    maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: gpu.limits.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: gpu.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: gpu.limits.maxComputeWorkgroupSizeX,
    maxBindingsPerBindGroup: gpu.limits.maxBindingsPerBindGroup,
    maxStorageBuffersPerShaderStage: gpu.limits.maxStorageBuffersPerShaderStage,
  });
  const maximumAllocation = prepared.semantic.source.allocationByteLength > prepared.semantic.destination.allocationByteLength
    ? prepared.semantic.source.allocationByteLength
    : prepared.semantic.destination.allocationByteLength;
  if (maximumAllocation > BigInt(limits.maxBufferSize)) {
    fail("BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT", "$.device.limits.maxBufferSize", `allocation requires ${maximumAllocation} bytes; device limit is ${limits.maxBufferSize}`);
  }
  if (maximumAllocation > BigInt(limits.maxStorageBufferBindingSize)) {
    fail("BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT", "$.device.limits.maxStorageBufferBindingSize", `allocation requires ${maximumAllocation} bytes; device limit is ${limits.maxStorageBufferBindingSize}`);
  }
  if (prepared.workgroupSize > limits.maxComputeInvocationsPerWorkgroup || prepared.workgroupSize > limits.maxComputeWorkgroupSizeX) {
    fail("BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT", "$.workgroupSize", `workgroup size ${prepared.workgroupSize} exceeds device limits`);
  }
  if (limits.maxBindingsPerBindGroup < 2 || limits.maxStorageBuffersPerShaderStage < 2) {
    fail("BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT", "$.device.limits", "device cannot bind the two required storage allocations");
  }
  const workgroupCount = divideRoundUp(prepared.semantic.elementCount, BigInt(prepared.workgroupSize));
  if (workgroupCount > BigInt(limits.maxComputeWorkgroupsPerDimension)) {
    fail("BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT", "$.launch", `dispatch requires ${workgroupCount} workgroups; device limit is ${limits.maxComputeWorkgroupsPerDimension}`);
  }
  return Object.freeze({
    features: Object.freeze([...gpu.features].map(String).sort()),
    limits,
  });
}

async function runWithDeviceDiagnostics(
  device: KernelDevice,
  prepared: PreparedSemanticViewCopyWgsl,
  buffers: SemanticViewCopyWebGpuBuffers,
): Promise<WgslKernelRunResult> {
  const gpu = device.gpu;
  pushErrorScopes(gpu);
  const preparation = prepareWgslKernelProgramSequence(device, [{
    program: prepared.program,
    launch: prepared.launch,
  }], {
    buffers: {
      source_words: buffers.sourceWords,
      destination_words: buffers.destinationWords,
    },
    readback: ["destination_words"],
  }).then(
    (value) => ({ kind: "completed", value }) as const,
    (error: unknown) => ({ kind: "failed", error }) as const,
  );
  const preparationScopes = popAllErrorScopes(gpu);
  const preparationOutcome = await Promise.race([
    Promise.all([preparation, preparationScopes]).then(([outcome, scopes]) => ({ kind: "settled", outcome, scopes }) as const),
    gpu.lost.then((info) => ({ kind: "lost", info }) as const),
  ]);
  if (preparationOutcome.kind === "lost") {
    void preparation.then((outcome) => {
      if (outcome.kind === "completed") outcome.value.destroy();
    });
    fail("BG-WEBGPU-VIEW-COPY-DEVICE-LOST", "$.device", `WebGPU device lost (${preparationOutcome.info.reason}): ${preparationOutcome.info.message}`);
  }
  if (preparationOutcome.outcome.kind !== "completed") {
    classifyDiagnosticPhase(preparationOutcome.outcome, preparationOutcome.scopes, "$.pipeline");
    fail("BG-WEBGPU-VIEW-COPY-INTERNAL", "$.pipeline", "pipeline preparation failed without a classified diagnostic");
  }
  const sequence = preparationOutcome.outcome.value;
  try {
    classifyDiagnosticPhase(preparationOutcome.outcome, preparationOutcome.scopes, "$.pipeline");
  } catch (error) {
    sequence.destroy();
    throw error;
  }

  pushErrorScopes(gpu);
  const execution = sequence.run().then(
    (value) => ({ kind: "completed", value }) as const,
    (error: unknown) => ({ kind: "failed", error }) as const,
  );
  const executionScopes = popAllErrorScopes(gpu);
  try {
    const executionOutcome = await Promise.race([
      Promise.all([execution, executionScopes]).then(([outcome, scopes]) => ({ kind: "settled", outcome, scopes }) as const),
      gpu.lost.then((info) => ({ kind: "lost", info }) as const),
    ]);
    if (executionOutcome.kind === "lost") {
      fail("BG-WEBGPU-VIEW-COPY-DEVICE-LOST", "$.device", `WebGPU device lost (${executionOutcome.info.reason}): ${executionOutcome.info.message}`);
    }
    classifyDiagnosticPhase(executionOutcome.outcome, executionOutcome.scopes, "$.dispatch");
    if (executionOutcome.outcome.kind !== "completed") {
      fail("BG-WEBGPU-VIEW-COPY-INTERNAL", "$.dispatch", "dispatch failed without a classified diagnostic");
    }
    return executionOutcome.outcome.value;
  } finally {
    sequence.destroy();
  }
}

type DiagnosticPhaseOutcome<T> =
  | { readonly kind: "completed"; readonly value: T }
  | { readonly kind: "failed"; readonly error: unknown };

interface ErrorScopeResults {
  readonly validation: ErrorScopeAttempt;
  readonly outOfMemory: ErrorScopeAttempt;
  readonly internal: ErrorScopeAttempt;
}

function pushErrorScopes(gpu: GPUDevice): void {
  gpu.pushErrorScope("internal");
  gpu.pushErrorScope("out-of-memory");
  gpu.pushErrorScope("validation");
}

async function popAllErrorScopes(gpu: GPUDevice): Promise<ErrorScopeResults> {
  const validation = popErrorScopeAttempt(gpu);
  const outOfMemory = popErrorScopeAttempt(gpu);
  const internal = popErrorScopeAttempt(gpu);
  const [validationResult, outOfMemoryResult, internalResult] = await Promise.all([
    validation,
    outOfMemory,
    internal,
  ]);
  return { validation: validationResult, outOfMemory: outOfMemoryResult, internal: internalResult };
}

function classifyDiagnosticPhase<T>(
  outcome: DiagnosticPhaseOutcome<T>,
  scopes: ErrorScopeResults,
  path: string,
): void {
  const scopeFailure = scopes.validation.failure ?? scopes.outOfMemory.failure ?? scopes.internal.failure;
  if (scopeFailure !== undefined) throw scopeFailure;
  if (scopes.outOfMemory.value !== null) fail("BG-WEBGPU-VIEW-COPY-OUT-OF-MEMORY", path, scopes.outOfMemory.value.message);
  if (scopes.internal.value !== null) fail("BG-WEBGPU-VIEW-COPY-INTERNAL", path, scopes.internal.value.message);
  if (outcome.kind === "failed") {
    const message = outcome.error instanceof Error ? outcome.error.message : String(outcome.error);
    if (outcome.error instanceof WgslShaderCreationError) {
      throw new SemanticViewCopyWebGpuError("BG-WEBGPU-VIEW-COPY-SHADER", "$.shaderModule", message, { cause: outcome.error });
    }
    if (outcome.error instanceof WgslPipelineCreationError) {
      throw new SemanticViewCopyWebGpuError("BG-WEBGPU-VIEW-COPY-PIPELINE", "$.pipeline", message, { cause: outcome.error });
    }
  }
  if (scopes.validation.value !== null) fail("BG-WEBGPU-VIEW-COPY-VALIDATION", path, scopes.validation.value.message);
  if (outcome.kind === "failed") {
    const message = outcome.error instanceof Error ? outcome.error.message : String(outcome.error);
    throw new SemanticViewCopyWebGpuError("BG-WEBGPU-VIEW-COPY-EXECUTION", path, message, { cause: outcome.error });
  }
}

interface ErrorScopeAttempt {
  readonly value: GPUError | null;
  readonly failure?: unknown;
}

async function popErrorScopeAttempt(gpu: GPUDevice): Promise<ErrorScopeAttempt> {
  try {
    return { value: await popErrorScope(gpu) };
  } catch (error) {
    return { value: null, failure: error };
  }
}

async function popErrorScope(gpu: GPUDevice): Promise<GPUError | null> {
  try {
    return await gpu.popErrorScope();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    fail("BG-WEBGPU-VIEW-COPY-INTERNAL", "$.device.errorScope", message);
  }
}

function createTrace(
  prepared: PreparedSemanticViewCopyWgsl,
  backendSpecializationHash: string,
  device: SemanticViewCopyWebGpuDeviceFacts,
  submitted: boolean,
): SemanticViewCopyWebGpuTrace {
  return Object.freeze({
    operationId: prepared.semantic.operation.operationId,
    semanticSpecializationHash: prepared.semantic.specializationHash,
    backendSpecializationHash,
    layoutSemanticHash: prepared.semantic.layoutSemanticHash,
    kernelSemanticHash: prepared.semantic.kernelSemanticHash,
    wgslModuleHash: prepared.wgslModuleHash,
    backendProfile: prepared.backendProfile,
    backendVersion: prepared.backendVersion,
    logicalShape: Object.freeze(prepared.semantic.logicalShape.map((extent) => encodeWireU64(extent))),
    elementCount: encodeWireU64(prepared.semantic.elementCount),
    bytesRead: encodeWireU64(prepared.semantic.readElements * 4n),
    bytesWritten: encodeWireU64(prepared.semantic.elementCount * 4n),
    plannedTransientGpuBytes: prepared.plannedTransientGpuBytes,
    plannedTransientHostBytes: prepared.plannedTransientHostBytes,
    plannedTransientWorkingSetBytes: prepared.plannedTransientWorkingSetBytes,
    maxTransientWorkingSetBytes: prepared.maxTransientWorkingSetBytes,
    submitted,
    device,
  });
}

interface TypedArraySlots {
  readonly buffer: ArrayBufferLike;
  readonly byteLength: number;
  readonly byteOffset: number;
}

const TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(Uint32Array.prototype) as object;
const TYPED_ARRAY_BUFFER_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "buffer");
const TYPED_ARRAY_BYTE_LENGTH_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteLength");
const TYPED_ARRAY_BYTE_OFFSET_GETTER = requiredGetter(TYPED_ARRAY_PROTOTYPE, "byteOffset");
const SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER = typeof SharedArrayBuffer === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(SharedArrayBuffer.prototype, "byteLength")?.get;
const HOST_IS_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;

function validateWords(value: Uint32Array, expectedBytes: bigint, alignment: number, path: string): TypedArraySlots {
  if (!(value instanceof Uint32Array) || Object.getPrototypeOf(value) !== Uint32Array.prototype) {
    fail("BG-WEBGPU-VIEW-COPY-INVALID-BINDING", path, "WebGPU allocation bindings must be direct Uint32Array values");
  }
  let slots: TypedArraySlots;
  try {
    slots = {
      buffer: TYPED_ARRAY_BUFFER_GETTER.call(value) as ArrayBufferLike,
      byteLength: TYPED_ARRAY_BYTE_LENGTH_GETTER.call(value) as number,
      byteOffset: TYPED_ARRAY_BYTE_OFFSET_GETTER.call(value) as number,
    };
  } catch (error) {
    throw new SemanticViewCopyWebGpuError("BG-WEBGPU-VIEW-COPY-INVALID-BINDING", path, "binding does not expose native typed-array slots", { cause: error });
  }
  if (isSharedArrayBuffer(slots.buffer)) {
    fail("BG-WEBGPU-VIEW-COPY-INVALID-BINDING", path, "shared allocation bindings require an explicit synchronization contract");
  }
  if (BigInt(slots.byteLength) !== expectedBytes) {
    fail("BG-WEBGPU-VIEW-COPY-INVALID-BINDING", path, `binding length ${slots.byteLength} does not equal declared allocation length ${expectedBytes}`);
  }
  if (slots.byteOffset % alignment !== 0) {
    fail("BG-WEBGPU-VIEW-COPY-INVALID-BINDING", path, `binding offset does not satisfy ${alignment}-byte allocation alignment`);
  }
  return slots;
}

function byteRangesOverlap(left: TypedArraySlots, right: TypedArraySlots): boolean {
  if (left.buffer !== right.buffer) return false;
  const leftEnd = left.byteOffset + left.byteLength;
  const rightEnd = right.byteOffset + right.byteLength;
  return left.byteOffset < rightEnd && right.byteOffset < leftEnd;
}

function isSharedArrayBuffer(buffer: ArrayBufferLike): boolean {
  if (SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) return false;
  try {
    SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER.call(buffer);
    return true;
  } catch {
    return false;
  }
}

function requiredGetter(target: object, name: string): (this: unknown) => unknown {
  const getter = Object.getOwnPropertyDescriptor(target, name)?.get;
  if (getter === undefined) throw new Error(`internal: missing typed-array ${name} getter`);
  return getter;
}

function copyWords(source: Uint32Array): Uint32Array {
  const result = new Uint32Array(source.length);
  for (let index = 0; index < source.length; index += 1) result[index] = source[index] as number;
  return result;
}

function nativeWords(slots: TypedArraySlots): Uint32Array {
  return new Uint32Array(slots.buffer as ArrayBuffer, slots.byteOffset, slots.byteLength / Uint32Array.BYTES_PER_ELEMENT);
}

function divideRoundUp(value: bigint, divisor: bigint): bigint {
  return value === 0n ? 0n : ((value - 1n) / divisor) + 1n;
}

function freezeProgram(program: WgslKernelProgram): WgslKernelProgram {
  const bindings = Object.freeze(program.bindings.map((binding) => Object.freeze({ ...binding })));
  const workgroupSize = Object.freeze([...program.workgroupSize]) as unknown as readonly [number, number, number];
  return Object.freeze({ ...program, bindings, workgroupSize });
}

function hashBackendSpecialization(
  prepared: PreparedSemanticViewCopyWgsl,
  deviceFacts: SemanticViewCopyWebGpuDeviceFacts,
): Promise<string> {
  return hashNamedComponents({
    profile: prepared.backendProfile,
    backendVersion: prepared.backendVersion,
    semanticSpecialization: prepared.semantic.specializationHash,
    wgslModule: prepared.wgslModuleHash,
    workgroupSize: prepared.workgroupSize,
    selectedFeatures: [],
    limits: deviceFacts.limits as unknown as JsonObject,
  });
}

async function awaitBoundedExecution<T>(
  execution: Promise<T>,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<T> {
  throwIfCancelled(signal);
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let abortHandler: (() => void) | undefined;
  const timeoutPromise = new Promise<never>((_resolve, reject) => {
    timeout = setTimeout(() => reject(new SemanticViewCopyWebGpuError(
      "BG-WEBGPU-VIEW-COPY-TIMEOUT",
      "$.timeoutMs",
      `WebGPU view-copy did not complete within ${timeoutMs}ms; any submitted work will finish cleanup without returning a stale result`,
    )), timeoutMs);
  });
  const abortPromise = signal === undefined
    ? new Promise<never>(() => undefined)
    : new Promise<never>((_resolve, reject) => {
        abortHandler = () => reject(cancelledError());
        signal.addEventListener("abort", abortHandler, { once: true });
      });
  try {
    return await Promise.race([execution, timeoutPromise, abortPromise]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
    if (signal !== undefined && abortHandler !== undefined) signal.removeEventListener("abort", abortHandler);
  }
}

function throwIfCancelled(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw cancelledError();
}

function cancelledError(): SemanticViewCopyWebGpuError {
  return new SemanticViewCopyWebGpuError(
    "BG-WEBGPU-VIEW-COPY-CANCELLED",
    "$.signal",
    "WebGPU view-copy was cancelled; submitted work, if any, will not publish a stale result",
  );
}

function invalidateLostDevice(device: KernelDevice): void {
  LOST_VIEW_COPY_DEVICES.add(device.gpu);
  clearWgslPipelineCache(device);
  device.clearCache();
}

function watchDeviceLoss(device: KernelDevice): void {
  if (WATCHED_VIEW_COPY_DEVICES.has(device as object)) return;
  WATCHED_VIEW_COPY_DEVICES.add(device as object);
  void device.gpu.lost.then(
    () => invalidateLostDevice(device),
    () => invalidateLostDevice(device),
  );
}

function resolvePositiveInteger(value: number | undefined, fallback: number, maximum: number, path: string): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > maximum) {
    fail("BG-WEBGPU-VIEW-COPY-RESOURCE-LIMIT", path, `value must be a positive safe integer no greater than ${maximum}`);
  }
  return resolved;
}

function fail(code: SemanticViewCopyWebGpuErrorCode, path: string, message: string): never {
  throw new SemanticViewCopyWebGpuError(code, path, message);
}

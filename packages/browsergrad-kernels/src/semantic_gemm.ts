import type { VerifiedLayoutArtifact } from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  copyCertifiedLogicalGemmExactF32Inputs,
  logicalGemmExactF32InputCertificatePayload,
  prepareLogicalGemmTileSpecialization,
  type PrepareLogicalGemmTileSpecializationRequest,
  type PreparedLogicalGemmTileSpecialization,
  type VerifiedLogicalGemmExactF32InputCertificate,
  type VerifiedLogicalGemmTileArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  prepareLogicalGemmTileSchedule,
  type PreparedLogicalGemmTileSchedule,
  type VerifiedLogicalGemmTileScheduleArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/schedule";
import {
  encodeWireU64,
  hashNamedComponents,
  hashSemanticArtifact,
  type JsonObject,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import type { DirectDispatchProfileOptions, DirectDispatchResult } from "./runner.js";
import type { KernelDevice } from "./types.js";
import {
  defineWgslKernelProgram,
  prepareWgslKernelProgramSequence,
  WgslPipelineCreationError,
  WgslShaderCreationError,
  type WgslKernelLaunch,
  type WgslKernelProgram,
} from "./wgsl_program.js";
import {
  emitSemanticGemmWgsl,
  SemanticGemmWgslLoweringError,
} from "./semantic_gemm_wgsl.js";

export const SEMANTIC_GEMM_WEBGPU_PROFILE = "browsergrad.webgpu.gemm.exact-f32@1";
export const SEMANTIC_GEMM_WEBGPU_BACKEND_VERSION = "1.0.0";

const DEFAULT_MAX_WGSL_BYTES = 64 * 1024;
const MAX_CONFIGURABLE_WGSL_BYTES = 1024 * 1024;
const DEFAULT_MAX_WORKGROUP_INVOCATIONS = 256;
const MAX_CONFIGURABLE_WORKGROUP_INVOCATIONS = 1024;
const DEFAULT_MAX_WORKGROUP_STORAGE_BYTES = 16 * 1024;
const MAX_CONFIGURABLE_WORKGROUP_STORAGE_BYTES = 64 * 1024;
const DEFAULT_MAX_TRANSIENT_WORKING_SET_BYTES = 256 * 1024 * 1024;
const MAX_CONFIGURABLE_TRANSIENT_WORKING_SET_BYTES = 1024 * 1024 * 1024;
const DEFAULT_EXECUTION_TIMEOUT_MS = 30_000;
const MAX_EXECUTION_TIMEOUT_MS = 5 * 60_000;
const PREPARED_SEMANTIC_GEMMS = new WeakSet<object>();
const LOST_SEMANTIC_GEMM_DEVICES = new WeakSet<GPUDevice>();
const WATCHED_SEMANTIC_GEMM_DEVICES = new WeakSet<object>();
const ACTIVE_SEMANTIC_GEMM_DEVICES = new WeakSet<GPUDevice>();
const OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const REFLECT_OWN_KEYS = Reflect.ownKeys;
const REFLECT_APPLY = Reflect.apply;
const ABORT_SIGNAL_PROTOTYPE = typeof globalThis.AbortSignal === "undefined"
  ? undefined
  : globalThis.AbortSignal.prototype;
const ABORT_SIGNAL_ABORTED_GETTER = ABORT_SIGNAL_PROTOTYPE === undefined
  ? undefined
  : OBJECT_GET_OWN_PROPERTY_DESCRIPTOR(ABORT_SIGNAL_PROTOTYPE, "aborted")?.get;
const EVENT_TARGET_PROTOTYPE = typeof globalThis.EventTarget === "undefined"
  ? undefined
  : globalThis.EventTarget.prototype;
const EVENT_TARGET_ADD_EVENT_LISTENER = EVENT_TARGET_PROTOTYPE === undefined
  ? undefined
  : EVENT_TARGET_PROTOTYPE.addEventListener;
const EVENT_TARGET_REMOVE_EVENT_LISTENER = EVENT_TARGET_PROTOTYPE === undefined
  ? undefined
  : EVENT_TARGET_PROTOTYPE.removeEventListener;

export interface PrepareSemanticGemmWgslRequest
  extends PrepareLogicalGemmTileSpecializationRequest {
  readonly maxWgslBytes?: number;
  readonly maxWorkgroupInvocations?: number;
  readonly maxWorkgroupStorageBytes?: number;
  /** Bounds owned host snapshots/results and GPU upload/output/readback storage. */
  readonly maxTransientWorkingSetBytes?: number;
}

export interface PreparedSemanticGemmWgsl {
  readonly semantic: PreparedLogicalGemmTileSpecialization;
  readonly scheduled: PreparedLogicalGemmTileSchedule;
  readonly backendProfile: typeof SEMANTIC_GEMM_WEBGPU_PROFILE;
  readonly backendVersion: typeof SEMANTIC_GEMM_WEBGPU_BACKEND_VERSION;
  readonly backendPreparationHash: string;
  readonly wgslModuleHash: string;
  readonly program: WgslKernelProgram;
  readonly launch: WgslKernelLaunch;
  readonly workgroupStorageBytes: WireU64;
  readonly plannedTransientGpuBytes: WireU64;
  readonly plannedTransientHostBytes: WireU64;
  readonly plannedTransientWorkingSetBytes: WireU64;
  readonly maxTransientWorkingSetBytes: WireU64;
}

export interface SemanticGemmResidentInput {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
}

export interface SemanticGemmResidentInputs {
  readonly lhs: SemanticGemmResidentInput;
  readonly rhs: SemanticGemmResidentInput;
}

export interface SemanticGemmResidentRunOptions {
  readonly profile?: DirectDispatchProfileOptions;
}

export interface SemanticGemmWebGpuRunOptions {
  /** Stops a queued submission or suppresses a result from submitted work. */
  readonly signal?: AbortSignal;
  /** Caller-visible wait budget; submitted work cleans up in the background. */
  readonly timeoutMs?: number;
}

export interface SemanticGemmWebGpuTrace {
  readonly operationId: string;
  readonly semanticSpecializationHash: string;
  readonly scheduleSpecializationHash: string;
  readonly backendPreparationHash: string;
  readonly backendSpecializationHash: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly scheduleSemanticHash: string;
  readonly inputCertificateSemanticHash: string;
  readonly wgslModuleHash: string;
  readonly backendProfile: typeof SEMANTIC_GEMM_WEBGPU_PROFILE;
  readonly backendVersion: typeof SEMANTIC_GEMM_WEBGPU_BACKEND_VERSION;
  readonly executionTier: "portable-webgpu-core";
  /** No CUDA/CuTe invocation schedule or native MMA facility is claimed. */
  readonly preservationLevel: "portable-relegalized";
  readonly numericalInputProfile: string;
  readonly numericalPreservation: "bit-exact-on-certified-inputs";
  readonly m: WireU64;
  readonly n: WireU64;
  readonly k: WireU64;
  readonly physicalTile: Readonly<{ readonly m: WireU64; readonly n: WireU64; readonly k: WireU64 }>;
  readonly dispatchWorkgroups: Readonly<{ readonly x: WireU64; readonly y: WireU64; readonly z: WireU64 }>;
  readonly multiplyAdds: WireU64;
  readonly logicalBytesRead: WireU64;
  readonly logicalBytesWritten: WireU64;
  readonly plannedTransientGpuBytes: WireU64;
  readonly plannedTransientHostBytes: WireU64;
  readonly plannedTransientWorkingSetBytes: WireU64;
  readonly maxTransientWorkingSetBytes: WireU64;
  readonly submitted: true;
  readonly device: SemanticGemmWebGpuDeviceFacts;
}

export interface SemanticGemmWebGpuResult {
  readonly destination: Uint8Array;
  readonly trace: SemanticGemmWebGpuTrace;
}

export type SemanticGemmWebGpuErrorCode =
  | "BG-WEBGPU-GEMM-UNSUPPORTED-PROFILE"
  | "BG-WEBGPU-GEMM-NUMERICAL-PROOF"
  | "BG-WEBGPU-GEMM-RESOURCE-LIMIT"
  | "BG-WEBGPU-GEMM-INVALID-BINDING"
  | "BG-WEBGPU-GEMM-DEVICE-LIMIT"
  | "BG-WEBGPU-GEMM-SHADER"
  | "BG-WEBGPU-GEMM-PIPELINE"
  | "BG-WEBGPU-GEMM-VALIDATION"
  | "BG-WEBGPU-GEMM-OUT-OF-MEMORY"
  | "BG-WEBGPU-GEMM-INTERNAL"
  | "BG-WEBGPU-GEMM-DEVICE-LOST"
  | "BG-WEBGPU-GEMM-CANCELLED"
  | "BG-WEBGPU-GEMM-TIMEOUT"
  | "BG-WEBGPU-GEMM-EXECUTION";

export class SemanticGemmWebGpuError extends Error {
  constructor(
    readonly code: SemanticGemmWebGpuErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SemanticGemmWebGpuError";
  }
}

/**
 * Composes verified logical GEMM and physical schedule artifacts, then lowers
 * only backend geometry. Numerical preservation is intentionally authorized
 * later by a concrete-input certificate at execution time.
 */
export async function prepareSemanticGemmWgsl(
  layoutArtifact: VerifiedLayoutArtifact,
  logicalGemmArtifact: VerifiedLogicalGemmTileArtifact,
  scheduleArtifact: VerifiedLogicalGemmTileScheduleArtifact,
  request: PrepareSemanticGemmWgslRequest,
): Promise<PreparedSemanticGemmWgsl> {
  const capturedRequest = capturePrepareRequest(request);
  const maxWgslBytes = positiveInteger(
    capturedRequest.maxWgslBytes,
    DEFAULT_MAX_WGSL_BYTES,
    MAX_CONFIGURABLE_WGSL_BYTES,
    "$.maxWgslBytes",
  );
  const maxWorkgroupInvocations = positiveInteger(
    capturedRequest.maxWorkgroupInvocations,
    DEFAULT_MAX_WORKGROUP_INVOCATIONS,
    MAX_CONFIGURABLE_WORKGROUP_INVOCATIONS,
    "$.maxWorkgroupInvocations",
  );
  const maxWorkgroupStorageBytes = positiveInteger(
    capturedRequest.maxWorkgroupStorageBytes,
    DEFAULT_MAX_WORKGROUP_STORAGE_BYTES,
    MAX_CONFIGURABLE_WORKGROUP_STORAGE_BYTES,
    "$.maxWorkgroupStorageBytes",
  );
  const maxTransientWorkingSetBytes = positiveInteger(
    capturedRequest.maxTransientWorkingSetBytes,
    DEFAULT_MAX_TRANSIENT_WORKING_SET_BYTES,
    MAX_CONFIGURABLE_TRANSIENT_WORKING_SET_BYTES,
    "$.maxTransientWorkingSetBytes",
  );

  const semantic = await prepareLogicalGemmTileSpecialization(
    layoutArtifact,
    logicalGemmArtifact,
    logicalRequest(capturedRequest),
  );
  requireNonemptyWebGpuProfile(semantic);
  const scheduled = await prepareLogicalGemmTileSchedule(
    semantic,
    logicalGemmArtifact,
    scheduleArtifact,
    {
      ...(capturedRequest.evaluationLimits === undefined
        ? {}
        : { evaluationLimits: capturedRequest.evaluationLimits }),
      maxWorkgroupInvocations,
      maxStagingElements: Math.floor(maxWorkgroupStorageBytes / 4),
    },
  );

  let emitted;
  try {
    emitted = emitSemanticGemmWgsl(semantic, scheduled);
  } catch (error) {
    if (error instanceof SemanticGemmWgslLoweringError) {
      throw new SemanticGemmWebGpuError(error.code, error.path, error.message, {
        cause: error,
      });
    }
    throw error;
  }
  if (emitted.workgroupStorageBytes > BigInt(maxWorkgroupStorageBytes)) {
    fail(
      "BG-WEBGPU-GEMM-RESOURCE-LIMIT",
      "$.maxWorkgroupStorageBytes",
      `generated workgroup staging requires ${emitted.workgroupStorageBytes} bytes; limit is ${maxWorkgroupStorageBytes}`,
    );
  }
  const wgslBytes = new TextEncoder().encode(emitted.source).byteLength;
  if (wgslBytes > maxWgslBytes) {
    fail(
      "BG-WEBGPU-GEMM-RESOURCE-LIMIT",
      "$.maxWgslBytes",
      `generated WGSL requires ${wgslBytes} bytes; limit is ${maxWgslBytes}`,
    );
  }

  const lhsBytes = semantic.lhs.allocationByteLength;
  const rhsBytes = semantic.rhs.allocationByteLength;
  const destinationBytes = semantic.destination.allocationByteLength;
  const plannedTransientGpuBytes = lhsBytes + rhsBytes + (destinationBytes * 2n);
  // Host peak includes certified lhs/rhs copies, the initialized destination
  // upload, the runner readback, and the returned byte-exact result copy.
  const plannedTransientHostBytes = lhsBytes + rhsBytes + (destinationBytes * 3n);
  const plannedTransientWorkingSetBytes = plannedTransientGpuBytes + plannedTransientHostBytes;
  if (plannedTransientWorkingSetBytes > BigInt(maxTransientWorkingSetBytes)) {
    fail(
      "BG-WEBGPU-GEMM-RESOURCE-LIMIT",
      "$.maxTransientWorkingSetBytes",
      `planned owned host and GPU buffers require ${plannedTransientWorkingSetBytes} bytes; limit is ${maxTransientWorkingSetBytes}`,
    );
  }

  const backendPreparationHash = await hashNamedComponents({
    profile: SEMANTIC_GEMM_WEBGPU_PROFILE,
    backendVersion: SEMANTIC_GEMM_WEBGPU_BACKEND_VERSION,
    semanticSpecialization: semantic.specializationHash,
    scheduleSpecialization: scheduled.scheduleSpecializationHash,
    workgroupStorageBytes: encodeWireU64(emitted.workgroupStorageBytes),
  });
  const wgslModuleHash = await hashNamedComponents({
    backendPreparation: backendPreparationHash,
    source: emitted.source,
  });
  const program = freezeProgram(defineWgslKernelProgram({
    name: `bg_semantic_gemm_${wgslModuleHash}`,
    wgsl: emitted.source,
    bindings: [
      { kind: "storage", name: "lhs_values", valueType: "f32", access: "read", binding: 0 },
      { kind: "storage", name: "rhs_values", valueType: "f32", access: "read", binding: 1 },
      { kind: "storage", name: "destination_values", valueType: "f32", access: "read_write", binding: 2 },
    ],
    workgroupSize: emitted.workgroupSize,
  }));
  const launch = Object.freeze({
    dispatchCount: Object.freeze([
      safeNumber(scheduled.dispatchX * scheduled.physicalN, "$.launch.x"),
      safeNumber(scheduled.dispatchY * scheduled.physicalM, "$.launch.y"),
      1,
    ] as const),
  });
  const prepared = Object.freeze({
    semantic,
    scheduled,
    backendProfile: SEMANTIC_GEMM_WEBGPU_PROFILE,
    backendVersion: SEMANTIC_GEMM_WEBGPU_BACKEND_VERSION,
    backendPreparationHash,
    wgslModuleHash,
    program,
    launch,
    workgroupStorageBytes: encodeWireU64(emitted.workgroupStorageBytes),
    plannedTransientGpuBytes: encodeWireU64(plannedTransientGpuBytes),
    plannedTransientHostBytes: encodeWireU64(plannedTransientHostBytes),
    plannedTransientWorkingSetBytes: encodeWireU64(plannedTransientWorkingSetBytes),
    maxTransientWorkingSetBytes: encodeWireU64(BigInt(maxTransientWorkingSetBytes)),
  });
  PREPARED_SEMANTIC_GEMMS.add(prepared);
  return prepared;
}

/**
 * A concrete host-byte certificate cannot authorize arbitrary resident GPU
 * buffers: their contents are not observable at this boundary and may have
 * changed after host validation. Keep the resident seam explicit and closed
 * until a trusted upload/provenance handle exists.
 */
export async function runPreparedSemanticGemmResident(
  _device: KernelDevice,
  prepared: PreparedSemanticGemmWgsl,
  _inputs: SemanticGemmResidentInputs,
  _options: SemanticGemmResidentRunOptions = {},
): Promise<DirectDispatchResult> {
  requirePrepared(prepared);
  fail(
    "BG-WEBGPU-GEMM-NUMERICAL-PROOF",
    "$.inputs",
    "resident GEMM requires trusted input provenance; a host-byte certificate cannot prove current GPUBuffer contents",
  );
}

/**
 * Uploads only fresh authority-retained copies of the certified bytes, runs the
 * scheduled GEMM, and reads the complete dense destination allocation back.
 * Caller-owned mutable input storage is never uploaded by this function.
 */
export async function runSemanticGemmWebGpu(
  device: KernelDevice,
  prepared: PreparedSemanticGemmWgsl,
  certificate: VerifiedLogicalGemmExactF32InputCertificate,
  options: SemanticGemmWebGpuRunOptions = {},
): Promise<SemanticGemmWebGpuResult> {
  requirePrepared(prepared);
  const capturedOptions = captureRunOptions(options);
  throwIfCancelled(capturedOptions.signal);
  const timeoutMs = positiveInteger(
    capturedOptions.timeoutMs,
    DEFAULT_EXECUTION_TIMEOUT_MS,
    MAX_EXECUTION_TIMEOUT_MS,
    "$.timeoutMs",
  );
  const certificatePayload = authorizeCertificate(prepared, certificate);
  const certifiedInputs = copyCertifiedLogicalGemmExactF32Inputs(certificate);
  validateCertifiedInputLength(
    certifiedInputs.lhs,
    prepared.semantic.lhs.allocationByteLength,
    "$.certificate.inputs.lhs",
  );
  validateCertifiedInputLength(
    certifiedInputs.rhs,
    prepared.semantic.rhs.allocationByteLength,
    "$.certificate.inputs.rhs",
  );
  throwIfCancelled(capturedOptions.signal);
  requireAvailableDevice(device);
  const deviceFacts = readAndVerifyDeviceFacts(device.gpu, prepared);
  throwIfCancelled(capturedOptions.signal);

  ACTIVE_SEMANTIC_GEMM_DEVICES.add(device.gpu);
  const execution = executeHostReadback(device, prepared, certifiedInputs);
  void execution.then(
    () => ACTIVE_SEMANTIC_GEMM_DEVICES.delete(device.gpu),
    () => ACTIVE_SEMANTIC_GEMM_DEVICES.delete(device.gpu),
  );
  try {
    const certificateSemanticHashPromise = hashSemanticArtifact(certificate);
    const backendSpecializationHashPromise = hashNamedComponents({
      backendPreparation: prepared.backendPreparationHash,
      deviceFeatures: [...deviceFacts.features],
      deviceLimits: deviceFacts.limits as unknown as JsonObject,
    });
    const [output, inputCertificateSemanticHash, backendSpecializationHash] = await Promise.all([
      awaitBoundedExecution(execution, device.gpu, capturedOptions.signal, timeoutMs),
      certificateSemanticHashPromise,
      backendSpecializationHashPromise,
    ]);
    throwIfCancelled(capturedOptions.signal);
    return Object.freeze({
      destination: output,
      trace: createTrace(
        prepared,
        certificatePayload.proof.profile,
        inputCertificateSemanticHash,
        backendSpecializationHash,
        deviceFacts,
      ),
    });
  } catch (error) {
    throw classifyExecutionError(error);
  }
}

function authorizeCertificate(
  prepared: PreparedSemanticGemmWgsl,
  certificate: VerifiedLogicalGemmExactF32InputCertificate,
) {
  const payload = logicalGemmExactF32InputCertificatePayload(certificate);
  if (payload.logicalGemmSemanticHash !== prepared.semantic.kernelSemanticHash) {
    fail(
      "BG-WEBGPU-GEMM-NUMERICAL-PROOF",
      "$.certificate.logicalGemmSemanticHash",
      "exact-input certificate belongs to a different logical GEMM artifact",
    );
  }
  if (payload.specializationHash !== prepared.semantic.specializationHash) {
    fail(
      "BG-WEBGPU-GEMM-NUMERICAL-PROOF",
      "$.certificate.specializationHash",
      "exact-input certificate belongs to a different logical GEMM specialization",
    );
  }
  if (payload.proof.guarantees.wgslF32Output !== "bit-exact-on-certified-inputs") {
    fail(
      "BG-WEBGPU-GEMM-NUMERICAL-PROOF",
      "$.certificate.proof.guarantees.wgslF32Output",
      "certificate does not authorize bit-exact WGSL f32 output",
    );
  }
  return payload;
}

function validateCertifiedInputLength(
  input: Uint8Array,
  expectedBytes: bigint,
  path: string,
): void {
  if (
    !(input instanceof Uint8Array)
    || Object.getPrototypeOf(input) !== Uint8Array.prototype
    || !(input.buffer instanceof ArrayBuffer)
    || input.byteOffset !== 0
    || BigInt(input.byteLength) !== expectedBytes
    || input.byteLength % 4 !== 0
  ) {
    fail(
      "BG-WEBGPU-GEMM-NUMERICAL-PROOF",
      path,
      "certificate authority did not return an exact unshared f32 allocation snapshot",
    );
  }
}

async function executeHostReadback(
  device: KernelDevice,
  prepared: PreparedSemanticGemmWgsl,
  inputs: Readonly<{ readonly lhs: Uint8Array; readonly rhs: Uint8Array }>,
): Promise<Uint8Array> {
  const destinationBytes = safeNumber(
    prepared.semantic.destination.allocationByteLength,
    "$.semantic.destination.allocationByteLength",
  );
  const result = await runWithDeviceDiagnostics(device, prepared, {
    buffers: {
      lhs_values: new Float32Array(inputs.lhs.buffer, 0, inputs.lhs.byteLength / 4),
      rhs_values: new Float32Array(inputs.rhs.buffer, 0, inputs.rhs.byteLength / 4),
      destination_values: new Float32Array(destinationBytes / 4),
    },
    readback: ["destination_values"],
  });
  const destination = result.buffers.destination_values;
  if (
    !(destination instanceof Float32Array)
    || Object.getPrototypeOf(destination) !== Float32Array.prototype
    || destination.byteLength !== destinationBytes
  ) {
    fail(
      "BG-WEBGPU-GEMM-INTERNAL",
      "$.result.destination",
      "WGSL runner returned an invalid f32 destination allocation",
    );
  }
  return new Uint8Array(
    destination.buffer.slice(
      destination.byteOffset,
      destination.byteOffset + destination.byteLength,
    ),
  );
}

async function runWithDeviceDiagnostics(
  device: KernelDevice,
  prepared: PreparedSemanticGemmWgsl,
  input: Parameters<typeof prepareWgslKernelProgramSequence>[2],
) {
  const gpu = device.gpu;
  pushErrorScopes(gpu);
  const preparation = prepareWgslKernelProgramSequence(device, [{
    program: prepared.program,
    launch: prepared.launch,
  }], input).then(
    (value) => ({ kind: "completed", value }) as const,
    (error: unknown) => ({ kind: "failed", error }) as const,
  );
  const preparationScopes = popAllErrorScopes(gpu);
  const preparationOutcome = await Promise.race([
    Promise.all([preparation, preparationScopes]).then(
      ([outcome, scopes]) => ({ kind: "settled", outcome, scopes }) as const,
    ),
    gpu.lost.then((info) => ({ kind: "lost", info }) as const),
  ]);
  if (preparationOutcome.kind === "lost") {
    void preparation.then((outcome) => {
      if (outcome.kind === "completed") outcome.value.destroy();
    });
    fail(
      "BG-WEBGPU-GEMM-DEVICE-LOST",
      "$.device",
      `WebGPU device lost (${preparationOutcome.info.reason}): ${preparationOutcome.info.message}`,
    );
  }
  if (preparationOutcome.outcome.kind !== "completed") {
    classifyDiagnosticPhase(
      preparationOutcome.outcome,
      preparationOutcome.scopes,
      "$.pipeline",
    );
    fail(
      "BG-WEBGPU-GEMM-INTERNAL",
      "$.pipeline",
      "pipeline preparation failed without a classified diagnostic",
    );
  }
  const sequence = preparationOutcome.outcome.value;
  try {
    classifyDiagnosticPhase(
      preparationOutcome.outcome,
      preparationOutcome.scopes,
      "$.pipeline",
    );
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
      Promise.all([execution, executionScopes]).then(
        ([outcome, scopes]) => ({ kind: "settled", outcome, scopes }) as const,
      ),
      gpu.lost.then((info) => ({ kind: "lost", info }) as const),
    ]);
    if (executionOutcome.kind === "lost") {
      fail(
        "BG-WEBGPU-GEMM-DEVICE-LOST",
        "$.device",
        `WebGPU device lost (${executionOutcome.info.reason}): ${executionOutcome.info.message}`,
      );
    }
    classifyDiagnosticPhase(
      executionOutcome.outcome,
      executionOutcome.scopes,
      "$.dispatch",
    );
    if (executionOutcome.outcome.kind !== "completed") {
      fail(
        "BG-WEBGPU-GEMM-INTERNAL",
        "$.dispatch",
        "dispatch failed without a classified diagnostic",
      );
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

interface ErrorScopeAttempt {
  readonly value: GPUError | null;
  readonly failure?: unknown;
}

function pushErrorScopes(gpu: GPUDevice): void {
  gpu.pushErrorScope("internal");
  gpu.pushErrorScope("out-of-memory");
  gpu.pushErrorScope("validation");
}

async function popAllErrorScopes(gpu: GPUDevice): Promise<ErrorScopeResults> {
  const [validation, outOfMemory, internal] = await Promise.all([
    popErrorScopeAttempt(gpu),
    popErrorScopeAttempt(gpu),
    popErrorScopeAttempt(gpu),
  ]);
  return { validation, outOfMemory, internal };
}

async function popErrorScopeAttempt(gpu: GPUDevice): Promise<ErrorScopeAttempt> {
  try {
    return { value: await gpu.popErrorScope() };
  } catch (failure) {
    return { value: null, failure };
  }
}

function classifyDiagnosticPhase<T>(
  outcome: DiagnosticPhaseOutcome<T>,
  scopes: ErrorScopeResults,
  path: string,
): void {
  const scopeFailure = scopes.validation.failure
    ?? scopes.outOfMemory.failure
    ?? scopes.internal.failure;
  if (scopeFailure !== undefined) {
    throw new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-INTERNAL",
      "$.device.errorScope",
      scopeFailure instanceof Error ? scopeFailure.message : String(scopeFailure),
      { cause: scopeFailure },
    );
  }
  if (scopes.outOfMemory.value !== null) {
    fail("BG-WEBGPU-GEMM-OUT-OF-MEMORY", path, scopes.outOfMemory.value.message);
  }
  if (scopes.internal.value !== null) {
    fail("BG-WEBGPU-GEMM-INTERNAL", path, scopes.internal.value.message);
  }
  if (outcome.kind === "failed") {
    if (outcome.error instanceof WgslShaderCreationError) {
      throw new SemanticGemmWebGpuError(
        "BG-WEBGPU-GEMM-SHADER",
        "$.shaderModule",
        outcome.error.message,
        { cause: outcome.error },
      );
    }
    if (outcome.error instanceof WgslPipelineCreationError) {
      throw new SemanticGemmWebGpuError(
        "BG-WEBGPU-GEMM-PIPELINE",
        "$.pipeline",
        outcome.error.message,
        { cause: outcome.error },
      );
    }
  }
  if (scopes.validation.value !== null) {
    fail("BG-WEBGPU-GEMM-VALIDATION", path, scopes.validation.value.message);
  }
  if (outcome.kind === "failed") {
    throw new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-EXECUTION",
      path,
      outcome.error instanceof Error ? outcome.error.message : String(outcome.error),
      { cause: outcome.error },
    );
  }
}

function createTrace(
  prepared: PreparedSemanticGemmWgsl,
  numericalInputProfile: string,
  inputCertificateSemanticHash: string,
  backendSpecializationHash: string,
  device: SemanticGemmWebGpuDeviceFacts,
): SemanticGemmWebGpuTrace {
  return Object.freeze({
    operationId: prepared.semantic.operation.operationId,
    semanticSpecializationHash: prepared.semantic.specializationHash,
    scheduleSpecializationHash: prepared.scheduled.scheduleSpecializationHash,
    backendPreparationHash: prepared.backendPreparationHash,
    backendSpecializationHash,
    layoutSemanticHash: prepared.semantic.layoutSemanticHash,
    kernelSemanticHash: prepared.semantic.kernelSemanticHash,
    scheduleSemanticHash: prepared.scheduled.scheduleSemanticHash,
    inputCertificateSemanticHash,
    wgslModuleHash: prepared.wgslModuleHash,
    backendProfile: prepared.backendProfile,
    backendVersion: prepared.backendVersion,
    executionTier: "portable-webgpu-core",
    preservationLevel: "portable-relegalized",
    numericalInputProfile,
    numericalPreservation: "bit-exact-on-certified-inputs",
    m: encodeWireU64(prepared.semantic.m),
    n: encodeWireU64(prepared.semantic.n),
    k: encodeWireU64(prepared.semantic.k),
    physicalTile: Object.freeze({
      m: encodeWireU64(prepared.scheduled.physicalM),
      n: encodeWireU64(prepared.scheduled.physicalN),
      k: encodeWireU64(prepared.scheduled.physicalK),
    }),
    dispatchWorkgroups: Object.freeze({
      x: encodeWireU64(prepared.scheduled.dispatchX),
      y: encodeWireU64(prepared.scheduled.dispatchY),
      z: encodeWireU64(prepared.scheduled.dispatchZ),
    }),
    multiplyAdds: encodeWireU64(prepared.semantic.multiplyAdds),
    logicalBytesRead: encodeWireU64(prepared.semantic.multiplyAdds * 8n),
    logicalBytesWritten: encodeWireU64(prepared.semantic.outputElements * 4n),
    plannedTransientGpuBytes: prepared.plannedTransientGpuBytes,
    plannedTransientHostBytes: prepared.plannedTransientHostBytes,
    plannedTransientWorkingSetBytes: prepared.plannedTransientWorkingSetBytes,
    maxTransientWorkingSetBytes: prepared.maxTransientWorkingSetBytes,
    submitted: true,
    device,
  });
}

async function awaitBoundedExecution<T>(
  execution: Promise<T>,
  gpu: GPUDevice,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let removeAbortListener: () => void = () => undefined;
  const timeoutPromise = new Promise<never>((_resolve, reject) => {
    timeout = setTimeout(() => reject(new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-TIMEOUT",
      "$.timeoutMs",
      `semantic GEMM exceeded caller wait budget of ${timeoutMs} ms`,
    )), timeoutMs);
  });
  const cancellationPromise = signal === undefined
    ? new Promise<never>(() => undefined)
    : new Promise<never>((_resolve, reject) => {
        const onAbort = () => reject(cancelledError());
        if (EVENT_TARGET_ADD_EVENT_LISTENER === undefined
          || EVENT_TARGET_REMOVE_EVENT_LISTENER === undefined) {
          reject(new SemanticGemmWebGpuError(
            "BG-WEBGPU-GEMM-INTERNAL",
            "$.signal",
            "native EventTarget methods are unavailable",
          ));
          return;
        }
        REFLECT_APPLY(EVENT_TARGET_ADD_EVENT_LISTENER, signal, ["abort", onAbort, { once: true }]);
        removeAbortListener = () => {
          REFLECT_APPLY(EVENT_TARGET_REMOVE_EVENT_LISTENER, signal, ["abort", onAbort]);
        };
        if (abortSignalAborted(signal)) onAbort();
      });
  const deviceLostPromise = gpu.lost.then((info) => {
    throw new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-DEVICE-LOST",
      "$.device",
      `WebGPU device lost (${info.reason}): ${info.message}`,
    );
  });
  try {
    return await Promise.race([
      execution,
      timeoutPromise,
      cancellationPromise,
      deviceLostPromise,
    ]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
    removeAbortListener();
  }
}

function classifyExecutionError(error: unknown): Error {
  if (error instanceof SemanticGemmWebGpuError) return error;
  if (error instanceof WgslShaderCreationError) {
    return new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-SHADER",
      "$.shaderModule",
      error.message,
      { cause: error },
    );
  }
  if (error instanceof WgslPipelineCreationError) {
    return new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-PIPELINE",
      "$.pipeline",
      error.message,
      { cause: error },
    );
  }
  return new SemanticGemmWebGpuError(
    "BG-WEBGPU-GEMM-EXECUTION",
    "$.dispatch",
    error instanceof Error ? error.message : String(error),
    { cause: error },
  );
}

function throwIfCancelled(signal: AbortSignal | undefined): void {
  if (signal !== undefined && abortSignalAborted(signal)) throw cancelledError();
}

function cancelledError(): SemanticGemmWebGpuError {
  return new SemanticGemmWebGpuError(
    "BG-WEBGPU-GEMM-CANCELLED",
    "$.signal",
    "semantic GEMM was cancelled",
  );
}

const PREPARE_REQUEST_FIELDS = Object.freeze([
  "operationId",
  "bindings",
  "evaluationLimits",
  "maxElements",
  "maxMultiplyAdds",
  "maxEvaluationSteps",
  "maxPreparationMs",
  "signal",
  "maxWgslBytes",
  "maxWorkgroupInvocations",
  "maxWorkgroupStorageBytes",
  "maxTransientWorkingSetBytes",
] as const);

function capturePrepareRequest(
  value: PrepareSemanticGemmWgslRequest,
): PrepareSemanticGemmWgslRequest {
  const captured = captureClosedDataRecord(
    value,
    PREPARE_REQUEST_FIELDS,
    ["operationId"],
    "$.request",
  ) as unknown as PrepareSemanticGemmWgslRequest;
  validateOptionalAbortSignal(captured.signal, "$.request.signal");
  return captured;
}

function captureRunOptions(
  value: SemanticGemmWebGpuRunOptions,
): SemanticGemmWebGpuRunOptions {
  const captured = captureClosedDataRecord(
    value,
    ["signal", "timeoutMs"],
    [],
    "$.options",
  ) as unknown as SemanticGemmWebGpuRunOptions;
  validateOptionalAbortSignal(captured.signal, "$.options.signal");
  return captured;
}

function captureClosedDataRecord(
  value: unknown,
  allowedFields: readonly string[],
  requiredFields: readonly string[],
  path: string,
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail("BG-WEBGPU-GEMM-INVALID-BINDING", path, "expected a plain options object");
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = OBJECT_GET_PROTOTYPE_OF(value);
    descriptors = OBJECT_GET_OWN_PROPERTY_DESCRIPTORS(value);
  } catch (error) {
    throw new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-INVALID-BINDING",
      path,
      "options object could not be captured without invoking properties",
      { cause: error },
    );
  }
  if (prototype !== Object.prototype && prototype !== null) {
    fail("BG-WEBGPU-GEMM-INVALID-BINDING", path, "expected a plain options object");
  }
  const allowed = new Set(allowedFields);
  const result = Object.create(null) as Record<string, unknown>;
  for (const key of REFLECT_OWN_KEYS(descriptors)) {
    if (typeof key !== "string" || !allowed.has(key)) {
      fail(
        "BG-WEBGPU-GEMM-INVALID-BINDING",
        path,
        `unknown closed-record field ${typeof key === "string" ? key : "symbol"}`,
      );
    }
    const descriptor = descriptors[key];
    if (
      descriptor === undefined
      || descriptor.enumerable !== true
      || !("value" in descriptor)
    ) {
      fail(
        "BG-WEBGPU-GEMM-INVALID-BINDING",
        `${path}.${key}`,
        "options must use enumerable own data properties without accessors",
      );
    }
    result[key] = descriptor.value;
  }
  for (const field of requiredFields) {
    if (!(field in result)) {
      fail(
        "BG-WEBGPU-GEMM-INVALID-BINDING",
        `${path}.${field}`,
        "required field is missing",
      );
    }
  }
  return Object.freeze(result);
}

function validateOptionalAbortSignal(value: AbortSignal | undefined, path: string): void {
  if (value === undefined) return;
  if (
    ABORT_SIGNAL_PROTOTYPE === undefined
    || ABORT_SIGNAL_ABORTED_GETTER === undefined
    || OBJECT_GET_PROTOTYPE_OF(value) !== ABORT_SIGNAL_PROTOTYPE
  ) {
    fail("BG-WEBGPU-GEMM-INVALID-BINDING", path, "signal must be a native AbortSignal");
  }
  try {
    if (typeof REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, value, []) !== "boolean") {
      throw new Error("missing native AbortSignal slots");
    }
  } catch (error) {
    throw new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-INVALID-BINDING",
      path,
      "signal does not expose native AbortSignal state",
      { cause: error },
    );
  }
}

function abortSignalAborted(signal: AbortSignal): boolean {
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined) {
    fail("BG-WEBGPU-GEMM-INTERNAL", "$.signal", "native AbortSignal getter is unavailable");
  }
  try {
    return REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []) as boolean;
  } catch (error) {
    throw new SemanticGemmWebGpuError(
      "BG-WEBGPU-GEMM-INVALID-BINDING",
      "$.signal",
      "signal does not expose native AbortSignal state",
      { cause: error },
    );
  }
}

function logicalRequest(
  request: PrepareSemanticGemmWgslRequest,
): PrepareLogicalGemmTileSpecializationRequest {
  return {
    operationId: request.operationId,
    ...(request.bindings === undefined ? {} : { bindings: request.bindings }),
    ...(request.evaluationLimits === undefined
      ? {}
      : { evaluationLimits: request.evaluationLimits }),
    ...(request.maxElements === undefined ? {} : { maxElements: request.maxElements }),
    ...(request.maxMultiplyAdds === undefined
      ? {}
      : { maxMultiplyAdds: request.maxMultiplyAdds }),
    ...(request.maxEvaluationSteps === undefined
      ? {}
      : { maxEvaluationSteps: request.maxEvaluationSteps }),
    ...(request.maxPreparationMs === undefined
      ? {}
      : { maxPreparationMs: request.maxPreparationMs }),
    ...(request.signal === undefined ? {} : { signal: request.signal }),
  };
}

function requireNonemptyWebGpuProfile(
  semantic: PreparedLogicalGemmTileSpecialization,
): void {
  for (const [name, value] of [
    ["m", semantic.m],
    ["n", semantic.n],
    ["k", semantic.k],
  ] as const) {
    if (value === 0n) {
      fail(
        "BG-WEBGPU-GEMM-UNSUPPORTED-PROFILE",
        `$.semantic.${name}`,
        "initial WebGPU GEMM profile requires nonempty M, N, and K dimensions",
      );
    }
  }
  const denseDestinationBytes = semantic.outputElements * 4n;
  if (
    semantic.destination.viewByteOffset !== 0n
    || semantic.destination.allocationByteLength !== denseDestinationBytes
  ) {
    fail(
      "BG-WEBGPU-GEMM-UNSUPPORTED-PROFILE",
      "$.semantic.destination",
      "host/readback GEMM requires a zero-offset dense destination that overwrites the complete root allocation",
    );
  }
}

function requirePrepared(prepared: PreparedSemanticGemmWgsl): void {
  if (!PREPARED_SEMANTIC_GEMMS.has(prepared as object)) {
    fail(
      "BG-WEBGPU-GEMM-INVALID-BINDING",
      "$.prepared",
      "prepared plan was not produced by prepareSemanticGemmWgsl in this module instance",
    );
  }
}

function positiveInteger(
  value: number | undefined,
  fallback: number,
  maximum: number,
  path: string,
): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > maximum) {
    fail(
      "BG-WEBGPU-GEMM-RESOURCE-LIMIT",
      path,
      `value must be a positive safe integer no greater than ${maximum}`,
    );
  }
  return resolved;
}

function safeNumber(value: bigint, path: string): number {
  if (value <= 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    fail(
      "BG-WEBGPU-GEMM-UNSUPPORTED-PROFILE",
      path,
      "value is outside exact JavaScript dispatch range",
    );
  }
  return Number(value);
}

function freezeProgram(program: WgslKernelProgram): WgslKernelProgram {
  return Object.freeze({
    ...program,
    bindings: Object.freeze(program.bindings.map((binding) => Object.freeze({ ...binding }))),
    workgroupSize: Object.freeze([...program.workgroupSize] as [number, number, number]),
  });
}

export interface SemanticGemmWebGpuDeviceFacts {
  readonly features: readonly string[];
  readonly limits: Readonly<{
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupsPerDimension: number;
    maxComputeInvocationsPerWorkgroup: number;
    maxComputeWorkgroupSizeX: number;
    maxComputeWorkgroupSizeY: number;
    maxComputeWorkgroupStorageSize: number;
    maxBindingsPerBindGroup: number;
    maxStorageBuffersPerShaderStage: number;
  }>;
}

function readAndVerifyDeviceFacts(
  gpu: GPUDevice,
  prepared: PreparedSemanticGemmWgsl,
): SemanticGemmWebGpuDeviceFacts {
  const limits = Object.freeze({
    maxBufferSize: gpu.limits.maxBufferSize,
    maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: gpu.limits.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: gpu.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: gpu.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: gpu.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupStorageSize: gpu.limits.maxComputeWorkgroupStorageSize,
    maxBindingsPerBindGroup: gpu.limits.maxBindingsPerBindGroup,
    maxStorageBuffersPerShaderStage: gpu.limits.maxStorageBuffersPerShaderStage,
  });
  const allocations = [
    prepared.semantic.lhs.allocationByteLength,
    prepared.semantic.rhs.allocationByteLength,
    prepared.semantic.destination.allocationByteLength,
  ];
  const maximumAllocation = allocations.reduce(
    (maximum, value) => value > maximum ? value : maximum,
    0n,
  );
  if (maximumAllocation > BigInt(limits.maxBufferSize)) {
    fail(
      "BG-WEBGPU-GEMM-DEVICE-LIMIT",
      "$.device.limits.maxBufferSize",
      `allocation requires ${maximumAllocation} bytes; device limit is ${limits.maxBufferSize}`,
    );
  }
  if (maximumAllocation > BigInt(limits.maxStorageBufferBindingSize)) {
    fail(
      "BG-WEBGPU-GEMM-DEVICE-LIMIT",
      "$.device.limits.maxStorageBufferBindingSize",
      `allocation requires ${maximumAllocation} bytes; device limit is ${limits.maxStorageBufferBindingSize}`,
    );
  }
  const [workgroupX, workgroupY] = prepared.program.workgroupSize;
  const invocations = workgroupX * workgroupY;
  if (
    invocations > limits.maxComputeInvocationsPerWorkgroup
    || workgroupX > limits.maxComputeWorkgroupSizeX
    || workgroupY > limits.maxComputeWorkgroupSizeY
  ) {
    fail(
      "BG-WEBGPU-GEMM-DEVICE-LIMIT",
      "$.device.limits.workgroupSize",
      `workgroup ${workgroupX}x${workgroupY} exceeds device limits`,
    );
  }
  if (BigInt(prepared.workgroupStorageBytes) > BigInt(limits.maxComputeWorkgroupStorageSize)) {
    fail(
      "BG-WEBGPU-GEMM-DEVICE-LIMIT",
      "$.device.limits.maxComputeWorkgroupStorageSize",
      `staging requires ${prepared.workgroupStorageBytes} bytes; device limit is ${limits.maxComputeWorkgroupStorageSize}`,
    );
  }
  if (limits.maxBindingsPerBindGroup < 3 || limits.maxStorageBuffersPerShaderStage < 3) {
    fail(
      "BG-WEBGPU-GEMM-DEVICE-LIMIT",
      "$.device.limits.bindings",
      "device cannot bind the three required storage allocations",
    );
  }
  if (
    prepared.scheduled.dispatchX > BigInt(limits.maxComputeWorkgroupsPerDimension)
    || prepared.scheduled.dispatchY > BigInt(limits.maxComputeWorkgroupsPerDimension)
  ) {
    fail(
      "BG-WEBGPU-GEMM-DEVICE-LIMIT",
      "$.launch",
      `dispatch requires ${prepared.scheduled.dispatchX}x${prepared.scheduled.dispatchY} workgroups; per-dimension limit is ${limits.maxComputeWorkgroupsPerDimension}`,
    );
  }
  return Object.freeze({
    features: Object.freeze([...gpu.features].map(String).sort()),
    limits,
  });
}

function watchDeviceLoss(device: KernelDevice): void {
  if (WATCHED_SEMANTIC_GEMM_DEVICES.has(device.gpu)) return;
  WATCHED_SEMANTIC_GEMM_DEVICES.add(device.gpu);
  void device.gpu.lost.then(() => {
    LOST_SEMANTIC_GEMM_DEVICES.add(device.gpu);
    device.clearCache();
  });
}

function requireAvailableDevice(device: KernelDevice): void {
  watchDeviceLoss(device);
  if (LOST_SEMANTIC_GEMM_DEVICES.has(device.gpu)) {
    fail(
      "BG-WEBGPU-GEMM-DEVICE-LOST",
      "$.device",
      "WebGPU device was previously lost and cannot execute this prepared plan",
    );
  }
  if (ACTIVE_SEMANTIC_GEMM_DEVICES.has(device.gpu)) {
    fail(
      "BG-WEBGPU-GEMM-RESOURCE-LIMIT",
      "$.device.inFlight",
      "only one semantic GEMM may be in flight per GPUDevice",
    );
  }
}

function fail(
  code: SemanticGemmWebGpuErrorCode,
  path: string,
  message: string,
): never {
  throw new SemanticGemmWebGpuError(code, path, message);
}

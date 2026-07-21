import type { VerifiedLayoutArtifact } from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  prepareAttentionForwardSpecialization,
  type PrepareAttentionForwardSpecializationRequest,
  type PreparedAttentionForwardSpecialization,
  type VerifiedAttentionForwardArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  prepareAttentionOnlineKvTileSchedule,
  type PreparedAttentionOnlineKvTileSchedule,
  type VerifiedAttentionOnlineKvTileScheduleArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/schedule";
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
  type WgslKernelLaunch,
  type WgslKernelProgram,
} from "./wgsl_program.js";
import {
  emitSemanticAttentionWgsl,
  SemanticAttentionWgslLoweringError,
} from "./semantic_attention_wgsl.js";
import { captureFiniteF32Snapshots } from "./native_f32_snapshots.js";
import {
  runPreparedSemanticWebGpuHostReadback,
  SemanticWebGpuHostError,
} from "./semantic_webgpu_host.js";

export const SEMANTIC_ATTENTION_WEBGPU_PROFILE =
  "browsergrad.webgpu.attention.block-tiled-online-softmax-f32@1";
export const SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION = "1.0.0";

const DEFAULT_MAX_WGSL_BYTES = 64 * 1024;
const MAX_CONFIGURABLE_WGSL_BYTES = 1024 * 1024;
const DEFAULT_MAX_WORKGROUP_INVOCATIONS = 256;
const MAX_CONFIGURABLE_WORKGROUP_INVOCATIONS = 1024;
const DEFAULT_MAX_WORKGROUP_STORAGE_BYTES = 16 * 1024;
const MAX_CONFIGURABLE_WORKGROUP_STORAGE_BYTES = 64 * 1024;
const DEFAULT_MAX_PRIVATE_ELEMENTS_PER_INVOCATION = 512;
const MAX_CONFIGURABLE_PRIVATE_ELEMENTS_PER_INVOCATION = 16_384;
const DEFAULT_MAX_KEY_TILES = 1_048_576;
const MAX_CONFIGURABLE_KEY_TILES = 16_777_216;
const DEFAULT_MAX_DISPATCH_WORKGROUPS = 16_777_216;
const MAX_CONFIGURABLE_DISPATCH_WORKGROUPS = 1_073_741_824;
const DEFAULT_MAX_TRANSIENT_WORKING_SET_BYTES = 256 * 1024 * 1024;
const MAX_CONFIGURABLE_TRANSIENT_WORKING_SET_BYTES = 1024 * 1024 * 1024;
const DEFAULT_MAX_INPUT_VALIDATION_MS = 30_000;
const MAX_INPUT_VALIDATION_MS = 5 * 60_000;
const DEFAULT_EXECUTION_TIMEOUT_MS = 30_000;
const MAX_EXECUTION_TIMEOUT_MS = 5 * 60_000;
const PREPARED_SEMANTIC_ATTENTIONS = new WeakSet<object>();
const LOST_SEMANTIC_ATTENTION_DEVICES = new WeakSet<GPUDevice>();
const WATCHED_SEMANTIC_ATTENTION_DEVICES = new WeakSet<object>();
const ACTIVE_SEMANTIC_ATTENTION_DEVICES = new WeakSet<GPUDevice>();
const OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const REFLECT_OWN_KEYS = Reflect.ownKeys;
const REFLECT_APPLY = Reflect.apply;
const NUMBER_IS_SAFE_INTEGER = Number.isSafeInteger;
const TEXT_ENCODER = new TextEncoder();
const TEXT_ENCODER_ENCODE = TextEncoder.prototype.encode;
const ABORT_SIGNAL_PROTOTYPE = typeof globalThis.AbortSignal === "undefined"
  ? undefined
  : globalThis.AbortSignal.prototype;
const ABORT_SIGNAL_ABORTED_GETTER = ABORT_SIGNAL_PROTOTYPE === undefined
  ? undefined
  : Object.getOwnPropertyDescriptor(ABORT_SIGNAL_PROTOTYPE, "aborted")?.get;
const EVENT_TARGET_PROTOTYPE = typeof globalThis.EventTarget === "undefined"
  ? undefined
  : globalThis.EventTarget.prototype;
const EVENT_TARGET_ADD_EVENT_LISTENER = EVENT_TARGET_PROTOTYPE === undefined
  ? undefined
  : EVENT_TARGET_PROTOTYPE.addEventListener;
const EVENT_TARGET_REMOVE_EVENT_LISTENER = EVENT_TARGET_PROTOTYPE === undefined
  ? undefined
  : EVENT_TARGET_PROTOTYPE.removeEventListener;
const FLOAT32_ARRAY_PROTOTYPE = Float32Array.prototype;
const FLOAT32_ARRAY_CONSTRUCTOR = Float32Array;
const UINT8_ARRAY_CONSTRUCTOR = Uint8Array;
const TYPED_ARRAY_PROTOTYPE = OBJECT_GET_PROTOTYPE_OF(FLOAT32_ARRAY_PROTOTYPE) as object;
const TYPED_ARRAY_BUFFER_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  "buffer",
)?.get;
const TYPED_ARRAY_BYTE_OFFSET_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  "byteOffset",
)?.get;
const TYPED_ARRAY_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  TYPED_ARRAY_PROTOTYPE,
  "byteLength",
)?.get;
const NUMBER_IS_FINITE = Number.isFinite;

export interface PrepareSemanticAttentionWgslRequest
  extends PrepareAttentionForwardSpecializationRequest {
  readonly maxWgslBytes?: number;
  readonly maxWorkgroupInvocations?: number;
  readonly maxWorkgroupStorageBytes?: number;
  readonly maxPrivateElementsPerInvocation?: number;
  readonly maxKeyTiles?: number;
  readonly maxDispatchWorkgroups?: number;
  /** Bounds owned input snapshots/results plus GPU upload/output/readback storage. */
  readonly maxTransientWorkingSetBytes?: number;
}

export interface PreparedSemanticAttentionWgsl {
  readonly semantic: PreparedAttentionForwardSpecialization;
  readonly scheduled: PreparedAttentionOnlineKvTileSchedule;
  readonly backendProfile: typeof SEMANTIC_ATTENTION_WEBGPU_PROFILE;
  readonly backendVersion: typeof SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION;
  readonly algorithmProfile: "block-tiled-kv-online-softmax-forward";
  readonly preservationLevel: "portable-relegalized";
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

export interface SemanticAttentionWebGpuInputs {
  readonly query: Uint8Array;
  readonly key: Uint8Array;
  readonly value: Uint8Array;
}

export interface SemanticAttentionWebGpuRunOptions {
  /** Stops queued work or suppresses publication from work already submitted. */
  readonly signal?: AbortSignal;
  /** Bounds finite-domain validation over the private input snapshots. */
  readonly maxInputValidationMs?: number;
  /** Caller-visible wait budget; submitted work completes cleanup in the background. */
  readonly timeoutMs?: number;
}

export interface SemanticAttentionWebGpuDeviceFacts {
  readonly features: readonly string[];
  readonly limits: Readonly<{
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
    readonly maxComputeWorkgroupsPerDimension: number;
    readonly maxComputeInvocationsPerWorkgroup: number;
    readonly maxComputeWorkgroupSizeX: number;
    readonly maxComputeWorkgroupStorageSize: number;
    readonly maxBindingsPerBindGroup: number;
    readonly maxStorageBuffersPerShaderStage: number;
  }>;
}

export interface SemanticAttentionWebGpuTrace {
  readonly operationId: string;
  readonly semanticSpecializationHash: string;
  readonly scheduleSpecializationHash: string;
  readonly backendPreparationHash: string;
  readonly backendSpecializationHash: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly scheduleSemanticHash: string;
  readonly wgslModuleHash: string;
  readonly backendProfile: typeof SEMANTIC_ATTENTION_WEBGPU_PROFILE;
  readonly backendVersion: typeof SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION;
  readonly algorithmProfile: "block-tiled-kv-online-softmax-forward";
  readonly executionTier: "portable-webgpu-core";
  readonly preservationLevel: "portable-relegalized";
  readonly numericalPreservation: "requires-declared-policy-comparison";
  readonly comparisonPolicyId: "browsergrad.attention-forward.f32-abs-relative@1";
  readonly mask: "none" | "causal-upper-left";
  readonly batch: WireU64;
  readonly heads: WireU64;
  readonly queryLength: WireU64;
  readonly keyLength: WireU64;
  readonly queryDepth: WireU64;
  readonly valueDepth: WireU64;
  readonly physicalTile: Readonly<{
    readonly queryRows: WireU64;
    readonly keyRows: WireU64;
  }>;
  readonly dispatchWorkgroups: Readonly<{
    readonly x: WireU64;
    readonly y: WireU64;
    readonly z: WireU64;
  }>;
  readonly validScoreElements: WireU64;
  readonly scoreMultiplyAdds: WireU64;
  readonly weightedValueMultiplyAdds: WireU64;
  readonly logicalBytesRead: WireU64;
  readonly logicalBytesWritten: WireU64;
  readonly plannedTransientGpuBytes: WireU64;
  readonly plannedTransientHostBytes: WireU64;
  readonly plannedTransientWorkingSetBytes: WireU64;
  readonly maxTransientWorkingSetBytes: WireU64;
  readonly submitted: true;
  readonly device: SemanticAttentionWebGpuDeviceFacts;
}

export interface SemanticAttentionWebGpuResult {
  readonly destination: Uint8Array;
  readonly trace: SemanticAttentionWebGpuTrace;
}

export type SemanticAttentionWebGpuErrorCode =
  | "BG-WEBGPU-ATTENTION-UNSUPPORTED-PROFILE"
  | "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT"
  | "BG-WEBGPU-ATTENTION-INVALID-BINDING"
  | "BG-WEBGPU-ATTENTION-NUMERICAL-DOMAIN"
  | "BG-WEBGPU-ATTENTION-DEVICE-LIMIT"
  | "BG-WEBGPU-ATTENTION-SHADER"
  | "BG-WEBGPU-ATTENTION-PIPELINE"
  | "BG-WEBGPU-ATTENTION-VALIDATION"
  | "BG-WEBGPU-ATTENTION-OUT-OF-MEMORY"
  | "BG-WEBGPU-ATTENTION-INTERNAL"
  | "BG-WEBGPU-ATTENTION-DEVICE-LOST"
  | "BG-WEBGPU-ATTENTION-CANCELLED"
  | "BG-WEBGPU-ATTENTION-TIMEOUT"
  | "BG-WEBGPU-ATTENTION-EXECUTION";

export class SemanticAttentionWebGpuError extends Error {
  constructor(
    readonly code: SemanticAttentionWebGpuErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SemanticAttentionWebGpuError";
  }
}

/**
 * Composes verified attention meaning and an independent online K/V-tile
 * schedule, then lowers only the portable scalar WebGPU realization.
 * Preparation grants no device execution or performance authority.
 */
export async function prepareSemanticAttentionWgsl(
  layoutArtifact: VerifiedLayoutArtifact,
  attentionArtifact: VerifiedAttentionForwardArtifact,
  scheduleArtifact: VerifiedAttentionOnlineKvTileScheduleArtifact,
  request: PrepareSemanticAttentionWgslRequest,
): Promise<PreparedSemanticAttentionWgsl> {
  const captured = capturePrepareRequest(request);
  const maxWgslBytes = positiveInteger(
    captured.maxWgslBytes,
    DEFAULT_MAX_WGSL_BYTES,
    MAX_CONFIGURABLE_WGSL_BYTES,
    "$.maxWgslBytes",
  );
  const maxWorkgroupInvocations = positiveInteger(
    captured.maxWorkgroupInvocations,
    DEFAULT_MAX_WORKGROUP_INVOCATIONS,
    MAX_CONFIGURABLE_WORKGROUP_INVOCATIONS,
    "$.maxWorkgroupInvocations",
  );
  const maxWorkgroupStorageBytes = positiveInteger(
    captured.maxWorkgroupStorageBytes,
    DEFAULT_MAX_WORKGROUP_STORAGE_BYTES,
    MAX_CONFIGURABLE_WORKGROUP_STORAGE_BYTES,
    "$.maxWorkgroupStorageBytes",
  );
  const maxPrivateElementsPerInvocation = positiveInteger(
    captured.maxPrivateElementsPerInvocation,
    DEFAULT_MAX_PRIVATE_ELEMENTS_PER_INVOCATION,
    MAX_CONFIGURABLE_PRIVATE_ELEMENTS_PER_INVOCATION,
    "$.maxPrivateElementsPerInvocation",
  );
  const maxKeyTiles = positiveInteger(
    captured.maxKeyTiles,
    DEFAULT_MAX_KEY_TILES,
    MAX_CONFIGURABLE_KEY_TILES,
    "$.maxKeyTiles",
  );
  const maxDispatchWorkgroups = positiveInteger(
    captured.maxDispatchWorkgroups,
    DEFAULT_MAX_DISPATCH_WORKGROUPS,
    MAX_CONFIGURABLE_DISPATCH_WORKGROUPS,
    "$.maxDispatchWorkgroups",
  );
  const maxTransientWorkingSetBytes = positiveInteger(
    captured.maxTransientWorkingSetBytes,
    DEFAULT_MAX_TRANSIENT_WORKING_SET_BYTES,
    MAX_CONFIGURABLE_TRANSIENT_WORKING_SET_BYTES,
    "$.maxTransientWorkingSetBytes",
  );

  const semantic = await prepareAttentionForwardSpecialization(
    layoutArtifact,
    attentionArtifact,
    logicalRequest(captured),
  );
  requireInitialWebGpuProfile(semantic);
  const scheduled = await prepareAttentionOnlineKvTileSchedule(
    semantic,
    attentionArtifact,
    scheduleArtifact,
    {
      ...(captured.evaluationLimits === undefined
        ? {}
        : { evaluationLimits: captured.evaluationLimits }),
      maxWorkgroupInvocations,
      maxStagingBytes: maxWorkgroupStorageBytes,
      maxPrivateElementsPerInvocation,
      maxKeyTiles,
      maxDispatchWorkgroups,
    },
  );

  let emitted;
  try {
    emitted = emitSemanticAttentionWgsl(semantic, scheduled);
  } catch (error) {
    if (error instanceof SemanticAttentionWgslLoweringError) {
      throw new SemanticAttentionWebGpuError(error.code, error.path, error.message, {
        cause: error,
      });
    }
    throw error;
  }
  if (emitted.workgroupStorageBytes > BigInt(maxWorkgroupStorageBytes)) {
    fail(
      "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT",
      "$.maxWorkgroupStorageBytes",
      `generated K/V staging requires ${emitted.workgroupStorageBytes} bytes; limit is ${maxWorkgroupStorageBytes}`,
    );
  }
  const wgslBytes = (REFLECT_APPLY(
    TEXT_ENCODER_ENCODE,
    TEXT_ENCODER,
    [emitted.source],
  ) as Uint8Array).byteLength;
  if (wgslBytes > maxWgslBytes) {
    fail(
      "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT",
      "$.maxWgslBytes",
      `generated WGSL requires ${wgslBytes} bytes; limit is ${maxWgslBytes}`,
    );
  }

  const inputBytes = semantic.query.allocationByteLength
    + semantic.key.allocationByteLength
    + semantic.value.allocationByteLength;
  const destinationBytes = semantic.destination.allocationByteLength;
  const plannedTransientGpuBytes = inputBytes + (destinationBytes * 2n);
  const plannedTransientHostBytes = inputBytes + (destinationBytes * 2n);
  const plannedTransientWorkingSetBytes = plannedTransientGpuBytes + plannedTransientHostBytes;
  if (plannedTransientWorkingSetBytes > BigInt(maxTransientWorkingSetBytes)) {
    fail(
      "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT",
      "$.maxTransientWorkingSetBytes",
      `planned owned host and GPU buffers require ${plannedTransientWorkingSetBytes} bytes; limit is ${maxTransientWorkingSetBytes}`,
    );
  }

  const backendPreparationHash = await hashNamedComponents({
    profile: SEMANTIC_ATTENTION_WEBGPU_PROFILE,
    backendVersion: SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION,
    algorithmProfile: "block-tiled-kv-online-softmax-forward",
    semanticSpecialization: semantic.specializationHash,
    scheduleSpecialization: scheduled.scheduleSpecializationHash,
    workgroupStorageBytes: encodeWireU64(emitted.workgroupStorageBytes),
  });
  const wgslModuleHash = await hashNamedComponents({
    backendPreparation: backendPreparationHash,
    source: emitted.source,
  });
  const program = freezeProgram(defineWgslKernelProgram({
    name: `bg_semantic_attention_${wgslModuleHash}`,
    wgsl: emitted.source,
    bindings: [
      { kind: "storage", name: "query_values", valueType: "f32", access: "read", binding: 0 },
      { kind: "storage", name: "key_values", valueType: "f32", access: "read", binding: 1 },
      { kind: "storage", name: "value_values", valueType: "f32", access: "read", binding: 2 },
      { kind: "storage", name: "destination_values", valueType: "f32", access: "read_write", binding: 3 },
    ],
    workgroupSize: emitted.workgroupSize,
  }));
  const launch = Object.freeze({
    dispatchCount: Object.freeze([
      safeNumber(scheduled.dispatchX * scheduled.queryRows, "$.launch.x"),
      safeNumber(scheduled.dispatchY, "$.launch.y"),
      safeNumber(scheduled.dispatchZ, "$.launch.z"),
    ] as const),
  });
  const prepared = Object.freeze({
    semantic,
    scheduled,
    backendProfile: SEMANTIC_ATTENTION_WEBGPU_PROFILE,
    backendVersion: SEMANTIC_ATTENTION_WEBGPU_BACKEND_VERSION,
    algorithmProfile: "block-tiled-kv-online-softmax-forward" as const,
    preservationLevel: "portable-relegalized" as const,
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
  PREPARED_SEMANTIC_ATTENTIONS.add(prepared);
  return prepared;
}

/**
 * Snapshots and validates finite Q/K/V host inputs before device access, runs
 * the exact prepared block-tiled module, and returns the complete dense output.
 * Numerical preservation remains a conformance comparison, not a run-time
 * self-assertion.
 */
export async function runSemanticAttentionWebGpu(
  device: KernelDevice,
  prepared: PreparedSemanticAttentionWgsl,
  inputs: SemanticAttentionWebGpuInputs,
  options: SemanticAttentionWebGpuRunOptions = {},
): Promise<SemanticAttentionWebGpuResult> {
  requirePrepared(prepared);
  const capturedOptions = captureRunOptions(options);
  throwIfCancelled(capturedOptions.signal);
  const maxInputValidationMs = positiveInteger(
    capturedOptions.maxInputValidationMs,
    DEFAULT_MAX_INPUT_VALIDATION_MS,
    MAX_INPUT_VALIDATION_MS,
    "$.maxInputValidationMs",
  );
  const timeoutMs = positiveInteger(
    capturedOptions.timeoutMs,
    DEFAULT_EXECUTION_TIMEOUT_MS,
    MAX_EXECUTION_TIMEOUT_MS,
    "$.timeoutMs",
  );
  const snapshots = await captureFiniteF32Snapshots(
    inputs,
    [
      {
        name: "query",
        expectedByteLength: prepared.semantic.query.allocationByteLength,
        alignmentBytes: prepared.semantic.query.allocationAlignmentBytes,
      },
      {
        name: "key",
        expectedByteLength: prepared.semantic.key.allocationByteLength,
        alignmentBytes: prepared.semantic.key.allocationAlignmentBytes,
      },
      {
        name: "value",
        expectedByteLength: prepared.semantic.value.allocationByteLength,
        alignmentBytes: prepared.semantic.value.allocationAlignmentBytes,
      },
    ] as const,
    "$.inputs",
    {
      maxValidationMs: maxInputValidationMs,
      ...(capturedOptions.signal === undefined ? {} : { signal: capturedOptions.signal }),
      fail: snapshotFailure,
    },
  );
  throwIfCancelled(capturedOptions.signal);
  requireAvailableDevice(device);
  const deviceFacts = readAndVerifyDeviceFacts(device.gpu, prepared);
  throwIfCancelled(capturedOptions.signal);

  ACTIVE_SEMANTIC_ATTENTION_DEVICES.add(device.gpu);
  const execution = executeHostReadback(device, prepared, snapshots);
  void execution.then(
    () => ACTIVE_SEMANTIC_ATTENTION_DEVICES.delete(device.gpu),
    () => ACTIVE_SEMANTIC_ATTENTION_DEVICES.delete(device.gpu),
  );
  try {
    const backendSpecializationHashPromise = hashNamedComponents({
      backendPreparation: prepared.backendPreparationHash,
      deviceFeatures: [...deviceFacts.features],
      deviceLimits: deviceFacts.limits as unknown as JsonObject,
    });
    const [destination, backendSpecializationHash] = await Promise.all([
      awaitBoundedExecution(
        execution,
        device.gpu,
        capturedOptions.signal,
        timeoutMs,
      ),
      backendSpecializationHashPromise,
    ]);
    throwIfCancelled(capturedOptions.signal);
    return Object.freeze({
      destination,
      trace: createTrace(prepared, backendSpecializationHash, deviceFacts),
    });
  } catch (error) {
    throw classifyExecutionError(error);
  }
}

async function executeHostReadback(
  device: KernelDevice,
  prepared: PreparedSemanticAttentionWgsl,
  inputs: Readonly<Record<"query" | "key" | "value", Uint8Array>>,
): Promise<Uint8Array> {
  const destinationBytes = safeNumber(
    prepared.semantic.destination.allocationByteLength,
    "$.semantic.destination.allocationByteLength",
  );
  const result = await runPreparedSemanticWebGpuHostReadback(
    device,
    [{ program: prepared.program, launch: prepared.launch }],
    {
      buffers: {
        query_values: new FLOAT32_ARRAY_CONSTRUCTOR(
          inputs.query.buffer,
          0,
          inputs.query.byteLength / 4,
        ),
        key_values: new FLOAT32_ARRAY_CONSTRUCTOR(
          inputs.key.buffer,
          0,
          inputs.key.byteLength / 4,
        ),
        value_values: new FLOAT32_ARRAY_CONSTRUCTOR(
          inputs.value.buffer,
          0,
          inputs.value.byteLength / 4,
        ),
        destination_values: new FLOAT32_ARRAY_CONSTRUCTOR(destinationBytes / 4),
      },
      readback: ["destination_values"],
    },
  );
  const destination = result.buffers.destination_values;
  if (
    !(destination instanceof FLOAT32_ARRAY_CONSTRUCTOR)
    || OBJECT_GET_PROTOTYPE_OF(destination) !== FLOAT32_ARRAY_PROTOTYPE
    || TYPED_ARRAY_BUFFER_GETTER === undefined
    || TYPED_ARRAY_BYTE_OFFSET_GETTER === undefined
    || TYPED_ARRAY_BYTE_LENGTH_GETTER === undefined
  ) {
    fail(
      "BG-WEBGPU-ATTENTION-INTERNAL",
      "$.result.destination",
      "WGSL runner returned an invalid f32 destination allocation",
    );
  }
  let buffer: ArrayBuffer;
  let byteOffset: number;
  let byteLength: number;
  try {
    buffer = REFLECT_APPLY(TYPED_ARRAY_BUFFER_GETTER, destination, []) as ArrayBuffer;
    byteOffset = REFLECT_APPLY(TYPED_ARRAY_BYTE_OFFSET_GETTER, destination, []) as number;
    byteLength = REFLECT_APPLY(TYPED_ARRAY_BYTE_LENGTH_GETTER, destination, []) as number;
  } catch (error) {
    throw new SemanticAttentionWebGpuError(
      "BG-WEBGPU-ATTENTION-INTERNAL",
      "$.result.destination",
      "WGSL destination does not expose native typed-array slots",
      { cause: error },
    );
  }
  if (byteLength !== destinationBytes || byteLength % 4 !== 0) {
    fail(
      "BG-WEBGPU-ATTENTION-INTERNAL",
      "$.result.destination",
      `WGSL destination length ${byteLength} does not equal ${destinationBytes}`,
    );
  }
  for (let index = 0; index < destination.length; index += 1) {
    if (!NUMBER_IS_FINITE(destination[index])) {
      fail(
        "BG-WEBGPU-ATTENTION-NUMERICAL-DOMAIN",
        "$.result.destination",
        `WebGPU attention output element ${index} is not finite f32`,
      );
    }
  }
  const output = new UINT8_ARRAY_CONSTRUCTOR(byteLength);
  output.set(new UINT8_ARRAY_CONSTRUCTOR(buffer, byteOffset, byteLength));
  return output;
}

function createTrace(
  prepared: PreparedSemanticAttentionWgsl,
  backendSpecializationHash: string,
  device: SemanticAttentionWebGpuDeviceFacts,
): SemanticAttentionWebGpuTrace {
  const validScoreElements = attentionValidScoreElements(prepared.semantic);
  const scoreMultiplyAdds = validScoreElements * prepared.semantic.queryDepth;
  const weightedValueMultiplyAdds = validScoreElements * prepared.semantic.valueDepth;
  return Object.freeze({
    operationId: prepared.semantic.operation.operationId,
    semanticSpecializationHash: prepared.semantic.specializationHash,
    scheduleSpecializationHash: prepared.scheduled.scheduleSpecializationHash,
    backendPreparationHash: prepared.backendPreparationHash,
    backendSpecializationHash,
    layoutSemanticHash: prepared.semantic.layoutSemanticHash,
    kernelSemanticHash: prepared.semantic.kernelSemanticHash,
    scheduleSemanticHash: prepared.scheduled.scheduleSemanticHash,
    wgslModuleHash: prepared.wgslModuleHash,
    backendProfile: prepared.backendProfile,
    backendVersion: prepared.backendVersion,
    algorithmProfile: prepared.algorithmProfile,
    executionTier: "portable-webgpu-core",
    preservationLevel: prepared.preservationLevel,
    numericalPreservation: "requires-declared-policy-comparison",
    comparisonPolicyId: "browsergrad.attention-forward.f32-abs-relative@1",
    mask: prepared.semantic.operation.mask.kind === "causal" ? "causal-upper-left" : "none",
    batch: encodeWireU64(prepared.semantic.batch),
    heads: encodeWireU64(prepared.semantic.heads),
    queryLength: encodeWireU64(prepared.semantic.queryLength),
    keyLength: encodeWireU64(prepared.semantic.keyLength),
    queryDepth: encodeWireU64(prepared.semantic.queryDepth),
    valueDepth: encodeWireU64(prepared.semantic.valueDepth),
    physicalTile: Object.freeze({
      queryRows: encodeWireU64(prepared.scheduled.queryRows),
      keyRows: encodeWireU64(prepared.scheduled.keyRows),
    }),
    dispatchWorkgroups: Object.freeze({
      x: encodeWireU64(prepared.scheduled.dispatchX),
      y: encodeWireU64(prepared.scheduled.dispatchY),
      z: encodeWireU64(prepared.scheduled.dispatchZ),
    }),
    validScoreElements: encodeWireU64(validScoreElements),
    scoreMultiplyAdds: encodeWireU64(scoreMultiplyAdds),
    weightedValueMultiplyAdds: encodeWireU64(weightedValueMultiplyAdds),
    logicalBytesRead: encodeWireU64((scoreMultiplyAdds * 8n) + (weightedValueMultiplyAdds * 4n)),
    logicalBytesWritten: encodeWireU64(prepared.semantic.outputElements * 4n),
    plannedTransientGpuBytes: prepared.plannedTransientGpuBytes,
    plannedTransientHostBytes: prepared.plannedTransientHostBytes,
    plannedTransientWorkingSetBytes: prepared.plannedTransientWorkingSetBytes,
    maxTransientWorkingSetBytes: prepared.maxTransientWorkingSetBytes,
    submitted: true,
    device,
  });
}

function attentionValidScoreElements(
  semantic: PreparedAttentionForwardSpecialization,
): bigint {
  if (semantic.operation.mask.kind === "none") return semantic.maximumScoreElements;
  let scoresPerBatchHead = 0n;
  for (let query = 0n; query < semantic.queryLength; query += 1n) {
    const causalKeys = query + 1n;
    scoresPerBatchHead += causalKeys < semantic.keyLength ? causalKeys : semantic.keyLength;
  }
  return semantic.batch * semantic.heads * scoresPerBatchHead;
}

function snapshotFailure(
  issue: "invalid-binding" | "numerical-domain" | "resource-limit" | "cancelled" | "internal",
  path: string,
  message: string,
  cause?: unknown,
): never {
  const codes: Readonly<Record<typeof issue, SemanticAttentionWebGpuErrorCode>> = {
    "invalid-binding": "BG-WEBGPU-ATTENTION-INVALID-BINDING",
    "numerical-domain": "BG-WEBGPU-ATTENTION-NUMERICAL-DOMAIN",
    "resource-limit": "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT",
    cancelled: "BG-WEBGPU-ATTENTION-CANCELLED",
    internal: "BG-WEBGPU-ATTENTION-INTERNAL",
  };
  throw new SemanticAttentionWebGpuError(
    codes[issue],
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}

function requirePrepared(prepared: PreparedSemanticAttentionWgsl): void {
  if (!PREPARED_SEMANTIC_ATTENTIONS.has(prepared as object)) {
    fail(
      "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      "$.prepared",
      "prepared plan was not produced by prepareSemanticAttentionWgsl in this module instance",
    );
  }
}

const PREPARE_REQUEST_FIELDS = Object.freeze([
  "operationId",
  "bindings",
  "evaluationLimits",
  "maxElements",
  "maxAllocationBytes",
  "maxScalarOperations",
  "maxEvaluationSteps",
  "maxPreparationMs",
  "signal",
  "maxWgslBytes",
  "maxWorkgroupInvocations",
  "maxWorkgroupStorageBytes",
  "maxPrivateElementsPerInvocation",
  "maxKeyTiles",
  "maxDispatchWorkgroups",
  "maxTransientWorkingSetBytes",
] as const);

function capturePrepareRequest(
  value: PrepareSemanticAttentionWgslRequest,
): PrepareSemanticAttentionWgslRequest {
  const captured = captureClosedDataRecord(
    value,
    PREPARE_REQUEST_FIELDS,
    ["operationId"],
    "$.request",
  ) as unknown as PrepareSemanticAttentionWgslRequest;
  validateOptionalAbortSignal(captured.signal, "$.request.signal");
  return captured;
}

function captureRunOptions(
  value: SemanticAttentionWebGpuRunOptions,
): SemanticAttentionWebGpuRunOptions {
  const captured = captureClosedDataRecord(
    value,
    ["signal", "maxInputValidationMs", "timeoutMs"],
    [],
    "$.options",
  ) as unknown as SemanticAttentionWebGpuRunOptions;
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
    fail("BG-WEBGPU-ATTENTION-INVALID-BINDING", path, "expected a plain options object");
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = OBJECT_GET_PROTOTYPE_OF(value);
    descriptors = OBJECT_GET_OWN_PROPERTY_DESCRIPTORS(value);
  } catch (error) {
    throw new SemanticAttentionWebGpuError(
      "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      path,
      "options object could not be captured without invoking properties",
      { cause: error },
    );
  }
  if (prototype !== Object.prototype && prototype !== null) {
    fail("BG-WEBGPU-ATTENTION-INVALID-BINDING", path, "expected a plain options object");
  }
  const allowed = new Set(allowedFields);
  const result = Object.create(null) as Record<string, unknown>;
  for (const key of REFLECT_OWN_KEYS(descriptors)) {
    if (typeof key !== "string" || !allowed.has(key)) {
      fail(
        "BG-WEBGPU-ATTENTION-INVALID-BINDING",
        path,
        `unknown closed-record field ${typeof key === "string" ? key : "symbol"}`,
      );
    }
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      fail(
        "BG-WEBGPU-ATTENTION-INVALID-BINDING",
        `${path}.${key}`,
        "options must use enumerable own data properties without accessors",
      );
    }
    result[key] = descriptor.value;
  }
  for (const field of requiredFields) {
    if (!(field in result)) {
      fail(
        "BG-WEBGPU-ATTENTION-INVALID-BINDING",
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
    fail("BG-WEBGPU-ATTENTION-INVALID-BINDING", path, "signal must be a native AbortSignal");
  }
  try {
    if (typeof REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, value, []) !== "boolean") {
      throw new Error("missing native AbortSignal slots");
    }
  } catch (error) {
    throw new SemanticAttentionWebGpuError(
      "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      path,
      "signal does not expose native AbortSignal state",
      { cause: error },
    );
  }
}

function logicalRequest(
  request: PrepareSemanticAttentionWgslRequest,
): PrepareAttentionForwardSpecializationRequest {
  return {
    operationId: request.operationId,
    ...(request.bindings === undefined ? {} : { bindings: request.bindings }),
    ...(request.evaluationLimits === undefined
      ? {}
      : { evaluationLimits: request.evaluationLimits }),
    ...(request.maxElements === undefined ? {} : { maxElements: request.maxElements }),
    ...(request.maxAllocationBytes === undefined
      ? {}
      : { maxAllocationBytes: request.maxAllocationBytes }),
    ...(request.maxScalarOperations === undefined
      ? {}
      : { maxScalarOperations: request.maxScalarOperations }),
    ...(request.maxEvaluationSteps === undefined
      ? {}
      : { maxEvaluationSteps: request.maxEvaluationSteps }),
    ...(request.maxPreparationMs === undefined
      ? {}
      : { maxPreparationMs: request.maxPreparationMs }),
    ...(request.signal === undefined ? {} : { signal: request.signal }),
  };
}

function requireInitialWebGpuProfile(
  semantic: PreparedAttentionForwardSpecialization,
): void {
  for (const [name, value] of [
    ["batch", semantic.batch],
    ["heads", semantic.heads],
    ["queryLength", semantic.queryLength],
    ["keyLength", semantic.keyLength],
    ["queryDepth", semantic.queryDepth],
    ["valueDepth", semantic.valueDepth],
  ] as const) {
    if (value === 0n) {
      fail(
        "BG-WEBGPU-ATTENTION-UNSUPPORTED-PROFILE",
        `$.semantic.${name}`,
        "initial WebGPU attention profile requires nonempty dimensions",
      );
    }
  }
  const denseDestinationBytes = semantic.outputElements * 4n;
  if (
    semantic.destination.viewByteOffset !== 0n
    || semantic.destination.allocationByteLength !== denseDestinationBytes
  ) {
    fail(
      "BG-WEBGPU-ATTENTION-UNSUPPORTED-PROFILE",
      "$.semantic.destination",
      "host/readback attention requires a zero-offset dense destination that overwrites the complete root allocation",
    );
  }
}

function readAndVerifyDeviceFacts(
  gpu: GPUDevice,
  prepared: PreparedSemanticAttentionWgsl,
): SemanticAttentionWebGpuDeviceFacts {
  const limits = Object.freeze({
    maxBufferSize: gpu.limits.maxBufferSize,
    maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: gpu.limits.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: gpu.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: gpu.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupStorageSize: gpu.limits.maxComputeWorkgroupStorageSize,
    maxBindingsPerBindGroup: gpu.limits.maxBindingsPerBindGroup,
    maxStorageBuffersPerShaderStage: gpu.limits.maxStorageBuffersPerShaderStage,
  });
  const allocations = [
    prepared.semantic.query.allocationByteLength,
    prepared.semantic.key.allocationByteLength,
    prepared.semantic.value.allocationByteLength,
    prepared.semantic.destination.allocationByteLength,
  ];
  const maximumAllocation = allocations.reduce(
    (maximum, value) => value > maximum ? value : maximum,
    0n,
  );
  if (maximumAllocation > BigInt(limits.maxBufferSize)) {
    fail(
      "BG-WEBGPU-ATTENTION-DEVICE-LIMIT",
      "$.device.limits.maxBufferSize",
      `allocation requires ${maximumAllocation} bytes; device limit is ${limits.maxBufferSize}`,
    );
  }
  if (maximumAllocation > BigInt(limits.maxStorageBufferBindingSize)) {
    fail(
      "BG-WEBGPU-ATTENTION-DEVICE-LIMIT",
      "$.device.limits.maxStorageBufferBindingSize",
      `allocation requires ${maximumAllocation} bytes; device limit is ${limits.maxStorageBufferBindingSize}`,
    );
  }
  const workgroupX = prepared.program.workgroupSize[0];
  if (
    workgroupX > limits.maxComputeInvocationsPerWorkgroup
    || workgroupX > limits.maxComputeWorkgroupSizeX
  ) {
    fail(
      "BG-WEBGPU-ATTENTION-DEVICE-LIMIT",
      "$.device.limits.workgroupSize",
      `workgroup size ${workgroupX} exceeds device limits`,
    );
  }
  if (BigInt(prepared.workgroupStorageBytes) > BigInt(limits.maxComputeWorkgroupStorageSize)) {
    fail(
      "BG-WEBGPU-ATTENTION-DEVICE-LIMIT",
      "$.device.limits.maxComputeWorkgroupStorageSize",
      `K/V staging requires ${prepared.workgroupStorageBytes} bytes; device limit is ${limits.maxComputeWorkgroupStorageSize}`,
    );
  }
  if (limits.maxBindingsPerBindGroup < 4 || limits.maxStorageBuffersPerShaderStage < 4) {
    fail(
      "BG-WEBGPU-ATTENTION-DEVICE-LIMIT",
      "$.device.limits.bindings",
      "device cannot bind the four required storage allocations",
    );
  }
  const dispatchLimit = BigInt(limits.maxComputeWorkgroupsPerDimension);
  if (
    prepared.scheduled.dispatchX > dispatchLimit
    || prepared.scheduled.dispatchY > dispatchLimit
    || prepared.scheduled.dispatchZ > dispatchLimit
  ) {
    fail(
      "BG-WEBGPU-ATTENTION-DEVICE-LIMIT",
      "$.launch",
      `dispatch requires ${prepared.scheduled.dispatchX}x${prepared.scheduled.dispatchY}x${prepared.scheduled.dispatchZ} workgroups; per-dimension limit is ${limits.maxComputeWorkgroupsPerDimension}`,
    );
  }
  return Object.freeze({
    features: Object.freeze([...gpu.features].map(String).sort()),
    limits,
  });
}

function watchDeviceLoss(device: KernelDevice): void {
  if (WATCHED_SEMANTIC_ATTENTION_DEVICES.has(device.gpu)) return;
  WATCHED_SEMANTIC_ATTENTION_DEVICES.add(device.gpu);
  void device.gpu.lost.then(
    () => invalidateLostDevice(device),
    () => invalidateLostDevice(device),
  );
}

function invalidateLostDevice(device: KernelDevice): void {
  LOST_SEMANTIC_ATTENTION_DEVICES.add(device.gpu);
  clearWgslPipelineCache(device);
  device.clearCache();
}

function requireAvailableDevice(device: KernelDevice): void {
  watchDeviceLoss(device);
  if (LOST_SEMANTIC_ATTENTION_DEVICES.has(device.gpu)) {
    fail(
      "BG-WEBGPU-ATTENTION-DEVICE-LOST",
      "$.device",
      "WebGPU device was previously lost and cannot execute this prepared plan",
    );
  }
  if (ACTIVE_SEMANTIC_ATTENTION_DEVICES.has(device.gpu)) {
    fail(
      "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT",
      "$.device.inFlight",
      "only one semantic attention operation may be in flight per GPUDevice",
    );
  }
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
    timeout = setTimeout(() => reject(new SemanticAttentionWebGpuError(
      "BG-WEBGPU-ATTENTION-TIMEOUT",
      "$.timeoutMs",
      `semantic attention exceeded caller wait budget of ${timeoutMs} ms`,
    )), timeoutMs);
  });
  const cancellationPromise = signal === undefined
    ? new Promise<never>(() => undefined)
    : new Promise<never>((_resolve, reject) => {
        const onAbort = () => reject(cancelledError());
        if (EVENT_TARGET_ADD_EVENT_LISTENER === undefined
          || EVENT_TARGET_REMOVE_EVENT_LISTENER === undefined) {
          reject(new SemanticAttentionWebGpuError(
            "BG-WEBGPU-ATTENTION-INTERNAL",
            "$.signal",
            "native EventTarget methods are unavailable",
          ));
          return;
        }
        REFLECT_APPLY(EVENT_TARGET_ADD_EVENT_LISTENER, signal, [
          "abort",
          onAbort,
          { once: true },
        ]);
        removeAbortListener = () => {
          REFLECT_APPLY(EVENT_TARGET_REMOVE_EVENT_LISTENER, signal, ["abort", onAbort]);
        };
        if (abortSignalAborted(signal)) onAbort();
      });
  const deviceLostPromise = gpu.lost.then((info) => {
    throw new SemanticAttentionWebGpuError(
      "BG-WEBGPU-ATTENTION-DEVICE-LOST",
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
  if (error instanceof SemanticAttentionWebGpuError) return error;
  if (error instanceof SemanticWebGpuHostError) {
    const codes: Readonly<Record<SemanticWebGpuHostError["issue"], SemanticAttentionWebGpuErrorCode>> = {
      shader: "BG-WEBGPU-ATTENTION-SHADER",
      pipeline: "BG-WEBGPU-ATTENTION-PIPELINE",
      validation: "BG-WEBGPU-ATTENTION-VALIDATION",
      "out-of-memory": "BG-WEBGPU-ATTENTION-OUT-OF-MEMORY",
      internal: "BG-WEBGPU-ATTENTION-INTERNAL",
      "device-lost": "BG-WEBGPU-ATTENTION-DEVICE-LOST",
      "error-scope": "BG-WEBGPU-ATTENTION-INTERNAL",
      execution: "BG-WEBGPU-ATTENTION-EXECUTION",
    };
    return new SemanticAttentionWebGpuError(
      codes[error.issue],
      error.path,
      error.message,
      { cause: error },
    );
  }
  return new SemanticAttentionWebGpuError(
    "BG-WEBGPU-ATTENTION-EXECUTION",
    "$.dispatch",
    error instanceof Error ? error.message : String(error),
    { cause: error },
  );
}

function throwIfCancelled(signal: AbortSignal | undefined): void {
  if (signal !== undefined && abortSignalAborted(signal)) throw cancelledError();
}

function abortSignalAborted(signal: AbortSignal): boolean {
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined) {
    fail("BG-WEBGPU-ATTENTION-INTERNAL", "$.signal", "native AbortSignal getter is unavailable");
  }
  try {
    return REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER, signal, []) as boolean;
  } catch (error) {
    throw new SemanticAttentionWebGpuError(
      "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      "$.signal",
      "signal does not expose native AbortSignal state",
      { cause: error },
    );
  }
}

function cancelledError(): SemanticAttentionWebGpuError {
  return new SemanticAttentionWebGpuError(
    "BG-WEBGPU-ATTENTION-CANCELLED",
    "$.signal",
    "semantic attention was cancelled",
  );
}

function positiveInteger(
  value: number | undefined,
  fallback: number,
  maximum: number,
  path: string,
): number {
  const resolved = value ?? fallback;
  if (!NUMBER_IS_SAFE_INTEGER(resolved) || resolved <= 0 || resolved > maximum) {
    fail(
      "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT",
      path,
      `value must be a positive safe integer no greater than ${maximum}`,
    );
  }
  return resolved;
}

function safeNumber(value: bigint, path: string): number {
  if (value <= 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    fail(
      "BG-WEBGPU-ATTENTION-UNSUPPORTED-PROFILE",
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

function fail(
  code: SemanticAttentionWebGpuErrorCode,
  path: string,
  message: string,
): never {
  throw new SemanticAttentionWebGpuError(code, path, message);
}

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
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  defineWgslKernelProgram,
  type WgslKernelLaunch,
  type WgslKernelProgram,
} from "./wgsl_program.js";
import {
  emitSemanticAttentionWgsl,
  SemanticAttentionWgslLoweringError,
} from "./semantic_attention_wgsl.js";

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
const PREPARED_SEMANTIC_ATTENTIONS = new WeakSet<object>();
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

export type SemanticAttentionWebGpuErrorCode =
  | "BG-WEBGPU-ATTENTION-UNSUPPORTED-PROFILE"
  | "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT"
  | "BG-WEBGPU-ATTENTION-INVALID-BINDING";

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

import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
  type InspectedUnsharedUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  canonicalCppCuteBrowserWorkerProfileRegionBytes,
  canonicalCppCuteBrowserWorkerRequestRegionBytes,
  copyCppCuteBrowserWorkerSourceSnapshots,
  unwrapPreparedCppCuteBrowserWorkerInvocation,
  type PreparedCppCuteBrowserWorkerInvocation,
} from "./cpp_cute_browser_worker_protocol.js";
import { unwrapPreparedCppCuteBrowserRuntimeAbiManifest } from "./cpp_cute_browser_runtime_abi.js";
import { CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE } from "./resources/cpp_cute_browser_runtime_abi_v1.js";

const UINT32_MAX = 0xffff_ffff;
const INPUT_FRAME_MAXIMUM_BYTE_LENGTH =
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.inputFrame.maxFrameByteLength;
const SOURCE_SNAPSHOT_MAXIMUM_COUNT = 10_000;
const SOURCE_SNAPSHOT_MAXIMUM_BYTE_LENGTH = 64 * 1024 * 1024;
const INPUT_FRAME_MAGIC_BYTES = Uint8Array.of(0x42, 0x47, 0x43, 0x43, 0x41, 0x42, 0x49, 0x31);
const PREPARED_INPUT_FRAMES = new WeakMap<object, PreparedInputFrameRecord>();
const INVOCATION_INPUT_FRAMES = new WeakMap<object, InvocationInputFrameMemo>();
const NATIVE_ARRAY_IS_ARRAY = Array.isArray;
const NATIVE_ARRAY_PROTOTYPE = Array.prototype;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_OBJECT_PROTOTYPE = Object.prototype;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;

interface PreparedInputFrameRecord {
  readonly invocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly bytes: Uint8Array;
}

type InvocationInputFrameMemo =
  | { readonly state: "in-flight"; readonly promise: Promise<PreparedCppCuteBrowserInputFrame> }
  | { readonly state: "prepared"; readonly prepared: WeakRef<PreparedCppCuteBrowserInputFrame> };

declare const preparedInputFrameBrand: unique symbol;

/** Opaque, immutable authority over one exact runtime-v1 input-frame snapshot. */
export interface PreparedCppCuteBrowserInputFrame {
  readonly [preparedInputFrameBrand]: true;
  readonly invocationId: string;
  readonly frameSha256: string;
  readonly frameByteLength: number;
  readonly profileOffset: number;
  readonly profileByteLength: number;
  readonly requestOffset: number;
  readonly requestByteLength: number;
}

export interface CppCuteBrowserInputFrameSourceSnapshot {
  readonly virtualPath: string;
  readonly bytes: Uint8Array;
}

export interface AssembleCppCuteBrowserInputFrameRegionsInput {
  /** Already-canonical runtime profile JSON bytes. */
  readonly profileRegionBytes: Uint8Array;
  /** Already-canonical producer-neutral request JSON bytes. */
  readonly requestRegionBytes: Uint8Array;
  /** Out-of-band source bytes. They are bounded but never embedded in the frame. */
  readonly sourceSnapshots: readonly CppCuteBrowserInputFrameSourceSnapshot[];
  readonly limits: {
    readonly maxFrameByteLength: number;
    readonly maxSourceSnapshotCount: number;
    readonly maxSourceSnapshotByteLength: number;
  };
}

/**
 * Caller-owned byte assembly only. This value is not opaque and proves no
 * invocation identity, Worker execution, source verification, or lowering.
 */
export interface CppCuteBrowserInputFrameRegionAssembly {
  readonly frameBytes: Uint8Array;
  readonly frameByteLength: number;
  readonly profileOffset: number;
  readonly profileByteLength: number;
  readonly requestOffset: number;
  readonly requestByteLength: number;
  readonly sourceSnapshotCount: number;
  readonly sourceSnapshotByteLength: number;
}

export type CppCuteBrowserInputFrameErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-UNVERIFIED";

export class CppCuteBrowserInputFrameError extends Error {
  constructor(
    readonly code: CppCuteBrowserInputFrameErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserInputFrameError";
  }
}

/**
 * Assembles exact runtime-v1 regions without consulting or minting authority.
 * Source snapshots remain out of band; their count and aggregate bytes are
 * checked before any frame allocation.
 */
export function assembleCppCuteBrowserInputFrameRegions(
  input: AssembleCppCuteBrowserInputFrameRegionsInput,
): CppCuteBrowserInputFrameRegionAssembly {
  const values = exactDataRecord(input, "$.input", [
    "profileRegionBytes",
    "requestRegionBytes",
    "sourceSnapshots",
    "limits",
  ]);
  const limits = exactAssemblyLimits(values["limits"]);
  const profile = inspectBytes(
    values["profileRegionBytes"],
    "$.input.profileRegionBytes",
  );
  const request = inspectBytes(
    values["requestRegionBytes"],
    "$.input.requestRegionBytes",
  );
  if (profile.inspection.byteLength > limits.maxFrameByteLength) {
    resource(
      "$.input.profileRegionBytes",
      `profile region exceeds input-frame ceiling ${limits.maxFrameByteLength}`,
    );
  }
  if (request.inspection.byteLength > limits.maxFrameByteLength) {
    resource(
      "$.input.requestRegionBytes",
      `request region exceeds input-frame ceiling ${limits.maxFrameByteLength}`,
    );
  }

  const profileOffset = 64;
  const requestOffset = alignUp(
    checkedAdd(profileOffset, profile.inspection.byteLength),
    8,
  );
  const frameByteLength = alignUp(
    checkedAdd(requestOffset, request.inspection.byteLength),
    8,
  );
  for (const [path, value] of [
    ["$.profileByteLength", profile.inspection.byteLength],
    ["$.requestByteLength", request.inspection.byteLength],
    ["$.requestOffset", requestOffset],
    ["$.frameByteLength", frameByteLength],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0 || value > UINT32_MAX) {
      resource(path, "runtime-v1 input-frame fields must fit unsigned 32-bit integers");
    }
  }
  if (frameByteLength > limits.maxFrameByteLength) {
    resource(
      "$.frameByteLength",
      `input frame length ${frameByteLength} exceeds runtime ABI ceiling ${limits.maxFrameByteLength}`,
    );
  }

  const sourceAccounting = inspectSourceSnapshots(
    values["sourceSnapshots"],
    limits.maxSourceSnapshotCount,
    limits.maxSourceSnapshotByteLength,
  );
  const profileBytes = copyBytes(profile, "$.input.profileRegionBytes");
  const requestBytes = copyBytes(request, "$.input.requestRegionBytes");
  const bytes = new Uint8Array(frameByteLength);
  bytes.set(INPUT_FRAME_MAGIC_BYTES, 0);
  const header = new DataView(bytes.buffer, bytes.byteOffset, 64);
  header.setUint16(8, 1, true);
  header.setUint16(10, 0, true);
  header.setUint32(12, 64, true);
  header.setUint32(16, frameByteLength, true);
  header.setUint32(20, 0, true);
  header.setUint32(24, profileOffset, true);
  header.setUint32(28, profileBytes.byteLength, true);
  header.setUint32(32, requestOffset, true);
  header.setUint32(36, requestBytes.byteLength, true);
  bytes.set(profileBytes, profileOffset);
  bytes.set(requestBytes, requestOffset);

  return NATIVE_OBJECT_FREEZE({
    frameBytes: bytes,
    frameByteLength,
    profileOffset,
    profileByteLength: profileBytes.byteLength,
    requestOffset,
    requestByteLength: requestBytes.byteLength,
    sourceSnapshotCount: sourceAccounting.count,
    sourceSnapshotByteLength: sourceAccounting.byteLength,
  });
}

/**
 * Materializes the fixed runtime-v1 binary frame from one prepared invocation.
 * Source snapshots remain out of band behind the Worker-owned VFS session.
 */
export function prepareCppCuteBrowserInputFrame(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): Promise<PreparedCppCuteBrowserInputFrame> {
  // Recheck liveness even when a memo exists; a resolved frame must never be
  // returned after its single-use invocation has terminalized.
  try {
    unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  } catch (cause) {
    return Promise.reject(cause);
  }
  const memo = INVOCATION_INPUT_FRAMES.get(invocation as object);
  if (memo?.state === "in-flight") return memo.promise;
  if (memo?.state === "prepared") {
    const prepared = memo.prepared.deref();
    if (prepared !== undefined) return Promise.resolve(prepared);
  }
  const promise = materializeCppCuteBrowserInputFrame(invocation);
  const inFlight: InvocationInputFrameMemo = Object.freeze({ state: "in-flight", promise });
  INVOCATION_INPUT_FRAMES.set(invocation as object, inFlight);
  void promise.then(
    (prepared) => {
      if (INVOCATION_INPUT_FRAMES.get(invocation as object) === inFlight) {
        INVOCATION_INPUT_FRAMES.set(
          invocation as object,
          Object.freeze({ state: "prepared", prepared: new WeakRef(prepared) }),
        );
      }
    },
    () => {
      if (INVOCATION_INPUT_FRAMES.get(invocation as object) === inFlight) {
        INVOCATION_INPUT_FRAMES.delete(invocation as object);
      }
    },
  );
  return promise;
}

async function materializeCppCuteBrowserInputFrame(
  invocation: PreparedCppCuteBrowserWorkerInvocation,
): Promise<PreparedCppCuteBrowserInputFrame> {
  const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  const inputFrame = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(
    invocationRecord.runtimeAbi,
  ).manifest.body.inputFrame;
  if (inputFrame.magicAscii !== "BGCCABI1" || inputFrame.major !== 1 || inputFrame.minor !== 0 ||
      inputFrame.headerByteLength !== 64 || inputFrame.alignmentByteLength !== 8) {
    invalid(
      "$.runtimeAbi.inputFrame",
      "prepared runtime ABI does not expose the fixed BGCCABI1 v1.0 64-byte aligned framing contract",
    );
  }
  const profileBytes = canonicalCppCuteBrowserWorkerProfileRegionBytes(invocation);
  const requestBytes = canonicalCppCuteBrowserWorkerRequestRegionBytes(invocation);
  const sourceSnapshots = copyCppCuteBrowserWorkerSourceSnapshots(invocation);
  const assembled = assembleCppCuteBrowserInputFrameRegions({
    profileRegionBytes: profileBytes,
    requestRegionBytes: requestBytes,
    sourceSnapshots,
    limits: {
      maxFrameByteLength: inputFrame.maxFrameByteLength,
      maxSourceSnapshotCount: invocationRecord.request.sourceFileCount,
      maxSourceSnapshotByteLength: Number(invocationRecord.request.sourceByteLength),
    },
  });
  const {
    frameBytes: bytes,
    frameByteLength,
    profileOffset,
    profileByteLength,
    requestOffset,
    requestByteLength,
  } = assembled;

  let frameSha256: string;
  try {
    frameSha256 = await sha256Hex(bytes);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-HASH-UNAVAILABLE",
      "$.frameSha256",
      "SHA-256 is unavailable for the runtime input frame",
      { cause },
    );
  }
  // Hashing yields to the event loop. Do not mint a frame after another path
  // has terminalized the single-use invocation in the meantime.
  unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  const prepared = Object.freeze({
    invocationId: invocation.invocationId,
    frameSha256,
    frameByteLength,
    profileOffset,
    profileByteLength,
    requestOffset,
    requestByteLength,
  }) as PreparedCppCuteBrowserInputFrame;
  PREPARED_INPUT_FRAMES.set(prepared, Object.freeze({ invocation, bytes }));
  return prepared;
}

/** Returns a caller-owned copy; the authority's frame snapshot cannot be mutated. */
export function copyPreparedCppCuteBrowserInputFrameBytes(
  prepared: PreparedCppCuteBrowserInputFrame,
): Uint8Array {
  const record = PREPARED_INPUT_FRAMES.get(prepared as object);
  if (record === undefined) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-UNVERIFIED",
      "$.prepared",
      "value is not an opaque prepared input-frame authority",
    );
  }
  // A prepared frame is usable only while its exact single-use invocation is
  // pending. Terminalization revokes all outstanding frame authorities.
  unwrapPreparedCppCuteBrowserWorkerInvocation(record.invocation);
  return new Uint8Array(record.bytes);
}

interface InspectedBytes {
  readonly value: unknown;
  readonly inspection: InspectedUnsharedUint8Array;
}

function exactAssemblyLimits(value: unknown): {
  readonly maxFrameByteLength: number;
  readonly maxSourceSnapshotCount: number;
  readonly maxSourceSnapshotByteLength: number;
} {
  const limits = exactDataRecord(value, "$.input.limits", [
    "maxFrameByteLength",
    "maxSourceSnapshotCount",
    "maxSourceSnapshotByteLength",
  ]);
  return NATIVE_OBJECT_FREEZE({
    maxFrameByteLength: boundedLimit(
      limits["maxFrameByteLength"],
      "$.input.limits.maxFrameByteLength",
      INPUT_FRAME_MAXIMUM_BYTE_LENGTH,
      false,
    ),
    maxSourceSnapshotCount: boundedLimit(
      limits["maxSourceSnapshotCount"],
      "$.input.limits.maxSourceSnapshotCount",
      SOURCE_SNAPSHOT_MAXIMUM_COUNT,
      true,
    ),
    maxSourceSnapshotByteLength: boundedLimit(
      limits["maxSourceSnapshotByteLength"],
      "$.input.limits.maxSourceSnapshotByteLength",
      SOURCE_SNAPSHOT_MAXIMUM_BYTE_LENGTH,
      true,
    ),
  });
}

function boundedLimit(
  value: unknown,
  path: string,
  maximum: number,
  allowZero: boolean,
): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) ||
      value < (allowZero ? 0 : 1) || value > maximum) {
    invalid(path, `limit must be a ${allowZero ? "nonnegative" : "positive"} safe integer at most ${maximum}`);
  }
  return value;
}

function inspectBytes(value: unknown, path: string): InspectedBytes {
  try {
    return NATIVE_OBJECT_FREEZE({
      value,
      inspection: inspectUnsharedPlainUint8Array(value),
    });
  } catch (cause) {
    invalid(path, "bytes must be an exact unshared plain Uint8Array", { cause });
  }
}

function copyBytes(value: InspectedBytes, path: string): Uint8Array {
  try {
    return copyInspectedUnsharedUint8Array(value.value, value.inspection);
  } catch (cause) {
    invalid(path, "bytes became unreadable while snapshotting", { cause });
  }
}

function inspectSourceSnapshots(
  value: unknown,
  maximumCount: number,
  maximumByteLength: number,
): { readonly count: number; readonly byteLength: number } {
  let prototype: unknown;
  let lengthDescriptor: PropertyDescriptor | undefined;
  try {
    if (!NATIVE_ARRAY_IS_ARRAY(value)) {
      invalid("$.input.sourceSnapshots", "source snapshots must be an array");
    }
    prototype = NATIVE_GET_PROTOTYPE_OF(value);
    lengthDescriptor = NATIVE_GET_OWN_PROPERTY_DESCRIPTOR(value, "length");
  } catch (cause) {
    invalid("$.input.sourceSnapshots", "source snapshot array is not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_ARRAY_PROTOTYPE || lengthDescriptor === undefined ||
      !("value" in lengthDescriptor) || typeof lengthDescriptor.value !== "number" ||
      !Number.isSafeInteger(lengthDescriptor.value) || lengthDescriptor.value < 0) {
    invalid("$.input.sourceSnapshots", "source snapshots must be a dense plain array");
  }
  const count = lengthDescriptor.value;
  if (count > maximumCount) {
    resource(
      "$.input.sourceSnapshots",
      `source snapshot count ${count} exceeds ceiling ${maximumCount}`,
    );
  }

  let keys: readonly PropertyKey[];
  let descriptors: PropertyDescriptorMap;
  try {
    keys = NATIVE_REFLECT_OWN_KEYS(value as object);
    descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(value as object);
  } catch (cause) {
    invalid("$.input.sourceSnapshots", "source snapshot array is not safely inspectable", { cause });
  }
  if (keys.length !== count + 1 || keys.some((key, index) =>
    index < count ? key !== String(index) : key !== "length")) {
    invalid("$.input.sourceSnapshots", "source snapshots must be a dense array without extra properties");
  }

  let byteLength = 0;
  for (let index = 0; index < count; index += 1) {
    const entryDescriptor = descriptors[String(index)];
    if (entryDescriptor === undefined || !("value" in entryDescriptor)) {
      invalid(`$.input.sourceSnapshots[${index}]`, "source snapshot must be a data element");
    }
    const fields = exactDataRecord(entryDescriptor.value, `$.input.sourceSnapshots[${index}]`, [
      "virtualPath",
      "bytes",
    ]);
    if (typeof fields["virtualPath"] !== "string") {
      invalid(`$.input.sourceSnapshots[${index}].virtualPath`, "expected canonical virtual-path string");
    }
    const bytes = inspectBytes(fields["bytes"], `$.input.sourceSnapshots[${index}].bytes`);
    byteLength = checkedSourceAdd(byteLength, bytes.inspection.byteLength);
    if (byteLength > maximumByteLength) {
      resource(
        "$.input.sourceSnapshots",
        `source snapshot bytes exceed ceiling ${maximumByteLength}`,
      );
    }
  }
  return NATIVE_OBJECT_FREEZE({ count, byteLength });
}

function exactDataRecord(
  value: unknown,
  path: string,
  expectedKeys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null) invalid(path, "expected plain data object");
  let prototype: unknown;
  let keys: readonly PropertyKey[];
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(value);
    keys = NATIVE_REFLECT_OWN_KEYS(value);
    descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(value);
  } catch (cause) {
    invalid(path, "object is not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_OBJECT_PROTOTYPE || keys.length !== expectedKeys.length ||
      keys.some((key) => typeof key !== "string" || !expectedKeys.includes(key))) {
    invalid(path, `expected exact data fields: ${expectedKeys.join(", ")}`);
  }
  const result: Record<string, unknown> = {};
  for (const key of expectedKeys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "field must be an own data property");
    }
    result[key] = descriptor.value;
  }
  return NATIVE_OBJECT_FREEZE(result);
}

function checkedSourceAdd(left: number, right: number): number {
  const result = left + right;
  if (!Number.isSafeInteger(result) || result < 0) {
    resource("$.input.sourceSnapshots", "source snapshot byte-length arithmetic overflowed");
  }
  return result;
}

function alignUp(value: number, alignment: number): number {
  if (!Number.isSafeInteger(alignment) || alignment <= 0) {
    invalid("$.runtimeAbi.inputFrame.alignmentByteLength", "alignment must be a positive safe integer");
  }
  const remainder = value % alignment;
  return remainder === 0 ? value : checkedAdd(value, alignment - remainder);
}

function checkedAdd(left: number, right: number): number {
  const result = left + right;
  if (!Number.isSafeInteger(result) || result < 0) {
    resource("$.frameByteLength", "input-frame length arithmetic overflowed");
  }
  return result;
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-INVALID", path, message, options);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-RESOURCE-LIMIT", path, message);
}

function fail(
  code: CppCuteBrowserInputFrameErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserInputFrameError(code, path, message, options);
}

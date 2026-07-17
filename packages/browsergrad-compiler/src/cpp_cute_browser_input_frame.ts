import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  canonicalCppCuteBrowserWorkerProfileRegionBytes,
  canonicalCppCuteBrowserWorkerRequestRegionBytes,
  unwrapPreparedCppCuteBrowserWorkerInvocation,
  type PreparedCppCuteBrowserWorkerInvocation,
} from "./cpp_cute_browser_worker_protocol.js";
import { unwrapPreparedCppCuteBrowserRuntimeAbiManifest } from "./cpp_cute_browser_runtime_abi.js";

const UINT32_MAX = 0xffff_ffff;
const INPUT_FRAME_MAGIC_BYTES = Uint8Array.of(0x42, 0x47, 0x43, 0x43, 0x41, 0x42, 0x49, 0x31);
const PREPARED_INPUT_FRAMES = new WeakMap<object, PreparedInputFrameRecord>();
const INVOCATION_INPUT_FRAMES = new WeakMap<object, InvocationInputFrameMemo>();

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
  const headerByteLength = inputFrame.headerByteLength;
  const alignmentByteLength = inputFrame.alignmentByteLength;
  const profileOffset = headerByteLength;
  const requestOffset = alignUp(checkedAdd(profileOffset, profileBytes.byteLength), alignmentByteLength);
  const frameByteLength = alignUp(checkedAdd(requestOffset, requestBytes.byteLength), alignmentByteLength);

  for (const [path, value] of [
    ["$.profileByteLength", profileBytes.byteLength],
    ["$.requestByteLength", requestBytes.byteLength],
    ["$.requestOffset", requestOffset],
    ["$.frameByteLength", frameByteLength],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0 || value > UINT32_MAX) {
      resource(path, "runtime-v1 input-frame fields must fit unsigned 32-bit integers");
    }
  }
  if (frameByteLength > inputFrame.maxFrameByteLength) {
    resource(
      "$.frameByteLength",
      `input frame length ${frameByteLength} exceeds runtime ABI ceiling ${inputFrame.maxFrameByteLength}`,
    );
  }

  const bytes = new Uint8Array(frameByteLength);
  bytes.set(INPUT_FRAME_MAGIC_BYTES, 0);
  const header = new DataView(bytes.buffer, bytes.byteOffset, headerByteLength);
  header.setUint16(8, inputFrame.major, true);
  header.setUint16(10, inputFrame.minor, true);
  header.setUint32(12, headerByteLength, true);
  header.setUint32(16, frameByteLength, true);
  header.setUint32(20, 0, true);
  header.setUint32(24, profileOffset, true);
  header.setUint32(28, profileBytes.byteLength, true);
  header.setUint32(32, requestOffset, true);
  header.setUint32(36, requestBytes.byteLength, true);
  bytes.set(profileBytes, profileOffset);
  bytes.set(requestBytes, requestOffset);

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
    profileByteLength: profileBytes.byteLength,
    requestOffset,
    requestByteLength: requestBytes.byteLength,
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

function invalid(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-INVALID", path, message);
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

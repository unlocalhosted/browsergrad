import {
  copyPreparedCppCuteBrowserInputFrameBytes,
  type PreparedCppCuteBrowserInputFrame,
} from "./cpp_cute_browser_input_frame.js";
import {
  copyCppCuteBrowserWorkerClangWasmBytes,
  discardCppCuteBrowserWorkerInvocation,
  unwrapPreparedCppCuteBrowserWorkerInvocation,
  type PreparedCppCuteBrowserWorkerInvocation,
} from "./cpp_cute_browser_worker_protocol.js";
import {
  cancelCppCuteBrowserVfsSession,
  createCppCuteBrowserVfsHostImports,
  observeCppCuteBrowserVfsSession,
  type CppCuteBrowserVfsHostImports,
  type PreparedCppCuteBrowserVfsSession,
} from "./cpp_cute_browser_vfs_session.js";

export const CPP_CUTE_BROWSER_WORKER_RUNTIME_PROTOCOL =
  "browsergrad.compiler.cpp-cute.package-worker-runtime@1";
export const CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS =
  "blocked-missing-reviewed-first-build-projections-and-package-emscripten-factory";
export const CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS = Object.freeze([
  "missing-reviewed-first-build-wasm-projections",
  "missing-self-contained-package-worker-bundle-bytes",
  "missing-package-owned-emscripten-factory-bytes",
] as const);

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const NATIVE_UINT8_ARRAY = Uint8Array;
const NATIVE_UINT8_ARRAY_SET = Uint8Array.prototype.set;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_PROTOTYPE = Object.prototype;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const NATIVE_WEAK_MAP_HAS = WeakMap.prototype.has;
const NATIVE_WEAK_MAP_DELETE = WeakMap.prototype.delete;
const NATIVE_REGEXP_TEST = RegExp.prototype.test;
const NATIVE_AGGREGATE_ERROR = AggregateError;
const NATIVE_SUBTLE = globalThis.crypto?.subtle;
const NATIVE_SUBTLE_DIGEST = NATIVE_SUBTLE === undefined
  ? undefined
  : NATIVE_SUBTLE.digest;
const NATIVE_WASM_COMPILE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.compile;
const NATIVE_WASM_INSTANTIATE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.instantiate;
const NATIVE_WASM_MEMORY = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.Memory;

declare const preparedRuntimeBindingBrand: unique symbol;

/**
 * Opaque composition of one verified invocation, one exact input frame, and
 * one active Worker VFS session. It is not Worker execution evidence.
 */
export interface PreparedCppCuteBrowserWorkerRuntimeBinding {
  readonly [preparedRuntimeBindingBrand]: true;
  readonly authority: "package-worker-runtime-binding-only";
  readonly protocol: typeof CPP_CUTE_BROWSER_WORKER_RUNTIME_PROTOCOL;
  readonly invocationId: string;
  readonly requestId: string;
  readonly profileHash: string;
  readonly inputFrameSha256: string;
  readonly inputFrameByteLength: number;
  readonly clangWasmSha256: string;
  readonly clangWasmByteLength: number;
  readonly vfsSessionOrdinal: number;
  readonly bundleStatus: typeof CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS;
  readonly blockers: typeof CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS;
  readonly networkAuthorityGranted: false;
  readonly workerExecutionObserved: false;
  readonly workerTerminationObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface PrepareCppCuteBrowserWorkerRuntimeBindingInput {
  readonly invocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly inputFrame: PreparedCppCuteBrowserInputFrame;
  readonly vfsSession: PreparedCppCuteBrowserVfsSession;
}

export interface CppCuteBrowserWorkerRuntimeBindingInspection {
  readonly state: "prepared" | "blocked-terminal";
  readonly invocationId: string;
  readonly inputFrameByteLength: number;
  readonly clangWasmByteLength: number;
  readonly vfsSessionOrdinal: number;
  readonly nativeIntrinsicSnapshot:
    "byte-copy-hash-wasm-object-inspection-and-authority-bookkeeping";
  readonly requiredWasmConstructionIntrinsicsAvailable: boolean;
  readonly networkAuthorityGranted: false;
  readonly factoryInvoked: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export type CppCuteBrowserWorkerRuntimeErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-DUPLICATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-UNVERIFIED";

export class CppCuteBrowserWorkerRuntimeError extends Error {
  constructor(
    readonly code: CppCuteBrowserWorkerRuntimeErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWorkerRuntimeError";
  }
}

interface ActiveRuntimeBinding {
  readonly invocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly inputFrameBytes: Uint8Array;
  readonly clangWasmBytes: Uint8Array;
  readonly vfsSession: PreparedCppCuteBrowserVfsSession;
  readonly vfsImports: CppCuteBrowserVfsHostImports;
}

interface StoredRuntimeBinding {
  state: "prepared" | "blocked-terminal";
  active: ActiveRuntimeBinding | null;
  readonly inspection: Omit<CppCuteBrowserWorkerRuntimeBindingInspection, "state">;
}

const BINDING_RESERVATIONS = new WeakMap<object, object>();
const FRAME_RESERVATIONS = new WeakMap<object, object>();
const VFS_RESERVATIONS = new WeakMap<object, object>();
const RUNTIME_BINDINGS = new WeakMap<object, StoredRuntimeBinding>();

export async function prepareCppCuteBrowserWorkerRuntimeBinding(
  input: PrepareCppCuteBrowserWorkerRuntimeBindingInput,
): Promise<PreparedCppCuteBrowserWorkerRuntimeBinding> {
  const values = exactDataRecord(input, "$.input", ["invocation", "inputFrame", "vfsSession"]);
  const invocation = values["invocation"] as PreparedCppCuteBrowserWorkerInvocation;
  const inputFrame = values["inputFrame"] as PreparedCppCuteBrowserInputFrame;
  const vfsSession = values["vfsSession"] as PreparedCppCuteBrowserVfsSession;
  const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
  // Validate the opaque frame before reading any of its public projection.
  // A forged object may carry hostile getters despite its TypeScript type.
  const initialInputFrameBytes = copyPreparedCppCuteBrowserInputFrameBytes(inputFrame);
  const vfsObservation = observeCppCuteBrowserVfsSession(vfsSession);
  if (inputFrame.invocationId !== invocation.invocationId) {
    mismatch("$.input.inputFrame", "input frame belongs to a different prepared invocation");
  }
  if (vfsObservation.state !== "active") {
    state("$.input.vfsSession", "VFS session must be active when the runtime binding is prepared");
  }
  if (vfsObservation.requestId !== invocation.requestId ||
      vfsObservation.profileHash !== invocation.profileHash) {
    mismatch("$.input.vfsSession", "VFS session differs from the exact invocation request or profile");
  }

  const reservation = NATIVE_OBJECT_FREEZE({});
  reserve(BINDING_RESERVATIONS, invocation as object, reservation, "$.input.invocation");
  try {
    reserve(FRAME_RESERVATIONS, inputFrame as object, reservation, "$.input.inputFrame");
    reserve(VFS_RESERVATIONS, vfsSession as object, reservation, "$.input.vfsSession");
  } catch (cause) {
    releaseReservation(BINDING_RESERVATIONS, invocation as object, reservation);
    releaseReservation(FRAME_RESERVATIONS, inputFrame as object, reservation);
    throw cause;
  }

  try {
    const inputFrameBytes = capturedCopy(initialInputFrameBytes);
    const clangWasmBytes = capturedCopy(copyCppCuteBrowserWorkerClangWasmBytes(invocation));
    const inputFrameSha256 = await nativeSha256Hex(
      inputFrameBytes,
      "$.input.inputFrame",
    );
    const clangWasmSha256 = await nativeSha256Hex(
      clangWasmBytes,
      "$.input.invocation.clangWasmBytes",
    );
    if (inputFrameBytes.byteLength !== inputFrame.frameByteLength ||
        inputFrameSha256 !== inputFrame.frameSha256) {
      mismatch("$.input.inputFrame", "copied input-frame bytes differ from the opaque frame authority");
    }
    if (!nativeRegexTest(SHA256_HEX, invocationRecord.rawWasmConformance.wasmSha256) ||
        clangWasmBytes.byteLength !== invocationRecord.rawWasmConformance.wasmByteLength ||
        clangWasmSha256 !== invocationRecord.rawWasmConformance.wasmSha256) {
      mismatch("$.input.invocation.clangWasmBytes", "copied Clang-Wasm bytes differ from raw-Wasm conformance authority");
    }

    // Hashing yields. Recheck every live opaque authority before minting the
    // composition so terminalization cannot race a stale runtime binding.
    unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
    copyPreparedCppCuteBrowserInputFrameBytes(inputFrame);
    if (observeCppCuteBrowserVfsSession(vfsSession).state !== "active") {
      state("$.input.vfsSession", "VFS session terminalized while runtime bytes were hashed");
    }
    const vfsImports = createCppCuteBrowserVfsHostImports(vfsSession);
    const prepared = NATIVE_OBJECT_FREEZE({
      authority: "package-worker-runtime-binding-only",
      protocol: CPP_CUTE_BROWSER_WORKER_RUNTIME_PROTOCOL,
      invocationId: invocation.invocationId,
      requestId: invocation.requestId,
      profileHash: invocation.profileHash,
      inputFrameSha256,
      inputFrameByteLength: inputFrameBytes.byteLength,
      clangWasmSha256,
      clangWasmByteLength: clangWasmBytes.byteLength,
      vfsSessionOrdinal: vfsSession.sessionOrdinal,
      bundleStatus: CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS,
      blockers: CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS,
      networkAuthorityGranted: false,
      workerExecutionObserved: false,
      workerTerminationObserved: false,
      loweringAuthorityMinted: false,
    }) as PreparedCppCuteBrowserWorkerRuntimeBinding;
    weakMapSet(RUNTIME_BINDINGS, prepared, {
      state: "prepared",
      active: NATIVE_OBJECT_FREEZE({
        invocation,
        inputFrameBytes,
        clangWasmBytes,
        vfsSession,
        vfsImports,
      }),
      inspection: NATIVE_OBJECT_FREEZE({
        invocationId: invocation.invocationId,
        inputFrameByteLength: inputFrameBytes.byteLength,
        clangWasmByteLength: clangWasmBytes.byteLength,
        vfsSessionOrdinal: vfsSession.sessionOrdinal,
        nativeIntrinsicSnapshot:
          "byte-copy-hash-wasm-object-inspection-and-authority-bookkeeping",
        requiredWasmConstructionIntrinsicsAvailable: NATIVE_WASM_COMPILE !== undefined &&
          NATIVE_WASM_INSTANTIATE !== undefined && NATIVE_WASM_MEMORY !== undefined,
        networkAuthorityGranted: false,
        factoryInvoked: false,
        workerExecutionObserved: false,
        loweringAuthorityMinted: false,
      }),
    });
    return prepared;
  } catch (cause) {
    releaseReservation(BINDING_RESERVATIONS, invocation as object, reservation);
    releaseReservation(FRAME_RESERVATIONS, inputFrame as object, reservation);
    releaseReservation(VFS_RESERVATIONS, vfsSession as object, reservation);
    throw cause;
  }
}

/**
 * Production runtime start remains deliberately unreachable until real,
 * internally pinned bundle/factory bytes and reviewed first-build projections
 * replace the blocker constants. No ambient resource-acquisition or
 * caller-supplied factory seam exists here.
 */
export async function startCppCuteBrowserWorkerRuntime(
  binding: PreparedCppCuteBrowserWorkerRuntimeBinding,
): Promise<never> {
  const stored = storedBinding(binding);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.binding", "runtime binding already reached its terminal blocked state");
  }
  const active = stored.active;
  stored.state = "blocked-terminal";
  stored.active = null;
  let vfsCleanupFailed = false;
  let invocationCleanupFailed = false;
  let vfsCleanupCause: unknown;
  let invocationCleanupCause: unknown;
  try {
    cancelCppCuteBrowserVfsSession(active.vfsSession);
  } catch (cause) {
    vfsCleanupFailed = true;
    vfsCleanupCause = cause;
  }
  try {
    discardCppCuteBrowserWorkerInvocation(active.invocation, "worker-unavailable");
  } catch (cause) {
    invocationCleanupFailed = true;
    invocationCleanupCause = cause;
  }
  if (vfsCleanupFailed || invocationCleanupFailed) {
    const cleanupCauses = vfsCleanupFailed && invocationCleanupFailed
      ? [vfsCleanupCause, invocationCleanupCause]
      : [vfsCleanupFailed ? vfsCleanupCause : invocationCleanupCause];
    cleanup(
      "$.binding.cleanup",
      "blocked runtime cleanup failed after both owned-authority cleanup attempts completed",
      new NATIVE_AGGREGATE_ERROR(cleanupCauses, "package Worker runtime cleanup failures"),
    );
  }
  capability(
    "$.bundle",
    "package Worker runtime is blocked: missing-reviewed-first-build-wasm-projections, " +
      "missing-self-contained-package-worker-bundle-bytes, " +
      "missing-package-owned-emscripten-factory-bytes",
  );
}

export function inspectCppCuteBrowserWorkerRuntimeBinding(
  binding: PreparedCppCuteBrowserWorkerRuntimeBinding,
): CppCuteBrowserWorkerRuntimeBindingInspection {
  const stored = storedBinding(binding);
  return NATIVE_OBJECT_FREEZE({ state: stored.state, ...stored.inspection });
}

function capturedCopy(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const copy = new NATIVE_UINT8_ARRAY(bytes.byteLength);
  NATIVE_REFLECT_APPLY(NATIVE_UINT8_ARRAY_SET, copy, [bytes]);
  return copy;
}

async function nativeSha256Hex(bytes: Uint8Array, path: string): Promise<string> {
  if (NATIVE_SUBTLE_DIGEST === undefined) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-HASH-UNAVAILABLE",
      path,
      "captured native SubtleCrypto SHA-256 is unavailable",
    );
  }
  let digest: ArrayBuffer;
  try {
    digest = await NATIVE_REFLECT_APPLY(
      NATIVE_SUBTLE_DIGEST,
      NATIVE_SUBTLE,
      ["SHA-256", capturedCopy(bytes)],
    ) as ArrayBuffer;
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-HASH-UNAVAILABLE",
      path,
      "captured native SubtleCrypto SHA-256 failed",
      { cause },
    );
  }
  const digestBytes = new NATIVE_UINT8_ARRAY(digest);
  if (digestBytes.byteLength !== 32) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-HASH-UNAVAILABLE",
      path,
      "captured native SubtleCrypto SHA-256 returned the wrong digest length",
    );
  }
  const hexDigits = "0123456789abcdef";
  let hex = "";
  for (let index = 0; index < digestBytes.byteLength; index += 1) {
    const value = digestBytes[index]!;
    hex += hexDigits[value >>> 4]! + hexDigits[value & 0x0f]!;
  }
  return hex;
}

function reserve(
  reservations: WeakMap<object, object>,
  authority: object,
  reservation: object,
  path: string,
): void {
  if (weakMapHas(reservations, authority)) {
    duplicate(path, "authority is already bound to a Worker runtime");
  }
  weakMapSet(reservations, authority, reservation);
}

function releaseReservation(
  reservations: WeakMap<object, object>,
  authority: object,
  reservation: object,
): void {
  if (weakMapGet(reservations, authority) === reservation) {
    weakMapDelete(reservations, authority);
  }
}

function storedBinding(
  binding: PreparedCppCuteBrowserWorkerRuntimeBinding,
): StoredRuntimeBinding {
  if (typeof binding !== "object" || binding === null) unverified("$.binding");
  const stored = weakMapGet(RUNTIME_BINDINGS, binding as object);
  if (stored === undefined) unverified("$.binding");
  return stored;
}

function exactDataRecord(
  value: unknown,
  path: string,
  expectedKeys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null) invalid(path, "expected a plain data record");
  let prototype: object | null;
  let keys: readonly PropertyKey[];
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(value);
    keys = NATIVE_REFLECT_OWN_KEYS(value);
    descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(value);
  } catch (cause) {
    invalid(path, "input record is not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_OBJECT_PROTOTYPE && prototype !== null) {
    invalid(path, "expected a plain data record");
  }
  const expectedKeyList = formatExpectedKeys(expectedKeys);
  if (keys.length !== expectedKeys.length) {
    invalid(path, `expected exactly ${expectedKeyList}`);
  }
  for (const key of keys) {
    if (typeof key !== "string" || !containsString(expectedKeys, key)) {
      invalid(path, `expected exactly ${expectedKeyList}`);
    }
  }
  const result: Record<string, unknown> = {};
  for (const key of expectedKeys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "expected an enumerable data property without accessors");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function weakMapHas<K extends object, V>(map: WeakMap<K, V>, key: K): boolean {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_HAS, map, [key]) as boolean;
}

function weakMapDelete<K extends object, V>(map: WeakMap<K, V>, key: K): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_DELETE, map, [key]);
}

function nativeRegexTest(pattern: RegExp, value: string): boolean {
  return NATIVE_REFLECT_APPLY(NATIVE_REGEXP_TEST, pattern, [value]) as boolean;
}

function containsString(values: readonly string[], expected: string): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] === expected) return true;
  }
  return false;
}

function formatExpectedKeys(values: readonly string[]): string {
  let result = "";
  for (let index = 0; index < values.length; index += 1) {
    if (index > 0) result += ", ";
    result += values[index];
  }
  return result;
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-INVALID", path, message, options);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH", path, message);
}

function duplicate(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-DUPLICATE", path, message);
}

function capability(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY", path, message);
}

function cleanup(path: string, message: string, cause: AggregateError): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CLEANUP",
    path,
    message,
    { cause },
  );
}

function state(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE", path, message, options);
}

function unverified(path: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-UNVERIFIED",
    path,
    "value is not an opaque package Worker runtime authority",
  );
}

function fail(
  code: CppCuteBrowserWorkerRuntimeErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserWorkerRuntimeError(code, path, message, options);
}

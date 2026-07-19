import {
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  copyPreparedCppCuteBrowserInputFrameBytes,
} from "./cpp_cute_browser_input_frame.js";
import {
  discardCppCuteBrowserEmscriptenFactory,
  inspectCppCuteBrowserEmscriptenFactory,
  prepareCppCuteBrowserEmscriptenFactory,
  type PreparedCppCuteBrowserEmscriptenFactory,
} from "./cpp_cute_browser_emscripten_factory.js";
import {
  CPP_CUTE_BROWSER_GENERATED_FACTORY,
} from "./cpp_cute_browser_generated_factory.js";
import {
  buildCanonicalCppCuteBrowserWorkerResultControl,
  consumeCppCuteBrowserWorkerInvocationResultControl,
  discardCppCuteBrowserWorkerInvocation,
  unwrapPreparedCppCuteBrowserWorkerInvocation,
  type CppCuteBrowserWorkerInvocationDiscardReason,
  type PreparedCppCuteBrowserWorkerInvocation,
} from "./cpp_cute_browser_worker_protocol.js";
import {
  createCppCuteBrowserVfsMountHostImports,
  discardCppCuteBrowserVfsMount,
  observeCppCuteBrowserVfsMount,
  type CppCuteBrowserVfsHostImports,
  type PreparedCppCuteBrowserVfsMount,
} from "./cpp_cute_browser_vfs_session.js";
import {
  executeCppCuteBrowserWasmCompiler,
} from "./cpp_cute_browser_wasm_compiler.js";
import {
  takeCppCuteBrowserWorkerRealmInput,
  type PreparedCppCuteBrowserWorkerRealmInput,
} from "./cpp_cute_browser_worker_transfer.js";

export const CPP_CUTE_BROWSER_WORKER_RUNTIME_PROTOCOL =
  "browsergrad.compiler.cpp-cute.package-worker-runtime@1";
export const CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS =
  "blocked-missing-self-contained-bundle";
export const CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS = Object.freeze([
  "missing-self-contained-package-worker-bundle-bytes",
] as const);

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const NATIVE_UINT8_ARRAY = Uint8Array;
const NATIVE_UINT8_ARRAY_SET = Uint8Array.prototype.set;
const NATIVE_UINT8_ARRAY_FILL = Uint8Array.prototype.fill;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_PROTOTYPE = Object.prototype;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
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
 * Opaque composition adopted inside the Worker realm before Wasm memory exists.
 * It is not Worker execution evidence.
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
  readonly vfsMountOrdinal: number;
  readonly bundleStatus: typeof CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS;
  readonly blockers: typeof CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS;
  readonly networkAuthorityGranted: false;
  readonly workerExecutionObserved: false;
  readonly workerTerminationObserved: false;
  readonly loweringAuthorityMinted: false;
}

export interface PrepareCppCuteBrowserWorkerRuntimeBindingInput {
  readonly realmInput: PreparedCppCuteBrowserWorkerRealmInput;
}

/**
 * One successful runtime terminal result. The entrypoint snapshots these bytes
 * into dedicated transferable buffers before handing them to the controller.
 */
export interface CppCuteBrowserWorkerRuntimeResult {
  readonly kind: "browsergrad-cpp-cute-runtime-result";
  readonly controlBytes: Uint8Array;
  readonly artifactBytes: Uint8Array;
}

export interface CppCuteBrowserWorkerRuntimeBindingInspection {
  readonly state: "prepared" | "execution-blocked-terminal" | "discarded";
  readonly invocationId: string;
  readonly inputFrameByteLength: number;
  readonly clangWasmByteLength: number;
  readonly vfsMountOrdinal: number;
  readonly nativeIntrinsicSnapshot:
    "byte-copy-hash-wasm-object-inspection-and-authority-bookkeeping";
  readonly requiredWasmConstructionIntrinsicsAvailable: boolean;
  readonly networkAuthorityGranted: false;
  readonly factoryInvoked: boolean;
  readonly cAbiExecutionObserved: boolean;
  readonly artifactVerificationObserved: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export type CppCuteBrowserWorkerRuntimeErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-EXECUTION"
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
  readonly vfsMount: PreparedCppCuteBrowserVfsMount;
  readonly vfsImports: CppCuteBrowserVfsHostImports;
}

interface StoredRuntimeBinding {
  state: "prepared" | "execution-blocked-terminal" | "discarded";
  active: ActiveRuntimeBinding | null;
  readonly inspection: Omit<
    CppCuteBrowserWorkerRuntimeBindingInspection,
    "state" | "factoryInvoked" | "cAbiExecutionObserved"
  >;
  factoryInvoked: boolean;
  cAbiExecutionObserved: boolean;
}

const RUNTIME_BINDINGS = new WeakMap<object, StoredRuntimeBinding>();

export async function prepareCppCuteBrowserWorkerRuntimeBinding(
  input: PrepareCppCuteBrowserWorkerRuntimeBindingInput,
): Promise<PreparedCppCuteBrowserWorkerRuntimeBinding> {
  const values = exactDataRecord(input, "$.input", ["realmInput"]);
  const realmInput = values["realmInput"] as PreparedCppCuteBrowserWorkerRealmInput;
  const adopted = takeCppCuteBrowserWorkerRealmInput(realmInput);
  const invocation = adopted.invocation;
  const inputFrame = adopted.inputFrame;
  const vfsMount = adopted.vfsMount;
  let inputFrameBytes: Uint8Array = new NATIVE_UINT8_ARRAY(0);
  let clangWasmBytes: Uint8Array = new NATIVE_UINT8_ARRAY(0);

  try {
    const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
    inputFrameBytes = copyPreparedCppCuteBrowserInputFrameBytes(inputFrame);
    clangWasmBytes = adopted.clangWasmBytes;
    const vfsObservation = observeCppCuteBrowserVfsMount(vfsMount);
    if (inputFrame.invocationId !== invocation.invocationId ||
        realmInput.invocationId !== invocation.invocationId) {
      mismatch("$.input.realmInput.inputFrame", "input frame belongs to a different invocation");
    }
    if (vfsObservation.state !== "prepared") {
      state("$.input.realmInput.vfsMount", "VFS mount must be unbound during runtime adoption");
    }
    if (vfsObservation.requestId !== invocation.requestId ||
        vfsObservation.profileHash !== invocation.profileHash ||
        vfsObservation.mountOrdinal !== realmInput.vfsMountOrdinal ||
        realmInput.requestId !== invocation.requestId ||
        realmInput.profileHash !== invocation.profileHash) {
      mismatch(
        "$.input.realmInput.vfsMount",
        "VFS mount differs from the exact invocation request or profile",
      );
    }
    const inputFrameSha256 = await nativeSha256Hex(
      inputFrameBytes,
      "$.input.inputFrame",
    );
    const clangWasmSha256 = await nativeSha256Hex(
      clangWasmBytes,
      "$.input.invocation.clangWasmBytes",
    );
    if (inputFrameBytes.byteLength !== inputFrame.frameByteLength ||
        inputFrameSha256 !== inputFrame.frameSha256 ||
        inputFrameBytes.byteLength !== realmInput.inputFrameByteLength ||
        inputFrameSha256 !== realmInput.inputFrameSha256) {
      mismatch("$.input.inputFrame", "copied input-frame bytes differ from the opaque frame authority");
    }
    if (!nativeRegexTest(SHA256_HEX, invocationRecord.rawWasmConformance.wasmSha256) ||
        clangWasmBytes.byteLength !== invocationRecord.rawWasmConformance.wasmByteLength ||
        clangWasmSha256 !== invocationRecord.rawWasmConformance.wasmSha256 ||
        clangWasmBytes.byteLength !== realmInput.clangWasmByteLength ||
        clangWasmSha256 !== realmInput.clangWasmSha256) {
      mismatch("$.input.invocation.clangWasmBytes", "copied Clang-Wasm bytes differ from raw-Wasm conformance authority");
    }

    // Hashing yields. Recheck every live opaque authority before minting the
    // composition so terminalization cannot race a stale runtime binding.
    unwrapPreparedCppCuteBrowserWorkerInvocation(invocation);
    copyPreparedCppCuteBrowserInputFrameBytes(inputFrame);
    if (observeCppCuteBrowserVfsMount(vfsMount).state !== "prepared") {
      state("$.input.realmInput.vfsMount", "VFS mount terminalized while runtime bytes were hashed");
    }
    const vfsImports = createCppCuteBrowserVfsMountHostImports(vfsMount);
    if (vfsImports !== adopted.vfsImports) {
      mismatch("$.input.realmInput.vfsImports", "VFS import table differs from reconstructed authority");
    }
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
      vfsMountOrdinal: vfsMount.mountOrdinal,
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
        vfsMount,
        vfsImports,
      }),
      inspection: NATIVE_OBJECT_FREEZE({
        invocationId: invocation.invocationId,
        inputFrameByteLength: inputFrameBytes.byteLength,
        clangWasmByteLength: clangWasmBytes.byteLength,
        vfsMountOrdinal: vfsMount.mountOrdinal,
        nativeIntrinsicSnapshot:
          "byte-copy-hash-wasm-object-inspection-and-authority-bookkeeping",
        requiredWasmConstructionIntrinsicsAvailable: NATIVE_WASM_COMPILE !== undefined &&
          NATIVE_WASM_INSTANTIATE !== undefined && NATIVE_WASM_MEMORY !== undefined,
        networkAuthorityGranted: false,
        workerExecutionObserved: false,
        artifactVerificationObserved: false,
        loweringAuthorityMinted: false,
      }),
      factoryInvoked: false,
      cAbiExecutionObserved: false,
    });
    return prepared;
  } catch (cause) {
    cleanupAdoptedRuntimeInput(
      { invocation, inputFrameBytes, clangWasmBytes,
        vfsMount, vfsImports: adopted.vfsImports },
      cause,
      "preparation",
      "abandoned",
    );
    throw cause;
  }
}

/** Executes the package-pinned factory and C ABI and emits canonical control. */
export async function startCppCuteBrowserWorkerRuntime(
  binding: PreparedCppCuteBrowserWorkerRuntimeBinding,
): Promise<CppCuteBrowserWorkerRuntimeResult> {
  const stored = storedBinding(binding);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.binding", "runtime binding already reached its terminal execution state");
  }
  const active = stored.active;
  stored.state = "execution-blocked-terminal";
  stored.active = null;
  let artifactBytes: Uint8Array | undefined;
  let preparedFactory: PreparedCppCuteBrowserEmscriptenFactory | undefined;
  try {
    const invocationRecord = unwrapPreparedCppCuteBrowserWorkerInvocation(active.invocation);
    preparedFactory = await prepareCppCuteBrowserEmscriptenFactory({
      factory: CPP_CUTE_BROWSER_GENERATED_FACTORY,
      clangWasmBytes: active.clangWasmBytes,
      expectedWasmSha256: binding.clangWasmSha256,
      expectedWasmByteLength: binding.clangWasmByteLength,
      vfsMount: active.vfsMount,
    });
    stored.factoryInvoked = true;
    const execution = executeCppCuteBrowserWasmCompiler({
      factory: preparedFactory,
      profile: invocationRecord.profile,
      inputFrameBytes: active.inputFrameBytes,
    });
    artifactBytes = execution.artifactBytes;
    let artifactByteLength: number;
    try {
      artifactByteLength = inspectUnsharedPlainUint8Array(artifactBytes).byteLength;
    } catch (cause) {
      invalid(
        "$.runtime.execution.artifactBytes",
        "local C ABI execution returned a non-plain or shared artifact view",
        { cause },
      );
    }
    if (execution.authority !== "wasm-c-abi-local-execution-only" ||
        execution.profileHash !== binding.profileHash ||
        execution.wasmSha256 !== binding.clangWasmSha256 ||
        execution.wasmByteLength !== binding.clangWasmByteLength ||
        execution.inputFrameByteLength !== binding.inputFrameByteLength ||
        execution.resultByteLength !== artifactByteLength ||
        execution.compileStatus.code !== 0 ||
        execution.cAbiExecutionObserved !== true ||
        execution.artifactVerificationObserved !== false ||
        execution.workerExecutionObserved !== false ||
        execution.loweringAuthorityMinted !== false) {
      mismatch(
        "$.runtime.execution",
        "local C ABI execution projection differs from the exact runtime binding",
      );
    }
    stored.cAbiExecutionObserved = true;
    const controlBytes = await buildCanonicalCppCuteBrowserWorkerResultControl(
      active.invocation,
      execution,
    );
    const result = NATIVE_OBJECT_FREEZE({
      kind: "browsergrad-cpp-cute-runtime-result" as const,
      controlBytes,
      artifactBytes,
    });
    consumeCppCuteBrowserWorkerInvocationResultControl(active.invocation);
    zeroBytes(active.inputFrameBytes);
    zeroBytes(active.clangWasmBytes);
    return result;
  } catch (cause) {
    const startCause = settlePreparedFactoryAfterFailedStart(preparedFactory, cause);
    zeroBytes(artifactBytes);
    cleanupAdoptedRuntimeInput(
      active,
      startCause,
      "failed C ABI start",
      "worker-unavailable",
    );
    executionFailure(
      "$.runtime.execution",
      "package-generated factory or exact C ABI execution failed",
      startCause,
    );
  }
}

function settlePreparedFactoryAfterFailedStart(
  prepared: PreparedCppCuteBrowserEmscriptenFactory | undefined,
  primaryCause: unknown,
): unknown {
  if (prepared === undefined) return primaryCause;
  try {
    if (inspectCppCuteBrowserEmscriptenFactory(prepared).state === "prepared") {
      discardCppCuteBrowserEmscriptenFactory(prepared);
    }
    return primaryCause;
  } catch (cause) {
    return new NATIVE_AGGREGATE_ERROR(
      [primaryCause, cause],
      "package-generated factory cleanup failed after Worker runtime start failure",
    );
  }
}

/** Abandons a prepared runtime binding without attempting Worker execution. */
export function discardCppCuteBrowserWorkerRuntimeBinding(
  binding: PreparedCppCuteBrowserWorkerRuntimeBinding,
): void {
  const stored = storedBinding(binding);
  if (stored.state !== "prepared" || stored.active === null) {
    state("$.binding", "only a prepared runtime binding may be discarded");
  }
  const active = stored.active;
  stored.state = "discarded";
  stored.active = null;
  cleanupAdoptedRuntimeInput(active, undefined, "abandoned binding", "abandoned");
}

function cleanupAdoptedRuntimeInput(
  active: ActiveRuntimeBinding,
  primaryCause: unknown,
  phase: string,
  discardReason: CppCuteBrowserWorkerInvocationDiscardReason,
): void {
  const cleanupCauses: unknown[] = [];
  try {
    if (observeCppCuteBrowserVfsMount(active.vfsMount).state === "prepared") {
      discardCppCuteBrowserVfsMount(active.vfsMount);
    }
  } catch (cause) {
    cleanupCauses.push(cause);
  }
  try {
    discardCppCuteBrowserWorkerInvocation(active.invocation, discardReason);
  } catch (cause) {
    cleanupCauses.push(cause);
  }
  zeroBytes(active.inputFrameBytes);
  zeroBytes(active.clangWasmBytes);
  if (cleanupCauses.length !== 0) {
    const causes = primaryCause === undefined
      ? cleanupCauses
      : [primaryCause, ...cleanupCauses];
    cleanup(
      "$.binding.cleanup",
      `${phase} cleanup failed after both owned-authority cleanup attempts completed`,
      new NATIVE_AGGREGATE_ERROR(causes, "package Worker runtime cleanup failures"),
    );
  }
}

function zeroBytes(bytes: Uint8Array | undefined): void {
  if (bytes === undefined) return;
  try {
    NATIVE_REFLECT_APPLY(NATIVE_UINT8_ARRAY_FILL, bytes, [0]);
  } catch {
    // Terminal cleanup is best effort after authority was already severed.
  }
}

export function inspectCppCuteBrowserWorkerRuntimeBinding(
  binding: PreparedCppCuteBrowserWorkerRuntimeBinding,
): CppCuteBrowserWorkerRuntimeBindingInspection {
  const stored = storedBinding(binding);
  return NATIVE_OBJECT_FREEZE({
    state: stored.state,
    ...stored.inspection,
    factoryInvoked: stored.factoryInvoked,
    cAbiExecutionObserved: stored.cAbiExecutionObserved,
  });
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

function executionFailure(path: string, message: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-EXECUTION",
    path,
    message,
    { cause },
  );
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

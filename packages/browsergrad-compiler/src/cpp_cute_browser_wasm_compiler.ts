import {
  wireIntegerToBigInt,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  takeCppCuteBrowserEmscriptenFactory,
  type PreparedCppCuteBrowserEmscriptenFactory,
  type TakenCppCuteBrowserEmscriptenFactory,
} from "./cpp_cute_browser_emscripten_factory.js";
import {
  beginCppCuteBrowserWasmRuntimePhase,
  cancelCppCuteBrowserWasmRuntimeMetrics,
  closeCppCuteBrowserWasmRuntimeMetrics,
  completeCppCuteBrowserWasmRuntimePhase,
  CppCuteBrowserWasmRuntimeMetricsError,
  prepareCppCuteBrowserWasmRuntimeMetrics,
  type CppCuteBrowserWasmRuntimeObservationV1,
  type PreparedCppCuteBrowserWasmRuntimeMetrics,
} from "./cpp_cute_browser_wasm_runtime_metrics.js";
import {
  closeCppCuteBrowserVfsSession,
  observeCppCuteBrowserVfsSession,
  unwrapClosedCppCuteBrowserVfsSession,
  type CppCuteBrowserVfsSessionObservation,
} from "./cpp_cute_browser_vfs_session.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import { CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE } from "./resources/cpp_cute_browser_runtime_abi_v1.js";

export const CPP_CUTE_BROWSER_WASM_COMPILER_PROTOCOL =
  "browsergrad.compiler.cpp-cute.wasm-c-abi-execution@1";

const RUNTIME_ABI = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body;
const ARTIFACT_READY_STATUS = 0;
const IDLE_STATUS = 1;
const INPUT_ALLOCATED_STATUS = 2;
const MAX_INPUT_FRAME_BYTE_LENGTH = RUNTIME_ABI.inputFrame.maxFrameByteLength;
const MAX_RESULT_BYTE_LENGTH = RUNTIME_ABI.result.maximumByteLength;
const U32_LIMIT = 0x1_0000_0000n;
const NATIVE_OBJECT_PROTOTYPE = Object.prototype;
const NATIVE_ARRAY_BUFFER_PROTOTYPE = ArrayBuffer.prototype;
const NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  NATIVE_ARRAY_BUFFER_PROTOTYPE,
  "byteLength",
)?.get;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_UINT8_ARRAY = Uint8Array;
const NATIVE_UINT8_ARRAY_SET = Uint8Array.prototype.set;
const NATIVE_UINT8_ARRAY_FILL = Uint8Array.prototype.fill;
const NATIVE_ARRAY_PUSH = Array.prototype.push;
const NATIVE_AGGREGATE_ERROR = AggregateError;
const NATIVE_BIGINT = BigInt;
const NATIVE_NUMBER_IS_INTEGER = Number.isInteger;
const NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.Memory.prototype;
const MEMORY_BUFFER_GETTER = NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE === undefined
  ? undefined
  : Object.getOwnPropertyDescriptor(NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE, "buffer")?.get;

export interface ExecuteCppCuteBrowserWasmCompilerInput {
  /** Single-use, package-generated factory authority. */
  readonly factory: PreparedCppCuteBrowserEmscriptenFactory;
  readonly profile: PreparedCppCuteFrontendProfile;
  /** Exact runtime-v1 frame copied into module-owned input storage. */
  readonly inputFrameBytes: Uint8Array;
}

/**
 * One local C-ABI execution observation. The artifact remains unverified and
 * this value mints neither Worker-execution nor lowering authority.
 */
export interface CppCuteBrowserWasmCompilerExecution {
  readonly authority: "wasm-c-abi-local-execution-only";
  readonly protocol: typeof CPP_CUTE_BROWSER_WASM_COMPILER_PROTOCOL;
  readonly profileHash: string;
  readonly wasmSha256: string;
  readonly wasmByteLength: number;
  readonly inputFrameByteLength: number;
  readonly resultByteLength: number;
  readonly compileStatus: {
    readonly code: 0;
    readonly name: "artifact-ready";
  };
  /** Caller-owned copy; canonical Artifact V3 verification is a later seam. */
  readonly artifactBytes: Uint8Array;
  readonly runtime: CppCuteBrowserWasmRuntimeObservationV1;
  readonly vfs: CppCuteBrowserVfsSessionObservation;
  readonly cAbiExecutionObserved: true;
  readonly artifactVerificationObserved: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
}

export type CppCuteBrowserWasmCompilerErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-CLEANUP"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-COMPILE-STATUS"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-TRAP";

export class CppCuteBrowserWasmCompilerError extends Error {
  constructor(
    readonly code: CppCuteBrowserWasmCompilerErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWasmCompilerError";
  }
}

interface ExecutionState {
  readonly taken: TakenCppCuteBrowserEmscriptenFactory;
  readonly frameBytes: Uint8Array;
  metrics: PreparedCppCuteBrowserWasmRuntimeMetrics | undefined;
  metricsClosed: boolean;
  vfsClosed: boolean;
}

interface MemoryEpoch {
  readonly memory: WebAssembly.Memory;
  readonly buffer: ArrayBuffer;
  readonly byteLength: number;
}

interface InputFrameSnapshot {
  readonly bytes: Uint8Array;
  readonly byteLength: number;
}

/**
 * Executes the exact exported runtime ABI synchronously. Dedicated-Worker
 * termination remains the only supported in-flight cancellation mechanism.
 */
export function executeCppCuteBrowserWasmCompiler(
  input: ExecuteCppCuteBrowserWasmCompilerInput,
): CppCuteBrowserWasmCompilerExecution {
  const values = exactDataRecord(input, "$.input", [
    "factory", "profile", "inputFrameBytes",
  ]);
  const factory = values["factory"] as PreparedCppCuteBrowserEmscriptenFactory;
  const profile = values["profile"] as PreparedCppCuteFrontendProfile;
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  if (profileRecord.profile.deployment.mode !== "browser-local") {
    invalid("$.input.profile", "Wasm execution requires one browser-local profile");
  }
  const frame = exactFrameCopy(values["inputFrameBytes"]);
  const frameBytes = frame.bytes;
  let taken: TakenCppCuteBrowserEmscriptenFactory;
  try {
    taken = takeCppCuteBrowserEmscriptenFactory(factory);
  } catch (cause) {
    zeroBytes(frameBytes);
    throw cause;
  }
  const state: ExecutionState = {
    taken,
    frameBytes,
    metrics: undefined,
    metricsClosed: false,
    vfsClosed: false,
  };

  try {
    if (taken.vfsSession.profileHash !== profile.profileHash) {
      mismatch("$.input.profile", "profile differs from the factory-bound VFS session");
    }
    state.metrics = prepareCppCuteBrowserWasmRuntimeMetrics({
      profile,
      memory: taken.memory,
      allocatorRecordPointer: factory.allocatorMetricsPointer,
    });
    expectStatus(taken, IDLE_STATUS, "$.runtime.initialStatus");

    beginCppCuteBrowserWasmRuntimePhase(state.metrics, "input-frame-copy");
    const inputPointer = callU32(
      taken.moduleFacade._bg_cpp_cute_alloc,
      [frame.byteLength],
      "$.runtime.alloc",
    );
    if (inputPointer === 0) {
      compileStatus("$.runtime.alloc", readStatus(taken));
    }
    expectStatus(taken, INPUT_ALLOCATED_STATUS, "$.runtime.statusAfterAlloc");
    const inputEpoch = inspectMemory(taken.memory, "$.runtime.inputMemory");
    checkedRange(inputEpoch, inputPointer, frame.byteLength, "$.runtime.inputRange");
    copyIntoMemory(inputEpoch, inputPointer, frameBytes, frame.byteLength, "$.runtime.inputCopy");
    completeCppCuteBrowserWasmRuntimePhase(state.metrics);

    beginCppCuteBrowserWasmRuntimePhase(state.metrics, "frontend-extractor");
    const compileReturn = callStatusExport(
      taken.moduleFacade._bg_cpp_cute_compile,
      [inputPointer, frame.byteLength],
      "$.runtime.compile",
    );
    const compileState = readStatus(taken);
    if (compileReturn !== compileState) {
      mismatch("$.runtime.compile", "compile return differs from the readable runtime status");
    }
    if (compileState !== ARTIFACT_READY_STATUS) {
      compileStatus("$.runtime.compile", compileState);
    }

    const resultPointer = callU32(
      taken.moduleFacade._bg_cpp_cute_result_pointer,
      [],
      "$.runtime.resultPointer",
    );
    const resultByteLength = callU32(
      taken.moduleFacade._bg_cpp_cute_result_length,
      [],
      "$.runtime.resultByteLength",
    );
    if (resultPointer === 0 || resultByteLength === 0) {
      mismatch("$.runtime.result", "artifact-ready status requires a nonempty result range");
    }
    const outputCeiling = profileRecord.profile.extractionLimits.maxOutputBytes;
    if (resultByteLength > MAX_RESULT_BYTE_LENGTH || resultByteLength > outputCeiling) {
      resource("$.runtime.resultByteLength", "result exceeds the ABI or invocation output ceiling");
    }
    const resultEpoch = inspectMemory(taken.memory, "$.runtime.resultMemory");
    checkedRange(resultEpoch, resultPointer, resultByteLength, "$.runtime.resultRange");
    if (rangesOverlap(inputPointer, frame.byteLength, resultPointer, resultByteLength)) {
      mismatch("$.runtime.resultRange", "module result aliases its live input allocation");
    }

    callVoid(
      taken.moduleFacade._bg_cpp_cute_free,
      [inputPointer, frame.byteLength],
      "$.runtime.freeInput",
    );
    expectStatus(taken, ARTIFACT_READY_STATUS, "$.runtime.statusAfterInputFree");
    expectStableResult(taken, resultPointer, resultByteLength, "$.runtime.resultAfterInputFree");
    completeCppCuteBrowserWasmRuntimePhase(state.metrics);

    beginCppCuteBrowserWasmRuntimePhase(state.metrics, "result-frame-copy");
    const artifactBytes = copyFromMemory(
      taken.memory,
      resultPointer,
      resultByteLength,
      "$.runtime.resultCopy",
    );
    expectStatus(taken, ARTIFACT_READY_STATUS, "$.runtime.statusAfterResultCopy");
    expectStableResult(taken, resultPointer, resultByteLength, "$.runtime.resultAfterCopy");
    callVoid(taken.moduleFacade._bg_cpp_cute_reset, [], "$.runtime.reset");
    expectStatus(taken, IDLE_STATUS, "$.runtime.statusAfterReset");
    if (callU32(
      taken.moduleFacade._bg_cpp_cute_result_pointer,
      [],
      "$.runtime.resultPointerAfterReset",
    ) !== 0 || callU32(
      taken.moduleFacade._bg_cpp_cute_result_length,
      [],
      "$.runtime.resultByteLengthAfterReset",
    ) !== 0) {
      mismatch("$.runtime.resultAfterReset", "reset did not revoke module-owned result storage");
    }
    completeCppCuteBrowserWasmRuntimePhase(state.metrics);
    const runtime = closeCppCuteBrowserWasmRuntimeMetrics(state.metrics);
    state.metricsClosed = true;

    const activeVfs = observeCppCuteBrowserVfsSession(taken.vfsSession);
    requireQuiescentVfs(activeVfs);
    const closedVfs = closeCppCuteBrowserVfsSession(taken.vfsSession, "completed");
    state.vfsClosed = true;
    const vfsRecord = unwrapClosedCppCuteBrowserVfsSession(closedVfs);
    if (vfsRecord.reason !== "completed" || vfsRecord.observation.state !== "disposed") {
      mismatch("$.runtime.vfs", "completed VFS closure did not produce a disposed receipt");
    }
    zeroBytes(frameBytes);

    return NATIVE_OBJECT_FREEZE({
      authority: "wasm-c-abi-local-execution-only",
      protocol: CPP_CUTE_BROWSER_WASM_COMPILER_PROTOCOL,
      profileHash: profile.profileHash,
      wasmSha256: factory.wasmSha256,
      wasmByteLength: factory.wasmByteLength,
      inputFrameByteLength: frame.byteLength,
      resultByteLength,
      compileStatus: NATIVE_OBJECT_FREEZE({
        code: ARTIFACT_READY_STATUS,
        name: "artifact-ready" as const,
      }),
      artifactBytes,
      runtime,
      vfs: vfsRecord.observation,
      cAbiExecutionObserved: true,
      artifactVerificationObserved: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
  } catch (cause) {
    cleanupFailedExecution(state, cause);
  }
}

function cleanupFailedExecution(state: ExecutionState, primaryCause: unknown): never {
  const cleanupCauses: unknown[] = [];
  if (state.metrics !== undefined && !state.metricsClosed) {
    try {
      cancelCppCuteBrowserWasmRuntimeMetrics(state.metrics);
    } catch (cause) {
      if (!(cause instanceof CppCuteBrowserWasmRuntimeMetricsError &&
            cause.code === "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE")) {
        arrayPush(cleanupCauses, cause);
      }
    }
  }
  if (!state.vfsClosed) {
    try {
      const observation = observeCppCuteBrowserVfsSession(state.taken.vfsSession);
      if (observation.state === "active") {
        closeCppCuteBrowserVfsSession(state.taken.vfsSession, "failed");
      }
      state.vfsClosed = true;
    } catch (cause) {
      arrayPush(cleanupCauses, cause);
    }
  }
  zeroBytes(state.frameBytes);
  if (cleanupCauses.length !== 0) {
    cleanup(
      "$.runtime.cleanup",
      "C-ABI execution failed and owned runtime authority cleanup also failed",
      new NATIVE_AGGREGATE_ERROR(
        [primaryCause, ...cleanupCauses],
        "C++/CuTe Wasm execution cleanup failures",
      ),
    );
  }
  throw primaryCause;
}

function exactFrameCopy(value: unknown): InputFrameSnapshot {
  let inspection;
  try {
    inspection = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid("$.input.inputFrameBytes", "expected one exact unshared Uint8Array", { cause });
  }
  if (inspection.byteLength === 0 || inspection.byteLength > MAX_INPUT_FRAME_BYTE_LENGTH) {
    resource("$.input.inputFrameBytes", "input frame exceeds the nonempty runtime ABI range");
  }
  try {
    return NATIVE_OBJECT_FREEZE({
      bytes: copyInspectedUnsharedUint8Array(value, inspection),
      byteLength: inspection.byteLength,
    });
  } catch (cause) {
    invalid("$.input.inputFrameBytes", "failed to snapshot the exact input-frame view", { cause });
  }
}

function inspectMemory(memory: WebAssembly.Memory, path: string): MemoryEpoch {
  if (MEMORY_BUFFER_GETTER === undefined) {
    mismatch(path, "captured WebAssembly.Memory buffer getter is unavailable");
  }
  let prototype: unknown;
  let buffer: unknown;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(memory);
    buffer = NATIVE_REFLECT_APPLY(MEMORY_BUFFER_GETTER, memory, []);
  } catch (cause) {
    mismatch(path, "module memory is not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE ||
      typeof buffer !== "object" || buffer === null ||
      NATIVE_GET_PROTOTYPE_OF(buffer) !== NATIVE_ARRAY_BUFFER_PROTOTYPE) {
    mismatch(path, "execution requires one exact unshared WebAssembly.Memory");
  }
  if (NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) {
    mismatch(path, "captured ArrayBuffer byte-length getter is unavailable");
  }
  const byteLength = NATIVE_REFLECT_APPLY(
    NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER,
    buffer,
    [],
  ) as unknown;
  if (typeof byteLength !== "number" || !NATIVE_NUMBER_IS_INTEGER(byteLength) || byteLength < 0) {
    mismatch(path, "module memory buffer exposed an invalid byte length");
  }
  return { memory, buffer: buffer as ArrayBuffer, byteLength };
}

function ensureMemoryEpoch(epoch: MemoryEpoch, path: string): void {
  if (MEMORY_BUFFER_GETTER === undefined) mismatch(path, "memory getter became unavailable");
  let current: unknown;
  try {
    current = NATIVE_REFLECT_APPLY(MEMORY_BUFFER_GETTER, epoch.memory, []);
  } catch (cause) {
    mismatch(path, "module memory became unavailable", { cause });
  }
  const currentByteLength = NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined
    ? undefined
    : NATIVE_REFLECT_APPLY(NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER, epoch.buffer, []);
  if (current !== epoch.buffer || currentByteLength !== epoch.byteLength) {
    mismatch(path, "module memory changed during one synchronous byte-copy epoch");
  }
}

function checkedRange(
  epoch: MemoryEpoch,
  pointer: number,
  byteLength: number,
  path: string,
): void {
  if (pointer <= 0 || byteLength <= 0) mismatch(path, "Wasm range must be nonempty and nonzero");
  const end = NATIVE_BIGINT(pointer) + NATIVE_BIGINT(byteLength);
  if (end > U32_LIMIT || end > NATIVE_BIGINT(epoch.byteLength)) {
    resource(path, "Wasm range exceeds the live wasm32 memory extent");
  }
}

function copyIntoMemory(
  epoch: MemoryEpoch,
  pointer: number,
  source: Uint8Array,
  sourceByteLength: number,
  path: string,
): void {
  try {
    const target = new NATIVE_UINT8_ARRAY(epoch.buffer, pointer, sourceByteLength);
    NATIVE_REFLECT_APPLY(NATIVE_UINT8_ARRAY_SET, target, [source]);
  } catch (cause) {
    mismatch(path, "failed to copy the input frame into the checked Wasm range", { cause });
  }
  ensureMemoryEpoch(epoch, `${path}.memoryEpoch`);
}

function copyFromMemory(
  memory: WebAssembly.Memory,
  pointer: number,
  byteLength: number,
  path: string,
): Uint8Array {
  const epoch = inspectMemory(memory, `${path}.memory`);
  checkedRange(epoch, pointer, byteLength, `${path}.range`);
  const copy = new NATIVE_UINT8_ARRAY(byteLength);
  try {
    const source = new NATIVE_UINT8_ARRAY(epoch.buffer, pointer, byteLength);
    NATIVE_REFLECT_APPLY(NATIVE_UINT8_ARRAY_SET, copy, [source]);
  } catch (cause) {
    mismatch(path, "failed to snapshot the checked module-owned result range", { cause });
  }
  ensureMemoryEpoch(epoch, `${path}.memoryEpoch`);
  return copy;
}

function expectStableResult(
  taken: TakenCppCuteBrowserEmscriptenFactory,
  pointer: number,
  byteLength: number,
  path: string,
): void {
  if (callU32(taken.moduleFacade._bg_cpp_cute_result_pointer, [], `${path}.pointer`) !== pointer ||
      callU32(taken.moduleFacade._bg_cpp_cute_result_length, [], `${path}.byteLength`) !== byteLength) {
    mismatch(path, "module-owned result range changed before reset");
  }
}

function expectStatus(
  taken: TakenCppCuteBrowserEmscriptenFactory,
  expected: number,
  path: string,
): void {
  const actual = readStatus(taken);
  if (actual !== expected) {
    mismatch(path, `expected C ABI status ${expected}, observed ${actual}`);
  }
}

function readStatus(taken: TakenCppCuteBrowserEmscriptenFactory): number {
  return callStatusExport(taken.moduleFacade._bg_cpp_cute_status, [], "$.runtime.status");
}

function callStatusExport(operation: Function, args: readonly number[], path: string): number {
  const value = callExport(operation, args, path);
  if (typeof value !== "number" || !NATIVE_NUMBER_IS_INTEGER(value)) {
    mismatch(path, "status export did not return one integer");
  }
  for (const status of RUNTIME_ABI.compileStatuses) {
    if (status.code === value) return value;
  }
  mismatch(path, "status export returned a code outside the pinned runtime ABI");
}

function callU32(operation: Function, args: readonly number[], path: string): number {
  const value = callExport(operation, args, path);
  if (typeof value !== "number" || !NATIVE_NUMBER_IS_INTEGER(value) || value < 0 || value > 0xffff_ffff) {
    mismatch(path, "Wasm export did not return one unsigned i32 value");
  }
  return value;
}

function callVoid(operation: Function, args: readonly number[], path: string): void {
  const value = callExport(operation, args, path);
  if (value !== undefined) mismatch(path, "void Wasm export returned a value");
}

function callExport(operation: Function, args: readonly number[], path: string): unknown {
  try {
    return NATIVE_REFLECT_APPLY(operation, undefined, args);
  } catch (cause) {
    trap(path, "Wasm export trapped or aborted during synchronous C ABI execution", cause);
  }
}

function requireQuiescentVfs(observation: CppCuteBrowserVfsSessionObservation): void {
  if (observation.state !== "active") {
    mismatch("$.runtime.vfs", "VFS session terminalized before successful C ABI closure");
  }
  for (const [field, value] of [
    ["currentLiveHandles", observation.counters.currentLiveHandles],
    ["currentLiveSourceLogicalReservationByteLength",
      observation.counters.currentLiveSourceLogicalReservationByteLength],
    ["currentLiveInstalledVfsLogicalReservationByteLength",
      observation.counters.currentLiveInstalledVfsLogicalReservationByteLength],
    ["currentLiveLogicalReservationByteLength",
      observation.counters.currentLiveLogicalReservationByteLength],
  ] as const) {
    if (wireIntegerToBigInt(value) !== 0n) {
      mismatch(`$.runtime.vfs.counters.${field}`, "successful compile leaked a live VFS handle or reservation");
    }
  }
}

function rangesOverlap(
  leftPointer: number,
  leftByteLength: number,
  rightPointer: number,
  rightByteLength: number,
): boolean {
  const leftBegin = NATIVE_BIGINT(leftPointer);
  const leftEnd = leftBegin + NATIVE_BIGINT(leftByteLength);
  const rightBegin = NATIVE_BIGINT(rightPointer);
  const rightEnd = rightBegin + NATIVE_BIGINT(rightByteLength);
  return leftBegin < rightEnd && rightBegin < leftEnd;
}

function compileStatus(path: string, status: number): never {
  let name = "unknown";
  for (const entry of RUNTIME_ABI.compileStatuses) {
    if (entry.code === status) {
      name = entry.name;
      break;
    }
  }
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-COMPILE-STATUS",
    path,
    `C ABI execution terminated with status ${status} (${name})`,
  );
}

function exactDataRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
): Record<string, unknown> {
  let prototype: unknown;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(value);
    descriptors = NATIVE_GET_OWN_PROPERTY_DESCRIPTORS(value);
  } catch (cause) {
    invalid(path, "expected an inspectable plain data record", { cause });
  }
  if (typeof value !== "object" || value === null || prototype !== NATIVE_OBJECT_PROTOTYPE) {
    invalid(path, "expected a plain data record");
  }
  const ownKeys = NATIVE_REFLECT_OWN_KEYS(descriptors);
  if (!hasExactKeys(ownKeys, keys)) {
    invalid(path, `expected exactly fields ${keys.join(", ")}`);
  }
  const result: Record<string, unknown> = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor) || descriptor.enumerable !== true) {
      invalid(`${path}.${key}`, "field must be an enumerable data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function hasExactKeys(actual: readonly PropertyKey[], expected: readonly string[]): boolean {
  if (actual.length !== expected.length) return false;
  for (let actualIndex = 0; actualIndex < actual.length; actualIndex += 1) {
    const candidate = actual[actualIndex];
    if (typeof candidate !== "string") return false;
    let found = false;
    for (let expectedIndex = 0; expectedIndex < expected.length; expectedIndex += 1) {
      if (expected[expectedIndex] === candidate) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

function arrayPush(values: unknown[], value: unknown): void {
  NATIVE_REFLECT_APPLY(NATIVE_ARRAY_PUSH, values, [value]);
}

function zeroBytes(bytes: Uint8Array): void {
  try {
    NATIVE_REFLECT_APPLY(NATIVE_UINT8_ARRAY_FILL, bytes, [0]);
  } catch {
    // The authority is already severed; terminal zeroing is best effort.
  }
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-INVALID", path, message, options);
}

function mismatch(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-MISMATCH", path, message, options);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-RESOURCE-LIMIT", path, message);
}

function trap(path: string, message: string, cause: unknown): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-TRAP", path, message, { cause });
}

function cleanup(path: string, message: string, cause: AggregateError): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-CLEANUP", path, message, { cause });
}

function fail(
  code: CppCuteBrowserWasmCompilerErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserWasmCompilerError(code, path, message, options);
}

import {
  encodeWireU64,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import { CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE } from "./resources/cpp_cute_browser_runtime_abi_v1.js";

export const CPP_CUTE_BROWSER_FRONTEND_WORK_METRICS_PROTOCOL =
  "browsergrad.compiler.cpp-cute.frontend-work-metrics@1";

const CONTRACT = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.frontendWorkMetricsRecord;
const MAGIC = CONTRACT.magicBytes;
const RECORD_BYTE_LENGTH = CONTRACT.byteLength;
const RECORD_ALIGNMENT = CONTRACT.alignmentByteLength;
const IDLE_PHASE = CONTRACT.lifecycle.idlePhase;
const COMPLETE_PHASE = CONTRACT.lifecycle.completePhase;
const HEALTHY_FLAG = CONTRACT.lifecycle.healthyFlag;
export const CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_MAGIC = CONTRACT.magicAscii;
export const CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_VERSION = CONTRACT.version.major;
export const CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_BYTE_LENGTH = RECORD_BYTE_LENGTH;
const U32_LIMIT = 0x1_0000_0000n;
const NATIVE_OBJECT_PROTOTYPE = Object.prototype;
const NATIVE_ARRAY_BUFFER_PROTOTYPE = ArrayBuffer.prototype;
const NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.Memory.prototype;
const MEMORY_BUFFER_GETTER = NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE === undefined
  ? undefined
  : Object.getOwnPropertyDescriptor(NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE, "buffer")?.get;
const ARRAY_BUFFER_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  NATIVE_ARRAY_BUFFER_PROTOTYPE,
  "byteLength",
)?.get;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const NATIVE_UINT8_ARRAY = Uint8Array;
const NATIVE_UINT8_ARRAY_SET = Uint8Array.prototype.set;
const NATIVE_DATA_VIEW = DataView;
const NATIVE_DATA_VIEW_GET_UINT32 = DataView.prototype.getUint32;
const NATIVE_DATA_VIEW_GET_BIG_UINT64 = DataView.prototype.getBigUint64;
const NATIVE_BIGINT = BigInt;
const NATIVE_NUMBER_IS_SAFE_INTEGER = Number.isSafeInteger;

export interface CppCuteBrowserFrontendWorkValuesV1 {
  readonly includeDepth: WireU64;
  readonly macroExpansions: WireU64;
  readonly preprocessedTokens: WireU64;
  readonly astNodes: WireU64;
  readonly constexprSteps: WireU64;
  readonly templateInstantiations: WireU64;
  readonly templateDepth: WireU64;
  readonly completedSemanticPasses: WireU64;
}

export interface CppCuteBrowserFrontendWorkObservationV1 {
  readonly authority: "wasm-frontend-work-local-observation-only";
  readonly protocol: typeof CPP_CUTE_BROWSER_FRONTEND_WORK_METRICS_PROTOCOL;
  readonly profileHash: string;
  readonly source: "wasm-memory-frontend-work-metrics-record-v1";
  readonly confidence: "record-exact-unverified-producer";
  readonly generation: WireU64;
  readonly values: CppCuteBrowserFrontendWorkValuesV1;
  readonly resetConfirmed: true;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityReady: false;
}

declare const preparedFrontendWorkBrand: unique symbol;

/** Single-use local record reader. It cannot mint Worker or lowering authority. */
export interface PreparedCppCuteBrowserFrontendWorkMetrics {
  readonly [preparedFrontendWorkBrand]: true;
  readonly profileHash: string;
  readonly recordPointer: number;
  readonly initialGeneration: WireU64;
  readonly localObservationOnly: true;
  readonly workerExecutionObserved: false;
}

export interface PrepareCppCuteBrowserFrontendWorkMetricsInput {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly memory: WebAssembly.Memory;
  readonly recordPointer: number;
}

export type CppCuteBrowserFrontendWorkMetricsErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-CAPABILITY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-STATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-UNVERIFIED";

export class CppCuteBrowserFrontendWorkMetricsError extends Error {
  constructor(
    readonly code: CppCuteBrowserFrontendWorkMetricsErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserFrontendWorkMetricsError";
  }
}

interface FrontendWorkCounters {
  readonly includeDepth: bigint;
  readonly macroExpansions: bigint;
  readonly preprocessedTokens: bigint;
  readonly astNodes: bigint;
  readonly constexprSteps: bigint;
  readonly templateInstantiations: bigint;
  readonly templateDepth: bigint;
  readonly completedSemanticPasses: bigint;
}

interface FrontendWorkRecord extends FrontendWorkCounters {
  readonly phase: number;
  readonly flags: number;
  readonly generation: bigint;
}

interface FrontendWorkLimits {
  readonly includeDepth: bigint;
  readonly macroExpansions: bigint;
  readonly preprocessedTokens: bigint;
  readonly astNodes: bigint;
  readonly constexprSteps: bigint;
  readonly templateInstantiations: bigint;
  readonly templateDepth: bigint;
}

interface StoredFrontendWork {
  readonly profileHash: string;
  readonly recordPointer: number;
  readonly memory: WebAssembly.Memory;
  readonly initialGeneration: bigint;
  readonly limits: FrontendWorkLimits;
  complete: FrontendWorkRecord | undefined;
  state: "prepared" | "complete" | "closed" | "failed";
}

interface MemoryEpoch {
  readonly memory: WebAssembly.Memory;
  readonly buffer: ArrayBuffer;
  readonly byteLength: number;
}

const FRONTEND_WORK = new WeakMap<object, StoredFrontendWork>();

export function prepareCppCuteBrowserFrontendWorkMetrics(
  input: PrepareCppCuteBrowserFrontendWorkMetricsInput,
): PreparedCppCuteBrowserFrontendWorkMetrics {
  const values = exactDataRecord(input, "$.input", [
    "profile", "memory", "recordPointer",
  ]);
  const profile = values["profile"] as PreparedCppCuteFrontendProfile;
  const memory = values["memory"] as WebAssembly.Memory;
  const recordPointer = u32Pointer(values["recordPointer"], "$.input.recordPointer");
  if (recordPointer % RECORD_ALIGNMENT !== 0) {
    mismatch("$.input.recordPointer", "frontend-work record pointer is misaligned");
  }
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  if (profileRecord.profile.deployment.mode !== "browser-local") {
    invalid("$.input.profile", "frontend-work metrics require a browser-local profile");
  }
  const extraction = profileRecord.profile.extractionLimits;
  const limits: FrontendWorkLimits = NATIVE_OBJECT_FREEZE({
    includeDepth: NATIVE_BIGINT(extraction.maxIncludeDepth),
    macroExpansions: NATIVE_BIGINT(extraction.maxMacroExpansions),
    preprocessedTokens: NATIVE_BIGINT(extraction.maxPreprocessedTokens),
    astNodes: NATIVE_BIGINT(extraction.maxAstNodes),
    constexprSteps: NATIVE_BIGINT(extraction.maxConstexprSteps),
    templateInstantiations: NATIVE_BIGINT(extraction.maxTemplateInstantiations),
    templateDepth: NATIVE_BIGINT(extraction.maxTemplateDepth),
  });
  const initial = readRecord(memory, recordPointer, "$.input.memory");
  if (initial.phase !== IDLE_PHASE || initial.flags !== HEALTHY_FLAG ||
      !countersAreZero(initial)) {
    mismatch(
      "$.input.memory.frontendWork",
      "frontend-work record must begin idle, healthy, and zeroed",
    );
  }
  const stored: StoredFrontendWork = {
    profileHash: profile.profileHash,
    recordPointer,
    memory,
    initialGeneration: initial.generation,
    limits,
    complete: undefined,
    state: "prepared",
  };
  const prepared = NATIVE_OBJECT_FREEZE({
    profileHash: profile.profileHash,
    recordPointer,
    initialGeneration: encodeWireU64(initial.generation),
    localObservationOnly: true,
    workerExecutionObserved: false,
  }) as PreparedCppCuteBrowserFrontendWorkMetrics;
  weakMapSet(FRONTEND_WORK, prepared, stored);
  return prepared;
}

/** Read the completed record after artifact-ready and before runtime reset. */
export function completeCppCuteBrowserFrontendWorkMetrics(
  prepared: PreparedCppCuteBrowserFrontendWorkMetrics,
): void {
  const stored = storedFrontendWork(prepared, "prepared");
  try {
    const record = readRecord(
      stored.memory,
      stored.recordPointer,
      "$.frontendWork.complete",
    );
    if (stored.initialGeneration === 0xffff_ffff_ffff_ffffn ||
        record.generation !== stored.initialGeneration + 1n) {
      mismatch(
        "$.frontendWork.complete.generation",
        "completed generation is not the exact next admitted invocation",
      );
    }
    if (record.phase !== COMPLETE_PHASE || record.flags !== HEALTHY_FLAG ||
        record.completedSemanticPasses !==
          NATIVE_BIGINT(CONTRACT.lifecycle.acceptedArtifactPassCount)) {
      mismatch(
        "$.frontendWork.complete",
        "artifact-ready requires one healthy complete two-pass frontend-work record",
      );
    }
    enforceLimits(record, stored.limits);
    stored.complete = record;
    stored.state = "complete";
  } catch (cause) {
    stored.state = "failed";
    throw cause;
  }
}

/** Confirm reset and emit the only public observation for this reader. */
export function closeCppCuteBrowserFrontendWorkMetrics(
  prepared: PreparedCppCuteBrowserFrontendWorkMetrics,
): CppCuteBrowserFrontendWorkObservationV1 {
  const stored = storedFrontendWork(prepared, "complete");
  const complete = stored.complete;
  if (complete === undefined) state("$.frontendWork", "completed record is unavailable");
  try {
    const reset = readRecord(
      stored.memory,
      stored.recordPointer,
      "$.frontendWork.reset",
    );
    if (reset.phase !== IDLE_PHASE || reset.flags !== HEALTHY_FLAG ||
        reset.generation !== complete.generation || !countersAreZero(reset)) {
      mismatch(
        "$.frontendWork.reset",
        "runtime reset did not preserve generation while returning the record to healthy idle zero state",
      );
    }
    stored.state = "closed";
    return NATIVE_OBJECT_FREEZE({
      authority: "wasm-frontend-work-local-observation-only",
      protocol: CPP_CUTE_BROWSER_FRONTEND_WORK_METRICS_PROTOCOL,
      profileHash: stored.profileHash,
      source: "wasm-memory-frontend-work-metrics-record-v1",
      confidence: "record-exact-unverified-producer",
      generation: encodeWireU64(complete.generation),
      values: publicCounters(complete),
      resetConfirmed: true,
      workerExecutionObserved: false,
      loweringAuthorityReady: false,
    });
  } catch (cause) {
    stored.state = "failed";
    throw cause;
  }
}

export function cancelCppCuteBrowserFrontendWorkMetrics(
  prepared: PreparedCppCuteBrowserFrontendWorkMetrics,
): void {
  const stored = authenticStored(prepared);
  if (stored.state !== "prepared" && stored.state !== "complete") {
    state("$.frontendWork", "only an active frontend-work reader may be cancelled");
  }
  stored.complete = undefined;
  stored.state = "failed";
}

function enforceLimits(record: FrontendWorkRecord, limits: FrontendWorkLimits): void {
  for (const [name, value, maximum] of [
    ["includeDepth", record.includeDepth, limits.includeDepth],
    ["macroExpansions", record.macroExpansions, limits.macroExpansions],
    ["preprocessedTokens", record.preprocessedTokens, limits.preprocessedTokens],
    ["astNodes", record.astNodes, limits.astNodes],
    ["constexprSteps", record.constexprSteps, limits.constexprSteps],
    ["templateInstantiations", record.templateInstantiations, limits.templateInstantiations],
    ["templateDepth", record.templateDepth, limits.templateDepth],
  ] as const) {
    if (value > maximum) {
      resource(`$.frontendWork.complete.${name}`, "frontend-work counter exceeds its prepared profile ceiling");
    }
  }
}

function publicCounters(record: FrontendWorkRecord): CppCuteBrowserFrontendWorkValuesV1 {
  return NATIVE_OBJECT_FREEZE({
    includeDepth: encodeWireU64(record.includeDepth),
    macroExpansions: encodeWireU64(record.macroExpansions),
    preprocessedTokens: encodeWireU64(record.preprocessedTokens),
    astNodes: encodeWireU64(record.astNodes),
    constexprSteps: encodeWireU64(record.constexprSteps),
    templateInstantiations: encodeWireU64(record.templateInstantiations),
    templateDepth: encodeWireU64(record.templateDepth),
    completedSemanticPasses: encodeWireU64(record.completedSemanticPasses),
  });
}

function readRecord(memory: WebAssembly.Memory, pointer: number, path: string): FrontendWorkRecord {
  const epoch = inspectMemory(memory, path);
  const end = NATIVE_BIGINT(pointer) + NATIVE_BIGINT(RECORD_BYTE_LENGTH);
  if (end > U32_LIMIT || end > NATIVE_BIGINT(epoch.byteLength)) {
    resource(path, "frontend-work record exceeds the live wasm32 memory extent");
  }
  const bytes = new NATIVE_UINT8_ARRAY(RECORD_BYTE_LENGTH);
  try {
    const source = new NATIVE_UINT8_ARRAY(epoch.buffer, pointer, RECORD_BYTE_LENGTH);
    NATIVE_REFLECT_APPLY(NATIVE_UINT8_ARRAY_SET, bytes, [source]);
  } catch (cause) {
    mismatch(path, "failed to snapshot the frontend-work record", { cause });
  }
  ensureMemoryEpoch(epoch, path);
  for (let index = 0; index < MAGIC.length; index += 1) {
    if (bytes[index] !== MAGIC[index]) mismatch(`${path}.magic`, "frontend-work record magic differs");
  }
  const view = new NATIVE_DATA_VIEW(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u32 = (offset: number): number =>
    NATIVE_REFLECT_APPLY(NATIVE_DATA_VIEW_GET_UINT32, view, [offset, true]) as number;
  const u64 = (offset: number): bigint =>
    NATIVE_REFLECT_APPLY(NATIVE_DATA_VIEW_GET_BIG_UINT64, view, [offset, true]) as bigint;
  if (u32(8) !== CONTRACT.version.major || u32(12) !== RECORD_BYTE_LENGTH) {
    mismatch(path, "frontend-work record version or byte length differs");
  }
  const phase = u32(16);
  if (phase !== IDLE_PHASE && phase !== CONTRACT.lifecycle.collectingPhase &&
      phase !== COMPLETE_PHASE && phase !== CONTRACT.lifecycle.failedPhase) {
    mismatch(`${path}.phase`, "frontend-work record phase is unknown");
  }
  const flags = u32(20);
  if (flags !== 0 && flags !== HEALTHY_FLAG) {
    mismatch(`${path}.flags`, "frontend-work record flags contain unknown bits");
  }
  return NATIVE_OBJECT_FREEZE({
    phase,
    flags,
    generation: u64(24),
    includeDepth: u64(32),
    macroExpansions: u64(40),
    preprocessedTokens: u64(48),
    astNodes: u64(56),
    constexprSteps: u64(64),
    templateInstantiations: u64(72),
    templateDepth: u64(80),
    completedSemanticPasses: u64(88),
  });
}

function countersAreZero(record: FrontendWorkRecord): boolean {
  return record.includeDepth === 0n && record.macroExpansions === 0n &&
    record.preprocessedTokens === 0n && record.astNodes === 0n &&
    record.constexprSteps === 0n && record.templateInstantiations === 0n &&
    record.templateDepth === 0n && record.completedSemanticPasses === 0n;
}

function inspectMemory(memory: WebAssembly.Memory, path: string): MemoryEpoch {
  if (MEMORY_BUFFER_GETTER === undefined || ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) {
    capability(path, "captured WebAssembly memory intrinsics are unavailable");
  }
  let prototype: unknown;
  let buffer: unknown;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(memory);
    buffer = NATIVE_REFLECT_APPLY(MEMORY_BUFFER_GETTER, memory, []);
  } catch (cause) {
    invalid(path, "Wasm memory is not safely inspectable", { cause });
  }
  if (prototype !== NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE ||
      typeof buffer !== "object" || buffer === null ||
      NATIVE_GET_PROTOTYPE_OF(buffer) !== NATIVE_ARRAY_BUFFER_PROTOTYPE) {
    invalid(path, "frontend-work metrics require one exact unshared WebAssembly.Memory");
  }
  const byteLength = NATIVE_REFLECT_APPLY(
    ARRAY_BUFFER_BYTE_LENGTH_GETTER,
    buffer,
    [],
  ) as unknown;
  if (typeof byteLength !== "number" || !NATIVE_NUMBER_IS_SAFE_INTEGER(byteLength) || byteLength < 0) {
    mismatch(path, "Wasm memory exposed an invalid byte length");
  }
  return { memory, buffer: buffer as ArrayBuffer, byteLength };
}

function ensureMemoryEpoch(epoch: MemoryEpoch, path: string): void {
  if (MEMORY_BUFFER_GETTER === undefined || ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) {
    capability(path, "captured WebAssembly memory intrinsics became unavailable");
  }
  const current = NATIVE_REFLECT_APPLY(MEMORY_BUFFER_GETTER, epoch.memory, []);
  const byteLength = NATIVE_REFLECT_APPLY(
    ARRAY_BUFFER_BYTE_LENGTH_GETTER,
    epoch.buffer,
    [],
  );
  if (current !== epoch.buffer || byteLength !== epoch.byteLength) {
    mismatch(path, "Wasm memory changed during one synchronous frontend-work snapshot");
  }
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
  if (ownKeys.length !== keys.length ||
      ownKeys.some((key) => typeof key !== "string" || !keys.includes(key))) {
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

function u32Pointer(value: unknown, path: string): number {
  if (typeof value !== "number" || !NATIVE_NUMBER_IS_SAFE_INTEGER(value) ||
      value <= 0 || value > 0xffff_ffff) {
    invalid(path, "expected one nonzero wasm32 pointer");
  }
  return value;
}

function authenticStored(
  prepared: PreparedCppCuteBrowserFrontendWorkMetrics,
): StoredFrontendWork {
  if ((typeof prepared !== "object" && typeof prepared !== "function") || prepared === null) {
    unverified("$.frontendWork", "frontend-work reader authority is not authentic");
  }
  const stored = NATIVE_REFLECT_APPLY(
    NATIVE_WEAK_MAP_GET,
    FRONTEND_WORK,
    [prepared],
  ) as StoredFrontendWork | undefined;
  if (stored === undefined) {
    unverified("$.frontendWork", "frontend-work reader authority is not authentic");
  }
  return stored;
}

function storedFrontendWork(
  prepared: PreparedCppCuteBrowserFrontendWorkMetrics,
  expected: StoredFrontendWork["state"],
): StoredFrontendWork {
  const stored = authenticStored(prepared);
  if (stored.state !== expected) {
    state("$.frontendWork", `frontend-work reader must be ${expected}`);
  }
  return stored;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-INVALID", path, message, options);
}

function mismatch(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-MISMATCH", path, message, options);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-RESOURCE-LIMIT", path, message);
}

function state(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-STATE", path, message);
}

function capability(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-CAPABILITY", path, message);
}

function unverified(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-FRONTEND-WORK-UNVERIFIED", path, message);
}

function fail(
  code: CppCuteBrowserFrontendWorkMetricsErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserFrontendWorkMetricsError(code, path, message, options);
}

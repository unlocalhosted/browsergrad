import {
  encodeWireU64,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import { CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE } from "./resources/cpp_cute_browser_runtime_abi_v1.js";

export const CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH =
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.wasm.memory.pageByteLength;
const ALLOCATOR_METRICS_RECORD =
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.allocatorMetricsRecord;
export const CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_MAGIC =
  ALLOCATOR_METRICS_RECORD.magicAscii;
export const CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_VERSION =
  ALLOCATOR_METRICS_RECORD.version.major;
export const CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH =
  ALLOCATOR_METRICS_RECORD.byteLength;

const U32_LIMIT = 0x1_0000_0000n;
const U32_LIMIT_NUMBER = 0x1_0000_0000;
const ALLOCATOR_RECORD_ALIGNMENT = ALLOCATOR_METRICS_RECORD.alignmentByteLength;
const ALLOCATOR_MAGIC_BYTES = ALLOCATOR_METRICS_RECORD.magicBytes;
const [
  ALLOCATOR_MAGIC_FIELD,
  ALLOCATOR_VERSION_FIELD,
  ALLOCATOR_BYTE_LENGTH_FIELD,
  ALLOCATOR_CURRENT_LIVE_FIELD,
  ALLOCATOR_PEAK_LIVE_FIELD,
  ALLOCATOR_CUMULATIVE_ALLOCATED_FIELD,
  ALLOCATOR_CUMULATIVE_FREED_FIELD,
  ALLOCATOR_SUCCESSFUL_ALLOCATION_COUNT_FIELD,
  ALLOCATOR_FREE_COUNT_FIELD,
  ALLOCATOR_FAILED_ALLOCATION_COUNT_FIELD,
] = ALLOCATOR_METRICS_RECORD.fields;
const PHASES = [
  "input-frame-copy",
  "frontend-extractor",
  "result-frame-copy",
] as const;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;
const NATIVE_ARRAY_PUSH = Array.prototype.push;
const NATIVE_WEAK_MAP_GET = WeakMap.prototype.get;
const NATIVE_WEAK_MAP_SET = WeakMap.prototype.set;
const NATIVE_UINT8_ARRAY = Uint8Array;
const NATIVE_BIGINT = BigInt;
const NATIVE_NUMBER_IS_FINITE = Number.isFinite;
const NATIVE_NUMBER_IS_SAFE_INTEGER = Number.isSafeInteger;
const NATIVE_MATH_ROUND = Math.round;
const NATIVE_OBJECT_PROTOTYPE = Object.prototype;
const NATIVE_ARRAY_BUFFER_PROTOTYPE = ArrayBuffer.prototype;
const NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE = typeof WebAssembly === "undefined"
  ? undefined
  : WebAssembly.Memory.prototype;
const MEMORY_BUFFER_GETTER = NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE === undefined
  ? undefined
  : Object.getOwnPropertyDescriptor(NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE, "buffer")?.get;
const ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;
const PERFORMANCE_NOW = typeof globalThis.performance?.now === "function"
  ? globalThis.performance.now.bind(globalThis.performance)
  : undefined;
const UINT8_ARRAY_SET = Uint8Array.prototype.set;

export type CppCuteBrowserWasmRuntimePhase = typeof PHASES[number];

export interface CppCuteBrowserAllocatorMetricsV1 {
  /** Exact live requested bytes across the instrumented module-global allocator. */
  readonly currentLiveGlobalRequestedByteLength: WireU64;
  /** Exact global peak requested bytes; this excludes allocator metadata. */
  readonly peakLiveGlobalRequestedByteLength: WireU64;
  readonly cumulativeGlobalAllocatedRequestedByteLength: WireU64;
  readonly cumulativeGlobalFreedRequestedByteLength: WireU64;
  readonly successfulAllocationCount: WireU64;
  readonly freeCount: WireU64;
  readonly failedAllocationCount: WireU64;
}

export interface CppCuteBrowserWasmRuntimeSampleV1 {
  readonly wasmMemory: {
    readonly source: "webassembly-memory-buffer-byte-length";
    readonly confidence: "exact";
    readonly pageByteLength: typeof CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH;
    readonly pages: WireU64;
    /** Linear-memory capacity, not resident memory or JavaScript heap usage. */
    readonly linearMemoryCapacityByteLength: WireU64;
  };
  readonly allocator: {
    readonly source: "wasm-memory-allocator-metrics-record-v1";
    readonly confidence: "record-exact-unverified-producer";
    readonly values: CppCuteBrowserAllocatorMetricsV1;
  };
}

export interface CppCuteBrowserWasmRuntimePhaseObservationV1 {
  readonly ordinal: number;
  readonly phase: CppCuteBrowserWasmRuntimePhase;
  readonly timing: {
    readonly source: "local-performance-now";
    readonly confidence: "exact";
    /** Monotonic elapsed wall time. It is not CPU time. */
    readonly elapsedMicroseconds: WireU64;
  };
  readonly start: CppCuteBrowserWasmRuntimeSampleV1;
  readonly end: CppCuteBrowserWasmRuntimeSampleV1;
}

export interface CppCuteBrowserWasmRuntimeObservationV1 {
  readonly authority: "wasm-runtime-local-observation-only";
  readonly profileHash: string;
  readonly initial: CppCuteBrowserWasmRuntimeSampleV1;
  readonly current: CppCuteBrowserWasmRuntimeSampleV1;
  readonly peakWasmMemoryPages: WireU64;
  readonly phases: readonly CppCuteBrowserWasmRuntimePhaseObservationV1[];
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityReady: false;
}

declare const preparedMetricsBrand: unique symbol;

/** Runtime-local collector. It is neither result-frame nor Worker authority. */
export interface PreparedCppCuteBrowserWasmRuntimeMetrics {
  readonly [preparedMetricsBrand]: true;
  readonly profileHash: string;
  readonly allocatorRecordPointer: number;
  readonly initialPages: number;
  readonly maximumPages: number;
  /** Profile extraction ceiling; it may be narrower than the ABI page maximum. */
  readonly maxLinearMemoryByteLength: number;
  /** Working + live input-frame + live result requested-byte coexistence bound. */
  readonly maxTrackedAllocatorRequestedByteLength: number;
  readonly localObservationOnly: true;
  readonly workerExecutionObserved: false;
}

export interface PrepareCppCuteBrowserWasmRuntimeMetricsInput {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly memory: WebAssembly.Memory;
  /**
   * Pointer to a local record in the sampled memory. This collector does not
   * prove which module produced the record or mint Worker-execution evidence.
   */
  readonly allocatorRecordPointer: number;
}

export interface CppCuteBrowserWasmRuntimeMetricsOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserWasmRuntimeMetricsErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-CAPABILITY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-UNVERIFIED";

export class CppCuteBrowserWasmRuntimeMetricsError extends Error {
  constructor(
    readonly code: CppCuteBrowserWasmRuntimeMetricsErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserWasmRuntimeMetricsError";
  }
}

interface AllocatorCounters {
  readonly currentLiveGlobalRequestedByteLength: bigint;
  readonly peakLiveGlobalRequestedByteLength: bigint;
  readonly cumulativeGlobalAllocatedRequestedByteLength: bigint;
  readonly cumulativeGlobalFreedRequestedByteLength: bigint;
  readonly successfulAllocationCount: bigint;
  readonly freeCount: bigint;
  readonly failedAllocationCount: bigint;
}

interface RuntimeSample {
  readonly pages: bigint;
  readonly capacityByteLength: bigint;
  readonly allocator: AllocatorCounters;
}

interface ActivePhase {
  readonly ordinal: number;
  readonly phase: CppCuteBrowserWasmRuntimePhase;
  readonly startedAtMilliseconds: number;
  readonly start: RuntimeSample;
}

interface StoredMetrics {
  readonly profileHash: string;
  readonly allocatorRecordPointer: number;
  readonly initialPages: bigint;
  readonly maximumPages: bigint;
  readonly stackByteLength: bigint;
  readonly maxLinearMemoryByteLength: bigint;
  readonly maxTrackedAllocatorRequestedByteLength: bigint;
  readonly maxWallTimeMilliseconds: number;
  memory: WebAssembly.Memory | undefined;
  initial: RuntimeSample;
  last: RuntimeSample;
  peakPages: bigint;
  nextPhaseOrdinal: number;
  activePhase: ActivePhase | undefined;
  readonly phases: CppCuteBrowserWasmRuntimePhaseObservationV1[];
  totalPhaseElapsedMicroseconds: bigint;
  state: "active" | "closed" | "cancelled" | "failed";
}

interface RuntimeSamplingContext {
  readonly allocatorRecordPointer: number;
  readonly maximumPages: bigint;
  readonly stackByteLength: bigint;
  readonly maxLinearMemoryByteLength: bigint;
  readonly maxTrackedAllocatorRequestedByteLength: bigint;
  readonly memory: WebAssembly.Memory;
}

interface MemoryEpoch {
  readonly memory: WebAssembly.Memory;
  readonly buffer: ArrayBuffer;
  readonly byteLength: number;
}

const METRICS = new WeakMap<object, StoredMetrics>();

export function prepareCppCuteBrowserWasmRuntimeMetrics(
  input: PrepareCppCuteBrowserWasmRuntimeMetricsInput,
  options: CppCuteBrowserWasmRuntimeMetricsOptions = {},
): PreparedCppCuteBrowserWasmRuntimeMetrics {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const values = exactDataRecord(input, "$.input", [
    "profile", "memory", "allocatorRecordPointer",
  ]);
  const profile = values["profile"] as PreparedCppCuteFrontendProfile;
  const memory = values["memory"] as WebAssembly.Memory;
  const allocatorRecordPointer = u32Pointer(
    values["allocatorRecordPointer"],
    "$.input.allocatorRecordPointer",
  );
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  if (profileRecord.profile.deployment.mode !== "browser-local") {
    invalid("$.input.profile", "Wasm runtime metrics require a browser-local profile");
  }
  const deployment = profileRecord.profile.deployment;
  const memoryProfile = deployment.compilerRuntime.memory;
  const extractionLimits = profileRecord.profile.extractionLimits;
  const abiMemory = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.wasm.memory;
  const maxTrackedAllocatorRequestedByteLength =
    memoryProfile.maxCompilerWorkingByteLength +
    abiMemory.maxInputFrameByteLength +
    extractionLimits.maxOutputBytes;
  const maxLinearMemoryByteLength = extractionLimits.maxMemoryBytes;
  if (!NATIVE_NUMBER_IS_SAFE_INTEGER(maxTrackedAllocatorRequestedByteLength) ||
      maxTrackedAllocatorRequestedByteLength + memoryProfile.stackByteLength >
        maxLinearMemoryByteLength) {
    resource(
      "$.input.profile",
      "global allocator coexistence bound exceeds the exact Wasm memory ceiling",
    );
  }
  if (maxLinearMemoryByteLength >
      memoryProfile.maximumPages * CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH) {
    mismatch(
      "$.input.profile",
      "profile linear-memory ceiling exceeds the runtime-ABI page maximum",
    );
  }
  const initialPages = NATIVE_BIGINT(memoryProfile.initialPages);
  const sampling: RuntimeSamplingContext = {
    allocatorRecordPointer,
    maximumPages: NATIVE_BIGINT(memoryProfile.maximumPages),
    stackByteLength: NATIVE_BIGINT(memoryProfile.stackByteLength),
    maxLinearMemoryByteLength: NATIVE_BIGINT(maxLinearMemoryByteLength),
    maxTrackedAllocatorRequestedByteLength:
      NATIVE_BIGINT(maxTrackedAllocatorRequestedByteLength),
    memory,
  };
  const initial = readRuntimeSample(sampling, "$.input.memory");
  if (initial.pages !== initialPages) {
    mismatch(
      "$.input.memory",
      "Wasm memory must begin at the exact browser-profile initial page count",
    );
  }
  const stored: StoredMetrics = {
    profileHash: profile.profileHash,
    ...sampling,
    initialPages,
    maxWallTimeMilliseconds: profileRecord.profile.extractionLimits.maxWallTimeMs,
    initial,
    last: initial,
    peakPages: initial.pages,
    nextPhaseOrdinal: 0,
    activePhase: undefined,
    phases: [],
    totalPhaseElapsedMicroseconds: 0n,
    state: "active",
  };
  throwIfAborted(signal, () => disposeStored(stored, "cancelled"));

  const prepared = NATIVE_OBJECT_FREEZE({
    profileHash: stored.profileHash,
    allocatorRecordPointer,
    initialPages: memoryProfile.initialPages,
    maximumPages: memoryProfile.maximumPages,
    maxLinearMemoryByteLength,
    maxTrackedAllocatorRequestedByteLength,
    localObservationOnly: true,
    workerExecutionObserved: false,
  }) as PreparedCppCuteBrowserWasmRuntimeMetrics;
  weakMapSet(METRICS, prepared, stored);
  return prepared;
}

export function beginCppCuteBrowserWasmRuntimePhase(
  prepared: PreparedCppCuteBrowserWasmRuntimeMetrics,
  phase: CppCuteBrowserWasmRuntimePhase,
  options: CppCuteBrowserWasmRuntimeMetricsOptions = {},
): void {
  const signal = normalizeOptions(options);
  const stored = activeStored(prepared);
  throwIfAborted(signal, () => disposeStored(stored, "cancelled"));
  if (stored.activePhase !== undefined) {
    state("$.phase", "a Wasm runtime phase is already active");
  }
  const expectedPhase = PHASES[stored.nextPhaseOrdinal];
  if (phase !== expectedPhase) {
    state("$.phase", `expected phase ${expectedPhase ?? "none"}`);
  }
  try {
    const start = sampleAndAdvance(stored, "$.phase.start");
    stored.activePhase = {
      ordinal: stored.nextPhaseOrdinal,
      phase,
      startedAtMilliseconds: monotonicNow("$.phase.startTime"),
      start,
    };
  } catch (cause) {
    failStoredIfActive(stored);
    throw cause;
  }
}

export function completeCppCuteBrowserWasmRuntimePhase(
  prepared: PreparedCppCuteBrowserWasmRuntimeMetrics,
  options: CppCuteBrowserWasmRuntimeMetricsOptions = {},
): CppCuteBrowserWasmRuntimePhaseObservationV1 {
  const signal = normalizeOptions(options);
  const stored = activeStored(prepared);
  throwIfAborted(signal, () => disposeStored(stored, "cancelled"));
  const active = stored.activePhase;
  if (active === undefined) state("$.phase", "no Wasm runtime phase is active");
  try {
    const endedAtMilliseconds = monotonicNow("$.phase.endTime");
    const elapsedMilliseconds = endedAtMilliseconds - active.startedAtMilliseconds;
    if (!NATIVE_NUMBER_IS_FINITE(elapsedMilliseconds) || elapsedMilliseconds < 0) {
      mismatch("$.phase.timing", "local monotonic clock moved backwards");
    }
    if (elapsedMilliseconds > stored.maxWallTimeMilliseconds) {
      resource("$.phase.timing", "phase elapsed time exceeds the prepared wall-time ceiling");
    }
    const elapsedMicroseconds = NATIVE_BIGINT(
      NATIVE_MATH_ROUND(elapsedMilliseconds * 1_000),
    );
    const totalElapsed = stored.totalPhaseElapsedMicroseconds + elapsedMicroseconds;
    if (totalElapsed > NATIVE_BIGINT(stored.maxWallTimeMilliseconds) * 1_000n) {
      resource("$.phase.timing", "aggregate phase elapsed time exceeds the prepared wall-time ceiling");
    }
    const end = sampleAndAdvance(stored, "$.phase.end");
    const observation = NATIVE_OBJECT_FREEZE({
      ordinal: active.ordinal,
      phase: active.phase,
      timing: NATIVE_OBJECT_FREEZE({
        source: "local-performance-now" as const,
        confidence: "exact" as const,
        elapsedMicroseconds: encodeWireU64(elapsedMicroseconds),
      }),
      start: publicSample(active.start),
      end: publicSample(end),
    });
    NATIVE_REFLECT_APPLY(NATIVE_ARRAY_PUSH, stored.phases, [observation]);
    stored.totalPhaseElapsedMicroseconds = totalElapsed;
    stored.nextPhaseOrdinal += 1;
    stored.activePhase = undefined;
    throwIfAborted(signal, () => disposeStored(stored, "cancelled"));
    return observation;
  } catch (cause) {
    failStoredIfActive(stored);
    throw cause;
  }
}

export function observeCppCuteBrowserWasmRuntimeMetrics(
  prepared: PreparedCppCuteBrowserWasmRuntimeMetrics,
  options: CppCuteBrowserWasmRuntimeMetricsOptions = {},
): CppCuteBrowserWasmRuntimeObservationV1 {
  const signal = normalizeOptions(options);
  const stored = activeStored(prepared);
  throwIfAborted(signal, () => disposeStored(stored, "cancelled"));
  if (stored.activePhase !== undefined) {
    state("$.phase", "cannot emit a runtime observation while a phase is active");
  }
  try {
    const current = sampleAndAdvance(stored, "$.current");
    throwIfAborted(signal, () => disposeStored(stored, "cancelled"));
    return NATIVE_OBJECT_FREEZE({
      authority: "wasm-runtime-local-observation-only",
      profileHash: stored.profileHash,
      initial: publicSample(stored.initial),
      current: publicSample(current),
      peakWasmMemoryPages: encodeWireU64(stored.peakPages),
      phases: NATIVE_OBJECT_FREEZE([...stored.phases]),
      workerExecutionObserved: false,
      loweringAuthorityReady: false,
    });
  } catch (cause) {
    failStoredIfActive(stored);
    throw cause;
  }
}

export function cancelCppCuteBrowserWasmRuntimeMetrics(
  prepared: PreparedCppCuteBrowserWasmRuntimeMetrics,
): void {
  const stored = activeStored(prepared);
  disposeStored(stored, "cancelled");
}

export function closeCppCuteBrowserWasmRuntimeMetrics(
  prepared: PreparedCppCuteBrowserWasmRuntimeMetrics,
  options: CppCuteBrowserWasmRuntimeMetricsOptions = {},
): CppCuteBrowserWasmRuntimeObservationV1 {
  const stored = activeStored(prepared);
  if (stored.activePhase !== undefined || stored.nextPhaseOrdinal !== PHASES.length) {
    state("$.phase", "all Wasm runtime phases must complete before normal closure");
  }
  const observation = observeCppCuteBrowserWasmRuntimeMetrics(prepared, options);
  disposeStored(stored, "closed");
  return observation;
}

function sampleAndAdvance(stored: StoredMetrics, path: string): RuntimeSample {
  const sample = readRuntimeSample(stored, path);
  if (sample.pages < stored.last.pages) {
    mismatch(`${path}.wasmMemory.pages`, "Wasm memory page count decreased");
  }
  assertAllocatorMonotonic(stored.last.allocator, sample.allocator, `${path}.allocator`);
  stored.last = sample;
  if (sample.pages > stored.peakPages) stored.peakPages = sample.pages;
  return sample;
}

function readRuntimeSample(
  stored: RuntimeSamplingContext | StoredMetrics,
  path: string,
): RuntimeSample {
  const memory = stored.memory;
  if (memory === undefined) state(path, "runtime metrics memory was disposed");
  const epoch = inspectMemory(memory, `${path}.wasmMemory`);
  const pages = NATIVE_BIGINT(epoch.byteLength / CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH);
  if (pages > stored.maximumPages) {
    resource(`${path}.wasmMemory.pages`, "Wasm memory exceeds the browser-profile maximum pages");
  }
  if (NATIVE_BIGINT(epoch.byteLength) > stored.maxLinearMemoryByteLength) {
    resource(
      `${path}.wasmMemory.linearMemoryCapacityByteLength`,
      "Wasm memory exceeds the profile extraction memory ceiling",
    );
  }
  const allocator = readAllocatorRecord(
    epoch,
    stored.allocatorRecordPointer,
    stored.maxTrackedAllocatorRequestedByteLength,
    stored.stackByteLength,
    `${path}.allocator`,
  );
  ensureMemoryEpoch(epoch, `${path}.wasmMemory`);
  return NATIVE_OBJECT_FREEZE({
    pages,
    capacityByteLength: NATIVE_BIGINT(epoch.byteLength),
    allocator,
  });
}

function inspectMemory(memory: WebAssembly.Memory, path: string): MemoryEpoch {
  if (MEMORY_BUFFER_GETTER === undefined) capability(path, "WebAssembly.Memory is unavailable");
  let prototype: unknown;
  try {
    prototype = NATIVE_GET_PROTOTYPE_OF(memory);
  } catch (cause) {
    invalid(path, "Wasm memory is not safely inspectable", { cause });
  }
  if (typeof memory !== "object" || memory === null ||
      prototype !== NATIVE_WEBASSEMBLY_MEMORY_PROTOTYPE) {
    invalid(path, "expected an exact WebAssembly.Memory instance");
  }
  let buffer: unknown;
  try {
    buffer = NATIVE_REFLECT_APPLY(MEMORY_BUFFER_GETTER, memory, []);
  } catch (cause) {
    invalid(path, "WebAssembly.Memory buffer is unavailable", { cause });
  }
  if (!isExactUnsharedArrayBuffer(buffer)) {
    invalid(path, "runtime metrics require one unshared Wasm memory");
  }
  if (buffer.byteLength % CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH !== 0) {
    mismatch(path, "Wasm memory byte length is not an exact page multiple");
  }
  return { memory, buffer, byteLength: buffer.byteLength };
}

function isExactUnsharedArrayBuffer(value: unknown): value is ArrayBuffer {
  if (typeof value !== "object" || value === null) return false;
  try {
    return NATIVE_GET_PROTOTYPE_OF(value) === NATIVE_ARRAY_BUFFER_PROTOTYPE;
  } catch {
    return false;
  }
}

function ensureMemoryEpoch(epoch: MemoryEpoch, path: string): void {
  if (MEMORY_BUFFER_GETTER === undefined) capability(path, "WebAssembly.Memory is unavailable");
  let current: unknown;
  try {
    current = NATIVE_REFLECT_APPLY(MEMORY_BUFFER_GETTER, epoch.memory, []);
  } catch (cause) {
    invalid(path, "WebAssembly.Memory buffer became unavailable", { cause });
  }
  if (current !== epoch.buffer || !isExactUnsharedArrayBuffer(current) ||
      current.byteLength !== epoch.byteLength) {
    mismatch(path, "Wasm memory changed during one metrics snapshot");
  }
}

function readAllocatorRecord(
  epoch: MemoryEpoch,
  pointer: number,
  maxTrackedAllocatorRequestedByteLength: bigint,
  stackByteLength: bigint,
  path: string,
): AllocatorCounters {
  const end = NATIVE_BIGINT(pointer) +
    NATIVE_BIGINT(CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH);
  if (end > U32_LIMIT || end > NATIVE_BIGINT(epoch.byteLength)) {
    resource(path, "allocator metrics record exceeds the wasm32 memory range");
  }
  const snapshot = new NATIVE_UINT8_ARRAY(
    CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH,
  );
  NATIVE_REFLECT_APPLY(
    UINT8_ARRAY_SET,
    snapshot,
    [new NATIVE_UINT8_ARRAY(
      epoch.buffer,
      pointer,
      CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH,
    )],
  );
  ensureMemoryEpoch(epoch, `${path}.memoryEpoch`);
  for (let index = 0; index < ALLOCATOR_MAGIC_BYTES.length; index += 1) {
    if (snapshot[ALLOCATOR_MAGIC_FIELD.offset + index] !== ALLOCATOR_MAGIC_BYTES[index]) {
      mismatch(`${path}.magic`, "allocator metrics record has the wrong magic");
    }
  }
  if (readU32Le(snapshot, ALLOCATOR_VERSION_FIELD.offset) !==
      CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_VERSION) {
    mismatch(`${path}.version`, "allocator metrics record has an unsupported version");
  }
  if (readU32Le(snapshot, ALLOCATOR_BYTE_LENGTH_FIELD.offset) !==
      CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH) {
    mismatch(`${path}.byteLength`, "allocator metrics record has the wrong byte length");
  }
  const counters = NATIVE_OBJECT_FREEZE({
    currentLiveGlobalRequestedByteLength: readU64Le(snapshot, ALLOCATOR_CURRENT_LIVE_FIELD.offset),
    peakLiveGlobalRequestedByteLength: readU64Le(snapshot, ALLOCATOR_PEAK_LIVE_FIELD.offset),
    cumulativeGlobalAllocatedRequestedByteLength:
      readU64Le(snapshot, ALLOCATOR_CUMULATIVE_ALLOCATED_FIELD.offset),
    cumulativeGlobalFreedRequestedByteLength:
      readU64Le(snapshot, ALLOCATOR_CUMULATIVE_FREED_FIELD.offset),
    successfulAllocationCount:
      readU64Le(snapshot, ALLOCATOR_SUCCESSFUL_ALLOCATION_COUNT_FIELD.offset),
    freeCount: readU64Le(snapshot, ALLOCATOR_FREE_COUNT_FIELD.offset),
    failedAllocationCount:
      readU64Le(snapshot, ALLOCATOR_FAILED_ALLOCATION_COUNT_FIELD.offset),
  });
  if (counters.cumulativeGlobalFreedRequestedByteLength >
      counters.cumulativeGlobalAllocatedRequestedByteLength ||
      counters.currentLiveGlobalRequestedByteLength !==
        counters.cumulativeGlobalAllocatedRequestedByteLength -
          counters.cumulativeGlobalFreedRequestedByteLength) {
    mismatch(path, "allocator requested-byte arithmetic is inconsistent");
  }
  if (counters.peakLiveGlobalRequestedByteLength <
        counters.currentLiveGlobalRequestedByteLength ||
      counters.peakLiveGlobalRequestedByteLength >
        counters.cumulativeGlobalAllocatedRequestedByteLength) {
    mismatch(path, "allocator peak requested bytes are inconsistent");
  }
  if (counters.freeCount > counters.successfulAllocationCount) {
    mismatch(path, "allocator free count exceeds successful allocations");
  }
  if ((counters.successfulAllocationCount === 0n &&
       counters.cumulativeGlobalAllocatedRequestedByteLength !== 0n) ||
      (counters.freeCount === 0n &&
       counters.cumulativeGlobalFreedRequestedByteLength !== 0n) ||
      (counters.freeCount === counters.successfulAllocationCount &&
       counters.currentLiveGlobalRequestedByteLength !== 0n)) {
    mismatch(path, "allocator allocation/free counts disagree with requested-byte counters");
  }
  if (counters.currentLiveGlobalRequestedByteLength >
        maxTrackedAllocatorRequestedByteLength ||
      counters.peakLiveGlobalRequestedByteLength >
        maxTrackedAllocatorRequestedByteLength) {
    resource(path, "global allocator live requested bytes exceed the checked coexistence ceiling");
  }
  if (counters.currentLiveGlobalRequestedByteLength + stackByteLength >
        NATIVE_BIGINT(epoch.byteLength) ||
      counters.peakLiveGlobalRequestedByteLength + stackByteLength >
        NATIVE_BIGINT(epoch.byteLength)) {
    resource(
      path,
      "allocator requested bytes and the fixed stack reservation exceed current Wasm memory capacity",
    );
  }
  if (counters.cumulativeGlobalAllocatedRequestedByteLength >
        counters.successfulAllocationCount * maxTrackedAllocatorRequestedByteLength ||
      counters.cumulativeGlobalFreedRequestedByteLength >
        counters.freeCount * maxTrackedAllocatorRequestedByteLength) {
    mismatch(path, "allocator cumulative requested bytes exceed their count-derived bound");
  }
  return counters;
}

function readU32Le(bytes: Uint8Array, offset: number): number {
  return (bytes[offset]! |
    (bytes[offset + 1]! << 8) |
    (bytes[offset + 2]! << 16) |
    (bytes[offset + 3]! << 24)) >>> 0;
}

function readU64Le(bytes: Uint8Array, offset: number): bigint {
  let value = 0n;
  for (let index = 0; index < 8; index += 1) {
    value |= NATIVE_BIGINT(bytes[offset + index]!) << NATIVE_BIGINT(index * 8);
  }
  return value;
}

function assertAllocatorMonotonic(
  previous: AllocatorCounters,
  current: AllocatorCounters,
  path: string,
): void {
  for (const field of [
    "peakLiveGlobalRequestedByteLength",
    "cumulativeGlobalAllocatedRequestedByteLength",
    "cumulativeGlobalFreedRequestedByteLength",
    "successfulAllocationCount",
    "freeCount",
    "failedAllocationCount",
  ] as const) {
    if (current[field] < previous[field]) {
      mismatch(`${path}.${field}`, "allocator cumulative counter decreased");
    }
  }
}

function publicSample(sample: RuntimeSample): CppCuteBrowserWasmRuntimeSampleV1 {
  return NATIVE_OBJECT_FREEZE({
    wasmMemory: NATIVE_OBJECT_FREEZE({
      source: "webassembly-memory-buffer-byte-length",
      confidence: "exact",
      pageByteLength: CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH,
      pages: encodeWireU64(sample.pages),
      linearMemoryCapacityByteLength: encodeWireU64(sample.capacityByteLength),
    }),
    allocator: NATIVE_OBJECT_FREEZE({
      source: "wasm-memory-allocator-metrics-record-v1",
      confidence: "record-exact-unverified-producer",
      values: NATIVE_OBJECT_FREEZE({
        currentLiveGlobalRequestedByteLength:
          encodeWireU64(sample.allocator.currentLiveGlobalRequestedByteLength),
        peakLiveGlobalRequestedByteLength:
          encodeWireU64(sample.allocator.peakLiveGlobalRequestedByteLength),
        cumulativeGlobalAllocatedRequestedByteLength:
          encodeWireU64(sample.allocator.cumulativeGlobalAllocatedRequestedByteLength),
        cumulativeGlobalFreedRequestedByteLength:
          encodeWireU64(sample.allocator.cumulativeGlobalFreedRequestedByteLength),
        successfulAllocationCount:
          encodeWireU64(sample.allocator.successfulAllocationCount),
        freeCount: encodeWireU64(sample.allocator.freeCount),
        failedAllocationCount:
          encodeWireU64(sample.allocator.failedAllocationCount),
      }),
    }),
  });
}

function activeStored(prepared: PreparedCppCuteBrowserWasmRuntimeMetrics): StoredMetrics {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = weakMapGet(METRICS, prepared);
  if (stored === undefined || prepared.localObservationOnly !== true ||
      prepared.workerExecutionObserved !== false) {
    unverified();
  }
  if (stored.state !== "active" || stored.memory === undefined) {
    state("$prepared", `runtime metrics collector is ${stored.state}`);
  }
  return stored;
}

function weakMapGet<K extends object, V>(map: WeakMap<K, V>, key: K): V | undefined {
  return NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_GET, map, [key]) as V | undefined;
}

function weakMapSet<K extends object, V>(map: WeakMap<K, V>, key: K, value: V): void {
  NATIVE_REFLECT_APPLY(NATIVE_WEAK_MAP_SET, map, [key, value]);
}

function disposeStored(
  stored: StoredMetrics,
  stateValue: "closed" | "cancelled" | "failed",
): void {
  stored.state = stateValue;
  stored.memory = undefined;
  stored.activePhase = undefined;
}

function failStoredIfActive(stored: StoredMetrics): void {
  if (stored.state === "active") disposeStored(stored, "failed");
}

function monotonicNow(path: string): number {
  if (PERFORMANCE_NOW === undefined) capability(path, "local monotonic clock is unavailable");
  let value: number;
  try {
    value = PERFORMANCE_NOW();
  } catch (cause) {
    capability(path, "local monotonic clock failed", { cause });
  }
  if (!NATIVE_NUMBER_IS_FINITE(value) || value < 0) {
    capability(path, "local monotonic clock returned an invalid value");
  }
  return value;
}

function u32Pointer(value: unknown, path: string): number {
  if (typeof value !== "number" || !NATIVE_NUMBER_IS_SAFE_INTEGER(value) ||
      value <= 0 || value >= U32_LIMIT_NUMBER) {
    invalid(path, "expected a nonzero unsigned wasm32 pointer");
  }
  if (value % ALLOCATOR_RECORD_ALIGNMENT !== 0) {
    invalid(path, `allocator metrics record must be ${ALLOCATOR_RECORD_ALIGNMENT}-byte aligned`);
  }
  return value;
}

function normalizeOptions(options: CppCuteBrowserWasmRuntimeMetricsOptions): AbortSignal | undefined {
  const values = exactDataRecordOptional(options, "$options", ["signal"]);
  const signal = values["signal"];
  if (signal !== undefined && !isAbortSignal(signal)) {
    invalid("$options.signal", "signal must be an AbortSignal");
  }
  return signal as AbortSignal | undefined;
}

function isAbortSignal(value: unknown): value is AbortSignal {
  if (ABORTED_GETTER === undefined || typeof value !== "object" || value === null) return false;
  try {
    return typeof NATIVE_REFLECT_APPLY(ABORTED_GETTER, value, []) === "boolean";
  } catch {
    return false;
  }
}

function throwIfAborted(signal: AbortSignal | undefined, onAbort?: () => void): void {
  if (signal === undefined || ABORTED_GETTER === undefined ||
      NATIVE_REFLECT_APPLY(ABORTED_GETTER, signal, []) !== true) return;
  onAbort?.();
  cancelled();
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
    invalid(path, "expected an inspectable plain object", { cause });
  }
  if (typeof value !== "object" || value === null ||
      prototype !== NATIVE_OBJECT_PROTOTYPE) {
    invalid(path, "expected a plain object");
  }
  const actualKeys = NATIVE_REFLECT_OWN_KEYS(descriptors);
  if (!hasExactKeys(actualKeys, keys)) {
    invalid(path, `expected exactly fields ${formatExpectedKeys(keys)}`);
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

function exactDataRecordOptional(
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
    invalid(path, "expected an inspectable plain object", { cause });
  }
  if (typeof value !== "object" || value === null ||
      prototype !== NATIVE_OBJECT_PROTOTYPE) {
    invalid(path, "expected a plain object");
  }
  const actualKeys = NATIVE_REFLECT_OWN_KEYS(descriptors);
  if (!hasKnownKeys(actualKeys, keys)) {
    invalid(path, "object contains unknown fields");
  }
  const result: Record<string, unknown> = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined) continue;
    if (!("value" in descriptor) || descriptor.enumerable !== true) {
      invalid(`${path}.${key}`, "field must be an enumerable data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function hasExactKeys(
  actual: readonly PropertyKey[],
  expected: readonly string[],
): boolean {
  return actual.length === expected.length && hasKnownKeys(actual, expected);
}

function hasKnownKeys(
  actual: readonly PropertyKey[],
  expected: readonly string[],
): boolean {
  for (const key of actual) {
    if (typeof key !== "string" || !containsString(expected, key)) return false;
  }
  return true;
}

function containsString(values: readonly string[], expected: string): boolean {
  for (const value of values) {
    if (value === expected) return true;
  }
  return false;
}

function formatExpectedKeys(values: readonly string[]): string {
  let result = "";
  for (const value of values) result += `${result.length === 0 ? "" : ", "}${value}`;
  return result;
}

function cancelled(): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-CANCELLED", "$options.signal", "operation was aborted");
}

function capability(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-CAPABILITY", path, message, options);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-INVALID", path, message, options);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-MISMATCH", path, message);
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-RESOURCE-LIMIT", path, message);
}

function state(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE", path, message);
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-UNVERIFIED",
    "$prepared",
    "expected an opaque local Wasm runtime metrics collector",
  );
}

function fail(
  code: CppCuteBrowserWasmRuntimeMetricsErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserWasmRuntimeMetricsError(code, path, message, options);
}

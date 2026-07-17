import { beforeAll, describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH,
  CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_MAGIC,
  CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_VERSION,
  CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH,
  CppCuteBrowserWasmRuntimeMetricsError,
  beginCppCuteBrowserWasmRuntimePhase,
  cancelCppCuteBrowserWasmRuntimeMetrics,
  closeCppCuteBrowserWasmRuntimeMetrics,
  completeCppCuteBrowserWasmRuntimePhase,
  observeCppCuteBrowserWasmRuntimeMetrics,
  prepareCppCuteBrowserWasmRuntimeMetrics,
} from "../../src/cpp_cute_browser_wasm_runtime_metrics.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import { CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE } from "../../src/resources/cpp_cute_browser_runtime_abi_v1.js";
import { createCppCuteBrowserProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const ALLOCATOR_RECORD_POINTER = 64;
const MIB = 1024 * 1024;

interface AllocatorRecord {
  readonly currentLiveGlobalRequestedByteLength: bigint;
  readonly peakLiveGlobalRequestedByteLength: bigint;
  readonly cumulativeGlobalAllocatedRequestedByteLength: bigint;
  readonly cumulativeGlobalFreedRequestedByteLength: bigint;
  readonly successfulAllocationCount: bigint;
  readonly freeCount: bigint;
  readonly failedAllocationCount: bigint;
}

const EMPTY_ALLOCATOR_RECORD: AllocatorRecord = Object.freeze({
  currentLiveGlobalRequestedByteLength: 0n,
  peakLiveGlobalRequestedByteLength: 0n,
  cumulativeGlobalAllocatedRequestedByteLength: 0n,
  cumulativeGlobalFreedRequestedByteLength: 0n,
  successfulAllocationCount: 0n,
  freeCount: 0n,
  failedAllocationCount: 0n,
});

let profile: PreparedCppCuteFrontendProfile;

beforeAll(async () => {
  profile = await prepareCppCuteFrontendProfile(createCppCuteBrowserProfileInput());
});

function memoryFixture(selectedProfile = profile): WebAssembly.Memory {
  const deployment = unwrapPreparedCppCuteBrowserFrontendProfile(selectedProfile)
    .profile.deployment;
  if (deployment.mode !== "browser-local") throw new Error("expected browser fixture");
  const memory = new WebAssembly.Memory({
    initial: deployment.compilerRuntime.memory.initialPages,
    maximum: deployment.compilerRuntime.memory.maximumPages,
  });
  writeAllocatorRecord(memory, EMPTY_ALLOCATOR_RECORD);
  return memory;
}

function writeAllocatorRecord(
  memory: WebAssembly.Memory,
  values: AllocatorRecord,
  pointer = ALLOCATOR_RECORD_POINTER,
): void {
  const bytes = new Uint8Array(
    memory.buffer,
    pointer,
    CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH,
  );
  bytes.fill(0);
  bytes.set(new TextEncoder().encode(CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_MAGIC));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  view.setUint32(8, CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_VERSION, true);
  view.setUint32(12, CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH, true);
  view.setBigUint64(16, values.currentLiveGlobalRequestedByteLength, true);
  view.setBigUint64(24, values.peakLiveGlobalRequestedByteLength, true);
  view.setBigUint64(32, values.cumulativeGlobalAllocatedRequestedByteLength, true);
  view.setBigUint64(40, values.cumulativeGlobalFreedRequestedByteLength, true);
  view.setBigUint64(48, values.successfulAllocationCount, true);
  view.setBigUint64(56, values.freeCount, true);
  view.setBigUint64(64, values.failedAllocationCount, true);
}

function expectMetricsError(
  operation: () => unknown,
  code: CppCuteBrowserWasmRuntimeMetricsError["code"],
  path: string,
): void {
  let observed: unknown;
  try {
    operation();
  } catch (error) {
    observed = error;
  }
  expect(observed).toBeInstanceOf(CppCuteBrowserWasmRuntimeMetricsError);
  expect(observed).toMatchObject({ code, path });
}

describe("C++/CuTe runtime-local Wasm metrics", () => {
  it("derives the local record decoder identity from the canonical runtime ABI", () => {
    const contract = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.allocatorMetricsRecord;
    expect(CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_MAGIC).toBe(contract.magicAscii);
    expect(CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_VERSION).toBe(contract.version.major);
    expect(CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH).toBe(contract.byteLength);
    expect(contract.pointerContract.zero).toBe("forbidden-for-conforming-live-module-instance");
  });

  it("observes exact pages, requested-byte allocator counters, and ordered phase wall time", () => {
    const memory = memoryFixture();
    const metrics = prepareCppCuteBrowserWasmRuntimeMetrics({
      profile,
      memory,
      allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
    });
    expect(metrics.maxLinearMemoryByteLength).toBe(1024 * MIB);
    expect(metrics.maxTrackedAllocatorRequestedByteLength).toBe(524 * MIB);
    const deployment = unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile.deployment;
    if (deployment.mode !== "browser-local") throw new Error("expected browser fixture");
    expect(metrics.maxTrackedAllocatorRequestedByteLength).toBe(
      deployment.compilerRuntime.memory.maxCompilerWorkingByteLength +
        4 * 1024 * 1024 + 8 * 1024 * 1024,
    );

    beginCppCuteBrowserWasmRuntimePhase(metrics, "input-frame-copy");
    expect(memory.grow(1)).toBe(metrics.initialPages);
    writeAllocatorRecord(memory, {
      currentLiveGlobalRequestedByteLength: 96n,
      peakLiveGlobalRequestedByteLength: 128n,
      cumulativeGlobalAllocatedRequestedByteLength: 160n,
      cumulativeGlobalFreedRequestedByteLength: 64n,
      successfulAllocationCount: 2n,
      freeCount: 1n,
      failedAllocationCount: 1n,
    });
    const inputPhase = completeCppCuteBrowserWasmRuntimePhase(metrics);

    beginCppCuteBrowserWasmRuntimePhase(metrics, "frontend-extractor");
    writeAllocatorRecord(memory, {
      currentLiveGlobalRequestedByteLength: 0n,
      peakLiveGlobalRequestedByteLength: 512n,
      cumulativeGlobalAllocatedRequestedByteLength: 576n,
      cumulativeGlobalFreedRequestedByteLength: 576n,
      successfulAllocationCount: 3n,
      freeCount: 3n,
      failedAllocationCount: 1n,
    });
    const frontendPhase = completeCppCuteBrowserWasmRuntimePhase(metrics);
    beginCppCuteBrowserWasmRuntimePhase(metrics, "result-frame-copy");
    const resultPhase = completeCppCuteBrowserWasmRuntimePhase(metrics);
    const observation = closeCppCuteBrowserWasmRuntimeMetrics(metrics);

    expect(inputPhase).toMatchObject({
      ordinal: 0,
      phase: "input-frame-copy",
      timing: {
        source: "local-performance-now",
        confidence: "exact",
        elapsedMicroseconds: expect.stringMatching(/^\d+$/u),
      },
      start: { wasmMemory: { pages: String(metrics.initialPages) } },
      end: { wasmMemory: { pages: String(metrics.initialPages + 1) } },
    });
    expect(frontendPhase).toMatchObject({ ordinal: 1, phase: "frontend-extractor" });
    expect(resultPhase).toMatchObject({ ordinal: 2, phase: "result-frame-copy" });
    expect(observation).toMatchObject({
      authority: "wasm-runtime-local-observation-only",
      profileHash: profile.profileHash,
      initial: {
        wasmMemory: {
          source: "webassembly-memory-buffer-byte-length",
          confidence: "exact",
          pageByteLength: CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH,
          pages: String(metrics.initialPages),
        },
      },
      current: {
        wasmMemory: {
          pages: String(metrics.initialPages + 1),
          linearMemoryCapacityByteLength: String(
            (metrics.initialPages + 1) * CPP_CUTE_BROWSER_WASM_PAGE_BYTE_LENGTH,
          ),
        },
        allocator: {
          source: "wasm-memory-allocator-metrics-record-v1",
          confidence: "record-exact-unverified-producer",
          values: {
            currentLiveGlobalRequestedByteLength: "0",
            peakLiveGlobalRequestedByteLength: "512",
            cumulativeGlobalAllocatedRequestedByteLength: "576",
            cumulativeGlobalFreedRequestedByteLength: "576",
            successfulAllocationCount: "3",
            freeCount: "3",
            failedAllocationCount: "1",
          },
        },
      },
      peakWasmMemoryPages: String(metrics.initialPages + 1),
      workerExecutionObserved: false,
      loweringAuthorityReady: false,
    });
    expect(observation.phases.map((phase) => phase.phase)).toEqual([
      "input-frame-copy",
      "frontend-extractor",
      "result-frame-copy",
    ]);
    expect(Object.isFrozen(observation.current.allocator.values)).toBe(true);
    expect(JSON.stringify(observation)).not.toMatch(
      /javascriptHeap|cpuTime|processMetrics|residentByte|logicalVfs/iu,
    );
    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics(metrics),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE",
      "$prepared",
    );
  });

  it("rejects pointer overflow, inconsistent or over-ceiling allocator records, and hostile memory", () => {
    const memory = memoryFixture();
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory,
        allocatorRecordPointer: 0xffff_fff8,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-RESOURCE-LIMIT",
      "$.input.memory.allocator",
    );

    const impossibleCountMemory = memoryFixture();
    writeAllocatorRecord(impossibleCountMemory, {
      ...EMPTY_ALLOCATOR_RECORD,
      currentLiveGlobalRequestedByteLength: 1n,
      peakLiveGlobalRequestedByteLength: 1n,
      cumulativeGlobalAllocatedRequestedByteLength: 1n,
    });
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory: impossibleCountMemory,
        allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-MISMATCH",
      "$.input.memory.allocator",
    );

    const impossibleCumulativeMemory = memoryFixture();
    const aboveOneAllocationBound = 525n * BigInt(MIB);
    writeAllocatorRecord(impossibleCumulativeMemory, {
      ...EMPTY_ALLOCATOR_RECORD,
      peakLiveGlobalRequestedByteLength: 1n,
      cumulativeGlobalAllocatedRequestedByteLength: aboveOneAllocationBound,
      cumulativeGlobalFreedRequestedByteLength: aboveOneAllocationBound,
      successfulAllocationCount: 1n,
      freeCount: 1n,
    });
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory: impossibleCumulativeMemory,
        allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-MISMATCH",
      "$.input.memory.allocator",
    );

    const stackCollisionMemory = memoryFixture();
    const allocatorWithoutStackHeadroom = 250n * BigInt(MIB);
    writeAllocatorRecord(stackCollisionMemory, {
      ...EMPTY_ALLOCATOR_RECORD,
      currentLiveGlobalRequestedByteLength: allocatorWithoutStackHeadroom,
      peakLiveGlobalRequestedByteLength: allocatorWithoutStackHeadroom,
      cumulativeGlobalAllocatedRequestedByteLength: allocatorWithoutStackHeadroom,
      successfulAllocationCount: 1n,
    });
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory: stackCollisionMemory,
        allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-RESOURCE-LIMIT",
      "$.input.memory.allocator",
    );
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory,
        allocatorRecordPointer: Number.MAX_SAFE_INTEGER,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-INVALID",
      "$.input.allocatorRecordPointer",
    );
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory,
        allocatorRecordPointer: 0,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-INVALID",
      "$.input.allocatorRecordPointer",
    );
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory,
        allocatorRecordPointer: 4,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-INVALID",
      "$.input.allocatorRecordPointer",
    );

    writeAllocatorRecord(memory, {
      ...EMPTY_ALLOCATOR_RECORD,
      currentLiveGlobalRequestedByteLength: 2n,
      peakLiveGlobalRequestedByteLength: 2n,
      cumulativeGlobalAllocatedRequestedByteLength: 3n,
    });
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory,
        allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-MISMATCH",
      "$.input.memory.allocator",
    );

    writeAllocatorRecord(memory, EMPTY_ALLOCATOR_RECORD);
    const metrics = prepareCppCuteBrowserWasmRuntimeMetrics({
      profile,
      memory,
      allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
    });
    const aboveAllocatorCoexistenceCeiling =
      BigInt(metrics.maxTrackedAllocatorRequestedByteLength) + 1n;
    writeAllocatorRecord(memory, {
      ...EMPTY_ALLOCATOR_RECORD,
      currentLiveGlobalRequestedByteLength: aboveAllocatorCoexistenceCeiling,
      peakLiveGlobalRequestedByteLength: aboveAllocatorCoexistenceCeiling,
      cumulativeGlobalAllocatedRequestedByteLength: aboveAllocatorCoexistenceCeiling,
      successfulAllocationCount: 1n,
    });
    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics(metrics),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-RESOURCE-LIMIT",
      "$.current.allocator",
    );

    const fakeMemory = Object.create(WebAssembly.Memory.prototype) as WebAssembly.Memory;
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory: fakeMemory,
        allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-INVALID",
      "$.input.memory.wasmMemory",
    );
    const sharedMemory = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory: sharedMemory,
        allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-INVALID",
      "$.input.memory.wasmMemory",
    );
  });

  it("enforces a profile linear-memory ceiling narrower than the ABI page maximum", async () => {
    const input = structuredClone(createCppCuteBrowserProfileInput()) as unknown as {
      deployment: {
        compilerRuntime: { memory: { maxCompilerWorkingByteLength: number } };
      };
      extractionLimits: { maxMemoryBytes: number };
    };
    input.deployment.compilerRuntime.memory.maxCompilerWorkingByteLength = 240 * MIB;
    input.extractionLimits.maxMemoryBytes = 268 * MIB;
    const narrowProfile = await prepareCppCuteFrontendProfile(input as never);
    const memory = memoryFixture(narrowProfile);
    const metrics = prepareCppCuteBrowserWasmRuntimeMetrics({
      profile: narrowProfile,
      memory,
      allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
    });
    expect(metrics.maxLinearMemoryByteLength).toBe(268 * MIB);
    expect(metrics.maxTrackedAllocatorRequestedByteLength).toBe(252 * MIB);

    expect(memory.grow(193)).toBe(metrics.initialPages);
    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics(metrics),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-RESOURCE-LIMIT",
      "$.current.wasmMemory.linearMemoryCapacityByteLength",
    );
  });

  it("keeps opaque-state and freezing behavior stable after intrinsic poisoning", () => {
    const memory = memoryFixture();
    const metrics = prepareCppCuteBrowserWasmRuntimeMetrics({
      profile,
      memory,
      allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
    });
    const originalWeakMapGet = WeakMap.prototype.get;
    const originalGetPrototypeOf = Object.getPrototypeOf;
    const originalFreeze = Object.freeze;
    let observation: ReturnType<typeof observeCppCuteBrowserWasmRuntimeMetrics> | undefined;
    try {
      WeakMap.prototype.get = (() => {
        throw new Error("poisoned WeakMap.get");
      }) as typeof WeakMap.prototype.get;
      Object.getPrototypeOf = (() => {
        throw new Error("poisoned Object.getPrototypeOf");
      }) as typeof Object.getPrototypeOf;
      Object.freeze = ((value: unknown) => value) as typeof Object.freeze;
      observation = observeCppCuteBrowserWasmRuntimeMetrics(metrics);
    } finally {
      WeakMap.prototype.get = originalWeakMapGet;
      Object.getPrototypeOf = originalGetPrototypeOf;
      Object.freeze = originalFreeze;
    }
    expect(observation).toBeDefined();
    expect(Object.isFrozen(observation)).toBe(true);
    expect(Object.isFrozen(observation?.current.allocator.values)).toBe(true);
  });

  it("rejects decreasing cumulative counters and disposes the collector after corruption", () => {
    const memory = memoryFixture();
    writeAllocatorRecord(memory, {
      ...EMPTY_ALLOCATOR_RECORD,
      currentLiveGlobalRequestedByteLength: 10n,
      peakLiveGlobalRequestedByteLength: 10n,
      cumulativeGlobalAllocatedRequestedByteLength: 10n,
      successfulAllocationCount: 1n,
    });
    const metrics = prepareCppCuteBrowserWasmRuntimeMetrics({
      profile,
      memory,
      allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
    });
    writeAllocatorRecord(memory, {
      ...EMPTY_ALLOCATOR_RECORD,
      currentLiveGlobalRequestedByteLength: 9n,
      peakLiveGlobalRequestedByteLength: 9n,
      cumulativeGlobalAllocatedRequestedByteLength: 9n,
      successfulAllocationCount: 1n,
    });

    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics(metrics),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-MISMATCH",
      "$.current.allocator.peakLiveGlobalRequestedByteLength",
    );
    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics(metrics),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE",
      "$prepared",
    );
  });

  it("fails closed on cancellation, phase-state misuse, forged authority, and VFS-residency input", () => {
    const memory = memoryFixture();
    const aborted = new AbortController();
    aborted.abort();
    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory,
        allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
      }, { signal: aborted.signal }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-CANCELLED",
      "$options.signal",
    );

    expectMetricsError(
      () => prepareCppCuteBrowserWasmRuntimeMetrics({
        profile,
        memory,
        allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
        logicalVfsReservationByteLength: 123,
      } as never),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-INVALID",
      "$.input",
    );

    const cancelledMetrics = prepareCppCuteBrowserWasmRuntimeMetrics({
      profile,
      memory,
      allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
    });
    expectMetricsError(
      () => beginCppCuteBrowserWasmRuntimePhase(
        cancelledMetrics,
        "input-frame-copy",
        { signal: aborted.signal },
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-CANCELLED",
      "$options.signal",
    );
    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics(cancelledMetrics),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE",
      "$prepared",
    );

    const stateMetrics = prepareCppCuteBrowserWasmRuntimeMetrics({
      profile,
      memory,
      allocatorRecordPointer: ALLOCATOR_RECORD_POINTER,
    });
    expectMetricsError(
      () => beginCppCuteBrowserWasmRuntimePhase(stateMetrics, "frontend-extractor"),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE",
      "$.phase",
    );
    beginCppCuteBrowserWasmRuntimePhase(stateMetrics, "input-frame-copy");
    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics(stateMetrics),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE",
      "$.phase",
    );
    completeCppCuteBrowserWasmRuntimePhase(stateMetrics);
    cancelCppCuteBrowserWasmRuntimeMetrics(stateMetrics);
    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics(stateMetrics),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-STATE",
      "$prepared",
    );
    expectMetricsError(
      () => observeCppCuteBrowserWasmRuntimeMetrics({ ...stateMetrics }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-METRICS-UNVERIFIED",
      "$prepared",
    );
  });
});

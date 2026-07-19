import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

interface MockFactoryState {
  state: "prepared" | "taken";
  readonly taken: object;
}

interface MockVfsState {
  state: "active" | "disposed";
  reason: "completed" | "failed" | undefined;
  closeCalls: number;
  readonly profileHash: string;
  readonly receipt: object;
}

const authorities = vi.hoisted(() => ({
  factories: new WeakMap<object, MockFactoryState>(),
  sessions: new WeakMap<object, MockVfsState>(),
  receipts: new WeakMap<object, MockVfsState>(),
}));

vi.mock("../../src/cpp_cute_browser_emscripten_factory.js", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../src/cpp_cute_browser_emscripten_factory.js")
  >();
  return {
    ...actual,
    takeCppCuteBrowserEmscriptenFactory: (prepared: object) => {
      const stored = authorities.factories.get(prepared);
      if (stored === undefined) {
        throw new actual.CppCuteBrowserEmscriptenFactoryError(
          "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-UNVERIFIED",
          "$.factory",
          "unregistered test factory",
        );
      }
      if (stored.state !== "prepared") {
        throw new actual.CppCuteBrowserEmscriptenFactoryError(
          "BG-COMPILER-CPP-CUTE-BROWSER-EMSCRIPTEN-FACTORY-STATE",
          "$.factory",
          "test factory already taken",
        );
      }
      stored.state = "taken";
      return stored.taken;
    },
  };
});

vi.mock("../../src/cpp_cute_browser_vfs_session.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_vfs_session.js")>();
  const storedSession = (session: object): MockVfsState => {
    const stored = authorities.sessions.get(session);
    if (stored === undefined) throw new Error("unregistered test VFS session");
    return stored;
  };
  const observation = (stored: MockVfsState) => Object.freeze({
    installationId: "bg.cpp.browser-vfs-installation.sha256." + "1".repeat(64),
    requestId: "bg.cpp.frontend-request.sha256." + "2".repeat(64),
    profileHash: stored.profileHash,
    state: stored.state,
    counters: zeroVfsCounters(),
    openedFiles: Object.freeze([]),
  });
  return {
    ...actual,
    observeCppCuteBrowserVfsSession: (session: object) => observation(storedSession(session)),
    closeCppCuteBrowserVfsSession: (session: object, reason: "completed" | "failed") => {
      const stored = storedSession(session);
      if (stored.state !== "active") throw new Error("test VFS session already closed");
      stored.closeCalls += 1;
      stored.reason = reason;
      stored.state = "disposed";
      authorities.receipts.set(stored.receipt, stored);
      return stored.receipt;
    },
    unwrapClosedCppCuteBrowserVfsSession: (receipt: object) => {
      const stored = authorities.receipts.get(receipt);
      if (stored === undefined || stored.reason === undefined) {
        throw new Error("unregistered test VFS receipt");
      }
      return Object.freeze({ observation: observation(stored), reason: stored.reason });
    },
  };
});

import {
  CppCuteBrowserWasmCompilerError,
  executeCppCuteBrowserWasmCompiler,
} from "../../src/cpp_cute_browser_wasm_compiler.js";
import type {
  PreparedCppCuteBrowserEmscriptenFactory,
  TakenCppCuteBrowserEmscriptenFactory,
} from "../../src/cpp_cute_browser_emscripten_factory.js";
import {
  CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH,
  CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_MAGIC,
  CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_VERSION,
} from "../../src/cpp_cute_browser_wasm_runtime_metrics.js";
import {
  CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_BYTE_LENGTH,
  CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_MAGIC,
  CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_VERSION,
} from "../../src/cpp_cute_browser_frontend_work_metrics.js";
import type { PreparedCppCuteBrowserVfsSession } from "../../src/cpp_cute_browser_vfs_session.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import { createCppCuteBrowserProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const ALLOCATOR_RECORD_POINTER = 64;
const FRONTEND_WORK_RECORD_POINTER = 256;
const INPUT_POINTER = 4_096;
const RESULT_POINTER = 65_536;
const WASM_SHA256 = "a".repeat(64);
const FRAME_BYTES = Uint8Array.of(0x42, 0x47, 0x43, 0x43, 1, 2, 3, 4);
const ARTIFACT_BYTES = new TextEncoder().encode(
  '{"schema":"browsergrad.compiler.cpp-cute.frontend-artifact","version":{"major":3,"minor":0}}',
);

interface AllocatorCounters {
  current: bigint;
  peak: bigint;
  allocated: bigint;
  freed: bigint;
  allocations: bigint;
  frees: bigint;
  failed: bigint;
}

interface RuntimeOptions {
  readonly compileReturn?: number;
  readonly readableCompileStatus?: number;
  readonly resultPointer?: number;
  readonly resultByteLength?: number;
  readonly trapCompile?: boolean;
}

interface RuntimeFixture {
  readonly factory: PreparedCppCuteBrowserEmscriptenFactory;
  readonly factoryState: MockFactoryState;
  readonly vfsState: MockVfsState;
  readonly calls: {
    alloc: number;
    compile: number;
    free: number;
    reset: number;
  };
}

let profile: PreparedCppCuteFrontendProfile;

beforeAll(async () => {
  profile = await prepareCppCuteFrontendProfile(createCppCuteBrowserProfileInput());
});

beforeEach(() => {
  vi.clearAllMocks();
});

describe("C++/CuTe local Wasm C ABI compiler execution", () => {
  it("copies one input, executes one compile, copies one result, and closes exact metrics and VFS lifecycles", () => {
    const fixture = runtimeFixture();
    const result = executeCppCuteBrowserWasmCompiler({
      factory: fixture.factory,
      profile,
      inputFrameBytes: FRAME_BYTES,
    });

    expect(result).toMatchObject({
      authority: "wasm-c-abi-local-execution-only",
      protocol: "browsergrad.compiler.cpp-cute.wasm-c-abi-execution@1",
      profileHash: profile.profileHash,
      wasmSha256: WASM_SHA256,
      wasmByteLength: 1024,
      inputFrameByteLength: FRAME_BYTES.byteLength,
      resultByteLength: ARTIFACT_BYTES.byteLength,
      compileStatus: { code: 0, name: "artifact-ready" },
      cAbiExecutionObserved: true,
      artifactVerificationObserved: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
      runtime: {
        authority: "wasm-runtime-local-observation-only",
        phases: [
          { ordinal: 0, phase: "input-frame-copy" },
          { ordinal: 1, phase: "frontend-extractor" },
          { ordinal: 2, phase: "result-frame-copy" },
        ],
        current: {
          allocator: {
            values: { currentLiveGlobalRequestedByteLength: "0" },
          },
        },
      },
      frontendWork: {
        authority: "wasm-frontend-work-local-observation-only",
        generation: "1",
        resetConfirmed: true,
        values: {
          includeDepth: "1",
          macroExpansions: "2",
          preprocessedTokens: "64",
          astNodes: "32",
          constexprSteps: "4",
          templateInstantiations: "8",
          templateDepth: "2",
          completedSemanticPasses: "2",
        },
      },
      vfs: { state: "disposed" },
    });
    expect(result.artifactBytes).toEqual(ARTIFACT_BYTES);
    expect(result.artifactBytes).not.toBe(ARTIFACT_BYTES);
    expect(FRAME_BYTES).toEqual(Uint8Array.of(0x42, 0x47, 0x43, 0x43, 1, 2, 3, 4));
    expect(fixture.calls).toEqual({ alloc: 1, compile: 1, free: 1, reset: 1 });
    expect(fixture.factoryState.state).toBe("taken");
    expect(fixture.vfsState).toMatchObject({
      state: "disposed",
      reason: "completed",
      closeCalls: 1,
    });
    expect(Object.isFrozen(result)).toBe(true);
    expect(Object.isFrozen(result.compileStatus)).toBe(true);
  });

  it("reports exact non-success status and discards the instance without further ABI calls", () => {
    const fixture = runtimeFixture({ compileReturn: 105, readableCompileStatus: 105 });
    expectCompilerError(
      () => executeCppCuteBrowserWasmCompiler({
        factory: fixture.factory,
        profile,
        inputFrameBytes: FRAME_BYTES,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-COMPILE-STATUS",
      "$.runtime.compile",
    );
    expect(fixture.calls).toEqual({ alloc: 1, compile: 1, free: 0, reset: 0 });
    expect(fixture.vfsState).toMatchObject({
      state: "disposed",
      reason: "failed",
      closeCalls: 1,
    });
  });

  it("fails closed on status disagreement, result aliasing, and traps", () => {
    for (const options of [
      { compileReturn: 0, readableCompileStatus: 1 },
      { resultPointer: INPUT_POINTER + 1 },
      { trapCompile: true },
    ] satisfies readonly RuntimeOptions[]) {
      const fixture = runtimeFixture(options);
      let observed: unknown;
      try {
        executeCppCuteBrowserWasmCompiler({
          factory: fixture.factory,
          profile,
          inputFrameBytes: FRAME_BYTES,
        });
      } catch (cause) {
        observed = cause;
      }
      expect(observed).toBeInstanceOf(CppCuteBrowserWasmCompilerError);
      expect(observed).toMatchObject({
        code: options.trapCompile === true
          ? "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-TRAP"
          : "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-MISMATCH",
      });
      expect(fixture.calls.free).toBe(0);
      expect(fixture.calls.reset).toBe(0);
      expect(fixture.vfsState.reason).toBe("failed");
    }
  });

  it("rejects forged or over-ceiling frame views before consuming the factory", () => {
    class DerivedBytes extends Uint8Array {}
    for (const bytes of [
      new Uint8Array(0),
      new DerivedBytes(8),
      new Uint8Array(4 * 1024 * 1024 + 1),
    ]) {
      const fixture = runtimeFixture();
      let observed: unknown;
      try {
        executeCppCuteBrowserWasmCompiler({
          factory: fixture.factory,
          profile,
          inputFrameBytes: bytes,
        });
      } catch (cause) {
        observed = cause;
      }
      expect(observed).toBeInstanceOf(CppCuteBrowserWasmCompilerError);
      expect(fixture.factoryState.state).toBe("prepared");
      expect(fixture.calls).toEqual({ alloc: 0, compile: 0, free: 0, reset: 0 });
      expect(fixture.vfsState.state).toBe("active");
    }
  });

  it("rejects a profile that differs from the bound VFS authority and closes it", async () => {
    const otherProfile = await prepareCppCuteFrontendProfile({
      ...createCppCuteBrowserProfileInput(),
      profileId: "browsergrad.compiler.cpp-cute.browser-clang-alternate@1",
    });
    const fixture = runtimeFixture();
    expectCompilerError(
      () => executeCppCuteBrowserWasmCompiler({
        factory: fixture.factory,
        profile: otherProfile,
        inputFrameBytes: FRAME_BYTES,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-MISMATCH",
      "$.input.profile",
    );
    expect(fixture.calls).toEqual({ alloc: 0, compile: 0, free: 0, reset: 0 });
    expect(fixture.vfsState.reason).toBe("failed");
  });
});

function runtimeFixture(options: RuntimeOptions = {}): RuntimeFixture {
  const deployment = unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile.deployment;
  if (deployment.mode !== "browser-local") throw new Error("expected browser profile");
  const memory = new WebAssembly.Memory({
    initial: deployment.compilerRuntime.memory.initialPages,
    maximum: deployment.compilerRuntime.memory.maximumPages,
  });
  const counters: AllocatorCounters = {
    current: 0n,
    peak: 0n,
    allocated: 0n,
    freed: 0n,
    allocations: 0n,
    frees: 0n,
    failed: 0n,
  };
  writeAllocatorRecord(memory, counters);
  writeFrontendWorkRecord(memory, 0, 1, 0n, zeroFrontendWork());
  const calls = { alloc: 0, compile: 0, free: 0, reset: 0 };
  let status = 1;
  let inputByteLength = 0;
  let resultPointer = 0;
  let resultByteLength = 0;

  const allocate = (byteLength: number): void => {
    counters.current += BigInt(byteLength);
    counters.allocated += BigInt(byteLength);
    counters.allocations += 1n;
    if (counters.current > counters.peak) counters.peak = counters.current;
    writeAllocatorRecord(memory, counters);
  };
  const release = (byteLength: number): void => {
    counters.current -= BigInt(byteLength);
    counters.freed += BigInt(byteLength);
    counters.frees += 1n;
    writeAllocatorRecord(memory, counters);
  };

  const facade = Object.freeze({
    _bg_cpp_cute_abi_version: () => 65_538,
    _bg_cpp_cute_alloc: (byteLength: number) => {
      calls.alloc += 1;
      inputByteLength = byteLength;
      allocate(byteLength);
      status = 2;
      return INPUT_POINTER;
    },
    _bg_cpp_cute_allocator_metrics_pointer: () => ALLOCATOR_RECORD_POINTER,
    _bg_cpp_cute_frontend_work_metrics_pointer: () => FRONTEND_WORK_RECORD_POINTER,
    _bg_cpp_cute_compile: (pointer: number, byteLength: number) => {
      calls.compile += 1;
      if (options.trapCompile === true) throw new WebAssembly.RuntimeError("test trap");
      if (pointer !== INPUT_POINTER || byteLength !== inputByteLength) {
        throw new Error("engine supplied the wrong input range");
      }
      expect(new Uint8Array(memory.buffer, pointer, byteLength)).toEqual(FRAME_BYTES);
      const compileReturn = options.compileReturn ?? 0;
      status = options.readableCompileStatus ?? compileReturn;
      if (compileReturn === 0) {
        resultPointer = options.resultPointer ?? RESULT_POINTER;
        resultByteLength = options.resultByteLength ?? ARTIFACT_BYTES.byteLength;
        allocate(resultByteLength);
        new Uint8Array(memory.buffer, resultPointer, ARTIFACT_BYTES.byteLength).set(ARTIFACT_BYTES);
        writeFrontendWorkRecord(memory, 2, 1, 1n, {
          includeDepth: 1n,
          macroExpansions: 2n,
          preprocessedTokens: 64n,
          astNodes: 32n,
          constexprSteps: 4n,
          templateInstantiations: 8n,
          templateDepth: 2n,
          completedSemanticPasses: 2n,
        });
      }
      return compileReturn;
    },
    _bg_cpp_cute_free: (pointer: number, byteLength: number) => {
      calls.free += 1;
      if (pointer !== INPUT_POINTER || byteLength !== inputByteLength) {
        throw new Error("engine freed the wrong input range");
      }
      release(byteLength);
    },
    _bg_cpp_cute_reset: () => {
      calls.reset += 1;
      release(resultByteLength);
      resultPointer = 0;
      resultByteLength = 0;
      inputByteLength = 0;
      status = 1;
      writeFrontendWorkRecord(memory, 0, 1, 1n, zeroFrontendWork());
    },
    _bg_cpp_cute_result_length: () => resultByteLength,
    _bg_cpp_cute_result_pointer: () => resultPointer,
    _bg_cpp_cute_status: () => status,
  });
  const session = Object.freeze({
    profileHash: profile.profileHash,
  }) as PreparedCppCuteBrowserVfsSession;
  const vfsState: MockVfsState = {
    state: "active",
    reason: undefined,
    closeCalls: 0,
    profileHash: profile.profileHash,
    receipt: Object.freeze({}),
  };
  authorities.sessions.set(session, vfsState);
  const taken = Object.freeze({
    instance: Object.freeze({}),
    memory,
    vfsSession: session,
    moduleFacade: facade,
    stdout: Object.freeze([]),
    stderr: Object.freeze([]),
  }) as unknown as TakenCppCuteBrowserEmscriptenFactory;
  const factory = Object.freeze({
    authority: "package-generated-factory-instantiation-only",
    protocol: "browsergrad.compiler.cpp-cute.emscripten-factory-binding@1",
    wasmSha256: WASM_SHA256,
    wasmByteLength: 1024,
    cAbiVersion: 65_538,
    allocatorMetricsPointer: ALLOCATOR_RECORD_POINTER,
    frontendWorkMetricsPointer: FRONTEND_WORK_RECORD_POINTER,
    generatedImportCount: 66,
    vfsImportCount: 6,
    networkAuthorityGranted: false,
    workerExecutionObserved: false,
    loweringAuthorityMinted: false,
  }) as PreparedCppCuteBrowserEmscriptenFactory;
  const factoryState: MockFactoryState = { state: "prepared", taken };
  authorities.factories.set(factory, factoryState);
  return { factory, factoryState, vfsState, calls };
}

function writeAllocatorRecord(memory: WebAssembly.Memory, counters: AllocatorCounters): void {
  const bytes = new Uint8Array(
    memory.buffer,
    ALLOCATOR_RECORD_POINTER,
    CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH,
  );
  bytes.fill(0);
  bytes.set(new TextEncoder().encode(CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_MAGIC));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  view.setUint32(8, CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_VERSION, true);
  view.setUint32(12, CPP_CUTE_BROWSER_ALLOCATOR_METRICS_RECORD_BYTE_LENGTH, true);
  view.setBigUint64(16, counters.current, true);
  view.setBigUint64(24, counters.peak, true);
  view.setBigUint64(32, counters.allocated, true);
  view.setBigUint64(40, counters.freed, true);
  view.setBigUint64(48, counters.allocations, true);
  view.setBigUint64(56, counters.frees, true);
  view.setBigUint64(64, counters.failed, true);
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

function zeroFrontendWork(): FrontendWorkCounters {
  return {
    includeDepth: 0n,
    macroExpansions: 0n,
    preprocessedTokens: 0n,
    astNodes: 0n,
    constexprSteps: 0n,
    templateInstantiations: 0n,
    templateDepth: 0n,
    completedSemanticPasses: 0n,
  };
}

function writeFrontendWorkRecord(
  memory: WebAssembly.Memory,
  phase: number,
  flags: number,
  generation: bigint,
  counters: FrontendWorkCounters,
): void {
  const bytes = new Uint8Array(
    memory.buffer,
    FRONTEND_WORK_RECORD_POINTER,
    CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_BYTE_LENGTH,
  );
  bytes.fill(0);
  bytes.set(new TextEncoder().encode(CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_MAGIC));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  view.setUint32(8, CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_VERSION, true);
  view.setUint32(12, CPP_CUTE_BROWSER_FRONTEND_WORK_RECORD_BYTE_LENGTH, true);
  view.setUint32(16, phase, true);
  view.setUint32(20, flags, true);
  view.setBigUint64(24, generation, true);
  view.setBigUint64(32, counters.includeDepth, true);
  view.setBigUint64(40, counters.macroExpansions, true);
  view.setBigUint64(48, counters.preprocessedTokens, true);
  view.setBigUint64(56, counters.astNodes, true);
  view.setBigUint64(64, counters.constexprSteps, true);
  view.setBigUint64(72, counters.templateInstantiations, true);
  view.setBigUint64(80, counters.templateDepth, true);
  view.setBigUint64(88, counters.completedSemanticPasses, true);
}

function zeroVfsCounters(): Readonly<Record<string, string>> {
  return Object.freeze({
    totalSessionCalls: "0",
    statusCalls: "0",
    openCalls: "0",
    readCalls: "0",
    closeCalls: "0",
    directoryCountCalls: "0",
    directoryEntryCalls: "0",
    currentLiveHandles: "0",
    peakLiveHandles: "0",
    currentLiveSourceLogicalReservationByteLength: "0",
    currentLiveInstalledVfsLogicalReservationByteLength: "0",
    currentLiveLogicalReservationByteLength: "0",
    peakLiveLogicalReservationByteLength: "0",
    indexedNodes: "1",
    indexLogicalByteLength: "1",
    logicalOpenedSourceByteLength: "0",
    logicalOpenedInstalledVfsByteLength: "0",
    logicalOpenedTotalByteLength: "0",
  });
}

function expectCompilerError(
  operation: () => unknown,
  code: CppCuteBrowserWasmCompilerError["code"],
  path: string,
): void {
  let observed: unknown;
  try {
    operation();
  } catch (cause) {
    observed = cause;
  }
  expect(observed).toBeInstanceOf(CppCuteBrowserWasmCompilerError);
  expect(observed).toMatchObject({ code, path });
}

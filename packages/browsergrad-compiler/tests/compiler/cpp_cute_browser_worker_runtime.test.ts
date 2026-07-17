import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it, vi } from "vitest";

interface InvocationState {
  record: {
    readonly rawWasmConformance: {
      readonly wasmSha256: string;
      readonly wasmByteLength: number;
    };
  };
  readonly clangWasmBytes: Uint8Array;
  active: boolean;
  discardCalls: number;
  copyCalls: number;
  discardFailure: Error | undefined;
}

interface VfsState {
  state: "active" | "disposed";
  readonly requestId: string;
  readonly profileHash: string;
  cancelCalls: number;
  importCalls: number;
  cancelFailure: Error | undefined;
}

const authorities = vi.hoisted(() => ({
  invocations: new WeakMap<object, InvocationState>(),
  frames: new WeakMap<object, Uint8Array>(),
  vfs: new WeakMap<object, VfsState>(),
}));

vi.mock("../../src/cpp_cute_browser_worker_protocol.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_worker_protocol.js")>();
  const invocation = (value: object): InvocationState => {
    const stored = authorities.invocations.get(value);
    if (stored === undefined) {
      throw new actual.CppCuteBrowserWorkerProtocolError(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNVERIFIED",
        "$.invocation",
        "unregistered invocation",
      );
    }
    if (!stored.active) {
      throw new actual.CppCuteBrowserWorkerProtocolError(
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-DUPLICATE-OR-LATE",
        "$.invocation",
        "terminal invocation",
      );
    }
    return stored;
  };
  return {
    ...actual,
    unwrapPreparedCppCuteBrowserWorkerInvocation: (value: object) => invocation(value).record,
    copyCppCuteBrowserWorkerClangWasmBytes: (value: object) => {
      const stored = invocation(value);
      stored.copyCalls += 1;
      return new Uint8Array(stored.clangWasmBytes);
    },
    discardCppCuteBrowserWorkerInvocation: (value: object) => {
      const stored = invocation(value);
      stored.discardCalls += 1;
      if (stored.discardFailure !== undefined) throw stored.discardFailure;
      stored.active = false;
      return Object.freeze({});
    },
  };
});

vi.mock("../../src/cpp_cute_browser_input_frame.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_input_frame.js")>();
  return {
    ...actual,
    copyPreparedCppCuteBrowserInputFrameBytes: (value: object) => {
      const bytes = authorities.frames.get(value);
      if (bytes === undefined) {
        throw new actual.CppCuteBrowserInputFrameError(
          "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-UNVERIFIED",
          "$.prepared",
          "unregistered input frame",
        );
      }
      return new Uint8Array(bytes);
    },
  };
});

vi.mock("../../src/cpp_cute_browser_vfs_session.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_vfs_session.js")>();
  const session = (value: object): VfsState => {
    const stored = authorities.vfs.get(value);
    if (stored === undefined) {
      throw new actual.CppCuteBrowserVfsSessionError(
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED",
        "$.session",
        "unregistered VFS session",
      );
    }
    return stored;
  };
  return {
    ...actual,
    observeCppCuteBrowserVfsSession: (value: object) => {
      const stored = session(value);
      return Object.freeze({
        state: stored.state,
        requestId: stored.requestId,
        profileHash: stored.profileHash,
      });
    },
    createCppCuteBrowserVfsHostImports: (value: object) => {
      const stored = session(value);
      if (stored.state !== "active") throw new Error("terminal VFS session");
      stored.importCalls += 1;
      return Object.freeze({
        bg_vfs_status: () => 0,
        bg_vfs_open: () => 0,
        bg_vfs_read: () => 0,
        bg_vfs_close: () => 0,
        bg_vfs_directory_count: () => 0,
        bg_vfs_directory_entry: () => 0,
      });
    },
    cancelCppCuteBrowserVfsSession: (value: object) => {
      const stored = session(value);
      if (stored.state !== "active") throw new Error("terminal VFS session");
      stored.cancelCalls += 1;
      if (stored.cancelFailure !== undefined) throw stored.cancelFailure;
      stored.state = "disposed";
      return Object.freeze({});
    },
  };
});

import {
  CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS,
  CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS,
  CPP_CUTE_BROWSER_WORKER_RUNTIME_PROTOCOL,
  inspectCppCuteBrowserWorkerRuntimeBinding,
  prepareCppCuteBrowserWorkerRuntimeBinding,
  startCppCuteBrowserWorkerRuntime,
  type PreparedCppCuteBrowserWorkerRuntimeBinding,
} from "../../src/cpp_cute_browser_worker_runtime.js";
import type { PreparedCppCuteBrowserInputFrame } from "../../src/cpp_cute_browser_input_frame.js";
import type { PreparedCppCuteBrowserWorkerInvocation } from "../../src/cpp_cute_browser_worker_protocol.js";
import type { PreparedCppCuteBrowserVfsSession } from "../../src/cpp_cute_browser_vfs_session.js";

const INVOCATION_ID = `bg.cpp.browser-worker-invocation.sha256.${"1".repeat(64)}`;
const REQUEST_ID = `bg.cpp.frontend-request.sha256.${"2".repeat(64)}`;
const PROFILE_HASH = "3".repeat(64);
const CLANG_WASM_BYTES = Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0);
const INPUT_FRAME_BYTES = new TextEncoder().encode("BGCCABI1-runtime-frame");

interface RuntimeFixture {
  readonly invocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly inputFrame: PreparedCppCuteBrowserInputFrame;
  readonly vfsSession: PreparedCppCuteBrowserVfsSession;
  readonly invocationState: InvocationState;
  readonly vfsState: VfsState;
}

function bindingInput(fixture: RuntimeFixture): {
  readonly invocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly inputFrame: PreparedCppCuteBrowserInputFrame;
  readonly vfsSession: PreparedCppCuteBrowserVfsSession;
} {
  return {
    invocation: fixture.invocation,
    inputFrame: fixture.inputFrame,
    vfsSession: fixture.vfsSession,
  };
}

async function runtimeFixture(
  options: {
    readonly frameInvocationId?: string;
    readonly vfsRequestId?: string;
    readonly vfsState?: "active" | "disposed";
    readonly expectedWasmSha256?: string;
    readonly discardFailure?: Error;
    readonly cancelFailure?: Error;
  } = {},
): Promise<RuntimeFixture> {
  const clangWasmSha256 = options.expectedWasmSha256 ?? await sha256Hex(CLANG_WASM_BYTES);
  const frameSha256 = await sha256Hex(INPUT_FRAME_BYTES);
  const invocation = Object.freeze({
    invocationId: INVOCATION_ID,
    requestId: REQUEST_ID,
    profileHash: PROFILE_HASH,
  }) as PreparedCppCuteBrowserWorkerInvocation;
  const inputFrame = Object.freeze({
    invocationId: options.frameInvocationId ?? INVOCATION_ID,
    frameSha256,
    frameByteLength: INPUT_FRAME_BYTES.byteLength,
  }) as PreparedCppCuteBrowserInputFrame;
  const vfsSession = Object.freeze({
    sessionOrdinal: 7,
    requestId: options.vfsRequestId ?? REQUEST_ID,
    profileHash: PROFILE_HASH,
  }) as PreparedCppCuteBrowserVfsSession;
  const invocationState: InvocationState = {
    record: {
      rawWasmConformance: {
        wasmSha256: clangWasmSha256,
        wasmByteLength: CLANG_WASM_BYTES.byteLength,
      },
    },
    clangWasmBytes: new Uint8Array(CLANG_WASM_BYTES),
    active: true,
    discardCalls: 0,
    copyCalls: 0,
    discardFailure: options.discardFailure,
  };
  const vfsState: VfsState = {
    state: options.vfsState ?? "active",
    requestId: options.vfsRequestId ?? REQUEST_ID,
    profileHash: PROFILE_HASH,
    cancelCalls: 0,
    importCalls: 0,
    cancelFailure: options.cancelFailure,
  };
  authorities.invocations.set(invocation, invocationState);
  authorities.frames.set(inputFrame, new Uint8Array(INPUT_FRAME_BYTES));
  authorities.vfs.set(vfsSession, vfsState);
  return { invocation, inputFrame, vfsSession, invocationState, vfsState };
}

describe("package-owned C++/CuTe Worker runtime boundary", () => {
  it("binds one verified invocation/frame/VFS composition without execution authority", async () => {
    const fixture = await runtimeFixture();
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));

    expect(binding).toMatchObject({
      authority: "package-worker-runtime-binding-only",
      protocol: CPP_CUTE_BROWSER_WORKER_RUNTIME_PROTOCOL,
      invocationId: INVOCATION_ID,
      requestId: REQUEST_ID,
      profileHash: PROFILE_HASH,
      bundleStatus: CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS,
      blockers: CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS,
      networkAuthorityGranted: false,
      workerExecutionObserved: false,
      workerTerminationObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(fixture.invocationState.copyCalls).toBe(1);
    expect(fixture.vfsState.importCalls).toBe(1);
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding)).toMatchObject({
      state: "prepared",
      nativeIntrinsicSnapshot:
        "byte-copy-hash-wasm-object-inspection-and-authority-bookkeeping",
      requiredWasmConstructionIntrinsicsAvailable: true,
      networkAuthorityGranted: false,
      factoryInvoked: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
  });

  it("fails start closed, retires invocation/VFS, and never claims execution", async () => {
    const fixture = await runtimeFixture();
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));

    await expect(startCppCuteBrowserWorkerRuntime(binding)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY",
      path: "$.bundle",
    });
    expect(fixture.vfsState.cancelCalls).toBe(1);
    expect(fixture.invocationState.discardCalls).toBe(1);
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding)).toMatchObject({
      state: "blocked-terminal",
      factoryInvoked: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    await expect(startCppCuteBrowserWorkerRuntime(binding)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE",
      path: "$.binding",
    });
  });

  it("settles both blocked-runtime owners and aggregates cleanup failures", async () => {
    const fixture = await runtimeFixture({
      cancelFailure: new Error("injected VFS cancellation failure"),
      discardFailure: new Error("injected invocation discard failure"),
    });
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));

    const observed: unknown = await startCppCuteBrowserWorkerRuntime(binding)
      .then(() => undefined, (error: unknown) => error);
    expect(observed).toBeInstanceOf(Error);
    expect(observed).toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CLEANUP",
      path: "$.binding.cleanup",
      cause: expect.any(AggregateError),
    });
    if (!(observed instanceof Error) || !(observed.cause instanceof AggregateError)) {
      throw new Error("expected aggregate runtime cleanup failure");
    }
    expect(observed.cause.errors).toHaveLength(2);
    expect(fixture.vfsState.cancelCalls).toBe(1);
    expect(fixture.invocationState.discardCalls).toBe(1);
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding).state).toBe("blocked-terminal");
  });

  it("rejects mismatched frame and VFS identities before reserving the invocation", async () => {
    const wrongFrame = await runtimeFixture({ frameInvocationId: `${INVOCATION_ID}-wrong` });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(wrongFrame))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
      path: "$.input.inputFrame",
    });
    const wrongVfs = await runtimeFixture({ vfsRequestId: `${REQUEST_ID}-wrong` });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(wrongVfs))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
      path: "$.input.vfsSession",
    });
  });

  it("releases reservations after transferred-byte hash mismatch", async () => {
    const fixture = await runtimeFixture({ expectedWasmSha256: "f".repeat(64) });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
      path: "$.input.invocation.clangWasmBytes",
    });
    fixture.invocationState.record = {
      rawWasmConformance: {
        wasmSha256: await sha256Hex(CLANG_WASM_BYTES),
        wasmByteLength: CLANG_WASM_BYTES.byteLength,
      },
    };
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture))).resolves.toMatchObject({
      invocationId: INVOCATION_ID,
    });
  });

  it("prevents invocation, frame, or VFS reuse across runtime bindings", async () => {
    const fixture = await runtimeFixture();
    await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-DUPLICATE",
      path: "$.input.invocation",
    });

    const reusedFrame = await runtimeFixture();
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding({
      invocation: reusedFrame.invocation,
      inputFrame: fixture.inputFrame,
      vfsSession: reusedFrame.vfsSession,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-DUPLICATE",
      path: "$.input.inputFrame",
    });

    const reusedVfs = await runtimeFixture();
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding({
      invocation: reusedVfs.invocation,
      inputFrame: reusedVfs.inputFrame,
      vfsSession: fixture.vfsSession,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-DUPLICATE",
      path: "$.input.vfsSession",
    });
  });

  it("rechecks liveness after async hashing and releases the failed reservation", async () => {
    const invocationRace = await runtimeFixture();
    const invocationPending = prepareCppCuteBrowserWorkerRuntimeBinding(
      bindingInput(invocationRace),
    );
    invocationRace.invocationState.active = false;
    await expect(invocationPending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-DUPLICATE-OR-LATE",
    });
    invocationRace.invocationState.active = true;
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(invocationRace)))
      .resolves.toMatchObject({ invocationId: INVOCATION_ID });

    const vfsRace = await runtimeFixture();
    const vfsPending = prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(vfsRace));
    vfsRace.vfsState.state = "disposed";
    await expect(vfsPending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE",
      path: "$.input.vfsSession",
    });
    vfsRace.vfsState.state = "active";
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(vfsRace)))
      .resolves.toMatchObject({ invocationId: INVOCATION_ID });
  });

  it("rejects terminal or forged authorities", async () => {
    const terminalVfs = await runtimeFixture({ vfsState: "disposed" });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(terminalVfs))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE",
      path: "$.input.vfsSession",
    });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding({
      invocation: { ...terminalVfs.invocation } as never,
      inputFrame: terminalVfs.inputFrame,
      vfsSession: terminalVfs.vfsSession,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-UNVERIFIED",
    });

    let frameGetterRead = false;
    const hostileFrame = Object.defineProperty({}, "invocationId", {
      enumerable: true,
      get: () => {
        frameGetterRead = true;
        throw new Error("forged frame getter must not run");
      },
    }) as PreparedCppCuteBrowserInputFrame;
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding({
      invocation: terminalVfs.invocation,
      inputFrame: hostileFrame,
      vfsSession: terminalVfs.vfsSession,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-UNVERIFIED",
    });
    expect(frameGetterRead).toBe(false);
  });

  it("rejects extra ambient-acquisition fields without invoking accessors", async () => {
    const fixture = await runtimeFixture();
    let accessorRead = false;
    const hostile = Object.defineProperty(bindingInput(fixture), "ambientLoader", {
      enumerable: true,
      get: () => {
        accessorRead = true;
        throw new Error("ambient loader must not be read");
      },
    });

    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(hostile as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-INVALID",
      path: "$.input",
    });
    expect(accessorRead).toBe(false);

    const proxy = new Proxy(bindingInput(fixture), {
      getPrototypeOf: () => {
        throw new Error("prototype trap must become a typed failure");
      },
    });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(proxy)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-INVALID",
      path: "$.input",
    });
  });

  it("uses the captured byte-copy and hash-format intrinsics after ambient mutation", async () => {
    const fixture = await runtimeFixture();
    const set = vi.spyOn(Uint8Array.prototype, "set").mockImplementation(() => {
      throw new Error("ambient Uint8Array.set must not be used");
    });
    const test = vi.spyOn(RegExp.prototype, "test").mockImplementation(() => false);
    try {
      await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture)))
        .resolves.toMatchObject({ invocationId: INVOCATION_ID });
    } finally {
      set.mockRestore();
      test.mockRestore();
    }
  });

  it("keeps structural copies outside the opaque runtime issuer", async () => {
    const fixture = await runtimeFixture();
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));
    expect(() => inspectCppCuteBrowserWorkerRuntimeBinding(
      { ...binding } as unknown as PreparedCppCuteBrowserWorkerRuntimeBinding,
    )).toThrow(/WORKER-RUNTIME-UNVERIFIED/u);
  });
});

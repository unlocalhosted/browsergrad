import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it, vi } from "vitest";

interface InvocationState {
  record: {
    readonly rawWasmConformance: {
      readonly wasmSha256: string;
      readonly wasmByteLength: number;
    };
  };
  active: boolean;
  unwrapCalls: number;
  discardCalls: number;
  discardReasons: string[];
  discardFailure: Error | undefined;
}

interface FrameState {
  readonly bytes: Uint8Array;
  copyCalls: number;
}

interface MountState {
  state: "prepared" | "bound" | "discarded";
  readonly requestId: string;
  readonly profileHash: string;
  readonly mountOrdinal: number;
  readonly imports: VfsImports;
  observeCalls: number;
  importCalls: number;
  discardCalls: number;
  discardFailure: Error | undefined;
}

interface RealmInputRecord {
  readonly profile: object;
  readonly request: object;
  readonly vfsInstallation: object;
  readonly runtimeAbiAsset: object;
  readonly rawWasmConformance: object;
  readonly invocation: object;
  readonly inputFrame: object;
  readonly vfsMount: object;
  readonly vfsImports: VfsImports;
  readonly clangWasmBytes: Uint8Array;
}

interface RealmInputState {
  state: "prepared" | "adopted";
  active: RealmInputRecord | null;
  takeCalls: number;
}

interface VfsImports {
  readonly bg_vfs_status: () => number;
  readonly bg_vfs_open: () => number;
  readonly bg_vfs_read: () => number;
  readonly bg_vfs_close: () => number;
  readonly bg_vfs_directory_count: () => number;
  readonly bg_vfs_directory_entry: () => number;
}

const authorities = vi.hoisted(() => ({
  invocations: new WeakMap<object, InvocationState>(),
  frames: new WeakMap<object, FrameState>(),
  mounts: new WeakMap<object, MountState>(),
  realmInputs: new WeakMap<object, RealmInputState>(),
}));

function authorityError(name: string, code: string, path: string, message: string): Error {
  const error = new Error(`${code}: ${message}`);
  return Object.assign(error, { name, code, path });
}

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
    unwrapPreparedCppCuteBrowserWorkerInvocation: (value: object) => {
      const stored = invocation(value);
      stored.unwrapCalls += 1;
      return stored.record;
    },
    discardCppCuteBrowserWorkerInvocation: (value: object, reason: string) => {
      const stored = invocation(value);
      stored.discardCalls += 1;
      stored.discardReasons.push(reason);
      stored.active = false;
      if (stored.discardFailure !== undefined) throw stored.discardFailure;
    },
  };
});

vi.mock("../../src/cpp_cute_browser_input_frame.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_input_frame.js")>();
  return {
    ...actual,
    copyPreparedCppCuteBrowserInputFrameBytes: (value: object) => {
      const stored = authorities.frames.get(value);
      if (stored === undefined) {
        throw new actual.CppCuteBrowserInputFrameError(
          "BG-COMPILER-CPP-CUTE-BROWSER-INPUT-FRAME-UNVERIFIED",
          "$.prepared",
          "unregistered input frame",
        );
      }
      stored.copyCalls += 1;
      return new Uint8Array(stored.bytes);
    },
  };
});

vi.mock("../../src/cpp_cute_browser_vfs_session.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_vfs_session.js")>();
  const mount = (value: object): MountState => {
    const stored = authorities.mounts.get(value);
    if (stored === undefined) {
      throw new actual.CppCuteBrowserVfsSessionError(
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED",
        "$.mount",
        "unregistered VFS mount",
      );
    }
    return stored;
  };
  return {
    ...actual,
    observeCppCuteBrowserVfsMount: (value: object) => {
      const stored = mount(value);
      stored.observeCalls += 1;
      return Object.freeze({
        state: stored.state,
        requestId: stored.requestId,
        profileHash: stored.profileHash,
        mountOrdinal: stored.mountOrdinal,
      });
    },
    createCppCuteBrowserVfsMountHostImports: (value: object) => {
      const stored = mount(value);
      if (stored.state !== "prepared") throw new Error("terminal VFS mount");
      stored.importCalls += 1;
      return stored.imports;
    },
    discardCppCuteBrowserVfsMount: (value: object) => {
      const stored = mount(value);
      if (stored.state !== "prepared") throw new Error("VFS mount is not discardable");
      stored.discardCalls += 1;
      stored.state = "discarded";
      if (stored.discardFailure !== undefined) throw stored.discardFailure;
    },
  };
});

vi.mock("../../src/cpp_cute_browser_worker_transfer.js", () => ({
  takeCppCuteBrowserWorkerRealmInput: (value: object) => {
    const stored = authorities.realmInputs.get(value);
    if (stored === undefined) {
      throw authorityError(
        "CppCuteBrowserWorkerTransferError",
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-UNVERIFIED",
        "$.prepared",
        "unregistered Worker-realm input",
      );
    }
    stored.takeCalls += 1;
    if (stored.state !== "prepared" || stored.active === null) {
      throw authorityError(
        "CppCuteBrowserWorkerTransferError",
        "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        "$.prepared",
        "Worker-realm input is terminal",
      );
    }
    const active = stored.active;
    stored.state = "adopted";
    stored.active = null;
    return active;
  },
}));

import {
  CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS,
  CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS,
  CPP_CUTE_BROWSER_WORKER_RUNTIME_PROTOCOL,
  discardCppCuteBrowserWorkerRuntimeBinding,
  inspectCppCuteBrowserWorkerRuntimeBinding,
  prepareCppCuteBrowserWorkerRuntimeBinding,
  startCppCuteBrowserWorkerRuntime,
  type PreparedCppCuteBrowserWorkerRuntimeBinding,
} from "../../src/cpp_cute_browser_worker_runtime.js";
import type { PreparedCppCuteBrowserInputFrame } from "../../src/cpp_cute_browser_input_frame.js";
import type { PreparedCppCuteBrowserWorkerInvocation } from "../../src/cpp_cute_browser_worker_protocol.js";
import type { PreparedCppCuteBrowserVfsMount } from "../../src/cpp_cute_browser_vfs_session.js";
import type { PreparedCppCuteBrowserWorkerRealmInput } from "../../src/cpp_cute_browser_worker_transfer.js";

const INVOCATION_ID = `bg.cpp.browser-worker-invocation.sha256.${"1".repeat(64)}`;
const REQUEST_ID = `bg.cpp.frontend-request.sha256.${"2".repeat(64)}`;
const PROFILE_HASH = "3".repeat(64);
const CLANG_WASM_BYTES = Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0);
const INPUT_FRAME_BYTES = new TextEncoder().encode("BGCCABI1-runtime-frame");
const VFS_SESSION_CLOSED = 9;

interface RuntimeFixture {
  readonly realmInput: PreparedCppCuteBrowserWorkerRealmInput;
  readonly invocation: PreparedCppCuteBrowserWorkerInvocation;
  readonly inputFrame: PreparedCppCuteBrowserInputFrame;
  readonly vfsMount: PreparedCppCuteBrowserVfsMount;
  readonly vfsImports: VfsImports;
  readonly invocationState: InvocationState;
  readonly frameState: FrameState;
  readonly mountState: MountState;
  readonly realmInputState: RealmInputState;
  readonly transferredClangWasmBytes: Uint8Array;
}

interface RuntimeFixtureOptions {
  readonly frameInvocationId?: string;
  readonly frameSha256?: string;
  readonly mountRequestId?: string;
  readonly mountProfileHash?: string;
  readonly mountState?: "prepared" | "bound" | "discarded";
  readonly realmInvocationId?: string;
  readonly realmRequestId?: string;
  readonly realmProfileHash?: string;
  readonly realmInputFrameSha256?: string;
  readonly realmInputFrameByteLength?: number;
  readonly realmClangWasmSha256?: string;
  readonly realmClangWasmByteLength?: number;
  readonly realmMountOrdinal?: number;
  readonly expectedWasmSha256?: string;
  readonly adoptedImportsDiffer?: boolean;
  readonly discardFailure?: Error;
  readonly mountDiscardFailure?: Error;
}

function bindingInput(fixture: RuntimeFixture): {
  readonly realmInput: PreparedCppCuteBrowserWorkerRealmInput;
} {
  return { realmInput: fixture.realmInput };
}

async function runtimeFixture(
  options: RuntimeFixtureOptions = {},
): Promise<RuntimeFixture> {
  const actualClangWasmSha256 = await sha256Hex(CLANG_WASM_BYTES);
  const expectedClangWasmSha256 = options.expectedWasmSha256 ?? actualClangWasmSha256;
  const actualFrameSha256 = await sha256Hex(INPUT_FRAME_BYTES);
  const invocation = Object.freeze({
    invocationId: INVOCATION_ID,
    requestId: REQUEST_ID,
    profileHash: PROFILE_HASH,
  }) as PreparedCppCuteBrowserWorkerInvocation;
  const inputFrame = Object.freeze({
    invocationId: options.frameInvocationId ?? INVOCATION_ID,
    frameSha256: options.frameSha256 ?? actualFrameSha256,
    frameByteLength: INPUT_FRAME_BYTES.byteLength,
  }) as PreparedCppCuteBrowserInputFrame;
  const vfsImports = Object.freeze({
    bg_vfs_status: () => VFS_SESSION_CLOSED,
    bg_vfs_open: () => VFS_SESSION_CLOSED,
    bg_vfs_read: () => VFS_SESSION_CLOSED,
    bg_vfs_close: () => VFS_SESSION_CLOSED,
    bg_vfs_directory_count: () => VFS_SESSION_CLOSED,
    bg_vfs_directory_entry: () => VFS_SESSION_CLOSED,
  });
  const vfsMount = Object.freeze({
    mountOrdinal: 7,
    requestId: options.mountRequestId ?? REQUEST_ID,
    profileHash: options.mountProfileHash ?? PROFILE_HASH,
  }) as PreparedCppCuteBrowserVfsMount;
  const invocationState: InvocationState = {
    record: {
      rawWasmConformance: {
        wasmSha256: expectedClangWasmSha256,
        wasmByteLength: CLANG_WASM_BYTES.byteLength,
      },
    },
    active: true,
    unwrapCalls: 0,
    discardCalls: 0,
    discardReasons: [],
    discardFailure: options.discardFailure,
  };
  const frameState: FrameState = {
    bytes: new Uint8Array(INPUT_FRAME_BYTES),
    copyCalls: 0,
  };
  const mountState: MountState = {
    state: options.mountState ?? "prepared",
    requestId: options.mountRequestId ?? REQUEST_ID,
    profileHash: options.mountProfileHash ?? PROFILE_HASH,
    mountOrdinal: 7,
    imports: vfsImports,
    observeCalls: 0,
    importCalls: 0,
    discardCalls: 0,
    discardFailure: options.mountDiscardFailure,
  };
  const transferredClangWasmBytes = new Uint8Array(CLANG_WASM_BYTES);
  const adoptedImports = options.adoptedImportsDiffer
    ? Object.freeze({ ...vfsImports })
    : vfsImports;
  const active: RealmInputRecord = Object.freeze({
    profile: Object.freeze({}),
    request: Object.freeze({}),
    vfsInstallation: Object.freeze({}),
    runtimeAbiAsset: Object.freeze({}),
    rawWasmConformance: Object.freeze({}),
    invocation,
    inputFrame,
    vfsMount,
    vfsImports: adoptedImports,
    clangWasmBytes: transferredClangWasmBytes,
  });
  const realmInput = Object.freeze({
    authority: "realm-local-runtime-input-only",
    invocationId: options.realmInvocationId ?? INVOCATION_ID,
    requestId: options.realmRequestId ?? REQUEST_ID,
    profileHash: options.realmProfileHash ?? PROFILE_HASH,
    inputFrameSha256: options.realmInputFrameSha256 ?? actualFrameSha256,
    inputFrameByteLength:
      options.realmInputFrameByteLength ?? INPUT_FRAME_BYTES.byteLength,
    clangWasmSha256: options.realmClangWasmSha256 ?? actualClangWasmSha256,
    clangWasmByteLength:
      options.realmClangWasmByteLength ?? CLANG_WASM_BYTES.byteLength,
    vfsMountOrdinal: options.realmMountOrdinal ?? 7,
    networkAuthorityGranted: false,
    workerExecutionObserved: false,
    workerTerminationObserved: false,
    loweringAuthorityMinted: false,
  }) as PreparedCppCuteBrowserWorkerRealmInput;
  const realmInputState: RealmInputState = {
    state: "prepared",
    active,
    takeCalls: 0,
  };
  authorities.invocations.set(invocation, invocationState);
  authorities.frames.set(inputFrame, frameState);
  authorities.mounts.set(vfsMount, mountState);
  authorities.realmInputs.set(realmInput, realmInputState);
  return {
    realmInput,
    invocation,
    inputFrame,
    vfsMount,
    vfsImports,
    invocationState,
    frameState,
    mountState,
    realmInputState,
    transferredClangWasmBytes,
  };
}

describe("package-owned C++/CuTe Worker runtime boundary", () => {
  it("adopts one reconstructed realm input with exact identity and stable pre-bind imports", async () => {
    const fixture = await runtimeFixture();
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));
    const inputFrameSha256 = await sha256Hex(INPUT_FRAME_BYTES);
    const clangWasmSha256 = await sha256Hex(CLANG_WASM_BYTES);

    expect(binding).toMatchObject({
      authority: "package-worker-runtime-binding-only",
      protocol: CPP_CUTE_BROWSER_WORKER_RUNTIME_PROTOCOL,
      invocationId: INVOCATION_ID,
      requestId: REQUEST_ID,
      profileHash: PROFILE_HASH,
      inputFrameSha256,
      inputFrameByteLength: INPUT_FRAME_BYTES.byteLength,
      clangWasmSha256,
      clangWasmByteLength: CLANG_WASM_BYTES.byteLength,
      vfsMountOrdinal: 7,
      bundleStatus: CPP_CUTE_BROWSER_WORKER_RUNTIME_BUNDLE_STATUS,
      blockers: CPP_CUTE_BROWSER_WORKER_RUNTIME_BLOCKERS,
      networkAuthorityGranted: false,
      workerExecutionObserved: false,
      workerTerminationObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(fixture.realmInputState).toMatchObject({ state: "adopted", takeCalls: 1 });
    expect(fixture.frameState.copyCalls).toBe(2);
    expect(fixture.invocationState.unwrapCalls).toBe(2);
    expect(fixture.mountState).toMatchObject({
      state: "prepared",
      observeCalls: 2,
      importCalls: 1,
      discardCalls: 0,
    });
    expect(Object.values(fixture.vfsImports).map((operation) => operation())).toEqual(
      Array(6).fill(VFS_SESSION_CLOSED),
    );
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding)).toMatchObject({
      state: "prepared",
      invocationId: INVOCATION_ID,
      inputFrameByteLength: INPUT_FRAME_BYTES.byteLength,
      clangWasmByteLength: CLANG_WASM_BYTES.byteLength,
      vfsMountOrdinal: 7,
      nativeIntrinsicSnapshot:
        "byte-copy-hash-wasm-object-inspection-and-authority-bookkeeping",
      requiredWasmConstructionIntrinsicsAvailable: true,
      networkAuthorityGranted: false,
      factoryInvoked: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(binding).not.toHaveProperty("factory");
    expect(binding).not.toHaveProperty("workerModuleBytes");
    expect(binding).not.toHaveProperty("vfsSessionOrdinal");
  });

  it("takes one realm input exactly once", async () => {
    const fixture = await runtimeFixture();
    await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));

    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        path: "$.prepared",
      });
    expect(fixture.realmInputState).toMatchObject({ state: "adopted", takeCalls: 2 });
  });

  it("fails start closed, terminalizes both adopted owners, and never claims execution", async () => {
    const fixture = await runtimeFixture();
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));

    await expect(startCppCuteBrowserWorkerRuntime(binding)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY",
      path: "$.bundle",
    });
    expect(fixture.mountState).toMatchObject({ state: "discarded", discardCalls: 1 });
    expect(fixture.invocationState).toMatchObject({
      active: false,
      discardCalls: 1,
      discardReasons: ["worker-unavailable"],
    });
    expect(fixture.transferredClangWasmBytes).toEqual(new Uint8Array(CLANG_WASM_BYTES.byteLength));
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

  it("settles both blocked-start owners and aggregates dual cleanup failures", async () => {
    const fixture = await runtimeFixture({
      mountDiscardFailure: new Error("injected VFS mount discard failure"),
      discardFailure: new Error("injected invocation discard failure"),
    });
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));

    const observed: unknown = await startCppCuteBrowserWorkerRuntime(binding)
      .then(() => undefined, (error: unknown) => error);
    expect(observed).toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CLEANUP",
      path: "$.binding.cleanup",
      cause: expect.any(AggregateError),
    });
    if (!(observed instanceof Error) || !(observed.cause instanceof AggregateError)) {
      throw new Error("expected aggregate runtime cleanup failure");
    }
    expect(observed.cause.errors).toHaveLength(2);
    expect(fixture.mountState).toMatchObject({ state: "discarded", discardCalls: 1 });
    expect(fixture.invocationState).toMatchObject({ active: false, discardCalls: 1 });
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding).state).toBe("blocked-terminal");
  });

  it("discards a prepared binding without attempting execution and rejects terminal reuse", async () => {
    const fixture = await runtimeFixture();
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));

    discardCppCuteBrowserWorkerRuntimeBinding(binding);

    expect(fixture.mountState).toMatchObject({ state: "discarded", discardCalls: 1 });
    expect(fixture.invocationState).toMatchObject({
      active: false,
      discardCalls: 1,
      discardReasons: ["abandoned"],
    });
    expect(fixture.transferredClangWasmBytes).toEqual(
      new Uint8Array(CLANG_WASM_BYTES.byteLength),
    );
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding)).toMatchObject({
      state: "discarded",
      factoryInvoked: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(() => discardCppCuteBrowserWorkerRuntimeBinding(binding)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE",
        path: "$.binding",
      }),
    );
    await expect(startCppCuteBrowserWorkerRuntime(binding)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE",
      path: "$.binding",
    });
  });

  it("keeps discard terminal after aggregating both owner-cleanup failures", async () => {
    const fixture = await runtimeFixture({
      mountDiscardFailure: new Error("injected discard mount failure"),
      discardFailure: new Error("injected discard invocation failure"),
    });
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));

    expect(() => discardCppCuteBrowserWorkerRuntimeBinding(binding)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CLEANUP",
        path: "$.binding.cleanup",
        cause: expect.any(AggregateError),
      }),
    );
    expect(fixture.mountState).toMatchObject({ state: "discarded", discardCalls: 1 });
    expect(fixture.invocationState).toMatchObject({ active: false, discardCalls: 1 });
    expect(fixture.transferredClangWasmBytes).toEqual(
      new Uint8Array(CLANG_WASM_BYTES.byteLength),
    );
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding)).toMatchObject({
      state: "discarded",
      factoryInvoked: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(() => discardCppCuteBrowserWorkerRuntimeBinding(binding)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE",
        path: "$.binding",
      }),
    );
    await expect(startCppCuteBrowserWorkerRuntime(binding)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-STATE",
      path: "$.binding",
    });
  });

  it("owns cleanup after an adopted Wasm or frame hash mismatch", async () => {
    const wasmMismatch = await runtimeFixture({ expectedWasmSha256: "f".repeat(64) });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(wasmMismatch)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
        path: "$.input.invocation.clangWasmBytes",
      });
    expect(wasmMismatch.realmInputState.state).toBe("adopted");
    expect(wasmMismatch.mountState).toMatchObject({ state: "discarded", discardCalls: 1 });
    expect(wasmMismatch.invocationState).toMatchObject({ active: false, discardCalls: 1 });
    expect(wasmMismatch.transferredClangWasmBytes).toEqual(
      new Uint8Array(CLANG_WASM_BYTES.byteLength),
    );

    const frameMismatch = await runtimeFixture({ frameSha256: "e".repeat(64) });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(frameMismatch)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
        path: "$.input.inputFrame",
      });
    expect(frameMismatch.realmInputState.state).toBe("adopted");
    expect(frameMismatch.mountState.state).toBe("discarded");
    expect(frameMismatch.invocationState.active).toBe(false);
    expect(frameMismatch.transferredClangWasmBytes).toEqual(
      new Uint8Array(CLANG_WASM_BYTES.byteLength),
    );
  });

  const projectionContinuityCases = [
    {
      label: "input-frame hash",
      options: { realmInputFrameSha256: "d".repeat(64) },
      path: "$.input.inputFrame",
    },
    {
      label: "input-frame byte length",
      options: { realmInputFrameByteLength: INPUT_FRAME_BYTES.byteLength + 1 },
      path: "$.input.inputFrame",
    },
    {
      label: "Clang-Wasm byte length",
      options: { realmClangWasmByteLength: CLANG_WASM_BYTES.byteLength + 1 },
      path: "$.input.invocation.clangWasmBytes",
    },
    {
      label: "VFS mount ordinal",
      options: { realmMountOrdinal: 8 },
      path: "$.input.realmInput.vfsMount",
    },
  ] satisfies readonly {
    readonly label: string;
    readonly options: RuntimeFixtureOptions;
    readonly path: string;
  }[];

  it.each(projectionContinuityCases)(
    "rejects realm projection drift for $label",
    async ({ options, path }) => {
      const fixture = await runtimeFixture(options);

      await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture)))
        .rejects.toMatchObject({
          code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
          path,
        });
      expect(fixture.realmInputState.state).toBe("adopted");
      expect(fixture.mountState.state).toBe("discarded");
      expect(fixture.invocationState.active).toBe(false);
      expect(fixture.transferredClangWasmBytes).toEqual(
        new Uint8Array(CLANG_WASM_BYTES.byteLength),
      );
    },
  );

  it("retains the preparation failure when adopted-owner cleanup also fails", async () => {
    const fixture = await runtimeFixture({
      expectedWasmSha256: "f".repeat(64),
      mountDiscardFailure: new Error("injected mount cleanup failure"),
      discardFailure: new Error("injected invocation cleanup failure"),
    });

    const observed: unknown = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture))
      .then(() => undefined, (error: unknown) => error);
    expect(observed).toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CLEANUP",
      path: "$.binding.cleanup",
      cause: expect.any(AggregateError),
    });
    if (!(observed instanceof Error) || !(observed.cause instanceof AggregateError)) {
      throw new Error("expected aggregate preparation cleanup failure");
    }
    expect(observed.cause.errors).toHaveLength(3);
    expect(observed.cause.errors[0]).toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
      path: "$.input.invocation.clangWasmBytes",
    });
    expect(fixture.mountState.state).toBe("discarded");
    expect(fixture.invocationState.active).toBe(false);
  });

  it("rejects realm projection, mount, and stable-import mismatches after adoption", async () => {
    const wrongRealm = await runtimeFixture({ realmInvocationId: `${INVOCATION_ID}-wrong` });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(wrongRealm)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
        path: "$.input.realmInput.inputFrame",
      });
    expect(wrongRealm.mountState.state).toBe("discarded");
    expect(wrongRealm.invocationState.active).toBe(false);

    const wrongMount = await runtimeFixture({ mountRequestId: `${REQUEST_ID}-wrong` });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(wrongMount)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
        path: "$.input.realmInput.vfsMount",
      });
    expect(wrongMount.mountState.state).toBe("discarded");
    expect(wrongMount.invocationState.active).toBe(false);

    const wrongImports = await runtimeFixture({ adoptedImportsDiffer: true });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(wrongImports)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-MISMATCH",
        path: "$.input.realmInput.vfsImports",
      });
    expect(wrongImports.mountState.state).toBe("discarded");
    expect(wrongImports.invocationState.active).toBe(false);
  });

  it("rejects forged or hostile binding input without invoking ambient accessors", async () => {
    const forged = Object.freeze({
      invocationId: INVOCATION_ID,
      requestId: REQUEST_ID,
      profileHash: PROFILE_HASH,
    }) as PreparedCppCuteBrowserWorkerRealmInput;
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding({ realmInput: forged }))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-UNVERIFIED",
        path: "$.prepared",
      });

    const fixture = await runtimeFixture();
    let accessorRead = false;
    const hostile = Object.defineProperty(bindingInput(fixture), "ambientFactory", {
      enumerable: true,
      get: () => {
        accessorRead = true;
        throw new Error("ambient factory must not be read");
      },
    });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(hostile as never))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-INVALID",
        path: "$.input",
      });
    expect(accessorRead).toBe(false);
    expect(fixture.realmInputState.takeCalls).toBe(0);

    const proxy = new Proxy(bindingInput(fixture), {
      getPrototypeOf: () => {
        throw new Error("prototype trap must become a typed failure");
      },
    });
    await expect(prepareCppCuteBrowserWorkerRuntimeBinding(proxy))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-INVALID",
        path: "$.input",
      });
    expect(fixture.realmInputState.takeCalls).toBe(0);
  });

  it("uses captured byte-copy, zeroing, and hash-format intrinsics", async () => {
    const fixture = await runtimeFixture();
    const set = vi.spyOn(Uint8Array.prototype, "set").mockImplementation(() => {
      throw new Error("ambient Uint8Array.set must not be used");
    });
    const fill = vi.spyOn(Uint8Array.prototype, "fill").mockImplementation(() => {
      throw new Error("ambient Uint8Array.fill must not be used");
    });
    const test = vi.spyOn(RegExp.prototype, "test").mockImplementation(() => false);
    try {
      const binding = await prepareCppCuteBrowserWorkerRuntimeBinding(bindingInput(fixture));
      await expect(startCppCuteBrowserWorkerRuntime(binding)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-CAPABILITY",
      });
    } finally {
      set.mockRestore();
      fill.mockRestore();
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

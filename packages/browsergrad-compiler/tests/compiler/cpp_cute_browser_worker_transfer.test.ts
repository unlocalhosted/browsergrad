import {
  canonicalJsonBytes,
  sha256Hex,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { beforeEach, describe, expect, it, vi } from "vitest";

interface MockProfileRecord {
  readonly profile: JsonValue;
}

interface MockManifestAsset {
  readonly assetId: string;
  readonly kind: "clang-extractor-wasm" | "dependency-header-pack";
}

interface MockManifestValue {
  readonly schema: string;
  readonly body: {
    readonly assets: readonly MockManifestAsset[];
  };
}

interface MockManifestRecord {
  readonly manifest: MockManifestValue;
  readonly bytes: Uint8Array;
}

interface MockAssetSetRecord {
  readonly manifest: object;
  readonly assets: readonly MockManifestAsset[];
  readonly bytes: ReadonlyMap<string, Uint8Array>;
}

interface MockInstallationRecord {
  readonly assetSet: object;
}

interface MockRequestRecord {
  readonly request: JsonValue;
  readonly sourceSnapshots: readonly MockSourceSnapshot[];
}

interface MockSourceSnapshot {
  readonly virtualPath: string;
  readonly bytes: Uint8Array;
}

interface MockInvocationValue {
  readonly invocationId: string;
  readonly invocationNonceSha256: string;
  readonly profileHash: string;
  readonly requestId: string;
}

interface MockInvocationRecord {
  readonly invocation: MockInvocationValue;
  readonly profile: object;
  readonly assetManifest: object;
  readonly vfsInstallation: object;
  readonly request: object & {
    readonly requestId: string;
    readonly profileHash: string;
    readonly sourceFileCount: number;
  };
  readonly sourceSnapshots: readonly MockSourceSnapshot[];
  readonly invocationBytes: Uint8Array;
  readonly profileBytes: Uint8Array;
  readonly requestBytes: Uint8Array;
  readonly clangWasmBytes: Uint8Array;
  readonly rawWasmConformance: {
    readonly wasmSha256: string;
    readonly wasmByteLength: number;
  };
}

const downstream = vi.hoisted(() => ({
  profileRecords: new WeakMap<object, MockProfileRecord>(),
  manifestRecords: new WeakMap<object, MockManifestRecord>(),
  assetSetRecords: new WeakMap<object, MockAssetSetRecord>(),
  installationRecords: new WeakMap<object, MockInstallationRecord>(),
  runtimeAbiRecords: new WeakMap<object, { readonly runtimeAbi: object }>(),
  requestRecords: new WeakMap<object, MockRequestRecord>(),
  invocationRecords: new WeakMap<object, MockInvocationRecord>(),
  inputFrameRecords: new WeakMap<object, { readonly bytes: Uint8Array }>(),
  mountRecords: new WeakMap<object, {
    state: "prepared" | "discarded";
    readonly requestId: string;
    readonly profileHash: string;
    readonly mountOrdinal: number;
    imports: object | undefined;
  }>(),
  expectedAssetBytes: new Map<string, Uint8Array>(),
  discardMount: vi.fn<(mount: object) => void>(),
  discardInvocation: vi.fn<(invocation: object, reason: string) => void>(),
  createImports: vi.fn<(mount: object) => object>(),
  prepareInputFrame: vi.fn<(invocation: object) => Promise<object>>(),
  nextMountOrdinal: 1,
}));

function required<T>(map: WeakMap<object, T>, value: unknown, label: string): T {
  if (typeof value !== "object" || value === null) throw new Error(`mock ${label} is not an object`);
  const record = map.get(value);
  if (record === undefined) throw new Error(`unregistered mock ${label}`);
  return record;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function parseJsonBytes(bytes: Uint8Array): Record<string, unknown> {
  return JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
}

vi.mock("../../src/cpp_cute_browser_assets.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_assets.js")>();
  return {
    ...actual,
    canonicalCppCuteBrowserAssetManifestBytes: (manifest: object) =>
      new Uint8Array(required(downstream.manifestRecords, manifest, "manifest").bytes),
    decodeCppCuteBrowserAssetManifest: async (bytes: Uint8Array) => {
      const value = parseJsonBytes(bytes) as unknown as MockManifestValue;
      const manifest = Object.freeze({ manifestId: "mock-worker-local-manifest" });
      downstream.manifestRecords.set(manifest, {
        manifest: value,
        bytes: new Uint8Array(bytes),
      });
      return manifest;
    },
    unwrapPreparedCppCuteBrowserAssetManifest: (manifest: object) =>
      required(downstream.manifestRecords, manifest, "manifest"),
  };
});

vi.mock("../../src/cpp_cute_browser_asset_installation.js", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../src/cpp_cute_browser_asset_installation.js")
  >();
  return {
    ...actual,
    copyVerifiedCppCuteBrowserAssetBytes: (assetSet: object, assetId: string) => {
      const bytes = required(downstream.assetSetRecords, assetSet, "asset set").bytes.get(assetId);
      if (bytes === undefined) throw new Error(`missing mock asset ${assetId}`);
      return new Uint8Array(bytes);
    },
    decodeAcquiredCppCuteBrowserRuntimeAbiAsset: async () => {
      const runtimeAbiAsset = Object.freeze({ runtimeAbiManifestId: "mock-runtime-abi" });
      downstream.runtimeAbiRecords.set(runtimeAbiAsset, {
        runtimeAbi: Object.freeze({ manifestId: "mock-runtime-abi" }),
      });
      return runtimeAbiAsset;
    },
    installCppCuteBrowserVfs: async (assetSet: object) => {
      const installation = Object.freeze({ installationId: "mock-worker-local-vfs" });
      downstream.installationRecords.set(installation, { assetSet });
      return installation;
    },
    unwrapVerifiedCppCuteBrowserAssetSet: (assetSet: object) =>
      required(downstream.assetSetRecords, assetSet, "asset set"),
    unwrapVerifiedCppCuteBrowserRuntimeAbiAsset: (runtimeAbiAsset: object) =>
      required(downstream.runtimeAbiRecords, runtimeAbiAsset, "runtime ABI"),
    unwrapVerifiedCppCuteBrowserVfsInstallation: (installation: object) =>
      required(downstream.installationRecords, installation, "VFS installation"),
    verifyTransferredCppCuteBrowserAssetSet: async (
      manifest: object,
      assets: readonly { readonly assetId: string; readonly bytes: Uint8Array }[],
    ) => {
      const manifestRecord = required(downstream.manifestRecords, manifest, "manifest");
      if (assets.length !== manifestRecord.manifest.body.assets.length) {
        throw new Error("mock transferred asset cardinality mismatch");
      }
      const bytes = new Map<string, Uint8Array>();
      for (let index = 0; index < assets.length; index += 1) {
        const asset = assets[index]!;
        const expectedDescriptor = manifestRecord.manifest.body.assets[index];
        const expectedBytes = downstream.expectedAssetBytes.get(asset.assetId);
        if (expectedDescriptor?.assetId !== asset.assetId || expectedBytes === undefined ||
            !equalBytes(asset.bytes, expectedBytes)) {
          throw new Error("mock transferred asset mismatch");
        }
        bytes.set(asset.assetId, new Uint8Array(asset.bytes));
      }
      const assetSet = Object.freeze({ assetSetSha256: "mock-worker-local-assets" });
      downstream.assetSetRecords.set(assetSet, {
        manifest,
        assets: manifestRecord.manifest.body.assets,
        bytes,
      });
      return assetSet;
    },
  };
});

vi.mock("../../src/cpp_cute_browser_input_frame.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_input_frame.js")>();
  return {
    ...actual,
    copyPreparedCppCuteBrowserInputFrameBytes: (frame: object) =>
      new Uint8Array(required(downstream.inputFrameRecords, frame, "input frame").bytes),
    prepareCppCuteBrowserInputFrame: async (invocation: object) => {
      const projection = await downstream.prepareInputFrame(invocation) as {
        readonly frameSha256: string;
        readonly frameByteLength: number;
      };
      const record = required(downstream.invocationRecords, invocation, "invocation");
      const frame = Object.freeze({
        invocationId: record.invocation.invocationId,
        ...projection,
      });
      downstream.inputFrameRecords.set(frame, { bytes: new Uint8Array(INPUT_FRAME_BYTES) });
      return frame;
    },
  };
});

vi.mock("../../src/cpp_cute_browser_worker_protocol.js", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../src/cpp_cute_browser_worker_protocol.js")
  >();
  return {
    ...actual,
    canonicalCppCuteBrowserWorkerInvocationBytes: (invocation: object) =>
      new Uint8Array(required(downstream.invocationRecords, invocation, "invocation").invocationBytes),
    canonicalCppCuteBrowserWorkerProfileRegionBytes: (invocation: object) =>
      new Uint8Array(required(downstream.invocationRecords, invocation, "invocation").profileBytes),
    canonicalCppCuteBrowserWorkerRequestRegionBytes: (invocation: object) =>
      new Uint8Array(required(downstream.invocationRecords, invocation, "invocation").requestBytes),
    copyCppCuteBrowserWorkerClangWasmBytes: (invocation: object) =>
      new Uint8Array(required(downstream.invocationRecords, invocation, "invocation").clangWasmBytes),
    copyCppCuteBrowserWorkerSourceSnapshots: (invocation: object) =>
      required(downstream.invocationRecords, invocation, "invocation").sourceSnapshots.map(
        (source) => ({ virtualPath: source.virtualPath, bytes: new Uint8Array(source.bytes) }),
      ),
    decodeCppCuteBrowserWorkerInvocation: async (
      bytes: Uint8Array,
      input: {
        readonly profile: object & { readonly profileHash: string };
        readonly assetManifest: object;
        readonly vfsInstallation: object;
        readonly request: object & {
          readonly requestId: string;
          readonly profileHash: string;
          readonly sourceFileCount: number;
        };
        readonly rawWasmConformance: {
          readonly wasmSha256: string;
          readonly wasmByteLength: number;
        };
      },
    ) => {
      const value = parseJsonBytes(bytes) as unknown as MockInvocationValue;
      if (value.profileHash !== input.profile.profileHash ||
          value.requestId !== input.request.requestId) {
        throw new Error("mock strict invocation mismatch");
      }
      const installation = required(
        downstream.installationRecords,
        input.vfsInstallation,
        "VFS installation",
      );
      const clangWasmBytes = required(
        downstream.assetSetRecords,
        installation.assetSet,
        "asset set",
      ).bytes.get("clang-wasm");
      if (clangWasmBytes === undefined) throw new Error("missing mock Clang Wasm");
      const invocation = Object.freeze({
        invocationId: value.invocationId,
        requestId: value.requestId,
        profileHash: value.profileHash,
      });
      downstream.invocationRecords.set(invocation, {
        invocation: value,
        profile: input.profile,
        assetManifest: input.assetManifest,
        vfsInstallation: input.vfsInstallation,
        request: input.request,
        sourceSnapshots: required(
          downstream.requestRecords,
          input.request,
          "request",
        ).sourceSnapshots,
        invocationBytes: new Uint8Array(bytes),
        profileBytes: canonicalJsonBytes(
          required(downstream.profileRecords, input.profile, "profile").profile,
        ),
        requestBytes: canonicalJsonBytes(
          required(downstream.requestRecords, input.request, "request").request,
        ),
        clangWasmBytes: new Uint8Array(clangWasmBytes),
        rawWasmConformance: input.rawWasmConformance,
      });
      return invocation;
    },
    discardCppCuteBrowserWorkerInvocation: (invocation: object, reason: string) =>
      downstream.discardInvocation(invocation, reason),
    unwrapPreparedCppCuteBrowserWorkerInvocation: (invocation: object) =>
      required(downstream.invocationRecords, invocation, "invocation"),
  };
});

vi.mock("../../src/cpp_cute_frontend_request.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_frontend_request.js")>();
  return {
    ...actual,
    copyPreparedCppCuteFrontendSourceSnapshots: (request: object) =>
      required(downstream.requestRecords, request, "request").sourceSnapshots.map(
        (source) => ({ virtualPath: source.virtualPath, bytes: new Uint8Array(source.bytes) }),
      ),
    prepareCppCuteFrontendRequest: async (
      profile: object & { readonly profileHash: string },
      value: Record<string, unknown>,
      sources: readonly MockSourceSnapshot[],
    ) => {
      if (value["profileHash"] !== profile.profileHash ||
          value["sourceFileCount"] !== sources.length) {
        throw new Error("mock request binding mismatch");
      }
      const request = Object.freeze({
        requestId: String(value["requestId"]),
        profileHash: profile.profileHash,
        sourceFileCount: sources.length,
      });
      downstream.requestRecords.set(request, {
        request: value as JsonValue,
        sourceSnapshots: sources.map((source) => ({
          virtualPath: source.virtualPath,
          bytes: new Uint8Array(source.bytes),
        })),
      });
      return request;
    },
    unwrapPreparedCppCuteFrontendRequest: (request: object) =>
      required(downstream.requestRecords, request, "request"),
  };
});

vi.mock("../../src/cpp_cute_frontend_profile.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_frontend_profile.js")>();
  return {
    ...actual,
    prepareCppCuteFrontendProfile: async (value: Record<string, unknown>) => {
      const profile = Object.freeze({ profileHash: String(value["profileHash"]) });
      downstream.profileRecords.set(profile, { profile: value as JsonValue });
      return profile;
    },
    unwrapPreparedCppCuteBrowserFrontendProfile: (profile: object) =>
      required(downstream.profileRecords, profile, "profile"),
  };
});

vi.mock("../../src/cpp_cute_browser_vfs_session.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/cpp_cute_browser_vfs_session.js")>();
  return {
    ...actual,
    createCppCuteBrowserVfsMountHostImports: (mount: object) => {
      const record = required(downstream.mountRecords, mount, "VFS mount");
      if (record.state !== "prepared") throw new Error("mock VFS mount is terminal");
      record.imports ??= downstream.createImports(mount);
      return record.imports;
    },
    discardCppCuteBrowserVfsMount: (mount: object) => {
      const record = required(downstream.mountRecords, mount, "VFS mount");
      if (record.state !== "prepared") throw new Error("mock VFS mount is terminal");
      record.state = "discarded";
      return downstream.discardMount(mount);
    },
    observeCppCuteBrowserVfsMount: (mount: object) => {
      const record = required(downstream.mountRecords, mount, "VFS mount");
      return Object.freeze({
        state: record.state,
        requestId: record.requestId,
        profileHash: record.profileHash,
        mountOrdinal: record.mountOrdinal,
      });
    },
    prepareCppCuteBrowserVfsMount: (input: {
      readonly request: { readonly requestId: string; readonly profileHash: string };
    }) => {
      const mountOrdinal = downstream.nextMountOrdinal++;
      const mount = Object.freeze({ mountOrdinal });
      downstream.mountRecords.set(mount, {
        state: "prepared",
        requestId: input.request.requestId,
        profileHash: input.request.profileHash,
        mountOrdinal,
        imports: undefined,
      });
      return mount;
    },
  };
});

vi.mock("../../src/cpp_cute_browser_wasm_inspection.js", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("../../src/cpp_cute_browser_wasm_inspection.js")
  >();
  return {
    ...actual,
    // Deliberate boundary mock: no pinned, reviewed Clang Wasm exists yet.
    verifyCppCuteBrowserWasmConformance: async (bytes: Uint8Array) => Object.freeze({
      wasmSha256: await sha256Hex(bytes),
      wasmByteLength: bytes.byteLength,
    }),
  };
});

import {
  CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
  CPP_CUTE_BROWSER_WORKER_TRANSFER_REGION_BYTE_LIMIT,
  discardCppCuteBrowserWorkerRealmInput,
  discardCppCuteBrowserWorkerTransfer,
  inspectCppCuteBrowserWorkerRealmInput,
  prepareCppCuteBrowserWorkerTransfer,
  reconstructCppCuteBrowserWorkerTransfer,
  takeCppCuteBrowserWorkerRealmInput,
  takeCppCuteBrowserWorkerTransfer,
  type CppCuteBrowserWorkerTransferMessage,
} from "../../src/cpp_cute_browser_worker_transfer.js";
import {
  discardCppCuteBrowserWorkerRuntimeBinding,
  inspectCppCuteBrowserWorkerRuntimeBinding,
  prepareCppCuteBrowserWorkerRuntimeBinding,
} from "../../src/cpp_cute_browser_worker_runtime.js";
import {
  handleCppCuteBrowserWorkerTransfer,
  installCppCuteBrowserWorkerEntry,
  type CppCuteBrowserWorkerEntryMessageListener,
  type CppCuteBrowserWorkerEntryScope,
} from "../../src/cpp_cute_browser_worker_entry.js";
import type {
  CppCuteBrowserWorkerControllerInboundMessage,
} from "../../src/cpp_cute_browser_worker_messages.js";

const PROFILE_HASH = "a".repeat(64);
const REQUEST_HASH = "b".repeat(64);
const INVOCATION_HASH = "c".repeat(64);
const NONCE = "e".repeat(64);
const INVOCATION_ID = `bg.cpp.browser-worker-invocation.sha256.${INVOCATION_HASH}`;
const REQUEST_ID = `bg.cpp.frontend-request.sha256.${REQUEST_HASH}`;
const SOURCE_BYTES = new TextEncoder().encode("auto layout = make_layout(Int<2>{});");
const CLANG_WASM_BYTES = Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0, 17);
const HEADER_PACK_BYTES = Uint8Array.of(66, 71, 86, 70, 83, 1, 2, 3);
const INPUT_FRAME_BYTES = new TextEncoder().encode("BGCCABI1-worker-transfer-integration");

interface TestEnvironment {
  readonly invocation: object;
  readonly sourceSnapshots: readonly MockSourceSnapshot[];
}

beforeEach(async () => {
  downstream.expectedAssetBytes.clear();
  downstream.expectedAssetBytes.set("clang-wasm", new Uint8Array(CLANG_WASM_BYTES));
  downstream.expectedAssetBytes.set("headers", new Uint8Array(HEADER_PACK_BYTES));
  downstream.discardMount.mockReset();
  downstream.discardInvocation.mockReset();
  downstream.createImports.mockReset();
  downstream.createImports.mockImplementation(() => Object.freeze({
    bg_vfs_status: () => -1,
  }));
  downstream.prepareInputFrame.mockReset();
  downstream.prepareInputFrame.mockResolvedValue(Object.freeze({
    frameSha256: await sha256Hex(INPUT_FRAME_BYTES),
    frameByteLength: INPUT_FRAME_BYTES.byteLength,
  }));
  downstream.nextMountOrdinal = 1;
});

describe("C++/CuTe browser Worker transfer boundary", () => {
  it("reserves one host transfer per invocation and rejects duplicate preparation", () => {
    const environment = createEnvironment();
    const prepared = prepareCppCuteBrowserWorkerTransfer(environment.invocation as never);

    expect(prepared).toMatchObject({
      authority: "host-prepared-worker-transfer-only",
      protocol: CPP_CUTE_BROWSER_WORKER_TRANSFER_PROTOCOL,
      invocationId: INVOCATION_ID,
      assetCount: 2,
      sourceSnapshotCount: 1,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(() => prepareCppCuteBrowserWorkerTransfer(environment.invocation as never))
      .toThrowError(expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        path: "$.invocation",
      }));
  });

  it("discards one untaken host transfer and terminalizes its invocation exactly once", () => {
    const environment = createEnvironment();
    const prepared = prepareCppCuteBrowserWorkerTransfer(environment.invocation as never);

    discardCppCuteBrowserWorkerTransfer(prepared);

    expect(downstream.discardInvocation).toHaveBeenCalledTimes(1);
    expect(downstream.discardInvocation).toHaveBeenCalledWith(
      environment.invocation,
      "abandoned",
    );
    expect(() => discardCppCuteBrowserWorkerTransfer(prepared)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        path: "$.prepared",
      }),
    );
    expect(() => takeCppCuteBrowserWorkerTransfer(prepared)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        path: "$.prepared",
      }),
    );
    expect(downstream.discardInvocation).toHaveBeenCalledTimes(1);
  });

  it("rejects host discard after take and rejects forged or cleanup-failed repeat transitions", () => {
    const takenEnvironment = createEnvironment();
    const takenPrepared = prepareCppCuteBrowserWorkerTransfer(
      takenEnvironment.invocation as never,
    );
    takeCppCuteBrowserWorkerTransfer(takenPrepared);
    expect(() => discardCppCuteBrowserWorkerTransfer(takenPrepared)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        path: "$.prepared",
      }),
    );
    expect(() => takeCppCuteBrowserWorkerTransfer(takenPrepared)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        path: "$.prepared",
      }),
    );

    const forgedEnvironment = createEnvironment();
    const forgedPrepared = prepareCppCuteBrowserWorkerTransfer(
      forgedEnvironment.invocation as never,
    );
    const forgedProjection = structuredClone(forgedPrepared);
    expect(() => discardCppCuteBrowserWorkerTransfer(forgedProjection as never)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-UNVERIFIED",
        path: "$.prepared",
      }),
    );

    downstream.discardInvocation.mockImplementationOnce(() => {
      throw new Error("mock host invocation cleanup failure");
    });
    expect(() => discardCppCuteBrowserWorkerTransfer(forgedPrepared)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-CLEANUP",
        path: "$.cleanup.invocation",
      }),
    );
    expect(() => discardCppCuteBrowserWorkerTransfer(forgedPrepared)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        path: "$.prepared",
      }),
    );
    expect(downstream.discardInvocation).toHaveBeenCalledTimes(1);
  });

  it("takes one immutable envelope with exact standalone unique transfer buffers", () => {
    const environment = createEnvironment();
    const prepared = prepareCppCuteBrowserWorkerTransfer(environment.invocation as never);
    const taken = takeCppCuteBrowserWorkerTransfer(prepared);
    const views = messageByteViews(taken.message);

    expect(taken.transferList).toHaveLength(4 + 2 + 1);
    expect(new Set(taken.transferList).size).toBe(taken.transferList.length);
    expect(taken.transferList).toEqual(views.map((view) => view.buffer));
    for (const view of views) {
      expect(view.byteOffset).toBe(0);
      expect(view.buffer).toBeInstanceOf(ArrayBuffer);
      expect(view.buffer.byteLength).toBe(view.byteLength);
    }
    expect(Object.isFrozen(taken)).toBe(true);
    expect(Object.isFrozen(taken.message)).toBe(true);
    expect(Object.isFrozen(taken.message.assets)).toBe(true);
    expect(Object.isFrozen(taken.transferList)).toBe(true);

    const takenClangBytes = taken.message.assets[0]!.bytes;
    takenClangBytes[0] = (takenClangBytes[0] ?? 0) ^ 0xff;
    expect(downstream.expectedAssetBytes.get("clang-wasm")).toEqual(CLANG_WASM_BYTES);
    expect(() => takeCppCuteBrowserWorkerTransfer(prepared)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE",
        path: "$.prepared",
      }),
    );
  });

  it("reconstructs and adopts local authorities with raw-Wasm conformance mocked, not execution proof", async () => {
    const taken = takeEnvironment();
    const transferredViews = messageByteViews(taken.message);
    const realmInput = await reconstructCppCuteBrowserWorkerTransfer(taken.message);

    expect(transferredViews.every((view) => view.byteLength === 0)).toBe(true);
    expect(realmInput).toMatchObject({
      authority: "realm-local-runtime-input-only",
      invocationId: INVOCATION_ID,
      invocationNonceSha256: NONCE,
      requestId: REQUEST_ID,
      profileHash: PROFILE_HASH,
      inputFrameSha256: await sha256Hex(INPUT_FRAME_BYTES),
      inputFrameByteLength: INPUT_FRAME_BYTES.byteLength,
      clangWasmSha256: await sha256Hex(CLANG_WASM_BYTES),
      clangWasmByteLength: CLANG_WASM_BYTES.byteLength,
      vfsMountOrdinal: 1,
      networkAuthorityGranted: false,
      workerExecutionObserved: false,
      workerTerminationObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(inspectCppCuteBrowserWorkerRealmInput(realmInput).state).toBe("prepared");

    const adopted = takeCppCuteBrowserWorkerRealmInput(realmInput);
    expect(adopted.clangWasmBytes).toEqual(CLANG_WASM_BYTES);
    expect(adopted.vfsImports).toHaveProperty("bg_vfs_status");
    expect(inspectCppCuteBrowserWorkerRealmInput(realmInput).state).toBe("adopted");
    expect(() => takeCppCuteBrowserWorkerRealmInput(realmInput)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE" }),
    );
    expect(() => discardCppCuteBrowserWorkerRealmInput(realmInput)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE" }),
    );
    expect(downstream.discardMount).not.toHaveBeenCalled();
    expect(downstream.discardInvocation).not.toHaveBeenCalled();
  });

  it("adopts reconstructed authority through the real runtime boundary without minting execution", async () => {
    const realmInput = await reconstructCppCuteBrowserWorkerTransfer(takeEnvironment().message);
    const binding = await prepareCppCuteBrowserWorkerRuntimeBinding({ realmInput });

    expect(inspectCppCuteBrowserWorkerRealmInput(realmInput).state).toBe("adopted");
    expect(binding).toMatchObject({
      invocationId: INVOCATION_ID,
      requestId: REQUEST_ID,
      profileHash: PROFILE_HASH,
      inputFrameSha256: await sha256Hex(INPUT_FRAME_BYTES),
      clangWasmSha256: await sha256Hex(CLANG_WASM_BYTES),
      vfsMountOrdinal: 1,
      networkAuthorityGranted: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding)).toMatchObject({
      state: "prepared",
      factoryInvoked: false,
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });

    discardCppCuteBrowserWorkerRuntimeBinding(binding);
    expect(inspectCppCuteBrowserWorkerRuntimeBinding(binding).state).toBe("discarded");
    expect(downstream.discardMount).toHaveBeenCalledTimes(1);
    expect(downstream.discardInvocation).toHaveBeenCalledWith(
      expect.any(Object),
      "abandoned",
    );
  });

  it("runs canonical transfer through the Worker entry and reports only typed package-execution failure", async () => {
    const taken = takeEnvironment();
    const transferredViews = messageByteViews(taken.message);
    const terminalMessages: CppCuteBrowserWorkerControllerInboundMessage[] = [];

    await handleCppCuteBrowserWorkerTransfer(
      taken.message,
      (message) => terminalMessages.push(message),
    );

    expect(transferredViews.every((view) => view.byteLength === 0)).toBe(true);
    expect(terminalMessages).toEqual([{
      kind: "browsergrad-cpp-cute-worker-failure",
      version: 1,
      controllerProtocol: "browsergrad.compiler.cpp-cute.browser-worker-controller@1",
      invocationId: INVOCATION_ID,
      invocationNonceSha256: NONCE,
      phase: "runtime-start",
      failureCode: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-RUNTIME-EXECUTION",
      failurePath: "$.runtime.execution",
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    }]);
    expect(downstream.discardMount).toHaveBeenCalledTimes(1);
    expect(downstream.discardInvocation).toHaveBeenCalledWith(
      expect.any(Object),
      "worker-unavailable",
    );
  });

  it("installs one one-shot Worker message listener with no loader or network effect", async () => {
    const listeners = new Set<CppCuteBrowserWorkerEntryMessageListener>();
    const terminalMessages: CppCuteBrowserWorkerControllerInboundMessage[] = [];
    const queuedFailures: (() => void)[] = [];
    const scope: CppCuteBrowserWorkerEntryScope = {
      addEventListener: (_type, listener) => listeners.add(listener),
      removeEventListener: (_type, listener) => listeners.delete(listener),
      postMessage: (message, transfer) => {
        expect(transfer).toEqual([]);
        terminalMessages.push(message);
      },
      queueMicrotask: (callback) => queuedFailures.push(callback),
    };
    installCppCuteBrowserWorkerEntry(scope);
    expect(listeners.size).toBe(1);

    const listener = [...listeners][0]!;
    listener({ data: takeEnvironment().message });
    expect(listeners.size).toBe(0);
    for (let attempt = 0; terminalMessages.length === 0 && attempt < 20; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    expect(terminalMessages).toHaveLength(1);
    expect(terminalMessages[0]).toMatchObject({
      kind: "browsergrad-cpp-cute-worker-failure",
      phase: "runtime-start",
      workerExecutionObserved: false,
      loweringAuthorityMinted: false,
    });
    expect(queuedFailures).toEqual([]);
  });

  it("routes an untrusted pre-identity transfer rejection to Worker error, not a terminal frame", async () => {
    const listeners = new Set<CppCuteBrowserWorkerEntryMessageListener>();
    const terminalMessages: CppCuteBrowserWorkerControllerInboundMessage[] = [];
    const queuedFailures: (() => void)[] = [];
    installCppCuteBrowserWorkerEntry({
      addEventListener: (_type, listener) => listeners.add(listener),
      removeEventListener: (_type, listener) => listeners.delete(listener),
      postMessage: (message) => terminalMessages.push(message),
      queueMicrotask: (callback) => queuedFailures.push(callback),
    });
    [...listeners][0]!({ data: Object.freeze({ kind: "attacker" }) });
    for (let attempt = 0; queuedFailures.length === 0 && attempt < 20; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    expect(terminalMessages).toEqual([]);
    expect(queuedFailures).toHaveLength(1);
    expect(() => queuedFailures[0]!()).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID",
      }),
    );
  });

  it("rejects forged authorities, discards once, and rejects replay", async () => {
    const environment = createEnvironment();
    const prepared = prepareCppCuteBrowserWorkerTransfer(environment.invocation as never);
    const forgedPrepared = structuredClone(prepared);
    expect(() => takeCppCuteBrowserWorkerTransfer(forgedPrepared as never)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-UNVERIFIED",
        path: "$.prepared",
      }),
    );

    const taken = takeCppCuteBrowserWorkerTransfer(prepared);
    const realmInput = await reconstructCppCuteBrowserWorkerTransfer(taken.message);
    const forgedRealmInput = structuredClone(realmInput);
    expect(() => takeCppCuteBrowserWorkerRealmInput(forgedRealmInput as never)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-UNVERIFIED" }),
    );

    discardCppCuteBrowserWorkerRealmInput(realmInput);
    expect(downstream.discardMount).toHaveBeenCalledTimes(1);
    expect(downstream.discardInvocation).toHaveBeenCalledWith(expect.any(Object), "abandoned");
    expect(inspectCppCuteBrowserWorkerRealmInput(realmInput).state).toBe("discarded");
    expect(() => discardCppCuteBrowserWorkerRealmInput(realmInput)).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-STATE" }),
    );
    await expect(reconstructCppCuteBrowserWorkerTransfer(taken.message)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID",
      path: "$.message",
    });
  });

  it("rejects envelope invocation ID and nonce mismatches after consuming every region", async () => {
    for (const mutate of [
      (message: MutableTransferMessage) => {
        message.invocationId = `bg.cpp.browser-worker-invocation.sha256.${"1".repeat(64)}`;
      },
      (message: MutableTransferMessage) => {
        message.invocationNonceSha256 = "2".repeat(64);
      },
    ]) {
      const message = mutableMessage(takeEnvironment().message);
      mutate(message);
      const views = messageByteViews(message);
      await expect(reconstructCppCuteBrowserWorkerTransfer(message)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-MISMATCH",
        path: "$.invocationId",
      });
      expect(views.every((view) => view.byteLength === 0)).toBe(true);
    }
  });

  it("relies on strict invocation and asset admission mocks to reject mutated authority bytes", async () => {
    const invocationMessage = mutableMessage(takeEnvironment().message);
    const invocationValue = parseJsonBytes(invocationMessage.invocationBytes);
    invocationValue["profileHash"] = "3".repeat(64);
    invocationMessage.invocationBytes = canonicalJsonBytes(invocationValue as JsonValue);
    await expect(reconstructCppCuteBrowserWorkerTransfer(invocationMessage)).rejects.toThrow(
      "mock strict invocation mismatch",
    );
    expect(messageByteViews(invocationMessage).every((view) => view.byteLength === 0)).toBe(true);

    const assetMessage = mutableMessage(takeEnvironment().message);
    const transferredClangBytes = assetMessage.assets[0]!.bytes;
    transferredClangBytes[0] = (transferredClangBytes[0] ?? 0) ^ 0xff;
    await expect(reconstructCppCuteBrowserWorkerTransfer(assetMessage)).rejects.toThrow(
      "mock transferred asset mismatch",
    );
    expect(messageByteViews(assetMessage).every((view) => view.byteLength === 0)).toBe(true);
  });

  it("rejects unknown fields and accessor-bearing envelopes before detaching buffers", async () => {
    const unknown = mutableMessage(takeEnvironment().message) as MutableTransferMessage & {
      unexpected?: boolean;
    };
    unknown.unexpected = true;
    const unknownViews = messageByteViews(unknown);
    await expect(reconstructCppCuteBrowserWorkerTransfer(unknown)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID",
      path: "$.message",
    });
    expect(unknownViews.every((view) => view.byteLength > 0)).toBe(true);

    const accessor = mutableMessage(takeEnvironment().message);
    Object.defineProperty(accessor, "invocationId", {
      enumerable: true,
      get: () => INVOCATION_ID,
    });
    const accessorViews = messageByteViews(accessor);
    await expect(reconstructCppCuteBrowserWorkerTransfer(accessor)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID",
      path: "$.message.invocationId",
    });
    expect(accessorViews.every((view) => view.byteLength > 0)).toBe(true);
  });

  it("rejects aliased, partial, and detached byte regions before reconstruction", async () => {
    const aliased = mutableMessage(takeEnvironment().message);
    aliased.requestRegionBytes = aliased.profileRegionBytes;
    await expect(reconstructCppCuteBrowserWorkerTransfer(aliased)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID",
      path: "$.message.requestRegionBytes",
    });
    expect(aliased.profileRegionBytes.byteLength).toBeGreaterThan(0);

    const partial = mutableMessage(takeEnvironment().message);
    const backing = new Uint8Array(partial.invocationBytes.byteLength + 2);
    backing.set(partial.invocationBytes, 1);
    partial.invocationBytes = backing.subarray(1, backing.byteLength - 1);
    await expect(reconstructCppCuteBrowserWorkerTransfer(partial)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID",
      path: "$.message.invocationBytes",
    });
    expect(backing.byteLength).toBeGreaterThan(0);

    const detached = mutableMessage(takeEnvironment().message);
    structuredClone(detached.invocationBytes.buffer, {
      transfer: [detached.invocationBytes.buffer],
    });
    await expect(reconstructCppCuteBrowserWorkerTransfer(detached)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-INVALID",
      path: "$.message",
    });
  });

  it("enforces array-count and canonical-region byte ceilings before consumption", async () => {
    const tooManyAssets = mutableMessage(takeEnvironment().message);
    tooManyAssets.assets = Array.from({ length: 257 }, (_, index) => ({
      assetId: `asset-${index}`,
      bytes: new Uint8Array(),
    }));
    await expect(reconstructCppCuteBrowserWorkerTransfer(tooManyAssets)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-RESOURCE-LIMIT",
      path: "$.message.assets",
    });
    expect(tooManyAssets.profileRegionBytes.byteLength).toBeGreaterThan(0);

    const oversizedRegion = mutableMessage(takeEnvironment().message);
    oversizedRegion.profileRegionBytes = new Uint8Array(
      CPP_CUTE_BROWSER_WORKER_TRANSFER_REGION_BYTE_LIMIT + 1,
    );
    await expect(reconstructCppCuteBrowserWorkerTransfer(oversizedRegion)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-RESOURCE-LIMIT",
      path: "$.message.profileRegionBytes",
    });
    expect(oversizedRegion.requestRegionBytes.byteLength).toBeGreaterThan(0);
  });

  it("keeps every message buffer intact when cancellation is already requested", async () => {
    const taken = takeEnvironment();
    const views = messageByteViews(taken.message);
    const before = views.map((view) => new Uint8Array(view));
    const controller = new AbortController();
    controller.abort();

    await expect(reconstructCppCuteBrowserWorkerTransfer(
      taken.message,
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-CANCELLED",
      path: "$.options.signal",
    });
    expect(views.map((view) => new Uint8Array(view))).toEqual(before);
  });

  it("settles both owners before reporting discard cleanup failures", async () => {
    const realmInput = await reconstructCppCuteBrowserWorkerTransfer(takeEnvironment().message);
    downstream.discardMount.mockImplementationOnce(() => {
      throw new Error("mock mount cleanup failure");
    });
    downstream.discardInvocation.mockImplementationOnce(() => {
      throw new Error("mock invocation cleanup failure");
    });

    expect(() => discardCppCuteBrowserWorkerRealmInput(realmInput)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-CLEANUP",
        path: "$.cleanup",
      }),
    );
    expect(downstream.discardMount).toHaveBeenCalledTimes(1);
    expect(downstream.discardInvocation).toHaveBeenCalledTimes(1);
    expect(inspectCppCuteBrowserWorkerRealmInput(realmInput).state).toBe("discarded");
  });

  it("attempts mount and invocation cleanup when reconstruction fails after both are minted", async () => {
    downstream.createImports.mockImplementationOnce(() => {
      throw new Error("mock import construction failure");
    });
    downstream.discardMount.mockImplementationOnce(() => {
      throw new Error("mock mount cleanup failure");
    });
    downstream.discardInvocation.mockImplementationOnce(() => {
      throw new Error("mock invocation cleanup failure");
    });
    const message = takeEnvironment().message;

    await expect(reconstructCppCuteBrowserWorkerTransfer(message)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WORKER-TRANSFER-CLEANUP",
      path: "$.cleanup",
    });
    expect(downstream.discardMount).toHaveBeenCalledTimes(1);
    expect(downstream.discardInvocation).toHaveBeenCalledWith(
      expect.any(Object),
      "worker-unavailable",
    );
    expect(messageByteViews(message).every((view) => view.byteLength === 0)).toBe(true);
  });
});

type MutableTransferMessage = {
  -readonly [Key in keyof CppCuteBrowserWorkerTransferMessage]:
    Key extends "assets"
      ? Array<{ assetId: string; bytes: Uint8Array }>
      : Key extends "sourceSnapshots"
        ? Array<{ virtualPath: string; bytes: Uint8Array }>
        : CppCuteBrowserWorkerTransferMessage[Key];
};

function createEnvironment(): TestEnvironment {
  const profileValue = {
    schema: "browsergrad.test.worker-profile",
    profileHash: PROFILE_HASH,
  } satisfies JsonValue;
  const requestValue = {
    schema: "browsergrad.test.worker-request",
    requestId: REQUEST_ID,
    profileHash: PROFILE_HASH,
    sourceFileCount: 1,
  } satisfies JsonValue;
  const manifestValue: MockManifestValue = {
    schema: "browsergrad.test.worker-assets",
    body: {
      assets: [
        { assetId: "clang-wasm", kind: "clang-extractor-wasm" },
        { assetId: "headers", kind: "dependency-header-pack" },
      ],
    },
  };
  const invocationValue: MockInvocationValue = {
    invocationId: INVOCATION_ID,
    invocationNonceSha256: NONCE,
    profileHash: PROFILE_HASH,
    requestId: REQUEST_ID,
  };
  const sourceSnapshots = [{
    virtualPath: "/src/main.cu",
    bytes: new Uint8Array(SOURCE_BYTES),
  }];

  const profile = Object.freeze({ profileHash: PROFILE_HASH });
  downstream.profileRecords.set(profile, { profile: profileValue });
  const manifest = Object.freeze({ manifestId: "mock-host-manifest" });
  downstream.manifestRecords.set(manifest, {
    manifest: manifestValue,
    bytes: canonicalJsonBytes(manifestValue as unknown as JsonValue),
  });
  const hostAssetBytes = new Map<string, Uint8Array>([
    ["clang-wasm", new Uint8Array(CLANG_WASM_BYTES)],
    ["headers", new Uint8Array(HEADER_PACK_BYTES)],
  ]);
  const assetSet = Object.freeze({ assetSetSha256: "mock-host-assets" });
  downstream.assetSetRecords.set(assetSet, {
    manifest,
    assets: manifestValue.body.assets,
    bytes: hostAssetBytes,
  });
  const installation = Object.freeze({ installationId: "mock-host-vfs" });
  downstream.installationRecords.set(installation, { assetSet });
  const request = Object.freeze({
    requestId: REQUEST_ID,
    profileHash: PROFILE_HASH,
    sourceFileCount: sourceSnapshots.length,
  });
  downstream.requestRecords.set(request, {
    request: requestValue,
    sourceSnapshots,
  });
  const invocation = Object.freeze({
    invocationId: INVOCATION_ID,
    requestId: REQUEST_ID,
    profileHash: PROFILE_HASH,
  });
  downstream.invocationRecords.set(invocation, {
    invocation: invocationValue,
    profile,
    assetManifest: manifest,
    vfsInstallation: installation,
    request,
    sourceSnapshots,
    invocationBytes: canonicalJsonBytes(invocationValue as unknown as JsonValue),
    profileBytes: canonicalJsonBytes(profileValue),
    requestBytes: canonicalJsonBytes(requestValue),
    clangWasmBytes: new Uint8Array(CLANG_WASM_BYTES),
    rawWasmConformance: {
      wasmSha256: "d".repeat(64),
      wasmByteLength: CLANG_WASM_BYTES.byteLength,
    },
  });
  return { invocation, sourceSnapshots };
}

function takeEnvironment(): ReturnType<typeof takeCppCuteBrowserWorkerTransfer> {
  const environment = createEnvironment();
  return takeCppCuteBrowserWorkerTransfer(
    prepareCppCuteBrowserWorkerTransfer(environment.invocation as never),
  );
}

function mutableMessage(
  message: CppCuteBrowserWorkerTransferMessage,
): MutableTransferMessage {
  return {
    kind: message.kind,
    version: { major: message.version.major, minor: message.version.minor },
    protocol: message.protocol,
    invocationId: message.invocationId,
    invocationNonceSha256: message.invocationNonceSha256,
    invocationBytes: new Uint8Array(message.invocationBytes),
    profileRegionBytes: new Uint8Array(message.profileRegionBytes),
    requestRegionBytes: new Uint8Array(message.requestRegionBytes),
    assetManifestBytes: new Uint8Array(message.assetManifestBytes),
    assets: message.assets.map((asset) => ({
      assetId: asset.assetId,
      bytes: new Uint8Array(asset.bytes),
    })),
    sourceSnapshots: message.sourceSnapshots.map((source) => ({
      virtualPath: source.virtualPath,
      bytes: new Uint8Array(source.bytes),
    })),
  };
}

function messageByteViews(message: CppCuteBrowserWorkerTransferMessage): Uint8Array[] {
  return [
    message.invocationBytes,
    message.profileRegionBytes,
    message.requestRegionBytes,
    message.assetManifestBytes,
    ...message.assets.map((asset) => asset.bytes),
    ...message.sourceSnapshots.map((source) => source.bytes),
  ];
}

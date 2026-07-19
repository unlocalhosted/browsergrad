import {
  encodeWireU64,
  hashCanonicalJson,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  acquireCppCuteBrowserAssetSet,
  decodeAcquiredCppCuteBrowserRuntimeAbiAsset,
  installCppCuteBrowserVfs,
  unwrapVerifiedCppCuteBrowserVfsInstallation,
  type CppCuteBrowserHostFetch,
  type VerifiedCppCuteBrowserRuntimeAbiAsset,
  type VerifiedCppCuteBrowserVfsInstallation,
} from "../../src/cpp_cute_browser_asset_installation.js";
import {
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
  CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
  cppCuteBrowserSourceAbi,
  deriveCppCuteBrowserAssetManifestId,
  deriveCppCuteBrowserAssetSetSha256,
  prepareCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetManifestBodyV1,
  type CppCuteBrowserAssetManifestV1,
  type CppCuteBrowserAssetV1,
  type PreparedCppCuteBrowserAssetManifest,
} from "../../src/cpp_cute_browser_assets.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
  cppCuteDiagnosticNormalizationResourceBytes,
} from "../../src/cpp_cute_diagnostic_normalization.js";
import {
  cppCuteSemanticAdapterManifestResourceBytes,
} from "../../src/cpp_cute_semantic_adapter_manifest.js";
import {
  CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
  type CppCuteFrontendEntryRequestV1,
  type CppCuteFrontendRequestBodyV1,
  type CppCuteFrontendRequestLimitsV1,
  type CppCuteFrontendRequestSourceFileV1,
  type CppCuteFrontendRequestV1,
  type PreparedCppCuteFrontendRequest,
} from "../../src/cpp_cute_frontend_request.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../../src/cpp_cute_frontend_types.js";
import {
  encodeCppCuteBrowserVfsPack,
  inspectCppCuteBrowserVfsPack,
} from "../../src/cpp_cute_browser_vfs_pack.js";
import {
  CPP_CUTE_BROWSER_VFS_STATUS,
  CppCuteBrowserVfsSessionError,
  bindCppCuteBrowserVfsMount,
  cancelCppCuteBrowserVfsSession,
  closeCppCuteBrowserVfsSession,
  closedCppCuteBrowserVfsSessionReceipt,
  cppCuteBrowserVfsClose,
  cppCuteBrowserVfsDirectoryCount,
  cppCuteBrowserVfsDirectoryEntry,
  cppCuteBrowserVfsOpen,
  cppCuteBrowserVfsRead,
  cppCuteBrowserVfsStatus,
  createCppCuteBrowserVfsHostImports,
  createCppCuteBrowserVfsMountHostImports,
  discardCppCuteBrowserVfsMount,
  nextCppCuteBrowserVfsSessionCallCount,
  observeCppCuteBrowserVfsMount,
  observeCppCuteBrowserVfsSession,
  prepareCppCuteBrowserVfsMount,
  prepareCppCuteBrowserVfsSession,
  unwrapClosedCppCuteBrowserVfsSession,
  type PreparedCppCuteBrowserVfsMount,
  type PreparedCppCuteBrowserVfsSession,
} from "../../src/cpp_cute_browser_vfs_session.js";
import { createCppCuteBrowserProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const ORIGIN = "https://vfs.example.test";
const BUILD_SUBJECT_ID = `bg.cpp.browser-build-subject.sha256.${"9".repeat(64)}`;
const BUILD_PROVENANCE_POLICY = {
  predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
  trustStoreSha256: "d".repeat(64),
  builderIds: ["https://builders.browsergrad.dev/cpp-cute-browser-test"],
} as const;
const MAIN_PATH = "/workspace/src/main.cu";
const HEADER_PATH = "/workspace/src/project.hpp";
const MAIN_BYTES = new TextEncoder().encode('#include "project.hpp"\nauto layout = make_layout(Int<2>{});\n');
const HEADER_BYTES = new TextEncoder().encode("constexpr int project_value = 2;\n");
const ENCODER = new TextEncoder();
const DECODER = new TextDecoder();
const MEMORY_INITIAL_PAGES = 4_096;
const MEMORY_MAXIMUM_PAGES = 16_384;
const NATIVE_DEFINE_PROPERTY = Object.defineProperty;
const NATIVE_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;

interface IntrinsicRestore {
  readonly target: object;
  readonly key: PropertyKey;
  readonly descriptor: PropertyDescriptor | undefined;
}

function poisonValue(
  restores: IntrinsicRestore[],
  target: object,
  key: PropertyKey,
  label: string,
): void {
  restores.push({
    target,
    key,
    descriptor: NATIVE_GET_OWN_PROPERTY_DESCRIPTOR(target, key),
  });
  NATIVE_DEFINE_PROPERTY(target, key, {
    configurable: true,
    writable: true,
    value: () => {
      throw new Error(`poisoned ${label}`);
    },
  });
}

function poisonGetter(
  restores: IntrinsicRestore[],
  target: object,
  key: PropertyKey,
  label: string,
): void {
  const descriptor = NATIVE_GET_OWN_PROPERTY_DESCRIPTOR(target, key);
  restores.push({ target, key, descriptor });
  NATIVE_DEFINE_PROPERTY(target, key, {
    configurable: true,
    enumerable: descriptor?.enumerable ?? false,
    get: () => {
      throw new Error(`poisoned ${label}`);
    },
  });
}

function restoreIntrinsics(restores: readonly IntrinsicRestore[]): void {
  for (let index = restores.length - 1; index >= 0; index -= 1) {
    const restore = restores[index]!;
    if (restore.descriptor === undefined) {
      delete (restore.target as Record<PropertyKey, unknown>)[restore.key];
    } else {
      NATIVE_DEFINE_PROPERTY(restore.target, restore.key, restore.descriptor);
    }
  }
}

interface AuthorityFixture {
  readonly installation: VerifiedCppCuteBrowserVfsInstallation;
  readonly runtimeAbiAsset: VerifiedCppCuteBrowserRuntimeAbiAsset;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly installedBytes: ReadonlyMap<string, Uint8Array>;
}

interface SessionFixture extends AuthorityFixture {
  readonly memory: WebAssembly.Memory;
  readonly session: PreparedCppCuteBrowserVfsSession;
}

let authorityFixturePromise: Promise<AuthorityFixture> | undefined;

function authorityFixture(): Promise<AuthorityFixture> {
  authorityFixturePromise ??= createAuthorityFixture();
  return authorityFixturePromise;
}

async function sessionFixture(): Promise<SessionFixture> {
  const authority = await authorityFixture();
  const memory = new WebAssembly.Memory({
    initial: MEMORY_INITIAL_PAGES,
    maximum: MEMORY_MAXIMUM_PAGES,
  });
  const session = prepareCppCuteBrowserVfsSession({
    installation: authority.installation,
    request: authority.request,
    runtimeAbiAsset: authority.runtimeAbiAsset,
    memory,
  });
  return { ...authority, memory, session };
}

describe("C++/CuTe Worker-owned aggregate lazy VFS session", () => {
  it("keeps the verified mount memory-independent until one exact bind mints the session imports", async () => {
    const authority = await authorityFixture();
    const mount = prepareCppCuteBrowserVfsMount({
      installation: authority.installation,
      request: authority.request,
      runtimeAbiAsset: authority.runtimeAbiAsset,
    });
    expect(observeCppCuteBrowserVfsMount(mount)).toMatchObject({
      mountOrdinal: mount.mountOrdinal,
      installationId: authority.installation.installationId,
      requestId: authority.request.requestId,
      profileHash: authority.request.profileHash,
      state: "prepared",
      workerExecutionObserved: false,
      loweringAuthorityReady: false,
    });
    const imports = createCppCuteBrowserVfsMountHostImports(mount);
    expect(createCppCuteBrowserVfsMountHostImports(mount)).toBe(imports);

    const memory = new WebAssembly.Memory({
      initial: MEMORY_INITIAL_PAGES,
      maximum: MEMORY_MAXIMUM_PAGES,
    });
    const sentinel = new Uint8Array(96).fill(0xa5);
    memoryBytes(memory, 512, sentinel.byteLength).set(sentinel);
    expect([
      imports.bg_vfs_status(-1, -1, -1),
      imports.bg_vfs_open(-1, -1, -1),
      imports.bg_vfs_read(-1, -1, -1, -1, -1),
      imports.bg_vfs_close(-1),
      imports.bg_vfs_directory_count(-1, -1, -1),
      imports.bg_vfs_directory_entry(-1, -1, -1, -1, -1, -1),
    ]).toEqual(Array.from({ length: 6 }, () => CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed));
    expect(memoryBytes(memory, 512, sentinel.byteLength)).toEqual(sentinel);

    const session = bindCppCuteBrowserVfsMount({ mount, memory });
    expect(observeCppCuteBrowserVfsMount(mount).state).toBe("bound");
    expect(createCppCuteBrowserVfsHostImports(session)).toBe(imports);
    expect(createCppCuteBrowserVfsMountHostImports(mount)).toBe(imports);
    expect(observeCppCuteBrowserVfsSession(session).counters).toMatchObject({
      totalSessionCalls: "0",
      statusCalls: "0",
      openCalls: "0",
      readCalls: "0",
      closeCalls: "0",
      directoryCountCalls: "0",
      directoryEntryCalls: "0",
    });
    writePath(memory, 64, MAIN_PATH);
    expect(imports.bg_vfs_status(64, byteLength(MAIN_PATH), 512)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expectSessionError(
      () => bindCppCuteBrowserVfsMount({ mount, memory }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-STATE",
      "$.mount",
    );
    closeCppCuteBrowserVfsSession(session, "completed");
    expect(imports.bg_vfs_status(64, byteLength(MAIN_PATH), 512)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed,
    );
  });

  it("rejects cloned mount authority and invalid memory without consuming the real mount", async () => {
    const authority = await authorityFixture();
    const mount = prepareCppCuteBrowserVfsMount({
      installation: authority.installation,
      request: authority.request,
      runtimeAbiAsset: authority.runtimeAbiAsset,
    });
    const cloned = structuredClone(mount) as PreparedCppCuteBrowserVfsMount;
    expectSessionError(
      () => createCppCuteBrowserVfsMountHostImports(cloned),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED",
      "$.mount",
    );
    expectSessionError(
      () => bindCppCuteBrowserVfsMount({
        mount: cloned,
        memory: new WebAssembly.Memory({ initial: 1, maximum: 1 }),
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED",
      "$.mount",
    );
    expectSessionError(
      () => bindCppCuteBrowserVfsMount({
        mount,
        memory: new WebAssembly.Memory({ initial: 1, maximum: 1 }),
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-MISMATCH",
      "$.memory",
    );
    class MemorySubclass extends WebAssembly.Memory {}
    expectSessionError(
      () => bindCppCuteBrowserVfsMount({
        mount,
        memory: new MemorySubclass({ initial: 1, maximum: 1 }),
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-INVALID",
      "$.memory",
    );
    const exactMemory = new WebAssembly.Memory({
      initial: MEMORY_INITIAL_PAGES,
      maximum: MEMORY_MAXIMUM_PAGES,
    });
    const accessor = {
      mount,
      get memory(): WebAssembly.Memory {
        return exactMemory;
      },
    };
    expectSessionError(
      () => bindCppCuteBrowserVfsMount(accessor),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-INVALID",
      "$.input.memory",
    );
    expect(observeCppCuteBrowserVfsMount(mount).state).toBe("prepared");
    const session = bindCppCuteBrowserVfsMount({ mount, memory: exactMemory });
    cancelCppCuteBrowserVfsSession(session);
  });

  it("destructively discards an unbound mount and refuses later bind or repeated cleanup", async () => {
    const authority = await authorityFixture();
    const mount = prepareCppCuteBrowserVfsMount({
      installation: authority.installation,
      request: authority.request,
      runtimeAbiAsset: authority.runtimeAbiAsset,
    });
    const imports = createCppCuteBrowserVfsMountHostImports(mount);
    discardCppCuteBrowserVfsMount(mount);
    expect(observeCppCuteBrowserVfsMount(mount).state).toBe("discarded");
    expect([
      imports.bg_vfs_status(0, 0, 0),
      imports.bg_vfs_open(0, 0, 0),
      imports.bg_vfs_read(0, 0, 0, 0, 0),
      imports.bg_vfs_close(0),
      imports.bg_vfs_directory_count(0, 0, 0),
      imports.bg_vfs_directory_entry(0, 0, 0, 0, 0, 0),
    ]).toEqual(Array.from({ length: 6 }, () => CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed));
    expectSessionError(
      () => bindCppCuteBrowserVfsMount({
        mount,
        memory: new WebAssembly.Memory({ initial: 1, maximum: 1 }),
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-STATE",
      "$.mount",
    );
    expectSessionError(
      () => discardCppCuteBrowserVfsMount(mount),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-STATE",
      "$.mount",
    );
  });

  it("binds same-realm authority through captured admission and memory intrinsics", async () => {
    const authority = await authorityFixture();
    const admissionRestores: IntrinsicRestore[] = [];
    let mount: PreparedCppCuteBrowserVfsMount | undefined;
    try {
      poisonValue(admissionRestores, WeakMap.prototype, "set", "WeakMap.set");
      poisonValue(admissionRestores, Object, "getPrototypeOf", "Object.getPrototypeOf");
      poisonValue(
        admissionRestores,
        Object,
        "getOwnPropertyDescriptors",
        "Object.getOwnPropertyDescriptors",
      );
      poisonValue(admissionRestores, Reflect, "ownKeys", "Reflect.ownKeys");
      poisonValue(admissionRestores, Reflect, "apply", "Reflect.apply");
      mount = prepareCppCuteBrowserVfsMount({
        installation: authority.installation,
        request: authority.request,
        runtimeAbiAsset: authority.runtimeAbiAsset,
      });
    } finally {
      restoreIntrinsics(admissionRestores);
    }
    expect(mount).toBeDefined();
    const memory = new WebAssembly.Memory({
      initial: MEMORY_INITIAL_PAGES,
      maximum: MEMORY_MAXIMUM_PAGES,
    });
    const memoryBufferDescriptor = NATIVE_GET_OWN_PROPERTY_DESCRIPTOR(
      WebAssembly.Memory.prototype,
      "buffer",
    );
    const memoryBufferGetter = memoryBufferDescriptor?.get;
    expect(memoryBufferGetter).toBeDefined();
    const restores: IntrinsicRestore[] = [];
    let session: PreparedCppCuteBrowserVfsSession | undefined;
    try {
      poisonValue(restores, WeakMap.prototype, "get", "WeakMap.get");
      poisonValue(restores, WeakMap.prototype, "set", "WeakMap.set");
      poisonValue(restores, Object, "getPrototypeOf", "Object.getPrototypeOf");
      poisonValue(
        restores,
        Object,
        "getOwnPropertyDescriptors",
        "Object.getOwnPropertyDescriptors",
      );
      poisonValue(restores, Object, "freeze", "Object.freeze");
      poisonValue(restores, Reflect, "ownKeys", "Reflect.ownKeys");
      poisonValue(restores, Reflect, "apply", "Reflect.apply");
      poisonValue(restores, ArrayBuffer, Symbol.hasInstance, "ArrayBuffer instanceof");
      poisonGetter(
        restores,
        WebAssembly.Memory.prototype,
        "buffer",
        "WebAssembly.Memory.buffer",
      );
      poisonValue(restores, memoryBufferGetter!, "call", "memory getter.call");
      session = bindCppCuteBrowserVfsMount({ mount: mount!, memory });
    } finally {
      restoreIntrinsics(restores);
    }
    expect(session).toBeDefined();
    expect(observeCppCuteBrowserVfsMount(mount!).state).toBe("bound");
    expect(createCppCuteBrowserVfsHostImports(session!)).toBe(
      createCppCuteBrowserVfsHostImports(session!),
    );
    cancelCppCuteBrowserVfsSession(session!);
  });

  it("settles mount and session cleanup while destructive prototype methods are hostile", async () => {
    const authority = await authorityFixture();
    const mount = prepareCppCuteBrowserVfsMount({
      installation: authority.installation,
      request: authority.request,
      runtimeAbiAsset: authority.runtimeAbiAsset,
    });
    const fixture = await sessionFixture();
    const restores: IntrinsicRestore[] = [];
    let closed: ReturnType<typeof cancelCppCuteBrowserVfsSession> | undefined;
    try {
      poisonValue(restores, Uint8Array.prototype, "fill", "Uint8Array.fill");
      poisonValue(restores, Map.prototype, "clear", "Map.clear");
      poisonValue(restores, Set.prototype, "clear", "Set.clear");
      discardCppCuteBrowserVfsMount(mount);
      closed = cancelCppCuteBrowserVfsSession(fixture.session);
    } finally {
      restoreIntrinsics(restores);
    }
    expect(observeCppCuteBrowserVfsMount(mount).state).toBe("discarded");
    expect(closed).toBeDefined();
    expect(closedCppCuteBrowserVfsSessionReceipt(fixture.session)).toBe(closed);
    expect(unwrapClosedCppCuteBrowserVfsSession(closed!).reason).toBe("cancelled");
    expect(observeCppCuteBrowserVfsSession(fixture.session)).toMatchObject({
      state: "disposed",
      openedFiles: [],
      counters: {
        currentLiveHandles: "0",
        currentLiveLogicalReservationByteLength: "0",
      },
    });
  });

  it("serves copied request sources and range-copied installed packs through one ABI memory", async () => {
    const fixture = await sessionFixture();
    const installedPath = [...fixture.installedBytes.entries()]
      .filter(([, bytes]) => bytes.byteLength > 1)
      .map(([path]) => path)
      .sort()[0]!;
    writePath(fixture.memory, 64, MAIN_PATH);
    expect(cppCuteBrowserVfsStatus(fixture.session, 64, byteLength(MAIN_PATH), 512)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expect(readU32(fixture.memory, 512)).toBe(1);
    expect(readU64(fixture.memory, 520)).toBe(BigInt(MAIN_BYTES.byteLength));

    expect(cppCuteBrowserVfsOpen(fixture.session, 64, byteLength(MAIN_PATH), 560)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    const sourceHandle = readU32(fixture.memory, 560);
    expect(readU64(fixture.memory, 568)).toBe(BigInt(MAIN_BYTES.byteLength));
    expect(cppCuteBrowserVfsRead(fixture.session, sourceHandle, 0, 0, 640, MAIN_BYTES.byteLength)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expect(memoryBytes(fixture.memory, 640, MAIN_BYTES.byteLength)).toEqual(MAIN_BYTES);
    memoryBytes(fixture.memory, 640, MAIN_BYTES.byteLength).fill(0);
    expect(cppCuteBrowserVfsRead(fixture.session, sourceHandle, 0, 0, 640, MAIN_BYTES.byteLength)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expect(memoryBytes(fixture.memory, 640, MAIN_BYTES.byteLength)).toEqual(MAIN_BYTES);
    expect(cppCuteBrowserVfsClose(fixture.session, sourceHandle)).toBe(CPP_CUTE_BROWSER_VFS_STATUS.ok);

    writePath(fixture.memory, 64, installedPath);
    expect(cppCuteBrowserVfsOpen(fixture.session, 64, byteLength(installedPath), 560)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    const installedHandle = readU32(fixture.memory, 560);
    const expectedInstalled = fixture.installedBytes.get(installedPath)!;
    expect(cppCuteBrowserVfsRead(
      fixture.session,
      installedHandle,
      1,
      0,
      640,
      expectedInstalled.byteLength - 1,
    )).toBe(CPP_CUTE_BROWSER_VFS_STATUS.ok);
    expect(memoryBytes(fixture.memory, 640, expectedInstalled.byteLength - 1)).toEqual(
      expectedInstalled.subarray(1),
    );
    expect(cppCuteBrowserVfsClose(fixture.session, installedHandle)).toBe(CPP_CUTE_BROWSER_VFS_STATUS.ok);

    const observation = observeCppCuteBrowserVfsSession(fixture.session);
    expect(observation.openedFiles.map((file) => file.virtualPath)).toEqual(
      [installedPath, MAIN_PATH].sort((left, right) => compareUtf8(left, right)),
    );
    expect(observation.counters.currentLiveHandles).toBe("0");
    expect(observation.counters.logicalOpenedSourceByteLength).toBe(String(MAIN_BYTES.byteLength));
    expect(observation.counters.logicalOpenedInstalledVfsByteLength).toBe(
      String(expectedInstalled.byteLength),
    );
    expect(observation.counters.logicalOpenedTotalByteLength).toBe(
      String(MAIN_BYTES.byteLength + expectedInstalled.byteLength),
    );
    const expectedIndex = indexAccounting([
      MAIN_PATH,
      HEADER_PATH,
      ...fixture.installedBytes.keys(),
    ]);
    expect(fixture.session.indexedNodes).toBe(expectedIndex.indexedNodes);
    expect(fixture.session.indexLogicalByteLength).toBe(expectedIndex.indexLogicalByteLength);
    expect(observation.counters.indexedNodes).toBe(String(expectedIndex.indexedNodes));
    expect(observation.counters.indexLogicalByteLength).toBe(
      String(expectedIndex.indexLogicalByteLength),
    );
    expect(observation).not.toHaveProperty("workerExecutionObserved");
    closeCppCuteBrowserVfsSession(fixture.session, "completed");
  });

  it("observes only unique files with successful content reads", async () => {
    const fixture = await sessionFixture();
    writePath(fixture.memory, 64, MAIN_PATH);
    expect(cppCuteBrowserVfsOpen(fixture.session, 64, byteLength(MAIN_PATH), 560)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    const handle = readU32(fixture.memory, 560);

    let observation = observeCppCuteBrowserVfsSession(fixture.session);
    expect(observation.openedFiles).toEqual([]);
    expect(observation.counters.logicalOpenedSourceByteLength).toBe("0");
    expect(observation.counters.currentLiveSourceLogicalReservationByteLength).toBe(
      String(MAIN_BYTES.byteLength),
    );

    expect(cppCuteBrowserVfsRead(fixture.session, handle, 0, 0, 640, 0)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expect(cppCuteBrowserVfsRead(
      fixture.session,
      handle,
      MAIN_BYTES.byteLength,
      0,
      640,
      1,
    )).toBe(CPP_CUTE_BROWSER_VFS_STATUS.outOfRange);
    expect(observeCppCuteBrowserVfsSession(fixture.session).openedFiles).toEqual([]);

    expect(cppCuteBrowserVfsRead(fixture.session, handle, 1, 0, 640, 1)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expect(cppCuteBrowserVfsRead(fixture.session, handle, 2, 0, 640, 2)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    observation = observeCppCuteBrowserVfsSession(fixture.session);
    expect(observation.openedFiles.map((file) => file.virtualPath)).toEqual([MAIN_PATH]);
    expect(observation.counters.logicalOpenedSourceByteLength).toBe(String(MAIN_BYTES.byteLength));
    expect(observation.counters.logicalOpenedTotalByteLength).toBe(String(MAIN_BYTES.byteLength));
    expect(observation.counters.currentLiveSourceLogicalReservationByteLength).toBe(
      String(MAIN_BYTES.byteLength),
    );

    const installedZeroBytePath = [...fixture.installedBytes.entries()]
      .find(([, bytes]) => bytes.byteLength === 0)?.[0];
    expect(installedZeroBytePath).toBeDefined();
    writePath(fixture.memory, 64, installedZeroBytePath!);
    expect(cppCuteBrowserVfsOpen(
      fixture.session,
      64,
      byteLength(installedZeroBytePath!),
      560,
    )).toBe(CPP_CUTE_BROWSER_VFS_STATUS.ok);
    const zeroByteHandle = readU32(fixture.memory, 560);
    expect(cppCuteBrowserVfsRead(fixture.session, zeroByteHandle, 0, 0, 640, 0)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expect(cppCuteBrowserVfsClose(fixture.session, zeroByteHandle)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expect(observeCppCuteBrowserVfsSession(fixture.session).openedFiles.map(
      (file) => file.virtualPath,
    )).toEqual([installedZeroBytePath!, MAIN_PATH].sort(compareUtf8));
    expect(observeCppCuteBrowserVfsSession(
      fixture.session,
    ).counters.logicalOpenedInstalledVfsByteLength).toBe("0");

    expect(cppCuteBrowserVfsClose(fixture.session, handle)).toBe(CPP_CUTE_BROWSER_VFS_STATUS.ok);
    closeCppCuteBrowserVfsSession(fixture.session, "completed");
  });

  it("snapshots overlapping path input and emits deterministic byte-ordered directory entries", async () => {
    const fixture = await sessionFixture();
    writePath(fixture.memory, 1_024, MAIN_PATH);
    expect(cppCuteBrowserVfsStatus(fixture.session, 1_024, byteLength(MAIN_PATH), 1_024)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    expect(readU32(fixture.memory, 1_024)).toBe(1);

    writePath(fixture.memory, 1_200, "/toolchain");
    expect(cppCuteBrowserVfsDirectoryCount(fixture.session, 1_200, byteLength("/toolchain"), 1_280)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    const count = readU32(fixture.memory, 1_280);
    const entries: string[] = [];
    for (let index = 0; index < count; index += 1) {
      memoryBytes(fixture.memory, 1_344, 64).fill(0xcc);
      memoryBytes(fixture.memory, 1_440, 32).fill(0xdd);
      expect(cppCuteBrowserVfsDirectoryEntry(
        fixture.session,
        1_200,
        byteLength("/toolchain"),
        index,
        1_344,
        64,
        1_440,
      )).toBe(CPP_CUTE_BROWSER_VFS_STATUS.ok);
      const nameLength = readU32(fixture.memory, 1_444);
      entries.push(DECODER.decode(memoryBytes(fixture.memory, 1_344, nameLength)));
      expect(readU32(fixture.memory, 1_440)).toBe(2);
    }
    expect(entries).toEqual([...entries].sort(compareUtf8));
    expect(entries).toEqual(["clang", "cuda", "cutlass", "cxx", "sysroot"]);

    memoryBytes(fixture.memory, 1_344, 64).fill(0xaa);
    memoryBytes(fixture.memory, 1_440, 32).fill(0xbb);
    expect(cppCuteBrowserVfsDirectoryEntry(
      fixture.session,
      1_200,
      byteLength("/toolchain"),
      0,
      1_344,
      1,
      1_440,
    )).toBe(CPP_CUTE_BROWSER_VFS_STATUS.bufferTooSmall);
    expect(memoryBytes(fixture.memory, 1_344, 64)).toEqual(new Uint8Array(64).fill(0xaa));
    expect(memoryBytes(fixture.memory, 1_440, 4)).toEqual(new Uint8Array(4).fill(0xbb));
    expect(readU32(fixture.memory, 1_444)).toBe(byteLength(entries[0]!));
    expect(memoryBytes(fixture.memory, 1_448, 24)).toEqual(new Uint8Array(24).fill(0xbb));
    cancelCppCuteBrowserVfsSession(fixture.session);
  });

  it("rejects hostile wasm32 ranges, malformed paths, misalignment, and overlapping outputs atomically", async () => {
    const fixture = await sessionFixture();
    const initialBuffer = fixture.memory.buffer;
    fixture.memory.grow(1);
    expect(fixture.memory.buffer).not.toBe(initialBuffer);
    const sentinelPointer = 2_048;
    const sentinel = new Uint8Array(96).fill(0xa5);
    memoryBytes(fixture.memory, sentinelPointer, sentinel.byteLength).set(sentinel);
    writePath(fixture.memory, 128, MAIN_PATH);

    expect(cppCuteBrowserVfsStatus(fixture.session, -1, 2, sentinelPointer)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.outOfRange,
    );
    expect(memoryBytes(fixture.memory, sentinelPointer, sentinel.byteLength)).toEqual(sentinel);
    expect(cppCuteBrowserVfsStatus(fixture.session, 128, byteLength(MAIN_PATH), sentinelPointer + 1)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.outOfRange,
    );
    expect(memoryBytes(fixture.memory, sentinelPointer, sentinel.byteLength)).toEqual(sentinel);

    memoryBytes(fixture.memory, 128, 2).set(Uint8Array.of(0xc0, 0xaf));
    expect(cppCuteBrowserVfsStatus(fixture.session, 128, 2, sentinelPointer)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.invalidPath,
    );
    expect(memoryBytes(fixture.memory, sentinelPointer, sentinel.byteLength)).toEqual(sentinel);

    writePath(fixture.memory, 128, "/toolchain");
    expect(cppCuteBrowserVfsDirectoryEntry(
      fixture.session,
      128,
      byteLength("/toolchain"),
      0,
      sentinelPointer,
      32,
      sentinelPointer,
    )).toBe(CPP_CUTE_BROWSER_VFS_STATUS.outOfRange);
    expect(memoryBytes(fixture.memory, sentinelPointer, sentinel.byteLength)).toEqual(sentinel);

    expect(cppCuteBrowserVfsRead(fixture.session, 0xffff_ffff, 0, 0, sentinelPointer, 32)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.invalidHandle,
    );
    expect(memoryBytes(fixture.memory, sentinelPointer, sentinel.byteLength)).toEqual(sentinel);
    cancelCppCuteBrowserVfsSession(fixture.session);
  });

  it("rejects structural capabilities, accessor records, and non-exact memory authorities", async () => {
    const fixture = await authorityFixture();
    const memory = new WebAssembly.Memory({ initial: MEMORY_INITIAL_PAGES, maximum: MEMORY_MAXIMUM_PAGES });
    const accessor = {
      installation: fixture.installation,
      request: fixture.request,
      runtimeAbiAsset: fixture.runtimeAbiAsset,
      get memory(): WebAssembly.Memory {
        return memory;
      },
    };
    expectSessionError(
      () => prepareCppCuteBrowserVfsSession(accessor),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-INVALID",
      "$.input.memory",
    );
    class MemorySubclass extends WebAssembly.Memory {}
    expectSessionError(
      () => prepareCppCuteBrowserVfsSession({
        installation: fixture.installation,
        request: fixture.request,
        runtimeAbiAsset: fixture.runtimeAbiAsset,
        memory: new MemorySubclass({ initial: MEMORY_INITIAL_PAGES, maximum: MEMORY_MAXIMUM_PAGES }),
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-INVALID",
      "$.memory",
    );
    expectSessionError(
      () => createCppCuteBrowserVfsHostImports({
        sessionOrdinal: 1,
        installationId: fixture.installation.installationId,
        requestId: fixture.request.requestId,
      } as PreparedCppCuteBrowserVfsSession),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-UNVERIFIED",
      "$.session",
    );
  });

  it("prevents stale-handle replay and disposes all live state on cancellation", async () => {
    const fixture = await sessionFixture();
    writePath(fixture.memory, 128, HEADER_PATH);
    expect(cppCuteBrowserVfsOpen(fixture.session, 128, byteLength(HEADER_PATH), 512)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    const firstHandle = readU32(fixture.memory, 512);
    expect(cppCuteBrowserVfsClose(fixture.session, firstHandle)).toBe(CPP_CUTE_BROWSER_VFS_STATUS.ok);
    expect(cppCuteBrowserVfsClose(fixture.session, firstHandle)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.invalidHandle,
    );
    expect(cppCuteBrowserVfsOpen(fixture.session, 128, byteLength(HEADER_PATH), 512)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.ok,
    );
    const secondHandle = readU32(fixture.memory, 512);
    expect(secondHandle).toBeGreaterThan(firstHandle);
    expect(cppCuteBrowserVfsRead(fixture.session, firstHandle, 0, 0, 640, 1)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.invalidHandle,
    );

    const closed = cancelCppCuteBrowserVfsSession(fixture.session);
    expect(closedCppCuteBrowserVfsSessionReceipt(fixture.session)).toBe(closed);
    const terminal = unwrapClosedCppCuteBrowserVfsSession(closed);
    expect(terminal.reason).toBe("cancelled");
    expect(terminal.observation.state).toBe("disposed");
    expect(terminal.observation.counters.currentLiveHandles).toBe("0");
    expect(terminal.observation.counters.currentLiveLogicalReservationByteLength).toBe("0");
    expect(terminal.observation.counters.peakLiveHandles).toBe("1");
    expect(terminal.observation.openedFiles).toEqual([]);
    expect(cppCuteBrowserVfsRead(fixture.session, secondHandle, 0, 0, 640, 1)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.sessionClosed,
    );
    expectSessionError(
      () => cancelCppCuteBrowserVfsSession(fixture.session),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-STATE",
      "$.session",
    );
  });

  it("enforces the exact call, live-handle, and aggregate-opened-byte ceilings", async () => {
    const fixture = await sessionFixture();
    const installedZeroBytePath = [...fixture.installedBytes.entries()]
      .find(([, bytes]) => bytes.byteLength === 0)?.[0];
    expect(installedZeroBytePath).toBeDefined();
    writePath(fixture.memory, 128, installedZeroBytePath!);
    for (let count = 0; count < fixture.session.maxLiveFileHandles; count += 1) {
      expect(cppCuteBrowserVfsOpen(fixture.session, 128, byteLength(installedZeroBytePath!), 512)).toBe(
        CPP_CUTE_BROWSER_VFS_STATUS.ok,
      );
    }
    expect(cppCuteBrowserVfsOpen(fixture.session, 128, byteLength(installedZeroBytePath!), 512)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.resourceLimit,
    );
    expect(observeCppCuteBrowserVfsSession(fixture.session).counters.peakLiveHandles).toBe(
      String(fixture.session.maxLiveFileHandles),
    );
    cancelCppCuteBrowserVfsSession(fixture.session);

    const limitedAuthority = await createAuthorityFixture({ narrowAggregate: true });
    const limitedMemory = new WebAssembly.Memory({
      initial: MEMORY_INITIAL_PAGES,
      maximum: MEMORY_MAXIMUM_PAGES,
    });
    const limitedSession = prepareCppCuteBrowserVfsSession({
      installation: limitedAuthority.installation,
      request: limitedAuthority.request,
      runtimeAbiAsset: limitedAuthority.runtimeAbiAsset,
      memory: limitedMemory,
    });
    writePath(limitedMemory, 128, HEADER_PATH);
    const allowedHeaderCopies = Math.floor(
      limitedSession.maxAggregateLiveOpenByteLength / HEADER_BYTES.byteLength,
    );
    for (let count = 0; count < allowedHeaderCopies; count += 1) {
      expect(cppCuteBrowserVfsOpen(limitedSession, 128, byteLength(HEADER_PATH), 512)).toBe(
        CPP_CUTE_BROWSER_VFS_STATUS.ok,
      );
    }
    expect(cppCuteBrowserVfsOpen(limitedSession, 128, byteLength(HEADER_PATH), 512)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.resourceLimit,
    );
    const limitedObservation = observeCppCuteBrowserVfsSession(limitedSession);
    expect(BigInt(
      limitedObservation.counters.currentLiveLogicalReservationByteLength,
    )).toBeLessThanOrEqual(
      BigInt(limitedSession.maxAggregateLiveOpenByteLength),
    );
    cancelCppCuteBrowserVfsSession(limitedSession);

    const callFixture = await sessionFixture();
    expect(cppCuteBrowserVfsClose(callFixture.session, 0)).toBe(
      CPP_CUTE_BROWSER_VFS_STATUS.invalidHandle,
    );
    expect(observeCppCuteBrowserVfsSession(callFixture.session).counters.totalSessionCalls).toBe("1");
    const penultimateCallCount = BigInt(callFixture.session.maxSessionCalls) - 1n;
    const finalCallCount = nextCppCuteBrowserVfsSessionCallCount(
      penultimateCallCount,
      callFixture.session.maxSessionCalls,
    );
    expect(finalCallCount).toBe(BigInt(callFixture.session.maxSessionCalls));
    expect(nextCppCuteBrowserVfsSessionCallCount(
      finalCallCount!,
      callFixture.session.maxSessionCalls,
    )).toBeUndefined();
    cancelCppCuteBrowserVfsSession(callFixture.session);
  });

  it("rejects overlong mounted paths and explicit expanded-index ceilings before authority", async () => {
    const longPathAuthority = await createAuthorityFixture({ longMountedPath: true });
    expectSessionError(
      () => prepareCppCuteBrowserVfsSession({
        installation: longPathAuthority.installation,
        request: longPathAuthority.request,
        runtimeAbiAsset: longPathAuthority.runtimeAbiAsset,
        memory: new WebAssembly.Memory({
          initial: MEMORY_INITIAL_PAGES,
          maximum: MEMORY_MAXIMUM_PAGES,
        }),
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-RESOURCE-LIMIT",
      "$.installation.files[0].virtualPath",
    );

    for (const [field, value, expectedPath] of [
      ["maxIndexedNodes", 1, "$.files"],
      ["maxIndexLogicalByteLength", 1, "$.index.root"],
    ] as const) {
      const authority = await createAuthorityFixture({ [field]: value });
      expectSessionError(
        () => prepareCppCuteBrowserVfsSession({
          installation: authority.installation,
          request: authority.request,
          runtimeAbiAsset: authority.runtimeAbiAsset,
          memory: new WebAssembly.Memory({
            initial: MEMORY_INITIAL_PAGES,
            maximum: MEMORY_MAXIMUM_PAGES,
          }),
        }),
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-SESSION-RESOURCE-LIMIT",
        expectedPath,
      );
    }
  });
});

async function createAuthorityFixture(
  options: {
    readonly narrowAggregate?: boolean;
    readonly longMountedPath?: boolean;
    readonly maxIndexedNodes?: number;
    readonly maxIndexLogicalByteLength?: number;
  } = {},
): Promise<AuthorityFixture> {
  const input = structuredClone(createCppCuteBrowserProfileInput());
  const profileVfs = input.deployment.compilerRuntime.virtualFileSystem as {
    maxAggregateLiveOpenByteLength: number;
    maxIndexedNodes: number;
    maxIndexLogicalByteLength: number;
  };
  if (options.maxIndexedNodes !== undefined) profileVfs.maxIndexedNodes = options.maxIndexedNodes;
  if (options.maxIndexLogicalByteLength !== undefined) {
    profileVfs.maxIndexLogicalByteLength = options.maxIndexLogicalByteLength;
  }
  if (options.longMountedPath === true) {
    const longRoot = input.virtualFileSystem.includeRoots.find((root) =>
      root.includeRootId === "cutlass");
    if (longRoot === undefined) throw new Error("fixture long-path root missing");
    (longRoot as { virtualPath: string }).virtualPath = `/${"r".repeat(1_023)}`;
  }
  if (options.narrowAggregate === true) {
    (input.extractionLimits as { maxSourceBytes: number; maxHeaderBytes: number }).maxSourceBytes =
      MAIN_BYTES.byteLength + HEADER_BYTES.byteLength;
    (input.extractionLimits as { maxSourceBytes: number; maxHeaderBytes: number }).maxHeaderBytes =
      1;
    profileVfs.maxAggregateLiveOpenByteLength = MAIN_BYTES.byteLength + HEADER_BYTES.byteLength + 1;
  }
  const adapterBytes = cppCuteSemanticAdapterManifestResourceBytes();
  const wasmBytes = Uint8Array.of(4, 5, 6, 7);
  const runtimeAbiBytes = cppCuteBrowserRuntimeAbiManifestResourceBytes();
  const diagnosticNormalizationBytes =
    cppCuteDiagnosticNormalizationResourceBytes();
  const adapterSha256 = await sha256Hex(adapterBytes);
  const wasmSha256 = await sha256Hex(wasmBytes);
  (input.toolchain.compiler as { binarySha256: string }).binarySha256 = wasmSha256;
  Object.assign(input.deployment.extractor, {
    binarySha256: wasmSha256,
    semanticAdapterManifestSha256: adapterSha256,
  });

  const packs = new Map<string, {
    readonly bytes: Uint8Array;
    readonly sha256: string;
    readonly contentSetSha256: string;
    readonly fileContentByteLength: WireU64;
    readonly fileBytes: Uint8Array;
    readonly markerBytes: Uint8Array | undefined;
    readonly relativePath: string;
  }>();
  let ordinal = 0;
  for (const root of input.virtualFileSystem.includeRoots) {
    if (root.owner.kind === "source") continue;
    const fileBytes = ordinal === 0
      ? new Uint8Array(0)
      : ENCODER.encode(`header-${root.includeRootId}`);
    const markerBytes = ordinal === 0 ? Uint8Array.of(0x7f) : undefined;
    const relativePath = options.longMountedPath === true && root.includeRootId === "cutlass"
      ? `${"s".repeat(3_070)}/header.h`
      : "header.h";
    const bytes = await encodeCppCuteBrowserVfsPack([
      { virtualPath: relativePath, bytes: fileBytes },
      ...(markerBytes === undefined ? [] : [{ virtualPath: "marker.h", bytes: markerBytes }]),
    ]);
    const inspected = await inspectCppCuteBrowserVfsPack(bytes);
    packs.set(root.includeRootId, {
      bytes,
      sha256: inspected.packSha256,
      contentSetSha256: inspected.contentSetSha256,
      fileContentByteLength: inspected.fileContentByteLength,
      fileBytes,
      markerBytes,
      relativePath,
    });
    (root as { manifestSha256: string }).manifestSha256 = inspected.contentSetSha256;
    if (root.owner.kind === "compiler-resource-directory") {
      (input.toolchain.compiler as { resourceDirectorySha256: string }).resourceDirectorySha256 =
        inspected.contentSetSha256;
    } else {
      const dependency = input.toolchain.dependencies.find((entry) => entry.dependencyId === root.owner.dependencyId);
      if (dependency === undefined) throw new Error("fixture dependency missing");
      (dependency as { headerSetSha256: string }).headerSetSha256 = inspected.contentSetSha256;
    }
    ordinal += 1;
  }

  const provisional = await prepareCppCuteFrontendProfile(input);
  const provisionalProfile = unwrapPreparedCppCuteBrowserFrontendProfile(provisional).profile;
  const sourceAbi = cppCuteBrowserSourceAbi(provisional);
  const sourceAbiSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-source-abi.v1",
    sourceAbi,
  });
  const assets: CppCuteBrowserAssetV1[] = [
    {
      assetId: "adapter",
      kind: "semantic-adapter-manifest",
      url: "/assets/adapter.json",
      urlPolicy: "same-origin-root-relative",
      sha256: adapterSha256,
      byteLength: wire(adapterBytes.byteLength),
      unpackedByteLength: wire(adapterBytes.byteLength),
      mediaType: "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      compression: "identity",
      buildSubjectId: BUILD_SUBJECT_ID,
    },
    {
      assetId: "clang-wasm",
      kind: "clang-extractor-wasm",
      url: "/assets/clang.wasm",
      urlPolicy: "same-origin-root-relative",
      sha256: wasmSha256,
      byteLength: wire(wasmBytes.byteLength),
      unpackedByteLength: wire(wasmBytes.byteLength),
      mediaType: "application/wasm",
      compression: "identity",
      buildSubjectId: BUILD_SUBJECT_ID,
      sourceAbiSha256,
    },
    {
      assetId: "diagnostic-normalization",
      kind: "diagnostic-normalization-manifest",
      url: "/assets/diagnostic-normalization.json",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
      byteLength: wire(diagnosticNormalizationBytes.byteLength),
      unpackedByteLength: wire(diagnosticNormalizationBytes.byteLength),
      mediaType:
        "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json",
      compression: "identity",
      buildSubjectId: BUILD_SUBJECT_ID,
    },
    {
      assetId: "runtime-abi",
      kind: "runtime-abi-manifest",
      url: "/assets/runtime-abi-manifest.json",
      urlPolicy: "same-origin-root-relative",
      sha256: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      byteLength: wire(runtimeAbiBytes.byteLength),
      unpackedByteLength: wire(runtimeAbiBytes.byteLength),
      mediaType: "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      compression: "identity",
      buildSubjectId: BUILD_SUBJECT_ID,
      runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1",
      runtimeAbiManifestId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
    },
  ];
  const installedBytes = new Map<string, Uint8Array>();
  for (const root of provisionalProfile.virtualFileSystem.includeRoots) {
    if (root.owner.kind === "source") continue;
    const pack = packs.get(root.includeRootId)!;
    const common = {
      assetId: root.owner.kind === "compiler-resource-directory"
        ? "compiler-resource"
        : `dependency.${root.owner.dependencyId}`,
      url: `/assets/${root.includeRootId}.bgvfs`,
      urlPolicy: "same-origin-root-relative" as const,
      sha256: pack.sha256,
      byteLength: wire(pack.bytes.byteLength),
      unpackedByteLength: wire(pack.bytes.byteLength),
      fileContentByteLength: pack.fileContentByteLength,
      mediaType: "application/vnd.browsergrad.vfs-pack.v1" as const,
      compression: "identity" as const,
      buildSubjectId: BUILD_SUBJECT_ID,
      includeRootId: root.includeRootId,
      mountedVirtualRoot: root.virtualPath,
      contentSetSha256: pack.contentSetSha256,
    };
    assets.push(root.owner.kind === "compiler-resource-directory"
      ? { ...common, kind: "compiler-resource-pack" }
      : { ...common, kind: "dependency-header-pack", dependencyId: root.owner.dependencyId });
    installedBytes.set(`${root.virtualPath}/${pack.relativePath}`, new Uint8Array(pack.fileBytes));
    if (pack.markerBytes !== undefined) {
      installedBytes.set(`${root.virtualPath}/marker.h`, new Uint8Array(pack.markerBytes));
    }
  }
  assets.sort((left, right) => left.assetId.localeCompare(right.assetId));
  const mountedVirtualRoots = assets.flatMap((asset): string[] =>
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? [asset.mountedVirtualRoot]
      : []).sort();
  const dependencyIds = provisionalProfile.toolchain.dependencies.map((entry) => entry.dependencyId);
  const assetSetSha256 = await deriveCppCuteBrowserAssetSetSha256({
    sourceAbiSha256,
    dependencyIds,
    buildSubjectIds: [BUILD_SUBJECT_ID],
    buildProvenancePolicy: BUILD_PROVENANCE_POLICY,
    mountedVirtualRoots,
    assets,
  });
  (input.deployment as { assetSetSha256: string }).assetSetSha256 = assetSetSha256;
  const profile = await prepareCppCuteFrontendProfile(input);
  const manifest = await prepareManifest(profile, sourceAbi, sourceAbiSha256, assetSetSha256, assets);
  const bytesByUrl = new Map<string, Uint8Array>([
    [`${ORIGIN}/assets/adapter.json`, adapterBytes],
    [`${ORIGIN}/assets/clang.wasm`, wasmBytes],
    [`${ORIGIN}/assets/diagnostic-normalization.json`,
      diagnosticNormalizationBytes],
    [`${ORIGIN}/assets/runtime-abi-manifest.json`, runtimeAbiBytes],
  ]);
  for (const [rootId, pack] of packs) bytesByUrl.set(`${ORIGIN}/assets/${rootId}.bgvfs`, pack.bytes);
  const assetSet = await acquireCppCuteBrowserAssetSet(manifest, ORIGIN, hostFetch(bytesByUrl));
  const installation = await installCppCuteBrowserVfs(assetSet);
  expect(unwrapVerifiedCppCuteBrowserVfsInstallation(installation).files).toHaveLength(installedBytes.size);
  return {
    installation,
    runtimeAbiAsset: await decodeAcquiredCppCuteBrowserRuntimeAbiAsset(assetSet),
    request: await createRequest(profile),
    installedBytes,
  };
}

async function prepareManifest(
  profile: PreparedCppCuteFrontendProfile,
  sourceAbi: ReturnType<typeof cppCuteBrowserSourceAbi>,
  sourceAbiSha256: string,
  assetSetSha256: string,
  assets: readonly CppCuteBrowserAssetV1[],
): Promise<PreparedCppCuteBrowserAssetManifest> {
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile).profile;
  const mountedVirtualRoots = assets.flatMap((asset): string[] =>
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? [asset.mountedVirtualRoot]
      : []).sort();
  const compressed = assets.reduce((total, asset) => total + BigInt(asset.byteLength), 0n);
  const fileContent = assets.reduce((total, asset) => total + (
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? BigInt(asset.fileContentByteLength)
      : 0n
  ), 0n);
  const body: CppCuteBrowserAssetManifestBodyV1 = {
    profileHash: profile.profileHash,
    sourceAbi,
    sourceAbiSha256,
    assetSetSha256,
    dependencyIds: profileRecord.toolchain.dependencies.map((entry) => entry.dependencyId),
    buildSubjectIds: [BUILD_SUBJECT_ID],
    buildProvenancePolicy: BUILD_PROVENANCE_POLICY,
    mountedVirtualRoots,
    assets,
    totals: {
      compressedByteLength: compressed.toString() as WireU64,
      unpackedByteLength: compressed.toString() as WireU64,
      fileContentByteLength: fileContent.toString() as WireU64,
    },
  };
  const input: CppCuteBrowserAssetManifestV1 = {
    schema: CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR,
      minor: CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR,
    },
    manifestId: await deriveCppCuteBrowserAssetManifestId(body),
    body,
  };
  return prepareCppCuteBrowserAssetManifest(input, profile);
}

async function createRequest(profile: PreparedCppCuteFrontendProfile): Promise<PreparedCppCuteFrontendRequest> {
  const files = await Promise.all([
    sourceFile("main-source", MAIN_PATH, null, MAIN_BYTES),
    sourceFile("project-header", HEADER_PATH, "workspace-source", HEADER_BYTES),
  ]);
  files.sort((left, right) => compareUtf8(left.virtualPath, right.virtualPath));
  const tokenStart = DECODER.decode(MAIN_BYTES).indexOf("layout");
  const anchor = {
    virtualPath: MAIN_PATH,
    beginByte: encodeWireU64(BigInt(tokenStart)),
    endByte: encodeWireU64(BigInt(tokenStart + "layout".length)),
    tokenSha256: await sha256Hex(MAIN_BYTES.subarray(tokenStart, tokenStart + "layout".length)),
  };
  const entryBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor,
  };
  const entryRequest: CppCuteFrontendEntryRequestV1 = {
    ...entryBody,
    requestId: await deriveCppCuteFrontendEntryRequestId(entryBody),
  };
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
    version: { major: 1, minor: 0 },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: MAIN_PATH,
    files,
    entryRequests: [entryRequest],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: { major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR, minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR },
    },
    limits: requestLimits(profile),
  };
  const request: CppCuteFrontendRequestV1 = {
    ...body,
    requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
  return prepareCppCuteFrontendRequest(profile, request, files.map((file) => ({
    virtualPath: file.virtualPath,
    bytes: file.virtualPath === MAIN_PATH ? MAIN_BYTES : HEADER_BYTES,
  })));
}

function requestLimits(profile: PreparedCppCuteFrontendProfile): CppCuteFrontendRequestLimitsV1 {
  const limits = profile.extractionLimits;
  return {
    maxSourceFiles: limits.maxSourceFiles,
    maxSourceBytes: limits.maxSourceBytes,
    maxHeaderFiles: limits.maxHeaderFiles,
    maxHeaderBytes: limits.maxHeaderBytes,
    maxIncludeDepth: limits.maxIncludeDepth,
    maxMacroExpansions: limits.maxMacroExpansions,
    maxPreprocessedTokens: limits.maxPreprocessedTokens,
    maxAstNodes: limits.maxAstNodes,
    maxConstexprSteps: limits.maxConstexprSteps,
    maxTemplateInstantiations: limits.maxTemplateInstantiations,
    maxTemplateDepth: limits.maxTemplateDepth,
    maxDeclarations: limits.maxDeclarations,
    maxTypes: limits.maxTypes,
    maxConstants: limits.maxConstants,
    maxLayouts: limits.maxLayouts,
    maxTensors: limits.maxTensors,
    maxOperations: limits.maxOperations,
    maxTargetIntrinsics: limits.maxTargetIntrinsics,
    maxDiagnostics: limits.maxDiagnostics,
    maxOutputBytes: limits.maxOutputBytes,
  };
}

async function sourceFile(
  role: CppCuteFrontendRequestSourceFileV1["role"],
  virtualPath: string,
  includeRootId: string | null,
  bytes: Uint8Array,
): Promise<CppCuteFrontendRequestSourceFileV1> {
  const body = {
    role,
    virtualPath,
    contentSha256: await sha256Hex(bytes),
    byteLength: encodeWireU64(BigInt(bytes.byteLength)),
    includeRootId,
  };
  return { ...body, fileId: await deriveCppCuteFrontendSourceFileId(body) };
}

function hostFetch(bytesByUrl: ReadonlyMap<string, Uint8Array>): CppCuteBrowserHostFetch {
  return async (url) => {
    const bytes = bytesByUrl.get(url);
    const response = new Response((bytes ?? new Uint8Array()).slice().buffer, {
      status: bytes === undefined ? 404 : 200,
      headers: { "content-length": String(bytes?.byteLength ?? 0) },
    });
    Object.defineProperty(response, "url", { configurable: true, value: url });
    Object.defineProperty(response, "redirected", { configurable: true, value: false });
    return response;
  };
}

function expectSessionError(
  operation: () => unknown,
  code: CppCuteBrowserVfsSessionError["code"],
  path: string,
): void {
  expect(operation).toThrowError(expect.objectContaining({ code, path }));
}

function writePath(memory: WebAssembly.Memory, pointer: number, path: string): void {
  memoryBytes(memory, pointer, byteLength(path)).set(ENCODER.encode(path));
}

function memoryBytes(memory: WebAssembly.Memory, pointer: number, length: number): Uint8Array {
  return new Uint8Array(memory.buffer, pointer, length);
}

function readU32(memory: WebAssembly.Memory, pointer: number): number {
  return new DataView(memory.buffer).getUint32(pointer, true);
}

function readU64(memory: WebAssembly.Memory, pointer: number): bigint {
  return new DataView(memory.buffer).getBigUint64(pointer, true);
}

function byteLength(value: string): number {
  return ENCODER.encode(value).byteLength;
}

function wire(value: number): WireU64 {
  return String(value) as WireU64;
}

function compareUtf8(left: string, right: string): number {
  const leftBytes = ENCODER.encode(left);
  const rightBytes = ENCODER.encode(right);
  const length = Math.min(leftBytes.byteLength, rightBytes.byteLength);
  for (let index = 0; index < length; index += 1) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index]! - rightBytes[index]!;
  }
  return leftBytes.byteLength - rightBytes.byteLength;
}

function indexAccounting(filePaths: readonly string[]): {
  readonly indexedNodes: number;
  readonly indexLogicalByteLength: number;
} {
  const nodes = new Set<string>(["/"]);
  for (const filePath of filePaths) {
    nodes.add(filePath);
    const segments = filePath.split("/").slice(1);
    let parent = "";
    for (let index = 0; index < segments.length - 1; index += 1) {
      parent += `/${segments[index]}`;
      nodes.add(parent);
    }
  }
  let indexLogicalByteLength = 0;
  for (const path of nodes) {
    const basename = path === "/" ? "" : path.slice(path.lastIndexOf("/") + 1);
    indexLogicalByteLength += 32 + byteLength(path) + byteLength(basename);
  }
  return { indexedNodes: nodes.size, indexLogicalByteLength };
}

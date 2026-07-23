import { sha256Hex, type WireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  CppCuteBrowserHeaderPackAssemblyError,
  assembleCppCuteBrowserHeaderPacks,
  canonicalCppCuteBrowserHeaderPackSelectionBytes,
  copyAssembledCppCuteBrowserHeaderPackBytes,
  prepareCppCuteBrowserHeaderPackSelection,
  unwrapAssembledCppCuteBrowserHeaderPacks,
  unwrapPreparedCppCuteBrowserHeaderPackSelection,
  type AssembleCppCuteBrowserHeaderPacksInput,
  type CppCuteBrowserHeaderPackSelectionPackInput,
  type CppCuteBrowserHeaderPackSourceFileInput,
  type PreparedCppCuteBrowserHeaderPackSelection,
} from "../../src/cpp_cute_browser_header_pack_assembly.js";
import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "../../src/cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES,
  deriveCppCuteBrowserVfsContentSetSha256,
  inspectCppCuteBrowserVfsPack,
  unwrapInspectedCppCuteBrowserVfsPack,
  type CppCuteBrowserVfsPackEntry,
} from "../../src/cpp_cute_browser_vfs_pack.js";
import {
  prepareCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import { createCppCuteBrowserProfileInput } from "./support/cpp_cute_frontend_fixtures.js";

const ENCODER = new TextEncoder();
const CUTLASS_COMMIT = "b78588d1630aa6643bf021613717bafb705df4ef";

interface FixtureFile {
  readonly includeRootId: string;
  readonly virtualPath: string;
  readonly bytes: Uint8Array;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
  readonly licenseComponentIds: readonly string[];
}

interface HeaderPackFixture {
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly packs: readonly CppCuteBrowserHeaderPackSelectionPackInput[];
  readonly files: readonly FixtureFile[];
  readonly selection: PreparedCppCuteBrowserHeaderPackSelection;
}

async function fixture(options: {
  readonly peakWorkingSetByteLimit?: number;
  readonly cutlassSelectionFileCount?: number;
} = {}): Promise<HeaderPackFixture> {
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  let files: FixtureFile[] = await Promise.all([
    fixtureFile("clang-resource", "__clang_cuda_runtime_wrapper.h", "// clang wrapper\n", "clang"),
    fixtureFile("cuda", "cuda_runtime.h", "// CUDA 12.6.3\n", "cuda-toolkit-12.6.3-headers"),
    fixtureFile("cutlass", "cute/layout.hpp", "// CuTe layout\n", "cutlass"),
    fixtureFile("cutlass", "cute/tensor.hpp", "// CuTe tensor\n", "cutlass"),
    fixtureFile("cxx-stdlib", "vector", "// libc++ vector\n", "libcxx"),
    fixtureFile("linux-sysroot", "stdint.h", "// sysroot stdint\n", "linux-sysroot"),
  ]);
  if (options.cutlassSelectionFileCount !== undefined) {
    const sharedTail = "x".repeat(54);
    files = [
      ...files.filter((file) => file.includeRootId !== "cutlass"),
      ...Array.from({ length: options.cutlassSelectionFileCount }, (_, index): FixtureFile => ({
        includeRootId: "cutlass",
        virtualPath: `${String(index).padStart(5, "0")}-${sharedTail}.h`,
        bytes: new Uint8Array(),
        contentSha256: "0".repeat(64),
        byteLength: "0" as WireU64,
        licenseComponentIds: ["cutlass"],
      })),
    ];
  }
  const hashes = new Map<string, string>();
  for (const includeRootId of ["clang-resource", "cuda", "cutlass", "cxx-stdlib", "linux-sysroot"]) {
    hashes.set(includeRootId, await contentSet(files.filter((file) => file.includeRootId === includeRootId)));
  }
  const profileInput = createCppCuteBrowserProfileInput({
    buildProvenanceLockSha256: buildInputLock.resourceSha256,
  }) as unknown as Record<string, unknown>;
  bindProfileToCurrentLock(profileInput, hashes);
  if (options.peakWorkingSetByteLimit !== undefined) {
    const deployment = profileInput["deployment"] as Record<string, unknown>;
    const compilerRuntime = deployment["compilerRuntime"] as Record<string, unknown>;
    const vfs = compilerRuntime["virtualFileSystem"] as Record<string, unknown>;
    vfs["maxRetainedHostPackByteLength"] = options.peakWorkingSetByteLimit;
  }
  const profile = await prepareCppCuteFrontendProfile(profileInput);
  const packs = packInputs(files);
  const selection = await prepareCppCuteBrowserHeaderPackSelection({
    buildInputLock,
    profile,
    packs: [...packs].reverse(),
  });
  return { buildInputLock, profile, packs, files, selection };
}

async function fixtureFile(
  includeRootId: string,
  virtualPath: string,
  contents: string,
  licenseComponentId: string,
): Promise<FixtureFile> {
  const bytes = ENCODER.encode(contents);
  return {
    includeRootId,
    virtualPath,
    bytes,
    contentSha256: await sha256Hex(bytes),
    byteLength: String(bytes.byteLength) as WireU64,
    licenseComponentIds: [licenseComponentId],
  };
}

async function contentSet(files: readonly FixtureFile[]): Promise<string> {
  const entries: CppCuteBrowserVfsPackEntry[] = files
    .map((file) => ({
      virtualPath: file.virtualPath,
      contentSha256: file.contentSha256,
      byteLength: file.byteLength,
    }))
    .sort((left, right) => left.virtualPath.localeCompare(right.virtualPath));
  return deriveCppCuteBrowserVfsContentSetSha256(entries);
}

function packInputs(files: readonly FixtureFile[]): readonly CppCuteBrowserHeaderPackSelectionPackInput[] {
  return ["clang-resource", "cuda", "cutlass", "cxx-stdlib", "linux-sysroot"].map(
    (includeRootId) => ({
      includeRootId,
      files: files
        .filter((file) => file.includeRootId === includeRootId)
        .map((file) => ({
          virtualPath: file.virtualPath,
          contentSha256: file.contentSha256,
          byteLength: file.byteLength,
          licenseComponentIds: file.licenseComponentIds,
        })),
    }),
  );
}

function bindProfileToCurrentLock(input: Record<string, unknown>, hashes: ReadonlyMap<string, string>): void {
  const toolchain = input["toolchain"] as Record<string, unknown>;
  const compiler = toolchain["compiler"] as Record<string, unknown>;
  compiler["version"] = "22.1.8";
  compiler["buildId"] = "llvmorg-22.1.8";
  compiler["resourceDirectorySha256"] = hashes.get("clang-resource")!;
  const dependencies = toolchain["dependencies"] as Record<string, unknown>[];
  for (const dependency of dependencies) {
    const dependencyId = dependency["dependencyId"];
    dependency["headerSetSha256"] = hashes.get(String(dependencyId))!;
    if (dependencyId === "cutlass") dependency["revision"] = CUTLASS_COMMIT;
    if (dependencyId === "cxx-stdlib") {
      dependency["version"] = "22.1.8";
      dependency["revision"] = "llvmorg-22.1.8";
    }
  }
  const vfs = input["virtualFileSystem"] as Record<string, unknown>;
  const roots = vfs["includeRoots"] as Record<string, unknown>[];
  for (const root of roots) {
    const includeRootId = String(root["includeRootId"]);
    if (includeRootId === "workspace-source") continue;
    root["manifestSha256"] = hashes.get(includeRootId)!;
    if (includeRootId === "clang-resource") {
      root["virtualPath"] = "/toolchain/clang/lib/clang/22/include";
    }
  }
  const language = input["language"] as Record<string, unknown>;
  const options = language["options"] as Record<string, unknown>[];
  const forced = options.find((option) => option["kind"] === "forced-include");
  if (forced === undefined) throw new Error("fixture lost forced include");
  forced["virtualPath"] = "/toolchain/clang/lib/clang/22/include/__clang_cuda_runtime_wrapper.h";
}

function assemblyInput(files: readonly FixtureFile[]): AssembleCppCuteBrowserHeaderPacksInput {
  return {
    files: [...files].reverse().map((file): CppCuteBrowserHeaderPackSourceFileInput => ({
      includeRootId: file.includeRootId,
      virtualPath: file.virtualPath,
      bytes: new Uint8Array(file.bytes),
    })),
  };
}

async function expectHeaderPackError(
  promise: Promise<unknown>,
  code: CppCuteBrowserHeaderPackAssemblyError["code"],
  path: string,
): Promise<void> {
  await expect(promise).rejects.toMatchObject({ code, path });
}

describe("C++/CuTe browser parsed-program header-pack assembly", () => {
  it("derives one deterministic complete-profile selection with a notice-policy projection", async () => {
    const first = await fixture();
    const reordered = first.packs.map((pack) => ({
      ...pack,
      files: [...pack.files].reverse(),
    })).reverse();
    const second = await prepareCppCuteBrowserHeaderPackSelection({
      buildInputLock: first.buildInputLock,
      profile: first.profile,
      packs: reordered,
    });

    expect(second).toEqual(first.selection);
    expect(canonicalCppCuteBrowserHeaderPackSelectionBytes(second)).toEqual(
      canonicalCppCuteBrowserHeaderPackSelectionBytes(first.selection),
    );
    const manifest = unwrapPreparedCppCuteBrowserHeaderPackSelection(first.selection).manifest;
    expect(manifest.body.policy).toEqual({
      scope: "complete-profile-header-sets-not-corpus-minimal-closures",
      network: "forbidden",
      buildSysrootReuse: "forbidden",
      inputBytes: "offline-caller-supplied-exact-inventory-match-required",
      noticeBytes: "exact-bytes-unverified",
      fileLicenseMapping: "derived-notice-policy-only-external-review-required",
      outputAuthority: "not-authorized",
      releaseAuthority: "not-authorized",
    });
    expect(manifest.body.packs.map((pack) => [pack.role, pack.outputPath])).toEqual([
      ["cuda", "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs"],
      ["cute", "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs"],
      ["cxx-standard-library", "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs"],
      ["compiler-resource", "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs"],
      ["linux-sysroot", "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs"],
    ]);
    expect(manifest.body.notices.map((notice) => [notice.componentId, notice.reviewStatus])).toEqual([
      ["clang", "reviewed"],
      ["cuda-toolkit-12.6.3-headers", "unresolved"],
      ["cutlass", "reviewed"],
      ["libcxx", "reviewed"],
      ["linux-sysroot", "unresolved"],
    ]);
    expect(manifest.body.licenseReviewComplete).toBe(false);
    expect(first.selection).toMatchObject({
      licenseReviewComplete: false,
      outputIdentityAuthorized: false,
      releaseReady: false,
    });
    expect(first.selection.releaseBlockerIds).toEqual(expect.arrayContaining([
      "cuda-header-redistribution",
      "distributed-file-license-manifest",
      "header-pack-acquisition-materialization-and-build-integration",
      "header-pack-exact-notice-bytes-verification",
      "header-pack-externally-reviewed-distributed-file-license-map",
      "linux-sysroot-redistribution",
      "reproducible-build-evidence",
    ]));
    expect(() => unwrapPreparedCppCuteBrowserHeaderPackSelection(
      { ...first.selection } as PreparedCppCuteBrowserHeaderPackSelection,
    )).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-UNVERIFIED",
    }));
  });

  it("composes exact offline bytes through the closed VFS writer without minting authority", async () => {
    const prepared = await fixture();
    const assembled = await assembleCppCuteBrowserHeaderPacks(
      prepared.selection,
      assemblyInput(prepared.files),
    );
    expect(assembled).toMatchObject({
      selectionId: prepared.selection.selectionId,
      buildInputLockId: prepared.buildInputLock.lockId,
      profileHash: prepared.profile.profileHash,
      networkAccessed: false,
      outputIdentityAuthorized: false,
      buildExecutionObserved: false,
      reproducibilityObserved: false,
      releaseReady: false,
    });
    expect(assembled.outputs).toHaveLength(5);
    expect(BigInt(assembled.peakWorkingSetUpperBoundByteLength)).toBeGreaterThan(
      assembled.outputs.reduce(
        (total, output) => total + BigInt(output.packByteLength),
        0n,
      ),
    );
    expect(BigInt(assembled.peakWorkingSetUpperBoundByteLength)).toBeLessThanOrEqual(
      BigInt(assembled.peakWorkingSetByteLimit),
    );
    const cutlassBytes = copyAssembledCppCuteBrowserHeaderPackBytes(assembled, "cutlass");
    const inspected = await inspectCppCuteBrowserVfsPack(cutlassBytes);
    expect(unwrapInspectedCppCuteBrowserVfsPack(inspected).entries.map((entry) => entry.virtualPath)).toEqual([
      "cute/layout.hpp",
      "cute/tensor.hpp",
    ]);
    cutlassBytes.fill(0);
    expect(copyAssembledCppCuteBrowserHeaderPackBytes(assembled, "cutlass")[0]).not.toBe(0);
    expect(unwrapAssembledCppCuteBrowserHeaderPacks(assembled)).toEqual({
      selection: prepared.selection,
      outputs: assembled.outputs,
    });
    expect(() => copyAssembledCppCuteBrowserHeaderPackBytes(
      { ...assembled } as typeof assembled,
      "cutlass",
    )).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-UNVERIFIED",
    }));
  });

  it("rejects profile/lock drift, incomplete packs, and caller-widened license notices", async () => {
    const prepared = await fixture();
    await expect(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: { ...prepared.buildInputLock } as PreparedCppCuteBrowserBuildInputLock,
        profile: prepared.profile,
        packs: prepared.packs,
      }),
    ).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-UNVERIFIED",
    });

    const wrongLockProfileInput = createCppCuteBrowserProfileInput() as unknown as Record<string, unknown>;
    const wrongLockProfile = await prepareCppCuteFrontendProfile(wrongLockProfileInput);
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: wrongLockProfile,
        packs: prepared.packs,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-MISMATCH",
      "$.input.profile",
    );

    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: prepared.packs.slice(1),
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-MISMATCH",
      "$.input.packs",
    );

    const widened = structuredClone(prepared.packs) as unknown as Array<Record<string, unknown>>;
    const clangPack = widened.find((pack) => pack["includeRootId"] === "clang-resource")!;
    const clangFiles = clangPack["files"] as Array<Record<string, unknown>>;
    clangFiles[0]!["licenseComponentIds"] = ["clang", "llvm"];
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: widened as unknown as readonly CppCuteBrowserHeaderPackSelectionPackInput[],
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-MISMATCH",
      "$.input.packs[0].files[0].licenseComponentIds",
    );

    const changedInventory = structuredClone(prepared.packs) as unknown as Array<Record<string, unknown>>;
    const cudaPack = changedInventory.find((pack) => pack["includeRootId"] === "cuda")!;
    const cudaFiles = cudaPack["files"] as Array<Record<string, unknown>>;
    cudaFiles[0]!["contentSha256"] = "0".repeat(64);
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: changedInventory as unknown as readonly CppCuteBrowserHeaderPackSelectionPackInput[],
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-MISMATCH",
      "$.input.packs[1].files",
    );
  });

  it("rejects non-enumerable and symbol shape widening without reading hidden values", async () => {
    const prepared = await fixture();
    const hiddenArrayProperty = structuredClone(prepared.packs);
    Object.defineProperty(hiddenArrayProperty[0]!.files, "hidden", {
      configurable: true,
      value: "must-not-be-semantic",
    });
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: hiddenArrayProperty,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-INVALID",
      "$.input.packs[0].files",
    );

    const symbolRecordProperty = structuredClone(prepared.packs);
    Object.defineProperty(symbolRecordProperty[0]!.files[0]!, Symbol("hidden"), {
      configurable: true,
      value: "must-not-be-semantic",
    });
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: symbolRecordProperty,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-INVALID",
      "$.input.packs[0].files[0]",
    );
  });

  it("rejects byte substitution, missing files, accessors, and overlong final mounts", async () => {
    const prepared = await fixture();
    const changed = assemblyInput(prepared.files) as { files: CppCuteBrowserHeaderPackSourceFileInput[] };
    changed.files[0] = {
      ...changed.files[0]!,
      bytes: new Uint8Array(changed.files[0]!.bytes).fill(0x78),
    };
    await expectHeaderPackError(
      assembleCppCuteBrowserHeaderPacks(prepared.selection, changed),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-HASH-MISMATCH",
      "$.input.files[0].bytes",
    );

    const missing = assemblyInput(prepared.files);
    await expectHeaderPackError(
      assembleCppCuteBrowserHeaderPacks(prepared.selection, { files: missing.files.slice(1) }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-MISMATCH",
      "$.input.files",
    );

    const accessor = assemblyInput(prepared.files) as { files: CppCuteBrowserHeaderPackSourceFileInput[] };
    let getterCalls = 0;
    Object.defineProperty(accessor.files[0], "bytes", {
      enumerable: true,
      get() {
        getterCalls += 1;
        return Uint8Array.of(1);
      },
    });
    await expectHeaderPackError(
      assembleCppCuteBrowserHeaderPacks(prepared.selection, accessor),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-INVALID",
      "$.input.files[0].bytes",
    );
    expect(getterCalls).toBe(0);

    const overlong = structuredClone(prepared.packs) as unknown as Array<Record<string, unknown>>;
    const clangPack = overlong.find((pack) => pack["includeRootId"] === "clang-resource")!;
    const clangFiles = clangPack["files"] as Array<Record<string, unknown>>;
    clangFiles[0]!["virtualPath"] = `${"a".repeat(4_060)}.h`;
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: overlong as unknown as readonly CppCuteBrowserHeaderPackSelectionPackInput[],
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-RESOURCE-LIMIT",
      "$.input.packs[0].files[0].virtualPath",
    );
  });

  it("enforces one global file-count and logical-metadata budget before inventory hashing", async () => {
    const prepared = await fixture();
    const overCount = prepared.packs.map((pack) => ({ ...pack }));
    overCount[0] = {
      ...overCount[0]!,
      files: Array.from({ length: 60_000 }) as unknown as
        CppCuteBrowserHeaderPackSelectionPackInput["files"],
    };
    overCount[1] = {
      ...overCount[1]!,
      files: Array.from({ length: 50_001 }) as unknown as
        CppCuteBrowserHeaderPackSelectionPackInput["files"],
    };
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: overCount,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-RESOURCE-LIMIT",
      "$.input.packs",
    );

    const overMetadata = prepared.packs.map((pack) => ({ ...pack }));
    const cutlassIndex = overMetadata.findIndex((pack) => pack.includeRootId === "cutlass");
    const sharedLongTail = "x".repeat(3_990);
    overMetadata[cutlassIndex] = {
      ...overMetadata[cutlassIndex]!,
      files: Array.from({ length: 6_300 }, (_, index) => ({
        virtualPath: `${String(index).padStart(4, "0")}-${sharedLongTail}.h`,
        contentSha256: "0".repeat(64),
        byteLength: "0" as WireU64,
        licenseComponentIds: ["cutlass"],
      })),
    };
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: overMetadata,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-RESOURCE-LIMIT",
      "$.input.packs",
    );
  });

  it("hashes and canonicalizes a matching 20k-entry selection beyond semantic-core defaults", async () => {
    const prepared = await fixture({ cutlassSelectionFileCount: 20_000 });
    expect(prepared.selection.fileCount).toBe(20_004);
    expect(canonicalCppCuteBrowserHeaderPackSelectionBytes(
      prepared.selection,
    ).byteLength).toBeGreaterThan(2 * 1024 * 1024);
    expect(unwrapPreparedCppCuteBrowserHeaderPackSelection(
      prepared.selection,
    ).manifest.body.packs.find((pack) => pack.includeRootId === "cutlass")?.fileCount).toBe(
      "20000",
    );
  });

  it("rejects cumulative content-set strings before the VFS hash boundary", async () => {
    const prepared = await fixture();
    const largeStrings = prepared.packs.map((pack) => ({ ...pack }));
    const cutlassIndex = largeStrings.findIndex((pack) => pack.includeRootId === "cutlass");
    const sharedTail = "x".repeat(840);
    largeStrings[cutlassIndex] = {
      ...largeStrings[cutlassIndex]!,
      files: Array.from({ length: 18_000 }, (_, index) => ({
        virtualPath: `${String(index).padStart(5, "0")}-${sharedTail}.h`,
        contentSha256: "0".repeat(64),
        byteLength: "0" as WireU64,
        licenseComponentIds: ["cutlass"],
      })),
    };
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: largeStrings,
      }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-RESOURCE-LIMIT",
      "$.input.packs[2].files",
    );
  });

  it("rejects a byte-copy peak that the former two-pack-copy projection admitted", async () => {
    const peakLimit = 400;
    const prepared = await fixture({ peakWorkingSetByteLimit: peakLimit });
    const firstPack = unwrapPreparedCppCuteBrowserHeaderPackSelection(
      prepared.selection,
    ).manifest.body.packs[0]!;
    const firstPackByteLength = CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES +
      firstPack.files.reduce(
        (total, file) => total + 42 + ENCODER.encode(file.virtualPath).byteLength +
          Number(file.byteLength),
        0,
      );
    expect(2 * firstPackByteLength).toBeLessThanOrEqual(peakLimit);
    expect(3 * firstPackByteLength).toBeGreaterThan(peakLimit);
    await expectHeaderPackError(
      assembleCppCuteBrowserHeaderPacks(
        prepared.selection,
        assemblyInput(prepared.files),
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-RESOURCE-LIMIT",
      "$.packs[0].workingSet",
    );
  });

  it("honors cancellation without changing release or execution claims", async () => {
    const prepared = await fixture();
    const controller = new AbortController();
    controller.abort();
    await expectHeaderPackError(
      prepareCppCuteBrowserHeaderPackSelection({
        buildInputLock: prepared.buildInputLock,
        profile: prepared.profile,
        packs: prepared.packs,
      }, { signal: controller.signal }),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-CANCELLED",
      "$.signal",
    );
    await expectHeaderPackError(
      assembleCppCuteBrowserHeaderPacks(
        prepared.selection,
        assemblyInput(prepared.files),
        { signal: controller.signal },
      ),
      "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-PACK-CANCELLED",
      "$.signal",
    );
  });
});

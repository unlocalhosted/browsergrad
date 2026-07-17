import {
  hashCanonicalJson,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";
import {
  prepareCppCuteBrowserAssetManifest,
} from "../../src/cpp_cute_browser_assets.js";
import {
  CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES,
  CppCuteBrowserVfsPackError,
  canonicalCppCuteBrowserVfsPackBytes,
  canonicalInspectedCppCuteBrowserVfsPackBytes,
  copyVerifiedCppCuteBrowserVfsPackFileRange,
  deriveCppCuteBrowserVfsContentSetSha256,
  inspectCppCuteBrowserVfsPack,
  unwrapInspectedCppCuteBrowserVfsPack,
  unwrapVerifiedCppCuteBrowserVfsPack,
  verifyCppCuteBrowserVfsPackAsset,
  type CppCuteBrowserVfsPackEntry,
} from "../../src/cpp_cute_browser_vfs_pack.js";
import { createCppCuteBrowserAssetFixture } from "./support/cpp_cute_browser_asset_fixtures.js";

const MAGIC = new TextEncoder().encode("BGVFSPK1");
const TEXT_ENCODER = new TextEncoder();

interface InputFile {
  readonly path: string;
  readonly bytes: Uint8Array;
}

interface PackFixture {
  readonly bytes: Uint8Array;
  readonly expected: {
    readonly packSha256: string;
    readonly packByteLength: WireU64;
    readonly fileContentByteLength: WireU64;
    readonly contentSetSha256: string;
  };
  readonly indexStart: number;
  readonly dataStart: number;
}

async function createPack(files: readonly InputFile[]): Promise<PackFixture> {
  const prepared = await Promise.all(files.map(async (file) => ({
    path: file.path,
    pathBytes: TEXT_ENCODER.encode(file.path),
    bytes: new Uint8Array(file.bytes),
    sha256: await sha256Hex(file.bytes),
  })));
  const indexLength = prepared.reduce((total, file) => total + 2 + file.pathBytes.byteLength + 8 + 32, 0);
  const dataLength = prepared.reduce((total, file) => total + file.bytes.byteLength, 0);
  const indexStart = CPP_CUTE_BROWSER_VFS_PACK_HEADER_BYTES;
  const dataStart = indexStart + indexLength;
  const bytes = new Uint8Array(dataStart + dataLength);
  bytes.set(MAGIC, 0);
  const header = new DataView(bytes.buffer);
  header.setUint16(8, 1, true);
  header.setUint16(10, 0, true);
  header.setUint32(12, prepared.length, true);
  header.setBigUint64(16, BigInt(indexLength), true);
  header.setBigUint64(24, BigInt(dataLength), true);

  const entries: CppCuteBrowserVfsPackEntry[] = [];
  let indexOffset = indexStart;
  let dataOffset = 0;
  for (const file of prepared) {
    header.setUint16(indexOffset, file.pathBytes.byteLength, true);
    indexOffset += 2;
    bytes.set(file.pathBytes, indexOffset);
    indexOffset += file.pathBytes.byteLength;
    header.setBigUint64(indexOffset, BigInt(file.bytes.byteLength), true);
    indexOffset += 8;
    bytes.set(hexBytes(file.sha256), indexOffset);
    indexOffset += 32;
    bytes.set(file.bytes, dataStart + dataOffset);
    entries.push({
      virtualPath: file.path,
      contentSha256: file.sha256,
      byteLength: wire(file.bytes.byteLength),
    });
    dataOffset += file.bytes.byteLength;
  }
  const indexBytes = bytes.slice(indexStart, dataStart);
  bytes.set(hexBytes(await sha256Hex(indexBytes)), 32);
  const contentSetSha256 = await deriveCppCuteBrowserVfsContentSetSha256(entries);
  bytes.set(hexBytes(contentSetSha256), 64);
  const packSha256 = await sha256Hex(bytes);
  return {
    bytes,
    indexStart,
    dataStart,
    expected: {
      packSha256,
      packByteLength: wire(bytes.byteLength),
      fileContentByteLength: wire(dataLength),
      contentSetSha256,
    },
  };
}

function wire(value: number): WireU64 {
  return String(value) as WireU64;
}

function hexBytes(value: string): Uint8Array {
  return Uint8Array.from(value.match(/../gu) ?? [], (byte) => Number.parseInt(byte, 16));
}

async function expectPackError(
  operation: Promise<unknown>,
  code: CppCuteBrowserVfsPackError["code"],
  path: string,
): Promise<void> {
  await expect(operation).rejects.toMatchObject({ code, path });
}

describe("C++/CuTe closed browser VFS pack", () => {
  it("verifies exact canonical regular-file bytes and returns opaque immutable authority", async () => {
    const fixture = await createPack([
      { path: "clang/22.1.8/include/__stddef_size_t.h", bytes: TEXT_ENCODER.encode("using size_t = __SIZE_TYPE__;\n") },
      { path: "cute/layout.hpp", bytes: TEXT_ENCODER.encode("#pragma once\n") },
      { path: "empty", bytes: new Uint8Array() },
    ]);
    const inspected = await inspectCppCuteBrowserVfsPack(fixture.bytes);

    expect(inspected).toMatchObject({
      packSha256: fixture.expected.packSha256,
      packByteLength: fixture.expected.packByteLength,
      fileContentByteLength: fixture.expected.fileContentByteLength,
      contentSetSha256: fixture.expected.contentSetSha256,
      fileCount: 3,
    });
    expect(Object.isFrozen(inspected)).toBe(true);
    const record = unwrapInspectedCppCuteBrowserVfsPack(inspected);
    expect(record.entries.map((entry) => entry.virtualPath)).toEqual([
      "clang/22.1.8/include/__stddef_size_t.h",
      "cute/layout.hpp",
      "empty",
    ]);
    expect(Object.isFrozen(record.entries)).toBe(true);
    expect(Object.isFrozen(record.entries[0])).toBe(true);
    expect(() => unwrapInspectedCppCuteBrowserVfsPack({ ...inspected })).toThrowError(
      expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-BROWSER-VFS-UNVERIFIED" }),
    );

    const copy = canonicalInspectedCppCuteBrowserVfsPackBytes(inspected);
    copy[0] = 0;
    expect(canonicalInspectedCppCuteBrowserVfsPackBytes(inspected)).toEqual(fixture.bytes);
  });

  it("rejects magic, version, reserved header growth, truncation, and trailing bytes", async () => {
    const fixture = await createPack([{ path: "a.h", bytes: Uint8Array.of(1) }]);
    for (const [offset, value, path] of [
      [0, 0, "$bytes.magic"],
      [8, 2, "$bytes.version.major"],
      [10, 1, "$bytes.version.minor"],
    ] as const) {
      const bytes = new Uint8Array(fixture.bytes);
      bytes[offset] = value;
      await expectPackError(
        inspectCppCuteBrowserVfsPack(bytes),
        offset === 8 || offset === 10
          ? "BG-COMPILER-CPP-CUTE-BROWSER-VFS-UNSUPPORTED-VERSION"
          : "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
        path,
      );
    }

    const truncated = fixture.bytes.slice(0, -1);
    await expectPackError(
      inspectCppCuteBrowserVfsPack(truncated),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$bytes",
    );
    const trailing = new Uint8Array(fixture.bytes.byteLength + 1);
    trailing.set(fixture.bytes);
    await expectPackError(
      inspectCppCuteBrowserVfsPack(trailing),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$bytes",
    );
  });

  it("rejects absolute, traversal, backslash, non-NFC, unsorted, duplicate, and file-directory collision paths", async () => {
    for (const paths of [
      ["/absolute.h"],
      ["../escape.h"],
      ["dir\\file.h"],
      ["e\u0301.h"],
      ["\ufeffhidden.h"],
      ["bidi\u202e.h"],
      ["space name.h"],
      ["z.h", "a.h"],
      ["a.h", "a.h"],
      ["dir", "dir/file.h"],
      ["dir", "dir.", "dir/file.h"],
    ]) {
      const fixture = await createPack(paths.map((path, index) => ({ path, bytes: Uint8Array.of(index) })));
      await expectPackError(
        inspectCppCuteBrowserVfsPack(fixture.bytes),
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
        paths.length === 1 ? "$.entries[0].virtualPath" : "$.entries",
      );
    }
  });

  it("rejects impossible data lengths, entry corruption, index corruption, and content-set drift", async () => {
    const fixture = await createPack([
      { path: "a.h", bytes: Uint8Array.of(1) },
      { path: "b.h", bytes: Uint8Array.of(2) },
    ]);

    const lengthDrift = new Uint8Array(fixture.bytes);
    const firstPathBytes = TEXT_ENCODER.encode("a.h").byteLength;
    new DataView(lengthDrift.buffer).setBigUint64(fixture.indexStart + 2 + firstPathBytes, 100n, true);
    lengthDrift.set(hexBytes(await sha256Hex(lengthDrift.slice(fixture.indexStart, fixture.dataStart))), 32);
    await expectPackError(
      inspectCppCuteBrowserVfsPack(lengthDrift),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.entries[0].byteLength",
    );

    const contentDrift = new Uint8Array(fixture.bytes);
    contentDrift[fixture.dataStart] = (contentDrift[fixture.dataStart] ?? 0) ^ 0xff;
    await expectPackError(
      inspectCppCuteBrowserVfsPack(contentDrift),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-MISMATCH",
      "$.entries[0].contentSha256",
    );

    const indexDrift = new Uint8Array(fixture.bytes);
    indexDrift[32] = (indexDrift[32] ?? 0) ^ 0xff;
    await expectPackError(
      inspectCppCuteBrowserVfsPack(indexDrift),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-MISMATCH",
      "$bytes.indexSha256",
    );

    const contentSetDrift = new Uint8Array(fixture.bytes);
    contentSetDrift[64] = (contentSetDrift[64] ?? 0) ^ 0xff;
    await expectPackError(
      inspectCppCuteBrowserVfsPack(contentSetDrift),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-MISMATCH",
      "$bytes.contentSetSha256",
    );
  });

  it("snapshots caller bytes before hashing and rejects hostile wrappers", async () => {
    const fixture = await createPack([{ path: "a.h", bytes: Uint8Array.of(1, 2, 3) }]);
    const mutableBytes = new Uint8Array(fixture.bytes);
    const pending = inspectCppCuteBrowserVfsPack(mutableBytes);
    mutableBytes.fill(0);
    await expect(pending).resolves.toMatchObject({ packSha256: fixture.expected.packSha256 });

    const getterOptions: Record<string, unknown> = {};
    Object.defineProperty(getterOptions, "limits", { enumerable: true, get: () => ({}) });
    await expectPackError(
      inspectCppCuteBrowserVfsPack(fixture.bytes, getterOptions),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      "$.options.limits",
    );

    if (typeof SharedArrayBuffer !== "undefined") {
      const shared = new Uint8Array(new SharedArrayBuffer(fixture.bytes.byteLength));
      shared.set(fixture.bytes);
      await expectPackError(
        inspectCppCuteBrowserVfsPack(shared),
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
        "$bytes",
      );
    }
  });

  it("binds exact pack bytes to one prepared manifest asset and mount", async () => {
    const packFixture = await createPack([
      { path: "include/__stddef_size_t.h", bytes: TEXT_ENCODER.encode("using size_t = __SIZE_TYPE__;\n") },
    ]);
    const assetFixture = await createCppCuteBrowserAssetFixture({
      packOverrides: {
        "clang-resource": {
          sha256: packFixture.expected.packSha256,
          byteLength: Number(packFixture.expected.packByteLength),
          fileContentByteLength: Number(packFixture.expected.fileContentByteLength),
          contentSetSha256: packFixture.expected.contentSetSha256,
        },
      },
    });
    const manifest = await prepareCppCuteBrowserAssetManifest(assetFixture.input, assetFixture.profile);
    const verified = await verifyCppCuteBrowserVfsPackAsset(
      packFixture.bytes,
      manifest,
      "compiler-resource",
    );

    expect(verified).toMatchObject({
      manifestId: manifest.manifestId,
      profileHash: manifest.profileHash,
      assetId: "compiler-resource",
      includeRootId: "clang-resource",
      mountedVirtualRoot: "/toolchain/clang/lib/clang/20/include",
      packSha256: packFixture.expected.packSha256,
      fileContentByteLength: packFixture.expected.fileContentByteLength,
      contentSetSha256: packFixture.expected.contentSetSha256,
    });
    const record = unwrapVerifiedCppCuteBrowserVfsPack(verified);
    expect(record.manifest).toBe(manifest);
    expect(record.asset.kind).toBe("compiler-resource-pack");
    expect(record.entries).toEqual(unwrapInspectedCppCuteBrowserVfsPack(record.pack).entries);
    expect(canonicalCppCuteBrowserVfsPackBytes(verified)).toEqual(packFixture.bytes);
    expect(copyVerifiedCppCuteBrowserVfsPackFileRange(
      verified,
      "include/__stddef_size_t.h",
      6,
      6,
    )).toEqual(TEXT_ENCODER.encode("size_t"));
    expect(() => copyVerifiedCppCuteBrowserVfsPackFileRange(
      verified,
      "include/__stddef_size_t.h",
      0,
      Number(packFixture.expected.fileContentByteLength) + 1,
    )).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VFS-INVALID",
      path: "$.byteLength",
    }));
    expect(() => copyVerifiedCppCuteBrowserVfsPackFileRange(
      { ...verified },
      "include/__stddef_size_t.h",
      0,
      1,
    )).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-VFS-UNVERIFIED",
    }));

    const otherPack = await createPack([{ path: "include/other.h", bytes: Uint8Array.of(1) }]);
    await expectPackError(
      verifyCppCuteBrowserVfsPackAsset(otherPack.bytes, manifest, "compiler-resource"),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-HASH-MISMATCH",
      "$.asset.sha256",
    );
  });

  it("enforces independent pack, index, file-count, path, content, and cancellation limits", async () => {
    const fixture = await createPack([
      { path: "a.h", bytes: Uint8Array.of(1) },
      { path: "b.h", bytes: Uint8Array.of(2) },
    ]);
    for (const [limits, path] of [
      [{ maxPackBytes: fixture.bytes.byteLength - 1 }, "$bytes"],
      [{ maxIndexBytes: 1 }, "$bytes.indexByteLength"],
      [{ maxFiles: 1 }, "$bytes.entryCount"],
      [{ maxPathBytes: 2 }, "$.entries[0].virtualPath"],
      [{ maxFileBytes: 0 }, "$.entries[0].byteLength"],
      [{ maxFileContentBytes: 1 }, "$bytes.dataByteLength"],
    ] as const) {
      await expectPackError(
        inspectCppCuteBrowserVfsPack(fixture.bytes, { limits }),
        "BG-COMPILER-CPP-CUTE-BROWSER-VFS-RESOURCE-LIMIT",
        path,
      );
    }

    const controller = new AbortController();
    controller.abort();
    await expectPackError(
      inspectCppCuteBrowserVfsPack(fixture.bytes, { signal: controller.signal }),
      "BG-COMPILER-CPP-CUTE-BROWSER-VFS-CANCELLED",
      "$.signal",
    );
  });

  it("uses the same canonical content-set projection embedded in the pack", async () => {
    const entries: CppCuteBrowserVfsPackEntry[] = [{
      virtualPath: "a.h",
      contentSha256: "1".repeat(64),
      byteLength: wire(3),
    }];
    expect(await deriveCppCuteBrowserVfsContentSetSha256(entries)).toBe(
      await hashCanonicalJson({
        domain: "browsergrad.compiler.cpp-cute.browser-vfs-content-set.v1",
        files: [{ virtualPath: "a.h", contentSha256: "1".repeat(64), byteLength: wire(3) }],
      }),
    );
  });

  it("uses explicit content-set limits above semantic-core cumulative-string defaults", async () => {
    const sharedTail = "x".repeat(54);
    const entries: CppCuteBrowserVfsPackEntry[] = Array.from(
      { length: 20_000 },
      (_, index) => ({
        virtualPath: `${String(index).padStart(5, "0")}-${sharedTail}.h`,
        contentSha256: "0".repeat(64),
        byteLength: wire(0),
      }),
    );
    await expect(deriveCppCuteBrowserVfsContentSetSha256(entries)).resolves.toMatch(
      /^[0-9a-f]{64}$/u,
    );
  });
});

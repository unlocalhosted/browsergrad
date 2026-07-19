import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { lstat, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { zstdCompressSync } from "node:zlib";
import { afterEach, describe, expect, it } from "vitest";

import {
  CppCuteBrowserArchiveNormalizationError,
  admitCppCuteBrowserBsdtarTool,
  admitPinnedCppCuteBrowserArchiveNormalizationEnvironment,
  copyCppCuteBrowserArchiveNormalizationFile,
  cppCuteBrowserArchiveNormalizationRoots,
  materializeCppCuteBrowserNormalizedArchive,
  requireCppCuteBrowserArchiveNormalizationAuthority,
  requireCppCuteBrowserBsdtarToolAuthority,
} from "./cpp_cute_browser_archive_normalization.mjs";

const TEST_ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("host archive normalization", () => {
  it("binds exact tool bytes and normalizes tar and Debian data streams", async () => {
    const root = await fixtureRoot("positive");
    const selectedTar = tar([file("pkg/include/header.hpp", "header-data\n")]);
    const toolPath = await fakeBsdtar(root, selectedTar, zstdCompressSync(selectedTar));
    const archivePath = join(root, "source.archive");
    await writeFile(archivePath, "caller-observed-archive", { mode: 0o400 });
    const tool = await admitCppCuteBrowserBsdtarTool({ executablePath: toolPath });

    expect(tool.observedVersion).toBe("bsdtar 3.5.3 - libarchive 3.5.3");
    expect(tool.claims.packageToolIdentityPinned).toBe(false);
    expect(tool.executableSha256).toBe(createHash("sha256").update(await readFile(toolPath)).digest("hex"));
    expect(() => requireCppCuteBrowserBsdtarToolAuthority({ ...tool } as never))
      .toThrow(CppCuteBrowserArchiveNormalizationError);

    const direct = await materializeCppCuteBrowserNormalizedArchive({
      archiveFormat: "tar.xz",
      archivePath,
      outputRoot: join(root, "direct"),
      selections: selection(),
      tool,
    });
    const deb = await materializeCppCuteBrowserNormalizedArchive({
      archiveFormat: "deb-data-tar-zstd",
      archivePath,
      outputRoot: join(root, "deb"),
      selections: selection(),
      tool,
    });

    expect(direct.claims).toMatchObject({
      observedArchiveBytesHashed: true,
      expectedArchiveIdentityBound: false,
      strictNormalizedTarParsed: true,
      callerSelectedPathsComplete: false,
    });
    expect(direct.normalizationId).not.toBe(deb.normalizationId);
    expect(deb.intermediate).toMatchObject({
      memberName: "data.tar.zst",
      decompressor: "node:zlib.createZstdDecompress",
      decompressedTarSha256: createHash("sha256").update(selectedTar).digest("hex"),
    });
    expect(deb.processes.map(({ stageId }) => stageId)).toEqual([
      "deb-data-member-read",
      "selected-pax-normalization",
    ]);
    expect(Buffer.from(await copyCppCuteBrowserArchiveNormalizationFile(
      direct,
      "headers",
      "header.hpp",
    )).toString("utf8")).toBe("header-data\n");
    expect(Buffer.from(await copyCppCuteBrowserArchiveNormalizationFile(
      deb,
      "headers",
      "header.hpp",
    )).toString("utf8")).toBe("header-data\n");
    expect(cppCuteBrowserArchiveNormalizationRoots(direct)[0]).toMatchObject({
      selectionId: "headers",
      storageRoot: join(root, "direct", "headers"),
    });
    expect(() => requireCppCuteBrowserArchiveNormalizationAuthority({ ...direct } as never))
      .toThrow(CppCuteBrowserArchiveNormalizationError);
    await expect(admitPinnedCppCuteBrowserArchiveNormalizationEnvironment({ executablePath: toolPath }))
      .rejects.toThrow(CppCuteBrowserArchiveNormalizationError);
  });

  it.runIf(process.platform === "darwin" && process.arch === "arm64" && process.version === "v25.9.0")(
    "pins the reviewed Darwin bsdtar and Node/Zstd builder closure",
    async () => {
      const child = spawnSync(process.execPath, [
        fileURLToPath(new URL("./cpp_cute_browser_archive_normalization.mjs", import.meta.url)),
        "--verify-pinned=/usr/bin/bsdtar",
      ], {
        encoding: "utf8",
        env: { LANG: "C", LC_ALL: "C", TZ: "UTC" },
        shell: false,
      });
      expect({ status: child.status, signal: child.signal, stderr: child.stderr }).toEqual({
        status: 0,
        signal: null,
        stderr: "",
      });
      const environment = JSON.parse(child.stdout) as Record<string, unknown>;

      expect(environment).toMatchObject({
        authority: "package-pinned-archive-normalization-environment",
        executableSha256: "2806c6e01f077f360f4046e597ef1a62d96c772eb937b5c35852ad97c9d0a625",
        nodeZstdRuntime: {
          runtimeVersion: "v25.9.0",
          executableSha256: "4b3fe8b384e30ee917e28a9f5b79a3ca64b72b13b70d9ab2273e6e9a823f4cbf",
          zstdVersion: "1.5.7",
        },
        claims: {
          packageToolIdentityPinned: true,
          nodeZstdRuntimeIdentityPinned: true,
          toolImplementationAttested: false,
          releaseReady: false,
        },
      });
    },
  );

  it("fails closed on nonzero tool execution and removes parser-owned output", async () => {
    const root = await fixtureRoot("failure");
    const toolPath = await fakeBsdtar(root, new Uint8Array(), new Uint8Array(), true);
    const archivePath = join(root, "source.archive");
    await writeFile(archivePath, "archive", { mode: 0o400 });
    const tool = await admitCppCuteBrowserBsdtarTool({ executablePath: toolPath });
    const outputRoot = join(root, "output");

    await expect(materializeCppCuteBrowserNormalizedArchive({
      archiveFormat: "tar.gz",
      archivePath,
      outputRoot,
      selections: selection(),
      tool,
    })).rejects.toSatisfy(expectMessage("exit=7"));
    expect(await exists(outputRoot)).toBe(false);
  });
});

async function fakeBsdtar(
  root: string,
  normalizedTar: Uint8Array,
  compressedMember: Uint8Array,
  fail = false,
): Promise<string> {
  const path = join(root, fail ? "failing-bsdtar" : "bsdtar");
  const source = `#!/bin/sh
if [ "$1" = "--version" ]; then
  printf '%s\\n' 'bsdtar 3.5.3 - libarchive 3.5.3'
elif [ "${fail ? "yes" : "no"}" = "yes" ]; then
  printf '%s\\n' 'fixture normalization failed' >&2
  exit 7
else
  for argument in "$@"; do
    if [ "$argument" = "-xOf" ]; then
      printf '%b' '${shellBytes(compressedMember)}'
      exit 0
    fi
  done
  printf '%b' '${shellBytes(normalizedTar)}'
fi
`;
  await writeFile(path, source, { mode: 0o500 });
  return path;
}

function shellBytes(value: Uint8Array): string {
  return [...value].map((byte) => `\\${byte.toString(8).padStart(3, "0")}`).join("");
}

function selection() {
  return [{
    selectionId: "headers",
    selectionKind: "subtree" as const,
    archiveSubtree: "pkg/include",
    outputSubdirectory: "headers",
  }];
}

type TarEntry = Readonly<{ path: string; bytes: Uint8Array }>;

function file(path: string, value: string): TarEntry {
  return { path, bytes: Buffer.from(value, "utf8") };
}

function tar(entries: readonly TarEntry[]): Buffer {
  const parts: Buffer[] = [];
  for (const value of entries) {
    const header = Buffer.alloc(512);
    writeAscii(header, 0, 100, value.path);
    writeOctal(header, 100, 8, 0o400);
    writeOctal(header, 108, 8, 0);
    writeOctal(header, 116, 8, 0);
    writeOctal(header, 124, 12, value.bytes.byteLength);
    writeOctal(header, 136, 12, 0);
    header.fill(0x20, 148, 156);
    header[156] = "0".charCodeAt(0);
    header.write("ustar\0", 257, "ascii");
    header.write("00", 263, "ascii");
    let checksum = 0;
    for (const byte of header) checksum += byte;
    header.write(checksum.toString(8).padStart(6, "0"), 148, "ascii");
    header[154] = 0;
    header[155] = 0x20;
    parts.push(header, Buffer.from(value.bytes));
    const padding = (512 - (value.bytes.byteLength % 512)) % 512;
    if (padding > 0) parts.push(Buffer.alloc(padding));
  }
  parts.push(Buffer.alloc(1024));
  return Buffer.concat(parts);
}

function writeAscii(buffer: Buffer, offset: number, length: number, value: string): void {
  const bytes = Buffer.from(value, "utf8");
  if (bytes.byteLength > length) throw new Error("fixture path exceeds tar header field");
  bytes.copy(buffer, offset);
}

function writeOctal(buffer: Buffer, offset: number, length: number, value: number): void {
  buffer.write(value.toString(8).padStart(length - 1, "0"), offset, "ascii");
  buffer[offset + length - 1] = 0;
}

async function fixtureRoot(name: string): Promise<string> {
  const root = await realpath(await mkdtemp(join(tmpdir(), `browsergrad-archive-normalize-${name}-`)));
  TEST_ROOTS.push(root);
  return root;
}

async function exists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch {
    return false;
  }
}

function expectMessage(message: string) {
  return (error: unknown) => error instanceof CppCuteBrowserArchiveNormalizationError &&
    error.message.includes(message);
}

import { createHash } from "node:crypto";
import { lstat, mkdtemp, readFile, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  CppCuteBrowserSelectedTarStreamError,
  cppCuteBrowserSelectedTarMaterializationRoots,
  materializeCppCuteBrowserSelectedTarStream,
  requireCppCuteBrowserSelectedTarMaterializationAuthority,
} from "./cpp_cute_browser_selected_tar_stream.mjs";

const TEST_ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("selected normalized tar stream", () => {
  it("streams complete selected trees into private root-independent outputs", async () => {
    const firstRoot = await fixtureRoot("first");
    const secondRoot = await fixtureRoot("second");
    const archive = tar([
      directory("pkg/include/"),
      file("pkg/include/cute/empty.hpp", ""),
      file("pkg/include/cute/layout.hpp", "layout\n"),
      file("pkg/include/cute/tensor.hpp", "tensor\n"),
      directory("pkg/libcxx/"),
      file("pkg/libcxx/vector", "vector\n"),
    ]);
    const selections = [
      { selectionId: "cutlass", archiveSubtree: "pkg/include", outputSubdirectory: "cutlass" },
      { selectionId: "libcxx", archiveSubtree: "pkg/libcxx", outputSubdirectory: "libcxx" },
    ] as const;
    const first = await materializeCppCuteBrowserSelectedTarStream({
      chunks: chunks(archive, [1, 509, 7, 1024, 3]),
      outputRoot: join(firstRoot, "out"),
      selections: [...selections].reverse(),
    });
    const second = await materializeCppCuteBrowserSelectedTarStream({
      chunks: [archive],
      outputRoot: join(secondRoot, "out"),
      selections,
    });

    expect(first).toEqual(second);
    expect(first.materializationId).toMatch(
      /^bg\.cpp\.selected-tar-materialization\.sha256\.[0-9a-f]{64}$/u,
    );
    expect(first.selections.map((selection) => [
      selection.selectionId,
      selection.fileCount,
      selection.fileContentByteLength,
    ])).toEqual([
      ["cutlass", 3, "14"],
      ["libcxx", 1, "7"],
    ]);
    expect(first.totals).toMatchObject({
      selectionCount: 2,
      fileCount: 4,
      fileContentByteLength: "21",
      consumedTarByteLength: String(archive.byteLength),
    });
    expect(await readFile(join(firstRoot, "out", "cutlass", "cute", "layout.hpp"), "utf8"))
      .toBe("layout\n");
    expect(await readFile(join(firstRoot, "out", "cutlass", "cute", "empty.hpp"), "utf8"))
      .toBe("");
    expect(await readFile(join(firstRoot, "out", "libcxx", "vector"), "utf8"))
      .toBe("vector\n");
    expect(cppCuteBrowserSelectedTarMaterializationRoots(first)).toEqual([
      expect.objectContaining({
        selectionId: "cutlass",
        sourceRoot: join(firstRoot, "out", "cutlass"),
      }),
      expect.objectContaining({
        selectionId: "libcxx",
        sourceRoot: join(firstRoot, "out", "libcxx"),
      }),
    ]);
    expect(() => requireCppCuteBrowserSelectedTarMaterializationAuthority({ ...first } as never))
      .toThrow(CppCuteBrowserSelectedTarStreamError);
  });

  it("accepts a strict PAX path override without accepting PAX size or link overrides", async () => {
    const root = await fixtureRoot("pax");
    const relative = `${"segment/".repeat(14)}header.hpp`;
    const selectedPath = `pkg/include/${relative}`;
    const archive = tar([
      pax({ mtime: "1700000000.25", path: selectedPath }),
      file("pkg/include/placeholder", "pax-data\n"),
    ]);
    const manifest = await materializeCppCuteBrowserSelectedTarStream({
      chunks: chunks(archive, [13, 511, 29]),
      outputRoot: join(root, "out"),
      selections: [
        { selectionId: "headers", archiveSubtree: "pkg/include", outputSubdirectory: "headers" },
      ],
    });

    expect(manifest.selections[0]?.files[0]).toMatchObject({
      relativePath: relative,
      contentSha256: createHash("sha256").update("pax-data\n").digest("hex"),
      byteLength: "9",
    });
    expect(await readFile(join(root, "out", "headers", relative), "utf8")).toBe("pax-data\n");

    const forbiddenRoot = await fixtureRoot("pax-size");
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [tar([pax({ size: "1" }), file("pkg/include/header.hpp", "value")])],
      outputRoot: join(forbiddenRoot, "out"),
      selections: [
        { selectionId: "headers", archiveSubtree: "pkg/include", outputSubdirectory: "headers" },
      ],
    })).rejects.toSatisfy(expectMessage("PAX key size is forbidden"));

    const consecutiveRoot = await fixtureRoot("pax-consecutive");
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [tar([
        pax({ path: "pkg/include/first.hpp" }),
        pax({ path: "pkg/include/second.hpp" }),
        file("pkg/include/placeholder", "value"),
      ])],
      outputRoot: join(consecutiveRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("consecutive PAX headers are forbidden"));

    const unknownRoot = await fixtureRoot("pax-unknown");
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [tar([pax({ comment: "ignored" }), file("pkg/include/header.hpp", "value")])],
      outputRoot: join(unknownRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("PAX key comment is not admitted"));
  });

  it("rejects links, traversal, files outside selection, and duplicate selected paths", async () => {
    const linkedRoot = await fixtureRoot("linked");
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [tar([entry("pkg/include/link.hpp", new Uint8Array(), "2")])],
      outputRoot: join(linkedRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("entry type"));
    expect(await exists(join(linkedRoot, "out"))).toBe(false);

    const traversalRoot = await fixtureRoot("traversal");
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [tar([file("pkg/include/../escape.hpp", "escape")])],
      outputRoot: join(traversalRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("parent segments"));

    const outsideRoot = await fixtureRoot("outside");
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [tar([file("pkg/other/file.hpp", "outside")])],
      outputRoot: join(outsideRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("outside selected subtrees"));

    const duplicateRoot = await fixtureRoot("duplicate");
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [tar([
        file("pkg/include/file.hpp", "one"),
        file("pkg/include/file.hpp", "two"),
      ])],
      outputRoot: join(duplicateRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("duplicate selected file path"));
  });

  it("rejects checksum, padding, truncation, and end-marker corruption", async () => {
    const checksumRoot = await fixtureRoot("checksum");
    const badChecksum = tar([file("pkg/include/file.hpp", "value")]);
    badChecksum[0] = badChecksum[0]! ^ 1;
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [badChecksum],
      outputRoot: join(checksumRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("checksum mismatch"));

    const paddingRoot = await fixtureRoot("padding");
    const badPadding = tar([file("pkg/include/file.hpp", "value")]);
    badPadding[512 + 5] = 1;
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [badPadding],
      outputRoot: join(paddingRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("nonzero padding"));

    const truncatedRoot = await fixtureRoot("truncated");
    const truncated = tar([file("pkg/include/file.hpp", "value")]).subarray(0, 514);
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [truncated],
      outputRoot: join(truncatedRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("ended inside an entry"));

    const markerRoot = await fixtureRoot("marker");
    const marker = tar([file("pkg/include/file.hpp", "value")]);
    marker[marker.byteLength - 1] = 1;
    await expect(materializeCppCuteBrowserSelectedTarStream({
      chunks: [marker],
      outputRoot: join(markerRoot, "out"),
      selections: selection(),
    })).rejects.toSatisfy(expectMessage("nonzero bytes follow"));
  });
});

type TarEntry = Readonly<{ path: string; bytes: Uint8Array; type: string }>;

function file(path: string, value: string): TarEntry {
  return entry(path, Buffer.from(value, "utf8"), "0");
}

function directory(path: string): TarEntry {
  return entry(path, new Uint8Array(), "5");
}

function pax(values: Readonly<Record<string, string>>): TarEntry {
  const records = Object.entries(values).map(([key, value]) => paxRecord(key, value)).join("");
  return entry("PaxHeaders/entry", Buffer.from(records, "utf8"), "x");
}

function paxRecord(key: string, value: string): string {
  const body = `${key}=${value}\n`;
  let length = Buffer.byteLength(body, "utf8") + 2;
  while (true) {
    const record = `${length} ${body}`;
    const actual = Buffer.byteLength(record, "utf8");
    if (actual === length) return record;
    length = actual;
  }
}

function entry(path: string, bytes: Uint8Array, type: string): TarEntry {
  return { path, bytes: new Uint8Array(bytes), type };
}

function tar(entries: readonly TarEntry[]): Buffer {
  const parts: Buffer[] = [];
  for (const value of entries) {
    const header = Buffer.alloc(512);
    writeAscii(header, 0, 100, value.path);
    writeOctal(header, 100, 8, value.type === "5" ? 0o700 : 0o400);
    writeOctal(header, 108, 8, 0);
    writeOctal(header, 116, 8, 0);
    writeOctal(header, 124, 12, value.bytes.byteLength);
    writeOctal(header, 136, 12, 0);
    header.fill(0x20, 148, 156);
    header[156] = value.type.charCodeAt(0);
    header.write("ustar\0", 257, "ascii");
    header.write("00", 263, "ascii");
    header.write("root", 265, "ascii");
    header.write("root", 297, "ascii");
    let checksum = 0;
    for (const byte of header) checksum += byte;
    const checksumText = checksum.toString(8).padStart(6, "0");
    header.write(checksumText, 148, "ascii");
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
  if (bytes.byteLength > length) throw new Error("fixture tar path exceeds header field");
  bytes.copy(buffer, offset);
}

function writeOctal(buffer: Buffer, offset: number, length: number, value: number): void {
  const text = value.toString(8).padStart(length - 1, "0");
  buffer.write(text, offset, "ascii");
  buffer[offset + length - 1] = 0;
}

function chunks(bytes: Uint8Array, sizes: readonly number[]): readonly Uint8Array[] {
  const output = [];
  let offset = 0;
  let index = 0;
  while (offset < bytes.byteLength) {
    const length = Math.min(sizes[index % sizes.length]!, bytes.byteLength - offset);
    output.push(bytes.slice(offset, offset + length));
    offset += length;
    index += 1;
  }
  return output;
}

function selection() {
  return [{ selectionId: "headers", archiveSubtree: "pkg/include", outputSubdirectory: "headers" }];
}

async function fixtureRoot(name: string): Promise<string> {
  const root = await realpath(await mkdtemp(join(tmpdir(), `browsergrad-selected-tar-${name}-`)));
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
  return (error: unknown) => error instanceof CppCuteBrowserSelectedTarStreamError &&
    error.message.includes(message);
}

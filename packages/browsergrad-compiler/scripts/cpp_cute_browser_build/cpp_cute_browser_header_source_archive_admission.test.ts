import { mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  CppCuteBrowserHeaderSourceArchiveAdmissionError,
  admitCppCuteBrowserHeaderSourcePlanArchives,
  canonicalCppCuteBrowserHeaderSourceArchiveAdmissionBytes,
  copyCppCuteBrowserHeaderSourceArchive,
  parseCppCuteBrowserHeaderSourceArchiveArguments,
  requireCppCuteBrowserHeaderSourceArchiveAuthority,
} from "./cpp_cute_browser_header_source_archive_admission.mjs";

const SOURCE_IDS = [
  "cuda-cccl-linux-x86-64",
  "cuda-cudart-linux-x86-64",
  "cuda-libcurand-linux-x86-64",
  "cuda-nvcc-linux-x86-64",
  "cutlass",
  "llvm-project",
  "ubuntu-noble-libc6-dev-amd64-cross",
  "ubuntu-noble-linux-libc-dev-amd64-cross",
] as const;
const TEST_ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("browser header-source archive admission", () => {
  it("parses only the exact eight source-plan archive arguments", () => {
    const argumentsInReverse = [...SOURCE_IDS].reverse().map((sourceId) =>
      `--${sourceId}=/tmp/${sourceId}.archive`);
    expect(parseCppCuteBrowserHeaderSourceArchiveArguments(argumentsInReverse)).toEqual({
      archives: SOURCE_IDS.map((sourceId) => ({
        sourceId,
        archivePath: `/tmp/${sourceId}.archive`,
      })),
    });
    expect(() => parseCppCuteBrowserHeaderSourceArchiveArguments(argumentsInReverse.slice(1)))
      .toThrow(CppCuteBrowserHeaderSourceArchiveAdmissionError);
    expect(() => parseCppCuteBrowserHeaderSourceArchiveArguments([
      ...argumentsInReverse.slice(0, -1),
      "--unknown=/tmp/unknown.archive",
    ])).toThrow(CppCuteBrowserHeaderSourceArchiveAdmissionError);
    expect(() => parseCppCuteBrowserHeaderSourceArchiveArguments([
      ...argumentsInReverse.slice(0, -1),
      `--${SOURCE_IDS[0]}=relative`,
    ])).toThrow(CppCuteBrowserHeaderSourceArchiveAdmissionError);
  });

  it("rejects a complete-shaped set whose bytes differ from the exact plan", async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "browsergrad-header-source-admission-")));
    TEST_ROOTS.push(root);
    const archives = [];
    for (const sourceId of SOURCE_IDS) {
      const archivePath = join(root, `${sourceId}.archive`);
      await writeFile(archivePath, sourceId, { mode: 0o400 });
      archives.push({ sourceId, archivePath });
    }

    await expect(admitCppCuteBrowserHeaderSourcePlanArchives({ archives }))
      .rejects.toSatisfy(expectMessage("byte length differs"));
    await expect(admitCppCuteBrowserHeaderSourcePlanArchives({
      archives: archives.map((archive, index) => index === 0
        ? { ...archive, sourceId: SOURCE_IDS[1] }
        : archive),
    })).rejects.toSatisfy(expectMessage("source IDs must be unique"));
  });

  it("keeps exact-plan authority and archive paths opaque", async () => {
    const forged = Object.freeze({
      schema: "browsergrad.compiler.cpp-cute.browser-header-source-archive-admission",
      claims: { exactCurrentHeaderSourcePlanArchiveBytesVerified: true },
    });
    expect(() => requireCppCuteBrowserHeaderSourceArchiveAuthority(forged as never))
      .toThrow(CppCuteBrowserHeaderSourceArchiveAdmissionError);
    expect(() => canonicalCppCuteBrowserHeaderSourceArchiveAdmissionBytes(forged as never))
      .toThrow(CppCuteBrowserHeaderSourceArchiveAdmissionError);
    await expect(copyCppCuteBrowserHeaderSourceArchive(
      forged as never,
      "cutlass",
      "/tmp/cutlass.copy",
    )).rejects.toSatisfy(expectPath("$.admission"));
  });
});

function expectMessage(message: string) {
  return (error: unknown) => error instanceof CppCuteBrowserHeaderSourceArchiveAdmissionError &&
    error.message.includes(message);
}

function expectPath(path: string) {
  return (error: unknown) => error instanceof CppCuteBrowserHeaderSourceArchiveAdmissionError &&
    error.path === path;
}

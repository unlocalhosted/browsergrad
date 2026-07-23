import { lstat, mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  CppCuteBrowserHeaderSourceExtractionError,
  extractCppCuteBrowserHeaderSourcePlan,
  parseCppCuteBrowserHeaderSourceExtractionArguments,
} from "./cpp_cute_browser_header_source_extraction.mjs";

const TEST_ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("exact header-source extraction", () => {
  it("parses the exact eight-source operational CLI with pnpm separator", () => {
    const parsed = parseCppCuteBrowserHeaderSourceExtractionArguments([
      "--",
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--output-root=/private/tmp/browsergrad-header-sources",
    ]);

    expect(parsed.archives.map(({ sourceId }) => sourceId)).toEqual([
      "cuda-cccl-linux-x86-64",
      "cuda-cudart-linux-x86-64",
      "cuda-libcurand-linux-x86-64",
      "cuda-nvcc-linux-x86-64",
      "cutlass",
      "llvm-project",
      "ubuntu-noble-libc6-dev-amd64-cross",
      "ubuntu-noble-linux-libc-dev-amd64-cross",
    ]);
    expect(parsed.bsdtarPath).toBe("/usr/bin/bsdtar");
    expect(parsed.outputRoot).toBe("/private/tmp/browsergrad-header-sources");
    expect(() => parseCppCuteBrowserHeaderSourceExtractionArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--bsdtar=/usr/local/bin/bsdtar",
    ])).toThrow(CppCuteBrowserHeaderSourceExtractionError);
  });

  it("rejects serialized authorities before creating an output root", async () => {
    const root = await fixtureRoot("forged");
    const outputRoot = join(root, "output");

    await expect(extractCppCuteBrowserHeaderSourcePlan({
      archiveAdmission: {} as never,
      bsdtarTool: {} as never,
      outputRoot,
    })).rejects.toSatisfy((error: unknown) =>
      error instanceof CppCuteBrowserHeaderSourceExtractionError &&
      error.message.includes("exact current header-source archive authority"));
    expect(await exists(outputRoot)).toBe(false);
  });
});

function archiveArguments(): string[] {
  return [
    "--cuda-cccl-linux-x86-64=/private/tmp/cuda-cccl.tar.xz",
    "--cuda-cudart-linux-x86-64=/private/tmp/cuda-cudart.tar.xz",
    "--cuda-libcurand-linux-x86-64=/private/tmp/cuda-libcurand.tar.xz",
    "--cuda-nvcc-linux-x86-64=/private/tmp/cuda-nvcc.tar.xz",
    "--cutlass=/private/tmp/cutlass.tar.gz",
    "--llvm-project=/private/tmp/llvm-project.tar.xz",
    "--ubuntu-noble-libc6-dev-amd64-cross=/private/tmp/libc6-dev.deb",
    "--ubuntu-noble-linux-libc-dev-amd64-cross=/private/tmp/linux-libc-dev.deb",
  ];
}

async function fixtureRoot(name: string): Promise<string> {
  const root = await realpath(await mkdtemp(join(tmpdir(), `browsergrad-header-extract-${name}-`)));
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

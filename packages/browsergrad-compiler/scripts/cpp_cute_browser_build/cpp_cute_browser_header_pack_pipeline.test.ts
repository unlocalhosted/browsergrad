import { chmod, lstat, mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  CppCuteBrowserHeaderPackPipelineError,
  createCppCuteBrowserPrivatePackOutputRoot,
  parseCppCuteBrowserHeaderPackPipelineArguments,
  requireCppCuteBrowserHeaderPackPipelineAuthority,
} from "./cpp_cute_browser_header_pack_pipeline.mjs";

const TEST_ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("exact header-pack pipeline", () => {
  it("rejects forged pipeline authority", () => {
    expect(() => requireCppCuteBrowserHeaderPackPipelineAuthority(Object.freeze({}) as never))
      .toThrow(CppCuteBrowserHeaderPackPipelineError);
  });

  it("parses one no-serialization pipeline invocation", () => {
    const parsed = parseCppCuteBrowserHeaderPackPipelineArguments([
      "--",
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--cuda-redistribution-index=/private/tmp/redistrib_12.6.3.json",
      "--output-root=/private/tmp/browsergrad-header-sources",
      "--pack-output-root=/private/tmp/browsergrad-header-packs",
    ]);

    expect(parsed.archives).toHaveLength(7);
    expect(parsed.bsdtarPath).toBe("/usr/bin/bsdtar");
    expect(parsed.cudaRedistributionIndexPath).toBe("/private/tmp/redistrib_12.6.3.json");
    expect(parsed.sourceOutputRoot).toBe("/private/tmp/browsergrad-header-sources");
    expect(parsed.packOutputRoot).toBe("/private/tmp/browsergrad-header-packs");
    expect(() => parseCppCuteBrowserHeaderPackPipelineArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--cuda-redistribution-index=/private/tmp/redistrib_12.6.3.json",
      "--output-root=/private/tmp/sources",
    ])).toThrow(CppCuteBrowserHeaderPackPipelineError);
    expect(() => parseCppCuteBrowserHeaderPackPipelineArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--cuda-redistribution-index=/private/tmp/redistrib_12.6.3.json",
      "--output-root=/private/tmp/sources",
      "--pack-output-root=/private/tmp/one",
      "--pack-output-root=/private/tmp/two",
    ])).toThrow(CppCuteBrowserHeaderPackPipelineError);
    expect(() => parseCppCuteBrowserHeaderPackPipelineArguments([
      ...archiveArguments(),
      "--bsdtar=/usr/bin/bsdtar",
      "--output-root=/private/tmp/sources",
      "--pack-output-root=/private/tmp/packs",
    ])).toThrow(CppCuteBrowserHeaderPackPipelineError);
  });

  it("creates one private no-clobber pack root and rejects unsafe parents", async () => {
    const parent = await fixtureRoot("output");
    const outputRoot = join(parent, "packs");
    const identity = await createCppCuteBrowserPrivatePackOutputRoot(outputRoot);
    const created = await lstat(outputRoot, { bigint: true });

    expect({ dev: created.dev, ino: created.ino }).toEqual(identity);
    expect(Number(created.mode & 0o077n)).toBe(0);
    await expect(createCppCuteBrowserPrivatePackOutputRoot(outputRoot))
      .rejects.toThrow(CppCuteBrowserHeaderPackPipelineError);

    const permissiveParent = await fixtureRoot("permissive");
    await chmod(permissiveParent, 0o755);
    await expect(createCppCuteBrowserPrivatePackOutputRoot(join(permissiveParent, "packs")))
      .rejects.toThrow(CppCuteBrowserHeaderPackPipelineError);
  });
});

async function fixtureRoot(name: string): Promise<string> {
  const root = await realpath(await mkdtemp(join(tmpdir(), `browsergrad-pack-pipeline-${name}-`)));
  TEST_ROOTS.push(root);
  return root;
}

function archiveArguments(): string[] {
  return [
    "--cuda-cccl-linux-x86-64=/private/tmp/cuda-cccl.tar.xz",
    "--cuda-cudart-linux-x86-64=/private/tmp/cuda-cudart.tar.xz",
    "--cuda-nvcc-linux-x86-64=/private/tmp/cuda-nvcc.tar.xz",
    "--cutlass=/private/tmp/cutlass.tar.gz",
    "--llvm-project=/private/tmp/llvm-project.tar.xz",
    "--ubuntu-noble-libc6-dev-amd64-cross=/private/tmp/libc6-dev.deb",
    "--ubuntu-noble-linux-libc-dev-amd64-cross=/private/tmp/linux-libc-dev.deb",
  ];
}

import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { inspectCppCuteBrowserVfsPack } from "../../src/cpp_cute_browser_vfs_pack.js";
import {
  inventoryCppCuteBrowserHeaderPackSources,
  type CppCuteBrowserHeaderPackInventoryInput,
} from "./cpp_cute_browser_header_pack_inventory.mjs";
import {
  CppCuteBrowserHeaderPackMaterializationError,
  canonicalCppCuteBrowserHeaderPackMaterializationBytes,
  materializeCppCuteBrowserHeaderPacks,
  parseCppCuteBrowserHeaderPackMaterializationArguments,
  requireCppCuteBrowserHeaderPackMaterializationAuthority,
} from "./cpp_cute_browser_header_pack_materialization.mjs";

const TEST_ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("browser header-pack materialization", () => {
  it("writes and independently inspects the exact five build-lock VFS packs", async () => {
    const fixture = await sourceFixture("exact");
    const inventory = await inventoryCppCuteBrowserHeaderPackSources(fixture.input);
    const outputRoot = await emptyDirectory(fixture.root, "outputs");
    const result = await materializeCppCuteBrowserHeaderPacks({ inventory, outputRoot });

    expect(result).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.browser-header-pack-materialization",
      version: 2,
      authority: "deterministic-vfs-pack-materialization-only",
      inventoryId: inventory.inventoryId,
      outputRoot,
      claims: {
        exactSourceBytesReverified: true,
        canonicalVfsPacksIndependentlyInspected: true,
        networkAccessed: false,
        licenseReviewComplete: false,
        assetManifestBound: false,
        buildExecuted: false,
        reproducibilityObserved: false,
        releaseReady: false,
      },
    });
    expect(result.outputs).toHaveLength(5);
    expect(canonicalCppCuteBrowserHeaderPackMaterializationBytes(result))
      .toEqual(canonicalJsonBytes(result));
    expect(() => requireCppCuteBrowserHeaderPackMaterializationAuthority({ ...result }))
      .toThrow(CppCuteBrowserHeaderPackMaterializationError);
    for (const output of result.outputs) {
      const bytes = await readFile(join(outputRoot, output.outputPath));
      const inspected = await inspectCppCuteBrowserVfsPack(new Uint8Array(bytes));
      expect(inspected).toMatchObject({
        packSha256: output.packSha256,
        packByteLength: output.packByteLength,
        fileContentByteLength: output.fileContentByteLength,
        contentSetSha256: output.contentSetSha256,
        fileCount: output.fileCount,
      });
      expect(output.contentSetSha256).toBe(
        inventory.packs.find((pack) => pack.includeRootId === output.includeRootId)?.contentSetSha256,
      );
    }
  });

  it("is byte-deterministic across distinct source and output roots", async () => {
    const first = await sourceFixture("first");
    const second = await sourceFixture("second");
    const firstInventory = await inventoryCppCuteBrowserHeaderPackSources(first.input);
    const secondInventory = await inventoryCppCuteBrowserHeaderPackSources(second.input);
    const firstOutput = await emptyDirectory(first.root, "outputs");
    const secondOutput = await emptyDirectory(second.root, "outputs");
    const one = await materializeCppCuteBrowserHeaderPacks({
      inventory: firstInventory,
      outputRoot: firstOutput,
    });
    const two = await materializeCppCuteBrowserHeaderPacks({
      inventory: secondInventory,
      outputRoot: secondOutput,
    });

    expect(one.outputs).toEqual(two.outputs);
    expect(one.totalPackByteLength).toBe(two.totalPackByteLength);
  });

  it("rejects nonempty and symlinked output roots before writing packs", async () => {
    const fixture = await sourceFixture("output-boundary");
    const inventory = await inventoryCppCuteBrowserHeaderPackSources(fixture.input);
    const nonempty = await emptyDirectory(fixture.root, "nonempty");
    await writeFile(join(nonempty, "owned.txt"), "keep\n");
    await expect(materializeCppCuteBrowserHeaderPacks({ inventory, outputRoot: nonempty }))
      .rejects.toSatisfy(expectMaterializationPath("$.input.outputRoot"));
    expect(await readFile(join(nonempty, "owned.txt"), "utf8")).toBe("keep\n");

    const permissive = await emptyDirectory(fixture.root, "permissive");
    await chmod(permissive, 0o755);
    await expect(materializeCppCuteBrowserHeaderPacks({ inventory, outputRoot: permissive }))
      .rejects.toSatisfy(expectMaterializationPath("$.input.outputRoot"));

    const realOutput = await emptyDirectory(fixture.root, "real-output");
    const linkedOutput = join(fixture.root, "linked-output");
    await symlink(realOutput, linkedOutput);
    await expect(materializeCppCuteBrowserHeaderPacks({ inventory, outputRoot: linkedOutput }))
      .rejects.toSatisfy(expectMaterializationPath("$.input.outputRoot"));
  });

  it("rejects forged inventory authority and malformed command arguments", async () => {
    const fixture = await sourceFixture("forged");
    const inventory = await inventoryCppCuteBrowserHeaderPackSources(fixture.input);
    const outputRoot = await emptyDirectory(fixture.root, "outputs");
    await expect(materializeCppCuteBrowserHeaderPacks({
      inventory: { ...inventory },
      outputRoot,
    })).rejects.toSatisfy(expectMaterializationPath("$.input.inventory"));
    expect(() => parseCppCuteBrowserHeaderPackMaterializationArguments([
      "--input=/tmp/spec.json",
    ])).toThrow(CppCuteBrowserHeaderPackMaterializationError);
    expect(() => parseCppCuteBrowserHeaderPackMaterializationArguments([
      "--input=/tmp/spec.json",
      "--output-root=relative",
    ])).toThrow(CppCuteBrowserHeaderPackMaterializationError);
  });
});

async function sourceFixture(name: string): Promise<{
  readonly root: string;
  readonly input: CppCuteBrowserHeaderPackInventoryInput;
}> {
  const root = await trackedTemporaryRoot(`browsergrad-header-materialize-${name}-`);
  const roots = {
    clang: join(root, "clang-resource"),
    cuda: join(root, "cuda"),
    cutlass: join(root, "cutlass"),
    cxx: join(root, "cxx"),
    sysroot: join(root, "sysroot"),
  };
  await Promise.all(Object.values(roots).map((path) => mkdir(path, { mode: 0o700 })));
  await mkdir(join(roots.cutlass, "cute"), { mode: 0o700 });
  await writeFile(join(roots.clang, "__clang_cuda_runtime_wrapper.h"), "// clang\n", { mode: 0o400 });
  await writeFile(join(roots.cuda, "cuda_runtime.h"), "// cuda\n", { mode: 0o400 });
  await writeFile(join(roots.cutlass, "cute", "layout.hpp"), "// layout\n", { mode: 0o400 });
  await writeFile(join(roots.cxx, "vector"), "// vector\n", { mode: 0o400 });
  await writeFile(join(roots.sysroot, "stdint.h"), "// stdint\n", { mode: 0o400 });
  return {
    root,
    input: {
      packs: [
        pack("cuda", roots.cuda, "cuda-toolkit-12.6.3-headers"),
        pack("cutlass", roots.cutlass, "cutlass"),
        pack("linux-sysroot", roots.sysroot, "linux-sysroot"),
        pack("clang-resource", roots.clang, "clang"),
        pack("cxx-stdlib", roots.cxx, "libcxx"),
      ],
    },
  };
}

function pack(
  includeRootId: string,
  sourceRoot: string,
  licenseComponentId: string,
): CppCuteBrowserHeaderPackInventoryInput["packs"][number] {
  return {
    includeRootId,
    sources: [{ sourceRoot, virtualPrefix: "", licenseComponentIds: [licenseComponentId] }],
  };
}

async function emptyDirectory(root: string, name: string): Promise<string> {
  const path = join(root, name);
  await mkdir(path, { mode: 0o700 });
  return path;
}

async function trackedTemporaryRoot(prefix: string): Promise<string> {
  const created = await mkdtemp(join(tmpdir(), prefix));
  const root = await realpath(created);
  TEST_ROOTS.push(root);
  return root;
}

function expectMaterializationPath(path: string): (error: unknown) => boolean {
  return (error) => error instanceof CppCuteBrowserHeaderPackMaterializationError &&
    error.path === path;
}

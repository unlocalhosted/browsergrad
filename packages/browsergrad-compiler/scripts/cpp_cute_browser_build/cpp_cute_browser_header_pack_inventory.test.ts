import { createHash } from "node:crypto";
import {
  chmod,
  link,
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

import {
  CppCuteBrowserHeaderPackInventoryError,
  authorCppCuteBrowserHeaderPackInventory,
  canonicalCppCuteBrowserHeaderPackInventoryBytes,
  copyCppCuteBrowserHeaderPackInventorySourceFile,
  inventoryCppCuteBrowserHeaderPackSources,
  parseCppCuteBrowserHeaderPackInventoryArguments,
  type CppCuteBrowserHeaderPackInventoryInput,
} from "./cpp_cute_browser_header_pack_inventory.mjs";

const TEST_ROOTS: string[] = [];

afterEach(async () => {
  await Promise.all(TEST_ROOTS.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("browser header-pack source inventory", () => {
  it("produces a root-independent canonical inventory with exact license mapping", async () => {
    const first = await fixture("first");
    const second = await fixture("second");
    const one = await inventoryCppCuteBrowserHeaderPackSources(first.input);
    const two = await inventoryCppCuteBrowserHeaderPackSources({
      packs: [...second.input.packs].reverse().map((pack) => ({
        ...pack,
        sources: [...pack.sources].reverse(),
      })),
    });

    expect(one).toEqual(two);
    expect(one.inventoryId).toMatch(/^bg\.cpp\.browser-header-pack-source-inventory\.sha256\.[0-9a-f]{64}$/u);
    expect(one.packs.map((pack) => pack.includeRootId)).toEqual([
      "clang-resource",
      "cuda",
      "cutlass",
      "cxx-stdlib",
      "linux-sysroot",
    ]);
    expect(one.packs[2]).toMatchObject({
      includeRootId: "cutlass",
      intendedAsset: "dependency-header-pack:cutlass",
      outputRole: "cutlass-header-vfs",
      outputPath: "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
      fileCount: 2,
      fileContentByteLength: "29",
    });
    expect(one.packs[2]?.files).toEqual([
      expect.objectContaining({
        virtualPath: "cute/layout.hpp",
        byteLength: "16",
        licenseComponentIds: ["cutlass"],
      }),
      expect.objectContaining({
        virtualPath: "cute/tensor.hpp",
        byteLength: "13",
        licenseComponentIds: ["cutlass"],
      }),
    ]);
    expect(one.claims).toEqual({
      exactReadableSourceTreesVerified: true,
      buildInputLockBound: true,
      headerInputProjectionBound: true,
      networkAccessed: false,
      archiveProvenanceVerified: false,
      generatedClangResourceHeadersComplete: false,
      configuredLibcxxHeaderComplete: false,
      licenseReviewComplete: false,
      headerPackSelectionPrepared: false,
      headerPacksAssembled: false,
      buildExecuted: false,
      releaseReady: false,
    });
    const bytes = canonicalCppCuteBrowserHeaderPackInventoryBytes(one);
    expect(bytes).toEqual(canonicalJsonBytes(one));
    expect(JSON.parse(new TextDecoder().decode(bytes))).toEqual(one);

    const copied = await copyCppCuteBrowserHeaderPackInventorySourceFile(
      one,
      "cutlass",
      "cute/layout.hpp",
    );
    expect(new TextDecoder().decode(copied)).toBe("// layout v1\nabc");
    await expect(copyCppCuteBrowserHeaderPackInventorySourceFile(
      { ...one },
      "cutlass",
      "cute/layout.hpp",
    )).rejects.toSatisfy(expectInventoryPath("$.inventory"));

    await chmod(join(first.cutlassRoot, "cute", "layout.hpp"), 0o600);
    await writeFile(join(first.cutlassRoot, "cute", "layout.hpp"), "changed\n");
    await expect(copyCppCuteBrowserHeaderPackInventorySourceFile(
      one,
      "cutlass",
      "cute/layout.hpp",
    )).rejects.toSatisfy(expectMessage("changed"));
  });

  it("authors one immutable canonical inventory through strict path arguments", async () => {
    const { root, input } = await fixture("author");
    const inputPath = join(root, "inventory-input.json");
    const outputPath = join(root, "inventory-output.json");
    await writeFile(inputPath, JSON.stringify(input), { mode: 0o400 });
    const parsed = parseCppCuteBrowserHeaderPackInventoryArguments([
      `--output=${outputPath}`,
      `--input=${inputPath}`,
    ]);
    const report = await authorCppCuteBrowserHeaderPackInventory(parsed);
    const bytes = await readFile(outputPath);

    expect(report).toMatchObject({
      outputPath,
      packCount: 5,
      fileCount: 6,
      inventorySha256: createHash("sha256").update(bytes).digest("hex"),
      inventoryByteLength: bytes.byteLength,
      releaseReady: false,
    });
    expect(bytes).toEqual(Buffer.from(canonicalJsonBytes(JSON.parse(bytes.toString("utf8")))));
    await expect(authorCppCuteBrowserHeaderPackInventory(parsed))
      .rejects.toSatisfy(expectInventoryPath("$.input.outputPath"));
  });

  it("rejects symlinks, hard links, and non-portable source names", async () => {
    const linked = await fixture("linked");
    await symlink(
      join(linked.cutlassRoot, "cute", "layout.hpp"),
      join(linked.cutlassRoot, "cute", "alias.hpp"),
    );
    await expect(inventoryCppCuteBrowserHeaderPackSources(linked.input))
      .rejects.toSatisfy(expectMessage("symbolic links are forbidden"));

    const hardLinked = await fixture("hard-linked");
    await link(
      join(hardLinked.cutlassRoot, "cute", "layout.hpp"),
      join(hardLinked.cutlassRoot, "cute", "alias.hpp"),
    );
    await expect(inventoryCppCuteBrowserHeaderPackSources(hardLinked.input))
      .rejects.toSatisfy(expectMessage("exactly one hard link"));

    const nonPortable = await fixture("non-portable");
    await writeFile(join(nonPortable.cutlassRoot, "cute", "bad name.hpp"), "bad\n");
    await expect(inventoryCppCuteBrowserHeaderPackSources(nonPortable.input))
      .rejects.toSatisfy(expectMessage("non-portable path segment"));
  });

  it("rejects cross-source virtual file and directory collisions", async () => {
    const { input, root } = await fixture("collision");
    const collisionRoot = join(root, "collision-source");
    await mkdir(collisionRoot, { mode: 0o700 });
    await writeFile(join(collisionRoot, "cute"), "collision\n", { mode: 0o400 });
    const cutlass = input.packs.find((pack) => pack.includeRootId === "cutlass")!;
    const conflicting: CppCuteBrowserHeaderPackInventoryInput = {
      packs: input.packs.map((pack) => pack === cutlass
        ? {
            ...pack,
            sources: [
              ...pack.sources,
              { sourceRoot: collisionRoot, virtualPrefix: "", licenseComponentIds: ["cutlass"] },
            ],
          }
        : pack),
    };
    await expect(inventoryCppCuteBrowserHeaderPackSources(conflicting))
      .rejects.toSatisfy(expectMessage("collides with a directory"));
  });

  it("rejects ambient object behavior and malformed arguments before filesystem access", async () => {
    const hostile = Object.defineProperty({}, "packs", { get: () => [] });
    await expect(inventoryCppCuteBrowserHeaderPackSources(hostile as never))
      .rejects.toSatisfy(expectInventoryPath("$.input.packs"));
    expect(() => parseCppCuteBrowserHeaderPackInventoryArguments(["--input=/tmp/input"])).toThrow(
      CppCuteBrowserHeaderPackInventoryError,
    );
    expect(() => parseCppCuteBrowserHeaderPackInventoryArguments([
      "--input=relative",
      "--output=/tmp/output",
    ])).toThrow(CppCuteBrowserHeaderPackInventoryError);

    const current = await fixture("wrong-license");
    const cutlass = current.input.packs.find((pack) => pack.includeRootId === "cutlass")!;
    await expect(inventoryCppCuteBrowserHeaderPackSources({
      packs: current.input.packs.map((pack) => pack === cutlass
        ? {
            ...pack,
            sources: pack.sources.map((source) => ({
              ...source,
              licenseComponentIds: ["linux-sysroot"],
            })),
          }
        : pack),
    })).rejects.toSatisfy(expectMessage("current build-lock notice policy"));
  });
});

async function fixture(name: string): Promise<{
  readonly root: string;
  readonly cutlassRoot: string;
  readonly input: CppCuteBrowserHeaderPackInventoryInput;
}> {
  const created = await mkdtemp(join(tmpdir(), `browsergrad-header-inventory-${name}-`));
  const root = await realpath(created);
  TEST_ROOTS.push(root);
  const cutlassRoot = join(root, "cutlass");
  const cuteRoot = join(cutlassRoot, "cute");
  const clangResource = join(root, "clang-resource");
  const cuda = join(root, "cuda");
  const cxx = join(root, "cxx");
  const sysroot = join(root, "sysroot");
  await mkdir(cuteRoot, { recursive: true, mode: 0o700 });
  await mkdir(clangResource, { mode: 0o700 });
  await mkdir(cuda, { mode: 0o700 });
  await mkdir(cxx, { mode: 0o700 });
  await mkdir(sysroot, { mode: 0o700 });
  await writeFile(join(cuteRoot, "layout.hpp"), "// layout v1\nabc", { mode: 0o400 });
  await writeFile(join(cuteRoot, "tensor.hpp"), "// tensor v1\n", { mode: 0o400 });
  await writeFile(join(clangResource, "__clang_cuda_runtime_wrapper.h"), "// clang\n", { mode: 0o400 });
  await writeFile(join(cuda, "cuda_runtime.h"), "// cuda\n", { mode: 0o400 });
  await writeFile(join(cxx, "vector"), "// vector\n", { mode: 0o400 });
  await writeFile(join(sysroot, "stdint.h"), "// stdint\n", { mode: 0o400 });
  return {
    root,
    cutlassRoot,
    input: {
      packs: [
        {
          includeRootId: "linux-sysroot",
          sources: [
            { sourceRoot: sysroot, virtualPrefix: "", licenseComponentIds: ["linux-sysroot"] },
          ],
        },
        {
          includeRootId: "cutlass",
          sources: [
            { sourceRoot: cuteRoot, virtualPrefix: "cute", licenseComponentIds: ["cutlass"] },
          ],
        },
        {
          includeRootId: "clang-resource",
          sources: [
            { sourceRoot: clangResource, virtualPrefix: "", licenseComponentIds: ["clang"] },
          ],
        },
        {
          includeRootId: "cxx-stdlib",
          sources: [
            { sourceRoot: cxx, virtualPrefix: "", licenseComponentIds: ["libcxx"] },
          ],
        },
        {
          includeRootId: "cuda",
          sources: [
            {
              sourceRoot: cuda,
              virtualPrefix: "",
              licenseComponentIds: ["cuda-toolkit-12.6.3-headers"],
            },
          ],
        },
      ],
    },
  };
}

function expectInventoryPath(path: string): (error: unknown) => boolean {
  return (error) => error instanceof CppCuteBrowserHeaderPackInventoryError && error.path === path;
}

function expectMessage(message: string): (error: unknown) => boolean {
  return (error) => error instanceof CppCuteBrowserHeaderPackInventoryError &&
    error.message.includes(message);
}

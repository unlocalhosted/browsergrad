import { chmod, lstat, mkdir, mkdtemp, rm, symlink, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  refreshCachedCppCuteToolchainOutputs,
} from "./cpp_cute_browser_build_cache_reuse.mjs";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("cached Clang toolchain reuse", () => {
  it("refreshes only owned object and archive outputs", async () => {
    const root = await privateRoot();
    const libraryRoot = join(root, "lib");
    await mkdir(libraryRoot, { mode: 0o700 });
    const objectPath = join(libraryRoot, "input.o");
    const archivePath = join(libraryRoot, "libclangAST.a");
    const ignoredPath = join(libraryRoot, "flags.make");
    await Promise.all([
      writeFile(objectPath, "object", { mode: 0o600 }),
      writeFile(archivePath, "archive", { mode: 0o600 }),
      writeFile(ignoredPath, "flags", { mode: 0o600 }),
    ]);
    await Promise.all([objectPath, archivePath, ignoredPath].map((path) => utimes(path, 1, 1)));

    await expect(refreshCachedCppCuteToolchainOutputs({ wasmBuildRoot: root })).resolves.toBe(2);
    expect((await lstat(objectPath)).mtimeMs).toBeGreaterThan(1_000);
    expect((await lstat(archivePath)).mtimeMs).toBeGreaterThan(1_000);
    expect((await lstat(ignoredPath)).mtimeMs).toBe(1_000);
  });

  it("rejects a symbolic-link toolchain output without following it", async () => {
    const root = await privateRoot();
    const target = join(root, "target");
    await writeFile(target, "target", { mode: 0o600 });
    await symlink(target, join(root, "poison.o"));

    await expect(refreshCachedCppCuteToolchainOutputs({ wasmBuildRoot: root }))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-CACHE-REUSE",
      });
  });
});

async function privateRoot() {
  const root = await mkdtemp(join(tmpdir(), "browsergrad-cache-reuse-"));
  roots.push(root);
  await chmod(root, 0o700);
  return root;
}

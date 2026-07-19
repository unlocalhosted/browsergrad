import { constants } from "node:fs";
import { lstat, open, readdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path/posix";

import {
  CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS,
} from "./cpp_cute_browser_build_executor_fs.mjs";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-CACHE-REUSE";
const MAX_SCANNED_CACHE_ENTRIES = 200_000;

export class CppCuteBrowserBuildCacheReuseError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserBuildCacheReuseError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Invalidates only BrowserGrad-owned extractor objects in an admitted CMake
 * tree. LLVM/Clang objects remain hot, while every source edit is recompiled.
 *
 * @param {{ wasmBuildRoot: string; sourcePaths: readonly string[] }} input
 */
export async function invalidateCachedCppCuteExtractorObjects(input) {
  if (typeof process.getuid !== "function") {
    fail("$input.wasmBuildRoot", "cache reuse requires POSIX uid support");
  }
  const currentUid = process.getuid();
  const targetRoot = join(
    input.wasmBuildRoot,
    "tools",
    "browsergrad_extractor",
    "CMakeFiles",
    "browsergrad-cpp-cute-extractor.dir",
  );
  const target = await lstatIfExists(targetRoot, "$targetRoot");
  if (target === undefined) return;
  if (!target.isDirectory() || target.isSymbolicLink() || target.uid !== currentUid ||
      (target.mode & 0o022) !== 0) {
    fail("$targetRoot", "cached extractor target must be a private owned directory");
  }

  for (const sourcePath of input.sourcePaths) {
    if (!/^[A-Za-z0-9._+/-]+$/u.test(sourcePath) || sourcePath.startsWith("/") ||
        sourcePath.split("/").some((component) => component === "" || component === "." || component === "..")) {
      fail(`$sourcePaths.${sourcePath}`, "extractor source path is not portable and relative");
    }
    if (!sourcePath.endsWith(".cpp")) continue;
    const diagnosticPath = `$objects.${sourcePath}`;
    const objectPath = join(targetRoot, `${sourcePath}.o`);
    const before = await lstatIfExists(objectPath, diagnosticPath);
    if (before === undefined) continue;
    if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1 ||
        before.uid !== currentUid) {
      fail(diagnosticPath, "cached extractor object is not an owned single-link regular file");
    }
    const rebound = await lstat(objectPath);
    if (before.dev !== rebound.dev || before.ino !== rebound.ino || rebound.isSymbolicLink()) {
      fail(diagnosticPath, "cached extractor object identity changed before invalidation");
    }
    await rm(objectPath, { force: false, maxRetries: 0 });
    await CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS.syncDirectory(dirname(objectPath));
  }
}

/**
 * Rebinds cached LLVM/Clang object and archive mtimes after CMake has refreshed
 * its generated dependency files. The cache key binds the immutable toolchain
 * inputs; this operation is diagnostic-only and never authorizes its contents.
 *
 * @param {{ wasmBuildRoot: string }} input
 * @returns {Promise<number>}
 */
export async function refreshCachedCppCuteToolchainOutputs(input) {
  if (typeof process.getuid !== "function") {
    fail("$input.wasmBuildRoot", "cache reuse requires POSIX uid support");
  }
  const currentUid = process.getuid();
  const refreshTimeSeconds = Date.now() / 1_000;
  const pending = [input.wasmBuildRoot];
  let scannedEntryCount = 0;
  let refreshedFileCount = 0;

  while (pending.length > 0) {
    const directoryPath = pending.pop();
    const directory = await lstat(directoryPath);
    if (!directory.isDirectory() || directory.isSymbolicLink() ||
        directory.uid !== currentUid || (directory.mode & 0o022) !== 0) {
      fail("$cacheTree", "cached build ancestry must contain only private owned directories");
    }
    const entries = (await readdir(directoryPath, { withFileTypes: true }))
      .sort((left, right) => left.name.localeCompare(right.name));
    scannedEntryCount += entries.length;
    if (scannedEntryCount > MAX_SCANNED_CACHE_ENTRIES) {
      fail("$cacheTree", "cached build tree exceeds the diagnostic entry limit");
    }
    for (const entry of entries) {
      const entryPath = join(directoryPath, entry.name);
      const stat = await lstat(entryPath);
      if (stat.isSymbolicLink()) {
        if (entry.name.endsWith(".o") || entry.name.endsWith(".a")) {
          fail("$cacheTree", "cached toolchain output must not be a symbolic link");
        }
        continue;
      }
      if (stat.isDirectory()) {
        pending.push(entryPath);
        continue;
      }
      if (!entry.name.endsWith(".o") && !entry.name.endsWith(".a")) continue;
      if (!stat.isFile() || stat.nlink !== 1 || stat.uid !== currentUid ||
          (stat.mode & 0o022) !== 0) {
        fail("$cacheTree", "cached toolchain output must be a private owned regular file");
      }
      const handle = await open(entryPath, constants.O_WRONLY | constants.O_NOFOLLOW);
      try {
        const before = await handle.stat();
        if (!sameFileIdentity(stat, before)) {
          fail("$cacheTree", "cached toolchain output identity changed before refresh");
        }
        await handle.utimes(refreshTimeSeconds, refreshTimeSeconds);
        await handle.sync();
        const after = await handle.stat();
        if (!sameFileIdentity(before, after)) {
          fail("$cacheTree", "cached toolchain output identity changed during refresh");
        }
      } finally {
        await handle.close();
      }
      refreshedFileCount += 1;
    }
  }
  return refreshedFileCount;
}

async function lstatIfExists(path, diagnosticPath) {
  try {
    return await lstat(path);
  } catch (cause) {
    if (cause instanceof Error && "code" in cause && cause.code === "ENOENT") {
      return undefined;
    }
    fail(diagnosticPath, "failed to inspect cached build path", { cause });
  }
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino &&
    left.size === right.size && left.nlink === right.nlink;
}

function fail(path, message, options) {
  throw new CppCuteBrowserBuildCacheReuseError(path, message, options);
}

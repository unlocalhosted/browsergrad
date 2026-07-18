import { lstat, rm } from "node:fs/promises";
import { dirname, join } from "node:path/posix";

import {
  CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS,
} from "./cpp_cute_browser_build_executor_fs.mjs";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-CACHE-REUSE";

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

function fail(path, message, options) {
  throw new CppCuteBrowserBuildCacheReuseError(path, message, options);
}

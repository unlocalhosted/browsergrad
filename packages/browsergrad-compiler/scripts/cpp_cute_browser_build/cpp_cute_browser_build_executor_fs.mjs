import { constants } from "node:fs";
import { link, lstat, open, rm } from "node:fs/promises";

/**
 * Internal filesystem effect boundary for the build executor.
 *
 * Production callers cannot replace this object. Tests mock this module itself
 * so commit/rollback failures can be exercised without adding a caller-owned
 * filesystem capability to the executor API.
 */
export const CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS = Object.freeze({
  link,
  lstat,
  rm,
  /** @param {string} path */
  syncDirectory: async (path) => {
    const handle = await open(path, constants.O_RDONLY | constants.O_DIRECTORY);
    try {
      await handle.sync();
    } finally {
      await handle.close();
    }
  },
  /** @param {import("node:fs/promises").FileHandle} handle @param {string} _purpose */
  closeFileHandle: async (handle, _purpose) => handle.close(),
});

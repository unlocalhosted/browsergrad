import {
  chmodSync,
  mkdtempSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  resolveNativeCompiler,
} from "./cpp_cute_browser_native_test_harness.js";

describe("native C++ test harness compiler discovery", () => {
  it("preserves a validated C++ driver symlink while recording canonical identity", () => {
    const root = mkdtempSync(join(tmpdir(), "browsergrad-native-cxx-"));
    const canonicalCompiler = join(root, "clang-18");
    const cxxDriver = join(root, "clang++");

    try {
      writeFileSync(
        canonicalCompiler,
        "#!/bin/sh\nprintf 'clang version 18.1.3\\n'\n",
        { mode: 0o755 },
      );
      chmodSync(canonicalCompiler, 0o755);
      symlinkSync(canonicalCompiler, cxxDriver);

      expect(resolveNativeCompiler([cxxDriver])).toEqual({
        path: cxxDriver,
        canonicalPath: realpathSync(canonicalCompiler),
        isClang: true,
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects relative compiler candidates", () => {
    expect(resolveNativeCompiler(["clang++"])).toBeUndefined();
  });
});

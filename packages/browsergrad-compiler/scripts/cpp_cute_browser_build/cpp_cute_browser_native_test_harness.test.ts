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
        [
          "#!/bin/sh",
          "if [ \"${1:-}\" = \"--version\" ]; then",
          "  printf 'Ubuntu clang version 22.1.8\\n'",
          "else",
          "  printf '#define __clang__ 1\\n'",
          "fi",
          "",
        ].join("\n"),
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

  it("does not trust a Clang-shaped version banner without compiler macros", () => {
    const root = mkdtempSync(join(tmpdir(), "browsergrad-fake-clang-"));
    const compiler = join(root, "clang++");
    try {
      writeFileSync(
        compiler,
        [
          "#!/bin/sh",
          "if [ \"${1:-}\" = \"--version\" ]; then",
          "  printf 'clang version 22.1.8\\n'",
          "else",
          "  printf '#define __GNUC__ 14\\n'",
          "fi",
          "",
        ].join("\n"),
        { mode: 0o755 },
      );
      expect(resolveNativeCompiler([compiler])).toMatchObject({
        path: compiler,
        isClang: false,
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

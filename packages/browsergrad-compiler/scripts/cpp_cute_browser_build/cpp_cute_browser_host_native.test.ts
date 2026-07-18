import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  nativeCompiler as compiler,
  nativeCompilerUnavailableUnlessOptional,
} from "./cpp_cute_browser_native_test_harness.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const nativeSource = join(scriptRoot, "cpp_cute_browser_host_native_test.cpp");

describe("Clang-Wasm browser host boundary", () => {
  it.skipIf(nativeCompilerUnavailableUnlessOptional)(
    "returns deterministic identity-free values and fails ambient host operations",
    () => {
      if (compiler === undefined) throw new Error("native C++ compiler unavailable");
      const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-host-"));
      const executable = join(workingDirectory, "browser-host-native-test");
      try {
        const compilation = spawnSync(compiler, [
          "-std=c++20",
          "-O2",
          "-Wall",
          "-Wextra",
          "-Wpedantic",
          "-Werror",
          nativeSource,
          "-o",
          executable,
        ], { encoding: "utf8", timeout: 60_000 });
        expect(compilation.error).toBeUndefined();
        expect(compilation.status, compilation.stderr).toBe(0);

        const execution = spawnSync(executable, [], {
          encoding: "utf8",
          timeout: 30_000,
        });
        expect(execution.error).toBeUndefined();
        expect(
          execution.status,
          `signal=${execution.signal ?? "none"}\n${execution.stdout}\n${execution.stderr}`,
        ).toBe(0);
      } finally {
        rmSync(workingDirectory, { recursive: true, force: true });
      }
    },
    90_000,
  );
});

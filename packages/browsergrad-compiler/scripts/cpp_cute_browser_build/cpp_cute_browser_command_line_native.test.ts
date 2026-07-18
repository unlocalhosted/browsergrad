import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE,
} from "../../src/resources/cpp_cute_semantic_adapter_manifest_v1.js";
import {
  cppCuteCommandLinePolicyIncludeMatches,
} from "./cpp_cute_browser_command_line_policy_codegen.mjs";
import {
  nativeCompiler as compiler,
  nativeCompilerIsClang,
  nativeCompilerUnavailableUnlessOptional,
} from "./cpp_cute_browser_native_test_harness.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const nativeSource = join(scriptRoot, "cpp_cute_browser_command_line_native_test.cpp");
const implementationSource = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCuteCommandLine.cpp",
);
const generatedPolicySource = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCuteCommandLinePolicy.inc",
);
function compileAndRun(extraFlags: readonly string[]): void {
  if (compiler === undefined) throw new Error("native C++ compiler unavailable");
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-command-line-"));
  const executable = join(workingDirectory, "command-line-native-test");
  try {
    const compilation = spawnSync(compiler, [
      "-std=c++20",
      "-O2",
      "-Wall",
      "-Wextra",
      "-Wpedantic",
      "-Werror",
      "-fno-omit-frame-pointer",
      ...extraFlags,
      nativeSource,
      implementationSource,
      "-o",
      executable,
    ], {
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(compilation.error).toBeUndefined();
    expect(compilation.status, compilation.stderr).toBe(0);

    const execution = spawnSync(executable, [], {
      encoding: "utf8",
      timeout: 30_000,
      env: {
        ...process.env,
        ASAN_OPTIONS: process.platform === "darwin"
          ? "detect_leaks=0:halt_on_error=1"
          : "detect_leaks=1:halt_on_error=1",
        UBSAN_OPTIONS: "halt_on_error=1:print_stacktrace=1",
      },
    });
    expect(execution.error).toBeUndefined();
    expect(
      execution.status,
      `signal=${execution.signal ?? "none"}\n${execution.stdout}\n${execution.stderr}`,
    ).toBe(0);
  } finally {
    rmSync(workingDirectory, { recursive: true, force: true });
  }
}

describe("Clang-Wasm closed compiler-policy materializer", () => {
  it("matches exact generated manifest policy bytes", () => {
    expect(cppCuteCommandLinePolicyIncludeMatches(
      CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE,
      readFileSync(generatedPolicySource),
    )).toBe(true);
  });

  it.skipIf(nativeCompilerUnavailableUnlessOptional)(
    "emits exact argv and rejects unknown, duplicate, reserved, or temporal policy",
    () => compileAndRun([]),
    90_000,
  );

  it.skipIf(!nativeCompilerIsClang)(
    "stays clean under the undefined-behavior sanitizer",
    () => compileAndRun(["-fsanitize=undefined"]),
    90_000,
  );

  // Apple clang's ASan runtime deadlocks inside dyld initialization on this
  // Darwin runner. ASan remains enabled on non-Darwin clang runners.
  it.skipIf(!nativeCompilerIsClang || process.platform === "darwin")(
    "stays clean under the address sanitizer",
    () => compileAndRun(["-fsanitize=address"]),
    90_000,
  );
});

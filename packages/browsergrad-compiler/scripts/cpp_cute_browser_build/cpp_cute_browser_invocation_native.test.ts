import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  nativeCompiler as compiler,
  nativeCompilerIsClang,
  nativeCompilerUnavailableUnlessOptional,
  runNativeTestProcess,
} from "./cpp_cute_browser_native_test_harness.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const nativeSource = join(
  scriptRoot,
  "cpp_cute_browser_invocation_native_test.cpp",
);
const invocationSource = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCuteInvocation.cpp",
);
const commandLineSource = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCuteCommandLine.cpp",
);
const virtualPathSource = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCuteVirtualPath.cpp",
);
async function compileAndRun(extraFlags: readonly string[]): Promise<void> {
  if (compiler === undefined) throw new Error("native C++ compiler unavailable");
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-invocation-"));
  const executable = join(workingDirectory, "invocation-native-test");
  try {
    const compilation = await runNativeTestProcess(compiler, [
      "-std=c++20",
      "-O2",
      "-Wall",
      "-Wextra",
      "-Wpedantic",
      "-Werror",
      "-fno-omit-frame-pointer",
      ...extraFlags,
      nativeSource,
      invocationSource,
      commandLineSource,
      virtualPathSource,
      "-o",
      executable,
    ], {
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(compilation.error).toBeUndefined();
    expect(compilation.status, compilation.stderr).toBe(0);

    const execution = await runNativeTestProcess(executable, [], {
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

describe("sealed Clang 22.1.8 CUDA invocation materializer", () => {
  it.skipIf(nativeCompilerUnavailableUnlessOptional)(
    "owns exact device/host argv and rejects unsafe or conflicting typed input",
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

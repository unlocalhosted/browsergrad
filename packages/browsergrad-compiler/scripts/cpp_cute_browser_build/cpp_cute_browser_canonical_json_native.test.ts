import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import {
  nativeCompiler as compiler,
  nativeCompilerIsClang,
  nativeCompilerUnavailableUnlessOptional,
} from "./cpp_cute_browser_native_test_harness.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const nativeSource = join(scriptRoot, "cpp_cute_browser_canonical_json_native_test.cpp");
const implementationSource = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCuteCanonicalJson.cpp",
);
function compileAndRun(extraFlags: readonly string[]): void {
  if (compiler === undefined) throw new Error("native C++ compiler unavailable");
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-canonical-json-"));
  const executable = join(workingDirectory, "canonical-json-native-test");
  const fixturePath = join(workingDirectory, "canonical.json");
  try {
    writeFileSync(fixturePath, canonicalJsonBytes({
      ascii: "A",
      controls: "\b\t\n\f\r\0",
      nested: [{ maximum: Number.MAX_SAFE_INTEGER, minimum: Number.MIN_SAFE_INTEGER }],
      unicode: ["e\u0301", "é", "😀", "\ue000"],
    }));
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

    const execution = spawnSync(executable, [fixturePath], {
      encoding: "utf8",
      timeout: 30_000,
      env: {
        ...process.env,
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

describe("Clang-Wasm allocation-free canonical JSON validator", () => {
  it.skipIf(nativeCompilerUnavailableUnlessOptional)(
    "matches TypeScript canonical bytes and rejects hostile wire forms",
    () => compileAndRun([]),
    90_000,
  );

  it.skipIf(!nativeCompilerIsClang)(
    "stays clean under the undefined-behavior sanitizer",
    () => compileAndRun(["-fsanitize=undefined"]),
    90_000,
  );
});

import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  canonicalCppCuteFrontendArtifactBytes,
  decodeCppCuteFrontendArtifact,
  verifyCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_artifact.js";
import { createCppCuteArtifactInput } from "../../tests/compiler/support/cpp_cute_frontend_fixtures.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const nativeSource = join(scriptRoot, "cpp_cute_browser_runtime_native_test.cpp");
const compiler = [
  "/usr/bin/clang++",
  "/usr/bin/c++",
  "/usr/bin/g++",
].find((candidate) => existsSync(candidate));

async function canonicalArtifactFixtureBytes(): Promise<Uint8Array> {
  const verified = await verifyCppCuteFrontendArtifact(
    await createCppCuteArtifactInput(),
  );
  const bytes = canonicalCppCuteFrontendArtifactBytes(verified);
  await decodeCppCuteFrontendArtifact(bytes);
  return bytes;
}

function compileAndRun(
  extraFlags: readonly string[],
  artifactBytes: Uint8Array,
): void {
  if (compiler === undefined) throw new Error("native C++ compiler unavailable");
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-runtime-"));
  const executable = join(workingDirectory, "runtime-native-test");
  const artifactPath = join(workingDirectory, "canonical-artifact-v3.json");
  try {
    writeFileSync(artifactPath, artifactBytes);
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
      "-o",
      executable,
    ], {
      encoding: "utf8",
      timeout: 60_000,
    });
    expect(compilation.error).toBeUndefined();
    expect(compilation.status, compilation.stderr).toBe(0);

    const execution = spawnSync(executable, [artifactPath], {
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

describe("Clang-Wasm ABI-1.1 result lifecycle native behavioral model", () => {
  it.skipIf(compiler === undefined)(
    "owns one immutable bounded result across compile, free, failure, and reset",
    async () => compileAndRun([], await canonicalArtifactFixtureBytes()),
    90_000,
  );

  it.skipIf(compiler !== "/usr/bin/clang++")(
    "stays clean under the undefined-behavior sanitizer",
    async () => compileAndRun(
      ["-fsanitize=undefined"],
      await canonicalArtifactFixtureBytes(),
    ),
    90_000,
  );

  // Apple clang's ASan runtime deadlocks inside dyld initialization on this
  // Darwin runner; Linux CI owns address-sanitizer coverage for this model.
  it.skipIf(compiler !== "/usr/bin/clang++" || process.platform === "darwin")(
    "stays clean under the address sanitizer",
    async () => compileAndRun(
      ["-fsanitize=address"],
      await canonicalArtifactFixtureBytes(),
    ),
    90_000,
  );
});

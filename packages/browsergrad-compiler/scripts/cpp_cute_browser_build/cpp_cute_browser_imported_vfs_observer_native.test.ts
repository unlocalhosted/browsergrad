import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  nativeCompiler as compiler,
  nativeCompilerIsClang,
  nativeCompilerUnavailableUnlessOptional,
} from "./cpp_cute_browser_native_test_harness.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const nativeSource = join(
  scriptRoot,
  "cpp_cute_browser_imported_vfs_observer_native_test.cpp",
);
const importedVfsSource = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCuteImportedVfs.cpp",
);
const virtualPathSource = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCuteVirtualPath.cpp",
);
function installLlvmDeclarationStubs(workingDirectory: string): void {
  const adtDirectory = join(workingDirectory, "llvm", "ADT");
  const supportDirectory = join(workingDirectory, "llvm", "Support");
  mkdirSync(adtDirectory, { recursive: true });
  mkdirSync(supportDirectory, { recursive: true });
  writeFileSync(
    join(adtDirectory, "IntrusiveRefCntPtr.h"),
    [
      "#pragma once",
      "namespace llvm { template <typename T> class IntrusiveRefCntPtr; }",
      "",
    ].join("\n"),
  );
  writeFileSync(
    join(supportDirectory, "VirtualFileSystem.h"),
    [
      "#pragma once",
      "namespace llvm::vfs { class FileSystem; }",
      "",
    ].join("\n"),
  );
}

function compileAndRun(extraFlags: readonly string[]): void {
  if (compiler === undefined) throw new Error("native C++ compiler unavailable");
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-vfs-observer-"));
  const executable = join(workingDirectory, "imported-vfs-observer-native-test");
  try {
    installLlvmDeclarationStubs(workingDirectory);
    const compilation = spawnSync(compiler, [
      "-std=c++20",
      "-O2",
      "-Wall",
      "-Wextra",
      "-Wpedantic",
      "-Werror",
      "-fno-omit-frame-pointer",
      ...extraFlags,
      "-I",
      workingDirectory,
      nativeSource,
      virtualPathSource,
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

describe("Clang-Wasm pass-scoped ImportedVFS observer", () => {
  it("commits observation only after the complete imported read loop", () => {
    const source = readFileSync(importedVfsSource, "utf8");
    const getBufferStart = source.indexOf("getBuffer(");
    const closeStart = source.indexOf("std::error_code close() override", getBufferStart);
    const getBuffer = source.slice(getBufferStart, closeStart);
    const readCall = getBuffer.indexOf("bg_vfs_read(");
    const failedReadReturn = getBuffer.indexOf(
      "return imported_vfs_error(read_status);",
      readCall,
    );
    const observationCommit = getBuffer.indexOf(
      "observer_->record_successful_read(",
      failedReadReturn,
    );
    expect(getBufferStart).toBeGreaterThanOrEqual(0);
    expect(closeStart).toBeGreaterThan(getBufferStart);
    expect(readCall).toBeGreaterThanOrEqual(0);
    expect(failedReadReturn).toBeGreaterThan(readCall);
    expect(observationCommit).toBeGreaterThan(failedReadReturn);
    expect(source.slice(0, getBufferStart)).not.toContain(
      "record_successful_read",
    );
  });

  it.skipIf(nativeCompilerUnavailableUnlessOptional)(
    "keeps unique bounded pass records deterministic and probe-free",
    () => compileAndRun([]),
    90_000,
  );

  it.skipIf(!nativeCompilerIsClang)(
    "stays clean under the undefined-behavior sanitizer",
    () => compileAndRun(["-fsanitize=undefined"]),
    90_000,
  );

  // Apple clang's ASan runtime deadlocks inside dyld initialization on this
  // Darwin runner; Linux CI owns address-sanitizer coverage for this model.
  it.skipIf(!nativeCompilerIsClang || process.platform === "darwin")(
    "stays clean under the address sanitizer",
    () => compileAndRun(["-fsanitize=address"]),
    90_000,
  );
});

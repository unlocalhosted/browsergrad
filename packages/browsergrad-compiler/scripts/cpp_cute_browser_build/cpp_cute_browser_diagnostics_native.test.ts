import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  cppCuteDiagnosticNormalizationResourceBytes,
  decodeCppCuteDiagnosticNormalization,
  deriveCppCuteNormalizedDiagnosticId,
  unwrapPreparedCppCuteDiagnosticNormalization,
  type CppCuteSerializedDiagnosticIdMaterial,
} from "../../src/cpp_cute_diagnostic_normalization.js";
import { CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE } from
  "../../src/resources/cpp_cute_diagnostic_normalization_v1.js";
import {
  CPP_CUTE_DIAGNOSTICS_POLICY_INCLUDE_PATH,
  cppCuteDiagnosticsPolicyIncludeMatches,
  renderCppCuteDiagnosticsPolicyInclude,
} from "./cpp_cute_browser_diagnostics_policy_codegen.mjs";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const extractorRoot = join(scriptRoot, "extractor");
const nativeSource = join(scriptRoot, "cpp_cute_browser_diagnostics_native_test.cpp");
const compiler = ["/usr/bin/clang++", "/usr/bin/c++", "/usr/bin/g++"]
  .find((candidate) => existsSync(candidate));
const contractHash = "2".repeat(64);
const spanOne = `bg.cpp.span.sha256.${"1".repeat(64)}`;
const spanTwo = `bg.cpp.span.sha256.${"3".repeat(64)}`;

async function expectedIds(): Promise<readonly [string, string, string]> {
  const prepared = await decodeCppCuteDiagnosticNormalization(
    cppCuteDiagnosticNormalizationResourceBytes(),
  );
  const root: CppCuteSerializedDiagnosticIdMaterial = {
    phase: "parsing",
    severity: "error",
    code: "clang:diag-1234",
    renderedMessage: "expected expression",
    location: { kind: "source", primarySpanId: spanOne, related: [] },
    subject: { kind: "compiler" },
    parentDiagnosticId: null,
  };
  const rootId = await deriveCppCuteNormalizedDiagnosticId(prepared, {
    compilationContractHash: contractHash,
    ownerPassId: "cuda-device-sema",
    diagnostic: root,
  });
  const noteId = await deriveCppCuteNormalizedDiagnosticId(prepared, {
    compilationContractHash: contractHash,
    ownerPassId: "cuda-device-sema",
    diagnostic: {
      phase: "parsing",
      severity: "note",
      code: "clang:diag-1235",
      renderedMessage: "while parsing template",
      location: {
        kind: "source",
        primarySpanId: spanTwo,
        related: [{ spanId: spanOne, message: "instantiated here" }],
      },
      subject: { kind: "compiler" },
      parentDiagnosticId: rootId,
    },
  });
  const customId = await deriveCppCuteNormalizedDiagnosticId(prepared, {
    compilationContractHash: contractHash,
    ownerPassId: "cuda-device-sema",
    diagnostic: {
      phase: "preprocessing",
      severity: "error",
      code: "browsergrad.cpp-cute:temporal-macro-forbidden",
      renderedMessage: "temporal macro is forbidden",
      location: { kind: "source", primarySpanId: spanOne, related: [] },
      subject: { kind: "compiler" },
      parentDiagnosticId: null,
    },
  });
  return [rootId, noteId, customId];
}

async function compileAndRun(extraFlags: readonly string[]): Promise<void> {
  if (compiler === undefined) throw new Error("native C++ compiler unavailable");
  const expected = await expectedIds();
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-diagnostics-"));
  const executable = join(workingDirectory, "diagnostics-native-test");
  try {
    const compilation = spawnSync(compiler, [
      "-std=c++20", "-O1", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
      "-fno-omit-frame-pointer", ...extraFlags,
      nativeSource,
      join(extractorRoot, "BrowserGradCppCuteDiagnostics.cpp"),
      join(extractorRoot, "BrowserGradCppCuteSha256.cpp"),
      join(extractorRoot, "BrowserGradCppCuteVirtualPath.cpp"),
      "-I", scriptRoot,
      "-o", executable,
    ], { encoding: "utf8", timeout: 60_000 });
    expect(compilation.error).toBeUndefined();
    expect(compilation.status, compilation.stderr).toBe(0);

    const execution = spawnSync(executable, expected, {
      encoding: "utf8",
      timeout: 30_000,
      env: {
        ...process.env,
        ASAN_OPTIONS: "detect_leaks=1:halt_on_error=1",
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

describe("native C++/CuTe diagnostic normalization", () => {
  it("keeps generated native policy byte-exact and rejects resource drift", () => {
    const actual = readFileSync(CPP_CUTE_DIAGNOSTICS_POLICY_INCLUDE_PATH);
    expect(cppCuteDiagnosticsPolicyIncludeMatches(
      CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE,
      actual,
    )).toBe(true);
    const drifted = structuredClone(CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE) as unknown as {
      body: { renderedTextPolicy: { maxRenderedMessageBytes: number } };
    };
    drifted.body.renderedTextPolicy.maxRenderedMessageBytes += 1;
    expect(() => renderCppCuteDiagnosticsPolicyInclude(drifted)).toThrow(
      /manifestId/u,
    );
  });

  it.skipIf(compiler === undefined)(
    "matches TypeScript stable IDs and fails closed for hostile states",
    () => compileAndRun([]),
    90_000,
  );

  it.skipIf(compiler !== "/usr/bin/clang++")(
    "stays clean under undefined-behavior sanitizer coverage",
    () => compileAndRun(["-fsanitize=undefined"]),
    90_000,
  );

  it.skipIf(compiler !== "/usr/bin/clang++" || process.platform === "darwin")(
    "stays clean under address and leak sanitizer coverage",
    () => compileAndRun(["-fsanitize=address"]),
    90_000,
  );

  it("pins the exact decoded policy used to derive parity fixtures", async () => {
    const prepared = await decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    );
    expect(unwrapPreparedCppCuteDiagnosticNormalization(prepared)
      .normalization.manifestId).toBe(
      CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE.manifestId,
    );
  });
});

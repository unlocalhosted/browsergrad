import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE,
} from "../../src/resources/cpp_cute_semantic_adapter_manifest_v1.js";
import {
  nativeCompiler as compiler,
  nativeCompilerIsClang,
  nativeCompilerUnavailableUnlessOptional,
  runNativeTestProcess,
} from "./cpp_cute_browser_native_test_harness.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const nativeSource = join(
  scriptRoot,
  "cpp_cute_browser_preprocessor_policy_native_test.cpp",
);
const policyHeader = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCutePreprocessorPolicy.h",
);
const policyImplementation = join(
  scriptRoot,
  "extractor",
  "BrowserGradCppCutePreprocessorPolicy.cpp",
);
async function compileAndRun(extraFlags: readonly string[]): Promise<void> {
  if (compiler === undefined) throw new Error("native C++ compiler unavailable");
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-pp-policy-"));
  const executable = join(workingDirectory, "preprocessor-policy-native-test");
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

describe("Clang 22.1.8 temporal macro preprocessor policy", () => {
  it("keeps native policy constants closed over the semantic-adapter manifest", () => {
    const header = readFileSync(policyHeader, "utf8");
    const temporal = CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE
      .body.temporalMacros;
    expect(header).toContain(`"${temporal.policyId}"`);
    expect(header).toContain(`"${temporal.diagnosticCodes.consultation}"`);
    expect(header).toContain(`"${temporal.diagnosticCodes.mutation}"`);
    for (const macro of temporal.macroNames) {
      expect(header).toContain(`name == "${macro}"`);
    }
    expect(temporal.mode).toBe("reject");
    expect(temporal.consultation).toBe("forbidden");
    expect(temporal.mutation).toBe("forbidden");
    expect(temporal.enforcement).toBe("preprocessor-callback-before-expansion");
  });

  it("uses the exact pinned PPCallbacks surface and fail-closed install hook", () => {
    const source = readFileSync(policyImplementation, "utf8");
    expect(source).toContain('#include "clang/Lex/PPCallbacks.h"');
    expect(source).toContain('#include "clang/Frontend/CompilerInstance.h"');
    expect(source).toContain("class CppCuteTemporalMacroCallbacks final");
    for (const callback of [
      "MacroExpands",
      "Defined",
      "Ifdef",
      "Ifndef",
      "Elifdef",
      "Elifndef",
      "MacroDefined",
      "MacroUndefined",
    ]) {
      expect(source).toMatch(new RegExp(`void ${callback}\\([^)]*`, "u"));
    }
    expect(source).toContain("const clang::MacroArgs*) override");
    expect(source).toContain("const clang::MacroDirective*) override");
    expect(source).toContain("const clang::MacroDefinition&,");
    expect(source).toContain("compiler.hasDiagnostics()");
    expect(source).toContain("compiler.hasPreprocessor()");
    expect(source).toContain("preprocessor.addPPCallbacks(");
    expect(source).toContain("clang::DiagnosticsEngine::Error");
    expect(source).not.toContain("clang::DiagnosticsEngine::Warning");
    expect(source).not.toContain("isIgnored(");
  });

  it("records unsuppressible failure before rendering fixed diagnostics", () => {
    const source = readFileSync(policyImplementation, "utf8");
    const reportStart = source.indexOf(
      "void report(const clang::Token& token, TemporalMacroUse use)",
    );
    const mutationStart = source.indexOf(
      "void report_source_mutation(const clang::Token& token,",
    );
    const fieldsStart = source.indexOf("clang::Preprocessor& preprocessor_;");
    const consultation = source.slice(reportStart, mutationStart);
    const mutation = source.slice(mutationStart, fieldsStart);
    expect(reportStart).toBeGreaterThanOrEqual(0);
    expect(mutationStart).toBeGreaterThan(reportStart);
    expect(fieldsStart).toBeGreaterThan(mutationStart);
    expect(consultation.indexOf("state_->record(")).toBeLessThan(
      consultation.indexOf(".Report("),
    );
    expect(mutation.indexOf("state_->record(")).toBeLessThan(
      mutation.indexOf(".Report("),
    );
    expect(source).toContain("source_manager.getFileID(spelling_location) != predefines_file");
    expect(source).toContain("compiler.getPreprocessorOpts().SourceDateEpoch");
    expect(source).toContain("SourceDateEpoch =");
    expect(source).toContain("kTemporalMacroRejectedRecoveryEpoch");
  });

  it("relies on semantic macro events, leaving comments and strings inert", () => {
    const source = readFileSync(policyImplementation, "utf8");
    expect(source).toContain("token.getIdentifierInfo()");
    expect(source).not.toMatch(
      /Lexer::getSourceText|getBufferData|getMemoryBuffer|std::regex|regex_search/u,
    );
  });

  it.skipIf(nativeCompilerUnavailableUnlessOptional)(
    "classifies only exact macro identifiers and fixed diagnostic families",
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

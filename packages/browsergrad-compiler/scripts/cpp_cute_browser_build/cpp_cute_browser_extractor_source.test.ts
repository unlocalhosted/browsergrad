import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { hashCanonicalJson } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { beforeAll, describe, expect, it } from "vitest";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

const extractorRoot = join(dirname(fileURLToPath(import.meta.url)), "extractor");

let sourceFiles: readonly {
  readonly path: string;
  readonly sha256: string;
  readonly byteLength: string;
}[];
let sourceSetSha256: string;

beforeAll(async () => {
  const prepared = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const source = unwrapPreparedCppCuteBrowserBuildInputLock(prepared)
    .lock.body.recipe.extractorSource;
  sourceFiles = source.files;
  sourceSetSha256 = source.sourceSetSha256;
});

describe("BrowserGrad-owned Clang-WASM extractor source", () => {
  it("matches the exact source file and source-set identities in the build lock", async () => {
    expect(sourceFiles.map((file) => file.path)).toEqual([
      "BrowserGradCppCuteExtractor.cpp",
      "CMakeLists.txt",
    ]);
    for (const file of sourceFiles) {
      const bytes = await readFile(join(extractorRoot, file.path));
      expect(String(bytes.byteLength)).toBe(file.byteLength);
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(file.sha256);
    }
    expect(await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.browser-extractor-source-set.v1",
      files: sourceFiles,
    })).toBe(sourceSetSha256);
  });

  it("builds an Emscripten ES-module factory plus Wasm sidecar, not a misleading standalone Wasm target", async () => {
    const cmake = await readFile(join(extractorRoot, "CMakeLists.txt"), "utf8");

    expect(cmake).toContain("add_llvm_executable(browsergrad-cpp-cute-extractor");
    expect(cmake).toContain("BROWSERGRAD_EXTRACTOR_FACTORY_OUTPUT_PATH");
    expect(cmake).toContain('SUFFIX ".mjs"');
    expect(cmake).not.toContain('SUFFIX ".wasm"');
    for (const library of [
      "clangAST", "clangBasic", "clangDriver", "clangFrontend", "clangIndex",
      "clangLex", "clangParse", "clangSema", "clangSerialization", "clangTooling",
    ]) {
      expect(cmake).toContain(`  ${library}\n`);
    }
  });

  it("contains a real closed-VFS LibTooling action but keeps artifact-ready unreachable", async () => {
    const source = await readFile(
      join(extractorRoot, "BrowserGradCppCuteExtractor.cpp"),
      "utf8",
    );

    expect(source).toContain("class LayoutTraceVisitor final");
    expect(source).toContain("class LayoutTraceAction final");
    expect(source).toContain("clang::tooling::ToolInvocation invocation");
    expect(source).toContain("llvm::vfs::FileSystem");
    expect(source).toContain("ReviewOnlyBlocker::kCustomVfsUnavailable");
    expect(source).not.toMatch(/g_runtime\.status\s*=\s*WireCompileStatus::kArtifactReady/u);
    expect(source).toMatch(/bg_cpp_cute_result_length\(void\)[\s\S]*?return 0U;/u);
    expect(source).toMatch(/bg_cpp_cute_result_pointer\(void\)[\s\S]*?return 0U;/u);
  });
});

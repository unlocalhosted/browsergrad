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
import {
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
  unwrapPreparedCppCuteBrowserRuntimeAbiManifest,
} from "../../dist/cpp_cute_browser_runtime_abi.js";

const extractorRoot = join(dirname(fileURLToPath(import.meta.url)), "extractor");

let sourceFiles: readonly {
  readonly path: string;
  readonly sha256: string;
  readonly byteLength: string;
}[];
let sourceSetSha256: string;
let applicationImportNames: readonly string[];
let runtimeVfsLimits: {
  readonly maxPathByteLength: number;
  readonly maxIndexedNodes: number;
  readonly maxLiveFileHandles: number;
  readonly maxAggregateLiveOpenByteLength: number;
};

beforeAll(async () => {
  const prepared = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const source = unwrapPreparedCppCuteBrowserBuildInputLock(prepared)
    .lock.body.recipe.extractorSource;
  sourceFiles = source.files;
  sourceSetSha256 = source.sourceSetSha256;
  const runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
    cppCuteBrowserRuntimeAbiManifestResourceBytes(),
  );
  const body = unwrapPreparedCppCuteBrowserRuntimeAbiManifest(runtimeAbi)
    .manifest.body;
  applicationImportNames = body.hostImports.functions.map((entry) => entry.fieldName);
  runtimeVfsLimits = body.vfs;
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

  it("declares and consumes only the exact closed-VFS application imports", async () => {
    const source = await readFile(
      join(extractorRoot, "BrowserGradCppCuteExtractor.cpp"),
      "utf8",
    );

    expect([...source.matchAll(/BG_CPP_CUTE_VFS_IMPORT\("([^"]+)"\)/gu)]
      .map((match) => match[1])).toEqual(applicationImportNames);
    expect(source).toContain('__attribute__((import_module("browsergrad_vfs_v1")))');
    for (const name of applicationImportNames) {
      expect([...source.matchAll(new RegExp(`\\b${name}\\(`, "gu"))].length)
        .toBeGreaterThanOrEqual(2);
    }
    expect(source).not.toMatch(
      /getRealFileSystem|createPhysicalFileSystem|std::ifstream|\bfopen\(|\bfetch\(/u,
    );
  });

  it("bounds paths and reads while streaming directories and fail-stopping handle errors", async () => {
    const source = await readFile(
      join(extractorRoot, "BrowserGradCppCuteExtractor.cpp"),
      "utf8",
    );

    expect(source).toContain(
      `kVfsMaximumPathByteLength = ${runtimeVfsLimits.maxPathByteLength}U`,
    );
    expect(source).toContain(
      `kVfsMaximumDirectoryEntryCount = ${runtimeVfsLimits.maxIndexedNodes}U`,
    );
    expect(source).toContain(
      `kVfsMaximumLiveHandleCount = ${runtimeVfsLimits.maxLiveFileHandles}U`,
    );
    expect(source).toContain(
      `kVfsMaximumReadableFileByteLength = ${runtimeVfsLimits.maxAggregateLiveOpenByteLength}ULL`,
    );
    expect(source).toContain("kVfsReadChunkByteLength = 64U * 1024U");
    expect(source).toContain("valid_canonical_path");
    expect(source).toContain("valid_utf8");
    expect(source).toContain("!utf8_byte_less(previous_name_, name)");
    expect(source).toContain("std::error_code load_current()");
    expect(source).not.toContain("std::vector<llvm::vfs::directory_entry>");
    expect(source).toContain("class ImportedVfsOpenGuard final");
    expect(source).toContain("handle_budget_->terminal_error");
    expect(source).toContain("live_ = false;");
    expect(source).toContain("std::errc::operation_canceled");
    expect(source).toContain("std::errc::protocol_error");
    expect(source).not.toMatch(/bg_vfs_(?:cancel|abort|terminate)/u);
  });

  it("wires the imported VFS into LibTooling but keeps artifact-ready unreachable", async () => {
    const source = await readFile(
      join(extractorRoot, "BrowserGradCppCuteExtractor.cpp"),
      "utf8",
    );

    expect(source).toContain("class ImportedVfsFileSystem final");
    expect(source).toContain("class ImportedVfsFile final");
    expect(source).toContain("class ImportedVfsDirectoryIterator final");
    expect(source).toContain("class LayoutTraceVisitor final");
    expect(source).toContain("class LayoutTraceAction final");
    expect(source).toContain("clang::tooling::ToolInvocation invocation");
    expect(source).toContain("clang::FileManager files(file_system_options, imported_closed_vfs())");
    expect(source).toContain("ReviewOnlyBlocker::kCudaDualPassUnavailable");
    expect(source).not.toContain("kCustomVfsUnavailable");
    expect(source).not.toMatch(/g_runtime\.status\s*=\s*WireCompileStatus::kArtifactReady/u);
    expect(source).toMatch(/bg_cpp_cute_result_length\(void\)[\s\S]*?return 0U;/u);
    expect(source).toMatch(/bg_cpp_cute_result_pointer\(void\)[\s\S]*?return 0U;/u);
  });
});

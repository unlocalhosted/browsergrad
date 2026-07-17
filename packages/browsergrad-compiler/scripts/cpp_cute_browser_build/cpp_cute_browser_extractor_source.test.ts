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
const expectedSourcePaths = [
  "BrowserGradCppCuteArtifactV3.cpp",
  "BrowserGradCppCuteArtifactV3.h",
  "BrowserGradCppCuteCanonicalJson.cpp",
  "BrowserGradCppCuteCanonicalJson.h",
  "BrowserGradCppCuteClangAction.cpp",
  "BrowserGradCppCuteClangAction.h",
  "BrowserGradCppCuteCommandLine.cpp",
  "BrowserGradCppCuteCommandLine.h",
  "BrowserGradCppCuteCommandLinePolicy.inc",
  "BrowserGradCppCuteExtractor.cpp",
  "BrowserGradCppCuteImportedVfs.cpp",
  "BrowserGradCppCuteImportedVfs.h",
  "BrowserGradCppCuteMetrics.cpp",
  "BrowserGradCppCuteMetrics.h",
  "BrowserGradCppCutePreprocessorPolicy.cpp",
  "BrowserGradCppCutePreprocessorPolicy.h",
  "BrowserGradCppCuteRuntime.cpp",
  "BrowserGradCppCuteRuntime.h",
  "BrowserGradCppCuteSha256.cpp",
  "BrowserGradCppCuteSha256.h",
  "CMakeLists.txt",
] as const;

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
let runtimeJsonLimits: {
  readonly maxDocumentByteLength: number;
  readonly maxNestingDepth: number;
  readonly maxNodeCount: number;
  readonly maxCumulativeStringByteLength: number;
  readonly maxArrayElementCount: number;
  readonly maxObjectPropertyCount: number;
};
let allocatorInterception: {
  readonly exactEntrypoints: readonly string[];
  readonly forbiddenEntrypoints: readonly string[];
  readonly underlyingBypassEntrypoints: readonly string[];
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
  runtimeJsonLimits = body.inputFrame.decodeLimits;
  allocatorInterception = body.allocatorMetricsRecord.accounting.interception;
});

async function extractorSource(path: typeof expectedSourcePaths[number]): Promise<string> {
  return readFile(join(extractorRoot, path), "utf8");
}

describe("BrowserGrad-owned Clang-WASM extractor source", () => {
  it("matches the exact split source closure and source-set identity in the build lock", async () => {
    expect(sourceFiles.map((file) => file.path)).toEqual(expectedSourcePaths);
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

  it("builds every owned translation unit into the Emscripten factory target", async () => {
    const cmake = await extractorSource("CMakeLists.txt");
    expect(cmake).toContain("add_llvm_executable(browsergrad-cpp-cute-extractor");
    for (const path of expectedSourcePaths.filter((path) => path !== "CMakeLists.txt")) {
      expect(cmake).toContain(`  ${path}\n`);
    }
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

  it("keeps the top-level translation unit as exact ABI-1.1 composition only", async () => {
    const source = await extractorSource("BrowserGradCppCuteExtractor.cpp");
    expect(source).toContain('#include "BrowserGradCppCuteArtifactV3.h"');
    expect(source).toContain('#include "BrowserGradCppCuteRuntime.h"');
    expect(source).not.toMatch(/clang\/|llvm\/|ImportedVfs|LayoutTrace|g_runtime|std::malloc/u);
    expect([...source.matchAll(/BG_CPP_CUTE_EXPORT[\s\S]*?\b(bg_cpp_cute_[a-z_]+)\(/gu)]
      .map((match) => match[1])).toEqual([
        "bg_cpp_cute_abi_version",
        "bg_cpp_cute_alloc",
        "bg_cpp_cute_allocator_metrics_pointer",
        "bg_cpp_cute_compile",
        "bg_cpp_cute_free",
        "bg_cpp_cute_reset",
        "bg_cpp_cute_result_length",
        "bg_cpp_cute_result_pointer",
        "bg_cpp_cute_status",
      ]);
  });

  it("owns frame validation and the bounded module-owned result lifecycle", async () => {
    const header = await extractorSource("BrowserGradCppCuteRuntime.h");
    const source = await extractorSource("BrowserGradCppCuteRuntime.cpp");
    expect(source).toContain("kRuntimeAbiVersion = 0x0001'0001U");
    expect(source).toContain("validate_frame_envelope");
    expect(source).toContain("RuntimeState g_runtime");
    expect(source).toContain("ArtifactV3ResultSink::allocate");
    expect(source).toContain("RuntimePhase::kCompiling");
    expect(source).toContain("RuntimePhase::kArtifactReady");
    expect(source).toContain("result.blocker.has_value()");
    expect(source).toContain("ranges_overlap");
    expect(source).toMatch(/runtime_result_length\(\)[\s\S]*?g_runtime\.result_byte_length/u);
    expect(source).toMatch(/runtime_result_pointer\(\)[\s\S]*?g_runtime\.result_wire_pointer/u);
    expect(header).toContain("class ArtifactV3ResultSink final");
    expect(header).toContain("kAbiMaximumByteLength");
    expect(header).toContain("8U * 1024U * 1024U");
    expect(header).toContain("bind_invocation_maximum_byte_length");
    expect(header).toContain("ArtifactV3ResultSink(const ArtifactV3ResultSink&) = delete");
    expect(source).not.toMatch(/clang\/|llvm\/|bg_vfs_/u);
  });

  it("keeps canonical JSON and SHA-256 deterministic, bounded, and ambient-free", async () => {
    const canonicalHeader = await extractorSource("BrowserGradCppCuteCanonicalJson.h");
    const canonicalSource = await extractorSource("BrowserGradCppCuteCanonicalJson.cpp");
    const shaHeader = await extractorSource("BrowserGradCppCuteSha256.h");
    const shaSource = await extractorSource("BrowserGradCppCuteSha256.cpp");
    const primitives = [canonicalHeader, canonicalSource, shaHeader, shaSource].join("\n");

    expect(canonicalHeader).toContain("struct CanonicalJsonLimits");
    expect(canonicalHeader).toContain("validate_canonical_json");
    for (const [name, value] of [
      ["kRuntimeV1MaxDocumentByteLength", runtimeJsonLimits.maxDocumentByteLength],
      ["kRuntimeV1MaxNestingDepth", runtimeJsonLimits.maxNestingDepth],
      ["kRuntimeV1MaxNodeCount", runtimeJsonLimits.maxNodeCount],
      ["kRuntimeV1MaxStringByteLength", runtimeJsonLimits.maxCumulativeStringByteLength],
      ["kRuntimeV1MaxArrayElementCount", runtimeJsonLimits.maxArrayElementCount],
      ["kRuntimeV1MaxObjectPropertyCount", runtimeJsonLimits.maxObjectPropertyCount],
    ] as const) {
      expect(canonicalHeader).toContain(`${name} = ${value}U`);
    }
    expect(canonicalSource).toContain("kMaximumSafeInteger");
    expect(canonicalSource).toContain("compare_utf16");
    expect(canonicalSource).toContain("depth > limits_.max_depth");
    expect(shaHeader).toContain("class Sha256 final");
    expect(shaHeader).toContain("Sha256(const Sha256&) = delete");
    expect(shaSource).toContain("kRoundConstants");
    expect(shaSource).toContain("kMaximumMessageByteLength");
    expect(primitives).not.toMatch(
      /clang\/|llvm\/|std::(?:filesystem|locale|random_device|thread)|\b(?:new|delete|malloc|calloc|realloc|free|fopen|getenv|setlocale|time|clock|fetch)\s*\(/u,
    );
  });

  it("owns exact requested-byte allocator metrics without recursive allocation", async () => {
    const header = await extractorSource("BrowserGradCppCuteMetrics.h");
    const source = await extractorSource("BrowserGradCppCuteMetrics.cpp");
    const allOtherSource = (await Promise.all(expectedSourcePaths
      .filter((path) => path !== "BrowserGradCppCuteMetrics.cpp")
      .filter((path) => path.endsWith(".cpp") || path.endsWith(".h") || path.endsWith(".inc"))
      .map(extractorSource))).join("\n");
    expect(header).toContain("struct alignas(8) AllocatorMetricsRecordV1");
    expect(header).toContain("sizeof(AllocatorMetricsRecordV1) == 72U");
    for (const offset of [
      "magic) == 0U", "version) == 8U", "byte_length) == 12U",
      "current_live_global_requested_byte_length) == 16U",
      "peak_live_global_requested_byte_length) == 24U",
      "cumulative_global_allocated_requested_byte_length) == 32U",
      "cumulative_global_freed_requested_byte_length) == 40U",
      "successful_allocation_count) == 48U", "free_count) == 56U",
      "failed_allocation_count) == 64U",
    ]) {
      expect(header).toContain(offset);
    }
    expect(source).toContain("emscripten_builtin_malloc");
    expect(source).toContain("emscripten_builtin_calloc");
    expect(source).toContain("emscripten_builtin_realloc");
    expect(source).toContain("emscripten_builtin_free");
    expect(source).toContain("emscripten_builtin_memalign");
    expect(source).toContain("kMaximumRecordStart");
    expect(source).toContain("g_metrics_healthy");
    expect(source).toContain("g_allocator_hook_active");
    expect(source).toContain("ensure_allocation_table_capacity");
    expect(source).toContain("erase_allocation");
    const overrides = [...source.matchAll(
      /BG_CPP_CUTE_ALLOCATOR_OVERRIDE\s+(?:void|int|std::size_t|void\*)\s*\**\s*([_a-z][_a-z0-9]*)\s*\(/gu,
    )].map((match) => match[1]);
    expect(new Set(overrides)).toEqual(new Set(allocatorInterception.exactEntrypoints));
    expect(overrides).toHaveLength(allocatorInterception.exactEntrypoints.length);
    for (const bypass of allocatorInterception.underlyingBypassEntrypoints) {
      expect(source).toContain(`${bypass}(`);
    }
    expect(source).not.toMatch(/std::(?:unordered_)?map|std::vector|new\s/u);
    expect(allOtherSource).not.toMatch(/emscripten_builtin_|\bdl(?:malloc|calloc|realloc|free|memalign)\b/u);
    for (const forbidden of allocatorInterception.forbiddenEntrypoints) {
      expect(overrides).not.toContain(forbidden);
    }
  });

  it("declares and consumes only the exact imported-VFS application surface", async () => {
    const source = await extractorSource("BrowserGradCppCuteImportedVfs.cpp");
    expect([...source.matchAll(/BG_CPP_CUTE_VFS_IMPORT\("([^"]+)"\)/gu)]
      .map((match) => match[1])).toEqual(applicationImportNames);
    expect(source).toContain('__attribute__((import_module("browsergrad_vfs_v1")))');
    for (const name of applicationImportNames) {
      expect([...source.matchAll(new RegExp(`\\b${name}\\(`, "gu"))].length)
        .toBeGreaterThanOrEqual(2);
    }
    expect(source).toContain(`kVfsMaximumPathByteLength = ${runtimeVfsLimits.maxPathByteLength}U`);
    expect(source).toContain(`kVfsMaximumDirectoryEntryCount = ${runtimeVfsLimits.maxIndexedNodes}U`);
    expect(source).toContain(`kVfsMaximumLiveHandleCount = ${runtimeVfsLimits.maxLiveFileHandles}U`);
    expect(source).toContain(
      `kVfsMaximumReadableFileByteLength = ${runtimeVfsLimits.maxAggregateLiveOpenByteLength}ULL`,
    );
    expect(source).toContain("class ImportedVfsFileSystem final");
    expect(source).toContain("class ImportedVfsFile final");
    expect(source).toContain("class ImportedVfsDirectoryIterator final");
    expect(source).toContain("class ImportedVfsOpenGuard final");
    expect(source).toContain("kVfsReadChunkByteLength = 64U * 1024U");
    expect(source).toContain("valid_canonical_path");
    expect(source).toContain("valid_utf8");
    expect(source).toContain("!utf8_byte_less(previous_name_, name)");
    expect(source).toContain("std::error_code load_current()");
    expect(source).not.toContain("std::vector<llvm::vfs::directory_entry>");
    expect(source).toContain("handle_budget_->terminal_error");
    expect(source).toContain("std::errc::operation_canceled");
    expect(source).toContain("std::errc::protocol_error");
    expect(source).not.toMatch(/bg_vfs_(?:cancel|abort|terminate)/u);
  });

  it("keeps Clang instrumentation on the imported VFS and nowhere on a physical filesystem", async () => {
    const action = await extractorSource("BrowserGradCppCuteClangAction.cpp");
    const allSource = (await Promise.all(expectedSourcePaths
      .filter((path) => path.endsWith(".cpp") || path.endsWith(".h") || path.endsWith(".inc"))
      .map(extractorSource))).join("\n");
    expect(action).toContain('#include "BrowserGradCppCuteImportedVfs.h"');
    expect(action).toContain("class LayoutTraceVisitor final");
    expect(action).toContain("class LayoutTraceAction final");
    expect(action).toContain("clang::tooling::ToolInvocation invocation");
    expect(action).toContain("clang::FileManager files(file_system_options, imported_closed_vfs())");
    expect(allSource).not.toMatch(
      /getRealFileSystem|createPhysicalFileSystem|std::ifstream|std::filesystem|\bfopen\(|\bfetch\(/u,
    );
  });

  it("keeps artifact-v3 as a wired non-authoritative placeholder", async () => {
    const artifact = await extractorSource("BrowserGradCppCuteArtifactV3.cpp");
    expect(artifact).toContain("ReviewOnlyBlocker::kCudaDualPassUnavailable");
    expect(artifact).toContain("WireCompileStatus::kInternalError");
    expect(artifact).not.toContain("run_layout_trace_for_review");
  });
});

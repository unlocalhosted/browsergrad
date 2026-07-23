import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";

vi.mock("./cpp_cute_browser_header_source_extraction.mjs", () => ({
  requireCppCuteBrowserHeaderSourceExtractionAuthority(extraction: FixtureExtraction) {
    if (extraction.fixtureAuthority !== true) throw new Error("forged extraction");
  },
  cppCuteBrowserExtractedHeaderSourceFiles(
    extraction: FixtureExtraction,
    sourceId: string,
    includeRootId: string,
  ) {
    return extraction.fixtureFiles.get(key(sourceId, includeRootId));
  },
  async copyCppCuteBrowserExtractedHeaderSourceFile(
    extraction: FixtureExtraction,
    sourceId: string,
    includeRootId: string,
    relativePath: string,
  ) {
    const bytes = extraction.fixtureContents.get(key(sourceId, includeRootId, relativePath));
    if (bytes === undefined) throw new Error("missing fixture bytes");
    return new Uint8Array(bytes);
  },
  async copyCppCuteBrowserExtractedHeaderSupplementalFile(
    extraction: FixtureExtraction,
    sourceId: string,
    supplementalFileId: string,
  ) {
    const bytes = extraction.fixtureContents.get(
      key(sourceId, "supplemental", supplementalFileId),
    );
    if (bytes === undefined) throw new Error("missing supplemental fixture bytes");
    return new Uint8Array(bytes);
  },
}));

import {
  CppCuteBrowserHeaderPackInventoryError,
  copyCppCuteBrowserHeaderPackInventorySourceFile,
  inventoryCppCuteBrowserExtractedHeaderSources,
} from "./cpp_cute_browser_header_pack_inventory.mjs";
import {
  materializeCppCuteBrowserLibcxxConfigSite,
  materializeCppCuteBrowserLibcxxModuleMap,
} from "./cpp_cute_browser_libcxx_config_site.mjs";

describe("extracted header-source inventory", () => {
  it("deduplicates identical overlays and retains opaque reread authority", async () => {
    const extraction = await fixtureExtraction();
    const inventory = await inventoryCppCuteBrowserExtractedHeaderSources(extraction as never);

    expect(inventory.authority).toBe("exact-extraction-source-inventory-only");
    expect(inventory.headerSourceExtractionId).toBe(extraction.extractionId);
    const templateBytes = Buffer.from(configSiteTemplate(), "utf8");
    const configuredBytes =
      materializeCppCuteBrowserLibcxxConfigSite(templateBytes).bytes;
    const moduleTemplateBytes = Buffer.from(moduleMapTemplate(), "utf8");
    const configuredModuleBytes =
      materializeCppCuteBrowserLibcxxModuleMap(moduleTemplateBytes).bytes;
    const assertionHandlerBytes = Buffer.from("// assertion handler\n", "utf8");
    expect(inventory.totals).toEqual({
      packCount: 5,
      sourceCount: 9,
      fileCount: 12,
      fileContentByteLength: String(
        91 +
        templateBytes.byteLength +
        configuredBytes.byteLength +
        moduleTemplateBytes.byteLength +
        configuredModuleBytes.byteLength +
        assertionHandlerBytes.byteLength,
      ),
    });
    expect(inventory.packs.find(({ includeRootId }) => includeRootId === "cuda"))
      .toMatchObject({ fileCount: 2, fileContentByteLength: "26" });
    expect(inventory.packs.find(({ includeRootId }) => includeRootId === "clang-resource"))
      .toMatchObject({ fileCount: 1, files: [expect.objectContaining({ virtualPath: "stddef.h" })] });
    expect(inventory.claims.generatedClangResourceHeadersComplete).toBe(true);
    expect(inventory.claims.configuredLibcxxHeaderComplete).toBe(true);
    expect(Buffer.from(await copyCppCuteBrowserHeaderPackInventorySourceFile(
      inventory,
      "cxx-stdlib",
      "__config_site",
    )).toString("utf8")).toContain("#define _LIBCPP_HAS_THREADS 1");
    expect(Buffer.from(await copyCppCuteBrowserHeaderPackInventorySourceFile(
      inventory,
      "cxx-stdlib",
      "module.modulemap",
    )).toString("utf8")).toContain('textual header "__config_site"');
    expect(Buffer.from(await copyCppCuteBrowserHeaderPackInventorySourceFile(
      inventory,
      "cxx-stdlib",
      "__assertion_handler",
    )).toString("utf8")).toBe("// assertion handler\n");
    expect(Buffer.from(await copyCppCuteBrowserHeaderPackInventorySourceFile(
      inventory,
      "cuda",
      "cuda.h",
    )).toString("utf8")).toBe("cuda-header\n");
    expect(Buffer.from(await copyCppCuteBrowserHeaderPackInventorySourceFile(
      inventory,
      "cuda",
      "curand_mtgp32_kernel.h",
    )).toString("utf8")).toBe("curand-header\n");

    extraction.fixtureContents.set(
      key("cuda-cccl-linux-x86-64", "cuda", "cuda.h"),
      Buffer.from("changed", "utf8"),
    );
    await expect(copyCppCuteBrowserHeaderPackInventorySourceFile(inventory, "cuda", "cuda.h"))
      .rejects.toThrow(CppCuteBrowserHeaderPackInventoryError);
  });

  it("rejects conflicting exact-source overlays", async () => {
    const extraction = await fixtureExtraction();
    setFile(
      extraction,
      "cuda-cudart-linux-x86-64",
      "cuda",
      "cuda.h",
      "different-cuda\n",
    );

    await expect(inventoryCppCuteBrowserExtractedHeaderSources(extraction as never))
      .rejects.toSatisfy((error: unknown) =>
        error instanceof CppCuteBrowserHeaderPackInventoryError &&
        error.message.includes("conflicting extracted overlay"));
  });
});

interface FixtureExtraction {
  fixtureAuthority: true;
  extractionId: string;
  buildInputLockId: string;
  buildInputLockResourceSha256: string;
  archives: Array<{
    sourceId: string;
    licenseComponentId: string;
    selections: Array<{
      includeRootId: string;
      virtualPrefix: "";
      intendedAsset: string;
      licenseComponentIds: readonly string[];
      configuredResourceOutput?: {
        upstreamBuildManifest: { virtualPath: string; sha256: string; byteLength: string };
        buildStageId: string;
        llvmTargetsToBuild: string;
        clangEnableHlsl: string;
        generatedVirtualPaths: readonly string[];
        omittedSourceVirtualPaths: readonly string[];
      };
    }>;
    supplementalFiles: Array<{
      supplementalFileId: string;
      includeRootId: string;
      virtualPath: string;
      intendedAsset: string;
      licenseComponentIds: readonly string[];
      sha256: string;
      byteLength: string;
    }>;
  }>;
  totals: { selectedSubtreeCount: number };
  fixtureFiles: Map<string, Array<{
    relativePath: string;
    contentSha256: string;
    byteLength: string;
  }>>;
  fixtureContents: Map<string, Uint8Array>;
}

async function fixtureExtraction(): Promise<FixtureExtraction> {
  const lock = await decodeCppCuteBrowserBuildInputLock(cppCuteBrowserBuildInputLockResourceBytes());
  const extraction: FixtureExtraction = {
    fixtureAuthority: true,
    extractionId: "bg.cpp.browser-header-source-extraction.sha256.fixture",
    buildInputLockId: lock.lockId,
    buildInputLockResourceSha256: lock.resourceSha256,
    archives: [
      source("cuda-cccl-linux-x86-64", "cuda-toolkit-12.6.3-headers", ["cuda"]),
      source("cuda-cudart-linux-x86-64", "cuda-toolkit-12.6.3-headers", ["cuda"]),
      source("cuda-libcurand-linux-x86-64", "cuda-toolkit-12.6.3-headers", ["cuda"]),
      source("cuda-nvcc-linux-x86-64", "cuda-toolkit-12.6.3-headers", ["cuda"]),
      source("cutlass", "cutlass", ["cutlass"]),
      source("llvm-project", "clang-and-libcxx", ["clang-resource", "cxx-stdlib"]),
      source("ubuntu-noble-libc6-dev-amd64-cross", "linux-sysroot", ["linux-sysroot"]),
      source("ubuntu-noble-linux-libc-dev-amd64-cross", "linux-sysroot", ["linux-sysroot"]),
    ],
    totals: { selectedSubtreeCount: 9 },
    fixtureFiles: new Map(),
    fixtureContents: new Map(),
  };
  for (const sourceId of [
    "cuda-cccl-linux-x86-64",
    "cuda-cudart-linux-x86-64",
    "cuda-nvcc-linux-x86-64",
  ]) {
    setFile(extraction, sourceId, "cuda", "cuda.h", "cuda-header\n");
  }
  setFile(
    extraction,
    "cuda-libcurand-linux-x86-64",
    "cuda",
    "curand_mtgp32_kernel.h",
    "curand-header\n",
  );
  setFile(extraction, "cutlass", "cutlass", "cute/tensor.hpp", "cute-header\n");
  setFile(extraction, "llvm-project", "clang-resource", "CMakeLists.txt", "fixture-cmake\n");
  setFile(extraction, "llvm-project", "clang-resource", "stddef.h", "clang-header\n");
  setFile(extraction, "llvm-project", "cxx-stdlib", "vector", "libcxx-header\n");
  setFile(
    extraction,
    "llvm-project",
    "cxx-stdlib",
    "__config_site.in",
    configSiteTemplate(),
  );
  setFile(
    extraction,
    "llvm-project",
    "cxx-stdlib",
    "module.modulemap.in",
    moduleMapTemplate(),
  );
  const assertionHandlerBytes = Buffer.from("// assertion handler\n", "utf8");
  const llvm = extraction.archives.find(({ sourceId }) => sourceId === "llvm-project");
  if (llvm === undefined) throw new Error("missing LLVM fixture source");
  llvm.supplementalFiles.push({
    supplementalFileId: "libcxx-default-assertion-handler",
    includeRootId: "cxx-stdlib",
    virtualPath: "__assertion_handler",
    intendedAsset: "dependency-header-pack:cxx-stdlib",
    licenseComponentIds: ["libcxx"],
    sha256: createHash("sha256").update(assertionHandlerBytes).digest("hex"),
    byteLength: String(assertionHandlerBytes.byteLength),
  });
  extraction.fixtureContents.set(
    key("llvm-project", "supplemental", "libcxx-default-assertion-handler"),
    assertionHandlerBytes,
  );
  setFile(
    extraction,
    "ubuntu-noble-libc6-dev-amd64-cross",
    "linux-sysroot",
    "assert.h",
    "glibc-header\n",
  );
  setFile(
    extraction,
    "ubuntu-noble-linux-libc-dev-amd64-cross",
    "linux-sysroot",
    "linux/types.h",
    "linux-header\n",
  );
  return extraction;
}

function configSiteTemplate(): string {
  return [
    "#ifndef _LIBCPP___CONFIG_SITE",
    "#define _LIBCPP___CONFIG_SITE",
    "#cmakedefine _LIBCPP_ABI_VERSION @_LIBCPP_ABI_VERSION@",
    "#cmakedefine _LIBCPP_ABI_NAMESPACE @_LIBCPP_ABI_NAMESPACE@",
    "#cmakedefine01 _LIBCPP_ABI_FORCE_ITANIUM",
    "#cmakedefine01 _LIBCPP_ABI_FORCE_MICROSOFT",
    "#cmakedefine01 _LIBCPP_HAS_THREADS",
    "#cmakedefine01 _LIBCPP_HAS_MONOTONIC_CLOCK",
    "#cmakedefine01 _LIBCPP_HAS_TERMINAL",
    "#cmakedefine01 _LIBCPP_HAS_MUSL_LIBC",
    "#cmakedefine01 _LIBCPP_HAS_THREAD_API_PTHREAD",
    "#cmakedefine01 _LIBCPP_HAS_THREAD_API_EXTERNAL",
    "#cmakedefine01 _LIBCPP_HAS_THREAD_API_WIN32",
    "#define _LIBCPP_HAS_THREAD_API_C11 0 // FIXME: Is this guarding dead code?",
    "#cmakedefine _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS",
    "#cmakedefine01 _LIBCPP_HAS_VENDOR_AVAILABILITY_ANNOTATIONS",
    "#cmakedefine _LIBCPP_NO_VCRUNTIME",
    "#cmakedefine _LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION @_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION@",
    "#cmakedefine01 _LIBCPP_HAS_FILESYSTEM",
    "#cmakedefine01 _LIBCPP_HAS_RANDOM_DEVICE",
    "#cmakedefine01 _LIBCPP_HAS_LOCALIZATION",
    "#cmakedefine01 _LIBCPP_HAS_UNICODE",
    "#cmakedefine01 _LIBCPP_HAS_WIDE_CHARACTERS",
    "#cmakedefine01 _LIBCPP_HAS_TIME_ZONE_DATABASE",
    "#cmakedefine01 _LIBCPP_INSTRUMENTED_WITH_ASAN",
    "#cmakedefine _LIBCPP_PSTL_BACKEND_SERIAL",
    "#cmakedefine _LIBCPP_PSTL_BACKEND_STD_THREAD",
    "#cmakedefine _LIBCPP_PSTL_BACKEND_LIBDISPATCH",
    "#cmakedefine _LIBCPP_HARDENING_MODE_DEFAULT @_LIBCPP_HARDENING_MODE_DEFAULT@",
    "#cmakedefine _LIBCPP_ASSERTION_SEMANTIC_DEFAULT @_LIBCPP_ASSERTION_SEMANTIC_DEFAULT@",
    "#cmakedefine01 _LIBCPP_LIBC_PICOLIBC",
    "#cmakedefine01 _LIBCPP_LIBC_NEWLIB",
    "@_LIBCPP_ABI_DEFINES@",
    "@_LIBCPP_EXTRA_SITE_DEFINES@",
    "#endif // _LIBCPP___CONFIG_SITE",
    "",
  ].join("\n");
}

function moduleMapTemplate(): string {
  return [
    "// module fixture",
    "module std_config [system] {",
    "  @LIBCXX_CONFIG_SITE_MODULE_ENTRY@ // generated via CMake",
    '  textual header "__config"',
    "}",
    "",
  ].join("\n");
}

function source(sourceId: string, licenseComponentId: string, includeRootIds: string[]) {
  return {
    sourceId,
    licenseComponentId,
    supplementalFiles: [],
    selections: includeRootIds.map((includeRootId) => ({
      includeRootId,
      virtualPrefix: "" as const,
      ...selectionPolicy(includeRootId),
    })),
  };
}

function selectionPolicy(includeRootId: string) {
  const policies: Record<string, {
    intendedAsset: string;
    licenseComponentIds: readonly string[];
    configuredResourceOutput?: ReturnType<typeof configuredResourceOutput>;
  }> = {
    "clang-resource": {
      intendedAsset: "compiler-resource-pack",
      licenseComponentIds: ["clang"],
      configuredResourceOutput: configuredResourceOutput("fixture-cmake\n"),
    },
    cuda: {
      intendedAsset: "dependency-header-pack:cuda",
      licenseComponentIds: ["cuda-toolkit-12.6.3-headers"],
    },
    cutlass: {
      intendedAsset: "dependency-header-pack:cutlass",
      licenseComponentIds: ["cutlass"],
    },
    "cxx-stdlib": {
      intendedAsset: "dependency-header-pack:cxx-stdlib",
      licenseComponentIds: ["libcxx"],
    },
    "linux-sysroot": {
      intendedAsset: "dependency-header-pack:linux-sysroot",
      licenseComponentIds: ["linux-sysroot"],
    },
  };
  const policy = policies[includeRootId];
  if (policy === undefined) throw new Error(`missing fixture policy ${includeRootId}`);
  return policy;
}

function configuredResourceOutput(value: string) {
  const bytes = Buffer.from(value, "utf8");
  return {
    upstreamBuildManifest: {
      virtualPath: "CMakeLists.txt",
      sha256: createHash("sha256").update(bytes).digest("hex"),
      byteLength: String(bytes.byteLength),
    },
    buildStageId: "clang-extractor-wasm",
    llvmTargetsToBuild: "WebAssembly",
    clangEnableHlsl: "OFF",
    generatedVirtualPaths: [] as const,
    omittedSourceVirtualPaths: ["CMakeLists.txt"] as const,
  };
}

function setFile(
  extraction: FixtureExtraction,
  sourceId: string,
  includeRootId: string,
  relativePath: string,
  value: string,
): void {
  const bytes = Buffer.from(value, "utf8");
  const files = extraction.fixtureFiles.get(key(sourceId, includeRootId)) ?? [];
  const evidence = {
    relativePath,
    contentSha256: createHash("sha256").update(bytes).digest("hex"),
    byteLength: String(bytes.byteLength),
  };
  const existingIndex = files.findIndex((file) => file.relativePath === relativePath);
  if (existingIndex === -1) files.push(evidence);
  else files[existingIndex] = evidence;
  extraction.fixtureFiles.set(key(sourceId, includeRootId), files);
  extraction.fixtureContents.set(key(sourceId, includeRootId, relativePath), bytes);
}

function key(...parts: string[]): string {
  return parts.join("\0");
}

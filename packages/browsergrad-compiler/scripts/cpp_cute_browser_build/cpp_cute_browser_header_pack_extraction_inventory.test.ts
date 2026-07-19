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
}));

import {
  CppCuteBrowserHeaderPackInventoryError,
  copyCppCuteBrowserHeaderPackInventorySourceFile,
  inventoryCppCuteBrowserExtractedHeaderSources,
} from "./cpp_cute_browser_header_pack_inventory.mjs";

describe("extracted header-source inventory", () => {
  it("deduplicates identical overlays and retains opaque reread authority", async () => {
    const extraction = await fixtureExtraction();
    const inventory = await inventoryCppCuteBrowserExtractedHeaderSources(extraction as never);

    expect(inventory.authority).toBe("exact-extraction-source-inventory-only");
    expect(inventory.headerSourceExtractionId).toBe(extraction.extractionId);
    expect(inventory.totals).toEqual({
      packCount: 5,
      sourceCount: 8,
      fileCount: 6,
      fileContentByteLength: "77",
    });
    expect(inventory.packs.find(({ includeRootId }) => includeRootId === "cuda"))
      .toMatchObject({ fileCount: 1, fileContentByteLength: "12" });
    expect(Buffer.from(await copyCppCuteBrowserHeaderPackInventorySourceFile(
      inventory,
      "cuda",
      "cuda.h",
    )).toString("utf8")).toBe("cuda-header\n");

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
      source("cuda-nvcc-linux-x86-64", "cuda-toolkit-12.6.3-headers", ["cuda"]),
      source("cutlass", "cutlass", ["cutlass"]),
      source("llvm-project", "clang-and-libcxx", ["clang-resource", "cxx-stdlib"]),
      source("ubuntu-noble-libc6-dev-amd64-cross", "linux-sysroot", ["linux-sysroot"]),
      source("ubuntu-noble-linux-libc-dev-amd64-cross", "linux-sysroot", ["linux-sysroot"]),
    ],
    totals: { selectedSubtreeCount: 8 },
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
  setFile(extraction, "cutlass", "cutlass", "cute/tensor.hpp", "cute-header\n");
  setFile(extraction, "llvm-project", "clang-resource", "stddef.h", "clang-header\n");
  setFile(extraction, "llvm-project", "cxx-stdlib", "vector", "libcxx-header\n");
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

function source(sourceId: string, licenseComponentId: string, includeRootIds: string[]) {
  return {
    sourceId,
    licenseComponentId,
    selections: includeRootIds.map((includeRootId) => ({
      includeRootId,
      virtualPrefix: "" as const,
      ...selectionPolicy(includeRootId),
    })),
  };
}

function selectionPolicy(includeRootId: string) {
  const policies: Record<string, { intendedAsset: string; licenseComponentIds: readonly string[] }> = {
    "clang-resource": {
      intendedAsset: "compiler-resource-pack",
      licenseComponentIds: ["clang"],
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

function setFile(
  extraction: FixtureExtraction,
  sourceId: string,
  includeRootId: string,
  relativePath: string,
  value: string,
): void {
  const bytes = Buffer.from(value, "utf8");
  extraction.fixtureFiles.set(key(sourceId, includeRootId), [{
    relativePath,
    contentSha256: createHash("sha256").update(bytes).digest("hex"),
    byteLength: String(bytes.byteLength),
  }]);
  extraction.fixtureContents.set(key(sourceId, includeRootId, relativePath), bytes);
}

function key(...parts: string[]): string {
  return parts.join("\0");
}

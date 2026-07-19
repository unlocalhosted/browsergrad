import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  CppCuteBrowserCudaRedistributionIndexError,
  inspectCppCuteBrowserCudaRedistributionIndexBytes,
  requireCppCuteBrowserCudaRedistributionIndexAuthority,
} from "./cpp_cute_browser_cuda_redistribution_index.mjs";

const SOURCE = Object.freeze({
  sourceId: "cuda-component-linux-x86-64",
  componentKey: "cuda_component",
  version: "12.6.1",
  relativePath: "cuda_component/linux-x86_64/cuda-component.tar.xz",
  sha256: "a".repeat(64),
  byteLength: "1234",
});

describe("CUDA redistribution index admission", () => {
  it("projects exact selected component and license metadata without minting current-plan authority", () => {
    const bytes = indexBytes();
    const inspected = inspectCppCuteBrowserCudaRedistributionIndexBytes({
      bytes,
      expected: expected(bytes),
    });

    expect(inspected).toMatchObject({
      authority: "caller-expected-cuda-redistribution-index-inspection-only",
      releaseLabel: "12.6.3",
      releaseProduct: "cuda",
      releaseDate: "2024-11-20",
      components: [{
        sourceId: SOURCE.sourceId,
        componentKey: SOURCE.componentKey,
        license: "CUDA Toolkit",
        licensePath: "cuda_component/LICENSE.txt",
        archiveSha256: SOURCE.sha256,
        archiveByteLength: SOURCE.byteLength,
      }],
      claims: {
        exactIndexBytesVerified: true,
        selectedComponentMetadataVerified: true,
        exactCurrentHeaderSourcePlanBound: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        releaseReady: false,
      },
    });
    expect(() => requireCppCuteBrowserCudaRedistributionIndexAuthority(inspected))
      .toThrow(CppCuteBrowserCudaRedistributionIndexError);
  });

  it("fails closed on byte, component, license, and accessor substitution", () => {
    const bytes = indexBytes();
    const changed = bytes.slice();
    changed[0] = (changed[0] ?? 0) ^ 1;
    expect(() => inspectCppCuteBrowserCudaRedistributionIndexBytes({
      bytes: changed,
      expected: expected(bytes),
    })).toThrow(CppCuteBrowserCudaRedistributionIndexError);

    const wrongLicense = indexBytes({ license: "unreviewed" });
    expect(() => inspectCppCuteBrowserCudaRedistributionIndexBytes({
      bytes: wrongLicense,
      expected: expected(wrongLicense),
    })).toThrow(CppCuteBrowserCudaRedistributionIndexError);

    const wrongArchive = indexBytes({ sha256: "b".repeat(64) });
    expect(() => inspectCppCuteBrowserCudaRedistributionIndexBytes({
      bytes: wrongArchive,
      expected: expected(wrongArchive),
    })).toThrow(CppCuteBrowserCudaRedistributionIndexError);

    expect(() => inspectCppCuteBrowserCudaRedistributionIndexBytes(Object.defineProperty(
      { expected: expected(bytes) },
      "bytes",
      { enumerable: true, get: () => bytes },
    ) as { bytes: Uint8Array; expected: ReturnType<typeof expected> }))
      .toThrow(CppCuteBrowserCudaRedistributionIndexError);
  });
});

function indexBytes(overrides: Readonly<{ license?: string; sha256?: string }> = {}): Uint8Array {
  return new TextEncoder().encode(JSON.stringify({
    release_date: "2024-11-20",
    release_label: "12.6.3",
    release_product: "cuda",
    cuda_component: {
      name: "CUDA Component",
      license: overrides.license ?? "CUDA Toolkit",
      license_path: "cuda_component/LICENSE.txt",
      version: SOURCE.version,
      "linux-x86_64": {
        relative_path: SOURCE.relativePath,
        sha256: overrides.sha256 ?? SOURCE.sha256,
        size: SOURCE.byteLength,
      },
    },
  }));
}

function expected(bytes: Uint8Array) {
  return {
    url: "https://example.invalid/redist/redistrib_12.6.3.json",
    releaseLabel: "12.6.3",
    sha256: createHash("sha256").update(bytes).digest("hex"),
    byteLength: String(bytes.byteLength),
    components: [SOURCE],
  } as const;
}

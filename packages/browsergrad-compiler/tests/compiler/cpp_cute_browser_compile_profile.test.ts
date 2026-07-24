import { describe, expect, it } from "vitest";

import {
  createCppCuteBrowserCompileProfileInput,
  deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256,
} from "../../src/cpp_cute_browser_compile_profile.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
} from "../../src/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
} from "../../src/cpp_cute_diagnostic_normalization.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteBrowserFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
} from "../../src/cpp_cute_semantic_adapter_manifest.js";

describe("package-owned browser C++/CuTe compile profile", () => {
  it("constructs the closed exact production shape without fixture identities", async () => {
    const sourceRootManifestSha256 =
      await deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256();
    expect(sourceRootManifestSha256).toBe(
      "6076ac6ed221c1ce33a656d14113c1099c60bd6781ae65928cdb85ed55ab9c91",
    );
    const input = createCppCuteBrowserCompileProfileInput({
      assetSetSha256: "1".repeat(64),
      buildProvenanceLockSha256: "2".repeat(64),
      extractorWasmSha256: "3".repeat(64),
      runtimeAbiManifestSha256:
        CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      semanticAdapterManifestSha256:
        CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
      sourceRootManifestSha256,
      workerModuleSha256: "5".repeat(64),
      workerModuleByteLength: 584_660,
      headerContentSets: {
        clangResource: "6".repeat(64),
        cuda: "7".repeat(64),
        cutlass: "8".repeat(64),
        cxxStdlib: "9".repeat(64),
        linuxSysroot: "a".repeat(64),
      },
    });

    expect(input.language.diagnostics.normalizationManifestSha256).toBe(
      CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
    );
    expect(input.toolchain.compiler).toMatchObject({
      version: "22.1.8",
      buildId: "llvmorg-22.1.8",
      binarySha256: "3".repeat(64),
      resourceDirectorySha256: "6".repeat(64),
    });
    expect(input.toolchain.dependencies).toEqual([
      expect.objectContaining({
        dependencyId: "cuda",
        version: "12.6.3",
        revision: "12.6.3",
        headerSetSha256: "7".repeat(64),
      }),
      expect.objectContaining({
        dependencyId: "cutlass",
        version: "3.7.0",
        revision: "b78588d1630aa6643bf021613717bafb705df4ef",
        headerSetSha256: "8".repeat(64),
      }),
      expect.objectContaining({
        dependencyId: "cxx-stdlib",
        version: "22.1.8",
        revision: "llvmorg-22.1.8",
        headerSetSha256: "9".repeat(64),
      }),
      expect.objectContaining({
        dependencyId: "linux-sysroot",
        version: "ubuntu-24.04",
        revision: "ubuntu-24.04-amd64",
        headerSetSha256: "a".repeat(64),
      }),
    ]);
    expect(input.virtualFileSystem.includeRoots[0]).toMatchObject({
      includeRootId: "workspace-source",
      manifestSha256: sourceRootManifestSha256,
      owner: { kind: "source" },
    });
    expect(input.deployment).toMatchObject({
      assetSetSha256: "1".repeat(64),
      buildProvenanceLockSha256: "2".repeat(64),
      extractor: {
        binarySha256: "3".repeat(64),
        semanticAdapterManifestSha256:
          CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
      },
      worker: {
        moduleSha256: "5".repeat(64),
        moduleByteLength: 584_660,
      },
      compilerRuntime: {
        runtimeAbiManifestSha256:
          CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      },
    });

    const prepared = await prepareCppCuteFrontendProfile(input);
    expect(
      unwrapPreparedCppCuteBrowserFrontendProfile(prepared).profile,
    ).toEqual(input);
  });
});

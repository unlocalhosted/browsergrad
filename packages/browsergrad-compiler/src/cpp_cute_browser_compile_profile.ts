import {
  hashCanonicalJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID,
  CPP_CUTE_FRONTEND_WARNING_BASELINE,
  CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID,
} from "./cpp_cute_frontend_compiler_policy.js";
import {
  type CppCuteFrontendBrowserDeploymentProfile,
  type CppCuteFrontendProfileV2,
} from "./cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
} from "./cpp_cute_diagnostic_normalization.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE,
} from "./resources/cpp_cute_browser_runtime_abi_v1.js";

export interface CppCuteBrowserCompileProfileInput {
  readonly assetSetSha256: string;
  readonly buildProvenanceLockSha256: string;
  readonly extractorWasmSha256: string;
  readonly runtimeAbiManifestSha256: string;
  readonly semanticAdapterManifestSha256: string;
  readonly sourceRootManifestSha256: string;
  readonly workerModuleSha256: string;
  readonly workerModuleByteLength: number;
  readonly headerContentSets: Readonly<{
    readonly clangResource: string;
    readonly cuda: string;
    readonly cutlass: string;
    readonly cxxStdlib: string;
    readonly linuxSysroot: string;
  }>;
}

export type CppCuteBrowserCompileProfile =
  CppCuteFrontendProfileV2 & {
    readonly deployment: CppCuteFrontendBrowserDeploymentProfile;
  };

/**
 * The real compile lane supplies no project headers. Main-source bytes remain
 * request-bound and must not change the compiler profile or header asset-set
 * identity from one user program to another.
 */
export function deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256():
Promise<string> {
  return hashCanonicalJson({
    domain:
      "browsergrad.compiler.cpp-cute.empty-source-include-root-manifest.v1",
    includeRootId: "workspace-source",
    virtualPath: "/workspace/src",
    files: [],
  });
}

/**
 * Constructs the one closed browser-local profile input used by the real
 * compile lane. Every mutable output identity is caller-supplied from an
 * independently verified package or asset boundary. This raw input grants no
 * authority until prepareCppCuteFrontendProfile admits it.
 */
export function createCppCuteBrowserCompileProfileInput(
  input: CppCuteBrowserCompileProfileInput,
): CppCuteBrowserCompileProfile {
  return {
    schema: "browsergrad.compiler.cpp-cute.frontend-profile",
    version: { major: 2, minor: 6 },
    profileId: "browsergrad.compiler.cpp-cute.browser-clang@1",
    deployment: {
      mode: "browser-local",
      contractId: "browsergrad.compiler.cpp-cute.browser-worker@1",
      assetSetSha256: input.assetSetSha256,
      buildProvenanceLockSha256: input.buildProvenanceLockSha256,
      extractor: {
        id: "browsergrad-tools/cpp-cute-clang-wasm",
        version: "0.1.0",
        buildId: "browsergrad-cpp-cute-clang-wasm-0.1.0",
        binarySha256: input.extractorWasmSha256,
        semanticAdapterManifestSha256:
          input.semanticAdapterManifestSha256,
      },
      worker: {
        protocolId: "browsergrad.compiler.cpp-cute.browser-worker@1",
        buildId: "browsergrad-cpp-cute-browser-worker-0.1.0",
        moduleSha256: input.workerModuleSha256,
        moduleByteLength: input.workerModuleByteLength,
        moduleFormat: "self-contained-es-module",
        construction: "host-verified-blob-url",
        isolation: "dedicated-worker",
        threading: "single-thread",
        cancellation: "terminate-worker",
        network: "forbidden",
        assetDelivery: "host-verified-transfer",
      },
      compilerRuntime: {
        runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1",
        runtimeAbiManifestSha256: input.runtimeAbiManifestSha256,
        wasmAddressBits: 32,
        requiredWasmFeatures: [
          ...CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.wasm.requiredFeatures,
        ],
        moduleHandoff: "host-verified-module-or-bytes",
        workerSideFetch: "forbidden",
        memory: {
          sharing: "unshared",
          ownership: "worker",
          initialPages: 4_096,
          maximumPages: 16_384,
          stackByteLength: 16 * 1024 * 1024,
          maxCompilerWorkingByteLength: 512 * 1024 * 1024,
        },
        virtualFileSystem: {
          storage: "host-backed-lazy",
          maxRetainedHostPackByteLength: 512 * 1024 * 1024,
          maxAggregateLiveOpenByteLength: 384 * 1024 * 1024,
          maxIndexedNodes: 65_536,
          maxIndexLogicalByteLength: 32 * 1024 * 1024,
        },
      },
      assetLimits: {
        maxAssets: 32,
        maxAssetCompressedByteLength: 256 * 1024 * 1024,
        maxAssetUnpackedByteLength: 512 * 1024 * 1024,
        maxAssetFileContentByteLength: 256 * 1024 * 1024,
        maxTotalCompressedByteLength: 512 * 1024 * 1024,
        maxTotalUnpackedByteLength: 1024 * 1024 * 1024,
        maxTotalFileContentByteLength: 512 * 1024 * 1024,
      },
    },
    language: {
      cxxStandard: "c++17",
      cudaCompatibility: "12.6",
      preprocessing: {
        temporalMacros: {
          policyId: CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID,
          mode: "reject",
        },
      },
      diagnostics: {
        warningRegistryId: CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID,
        baseline: CPP_CUTE_FRONTEND_WARNING_BASELINE,
        normalizationManifestSha256:
          CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
      },
      semanticPasses: [
        {
          ordinal: 0,
          passId: "cuda-device-sema",
          domain: "device",
          role: "semantic-extraction",
          invocationMode: "cuda-device-only",
          targetTriple: "nvptx64-nvidia-cuda",
          auxiliaryTargetTriple: "x86_64-unknown-linux-gnu",
          deviceArchitecture: "sm_80",
        },
        {
          ordinal: 1,
          passId: "cuda-host-sema",
          domain: "host",
          role: "validation",
          invocationMode: "cuda-host-only",
          targetTriple: "x86_64-unknown-linux-gnu",
          auxiliaryTargetTriple: "nvptx64-nvidia-cuda",
          deviceArchitecture: "sm_80",
        },
      ],
      options: [
        { kind: "define", name: "CUTE_SM80_ENABLED", value: "1" },
        {
          kind: "forced-include",
          includeRootId: "clang-resource",
          virtualPath:
            "/toolchain/clang/lib/clang/22/include/__clang_cuda_runtime_wrapper.h",
        },
        { kind: "frontend-option", id: "syntax-only", value: null },
        { kind: "frontend-option", id: "error-limit", value: "100000" },
      ],
    },
    target: {
      host: {
        triple: "x86_64-unknown-linux-gnu",
        endianness: "little",
        pointerBits: 64,
        dataLayout:
          "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
      },
      device: {
        triple: "nvptx64-nvidia-cuda",
        architecture: "sm_80",
        endianness: "little",
        pointerBits: 64,
        dataLayout: "e-i64:64-i128:128-v16:16-v32:32-n16:32:64",
      },
    },
    toolchain: {
      compiler: {
        id: "clang",
        version: "22.1.8",
        buildId: "llvmorg-22.1.8",
        binarySha256: input.extractorWasmSha256,
        resourceDirectoryVirtualPath: "/toolchain/clang/lib/clang/22",
        resourceDirectorySha256: input.headerContentSets.clangResource,
      },
      dependencies: [
        {
          dependencyId: "cuda",
          kind: "cuda-toolkit",
          version: "12.6.3",
          revision: "12.6.3",
          headerSetSha256: input.headerContentSets.cuda,
        },
        {
          dependencyId: "cutlass",
          kind: "cutlass",
          version: "3.7.0",
          revision: "b78588d1630aa6643bf021613717bafb705df4ef",
          headerSetSha256: input.headerContentSets.cutlass,
        },
        {
          dependencyId: "cxx-stdlib",
          kind: "cxx-standard-library",
          version: "22.1.8",
          revision: "llvmorg-22.1.8",
          headerSetSha256: input.headerContentSets.cxxStdlib,
        },
        {
          dependencyId: "linux-sysroot",
          kind: "linux-sysroot",
          version: "ubuntu-24.04",
          revision: "ubuntu-24.04-amd64",
          headerSetSha256: input.headerContentSets.linuxSysroot,
        },
      ],
    },
    virtualFileSystem: {
      sourceRoots: ["/workspace/src"],
      includeRoots: [
        {
          includeRootId: "workspace-source",
          mode: "quote",
          virtualPath: "/workspace/src",
          manifestSha256: input.sourceRootManifestSha256,
          owner: { kind: "source" },
        },
        {
          includeRootId: "cuda",
          mode: "system",
          virtualPath: "/toolchain/cuda/include",
          manifestSha256: input.headerContentSets.cuda,
          owner: { kind: "dependency", dependencyId: "cuda" },
        },
        {
          includeRootId: "cutlass",
          mode: "system",
          virtualPath: "/toolchain/cutlass/include",
          manifestSha256: input.headerContentSets.cutlass,
          owner: { kind: "dependency", dependencyId: "cutlass" },
        },
        {
          includeRootId: "cxx-stdlib",
          mode: "system",
          virtualPath: "/toolchain/cxx/include/c++/v1",
          manifestSha256: input.headerContentSets.cxxStdlib,
          owner: { kind: "dependency", dependencyId: "cxx-stdlib" },
        },
        {
          includeRootId: "clang-resource",
          mode: "system",
          virtualPath: "/toolchain/clang/lib/clang/22/include",
          manifestSha256: input.headerContentSets.clangResource,
          owner: { kind: "compiler-resource-directory" },
        },
        {
          includeRootId: "linux-sysroot",
          mode: "system",
          virtualPath: "/toolchain/sysroot/usr/include",
          manifestSha256: input.headerContentSets.linuxSysroot,
          owner: { kind: "dependency", dependencyId: "linux-sysroot" },
        },
      ],
    },
    compatibility: {
      supportedSourceFeatures: [
        "cuda:language@1",
        "cute:layout-algebra@1",
        "cxx:templates@1",
      ],
      unsupportedSourceFeatures: ["cxx:temporal-macros@1"],
      unsupportedIntrinsicFamilies: ["nvidia:tma@1", "nvidia:wgmma@1"],
    },
    extractionLimits: {
      maxSourceFiles: 8,
      maxSourceBytes: 1_048_576,
      maxHeaderFiles: 20_000,
      maxHeaderBytes: 268_435_456,
      maxIncludeDepth: 256,
      maxMacroExpansions: 1_000_000,
      maxPreprocessedTokens: 10_000_000,
      maxAstNodes: 5_000_000,
      maxConstexprSteps: 10_000_000,
      maxTemplateInstantiations: 1_000_000,
      maxTemplateDepth: 1_024,
      maxDeclarations: 1_000_000,
      maxTypes: 1_000_000,
      maxConstants: 1_000_000,
      maxLayouts: 100_000,
      maxTensors: 100_000,
      maxOperations: 1_000_000,
      maxTargetIntrinsics: 100_000,
      maxDiagnostics: 100_000,
      maxOutputBytes: 33_554_432,
      maxWallTimeMs: 120_000,
      maxCpuTimeMs: 120_000,
      maxMemoryBytes: 1024 * 1024 * 1024,
      maxProcesses: 1,
    },
  };
}

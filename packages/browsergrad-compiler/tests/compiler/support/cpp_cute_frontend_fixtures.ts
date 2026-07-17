import {
  CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE,
  type CppCuteFrontendAotDeploymentProfile,
  type CppCuteFrontendBrowserDeploymentProfile,
  type CppCuteFrontendIncludeRoot,
  type CppCuteFrontendProfileV2,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
} from "../../../src/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
} from "../../../src/cpp_cute_semantic_adapter_manifest.js";
import {
  CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID,
  CPP_CUTE_FRONTEND_WARNING_BASELINE,
  CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID,
} from "../../../src/cpp_cute_frontend_compiler_policy.js";
import { CPP_CUTE_AOT_SANDBOX_POLICY_SHA256 } from "../../../src/cpp_cute_aot_policy.js";
import { deriveCppCuteFrontendArtifactId } from "../../../src/cpp_cute_frontend_artifact.js";
import {
  computeCppCuteInputHashes,
  computeCppCuteSemanticPassInputClosureHash,
  computeCppCuteSharedSurfaceHash,
  deriveCppCuteSourceEntityId,
} from "../../../src/cpp_cute_frontend_verify.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../../../src/cpp_cute_frontend_types.js";
import type {
  CppCuteFrontendArtifactV3,
  CppCuteFrontendPayloadV3,
} from "../../../src/cpp_cute_frontend_types.js";

export const CPP_CUTE_FIXTURE_HEADER_SET_HASH = "2".repeat(64);
export const CPP_CUTE_FIXTURE_COMPILER_HASH = "3".repeat(64);
export const CPP_CUTE_FIXTURE_COMPILER_RESOURCE_HASH = "b".repeat(64);
export const CPP_CUTE_FIXTURE_SEMANTIC_ADAPTER_HASH =
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256;
export const CPP_CUTE_FIXTURE_RUNTIME_ABI_MANIFEST_SHA256 =
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256;
export const CPP_CUTE_FIXTURE_CONTAINER_DIGEST = `sha256:${"4".repeat(64)}`;
export const CPP_CUTE_FIXTURE_CONTAINER_CONFIG_DIGEST = `sha256:${"1".repeat(64)}`;
export const CPP_CUTE_FIXTURE_EXECUTION_ENVIRONMENT_HASH = "0".repeat(64);
export const CPP_CUTE_FIXTURE_CUDA_HEADER_HASH = "5".repeat(64);
export const CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH = "6".repeat(64);
export const CPP_CUTE_FIXTURE_CUTLASS_REVISION = "7".repeat(40);
export const CPP_CUTE_FIXTURE_CXX_HEADER_HASH = "c".repeat(64);
export const CPP_CUTE_FIXTURE_LINUX_SYSROOT_HEADER_HASH = "f".repeat(64);
export const CPP_CUTE_FIXTURE_SOURCE_INCLUDE_HASH = "1".repeat(64);
export const CPP_CUTE_FIXTURE_BUILDER_ID =
  "https://github.com/unlocalhosted/browsergrad/.github/workflows/cpp-cute-aot.yml";
export const CPP_CUTE_FIXTURE_SOURCE_REPOSITORY = "https://github.com/unlocalhosted/browsergrad";
export const CPP_CUTE_FIXTURE_SOURCE_REVISION = Object.freeze({
  algorithm: "git-sha1" as const,
  value: "8".repeat(40),
});

export interface CppCuteProfileFixtureOptions {
  readonly trustStoreSha256?: string;
  readonly sourceRoots?: readonly string[];
  readonly includeRoots?: readonly CppCuteFrontendIncludeRoot[];
  readonly sandboxPolicySha256?: string;
  readonly executionEnvironmentManifestSha256?: string;
  readonly containerManifestDigest?: string;
  readonly containerConfigDigest?: string;
}

export interface CppCuteBrowserProfileFixtureOptions extends CppCuteProfileFixtureOptions {
  readonly assetSetSha256?: string;
  readonly buildProvenanceLockSha256?: string;
  readonly runtimeAbiManifestSha256?: string;
}

export function createCppCuteProfileInput(
  options: CppCuteProfileFixtureOptions = {},
): CppCuteFrontendProfileV2 & { readonly deployment: CppCuteFrontendAotDeploymentProfile } {
  const sourceRoots = options.sourceRoots ?? ["/workspace/src"];
  return {
    schema: "browsergrad.compiler.cpp-cute.frontend-profile",
    version: { major: 2, minor: 5 },
    profileId: "browsergrad.compiler.cpp-cute.layout-tracer@2",
    deployment: {
      mode: "ahead-of-time",
      contractId: "browsergrad.compiler.cpp-cute.aot@1",
      sandboxPolicySha256: options.sandboxPolicySha256 ?? CPP_CUTE_AOT_SANDBOX_POLICY_SHA256,
      executionEnvironmentManifestSha256:
        options.executionEnvironmentManifestSha256 ?? CPP_CUTE_FIXTURE_EXECUTION_ENVIRONMENT_HASH,
      extractor: {
        id: "browsergrad-tools/cpp-cute-frontend",
        version: "0.1.0",
        buildId: "browsergrad-cpp-cute-extractor-0.1.0",
        binarySha256: "a".repeat(64),
        semanticAdapterManifestSha256: CPP_CUTE_FIXTURE_SEMANTIC_ADAPTER_HASH,
      },
      runner: {
        id: "browsergrad-tools/cpp-cute-aot-runner",
        version: "0.1.0",
        binarySha256: "9".repeat(64),
      },
      container: {
        runtime: "docker",
        repository: "ghcr.io/unlocalhosted/browsergrad-cpp-cute-aot",
        platform: "linux/amd64",
        manifestDigest: options.containerManifestDigest ?? CPP_CUTE_FIXTURE_CONTAINER_DIGEST,
        configDigest: options.containerConfigDigest ?? CPP_CUTE_FIXTURE_CONTAINER_CONFIG_DIGEST,
      },
      provenance: {
        kind: "external-attestation",
        predicateType: CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE,
        trustStoreSha256: options.trustStoreSha256 ?? "d".repeat(64),
        builderIds: [CPP_CUTE_FIXTURE_BUILDER_ID],
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
          virtualPath: "/toolchain/clang/lib/clang/22/include/__clang_cuda_runtime_wrapper.h",
        },
        { kind: "frontend-option", id: "syntax-only", value: null },
      ],
    },
    target: {
      host: {
        triple: "x86_64-unknown-linux-gnu",
        endianness: "little",
        pointerBits: 64,
        dataLayout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
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
        binarySha256: CPP_CUTE_FIXTURE_COMPILER_HASH,
        resourceDirectorySha256: CPP_CUTE_FIXTURE_COMPILER_RESOURCE_HASH,
      },
      dependencies: [
        {
          dependencyId: "cuda",
          kind: "cuda-toolkit",
          version: "12.6.3",
          revision: "12.6.3",
          headerSetSha256: CPP_CUTE_FIXTURE_CUDA_HEADER_HASH,
        },
        {
          dependencyId: "cutlass",
          kind: "cutlass",
          version: "3.7.0",
          revision: CPP_CUTE_FIXTURE_CUTLASS_REVISION,
          headerSetSha256: CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH,
        },
        {
          dependencyId: "cxx-stdlib",
          kind: "cxx-standard-library",
          version: "18.1.8",
          revision: "llvmorg-18.1.8",
          headerSetSha256: CPP_CUTE_FIXTURE_CXX_HEADER_HASH,
        },
        {
          dependencyId: "linux-sysroot",
          kind: "linux-sysroot",
          version: "ubuntu-24.04",
          revision: "ubuntu-24.04-amd64",
          headerSetSha256: CPP_CUTE_FIXTURE_LINUX_SYSROOT_HEADER_HASH,
        },
      ],
    },
    virtualFileSystem: {
      sourceRoots,
      includeRoots: options.includeRoots ?? createDefaultCppCuteIncludeRoots(sourceRoots[0] ?? "/workspace/src"),
    },
    compatibility: {
      supportedSourceFeatures: ["cuda:language@1", "cute:layout-algebra@1", "cxx:templates@1"],
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
      maxOutputBytes: 8_388_608,
      maxWallTimeMs: 120_000,
      maxCpuTimeMs: 120_000,
      maxMemoryBytes: 4_294_967_296,
      maxProcesses: 32,
    },
  };
}

export function createCppCuteBrowserProfileInput(
  options: CppCuteBrowserProfileFixtureOptions = {},
): CppCuteFrontendProfileV2 & { readonly deployment: CppCuteFrontendBrowserDeploymentProfile } {
  const aot = createCppCuteProfileInput(options);
  const browserExtractorSha256 = "a".repeat(64);
  return {
    ...aot,
    profileId: "browsergrad.compiler.cpp-cute.browser-clang@1",
    toolchain: {
      ...aot.toolchain,
      compiler: {
        ...aot.toolchain.compiler,
        binarySha256: browserExtractorSha256,
      },
    },
    deployment: {
      mode: "browser-local",
      contractId: "browsergrad.compiler.cpp-cute.browser-worker@1",
      assetSetSha256: options.assetSetSha256 ?? "8".repeat(64),
      buildProvenanceLockSha256: options.buildProvenanceLockSha256 ?? "7".repeat(64),
      extractor: {
        id: "browsergrad-tools/cpp-cute-clang-wasm",
        version: "0.1.0",
        buildId: "browsergrad-cpp-cute-clang-wasm-0.1.0",
        binarySha256: browserExtractorSha256,
        semanticAdapterManifestSha256: CPP_CUTE_FIXTURE_SEMANTIC_ADAPTER_HASH,
      },
      worker: {
        protocolId: "browsergrad.compiler.cpp-cute.browser-worker@1",
        buildId: "browsergrad-cpp-cute-browser-worker-0.1.0",
        moduleSha256: "9".repeat(64),
        moduleByteLength: 65_536,
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
        runtimeAbiManifestSha256:
          options.runtimeAbiManifestSha256 ?? CPP_CUTE_FIXTURE_RUNTIME_ABI_MANIFEST_SHA256,
        wasmAddressBits: 32,
        requiredWasmFeatures: [
          "bulk-memory",
          "mutable-globals",
          "nontrapping-fptoint",
          "sign-extension",
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
    extractionLimits: {
      ...aot.extractionLimits,
      maxMemoryBytes: 1024 * 1024 * 1024,
      maxProcesses: 1,
    },
  };
}

function createDefaultCppCuteIncludeRoots(sourceRoot: string): readonly CppCuteFrontendIncludeRoot[] {
  return [
    {
      includeRootId: "workspace-source",
      mode: "quote",
      virtualPath: sourceRoot,
      manifestSha256: CPP_CUTE_FIXTURE_SOURCE_INCLUDE_HASH,
      owner: { kind: "source" },
    },
    {
      includeRootId: "clang-resource",
      mode: "system",
      virtualPath: "/toolchain/clang/lib/clang/22/include",
      manifestSha256: CPP_CUTE_FIXTURE_COMPILER_RESOURCE_HASH,
      owner: { kind: "compiler-resource-directory" },
    },
    {
      includeRootId: "cuda",
      mode: "system",
      virtualPath: "/toolchain/cuda/include",
      manifestSha256: CPP_CUTE_FIXTURE_CUDA_HEADER_HASH,
      owner: { kind: "dependency", dependencyId: "cuda" },
    },
    {
      includeRootId: "cutlass",
      mode: "system",
      virtualPath: "/toolchain/cutlass/include",
      manifestSha256: CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH,
      owner: { kind: "dependency", dependencyId: "cutlass" },
    },
    {
      includeRootId: "cxx-stdlib",
      mode: "system",
      virtualPath: "/toolchain/cxx/include/c++/v1",
      manifestSha256: CPP_CUTE_FIXTURE_CXX_HEADER_HASH,
      owner: { kind: "dependency", dependencyId: "cxx-stdlib" },
    },
    {
      includeRootId: "linux-sysroot",
      mode: "system",
      virtualPath: "/toolchain/sysroot/usr/include",
      manifestSha256: CPP_CUTE_FIXTURE_LINUX_SYSROOT_HEADER_HASH,
      owner: { kind: "dependency", dependencyId: "linux-sysroot" },
    },
  ];
}

export function cloneCppCuteProfileInput(
  options: CppCuteProfileFixtureOptions = {},
): Record<string, unknown> {
  return structuredClone(createCppCuteProfileInput(options)) as unknown as Record<string, unknown>;
}

export const CPP_CUTE_FIXTURE_COMPILATION_CONTRACT_HASH = "a".repeat(64);
export const CPP_CUTE_FIXTURE_MAIN_FILE_ID = stableId("file", "1");
export const CPP_CUTE_FIXTURE_COMPILER_HEADER_FILE_ID = stableId("file", "d");
export const CPP_CUTE_FIXTURE_HEADER_FILE_ID = stableId("file", "e");
export const CPP_CUTE_FIXTURE_SOURCE_INCLUDE_ROOT_ID = "workspace-source";
export const CPP_CUTE_FIXTURE_COMPILER_INCLUDE_ROOT_ID = "clang-resource";
export const CPP_CUTE_FIXTURE_CUDA_INCLUDE_ROOT_ID = "cuda";
export const CPP_CUTE_FIXTURE_CXX_INCLUDE_ROOT_ID = "cxx-stdlib";
export const CPP_CUTE_FIXTURE_LINUX_SYSROOT_INCLUDE_ROOT_ID = "linux-sysroot";
export const CPP_CUTE_FIXTURE_INCLUDE_ROOT_ID = "cutlass";
export const CPP_CUTE_FIXTURE_INCLUDE_EDGE_ID = stableId("include-edge", "0");
export const CPP_CUTE_FIXTURE_FORCED_INCLUDE_EDGE_ID = stableId("include-edge", "1");
export const CPP_CUTE_FIXTURE_SPAN_ID = stableId("span", "2");
export const CPP_CUTE_FIXTURE_INT_TYPE_ID = stableId("type", "3");
const CPP_CUTE_FIXTURE_INT_SOURCE_ENTITY_PLACEHOLDER = stableId("source-entity", "3");
const CPP_CUTE_FIXTURE_LAYOUT_SOURCE_ENTITY_PLACEHOLDER = stableId("source-entity", "7");
export const CPP_CUTE_FIXTURE_LAYOUT_TYPE_ID = stableId("type", "4");
export const CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID = stableId("declaration", "5");
export const CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID = stableId("declaration", "6");
export const CPP_CUTE_FIXTURE_VARIABLE_DECLARATION_ID = stableId("declaration", "7");
export const CPP_CUTE_FIXTURE_INSTANTIATION_ID = stableId("template-instantiation", "8");
export const CPP_CUTE_FIXTURE_LAYOUT_FACT_ID = stableId("fact", "9");
export const CPP_CUTE_FIXTURE_INTRINSIC_FACT_ID = stableId("fact", "a");
export const CPP_CUTE_FIXTURE_ENTRY_ID = stableId("entry", "b");
export const CPP_CUTE_FIXTURE_DIAGNOSTIC_ID = stableId("diagnostic", "c");
const ZERO_HASH = "0".repeat(64);

function stableId(kind: string, digit: string): string {
  return `bg.cpp.${kind}.sha256.${digit.repeat(64)}`;
}

function sourceOrigin(): { readonly kind: "source"; readonly spanId: string } {
  return { kind: "source", spanId: CPP_CUTE_FIXTURE_SPAN_ID };
}

function qualifiers(): { readonly const: boolean; readonly volatile: boolean; readonly restrict: boolean } {
  return { const: false, volatile: false, restrict: false };
}

export async function createCppCutePayloadInput(
  compilationContractHash = CPP_CUTE_FIXTURE_COMPILATION_CONTRACT_HASH,
): Promise<CppCuteFrontendPayloadV3> {
  const payload: CppCuteFrontendPayloadV3 = {
    compilationContractHash,
    inputs: {
      mainFileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID,
      includeRoots: [
        {
          includeRootId: CPP_CUTE_FIXTURE_SOURCE_INCLUDE_ROOT_ID,
          ordinal: 0,
          mode: "quote",
          virtualPath: "/src",
          manifestSha256: CPP_CUTE_FIXTURE_SOURCE_INCLUDE_HASH,
          owner: { kind: "source" },
        },
        {
          includeRootId: CPP_CUTE_FIXTURE_COMPILER_INCLUDE_ROOT_ID,
          ordinal: 1,
          mode: "system",
          virtualPath: "/toolchain/clang/lib/clang/22/include",
          manifestSha256: CPP_CUTE_FIXTURE_COMPILER_RESOURCE_HASH,
          owner: { kind: "compiler-resource-directory" },
        },
        {
          includeRootId: CPP_CUTE_FIXTURE_CUDA_INCLUDE_ROOT_ID,
          ordinal: 2,
          mode: "system",
          virtualPath: "/toolchain/cuda/include",
          manifestSha256: CPP_CUTE_FIXTURE_CUDA_HEADER_HASH,
          owner: { kind: "dependency", dependencyId: "cuda" },
        },
        {
          includeRootId: CPP_CUTE_FIXTURE_INCLUDE_ROOT_ID,
          ordinal: 3,
          mode: "system",
          virtualPath: "/toolchain/cutlass/include",
          manifestSha256: CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH,
          owner: { kind: "dependency", dependencyId: "cutlass" },
        },
        {
          includeRootId: CPP_CUTE_FIXTURE_CXX_INCLUDE_ROOT_ID,
          ordinal: 4,
          mode: "system",
          virtualPath: "/toolchain/cxx/include/c++/v1",
          manifestSha256: CPP_CUTE_FIXTURE_CXX_HEADER_HASH,
          owner: { kind: "dependency", dependencyId: "cxx-stdlib" },
        },
        {
          includeRootId: CPP_CUTE_FIXTURE_LINUX_SYSROOT_INCLUDE_ROOT_ID,
          ordinal: 5,
          mode: "system",
          virtualPath: "/toolchain/sysroot/usr/include",
          manifestSha256: CPP_CUTE_FIXTURE_LINUX_SYSROOT_HEADER_HASH,
          owner: { kind: "dependency", dependencyId: "linux-sysroot" },
        },
      ],
      files: [
        {
          fileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID,
          role: "main-source",
          virtualPath: "/src/layout.cu",
          contentSha256: "d".repeat(64),
          byteLength: "100" as never,
          owner: { kind: "source" },
          includeRootId: null,
        },
        {
          fileId: CPP_CUTE_FIXTURE_COMPILER_HEADER_FILE_ID,
          role: "compiler-header",
          virtualPath: "/toolchain/clang/lib/clang/22/include/__clang_cuda_runtime_wrapper.h",
          contentSha256: "c".repeat(64),
          byteLength: "300" as never,
          owner: { kind: "compiler-resource-directory" },
          includeRootId: CPP_CUTE_FIXTURE_COMPILER_INCLUDE_ROOT_ID,
        },
        {
          fileId: CPP_CUTE_FIXTURE_HEADER_FILE_ID,
          role: "dependency-header",
          virtualPath: "/toolchain/cutlass/include/cute/layout.hpp",
          contentSha256: "e".repeat(64),
          byteLength: "200" as never,
          owner: { kind: "dependency", dependencyId: "cutlass" },
          includeRootId: CPP_CUTE_FIXTURE_INCLUDE_ROOT_ID,
        },
      ],
      includeEdges: [
        {
          kind: "source-directive",
          includeEdgeId: CPP_CUTE_FIXTURE_INCLUDE_EDGE_ID,
          includingFileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID,
          directiveSpanId: CPP_CUTE_FIXTURE_SPAN_ID,
          spelling: "cute/layout.hpp",
          mode: "angle",
          resolution: {
            kind: "resolved",
            fileId: CPP_CUTE_FIXTURE_HEADER_FILE_ID,
            includeRootId: CPP_CUTE_FIXTURE_INCLUDE_ROOT_ID,
          },
        },
        {
          kind: "compiler-forced",
          includeEdgeId: CPP_CUTE_FIXTURE_FORCED_INCLUDE_EDGE_ID,
          fileId: CPP_CUTE_FIXTURE_COMPILER_HEADER_FILE_ID,
          includeRootId: CPP_CUTE_FIXTURE_COMPILER_INCLUDE_ROOT_ID,
          compilerOptionOrdinal: 1,
        },
      ],
      sourceSetSha256: ZERO_HASH,
      headerSetSha256: ZERO_HASH,
      closureSha256: ZERO_HASH,
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
        status: "succeeded",
        openedFileIds: [
          CPP_CUTE_FIXTURE_MAIN_FILE_ID,
          CPP_CUTE_FIXTURE_COMPILER_HEADER_FILE_ID,
          CPP_CUTE_FIXTURE_HEADER_FILE_ID,
        ].sort(),
        includeEdgeIds: [CPP_CUTE_FIXTURE_INCLUDE_EDGE_ID, CPP_CUTE_FIXTURE_FORCED_INCLUDE_EDGE_ID].sort(),
        observedInputClosureSha256: ZERO_HASH,
        sharedSurfaceSha256: ZERO_HASH,
        selectedSourceRootEntityIds: [CPP_CUTE_FIXTURE_LAYOUT_SOURCE_ENTITY_PLACEHOLDER],
        factIds: [CPP_CUTE_FIXTURE_LAYOUT_FACT_ID, CPP_CUTE_FIXTURE_INTRINSIC_FACT_ID],
        diagnosticIds: [CPP_CUTE_FIXTURE_DIAGNOSTIC_ID],
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
        status: "succeeded",
        openedFileIds: [
          CPP_CUTE_FIXTURE_MAIN_FILE_ID,
          CPP_CUTE_FIXTURE_COMPILER_HEADER_FILE_ID,
          CPP_CUTE_FIXTURE_HEADER_FILE_ID,
        ].sort(),
        includeEdgeIds: [CPP_CUTE_FIXTURE_INCLUDE_EDGE_ID, CPP_CUTE_FIXTURE_FORCED_INCLUDE_EDGE_ID].sort(),
        observedInputClosureSha256: ZERO_HASH,
        sharedSurfaceSha256: ZERO_HASH,
        selectedSourceRootEntityIds: [CPP_CUTE_FIXTURE_LAYOUT_SOURCE_ENTITY_PLACEHOLDER],
        factIds: [],
        diagnosticIds: [],
      },
    ],
    semanticGraphOwnerPassId: "cuda-device-sema",
    spans: [{
      spanId: CPP_CUTE_FIXTURE_SPAN_ID,
      spelling: {
        fileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID,
        startByte: "0" as never,
        endByte: "100" as never,
      },
      expansion: {
        fileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID,
        startByte: "0" as never,
        endByte: "100" as never,
      },
      macroExpansionId: null,
    }],
    macroExpansions: [],
    types: [
      {
        typeId: CPP_CUTE_FIXTURE_INT_TYPE_ID,
        kind: "builtin",
        canonicalName: "int",
        qualifiers: qualifiers(),
        origin: sourceOrigin(),
        builtin: "int",
      },
      {
        typeId: CPP_CUTE_FIXTURE_LAYOUT_TYPE_ID,
        kind: "template-specialization",
        canonicalName: "cute::Layout<cute::Shape<cute::Int<3>, cute::Int<2>>, cute::Stride<cute::Int<1>, cute::Int<3>>>",
        qualifiers: qualifiers(),
        origin: sourceOrigin(),
        templateDeclarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
        arguments: [],
      },
    ],
    constants: [],
    initializerExpressions: [],
    declarations: [
      {
        declarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
        kind: "template",
        canonicalUsr: "c:@N@cute@ST>1#T@Layout",
        canonicalName: "cute::Layout",
        lexicalParentId: null,
        semanticParentId: null,
        typeId: CPP_CUTE_FIXTURE_INT_TYPE_ID,
        targetTypeId: null,
        initializerExpressionId: null,
        origin: sourceOrigin(),
        identitySpanId: CPP_CUTE_FIXTURE_SPAN_ID,
        definitionKind: "external",
        linkage: "external",
        storageDuration: "none",
        memorySpace: "generic",
        mangledName: null,
        cudaAttributes: { host: true, device: true, global: false, forceInline: true },
      },
      {
        declarationId: CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID,
        kind: "record",
        canonicalUsr: "c:@N@cute@S@Layout>#I#I",
        canonicalName: "cute::Layout<cute::Shape<cute::Int<3>, cute::Int<2>>, cute::Stride<cute::Int<1>, cute::Int<3>>>",
        lexicalParentId: null,
        semanticParentId: null,
        typeId: CPP_CUTE_FIXTURE_LAYOUT_TYPE_ID,
        targetTypeId: null,
        initializerExpressionId: null,
        origin: sourceOrigin(),
        identitySpanId: CPP_CUTE_FIXTURE_SPAN_ID,
        definitionKind: "external",
        linkage: "external",
        storageDuration: "none",
        memorySpace: "host",
        mangledName: null,
        cudaAttributes: { host: true, device: true, global: false, forceInline: false },
      },
      {
        declarationId: CPP_CUTE_FIXTURE_VARIABLE_DECLARATION_ID,
        kind: "variable",
        canonicalUsr: "c:@layout",
        canonicalName: "layout",
        lexicalParentId: null,
        semanticParentId: null,
        typeId: CPP_CUTE_FIXTURE_LAYOUT_TYPE_ID,
        targetTypeId: null,
        initializerExpressionId: null,
        origin: sourceOrigin(),
        identitySpanId: CPP_CUTE_FIXTURE_SPAN_ID,
        definitionKind: "definition",
        linkage: "internal",
        storageDuration: "static",
        memorySpace: "generic",
        mangledName: "_ZL6layout",
        cudaAttributes: { host: true, device: true, global: false, forceInline: false },
      },
    ],
    templateInstantiations: [{
      instantiationId: CPP_CUTE_FIXTURE_INSTANTIATION_ID,
      templateDeclarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
      specializationDeclarationId: CPP_CUTE_FIXTURE_RECORD_DECLARATION_ID,
      arguments: [],
      pointOfInstantiationSpanId: CPP_CUTE_FIXTURE_SPAN_ID,
      parentInstantiationId: null,
      depth: 0,
    }],
    overloadResolutions: [],
    sourceEntities: [
      {
        sourceEntityId: CPP_CUTE_FIXTURE_INT_SOURCE_ENTITY_PLACEHOLDER,
        entityKind: "type",
        canonicalIdentity: "int",
        origin: sourceOrigin(),
        domains: ["device", "host"],
      },
      {
        sourceEntityId: CPP_CUTE_FIXTURE_LAYOUT_SOURCE_ENTITY_PLACEHOLDER,
        entityKind: "variable",
        canonicalIdentity: "c:@layout",
        origin: sourceOrigin(),
        domains: ["device", "host"],
      },
    ],
    sourceAbi: {
      types: [
        {
          domain: "device",
          shared: true,
          sourceTypeEntityId: CPP_CUTE_FIXTURE_INT_SOURCE_ENTITY_PLACEHOLDER,
          deviceTypeId: CPP_CUTE_FIXTURE_INT_TYPE_ID,
          sizeBits: "32" as never,
          alignmentBits: "32" as never,
          fields: [],
          bases: [],
        },
        {
          domain: "host",
          shared: true,
          sourceTypeEntityId: CPP_CUTE_FIXTURE_INT_SOURCE_ENTITY_PLACEHOLDER,
          deviceTypeId: null,
          sizeBits: "32" as never,
          alignmentBits: "32" as never,
          fields: [],
          bases: [],
        },
      ],
      functions: [],
    },
    functionBodies: [],
    facts: [
      {
        factId: CPP_CUTE_FIXTURE_LAYOUT_FACT_ID,
        kind: "affine-layout",
        origin: sourceOrigin(),
        resultDeclarationId: CPP_CUTE_FIXTURE_VARIABLE_DECLARATION_ID,
        shape: {
          kind: "tuple",
          elements: [
            { kind: "scalar", value: { kind: "integer", value: "3" as never } },
            { kind: "scalar", value: { kind: "integer", value: "2" as never } },
          ],
        },
        stride: {
          kind: "tuple",
          elements: [
            { kind: "scalar", value: { kind: "integer", value: "1" as never } },
            { kind: "scalar", value: { kind: "integer", value: "3" as never } },
          ],
        },
        rank: 2,
        leafRank: 2,
        size: { kind: "integer", value: "6" as never },
        cosize: { kind: "integer", value: "6" as never },
      },
      {
        factId: CPP_CUTE_FIXTURE_INTRINSIC_FACT_ID,
        kind: "target-intrinsic",
        origin: sourceOrigin(),
        familyId: "nvidia:wgmma@1",
        operation: { kind: "capability", capabilityId: "nvidia:wgmma@1" },
        operandExpressionIds: [],
        resultTypeId: null,
        effects: { readsMemory: false, writesMemory: false, synchronizes: true, convergent: true },
        availability: { kind: "recognized-unsupported", diagnosticId: CPP_CUTE_FIXTURE_DIAGNOSTIC_ID },
      },
    ],
    entries: [{
      entryId: CPP_CUTE_FIXTURE_ENTRY_ID,
      kind: "layout",
      layoutFactId: CPP_CUTE_FIXTURE_LAYOUT_FACT_ID,
      selectedRootDeclarationIds: [CPP_CUTE_FIXTURE_VARIABLE_DECLARATION_ID],
    }],
    diagnostics: [{
      diagnosticId: CPP_CUTE_FIXTURE_DIAGNOSTIC_ID,
      phase: "artifact-extraction",
      severity: "warning",
      code: "browsergrad.cpp-cute:recognized-unsupported-intrinsic",
      renderedMessage: "WGMMA preserved as a typed unsupported target capability.",
      location: { kind: "source", primarySpanId: CPP_CUTE_FIXTURE_SPAN_ID, related: [] },
      subject: { kind: "fact", factId: CPP_CUTE_FIXTURE_INTRINSIC_FACT_ID },
      parentDiagnosticId: null,
    }],
    outcome: { kind: "accepted", selectedEntryIds: [CPP_CUTE_FIXTURE_ENTRY_ID] },
    extraction: {
      compilationContractHash,
      inputClosureSha256: ZERO_HASH,
      appliedTransforms: [],
    },
  };
  const intEntity = payload.sourceEntities.find((entity) => entity.entityKind === "type");
  const layoutEntity = payload.sourceEntities.find((entity) => entity.entityKind === "variable");
  if (intEntity === undefined) throw new Error("fixture lost integer source identity");
  if (layoutEntity === undefined) throw new Error("fixture lost layout source identity");
  const intSourceEntityId = await deriveCppCuteSourceEntityId(payload, {
    entityKind: intEntity.entityKind,
    canonicalIdentity: intEntity.canonicalIdentity,
    origin: intEntity.origin,
    domains: intEntity.domains,
  });
  const layoutSourceEntityId = await deriveCppCuteSourceEntityId(payload, {
    entityKind: layoutEntity.entityKind,
    canonicalIdentity: layoutEntity.canonicalIdentity,
    origin: layoutEntity.origin,
    domains: layoutEntity.domains,
  });
  const sourceEntities = [
    { ...intEntity, sourceEntityId: intSourceEntityId },
    { ...layoutEntity, sourceEntityId: layoutSourceEntityId },
  ].sort((left, right) => left.sourceEntityId.localeCompare(right.sourceEntityId));
  const identityPayload: CppCuteFrontendPayloadV3 = {
    ...payload,
    semanticPasses: payload.semanticPasses.map((pass) => ({
      ...pass,
      selectedSourceRootEntityIds: [layoutSourceEntityId],
    })),
    sourceEntities,
    sourceAbi: {
      ...payload.sourceAbi,
      types: payload.sourceAbi.types.map((entry) => ({ ...entry, sourceTypeEntityId: intSourceEntityId })),
    },
  };
  const hashes = await computeCppCuteInputHashes(identityPayload);
  const boundPayload: CppCuteFrontendPayloadV3 = {
    ...identityPayload,
    inputs: {
      ...identityPayload.inputs,
      sourceSetSha256: hashes.sourceSetSha256,
      headerSetSha256: hashes.headerSetSha256,
      closureSha256: hashes.closureSha256,
    },
    extraction: { ...identityPayload.extraction, inputClosureSha256: hashes.closureSha256 },
  };
  const semanticPasses = await Promise.all(boundPayload.semanticPasses.map(async (pass, index) => ({
    ...pass,
    observedInputClosureSha256: await computeCppCuteSemanticPassInputClosureHash(boundPayload, index),
    sharedSurfaceSha256: await computeCppCuteSharedSurfaceHash(boundPayload, pass.domain),
  })));
  return { ...boundPayload, semanticPasses };
}

export async function createCppCuteArtifactInput(
  compilationContractHash = CPP_CUTE_FIXTURE_COMPILATION_CONTRACT_HASH,
): Promise<CppCuteFrontendArtifactV3> {
  const payload = await createCppCutePayloadInput(compilationContractHash);
  return {
    schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
    version: { major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR, minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR },
    producer: { id: "browsergrad-tools/cpp-cute-frontend", version: "0.1.0" },
    artifactId: await deriveCppCuteFrontendArtifactId(payload),
    payload,
    requiredExtensions: [],
  };
}

export async function cloneCppCuteArtifactInput(
  compilationContractHash = CPP_CUTE_FIXTURE_COMPILATION_CONTRACT_HASH,
): Promise<Record<string, unknown>> {
  return structuredClone(await createCppCuteArtifactInput(compilationContractHash)) as unknown as Record<string, unknown>;
}

export async function rebindCppCuteFixtureSourceEntityIds(payload: CppCuteFrontendPayloadV3): Promise<void> {
  const replacements = new Map<string, string>();
  const sourceEntities = await Promise.all(payload.sourceEntities.map(async (entity) => {
    const sourceEntityId = await deriveCppCuteSourceEntityId(payload, {
      entityKind: entity.entityKind,
      canonicalIdentity: entity.canonicalIdentity,
      origin: entity.origin,
      domains: entity.domains,
    });
    replacements.set(entity.sourceEntityId, sourceEntityId);
    return { ...entity, sourceEntityId };
  }));
  sourceEntities.sort((left, right) => left.sourceEntityId.localeCompare(right.sourceEntityId));
  const replace = (sourceEntityId: string): string => replacements.get(sourceEntityId) ?? sourceEntityId;
  const sourceAbi: CppCuteFrontendPayloadV3["sourceAbi"] = {
    types: payload.sourceAbi.types.map((entry) => ({
      ...entry,
      sourceTypeEntityId: replace(entry.sourceTypeEntityId),
      fields: entry.fields.map((field) => ({
        ...field,
        sourceEntityId: replace(field.sourceEntityId),
        sourceTypeEntityId: replace(field.sourceTypeEntityId),
      })),
      bases: entry.bases.map((base) => ({
        ...base,
        sourceTypeEntityId: replace(base.sourceTypeEntityId),
      })),
    })),
    functions: payload.sourceAbi.functions.map((entry) => ({
      ...entry,
      sourceEntityId: replace(entry.sourceEntityId),
      returnSourceTypeEntityId: replace(entry.returnSourceTypeEntityId),
      parameters: entry.parameters.map((parameter) => ({
        ...parameter,
        sourceEntityId: replace(parameter.sourceEntityId),
        sourceTypeEntityId: replace(parameter.sourceTypeEntityId),
      })),
    })),
  };
  (payload as { sourceEntities: CppCuteFrontendPayloadV3["sourceEntities"] }).sourceEntities = sourceEntities;
  (payload as { semanticPasses: CppCuteFrontendPayloadV3["semanticPasses"] }).semanticPasses =
    payload.semanticPasses.map((pass) => ({
      ...pass,
      selectedSourceRootEntityIds: pass.selectedSourceRootEntityIds.map(replace).sort(),
    }));
  (payload as { sourceAbi: CppCuteFrontendPayloadV3["sourceAbi"] }).sourceAbi = sourceAbi;
}

export function artifactCompatibleProfileOptions(
  trustStoreSha256: string,
): CppCuteProfileFixtureOptions {
  return {
    trustStoreSha256,
    sourceRoots: ["/src"],
  };
}

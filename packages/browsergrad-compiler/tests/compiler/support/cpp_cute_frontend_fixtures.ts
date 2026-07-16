import {
  CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE,
  type CppCuteFrontendIncludeRoot,
  type CppCuteFrontendProfileV1,
} from "../../../src/cpp_cute_frontend_profile.js";
import { CPP_CUTE_AOT_SANDBOX_POLICY_SHA256 } from "../../../src/cpp_cute_aot_policy.js";
import { deriveCppCuteFrontendArtifactId } from "../../../src/cpp_cute_frontend_artifact.js";
import { computeCppCuteInputHashes } from "../../../src/cpp_cute_frontend_verify.js";
import type {
  CppCuteFrontendArtifactV1,
  CppCuteFrontendPayloadV1,
} from "../../../src/cpp_cute_frontend_types.js";

export const CPP_CUTE_FIXTURE_HEADER_SET_HASH = "2".repeat(64);
export const CPP_CUTE_FIXTURE_COMPILER_HASH = "3".repeat(64);
export const CPP_CUTE_FIXTURE_CONTAINER_DIGEST = `sha256:${"4".repeat(64)}`;
export const CPP_CUTE_FIXTURE_CONTAINER_CONFIG_DIGEST = `sha256:${"1".repeat(64)}`;
export const CPP_CUTE_FIXTURE_EXECUTION_ENVIRONMENT_HASH = "0".repeat(64);
export const CPP_CUTE_FIXTURE_CUDA_HEADER_HASH = "5".repeat(64);
export const CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH = "6".repeat(64);
export const CPP_CUTE_FIXTURE_CUTLASS_REVISION = "7".repeat(40);
export const CPP_CUTE_FIXTURE_BUILDER_ID =
  "https://github.com/unlocalhosted/browsergrad/.github/workflows/cpp-cute-aot.yml";
export const CPP_CUTE_FIXTURE_SOURCE_REPOSITORY = "https://github.com/unlocalhosted/browsergrad";
export const CPP_CUTE_FIXTURE_SOURCE_REVISION = Object.freeze({
  algorithm: "git-sha1" as const,
  value: "8".repeat(40),
});

export interface CppCuteProfileFixtureOptions {
  readonly trustStoreSha256?: string;
  readonly expectedHeaderSetSha256?: string;
  readonly sourceRoots?: readonly string[];
  readonly includeRoots?: readonly CppCuteFrontendIncludeRoot[];
  readonly sandboxPolicySha256?: string;
  readonly executionEnvironmentManifestSha256?: string;
  readonly containerConfigDigest?: string;
}

export function createCppCuteProfileInput(
  options: CppCuteProfileFixtureOptions = {},
): CppCuteFrontendProfileV1 {
  return {
    schema: "browsergrad.compiler.cpp-cute.frontend-profile",
    version: { major: 1, minor: 0 },
    profileId: "browsergrad.compiler.cpp-cute.layout-tracer@1",
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
        manifestDigest: CPP_CUTE_FIXTURE_CONTAINER_DIGEST,
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
      options: [
        { kind: "define", name: "CUTE_SM80_ENABLED", value: "1" },
        { kind: "frontend-option", id: "cuda-host-only", value: null },
        { kind: "frontend-option", id: "syntax-only", value: null },
      ],
    },
    target: {
      hostTriple: "x86_64-unknown-linux-gnu",
      deviceArchitecture: "sm_80",
      endianness: "little",
      pointerBits: 64,
    },
    toolchain: {
      compiler: {
        id: "clang",
        version: "20.1.8",
        buildId: "llvmorg-20.1.8",
        binarySha256: CPP_CUTE_FIXTURE_COMPILER_HASH,
        resourceDirectorySha256: "b".repeat(64),
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
      ],
    },
    virtualFileSystem: {
      sourceRoots: options.sourceRoots ?? ["/workspace/src"],
      includeRoots: options.includeRoots ?? [
        {
          includeRootId: "cuda",
          mode: "system",
          virtualPath: "/toolchain/cuda/include",
          manifestSha256: CPP_CUTE_FIXTURE_CUDA_HEADER_HASH,
        },
        {
          includeRootId: "cutlass",
          mode: "system",
          virtualPath: "/toolchain/cutlass/include",
          manifestSha256: CPP_CUTE_FIXTURE_CUTLASS_HEADER_HASH,
        },
      ],
    },
    compatibility: {
      expectedHeaderSetSha256: options.expectedHeaderSetSha256 ?? CPP_CUTE_FIXTURE_HEADER_SET_HASH,
      supportedSourceFeatures: ["cuda:language@1", "cute:layout-algebra@1", "cxx:templates@1"],
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

export function cloneCppCuteProfileInput(
  options: CppCuteProfileFixtureOptions = {},
): Record<string, unknown> {
  return structuredClone(createCppCuteProfileInput(options)) as unknown as Record<string, unknown>;
}

export const CPP_CUTE_FIXTURE_PROFILE_HASH = "a".repeat(64);
export const CPP_CUTE_FIXTURE_MAIN_FILE_ID = stableId("file", "1");
export const CPP_CUTE_FIXTURE_HEADER_FILE_ID = stableId("file", "e");
export const CPP_CUTE_FIXTURE_INCLUDE_ROOT_ID = stableId("include-root", "f");
export const CPP_CUTE_FIXTURE_INCLUDE_EDGE_ID = stableId("include-edge", "0");
export const CPP_CUTE_FIXTURE_SPAN_ID = stableId("span", "2");
export const CPP_CUTE_FIXTURE_INT_TYPE_ID = stableId("type", "3");
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
  profileHash = CPP_CUTE_FIXTURE_PROFILE_HASH,
): Promise<CppCuteFrontendPayloadV1> {
  const payload: CppCuteFrontendPayloadV1 = {
    profileHash,
    inputs: {
      mainFileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID,
      includeRoots: [{
        includeRootId: CPP_CUTE_FIXTURE_INCLUDE_ROOT_ID,
        ordinal: 0,
        mode: "system",
        virtualPath: "/toolchain/cutlass/include",
        manifestSha256: "f".repeat(64),
      }],
      files: [
        {
          fileId: CPP_CUTE_FIXTURE_MAIN_FILE_ID,
          role: "main-source",
          virtualPath: "/src/layout.cu",
          contentSha256: "d".repeat(64),
          byteLength: "100" as never,
          profileDependency: "none",
        },
        {
          fileId: CPP_CUTE_FIXTURE_HEADER_FILE_ID,
          role: "cute-header",
          virtualPath: "/toolchain/cutlass/include/cute/layout.hpp",
          contentSha256: "e".repeat(64),
          byteLength: "200" as never,
          profileDependency: "cute",
        },
      ],
      includeEdges: [{
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
      }],
      sourceSetSha256: ZERO_HASH,
      headerSetSha256: ZERO_HASH,
      closureSha256: ZERO_HASH,
    },
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
        origin: sourceOrigin(),
        definitionKind: "external",
        linkage: "external",
        storageDuration: "none",
        memorySpace: "host",
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
        origin: sourceOrigin(),
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
        origin: sourceOrigin(),
        definitionKind: "definition",
        linkage: "internal",
        storageDuration: "static",
        memorySpace: "host",
        mangledName: "_ZL6layout",
        cudaAttributes: { host: true, device: false, global: false, forceInline: false },
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
    sourceAbi: { types: [], functions: [] },
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
      primarySpanId: CPP_CUTE_FIXTURE_SPAN_ID,
      subject: { kind: "fact", factId: CPP_CUTE_FIXTURE_INTRINSIC_FACT_ID },
      parentDiagnosticId: null,
      related: [],
    }],
    outcome: { kind: "accepted", selectedEntryIds: [CPP_CUTE_FIXTURE_ENTRY_ID] },
    extraction: {
      profileHash,
      inputClosureSha256: ZERO_HASH,
      appliedTransforms: [],
    },
  };
  const hashes = await computeCppCuteInputHashes(payload);
  return {
    ...payload,
    inputs: {
      ...payload.inputs,
      sourceSetSha256: hashes.sourceSetSha256,
      headerSetSha256: hashes.headerSetSha256,
      closureSha256: hashes.closureSha256,
    },
    extraction: { ...payload.extraction, inputClosureSha256: hashes.closureSha256 },
  };
}

export async function createCppCuteArtifactInput(
  profileHash = CPP_CUTE_FIXTURE_PROFILE_HASH,
): Promise<CppCuteFrontendArtifactV1> {
  const payload = await createCppCutePayloadInput(profileHash);
  return {
    schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
    version: { major: 1, minor: 0 },
    producer: { id: "browsergrad-tools/cpp-cute-frontend", version: "0.1.0" },
    artifactId: await deriveCppCuteFrontendArtifactId(payload),
    payload,
    requiredExtensions: [],
  };
}

export async function cloneCppCuteArtifactInput(
  profileHash = CPP_CUTE_FIXTURE_PROFILE_HASH,
): Promise<Record<string, unknown>> {
  return structuredClone(await createCppCuteArtifactInput(profileHash)) as unknown as Record<string, unknown>;
}

export function artifactCompatibleProfileOptions(
  headerSetSha256: string,
  trustStoreSha256: string,
): CppCuteProfileFixtureOptions {
  return {
    trustStoreSha256,
    expectedHeaderSetSha256: headerSetSha256,
    sourceRoots: ["/src"],
    includeRoots: [{
      includeRootId: "cutlass",
      mode: "system",
      virtualPath: "/toolchain/cutlass/include",
      manifestSha256: "f".repeat(64),
    }],
  };
}

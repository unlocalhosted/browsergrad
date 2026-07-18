import {
  assertJsonValue,
  canonicalizeJson,
  compareCanonicalStrings,
  deepFreezeJson,
  hashCanonicalJson,
  isJsonObject,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
} from "./cpp_cute_semantic_adapter_manifest.js";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
} from "./cpp_cute_diagnostic_normalization.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE,
} from "./resources/cpp_cute_browser_runtime_abi_v1.js";
import {
  CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID,
  CPP_CUTE_FRONTEND_WARNING_BASELINE,
  CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID,
  cppCuteFrontendWarningPolicyMapping,
  isCppCuteFrontendTemporalMacroName,
  isCppCuteFrontendReservedMacroName,
  type CppCuteFrontendWarningPolicyId,
} from "./cpp_cute_frontend_compiler_policy.js";
import {
  findCppCuteVirtualPathError,
} from "./cpp_cute_virtual_path.js";

export const CPP_CUTE_FRONTEND_PROFILE_SCHEMA = "browsergrad.compiler.cpp-cute.frontend-profile";
export const CPP_CUTE_FRONTEND_PROFILE_MAJOR = 2;
export const CPP_CUTE_FRONTEND_PROFILE_MINOR = 6;
export const CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_SCHEMA =
  "browsergrad.compiler.cpp-cute.compilation-contract";
export const CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_MAJOR = 1;
export const CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_MINOR = 2;
export const CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE =
  "https://browsergrad.dev/provenance/cpp-cute-aot/v3";

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const OCI_SHA256 = /^sha256:[0-9a-f]{64}$/u;
const GIT_COMMIT = /^[0-9a-f]{40}$/u;
const PROFILE_ID = /^browsergrad\.compiler\.cpp-cute\.[a-z0-9][a-z0-9._-]*@[1-9][0-9]*$/u;
const CAPABILITY_ID = /^[a-z][a-z0-9.-]*:[a-z][a-z0-9._-]*(?:@[1-9][0-9]*)?$/u;
const DEPENDENCY_ID = /^[a-z][a-z0-9._-]*$/u;
const MACRO_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/u;
const TARGET_ARCHITECTURE = /^sm_[1-9][0-9][a-z]?$/u;
const LLVM_DATA_LAYOUT = /^[A-Za-z0-9:_+.-]+$/u;
const OCI_REPOSITORY = /^[a-z0-9.-]+(?::[1-9][0-9]*)?(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$/u;
const PREPARED_PROFILES = new WeakMap<object, PreparedCppCuteFrontendProfileRecord>();

const EXTRACTION_LIMIT_KEYS = [
  "maxSourceFiles",
  "maxSourceBytes",
  "maxHeaderFiles",
  "maxHeaderBytes",
  "maxIncludeDepth",
  "maxMacroExpansions",
  "maxPreprocessedTokens",
  "maxAstNodes",
  "maxConstexprSteps",
  "maxTemplateInstantiations",
  "maxTemplateDepth",
  "maxDeclarations",
  "maxTypes",
  "maxConstants",
  "maxLayouts",
  "maxTensors",
  "maxOperations",
  "maxTargetIntrinsics",
  "maxDiagnostics",
  "maxOutputBytes",
  "maxWallTimeMs",
  "maxCpuTimeMs",
  "maxMemoryBytes",
  "maxProcesses",
] as const;

export const CPP_CUTE_FRONTEND_SEMANTIC_EXTRACTION_LIMIT_KEYS = [
  "maxSourceFiles",
  "maxSourceBytes",
  "maxHeaderFiles",
  "maxHeaderBytes",
  "maxIncludeDepth",
  "maxMacroExpansions",
  "maxPreprocessedTokens",
  "maxAstNodes",
  "maxConstexprSteps",
  "maxTemplateInstantiations",
  "maxTemplateDepth",
  "maxDeclarations",
  "maxTypes",
  "maxConstants",
  "maxLayouts",
  "maxTensors",
  "maxOperations",
  "maxTargetIntrinsics",
  "maxDiagnostics",
  "maxOutputBytes",
] as const satisfies readonly CppCuteExtractionLimitName[];

export type CppCuteExtractionLimitName = (typeof EXTRACTION_LIMIT_KEYS)[number];

const MAXIMUM_EXTRACTION_LIMITS: Readonly<Record<CppCuteExtractionLimitName, number>> = Object.freeze({
  maxSourceFiles: 10_000,
  maxSourceBytes: 64 * 1024 * 1024,
  maxHeaderFiles: 100_000,
  maxHeaderBytes: 512 * 1024 * 1024,
  maxIncludeDepth: 1_024,
  maxMacroExpansions: 10_000_000,
  maxPreprocessedTokens: 100_000_000,
  maxAstNodes: 20_000_000,
  maxConstexprSteps: 100_000_000,
  maxTemplateInstantiations: 5_000_000,
  maxTemplateDepth: 4_096,
  maxDeclarations: 5_000_000,
  maxTypes: 5_000_000,
  maxConstants: 5_000_000,
  maxLayouts: 1_000_000,
  maxTensors: 1_000_000,
  maxOperations: 5_000_000,
  maxTargetIntrinsics: 1_000_000,
  maxDiagnostics: 1_000_000,
  maxOutputBytes: 64 * 1024 * 1024,
  maxWallTimeMs: 30 * 60 * 1_000,
  maxCpuTimeMs: 30 * 60 * 1_000,
  maxMemoryBytes: 16 * 1024 * 1024 * 1024,
  maxProcesses: 1_024,
});

export interface CppCuteFrontendProfileVersion extends JsonObject {
  readonly major: typeof CPP_CUTE_FRONTEND_PROFILE_MAJOR;
  readonly minor: typeof CPP_CUTE_FRONTEND_PROFILE_MINOR;
}

export interface CppCuteFrontendProducer extends JsonObject {
  readonly id: string;
  readonly version: string;
}

export interface CppCuteFrontendExtractorProfile extends JsonObject {
  readonly id: string;
  readonly version: string;
  readonly buildId: string;
  readonly binarySha256: string;
  /** Exact adapter inventory that maps resolved frontend facts into the artifact schema. */
  readonly semanticAdapterManifestSha256: string;
}

export interface CppCuteFrontendRunnerProfile extends JsonObject {
  readonly id: string;
  readonly version: string;
  readonly binarySha256: string;
}

export interface CppCuteFrontendContainerProfile extends JsonObject {
  readonly runtime: "docker";
  readonly repository: string;
  readonly platform: "linux/amd64";
  /** Resolved platform-manifest digest, never a mutable tag or multi-platform index. */
  readonly manifestDigest: string;
  /** OCI image-configuration digest transitively bound by manifestDigest. */
  readonly configDigest: string;
}

export interface CppCuteFrontendProvenancePolicy extends JsonObject {
  readonly kind: "external-attestation";
  readonly predicateType: typeof CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE;
  readonly trustStoreSha256: string;
  readonly builderIds: readonly string[];
}

export interface CppCuteFrontendAotDeploymentProfile extends JsonObject {
  readonly mode: "ahead-of-time";
  readonly contractId: "browsergrad.compiler.cpp-cute.aot@1";
  readonly sandboxPolicySha256: string;
  /** Builder/runtime/kernel/cgroup/security-module environment manifest. */
  readonly executionEnvironmentManifestSha256: string;
  readonly extractor: CppCuteFrontendExtractorProfile;
  readonly runner: CppCuteFrontendRunnerProfile;
  readonly container: CppCuteFrontendContainerProfile;
  readonly provenance: CppCuteFrontendProvenancePolicy;
}

export interface CppCuteFrontendBrowserWorkerProfile extends JsonObject {
  readonly protocolId: "browsergrad.compiler.cpp-cute.browser-worker@1";
  /** Package-owned worker module build identity. */
  readonly buildId: string;
  /** Package-owned worker module bytes; executable JS is not asset-manifest supplied. */
  readonly moduleSha256: string;
  readonly moduleByteLength: number;
  readonly moduleFormat: "self-contained-es-module";
  readonly construction: "host-verified-blob-url";
  readonly isolation: "dedicated-worker";
  readonly threading: "single-thread";
  readonly cancellation: "terminate-worker";
  readonly network: "forbidden";
  readonly assetDelivery: "host-verified-transfer";
}

export type CppCuteBrowserRequiredWasmFeature =
  | "bulk-memory"
  | "mutable-globals"
  | "nontrapping-fptoint"
  | "sign-extension";

export interface CppCuteFrontendBrowserCompilerRuntimeProfile extends JsonObject {
  readonly runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1";
  /** SHA-256 of the exact canonical runtime-ABI manifest resource. */
  readonly runtimeAbiManifestSha256: string;
  readonly wasmAddressBits: 32;
  readonly requiredWasmFeatures: readonly CppCuteBrowserRequiredWasmFeature[];
  readonly moduleHandoff: "host-verified-module-or-bytes";
  readonly workerSideFetch: "forbidden";
  readonly memory: JsonObject & {
    readonly sharing: "unshared";
    readonly ownership: "worker";
    readonly initialPages: number;
    readonly maximumPages: number;
    readonly stackByteLength: number;
    readonly maxCompilerWorkingByteLength: number;
  };
  readonly virtualFileSystem: JsonObject & {
    readonly storage: "host-backed-lazy";
    readonly maxRetainedHostPackByteLength: number;
    readonly maxAggregateLiveOpenByteLength: number;
    readonly maxIndexedNodes: number;
    readonly maxIndexLogicalByteLength: number;
  };
}

export interface CppCuteFrontendBrowserAssetLimits extends JsonObject {
  readonly maxAssets: number;
  readonly maxAssetCompressedByteLength: number;
  readonly maxAssetUnpackedByteLength: number;
  readonly maxAssetFileContentByteLength: number;
  readonly maxTotalCompressedByteLength: number;
  readonly maxTotalUnpackedByteLength: number;
  readonly maxTotalFileContentByteLength: number;
}

export interface CppCuteFrontendBrowserDeploymentProfile extends JsonObject {
  readonly mode: "browser-local";
  readonly contractId: "browsergrad.compiler.cpp-cute.browser-worker@1";
  /** Hash of the exact content-addressed browser asset identity projection. */
  readonly assetSetSha256: string;
  /** Hash of release-verified build evidence for every member of the asset set. */
  readonly buildProvenanceLockSha256: string;
  readonly extractor: CppCuteFrontendExtractorProfile;
  readonly worker: CppCuteFrontendBrowserWorkerProfile;
  readonly compilerRuntime: CppCuteFrontendBrowserCompilerRuntimeProfile;
  readonly assetLimits: CppCuteFrontendBrowserAssetLimits;
}

export type CppCuteFrontendDeploymentProfile =
  | CppCuteFrontendAotDeploymentProfile
  | CppCuteFrontendBrowserDeploymentProfile;

export type CppCuteFrontendCompilerOption =
  | (JsonObject & {
      readonly kind: "define";
      readonly name: string;
      readonly value: string | null;
    })
  | (JsonObject & {
      readonly kind: "undefine";
      readonly name: string;
    })
  | (JsonObject & {
      readonly kind: "frontend-option";
      readonly id: "syntax-only" | "error-limit";
      readonly value: string | null;
    })
  | (JsonObject & {
      readonly kind: "warning-policy";
      readonly id: CppCuteFrontendWarningPolicyId;
      readonly disposition: "ignore" | "warn" | "error";
    })
  | (JsonObject & {
      readonly kind: "forced-include";
      readonly includeRootId: string;
      readonly virtualPath: string;
    });

export type CppCuteFrontendSemanticDomain = "host" | "device";

export interface CppCuteFrontendSemanticPassProfile extends JsonObject {
  /** Array position and ordinal are both semantic. */
  readonly ordinal: number;
  readonly passId: "cuda-device-sema" | "cuda-host-sema";
  readonly domain: CppCuteFrontendSemanticDomain;
  /** Device pass owns the extracted graph; host pass validates its shared surface. */
  readonly role: "semantic-extraction" | "validation";
  /** Exact Clang frontend mode; this is not a free-form compiler option. */
  readonly invocationMode: "cuda-device-only" | "cuda-host-only";
  /** Exact primary and auxiliary Clang target identities for the pass. */
  readonly targetTriple: string;
  readonly auxiliaryTargetTriple: string;
  /** Exact CUDA target CPU used by both the primary and auxiliary compilation views. */
  readonly deviceArchitecture: string;
}

export interface CppCuteFrontendPreprocessingPolicy extends JsonObject {
  readonly temporalMacros: JsonObject & {
    readonly policyId: typeof CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID;
    readonly mode: "reject";
  };
}

export interface CppCuteFrontendDiagnosticPolicy extends JsonObject {
  readonly warningRegistryId: typeof CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID;
  readonly baseline: typeof CPP_CUTE_FRONTEND_WARNING_BASELINE;
  /** Exact canonical Clang-to-Artifact-V3 normalization authority. */
  readonly normalizationManifestSha256:
    typeof CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256;
}

export interface CppCuteFrontendLanguageProfile extends JsonObject {
  readonly cxxStandard: "c++17";
  readonly cudaCompatibility: string;
  readonly preprocessing: CppCuteFrontendPreprocessingPolicy;
  readonly diagnostics: CppCuteFrontendDiagnosticPolicy;
  /** Exactly device extraction then host validation over the same unmodified input closure. */
  readonly semanticPasses: readonly CppCuteFrontendSemanticPassProfile[];
  /** Compiler option order is semantic and therefore preserved, not sorted. */
  readonly options: readonly CppCuteFrontendCompilerOption[];
}

export interface CppCuteFrontendTargetProfile extends JsonObject {
  readonly host: JsonObject & {
    readonly triple: string;
    readonly endianness: "little";
    readonly pointerBits: 64;
    readonly dataLayout: string;
  };
  readonly device: JsonObject & {
    readonly triple: "nvptx64-nvidia-cuda";
    readonly architecture: string;
    readonly endianness: "little";
    readonly pointerBits: 64;
    readonly dataLayout: string;
  };
}

export interface CppCuteFrontendCompilerProfile extends JsonObject {
  readonly id: string;
  readonly version: string;
  readonly buildId: string;
  readonly binarySha256: string;
  /** Canonical VFS directory passed exactly to Clang `-resource-dir`. */
  readonly resourceDirectoryVirtualPath: string;
  readonly resourceDirectorySha256: string;
}

export type CppCuteFrontendDependencyKind =
  | "cuda-toolkit"
  | "cutlass"
  | "cccl"
  | "cxx-standard-library"
  | "c-system-headers"
  | "linux-sysroot";

export interface CppCuteFrontendDependencyProfile extends JsonObject {
  readonly dependencyId: string;
  readonly kind: CppCuteFrontendDependencyKind;
  readonly version: string;
  readonly revision: string;
  readonly headerSetSha256: string;
}

export interface CppCuteFrontendToolchainProfile extends JsonObject {
  readonly compiler: CppCuteFrontendCompilerProfile;
  readonly dependencies: readonly CppCuteFrontendDependencyProfile[];
}

export interface CppCuteFrontendVirtualFileSystemProfile extends JsonObject {
  /** Resolution order is semantic and therefore preserved, not sorted. */
  readonly sourceRoots: readonly string[];
  /** Resolution order is semantic and therefore preserved, not sorted. */
  readonly includeRoots: readonly CppCuteFrontendIncludeRoot[];
}

export type CppCuteFrontendIncludeRootOwner =
  | (JsonObject & { readonly kind: "source" })
  | (JsonObject & { readonly kind: "compiler-resource-directory" })
  | (JsonObject & { readonly kind: "dependency"; readonly dependencyId: string });

export interface CppCuteFrontendIncludeRoot extends JsonObject {
  readonly includeRootId: string;
  readonly mode: "quote" | "system";
  readonly virtualPath: string;
  readonly manifestSha256: string;
  readonly owner: CppCuteFrontendIncludeRootOwner;
}

export interface CppCuteFrontendCompatibilityProfile extends JsonObject {
  readonly supportedSourceFeatures: readonly string[];
  readonly unsupportedSourceFeatures: readonly string[];
  readonly unsupportedIntrinsicFamilies: readonly string[];
}

export type CppCuteFrontendExtractionLimits = JsonObject & Readonly<Record<CppCuteExtractionLimitName, number>>;

export interface CppCuteFrontendProfileV2 extends JsonObject {
  readonly schema: typeof CPP_CUTE_FRONTEND_PROFILE_SCHEMA;
  readonly version: CppCuteFrontendProfileVersion;
  readonly profileId: string;
  readonly deployment: CppCuteFrontendDeploymentProfile;
  readonly language: CppCuteFrontendLanguageProfile;
  readonly target: CppCuteFrontendTargetProfile;
  readonly toolchain: CppCuteFrontendToolchainProfile;
  readonly virtualFileSystem: CppCuteFrontendVirtualFileSystemProfile;
  readonly compatibility: CppCuteFrontendCompatibilityProfile;
  readonly extractionLimits: CppCuteFrontendExtractionLimits;
}

export interface CppCuteFrontendCompilationContractVersionV1 extends JsonObject {
  readonly major: typeof CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_MAJOR;
  readonly minor: typeof CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_MINOR;
}

export interface CppCuteFrontendCompilationContractV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_SCHEMA;
  readonly version: CppCuteFrontendCompilationContractVersionV1;
  readonly language: CppCuteFrontendLanguageProfile;
  readonly target: CppCuteFrontendTargetProfile;
  readonly compiler: JsonObject & {
    readonly id: string;
    readonly version: string;
    readonly buildId: string;
    readonly resourceDirectoryVirtualPath: string;
    readonly resourceDirectorySha256: string;
  };
  readonly dependencies: readonly CppCuteFrontendDependencyProfile[];
  readonly virtualFileSystem: CppCuteFrontendVirtualFileSystemProfile;
  readonly compatibility: CppCuteFrontendCompatibilityProfile;
  readonly semanticAdapterManifestSha256: string;
  readonly extractionLimits: JsonObject & Readonly<Record<
    (typeof CPP_CUTE_FRONTEND_SEMANTIC_EXTRACTION_LIMIT_KEYS)[number],
    number
  >>;
}

declare const preparedCppCuteFrontendProfileBrand: unique symbol;

export interface PreparedCppCuteFrontendProfile {
  readonly [preparedCppCuteFrontendProfileBrand]: true;
  readonly profileId: string;
  readonly profileHash: string;
  /** Producer-neutral source-analysis contract used inside frontend artifacts. */
  readonly compilationContractHash: string;
  readonly deploymentMode: "ahead-of-time" | "browser-local";
  readonly extractionLimits: CppCuteFrontendExtractionLimits;
}

export interface PrepareCppCuteFrontendProfileOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteFrontendProfileErrorCode =
  | "BG-COMPILER-CPP-CUTE-PROFILE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-PROFILE-INVALID"
  | "BG-COMPILER-CPP-CUTE-PROFILE-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-PROFILE-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-PROFILE-UNVERIFIED";

export class CppCuteFrontendProfileError extends Error {
  constructor(
    readonly code: CppCuteFrontendProfileErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteFrontendProfileError";
  }
}

export interface PreparedCppCuteFrontendProfileRecord {
  readonly profile: CppCuteFrontendProfileV2;
  readonly profileHash: string;
  readonly compilationContract: CppCuteFrontendCompilationContractV1;
  readonly compilationContractHash: string;
}

export interface PreparedCppCuteAotFrontendProfileRecord extends PreparedCppCuteFrontendProfileRecord {
  readonly profile: CppCuteFrontendProfileV2 & {
    readonly deployment: CppCuteFrontendAotDeploymentProfile;
  };
}

export interface PreparedCppCuteBrowserFrontendProfileRecord extends PreparedCppCuteFrontendProfileRecord {
  readonly profile: CppCuteFrontendProfileV2 & {
    readonly deployment: CppCuteFrontendBrowserDeploymentProfile;
  };
}

export async function prepareCppCuteFrontendProfile(
  value: unknown,
  options: PrepareCppCuteFrontendProfileOptions = {},
): Promise<PreparedCppCuteFrontendProfile> {
  throwIfAborted(options.signal);
  const profile = parseProfile(value);
  canonicalizeJson(profile);
  const profileHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.frontend-profile.v2",
    profile,
  });
  const compilationContract = cppCuteFrontendCompilationContract(profile);
  const compilationContractHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.compilation-contract.v1",
    contract: compilationContract,
  });
  throwIfAborted(options.signal);
  const prepared = Object.freeze({
    profileId: profile.profileId,
    profileHash,
    compilationContractHash,
    deploymentMode: profile.deployment.mode,
    extractionLimits: profile.extractionLimits,
  }) as PreparedCppCuteFrontendProfile;
  PREPARED_PROFILES.set(prepared, Object.freeze({
    profile,
    profileHash,
    compilationContract,
    compilationContractHash,
  }));
  return prepared;
}

export function cppCuteFrontendCompilationContract(
  profile: CppCuteFrontendProfileV2,
): CppCuteFrontendCompilationContractV1 {
  const semanticLimits = Object.fromEntries(
    CPP_CUTE_FRONTEND_SEMANTIC_EXTRACTION_LIMIT_KEYS.map((key) => [key, profile.extractionLimits[key]]),
  ) as CppCuteFrontendCompilationContractV1["extractionLimits"];
  return deepFreezeJson({
    schema: CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_SCHEMA,
    version: {
      major: CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_MAJOR,
      minor: CPP_CUTE_FRONTEND_COMPILATION_CONTRACT_MINOR,
    },
    language: profile.language,
    target: profile.target,
    compiler: {
      id: profile.toolchain.compiler.id,
      version: profile.toolchain.compiler.version,
      buildId: profile.toolchain.compiler.buildId,
      resourceDirectoryVirtualPath:
        profile.toolchain.compiler.resourceDirectoryVirtualPath,
      resourceDirectorySha256: profile.toolchain.compiler.resourceDirectorySha256,
    },
    dependencies: profile.toolchain.dependencies,
    virtualFileSystem: profile.virtualFileSystem,
    compatibility: profile.compatibility,
    semanticAdapterManifestSha256: profile.deployment.extractor.semanticAdapterManifestSha256,
    extractionLimits: semanticLimits,
  });
}

export function unwrapPreparedCppCuteFrontendProfile(
  prepared: PreparedCppCuteFrontendProfile,
): PreparedCppCuteFrontendProfileRecord {
  return getPreparedCppCuteFrontendProfileRecord(prepared);
}

function getPreparedCppCuteFrontendProfileRecord(
  prepared: PreparedCppCuteFrontendProfile,
): PreparedCppCuteFrontendProfileRecord {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const record = PREPARED_PROFILES.get(prepared as object);
  if (record === undefined) unverified();
  return record;
}

export function unwrapPreparedCppCuteAotFrontendProfile(
  prepared: PreparedCppCuteFrontendProfile,
): PreparedCppCuteAotFrontendProfileRecord {
  const record = getPreparedCppCuteFrontendProfileRecord(prepared);
  if (record.profile.deployment.mode !== "ahead-of-time") {
    invalid("$.deployment.mode", "prepared frontend profile is not an ahead-of-time deployment");
  }
  return record as PreparedCppCuteAotFrontendProfileRecord;
}

export function unwrapPreparedCppCuteBrowserFrontendProfile(
  prepared: PreparedCppCuteFrontendProfile,
): PreparedCppCuteBrowserFrontendProfileRecord {
  const record = getPreparedCppCuteFrontendProfileRecord(prepared);
  if (record.profile.deployment.mode !== "browser-local") {
    invalid("$.deployment.mode", "prepared frontend profile is not a browser-local deployment");
  }
  return record as PreparedCppCuteBrowserFrontendProfileRecord;
}

function parseProfile(value: unknown): CppCuteFrontendProfileV2 {
  assertJsonValue(value);
  const object = closedObject(value, [
    "schema",
    "version",
    "profileId",
    "deployment",
    "language",
    "target",
    "toolchain",
    "virtualFileSystem",
    "compatibility",
    "extractionLimits",
  ], "$");
  if (object.schema !== CPP_CUTE_FRONTEND_PROFILE_SCHEMA) {
    invalid("$.schema", `expected ${CPP_CUTE_FRONTEND_PROFILE_SCHEMA}`);
  }
  const version = parseVersion(field(object, "version", "$"), "$.version");
  const profileId = stringValue(field(object, "profileId", "$"), "$.profileId");
  if (!PROFILE_ID.test(profileId)) {
    invalid("$.profileId", "profileId must be a versioned BrowserGrad C++/CuTe profile identifier");
  }
  const language = parseLanguage(field(object, "language", "$"), "$.language");
  const target = parseTarget(field(object, "target", "$"), "$.target");
  validateSemanticPassTargets(language.semanticPasses, target);
  const toolchain = parseToolchain(field(object, "toolchain", "$"), "$.toolchain");
  const virtualFileSystem = parseVirtualFileSystem(
    field(object, "virtualFileSystem", "$"),
    "$.virtualFileSystem",
  );
  validateProfileReferences(language, toolchain, virtualFileSystem);
  const compatibility = parseCompatibility(field(object, "compatibility", "$"), "$.compatibility");
  validateLanguageCompatibility(language, compatibility);
  const deployment = parseDeployment(field(object, "deployment", "$"), "$.deployment");
  validateSemanticAdapterBinding(toolchain, deployment);
  const extractionLimits = parseExtractionLimits(field(object, "extractionLimits", "$"), "$.extractionLimits");
  if (deployment.mode === "browser-local" &&
      toolchain.compiler.binarySha256 !== deployment.extractor.binarySha256) {
    invalid(
      "$.toolchain.compiler.binarySha256",
      "browser-local profile requires one content-identical monolithic Clang/extractor WASM binary",
    );
  }
  if (deployment.mode === "browser-local") {
    const runtime = deployment.compilerRuntime;
    const abiMemory = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.wasm.memory;
    const maximumLinearBytes = runtime.memory.maximumPages * abiMemory.pageByteLength;
    if (extractionLimits.maxMemoryBytes > maximumLinearBytes) {
      invalid(
        "$.extractionLimits.maxMemoryBytes",
        "browser producer memory limit exceeds the runtime maximum linear memory",
      );
    }
    if (extractionLimits.maxOutputBytes > abiMemory.maxResultByteLength) {
      invalid(
        "$.extractionLimits.maxOutputBytes",
        "browser output ceiling exceeds the canonical runtime-ABI result ceiling",
      );
    }
    const reservedLinearBytes = runtime.memory.stackByteLength +
      runtime.memory.maxCompilerWorkingByteLength +
      abiMemory.maxInputFrameByteLength +
      extractionLimits.maxOutputBytes;
    if (reservedLinearBytes > extractionLimits.maxMemoryBytes) {
      invalid(
        "$.extractionLimits.maxMemoryBytes",
        "browser producer memory limit cannot cover stack, compiler working memory, input frame, and output reservations",
      );
    }
    if (extractionLimits.maxSourceBytes + extractionLimits.maxHeaderBytes >
        runtime.virtualFileSystem.maxAggregateLiveOpenByteLength) {
      invalid(
        "$.deployment.compilerRuntime.virtualFileSystem.maxAggregateLiveOpenByteLength",
        "aggregate live-open logical reservation cannot cover the profile source and header byte limits",
      );
    }
  }
  const profile = {
    schema: CPP_CUTE_FRONTEND_PROFILE_SCHEMA,
    version,
    profileId,
    deployment,
    language,
    target,
    toolchain,
    virtualFileSystem,
    compatibility,
    extractionLimits,
  } as CppCuteFrontendProfileV2;
  return deepFreezeJson(profile);
}

function validateSemanticAdapterBinding(
  toolchain: CppCuteFrontendToolchainProfile,
  deployment: CppCuteFrontendDeploymentProfile,
): void {
  if (toolchain.compiler.id !== "clang" || toolchain.compiler.version !== "22.1.8") {
    invalid(
      "$.toolchain.compiler",
      "profile v2.6 semantic adapter requires exact Clang 22.1.8 identity",
    );
  }
  if (deployment.extractor.semanticAdapterManifestSha256 !==
      CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256) {
    invalid(
      "$.deployment.extractor.semanticAdapterManifestSha256",
      "profile v2.6 requires the package canonical semantic-adapter manifest",
    );
  }
}

function parseVersion(value: JsonValue, path: string): CppCuteFrontendProfileVersion {
  const object = closedObject(value, ["major", "minor"], path);
  if (object.major !== CPP_CUTE_FRONTEND_PROFILE_MAJOR) {
    fail(
      "BG-COMPILER-CPP-CUTE-PROFILE-UNSUPPORTED-VERSION",
      `${path}.major`,
      `reader supports profile major ${CPP_CUTE_FRONTEND_PROFILE_MAJOR}`,
    );
  }
  if (object.minor !== CPP_CUTE_FRONTEND_PROFILE_MINOR) {
    fail(
      "BG-COMPILER-CPP-CUTE-PROFILE-UNSUPPORTED-VERSION",
      `${path}.minor`,
      `reader supports closed profile version ${CPP_CUTE_FRONTEND_PROFILE_MAJOR}.${CPP_CUTE_FRONTEND_PROFILE_MINOR}`,
    );
  }
  return { major: CPP_CUTE_FRONTEND_PROFILE_MAJOR, minor: CPP_CUTE_FRONTEND_PROFILE_MINOR };
}

function parseDeployment(value: JsonValue, path: string): CppCuteFrontendDeploymentProfile {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const mode = stringValue(field(value, "mode", path), `${path}.mode`);
  if (mode === "browser-local") return parseBrowserDeployment(value, path);
  if (mode !== "ahead-of-time") {
    invalid(`${path}.mode`, `unknown deployment mode ${JSON.stringify(mode)}`);
  }
  return parseAotDeployment(value, path);
}

function parseAotDeployment(value: JsonValue, path: string): CppCuteFrontendAotDeploymentProfile {
  const object = closedObject(
    value,
    [
      "mode", "contractId", "sandboxPolicySha256", "executionEnvironmentManifestSha256",
      "extractor", "runner", "container", "provenance",
    ],
    path,
  );
  if (object.contractId !== "browsergrad.compiler.cpp-cute.aot@1") {
    invalid(`${path}.contractId`, "profile v2 requires browsergrad.compiler.cpp-cute.aot@1 deployment contract");
  }
  const extractor = parseExtractorProfile(field(object, "extractor", path), `${path}.extractor`);
  const runnerObject = closedObject(
    field(object, "runner", path),
    ["id", "version", "binarySha256"],
    `${path}.runner`,
  );
  const runner = {
    id: boundedString(field(runnerObject, "id", `${path}.runner`), `${path}.runner.id`, 256),
    version: boundedString(field(runnerObject, "version", `${path}.runner`), `${path}.runner.version`, 128),
    binarySha256: sha256(field(runnerObject, "binarySha256", `${path}.runner`), `${path}.runner.binarySha256`),
  };
  const containerObject = closedObject(
    field(object, "container", path),
    ["runtime", "repository", "platform", "manifestDigest", "configDigest"],
    `${path}.container`,
  );
  if (containerObject.runtime !== "docker" || containerObject.platform !== "linux/amd64") {
    invalid(`${path}.container`, "profile v2 requires Docker with resolved linux/amd64 platform manifest");
  }
  const repository = stringValue(field(containerObject, "repository", `${path}.container`), `${path}.container.repository`);
  if (!OCI_REPOSITORY.test(repository)) {
    invalid(`${path}.container.repository`, "container repository must be a canonical lowercase OCI repository without tag");
  }
  const container: CppCuteFrontendContainerProfile = {
    runtime: "docker",
    repository,
    platform: "linux/amd64",
    manifestDigest: ociSha256(
      field(containerObject, "manifestDigest", `${path}.container`),
      `${path}.container.manifestDigest`,
    ),
    configDigest: ociSha256(
      field(containerObject, "configDigest", `${path}.container`),
      `${path}.container.configDigest`,
    ),
  };
  const provenanceObject = closedObject(
    field(object, "provenance", path),
    ["kind", "predicateType", "trustStoreSha256", "builderIds"],
    `${path}.provenance`,
  );
  if (provenanceObject.kind !== "external-attestation") {
    invalid(`${path}.provenance.kind`, "AOT profile requires detached external attestation");
  }
  const builderIds = sortedUniqueStrings(
    field(provenanceObject, "builderIds", `${path}.provenance`),
    `${path}.provenance.builderIds`,
    256,
  );
  if (builderIds.length === 0) invalid(`${path}.provenance.builderIds`, "at least one allowlisted builder is required");
  builderIds.forEach((builderId, index) =>
    validateCanonicalHttpsIdentifier(builderId, `${path}.provenance.builderIds[${index}]`));
  if (provenanceObject.predicateType !== CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE) {
    invalid(
      `${path}.provenance.predicateType`,
      `profile v2 requires ${CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE}`,
    );
  }
  const provenance: CppCuteFrontendProvenancePolicy = {
    kind: "external-attestation" as const,
    predicateType: CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE,
    trustStoreSha256: sha256(
      field(provenanceObject, "trustStoreSha256", `${path}.provenance`),
      `${path}.provenance.trustStoreSha256`,
    ),
    builderIds,
  };
  return {
    mode: "ahead-of-time",
    contractId: "browsergrad.compiler.cpp-cute.aot@1",
    sandboxPolicySha256: sha256(field(object, "sandboxPolicySha256", path), `${path}.sandboxPolicySha256`),
    executionEnvironmentManifestSha256: sha256(
      field(object, "executionEnvironmentManifestSha256", path),
      `${path}.executionEnvironmentManifestSha256`,
    ),
    extractor,
    runner,
    container,
    provenance,
  };
}

function parseBrowserDeployment(value: JsonValue, path: string): CppCuteFrontendBrowserDeploymentProfile {
  const object = closedObject(
    value,
    [
      "mode",
      "contractId",
      "assetSetSha256",
      "buildProvenanceLockSha256",
      "extractor",
      "worker",
      "compilerRuntime",
      "assetLimits",
    ],
    path,
  );
  if (object.contractId !== "browsergrad.compiler.cpp-cute.browser-worker@1") {
    invalid(
      `${path}.contractId`,
      "browser-local profile requires browsergrad.compiler.cpp-cute.browser-worker@1 deployment contract",
    );
  }
  const workerObject = closedObject(
    field(object, "worker", path),
    [
      "protocolId",
      "buildId",
      "moduleSha256",
      "moduleByteLength",
      "moduleFormat",
      "construction",
      "isolation",
      "threading",
      "cancellation",
      "network",
      "assetDelivery",
    ],
    `${path}.worker`,
  );
  requireLiteral(
    field(workerObject, "protocolId", `${path}.worker`),
    "browsergrad.compiler.cpp-cute.browser-worker@1",
    `${path}.worker.protocolId`,
  );
  requireLiteral(
    field(workerObject, "moduleFormat", `${path}.worker`),
    "self-contained-es-module",
    `${path}.worker.moduleFormat`,
  );
  requireLiteral(
    field(workerObject, "construction", `${path}.worker`),
    "host-verified-blob-url",
    `${path}.worker.construction`,
  );
  requireLiteral(field(workerObject, "isolation", `${path}.worker`), "dedicated-worker", `${path}.worker.isolation`);
  requireLiteral(field(workerObject, "threading", `${path}.worker`), "single-thread", `${path}.worker.threading`);
  requireLiteral(field(workerObject, "cancellation", `${path}.worker`), "terminate-worker", `${path}.worker.cancellation`);
  requireLiteral(
    field(workerObject, "network", `${path}.worker`),
    "forbidden",
    `${path}.worker.network`,
  );
  requireLiteral(
    field(workerObject, "assetDelivery", `${path}.worker`),
    "host-verified-transfer",
    `${path}.worker.assetDelivery`,
  );
  const worker: CppCuteFrontendBrowserWorkerProfile = {
    protocolId: "browsergrad.compiler.cpp-cute.browser-worker@1",
    buildId: boundedString(field(workerObject, "buildId", `${path}.worker`), `${path}.worker.buildId`, 256),
    moduleSha256: sha256(
      field(workerObject, "moduleSha256", `${path}.worker`),
      `${path}.worker.moduleSha256`,
    ),
    moduleByteLength: boundedPositiveInteger(
      field(workerObject, "moduleByteLength", `${path}.worker`),
      `${path}.worker.moduleByteLength`,
      64 * 1024 * 1024,
    ),
    moduleFormat: "self-contained-es-module",
    construction: "host-verified-blob-url",
    isolation: "dedicated-worker",
    threading: "single-thread",
    cancellation: "terminate-worker",
    network: "forbidden",
    assetDelivery: "host-verified-transfer",
  };
  const compilerRuntime = parseBrowserCompilerRuntime(
    field(object, "compilerRuntime", path),
    `${path}.compilerRuntime`,
  );
  const assetLimitsObject = closedObject(
    field(object, "assetLimits", path),
    [
      "maxAssets",
      "maxAssetCompressedByteLength",
      "maxAssetUnpackedByteLength",
      "maxAssetFileContentByteLength",
      "maxTotalCompressedByteLength",
      "maxTotalUnpackedByteLength",
      "maxTotalFileContentByteLength",
    ],
    `${path}.assetLimits`,
  );
  const assetLimits: CppCuteFrontendBrowserAssetLimits = {
    maxAssets: boundedPositiveInteger(
      field(assetLimitsObject, "maxAssets", `${path}.assetLimits`),
      `${path}.assetLimits.maxAssets`,
      256,
    ),
    maxAssetCompressedByteLength: boundedPositiveInteger(
      field(assetLimitsObject, "maxAssetCompressedByteLength", `${path}.assetLimits`),
      `${path}.assetLimits.maxAssetCompressedByteLength`,
      1024 * 1024 * 1024,
    ),
    maxAssetUnpackedByteLength: boundedPositiveInteger(
      field(assetLimitsObject, "maxAssetUnpackedByteLength", `${path}.assetLimits`),
      `${path}.assetLimits.maxAssetUnpackedByteLength`,
      2 * 1024 * 1024 * 1024,
    ),
    maxAssetFileContentByteLength: boundedPositiveInteger(
      field(assetLimitsObject, "maxAssetFileContentByteLength", `${path}.assetLimits`),
      `${path}.assetLimits.maxAssetFileContentByteLength`,
      1024 * 1024 * 1024,
    ),
    maxTotalCompressedByteLength: boundedPositiveInteger(
      field(assetLimitsObject, "maxTotalCompressedByteLength", `${path}.assetLimits`),
      `${path}.assetLimits.maxTotalCompressedByteLength`,
      2 * 1024 * 1024 * 1024,
    ),
    maxTotalUnpackedByteLength: boundedPositiveInteger(
      field(assetLimitsObject, "maxTotalUnpackedByteLength", `${path}.assetLimits`),
      `${path}.assetLimits.maxTotalUnpackedByteLength`,
      4 * 1024 * 1024 * 1024,
    ),
    maxTotalFileContentByteLength: boundedPositiveInteger(
      field(assetLimitsObject, "maxTotalFileContentByteLength", `${path}.assetLimits`),
      `${path}.assetLimits.maxTotalFileContentByteLength`,
      2 * 1024 * 1024 * 1024,
    ),
  };
  if (assetLimits.maxAssetCompressedByteLength > assetLimits.maxTotalCompressedByteLength) {
    invalid(
      `${path}.assetLimits.maxAssetCompressedByteLength`,
      "per-asset compressed limit must not exceed the total compressed limit",
    );
  }
  if (assetLimits.maxAssetUnpackedByteLength > assetLimits.maxTotalUnpackedByteLength) {
    invalid(
      `${path}.assetLimits.maxAssetUnpackedByteLength`,
      "per-asset unpacked limit must not exceed the total unpacked limit",
    );
  }
  if (assetLimits.maxAssetFileContentByteLength > assetLimits.maxTotalFileContentByteLength) {
    invalid(
      `${path}.assetLimits.maxAssetFileContentByteLength`,
      "per-asset file-content ceiling cannot exceed total file-content ceiling",
    );
  }
  if (compilerRuntime.virtualFileSystem.maxRetainedHostPackByteLength >
      assetLimits.maxTotalUnpackedByteLength) {
    invalid(
      `${path}.compilerRuntime.virtualFileSystem.maxRetainedHostPackByteLength`,
      "retained host-pack ceiling cannot exceed verified unpacked asset ceiling",
    );
  }
  if (compilerRuntime.virtualFileSystem.maxAggregateLiveOpenByteLength >
      assetLimits.maxTotalFileContentByteLength) {
    invalid(
      `${path}.compilerRuntime.virtualFileSystem.maxAggregateLiveOpenByteLength`,
      "aggregate live-open logical reservation cannot exceed mounted file-content ceiling",
    );
  }
  return {
    mode: "browser-local",
    contractId: "browsergrad.compiler.cpp-cute.browser-worker@1",
    assetSetSha256: sha256(field(object, "assetSetSha256", path), `${path}.assetSetSha256`),
    buildProvenanceLockSha256: sha256(
      field(object, "buildProvenanceLockSha256", path),
      `${path}.buildProvenanceLockSha256`,
    ),
    extractor: parseExtractorProfile(field(object, "extractor", path), `${path}.extractor`),
    worker,
    compilerRuntime,
    assetLimits,
  };
}

function parseBrowserCompilerRuntime(
  value: JsonValue,
  path: string,
): CppCuteFrontendBrowserCompilerRuntimeProfile {
  const object = closedObject(value, [
    "runtimeAbiId",
    "runtimeAbiManifestSha256",
    "wasmAddressBits",
    "requiredWasmFeatures",
    "moduleHandoff",
    "workerSideFetch",
    "memory",
    "virtualFileSystem",
  ], path);
  requireLiteral(
    field(object, "runtimeAbiId", path),
    CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.runtimeAbiId,
    `${path}.runtimeAbiId`,
  );
  const runtimeAbiManifestSha256 = sha256(
    field(object, "runtimeAbiManifestSha256", path),
    `${path}.runtimeAbiManifestSha256`,
  );
  if (runtimeAbiManifestSha256 !== CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256) {
    invalid(
      `${path}.runtimeAbiManifestSha256`,
      "runtime ABI must bind the exact built-in canonical manifest resource",
    );
  }
  if (field(object, "wasmAddressBits", path) !==
      CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.wasm.addressBits) {
    invalid(`${path}.wasmAddressBits`, "runtime address width differs from the canonical ABI manifest");
  }
  requireLiteral(
    field(object, "moduleHandoff", path),
    "host-verified-module-or-bytes",
    `${path}.moduleHandoff`,
  );
  requireLiteral(field(object, "workerSideFetch", path), "forbidden", `${path}.workerSideFetch`);
  const rawFeatures = sortedUniqueStrings(field(object, "requiredWasmFeatures", path), `${path}.requiredWasmFeatures`, 16);
  const expectedFeatures = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.wasm.requiredFeatures;
  if (rawFeatures.length !== expectedFeatures.length ||
      rawFeatures.some((feature, index) => feature !== expectedFeatures[index])) {
    invalid(`${path}.requiredWasmFeatures`, "required WASM features differ from the canonical ABI manifest");
  }
  const memoryObject = closedObject(field(object, "memory", path), [
    "sharing",
    "ownership",
    "initialPages",
    "maximumPages",
    "stackByteLength",
    "maxCompilerWorkingByteLength",
  ], `${path}.memory`);
  requireLiteral(field(memoryObject, "sharing", `${path}.memory`), "unshared", `${path}.memory.sharing`);
  requireLiteral(field(memoryObject, "ownership", `${path}.memory`), "worker", `${path}.memory.ownership`);
  const initialPages = boundedPositiveInteger(
    field(memoryObject, "initialPages", `${path}.memory`),
    `${path}.memory.initialPages`,
    32_768,
  );
  const maximumPages = boundedPositiveInteger(
    field(memoryObject, "maximumPages", `${path}.memory`),
    `${path}.memory.maximumPages`,
    32_768,
  );
  if (initialPages > maximumPages) invalid(`${path}.memory.initialPages`, "initial pages exceed maximum pages");
  const abiMemory = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.wasm.memory;
  if (initialPages !== abiMemory.initialPages) {
    invalid(`${path}.memory.initialPages`, "initial pages differ from the canonical ABI manifest");
  }
  if (maximumPages !== abiMemory.maximumPages) {
    invalid(`${path}.memory.maximumPages`, "maximum pages differ from the canonical ABI manifest");
  }
  const stackByteLength = boundedPositiveInteger(
    field(memoryObject, "stackByteLength", `${path}.memory`),
    `${path}.memory.stackByteLength`,
    256 * 1024 * 1024,
  );
  if (stackByteLength !== abiMemory.stackByteLength) {
    invalid(`${path}.memory.stackByteLength`, "stack reservation differs from the canonical ABI manifest");
  }
  const maxCompilerWorkingByteLength = boundedPositiveInteger(
    field(memoryObject, "maxCompilerWorkingByteLength", `${path}.memory`),
    `${path}.memory.maxCompilerWorkingByteLength`,
    2 * 1024 * 1024 * 1024,
  );
  if (maxCompilerWorkingByteLength > abiMemory.maxCompilerWorkingByteLength) {
    invalid(
      `${path}.memory.maxCompilerWorkingByteLength`,
      "compiler working ceiling exceeds the canonical ABI manifest",
    );
  }
  const memory = {
    sharing: "unshared" as const,
    ownership: "worker" as const,
    initialPages,
    maximumPages,
    stackByteLength,
    maxCompilerWorkingByteLength,
  };
  if (memory.stackByteLength > memory.initialPages * abiMemory.pageByteLength) {
    invalid(`${path}.memory.stackByteLength`, "stack reservation exceeds initial linear memory");
  }
  const vfsObject = closedObject(field(object, "virtualFileSystem", path), [
    "storage",
    "maxRetainedHostPackByteLength",
    "maxAggregateLiveOpenByteLength",
    "maxIndexedNodes",
    "maxIndexLogicalByteLength",
  ], `${path}.virtualFileSystem`);
  requireLiteral(
    field(vfsObject, "storage", `${path}.virtualFileSystem`),
    "host-backed-lazy",
    `${path}.virtualFileSystem.storage`,
  );
  const maxAggregateLiveOpenByteLength = boundedPositiveInteger(
    field(vfsObject, "maxAggregateLiveOpenByteLength", `${path}.virtualFileSystem`),
    `${path}.virtualFileSystem.maxAggregateLiveOpenByteLength`,
    2 * 1024 * 1024 * 1024,
  );
  const abiVfs = CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.vfs;
  if (maxAggregateLiveOpenByteLength > abiVfs.maxAggregateLiveOpenByteLength) {
    invalid(
      `${path}.virtualFileSystem.maxAggregateLiveOpenByteLength`,
      "aggregate live-open logical reservation exceeds the canonical ABI manifest",
    );
  }
  const maxIndexedNodes = boundedPositiveInteger(
    field(vfsObject, "maxIndexedNodes", `${path}.virtualFileSystem`),
    `${path}.virtualFileSystem.maxIndexedNodes`,
    abiVfs.maxIndexedNodes,
  );
  if (maxIndexedNodes > abiVfs.maxIndexedNodes) {
    invalid(
      `${path}.virtualFileSystem.maxIndexedNodes`,
      "indexed-node ceiling exceeds the canonical ABI manifest",
    );
  }
  const maxIndexLogicalByteLength = boundedPositiveInteger(
    field(vfsObject, "maxIndexLogicalByteLength", `${path}.virtualFileSystem`),
    `${path}.virtualFileSystem.maxIndexLogicalByteLength`,
    abiVfs.maxIndexLogicalByteLength,
  );
  if (maxIndexLogicalByteLength > abiVfs.maxIndexLogicalByteLength) {
    invalid(
      `${path}.virtualFileSystem.maxIndexLogicalByteLength`,
      "logical index-byte ceiling exceeds the canonical ABI manifest",
    );
  }
  return {
    runtimeAbiId: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.runtimeAbiId,
    runtimeAbiManifestSha256,
    wasmAddressBits: CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE.body.wasm.addressBits,
    requiredWasmFeatures: rawFeatures as readonly CppCuteBrowserRequiredWasmFeature[],
    moduleHandoff: "host-verified-module-or-bytes",
    workerSideFetch: "forbidden",
    memory,
    virtualFileSystem: {
      storage: "host-backed-lazy",
      maxRetainedHostPackByteLength: boundedPositiveInteger(
        field(vfsObject, "maxRetainedHostPackByteLength", `${path}.virtualFileSystem`),
        `${path}.virtualFileSystem.maxRetainedHostPackByteLength`,
        4 * 1024 * 1024 * 1024,
      ),
      maxAggregateLiveOpenByteLength,
      maxIndexedNodes,
      maxIndexLogicalByteLength,
    },
  };
}

function parseExtractorProfile(value: JsonValue, path: string): CppCuteFrontendExtractorProfile {
  const object = closedObject(
    value,
    ["id", "version", "buildId", "binarySha256", "semanticAdapterManifestSha256"],
    path,
  );
  return {
    id: boundedString(field(object, "id", path), `${path}.id`, 256),
    version: boundedString(field(object, "version", path), `${path}.version`, 128),
    buildId: boundedString(field(object, "buildId", path), `${path}.buildId`, 256),
    binarySha256: sha256(field(object, "binarySha256", path), `${path}.binarySha256`),
    semanticAdapterManifestSha256: sha256(
      field(object, "semanticAdapterManifestSha256", path),
      `${path}.semanticAdapterManifestSha256`,
    ),
  };
}

function parseLanguage(value: JsonValue, path: string): CppCuteFrontendLanguageProfile {
  const object = closedObject(value, [
    "cxxStandard",
    "cudaCompatibility",
    "preprocessing",
    "diagnostics",
    "semanticPasses",
    "options",
  ], path);
  if (object.cxxStandard !== "c++17") invalid(`${path}.cxxStandard`, "profile v2 supports c++17 only");
  return {
    cxxStandard: "c++17",
    cudaCompatibility: boundedString(field(object, "cudaCompatibility", path), `${path}.cudaCompatibility`, 128),
    preprocessing: parsePreprocessingPolicy(
      field(object, "preprocessing", path),
      `${path}.preprocessing`,
    ),
    diagnostics: parseDiagnosticPolicy(
      field(object, "diagnostics", path),
      `${path}.diagnostics`,
    ),
    semanticPasses: parseSemanticPasses(field(object, "semanticPasses", path), `${path}.semanticPasses`),
    options: parseCompilerOptions(field(object, "options", path), `${path}.options`),
  };
}

function parsePreprocessingPolicy(
  value: JsonValue,
  path: string,
): CppCuteFrontendPreprocessingPolicy {
  const object = closedObject(value, ["temporalMacros"], path);
  const temporalPath = `${path}.temporalMacros`;
  const temporal = closedObject(
    field(object, "temporalMacros", path),
    ["policyId", "mode"],
    temporalPath,
  );
  if (temporal.policyId !== CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID) {
    invalid(
      `${temporalPath}.policyId`,
      `temporal macro policy must equal ${CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID}`,
    );
  }
  if (temporal.mode !== "reject") {
    invalid(`${temporalPath}.mode`, "temporal macro policy v1 requires reject mode");
  }
  return {
    temporalMacros: {
      policyId: CPP_CUTE_FRONTEND_TEMPORAL_MACRO_POLICY_ID,
      mode: "reject",
    },
  };
}

function parseDiagnosticPolicy(
  value: JsonValue,
  path: string,
): CppCuteFrontendDiagnosticPolicy {
  const object = closedObject(
    value,
    ["warningRegistryId", "baseline", "normalizationManifestSha256"],
    path,
  );
  if (object.warningRegistryId !== CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID) {
    invalid(
      `${path}.warningRegistryId`,
      `warning registry must equal ${CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID}`,
    );
  }
  if (object.baseline !== CPP_CUTE_FRONTEND_WARNING_BASELINE) {
    invalid(`${path}.baseline`, `warning baseline must equal ${CPP_CUTE_FRONTEND_WARNING_BASELINE}`);
  }
  const normalizationManifestSha256 = sha256(
    field(object, "normalizationManifestSha256", path),
    `${path}.normalizationManifestSha256`,
  );
  if (normalizationManifestSha256 !==
      CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256) {
    invalid(
      `${path}.normalizationManifestSha256`,
      "profile v2.6 requires the package canonical diagnostic-normalization manifest",
    );
  }
  return {
    warningRegistryId: CPP_CUTE_FRONTEND_WARNING_POLICY_REGISTRY_ID,
    baseline: CPP_CUTE_FRONTEND_WARNING_BASELINE,
    normalizationManifestSha256:
      CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
  };
}

function parseSemanticPasses(value: JsonValue, path: string): readonly CppCuteFrontendSemanticPassProfile[] {
  const values = arrayValue(value, path);
  if (values.length !== 2) invalid(path, "CUDA profile requires exactly device extraction then host validation passes");
  const expected = [
    {
      ordinal: 0,
      passId: "cuda-device-sema",
      domain: "device",
      role: "semantic-extraction",
      invocationMode: "cuda-device-only",
    },
    {
      ordinal: 1,
      passId: "cuda-host-sema",
      domain: "host",
      role: "validation",
      invocationMode: "cuda-host-only",
    },
  ] as const;
  return values.map((entry, index) => {
    const passPath = `${path}[${index}]`;
    const object = closedObject(entry, [
      "ordinal", "passId", "domain", "role", "invocationMode",
      "targetTriple", "auxiliaryTargetTriple", "deviceArchitecture",
    ], passPath);
    const wanted = expected[index];
    if (wanted === undefined) invalid(passPath, "unexpected semantic pass");
    if (object.ordinal !== wanted.ordinal) invalid(`${passPath}.ordinal`, `semantic pass ordinal must equal ${wanted.ordinal}`);
    if (object.passId !== wanted.passId) invalid(`${passPath}.passId`, `semantic pass must be ${wanted.passId}`);
    if (object.domain !== wanted.domain) invalid(`${passPath}.domain`, `semantic pass domain must be ${wanted.domain}`);
    if (object.role !== wanted.role) invalid(`${passPath}.role`, `semantic pass role must be ${wanted.role}`);
    if (object.invocationMode !== wanted.invocationMode) {
      invalid(`${passPath}.invocationMode`, `semantic pass invocationMode must be ${wanted.invocationMode}`);
    }
    const deviceArchitecture = stringValue(
      field(object, "deviceArchitecture", passPath),
      `${passPath}.deviceArchitecture`,
    );
    if (!TARGET_ARCHITECTURE.test(deviceArchitecture)) {
      invalid(`${passPath}.deviceArchitecture`, "device architecture must be an sm_NN Clang CUDA target CPU");
    }
    return {
      ordinal: wanted.ordinal,
      passId: wanted.passId,
      domain: wanted.domain,
      role: wanted.role,
      invocationMode: wanted.invocationMode,
      targetTriple: boundedString(field(object, "targetTriple", passPath), `${passPath}.targetTriple`, 256),
      auxiliaryTargetTriple: boundedString(
        field(object, "auxiliaryTargetTriple", passPath),
        `${passPath}.auxiliaryTargetTriple`,
        256,
      ),
      deviceArchitecture,
    };
  });
}

function validateSemanticPassTargets(
  passes: readonly CppCuteFrontendSemanticPassProfile[],
  target: CppCuteFrontendTargetProfile,
): void {
  const devicePass = passes[0];
  if (devicePass?.targetTriple !== target.device.triple ||
      devicePass.auxiliaryTargetTriple !== target.host.triple ||
      devicePass.deviceArchitecture !== target.device.architecture) {
    invalid(
      "$.language.semanticPasses[0]",
      "device semantic pass must bind the exact device triple, host auxiliary triple, and device architecture",
    );
  }
  const hostPass = passes[1];
  if (hostPass?.targetTriple !== target.host.triple ||
      hostPass.auxiliaryTargetTriple !== target.device.triple ||
      hostPass.deviceArchitecture !== target.device.architecture) {
    invalid(
      "$.language.semanticPasses[1]",
      "host semantic pass must bind the exact host triple, device auxiliary triple, and device architecture",
    );
  }
}

function parseCompilerOptions(value: JsonValue, path: string): readonly CppCuteFrontendCompilerOption[] {
  const values = arrayValue(value, path);
  if (values.length > 4_096) resource(path, "compiler option count exceeds 4096");
  const seen = new Set<string>();
  const options = values.map((entry, index) => {
    const optionPath = `${path}[${index}]`;
    if (!isJsonObject(entry)) invalid(optionPath, "compiler option must be an object");
    const kind = stringValue(field(entry, "kind", optionPath), `${optionPath}.kind`);
    let option: CppCuteFrontendCompilerOption;
    let singleton: string;
    if (kind === "define") {
      const object = closedObject(entry, ["kind", "name", "value"], optionPath);
      const name = macroName(field(object, "name", optionPath), `${optionPath}.name`);
      if (isCppCuteFrontendTemporalMacroName(name)) {
        invalid(`${optionPath}.name`, "temporal macro policy forbids command-line definition");
      }
      if (isCppCuteFrontendReservedMacroName(name)) {
        invalid(`${optionPath}.name`, "reserved implementation macro cannot be defined by profile");
      }
      const rawValue = field(object, "value", optionPath);
      const defineValue = rawValue === null ? null : boundedString(rawValue, `${optionPath}.value`, 1_024);
      option = { kind, name, value: defineValue };
      singleton = `macro:${name}`;
    } else if (kind === "undefine") {
      const object = closedObject(entry, ["kind", "name"], optionPath);
      const name = macroName(field(object, "name", optionPath), `${optionPath}.name`);
      if (isCppCuteFrontendTemporalMacroName(name)) {
        invalid(`${optionPath}.name`, "temporal macro policy forbids command-line undefinition");
      }
      if (isCppCuteFrontendReservedMacroName(name)) {
        invalid(`${optionPath}.name`, "reserved implementation macro cannot be undefined by profile");
      }
      option = { kind, name };
      singleton = `macro:${name}`;
    } else if (kind === "frontend-option") {
      const object = closedObject(entry, ["kind", "id", "value"], optionPath);
      const id = stringValue(field(object, "id", optionPath), `${optionPath}.id`);
      if (id !== "syntax-only" && id !== "error-limit") {
        invalid(`${optionPath}.id`, `frontend option ${JSON.stringify(id)} is not allowlisted by profile v2`);
      }
      const rawValue = field(object, "value", optionPath);
      const optionValue = rawValue === null ? null : boundedString(rawValue, `${optionPath}.value`, 64);
      if (id === "syntax-only" && optionValue !== null) {
        invalid(`${optionPath}.value`, `${id} does not accept a value`);
      }
      if (id === "error-limit" && (optionValue === null || !/^[1-9][0-9]{0,5}$/u.test(optionValue))) {
        invalid(`${optionPath}.value`, "error-limit requires a canonical positive decimal no greater than six digits");
      }
      option = { kind, id, value: optionValue };
      singleton = `frontend:${id}`;
    } else if (kind === "warning-policy") {
      const object = closedObject(entry, ["kind", "id", "disposition"], optionPath);
      const id = stringValue(field(object, "id", optionPath), `${optionPath}.id`);
      const mapping = cppCuteFrontendWarningPolicyMapping(id);
      if (mapping === undefined) {
        invalid(`${optionPath}.id`, `warning policy ${JSON.stringify(id)} is not present in the closed registry`);
      }
      const disposition = stringValue(field(object, "disposition", optionPath), `${optionPath}.disposition`);
      if (disposition !== "ignore" && disposition !== "warn" && disposition !== "error") {
        invalid(`${optionPath}.disposition`, "warning disposition must be ignore, warn, or error");
      }
      option = { kind, id: mapping.policyId, disposition };
      singleton = `warning:${mapping.policyId}`;
    } else if (kind === "forced-include") {
      const object = closedObject(entry, ["kind", "includeRootId", "virtualPath"], optionPath);
      const includeRootId = dependencyId(
        field(object, "includeRootId", optionPath),
        `${optionPath}.includeRootId`,
      );
      const virtualPath = stringValue(field(object, "virtualPath", optionPath), `${optionPath}.virtualPath`);
      requireVirtualPath(virtualPath, `${optionPath}.virtualPath`);
      option = { kind, includeRootId, virtualPath };
      singleton = `forced-include:${includeRootId}:${virtualPath}`;
    } else {
      invalid(`${optionPath}.kind`, `unknown compiler option kind ${JSON.stringify(kind)}`);
    }
    if (seen.has(singleton)) invalid(optionPath, `conflicting or duplicate compiler option ${singleton}`);
    seen.add(singleton);
    return option;
  });
  for (const required of ["syntax-only", "error-limit"] as const) {
    if (!seen.has(`frontend:${required}`)) {
      invalid(path, `compiler options must contain exactly one ${required} frontend option`);
    }
  }
  return options;
}

function parseTarget(value: JsonValue, path: string): CppCuteFrontendTargetProfile {
  const object = closedObject(value, ["host", "device"], path);
  const hostPath = `${path}.host`;
  const host = closedObject(field(object, "host", path), ["triple", "endianness", "pointerBits", "dataLayout"], hostPath);
  const devicePath = `${path}.device`;
  const device = closedObject(
    field(object, "device", path),
    ["triple", "architecture", "endianness", "pointerBits", "dataLayout"],
    devicePath,
  );
  const deviceArchitecture = stringValue(field(device, "architecture", devicePath), `${devicePath}.architecture`);
  if (!TARGET_ARCHITECTURE.test(deviceArchitecture)) {
    invalid(`${devicePath}.architecture`, "device architecture must be an sm_NN Clang CUDA target CPU");
  }
  if (device.triple !== "nvptx64-nvidia-cuda") {
    invalid(`${devicePath}.triple`, "profile v2.6 requires the nvptx64-nvidia-cuda device triple");
  }
  if (host.endianness !== "little" || device.endianness !== "little") {
    invalid(path, "CUDA profile v2.6 requires little-endian host and NVPTX targets");
  }
  if (host.pointerBits !== 64 || device.pointerBits !== 64) {
    invalid(path, "CUDA profile v2.6 requires matching 64-bit host and device pointer widths");
  }
  const hostDataLayout = boundedString(field(host, "dataLayout", hostPath), `${hostPath}.dataLayout`, 1_024);
  const deviceDataLayout = boundedString(field(device, "dataLayout", devicePath), `${devicePath}.dataLayout`, 1_024);
  if (!LLVM_DATA_LAYOUT.test(hostDataLayout)) invalid(`${hostPath}.dataLayout`, "invalid closed LLVM data-layout string");
  if (!LLVM_DATA_LAYOUT.test(deviceDataLayout)) invalid(`${devicePath}.dataLayout`, "invalid closed LLVM data-layout string");
  return {
    host: {
      triple: boundedString(field(host, "triple", hostPath), `${hostPath}.triple`, 256),
      endianness: "little",
      pointerBits: 64,
      dataLayout: hostDataLayout,
    },
    device: {
      triple: "nvptx64-nvidia-cuda",
      architecture: deviceArchitecture,
      endianness: "little",
      pointerBits: 64,
      dataLayout: deviceDataLayout,
    },
  };
}

function parseToolchain(value: JsonValue, path: string): CppCuteFrontendToolchainProfile {
  const object = closedObject(value, ["compiler", "dependencies"], path);
  const compilerObject = closedObject(
    field(object, "compiler", path),
    [
      "id",
      "version",
      "buildId",
      "binarySha256",
      "resourceDirectoryVirtualPath",
      "resourceDirectorySha256",
    ],
    `${path}.compiler`,
  );
  const resourceDirectoryVirtualPath = stringValue(
    field(compilerObject, "resourceDirectoryVirtualPath", `${path}.compiler`),
    `${path}.compiler.resourceDirectoryVirtualPath`,
  );
  requireVirtualPath(
    resourceDirectoryVirtualPath,
    `${path}.compiler.resourceDirectoryVirtualPath`,
  );
  if (
    resourceDirectoryVirtualPath === "/"
    || resourceDirectoryVirtualPath.length + "/include".length > 4_096
  ) {
    invalid(
      `${path}.compiler.resourceDirectoryVirtualPath`,
      "compiler resource directory must be a non-root path with a representable include child",
    );
  }
  const compiler = {
    id: boundedString(field(compilerObject, "id", `${path}.compiler`), `${path}.compiler.id`, 128),
    version: boundedString(field(compilerObject, "version", `${path}.compiler`), `${path}.compiler.version`, 128),
    buildId: boundedString(field(compilerObject, "buildId", `${path}.compiler`), `${path}.compiler.buildId`, 256),
    binarySha256: sha256(field(compilerObject, "binarySha256", `${path}.compiler`), `${path}.compiler.binarySha256`),
    resourceDirectoryVirtualPath,
    resourceDirectorySha256: sha256(
      field(compilerObject, "resourceDirectorySha256", `${path}.compiler`),
      `${path}.compiler.resourceDirectorySha256`,
    ),
  };
  const dependencies = arrayValue(field(object, "dependencies", path), `${path}.dependencies`).map((entry, index) =>
    parseDependency(entry, `${path}.dependencies[${index}]`));
  requireSortedUnique(dependencies, (dependency) => dependency.dependencyId, `${path}.dependencies`);
  const dependencyKinds: readonly CppCuteFrontendDependencyKind[] = [
    "cuda-toolkit",
    "cutlass",
    "cccl",
    "cxx-standard-library",
    "c-system-headers",
    "linux-sysroot",
  ];
  for (const kind of dependencyKinds) {
    if (dependencies.filter((dependency) => dependency.kind === kind).length > 1) {
      invalid(`${path}.dependencies`, `toolchain may pin at most one ${kind} dependency`);
    }
  }
  for (const requiredKind of ["cuda-toolkit", "cutlass", "cxx-standard-library"] as const) {
    if (dependencies.filter((dependency) => dependency.kind === requiredKind).length !== 1) {
      invalid(`${path}.dependencies`, `toolchain must pin exactly one ${requiredKind} dependency`);
    }
  }
  const cSystemProviders = dependencies.filter(
    (dependency) => dependency.kind === "c-system-headers" || dependency.kind === "linux-sysroot",
  );
  if (cSystemProviders.length !== 1) {
    invalid(`${path}.dependencies`, "toolchain must pin exactly one C-system provider or Linux sysroot dependency");
  }
  return { compiler, dependencies };
}

function parseDependency(value: JsonValue, path: string): CppCuteFrontendDependencyProfile {
  const object = closedObject(value, ["dependencyId", "kind", "version", "revision", "headerSetSha256"], path);
  const parsedDependencyId = dependencyId(field(object, "dependencyId", path), `${path}.dependencyId`);
  const kind = stringValue(field(object, "kind", path), `${path}.kind`);
  if (kind !== "cuda-toolkit" && kind !== "cutlass" && kind !== "cccl" && kind !== "cxx-standard-library" &&
      kind !== "c-system-headers" && kind !== "linux-sysroot") {
    invalid(`${path}.kind`, `unknown dependency kind ${JSON.stringify(kind)}`);
  }
  const revision = boundedString(field(object, "revision", path), `${path}.revision`, 256);
  if ((kind === "cutlass" || kind === "cccl") && !GIT_COMMIT.test(revision)) {
    invalid(`${path}.revision`, `${kind} revision must be an exact 40-digit lowercase Git commit`);
  }
  return {
    dependencyId: parsedDependencyId,
    kind,
    version: boundedString(field(object, "version", path), `${path}.version`, 128),
    revision,
    headerSetSha256: sha256(field(object, "headerSetSha256", path), `${path}.headerSetSha256`),
  };
}

function parseVirtualFileSystem(value: JsonValue, path: string): CppCuteFrontendVirtualFileSystemProfile {
  const object = closedObject(value, ["sourceRoots", "includeRoots"], path);
  const sourceRoots = virtualRootArray(field(object, "sourceRoots", path), `${path}.sourceRoots`);
  const includeRoots = arrayValue(field(object, "includeRoots", path), `${path}.includeRoots`).map((entry, index) =>
    parseIncludeRoot(entry, `${path}.includeRoots[${index}]`));
  if (includeRoots.length > 256) resource(`${path}.includeRoots`, "include root count exceeds 256");
  if (new Set(includeRoots.map((root) => root.includeRootId)).size !== includeRoots.length) {
    invalid(`${path}.includeRoots`, "include root IDs must be unique");
  }
  if (new Set(includeRoots.map((root) => root.virtualPath)).size !== includeRoots.length) {
    invalid(`${path}.includeRoots`, "include root virtual paths must have unique ownership");
  }
  if (sourceRoots.length === 0) invalid(`${path}.sourceRoots`, "at least one source root is required");
  if (includeRoots.length === 0) invalid(`${path}.includeRoots`, "at least one include root is required");
  return { sourceRoots, includeRoots };
}

function parseIncludeRoot(value: JsonValue, path: string): CppCuteFrontendIncludeRoot {
  const object = closedObject(value, ["includeRootId", "mode", "virtualPath", "manifestSha256", "owner"], path);
  const includeRootId = dependencyId(field(object, "includeRootId", path), `${path}.includeRootId`);
  if (object.mode !== "quote" && object.mode !== "system") invalid(`${path}.mode`, "include root mode must be quote or system");
  const virtualPath = stringValue(field(object, "virtualPath", path), `${path}.virtualPath`);
  requireVirtualPath(virtualPath, `${path}.virtualPath`);
  return {
    includeRootId,
    mode: object.mode,
    virtualPath,
    manifestSha256: sha256(field(object, "manifestSha256", path), `${path}.manifestSha256`),
    owner: parseIncludeRootOwner(field(object, "owner", path), `${path}.owner`),
  };
}

function parseIncludeRootOwner(value: JsonValue, path: string): CppCuteFrontendIncludeRootOwner {
  if (!isJsonObject(value)) invalid(path, "include root owner must be an object");
  const kind = stringValue(field(value, "kind", path), `${path}.kind`);
  if (kind === "source" || kind === "compiler-resource-directory") {
    closedObject(value, ["kind"], path);
    return { kind };
  }
  if (kind === "dependency") {
    const object = closedObject(value, ["kind", "dependencyId"], path);
    return {
      kind,
      dependencyId: dependencyId(field(object, "dependencyId", path), `${path}.dependencyId`),
    };
  }
  invalid(`${path}.kind`, `unknown include root owner kind ${JSON.stringify(kind)}`);
}

function validateProfileReferences(
  language: CppCuteFrontendLanguageProfile,
  toolchain: CppCuteFrontendToolchainProfile,
  virtualFileSystem: CppCuteFrontendVirtualFileSystemProfile,
): void {
  const dependencies = new Map(toolchain.dependencies.map((dependency) => [dependency.dependencyId, dependency]));
  const includeRoots = new Map(
    virtualFileSystem.includeRoots.map((root, index) => [root.includeRootId, { root, index }]),
  );
  const ownedDependencies = new Set<string>();
  let compilerResourceRoots = 0;

  for (const [index, root] of virtualFileSystem.includeRoots.entries()) {
    const rootPath = `$.virtualFileSystem.includeRoots[${index}]`;
    if (root.owner.kind === "source") {
      const containers = virtualFileSystem.sourceRoots.filter((sourceRoot) =>
        virtualPathContains(sourceRoot, root.virtualPath));
      if (containers.length !== 1) {
        invalid(`${rootPath}.virtualPath`, "source-owned include root must belong to exactly one source root");
      }
      continue;
    }
    if (root.owner.kind === "compiler-resource-directory") {
      compilerResourceRoots += 1;
      if (
        root.mode !== "system"
        || root.virtualPath !== `${toolchain.compiler.resourceDirectoryVirtualPath}/include`
      ) {
        invalid(
          rootPath,
          "compiler-owned include root must be the exact system include child of resourceDirectoryVirtualPath",
        );
      }
      if (root.manifestSha256 !== toolchain.compiler.resourceDirectorySha256) {
        invalid(`${rootPath}.manifestSha256`, "compiler-owned include root must bind the compiler resource directory");
      }
      continue;
    }
    const dependency = dependencies.get(root.owner.dependencyId);
    if (dependency === undefined) {
      invalid(`${rootPath}.owner.dependencyId`, "include root owner does not name a pinned toolchain dependency");
    }
    if (root.manifestSha256 !== dependency.headerSetSha256) {
      invalid(`${rootPath}.manifestSha256`, "dependency-owned include root must bind its dependency header set");
    }
    ownedDependencies.add(dependency.dependencyId);
  }

  if (compilerResourceRoots !== 1) {
    invalid(
      "$.virtualFileSystem.includeRoots",
      "profile must expose exactly one compiler resource directory include root",
    );
  }
  for (const dependency of toolchain.dependencies) {
    if (!ownedDependencies.has(dependency.dependencyId)) {
      invalid(
        "$.virtualFileSystem.includeRoots",
        `dependency ${JSON.stringify(dependency.dependencyId)} must own at least one include root`,
      );
    }
  }

  for (const [index, option] of language.options.entries()) {
    if (option.kind !== "forced-include") continue;
    const optionPath = `$.language.options[${index}]`;
    const referenced = includeRoots.get(option.includeRootId);
    if (referenced === undefined) {
      invalid(`${optionPath}.includeRootId`, "forced include does not name a declared include root");
    }
    if (!virtualPathContains(referenced.root.virtualPath, option.virtualPath) ||
        referenced.root.virtualPath === option.virtualPath) {
      invalid(`${optionPath}.virtualPath`, "forced include must be a file contained by its declared include root");
    }
  }
}

function parseCompatibility(value: JsonValue, path: string): CppCuteFrontendCompatibilityProfile {
  const object = closedObject(value, [
    "supportedSourceFeatures",
    "unsupportedSourceFeatures",
    "unsupportedIntrinsicFamilies",
  ], path);
  const supportedSourceFeatures = capabilitySet(
    field(object, "supportedSourceFeatures", path),
    `${path}.supportedSourceFeatures`,
  );
  if (supportedSourceFeatures.length === 0) {
    invalid(`${path}.supportedSourceFeatures`, "profile must name supported source features");
  }
  return {
    supportedSourceFeatures,
    unsupportedSourceFeatures: capabilitySet(
      field(object, "unsupportedSourceFeatures", path),
      `${path}.unsupportedSourceFeatures`,
    ),
    unsupportedIntrinsicFamilies: capabilitySet(
      field(object, "unsupportedIntrinsicFamilies", path),
      `${path}.unsupportedIntrinsicFamilies`,
    ),
  };
}

function validateLanguageCompatibility(
  language: CppCuteFrontendLanguageProfile,
  compatibility: CppCuteFrontendCompatibilityProfile,
): void {
  const temporalFeature = "cxx:temporal-macros@1";
  if (language.preprocessing.temporalMacros.mode === "reject") {
    if (!compatibility.unsupportedSourceFeatures.includes(temporalFeature)) {
      invalid(
        "$.compatibility.unsupportedSourceFeatures",
        `reject temporal macro policy requires ${temporalFeature}`,
      );
    }
    if (compatibility.supportedSourceFeatures.includes(temporalFeature)) {
      invalid(
        "$.compatibility.supportedSourceFeatures",
        `reject temporal macro policy forbids supported claim ${temporalFeature}`,
      );
    }
  }
}

function parseExtractionLimits(value: JsonValue, path: string): CppCuteFrontendExtractionLimits {
  const object = closedObject(value, EXTRACTION_LIMIT_KEYS, path);
  const result = Object.create(null) as Record<CppCuteExtractionLimitName, number>;
  for (const key of EXTRACTION_LIMIT_KEYS) {
    const limit = field(object, key, path);
    const maximum = MAXIMUM_EXTRACTION_LIMITS[key];
    if (typeof limit !== "number" || !Number.isSafeInteger(limit) || limit <= 0 || limit > maximum) {
      fail(
        "BG-COMPILER-CPP-CUTE-PROFILE-RESOURCE-LIMIT",
        `${path}.${key}`,
        `${key} must be a positive safe integer no greater than ${maximum}`,
      );
    }
    result[key] = limit;
  }
  if (result.maxCpuTimeMs > result.maxWallTimeMs * result.maxProcesses) {
    invalid(`${path}.maxCpuTimeMs`, "aggregate CPU budget exceeds wall-time × process budget");
  }
  return result as CppCuteFrontendExtractionLimits;
}

function virtualRootArray(value: JsonValue, path: string): readonly string[] {
  const values = arrayValue(value, path);
  if (values.length > 256) resource(path, "array length exceeds 256");
  const roots = values.map((entry, index) => {
    const root = stringValue(entry, `${path}[${index}]`);
    requireVirtualPath(root, `${path}[${index}]`);
    return root;
  });
  if (new Set(roots).size !== roots.length) invalid(path, "virtual roots must be unique");
  return roots;
}

function requireVirtualPath(value: string, path: string): void {
  const error = findCppCuteVirtualPathError(value);
  if (error !== null) invalid(path, error);
}

function virtualPathContains(root: string, candidate: string): boolean {
  return root === "/" ? candidate.startsWith("/") : candidate === root || candidate.startsWith(`${root}/`);
}

function validateCanonicalHttpsIdentifier(value: string, path: string): void {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    invalid(path, "builder identity must be a canonical HTTPS URL");
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" || parsed.password !== "" || parsed.search !== "" ||
      parsed.hash !== "" || parsed.pathname === "/" || parsed.pathname.endsWith("/") ||
      `${parsed.origin}${parsed.pathname}` !== value) {
    invalid(path, "builder identity must be a canonical credential-free HTTPS URL without query, fragment, or trailing slash");
  }
}

function capabilitySet(value: JsonValue, path: string): readonly string[] {
  const values = sortedUniqueStrings(value, path, 4_096);
  values.forEach((entry, index) => {
    if (!CAPABILITY_ID.test(entry)) invalid(`${path}[${index}]`, "invalid namespaced capability ID");
  });
  return values;
}

function sortedUniqueStrings(value: JsonValue, path: string, maximum: number): readonly string[] {
  const values = stringArray(value, path, maximum, 1_024);
  requireSortedUnique(values, (entry) => entry, path);
  return values;
}

function stringArray(value: JsonValue, path: string, maximumLength: number, maximumStringBytes: number): readonly string[] {
  const values = arrayValue(value, path);
  if (values.length > maximumLength) resource(path, `array length exceeds ${maximumLength}`);
  return values.map((entry, index) => boundedString(entry, `${path}[${index}]`, maximumStringBytes));
}

function requireSortedUnique<T>(values: readonly T[], key: (value: T) => string, path: string): void {
  for (let index = 1; index < values.length; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    if (previous === undefined || current === undefined ||
        compareCanonicalStrings(key(previous), key(current)) >= 0) {
      invalid(path, "set-like entries must be strictly sorted and unique");
    }
  }
}

function closedObject(value: unknown, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value as JsonValue)) invalid(path, "expected object");
  const object = value as JsonObject;
  const unknown = Object.keys(object).filter((key) => !fields.includes(key));
  const missing = fields.filter((key) => !Object.hasOwn(object, key));
  if (unknown.length > 0) invalid(path, `unknown closed-record fields: ${unknown.sort().join(", ")}`);
  if (missing.length > 0) invalid(path, `missing required fields: ${missing.sort().join(", ")}`);
  return object;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(`${path}.${name}`, "field is required");
  return value;
}

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(path, "expected string");
  return value;
}

function macroName(value: JsonValue, path: string): string {
  const name = stringValue(value, path);
  if (!MACRO_NAME.test(name)) invalid(path, "macro name must be a C/C++ identifier");
  return name;
}

function dependencyId(value: JsonValue, path: string): string {
  const id = stringValue(value, path);
  if (!DEPENDENCY_ID.test(id)) invalid(path, "invalid dependency or include-root ID");
  return id;
}

function boundedString(value: JsonValue, path: string, maximumBytes: number): string {
  const text = stringValue(value, path);
  if (text.length === 0 || new TextEncoder().encode(text).byteLength > maximumBytes || text.includes("\0")) {
    invalid(path, `string must be non-empty, NUL-free, and at most ${maximumBytes} UTF-8 bytes`);
  }
  return text;
}

function boundedPositiveInteger(value: JsonValue, path: string, maximum: number): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0 || value > maximum) {
    resource(path, `value must be a positive safe integer no greater than ${maximum}`);
  }
  return value;
}

function requireLiteral(value: JsonValue, expected: string, path: string): void {
  if (value !== expected) invalid(path, `expected ${JSON.stringify(expected)}`);
}

function sha256(value: JsonValue, path: string): string {
  const text = stringValue(value, path);
  if (!SHA256_HEX.test(text)) invalid(path, "SHA-256 must be 64 lowercase hexadecimal digits");
  return text;
}

function ociSha256(value: JsonValue, path: string): string {
  const text = stringValue(value, path);
  if (!OCI_SHA256.test(text)) invalid(path, "OCI digest must use sha256:<64 lowercase hexadecimal digits>");
  return text;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    fail("BG-COMPILER-CPP-CUTE-PROFILE-CANCELLED", "$.signal", "profile preparation was aborted");
  }
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-PROFILE-UNVERIFIED",
    "$",
    "C++/CuTe artifact verification requires an opaque prepared frontend profile",
  );
}

function resource(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-PROFILE-RESOURCE-LIMIT", path, message);
}

function invalid(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-PROFILE-INVALID", path, message);
}

function fail(code: CppCuteFrontendProfileErrorCode, path: string, message: string): never {
  throw new CppCuteFrontendProfileError(code, path, message);
}

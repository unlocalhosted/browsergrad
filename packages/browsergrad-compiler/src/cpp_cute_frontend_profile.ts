import {
  assertJsonValue,
  canonicalizeJson,
  deepFreezeJson,
  hashCanonicalJson,
  isJsonObject,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

export const CPP_CUTE_FRONTEND_PROFILE_SCHEMA = "browsergrad.compiler.cpp-cute.frontend-profile";
export const CPP_CUTE_FRONTEND_PROFILE_MAJOR = 1;
export const CPP_CUTE_FRONTEND_PROFILE_MINOR = 0;
export const CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE =
  "https://browsergrad.dev/provenance/cpp-cute-aot/v1";

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const OCI_SHA256 = /^sha256:[0-9a-f]{64}$/u;
const GIT_COMMIT = /^[0-9a-f]{40}$/u;
const PROFILE_ID = /^browsergrad\.compiler\.cpp-cute\.[a-z0-9][a-z0-9._-]*@[1-9][0-9]*$/u;
const CAPABILITY_ID = /^[a-z][a-z0-9.-]*:[a-z][a-z0-9._-]*(?:@[1-9][0-9]*)?$/u;
const DEPENDENCY_ID = /^[a-z][a-z0-9._-]*$/u;
const MACRO_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/u;
const WARNING_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/u;
const TARGET_ARCHITECTURE = /^(?:sm|compute)_[1-9][0-9][a-z]?$/u;
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

export interface CppCuteFrontendDeploymentProfile extends JsonObject {
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
      readonly id: "syntax-only" | "cuda-host-only" | "error-limit";
      readonly value: string | null;
    })
  | (JsonObject & {
      readonly kind: "warning-policy";
      readonly id: string;
      readonly disposition: "ignore" | "warn" | "error";
    });

export interface CppCuteFrontendLanguageProfile extends JsonObject {
  readonly cxxStandard: "c++17";
  readonly cudaCompatibility: string;
  /** Compiler option order is semantic and therefore preserved, not sorted. */
  readonly options: readonly CppCuteFrontendCompilerOption[];
}

export interface CppCuteFrontendTargetProfile extends JsonObject {
  readonly hostTriple: string;
  readonly deviceArchitecture: string;
  readonly endianness: "little" | "big";
  readonly pointerBits: 32 | 64;
}

export interface CppCuteFrontendCompilerProfile extends JsonObject {
  readonly id: string;
  readonly version: string;
  readonly buildId: string;
  readonly binarySha256: string;
  readonly resourceDirectorySha256: string;
}

export interface CppCuteFrontendDependencyProfile extends JsonObject {
  readonly dependencyId: string;
  readonly kind: "cuda-toolkit" | "cutlass" | "cccl" | "cxx-standard-library";
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

export interface CppCuteFrontendIncludeRoot extends JsonObject {
  readonly includeRootId: string;
  readonly mode: "quote" | "system";
  readonly virtualPath: string;
  readonly manifestSha256: string;
}

export interface CppCuteFrontendCompatibilityProfile extends JsonObject {
  readonly expectedHeaderSetSha256: string;
  readonly supportedSourceFeatures: readonly string[];
  readonly unsupportedIntrinsicFamilies: readonly string[];
}

export type CppCuteFrontendExtractionLimits = JsonObject & Readonly<Record<CppCuteExtractionLimitName, number>>;

export interface CppCuteFrontendProfileV1 extends JsonObject {
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

declare const preparedCppCuteFrontendProfileBrand: unique symbol;

export interface PreparedCppCuteFrontendProfile {
  readonly [preparedCppCuteFrontendProfileBrand]: true;
  readonly profileId: string;
  readonly profileHash: string;
  readonly deploymentMode: "ahead-of-time";
  readonly expectedHeaderSetSha256: string;
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
  readonly profile: CppCuteFrontendProfileV1;
  readonly profileHash: string;
}

export async function prepareCppCuteFrontendProfile(
  value: unknown,
  options: PrepareCppCuteFrontendProfileOptions = {},
): Promise<PreparedCppCuteFrontendProfile> {
  throwIfAborted(options.signal);
  const profile = parseProfile(value);
  canonicalizeJson(profile);
  const profileHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.frontend-profile.v1",
    profile,
  });
  throwIfAborted(options.signal);
  const prepared = Object.freeze({
    profileId: profile.profileId,
    profileHash,
    deploymentMode: profile.deployment.mode,
    expectedHeaderSetSha256: profile.compatibility.expectedHeaderSetSha256,
    extractionLimits: profile.extractionLimits,
  }) as PreparedCppCuteFrontendProfile;
  PREPARED_PROFILES.set(prepared, Object.freeze({ profile, profileHash }));
  return prepared;
}

export function unwrapPreparedCppCuteFrontendProfile(
  prepared: PreparedCppCuteFrontendProfile,
): PreparedCppCuteFrontendProfileRecord {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const record = PREPARED_PROFILES.get(prepared as object);
  if (record === undefined) unverified();
  return record;
}

function parseProfile(value: unknown): CppCuteFrontendProfileV1 {
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
  const profile = {
    schema: CPP_CUTE_FRONTEND_PROFILE_SCHEMA,
    version,
    profileId,
    deployment: parseDeployment(field(object, "deployment", "$"), "$.deployment"),
    language: parseLanguage(field(object, "language", "$"), "$.language"),
    target: parseTarget(field(object, "target", "$"), "$.target"),
    toolchain: parseToolchain(field(object, "toolchain", "$"), "$.toolchain"),
    virtualFileSystem: parseVirtualFileSystem(field(object, "virtualFileSystem", "$"), "$.virtualFileSystem"),
    compatibility: parseCompatibility(field(object, "compatibility", "$"), "$.compatibility"),
    extractionLimits: parseExtractionLimits(field(object, "extractionLimits", "$"), "$.extractionLimits"),
  } as CppCuteFrontendProfileV1;
  return deepFreezeJson(profile);
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
  const object = closedObject(
    value,
    [
      "mode", "contractId", "sandboxPolicySha256", "executionEnvironmentManifestSha256",
      "extractor", "runner", "container", "provenance",
    ],
    path,
  );
  if (object.mode !== "ahead-of-time") invalid(`${path}.mode`, "profile v1 supports ahead-of-time extraction only");
  if (object.contractId !== "browsergrad.compiler.cpp-cute.aot@1") {
    invalid(`${path}.contractId`, "profile v1 requires browsergrad.compiler.cpp-cute.aot@1 deployment contract");
  }
  const extractorObject = closedObject(
    field(object, "extractor", path),
    ["id", "version", "buildId", "binarySha256"],
    `${path}.extractor`,
  );
  const extractor = {
    id: boundedString(field(extractorObject, "id", `${path}.extractor`), `${path}.extractor.id`, 256),
    version: boundedString(field(extractorObject, "version", `${path}.extractor`), `${path}.extractor.version`, 128),
    buildId: boundedString(field(extractorObject, "buildId", `${path}.extractor`), `${path}.extractor.buildId`, 256),
    binarySha256: sha256(field(extractorObject, "binarySha256", `${path}.extractor`), `${path}.extractor.binarySha256`),
  };
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
    invalid(`${path}.container`, "profile v1 requires Docker with resolved linux/amd64 platform manifest");
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
      `profile v1 requires ${CPP_CUTE_FRONTEND_PROVENANCE_PREDICATE_TYPE}`,
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

function parseLanguage(value: JsonValue, path: string): CppCuteFrontendLanguageProfile {
  const object = closedObject(value, ["cxxStandard", "cudaCompatibility", "options"], path);
  if (object.cxxStandard !== "c++17") invalid(`${path}.cxxStandard`, "profile v1 supports c++17 only");
  return {
    cxxStandard: "c++17",
    cudaCompatibility: boundedString(field(object, "cudaCompatibility", path), `${path}.cudaCompatibility`, 128),
    options: parseCompilerOptions(field(object, "options", path), `${path}.options`),
  };
}

function parseCompilerOptions(value: JsonValue, path: string): readonly CppCuteFrontendCompilerOption[] {
  const values = arrayValue(value, path);
  if (values.length > 4_096) resource(path, "compiler option count exceeds 4096");
  const seen = new Set<string>();
  return values.map((entry, index) => {
    const optionPath = `${path}[${index}]`;
    if (!isJsonObject(entry)) invalid(optionPath, "compiler option must be an object");
    const kind = stringValue(field(entry, "kind", optionPath), `${optionPath}.kind`);
    let option: CppCuteFrontendCompilerOption;
    let singleton: string;
    if (kind === "define") {
      const object = closedObject(entry, ["kind", "name", "value"], optionPath);
      const name = macroName(field(object, "name", optionPath), `${optionPath}.name`);
      const rawValue = field(object, "value", optionPath);
      const defineValue = rawValue === null ? null : boundedString(rawValue, `${optionPath}.value`, 1_024);
      option = { kind, name, value: defineValue };
      singleton = `macro:${name}`;
    } else if (kind === "undefine") {
      const object = closedObject(entry, ["kind", "name"], optionPath);
      const name = macroName(field(object, "name", optionPath), `${optionPath}.name`);
      option = { kind, name };
      singleton = `macro:${name}`;
    } else if (kind === "frontend-option") {
      const object = closedObject(entry, ["kind", "id", "value"], optionPath);
      const id = stringValue(field(object, "id", optionPath), `${optionPath}.id`);
      if (id !== "syntax-only" && id !== "cuda-host-only" && id !== "error-limit") {
        invalid(`${optionPath}.id`, `frontend option ${JSON.stringify(id)} is not allowlisted by profile v1`);
      }
      const rawValue = field(object, "value", optionPath);
      const optionValue = rawValue === null ? null : boundedString(rawValue, `${optionPath}.value`, 64);
      if ((id === "syntax-only" || id === "cuda-host-only") && optionValue !== null) {
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
      if (!WARNING_ID.test(id)) invalid(`${optionPath}.id`, "invalid warning policy ID");
      const disposition = stringValue(field(object, "disposition", optionPath), `${optionPath}.disposition`);
      if (disposition !== "ignore" && disposition !== "warn" && disposition !== "error") {
        invalid(`${optionPath}.disposition`, "warning disposition must be ignore, warn, or error");
      }
      option = { kind, id, disposition };
      singleton = `warning:${id}`;
    } else {
      invalid(`${optionPath}.kind`, `unknown compiler option kind ${JSON.stringify(kind)}`);
    }
    if (seen.has(singleton)) invalid(optionPath, `conflicting or duplicate compiler option ${singleton}`);
    seen.add(singleton);
    return option;
  });
}

function parseTarget(value: JsonValue, path: string): CppCuteFrontendTargetProfile {
  const object = closedObject(value, ["hostTriple", "deviceArchitecture", "endianness", "pointerBits"], path);
  const deviceArchitecture = stringValue(field(object, "deviceArchitecture", path), `${path}.deviceArchitecture`);
  if (!TARGET_ARCHITECTURE.test(deviceArchitecture)) {
    invalid(`${path}.deviceArchitecture`, "device architecture must be an sm_NN or compute_NN profile");
  }
  if (object.endianness !== "little" && object.endianness !== "big") {
    invalid(`${path}.endianness`, "endianness must be little or big");
  }
  if (object.pointerBits !== 32 && object.pointerBits !== 64) {
    invalid(`${path}.pointerBits`, "pointerBits must be 32 or 64");
  }
  return {
    hostTriple: boundedString(field(object, "hostTriple", path), `${path}.hostTriple`, 256),
    deviceArchitecture,
    endianness: object.endianness,
    pointerBits: object.pointerBits,
  };
}

function parseToolchain(value: JsonValue, path: string): CppCuteFrontendToolchainProfile {
  const object = closedObject(value, ["compiler", "dependencies"], path);
  const compilerObject = closedObject(
    field(object, "compiler", path),
    ["id", "version", "buildId", "binarySha256", "resourceDirectorySha256"],
    `${path}.compiler`,
  );
  const compiler = {
    id: boundedString(field(compilerObject, "id", `${path}.compiler`), `${path}.compiler.id`, 128),
    version: boundedString(field(compilerObject, "version", `${path}.compiler`), `${path}.compiler.version`, 128),
    buildId: boundedString(field(compilerObject, "buildId", `${path}.compiler`), `${path}.compiler.buildId`, 256),
    binarySha256: sha256(field(compilerObject, "binarySha256", `${path}.compiler`), `${path}.compiler.binarySha256`),
    resourceDirectorySha256: sha256(
      field(compilerObject, "resourceDirectorySha256", `${path}.compiler`),
      `${path}.compiler.resourceDirectorySha256`,
    ),
  };
  const dependencies = arrayValue(field(object, "dependencies", path), `${path}.dependencies`).map((entry, index) =>
    parseDependency(entry, `${path}.dependencies[${index}]`));
  requireSortedUnique(dependencies, (dependency) => dependency.dependencyId, `${path}.dependencies`);
  for (const requiredKind of ["cuda-toolkit", "cutlass"] as const) {
    if (dependencies.filter((dependency) => dependency.kind === requiredKind).length !== 1) {
      invalid(`${path}.dependencies`, `toolchain must pin exactly one ${requiredKind} dependency`);
    }
  }
  return { compiler, dependencies };
}

function parseDependency(value: JsonValue, path: string): CppCuteFrontendDependencyProfile {
  const object = closedObject(value, ["dependencyId", "kind", "version", "revision", "headerSetSha256"], path);
  const dependencyId = stringValue(field(object, "dependencyId", path), `${path}.dependencyId`);
  if (!DEPENDENCY_ID.test(dependencyId)) invalid(`${path}.dependencyId`, "invalid dependency ID");
  const kind = stringValue(field(object, "kind", path), `${path}.kind`);
  if (kind !== "cuda-toolkit" && kind !== "cutlass" && kind !== "cccl" && kind !== "cxx-standard-library") {
    invalid(`${path}.kind`, `unknown dependency kind ${JSON.stringify(kind)}`);
  }
  const revision = boundedString(field(object, "revision", path), `${path}.revision`, 256);
  if ((kind === "cutlass" || kind === "cccl") && !GIT_COMMIT.test(revision)) {
    invalid(`${path}.revision`, `${kind} revision must be an exact 40-digit lowercase Git commit`);
  }
  return {
    dependencyId,
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
  if (sourceRoots.length === 0) invalid(`${path}.sourceRoots`, "at least one source root is required");
  if (includeRoots.length === 0) invalid(`${path}.includeRoots`, "at least one include root is required");
  return { sourceRoots, includeRoots };
}

function parseIncludeRoot(value: JsonValue, path: string): CppCuteFrontendIncludeRoot {
  const object = closedObject(value, ["includeRootId", "mode", "virtualPath", "manifestSha256"], path);
  const includeRootId = stringValue(field(object, "includeRootId", path), `${path}.includeRootId`);
  if (!DEPENDENCY_ID.test(includeRootId)) invalid(`${path}.includeRootId`, "invalid include root ID");
  if (object.mode !== "quote" && object.mode !== "system") invalid(`${path}.mode`, "include root mode must be quote or system");
  const virtualPath = stringValue(field(object, "virtualPath", path), `${path}.virtualPath`);
  validateVirtualPath(virtualPath, `${path}.virtualPath`);
  return {
    includeRootId,
    mode: object.mode,
    virtualPath,
    manifestSha256: sha256(field(object, "manifestSha256", path), `${path}.manifestSha256`),
  };
}

function parseCompatibility(value: JsonValue, path: string): CppCuteFrontendCompatibilityProfile {
  const object = closedObject(value, [
    "expectedHeaderSetSha256",
    "supportedSourceFeatures",
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
    expectedHeaderSetSha256: sha256(field(object, "expectedHeaderSetSha256", path), `${path}.expectedHeaderSetSha256`),
    supportedSourceFeatures,
    unsupportedIntrinsicFamilies: capabilitySet(
      field(object, "unsupportedIntrinsicFamilies", path),
      `${path}.unsupportedIntrinsicFamilies`,
    ),
  };
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
  const roots = stringArray(value, path, 256, 1_024);
  if (new Set(roots).size !== roots.length) invalid(path, "virtual roots must be unique");
  roots.forEach((root, index) => validateVirtualPath(root, `${path}[${index}]`));
  return roots;
}

export function validateCppCuteVirtualPath(value: string, path = "$"): void {
  validateVirtualPath(value, path);
}

function validateVirtualPath(value: string, path: string): void {
  if (!value.startsWith("/") || value.length > 1_024 || value.includes("\\") || value.includes("\0")) {
    invalid(path, "virtual path must be bounded absolute POSIX syntax");
  }
  const segments = value.split("/");
  if (segments.some((segment, index) => index > 0 && (segment.length === 0 || segment === "." || segment === ".."))) {
    invalid(path, "virtual path must be normalized and must not contain empty, . or .. segments");
  }
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
    if (previous === undefined || current === undefined || key(previous).localeCompare(key(current)) >= 0) {
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

function boundedString(value: JsonValue, path: string, maximumBytes: number): string {
  const text = stringValue(value, path);
  if (text.length === 0 || new TextEncoder().encode(text).byteLength > maximumBytes || text.includes("\0")) {
    invalid(path, `string must be non-empty, NUL-free, and at most ${maximumBytes} UTF-8 bytes`);
  }
  return text;
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

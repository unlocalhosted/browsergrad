import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  decodeWireJson,
  hashCanonicalJson,
  sha256Hex,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
  type PreparedCppCuteBrowserRuntimeAbiManifest,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE,
  type CppCuteBrowserBuildInputLockV1Resource,
} from "./resources/cpp_cute_browser_build_lock_v1.js";

export const CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-build-input-lock";
export const CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MAJOR = 1;
export const CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MINOR = 0;
export const CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT = 256 * 1024;

const LOCK_ID = /^bg\.cpp\.browser-build-input-lock\.sha256\.[0-9a-f]{64}$/u;
const SHA1_HEX = /^[0-9a-f]{40}$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const SHA256_DIGEST = /^sha256:[0-9a-f]{64}$/u;
const WIRE_U64 = /^(?:0|[1-9][0-9]{0,19})$/u;
const SAFE_RELATIVE_PATH = /^(?!\/)(?!.*\\)[A-Za-z0-9._/-]+$/u;
const PLACEHOLDER = /@[A-Z][A-Z0-9_]*@/gu;
const MAX_U64 = 18_446_744_073_709_551_615n;
const HEADER_INPUT_PROJECTION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-input-projection.v1";
const HEADER_RESOURCE_DEFINITION_NAMES = new Set([
  "CLANG_ENABLE_HLSL",
  "LLVM_TARGETS_TO_BUILD",
]);
const HEADER_DISTRIBUTION_OUTPUT_ROLES = new Set([
  "clang-resource-header-vfs",
  "component-license",
  "cuda-header-vfs",
  "cutlass-header-vfs",
  "libcxx-header-vfs",
  "license-inventory",
  "linux-sysroot-header-vfs",
  "third-party-notices",
]);
const REQUIRED_BROWSER_ASSET_OUTPUTS = Object.freeze([
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/clang-extractor.wasm",
    role: "clang-extractor",
    mediaType: "application/wasm",
  }),
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
    role: "clang-resource-header-vfs",
    mediaType: "application/octet-stream",
  }),
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
    role: "cuda-header-vfs",
    mediaType: "application/octet-stream",
  }),
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
    role: "cutlass-header-vfs",
    mediaType: "application/octet-stream",
  }),
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/diagnostic-normalization.json",
    role: "diagnostic-normalization-manifest",
    mediaType:
      "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json",
  }),
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
    role: "libcxx-header-vfs",
    mediaType: "application/octet-stream",
  }),
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
    role: "linux-sysroot-header-vfs",
    mediaType: "application/octet-stream",
  }),
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/runtime-abi-manifest.json",
    role: "runtime-abi-manifest",
    mediaType:
      "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
  }),
  Object.freeze({
    path: "assets/browsergrad-cpp-cute/semantic-adapter-manifest.json",
    role: "semantic-adapter-manifest",
    mediaType:
      "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
  }),
]);
const PREPARED_LOCKS = new WeakMap<object, StoredCppCuteBrowserBuildInputLock>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export const CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT,
  maxDepth: 24,
  maxNodes: 16_384,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 512,
  maxObjectProperties: 128,
  maxRank: 1,
  maxIntegerBits: 64,
  maxArithmeticOperations: 32_768,
});

export type CppCuteBrowserBuildInputLockV1 = CppCuteBrowserBuildInputLockV1Resource;
export type CppCuteBrowserBuildInputLockBodyV1 = CppCuteBrowserBuildInputLockV1["body"];

declare const preparedCppCuteBrowserBuildInputLockBrand: unique symbol;

/**
 * Opaque authority over exact canonical input-lock bytes.
 *
 * This authority proves only selected build inputs and policy. It never proves
 * build execution, output identity, reproducibility, producer trust, or release
 * readiness.
 */
export interface PreparedCppCuteBrowserBuildInputLock {
  readonly [preparedCppCuteBrowserBuildInputLockBrand]: true;
  readonly lockId: string;
  readonly resourceSha256: string;
  readonly recipeSha256: string;
  readonly extractorSourceSetSha256: string;
  readonly noticeInventorySha256: string;
  readonly runtimeAbiManifestId: string;
  readonly runtimeAbiResourceSha256: string;
  readonly runtimeAbiResourceByteLength: number;
  readonly resourceByteLength: number;
  readonly releaseReady: false;
  readonly releaseBlockerIds: readonly string[];
}

export interface PreparedCppCuteBrowserBuildInputLockRecord {
  readonly lock: CppCuteBrowserBuildInputLockV1;
}

interface StoredCppCuteBrowserBuildInputLock extends PreparedCppCuteBrowserBuildInputLockRecord {
  readonly bytes: Uint8Array;
}

export interface PrepareCppCuteBrowserBuildInputLockOptions {
  readonly signal?: AbortSignal;
}

export interface CppCuteBrowserBuildReleaseReadiness {
  readonly ready: false;
  readonly blockerIds: readonly string[];
}

export type CppCuteBrowserBuildInputLockErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-NONCANONICAL-BYTES"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-UNVERIFIED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-RELEASE-BLOCKED";

export class CppCuteBrowserBuildInputLockError extends Error {
  constructor(
    readonly code: CppCuteBrowserBuildInputLockErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserBuildInputLockError";
  }
}

validateBodyInvariants(CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE.body);

const BUILTIN_RESOURCE_BYTES = canonicalResourceBytes(
  CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE,
);
const BUILTIN_BODY_BYTES = canonicalResourceBytes(
  CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE.body,
);

/** Returns a disposable copy of the exact selected input-lock resource. */
export function cppCuteBrowserBuildInputLockResourceBytes(): Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

export async function decodeCppCuteBrowserBuildInputLock(
  bytes: Uint8Array,
  options: PrepareCppCuteBrowserBuildInputLockOptions = {},
): Promise<PreparedCppCuteBrowserBuildInputLock> {
  const signal = normalizeOptions(options);
  const snapshot = snapshotBytes(bytes);
  throwIfAborted(signal);
  let value: JsonValue;
  try {
    value = decodeWireJson(snapshot, {
      limits: CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$bytes", "build-input lock decoding exceeded fixed resource limits", { cause });
    }
    invalid("$bytes", "build-input lock bytes are not bounded strict JSON", { cause });
  }
  const lock = parseLock(value);
  const canonical = canonicalResourceBytes(lock);
  if (!equalBytes(snapshot, canonical)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-NONCANONICAL-BYTES",
      "$bytes",
      "build-input lock bytes must exactly equal canonical JSON bytes",
    );
  }
  throwIfAborted(signal);
  const expectedLockId = await deriveCppCuteBrowserBuildInputLockId(lock.body);
  if (lock.lockId !== expectedLockId) {
    hashMismatch("$.lockId", `lock ID must equal ${expectedLockId}`);
  }
  throwIfAborted(signal);
  const releaseBlockerIds = deriveReleaseBlockerIds(lock.body);
  verifyReleasePolicy(lock.body, releaseBlockerIds);
  let runtimeAbi: PreparedCppCuteBrowserRuntimeAbiManifest;
  try {
    runtimeAbi = await decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    );
  } catch (cause) {
    throwIfAborted(signal);
    invalid(
      "$.body.runtimeAbiResource",
      "package canonical runtime-ABI resource failed strict decoding",
      { cause },
    );
  }
  if (runtimeAbi.manifestId !== lock.body.runtimeAbiResource.manifestId ||
      runtimeAbi.runtimeAbiId !== lock.body.runtimeAbiResource.runtimeAbiId ||
      runtimeAbi.resourceSha256 !== lock.body.runtimeAbiResource.resourceSha256 ||
      runtimeAbi.resourceByteLength !== Number(BigInt(lock.body.runtimeAbiResource.resourceByteLength))) {
    hashMismatch(
      "$.body.runtimeAbiResource",
      "build binding differs from strict-decoded package canonical runtime-ABI bytes",
    );
  }
  throwIfAborted(signal);
  const [
    resourceSha256,
    recipeSha256,
    extractorSourceSetSha256,
    noticeInventorySha256,
  ] = await Promise.all([
    hashBytes(snapshot, "$bytes"),
    hashJson({
      domain: "browsergrad.compiler.cpp-cute.browser-build-recipe.v1",
      recipe: lock.body.recipe,
    }, "$.body.recipe"),
    hashJson({
      domain: lock.body.recipe.extractorSource.hashDomain,
      files: lock.body.recipe.extractorSource.files,
    }, "$.body.recipe.extractorSource"),
    hashJson({
      domain: "browsergrad.compiler.cpp-cute.browser-build-notices.v1",
      notices: lock.body.notices,
    }, "$.body.notices"),
  ]);
  if (extractorSourceSetSha256 !== lock.body.recipe.extractorSource.sourceSetSha256) {
    hashMismatch(
      "$.body.recipe.extractorSource.sourceSetSha256",
      "extractor source-set digest differs from its exact file projection",
    );
  }
  throwIfAborted(signal);
  const prepared = Object.freeze({
    lockId: expectedLockId,
    resourceSha256,
    recipeSha256,
    extractorSourceSetSha256,
    noticeInventorySha256,
    runtimeAbiManifestId: runtimeAbi.manifestId,
    runtimeAbiResourceSha256: runtimeAbi.resourceSha256,
    runtimeAbiResourceByteLength: runtimeAbi.resourceByteLength,
    resourceByteLength: snapshot.byteLength,
    releaseReady: false,
    releaseBlockerIds: Object.freeze([...releaseBlockerIds]),
  }) as PreparedCppCuteBrowserBuildInputLock;
  PREPARED_LOCKS.set(prepared, Object.freeze({
    lock,
    bytes: new Uint8Array(snapshot),
  }));
  return prepared;
}

export async function deriveCppCuteBrowserBuildInputLockId(
  body: CppCuteBrowserBuildInputLockBodyV1,
): Promise<string> {
  const digest = await hashJson({
    domain: "browsergrad.compiler.cpp-cute.browser-build-input-lock-id.v1",
    body,
  }, "$.lockId");
  return `bg.cpp.browser-build-input-lock.sha256.${digest}`;
}

/**
 * Derives the narrow identity of build-lock fields that can affect the exact
 * header-distribution subset. This is an identity projection only: accepting
 * it does not verify a build lock or grant header, license, or release
 * authority.
 */
export async function deriveCppCuteBrowserHeaderInputProjectionId(
  body: CppCuteBrowserBuildInputLockBodyV1,
): Promise<string> {
  const clangWasmStages = body.recipe.stages.filter(
    (stage) => stage.stageId === "clang-extractor-wasm",
  );
  if (clangWasmStages.length !== 1) {
    invalid(
      "$.body.recipe.stages",
      "header-input projection requires one Clang-Wasm stage",
    );
  }
  const clangWasmStage = clangWasmStages[0];
  if (clangWasmStage === undefined) {
    invalid("$.body.recipe.stages", "header-input projection lost its Clang-Wasm stage");
  }
  const resourceDefinitions = clangWasmStage.definitions.filter(
    (definition) => HEADER_RESOURCE_DEFINITION_NAMES.has(definition.name),
  );
  const resourceDefinitionNames = new Set<string>(
    resourceDefinitions.map((definition) => definition.name),
  );
  if (resourceDefinitions.length !== HEADER_RESOURCE_DEFINITION_NAMES.size ||
      [...HEADER_RESOURCE_DEFINITION_NAMES].some(
        (name) => !resourceDefinitionNames.has(name),
      )) {
    invalid(
      "$.body.recipe.stages.clang-extractor-wasm.definitions",
      "header-input projection requires the complete configured-resource definition set",
    );
  }
  const outputs = body.recipe.distributedOutputPlan.outputs.filter(
    (output) => HEADER_DISTRIBUTION_OUTPUT_ROLES.has(output.role),
  );
  if (outputs.length !== 17) {
    invalid(
      "$.body.recipe.distributedOutputPlan.outputs",
      "header-input projection requires the exact 17-output header subset",
    );
  }
  const digest = await hashJson({
    domain: HEADER_INPUT_PROJECTION_HASH_DOMAIN,
    body: {
      sources: body.sources,
      configuredClangResourceHeaders: {
        stageId: clangWasmStage.stageId,
        definitions: resourceDefinitions,
      },
      distributedOutputPlan: {
        closure: body.recipe.distributedOutputPlan.closure,
        outputs,
      },
      notices: body.notices,
    },
  }, "$.headerInputProjectionId");
  return `bg.cpp.browser-header-input-projection.sha256.${digest}`;
}

/**
 * Derives the header-input identity only from verifier-issued build-lock
 * authority.
 */
export async function cppCuteBrowserHeaderInputProjectionId(
  prepared: PreparedCppCuteBrowserBuildInputLock,
): Promise<string> {
  return deriveCppCuteBrowserHeaderInputProjectionId(storedLock(prepared).lock.body);
}

export function unwrapPreparedCppCuteBrowserBuildInputLock(
  prepared: PreparedCppCuteBrowserBuildInputLock,
): PreparedCppCuteBrowserBuildInputLockRecord {
  const stored = storedLock(prepared);
  return Object.freeze({ lock: stored.lock });
}

export function canonicalCppCuteBrowserBuildInputLockBytes(
  prepared: PreparedCppCuteBrowserBuildInputLock,
): Uint8Array {
  return new Uint8Array(storedLock(prepared).bytes);
}

export function cppCuteBrowserBuildReleaseReadiness(
  prepared: PreparedCppCuteBrowserBuildInputLock,
): CppCuteBrowserBuildReleaseReadiness {
  const stored = storedLock(prepared);
  const blockerIds = deriveReleaseBlockerIds(stored.lock.body);
  return Object.freeze({ ready: false, blockerIds: Object.freeze([...blockerIds]) });
}

export function assertCppCuteBrowserBuildReleaseReady(
  prepared: PreparedCppCuteBrowserBuildInputLock,
): never {
  const readiness = cppCuteBrowserBuildReleaseReadiness(prepared);
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-RELEASE-BLOCKED",
    "$.body.releasePolicy",
    `input lock cannot grant release authority; unresolved blockers: ${readiness.blockerIds.join(", ")}`,
  );
}

function parseLock(value: JsonValue): CppCuteBrowserBuildInputLockV1 {
  const object = closedObject(value, ["schema", "version", "lockId", "body"], "$", true);
  literal(field(object, "schema", "$"), CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_SCHEMA, "$.schema");
  const version = closedObject(field(object, "version", "$"), ["major", "minor"], "$.version", true);
  if (version.major !== CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MAJOR) {
    unsupported("$.version.major", `reader supports major ${CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MAJOR}`);
  }
  if (version.minor !== CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MINOR) {
    unsupported(
      "$.version.minor",
      `closed reader supports ${CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MAJOR}.${CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MINOR} only`,
    );
  }
  const lockId = boundedPattern(field(object, "lockId", "$"), "$.lockId", LOCK_ID);
  const body = field(object, "body", "$");
  if (typeof body !== "object" || body === null || Array.isArray(body)) invalid("$.body", "expected object");
  validateBodyInvariants(body as JsonObject);
  const bodyBytes = canonicalResourceBytes(body);
  if (!equalBytes(bodyBytes, BUILTIN_BODY_BYTES)) {
    invalid("$.body", "body does not equal the single supported pinned build-input selection");
  }
  return {
    schema: CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MAJOR,
      minor: CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_MINOR,
    },
    lockId,
    body: CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_V1_RESOURCE.body,
  } as CppCuteBrowserBuildInputLockV1;
}

function validateBodyInvariants(value: JsonObject): void {
  const body = value as unknown as CppCuteBrowserBuildInputLockBodyV1;
  try {
    const runtimeAbiResourceBytes = cppCuteBrowserRuntimeAbiManifestResourceBytes();
    if (body.runtimeAbiResource.outputPath !==
          "assets/browsergrad-cpp-cute/runtime-abi-manifest.json" ||
        body.runtimeAbiResource.mediaType !==
          "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json" ||
        body.runtimeAbiResource.runtimeAbiId !==
          "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1" ||
        body.runtimeAbiResource.manifestId !== CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID ||
        body.runtimeAbiResource.resourceSha256 !==
          CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256 ||
        body.runtimeAbiResource.resourceByteLength !== String(runtimeAbiResourceBytes.byteLength) ||
        body.runtimeAbiResource.byteIdentity !== "must-equal-package-canonical-resource" ||
        body.runtimeAbiResource.authority !==
          "design-reference-only-no-wasm-conformance-worker-or-release-authority") {
      invalid(
        "$.body.runtimeAbiResource",
        "runtime-ABI build binding must equal the package canonical design resource without granting execution or release authority",
      );
    }
    assertSortedUniqueStrings(body.sources.map((source) => source.sourceId), "$.body.sources[*].sourceId");
    for (const [index, source] of body.sources.entries()) {
      const path = `$.body.sources[${index}]`;
      assertHttpsUrl(source.repository, `${path}.repository`);
      assertHttpsUrl(source.acquisitionUrl, `${path}.acquisitionUrl`);
      assertSha1(source.commit, `${path}.commit`);
      assertSha1(source.treeSha1, `${path}.treeSha1`);
      assertSha256(source.archiveSha256, `${path}.archiveSha256`);
      assertWireU64(source.archiveByteLength, `${path}.archiveByteLength`);
      if ("attestationUrl" in source) {
        assertHttpsUrl(source.attestationUrl, `${path}.attestationUrl`);
        assertSha256(source.attestationSha256, `${path}.attestationSha256`);
        assertWireU64(source.attestationByteLength, `${path}.attestationByteLength`);
      }
    }

    assertSha256Digest(body.builder.imageIndexDigest, "$.body.builder.imageIndexDigest");
    assertSha256Digest(body.builder.platformManifestDigest, "$.body.builder.platformManifestDigest");
    assertSha256Digest(body.builder.imageConfigDigest, "$.body.builder.imageConfigDigest");
    assertHttpsUrl(body.builder.emsdk.repository, "$.body.builder.emsdk.repository");
    assertSha1(body.builder.emsdk.commit, "$.body.builder.emsdk.commit");
    assertSha1(body.builder.emsdk.treeSha1, "$.body.builder.emsdk.treeSha1");
    assertSha1(body.builder.emsdk.releaseBundleCommit, "$.body.builder.emsdk.releaseBundleCommit");
    assertHttpsUrl(body.builder.emsdk.releaseBundleUrl, "$.body.builder.emsdk.releaseBundleUrl");
    assertSha256(body.builder.emsdk.releaseBundleSha256, "$.body.builder.emsdk.releaseBundleSha256");
    assertWireU64(body.builder.emsdk.releaseBundleByteLength, "$.body.builder.emsdk.releaseBundleByteLength");
    assertSha1(body.builder.emsdk.emscriptenCommit, "$.body.builder.emsdk.emscriptenCommit");
    assertSha1(body.builder.emsdk.llvmToolchainCommit, "$.body.builder.emsdk.llvmToolchainCommit");
    assertSha1(body.builder.emsdk.binaryenCommit, "$.body.builder.emsdk.binaryenCommit");

    assertWireU64(body.recipe.sourceDateEpoch, "$.body.recipe.sourceDateEpoch");
    const extractorSource = body.recipe.extractorSource;
    if (extractorSource.hashDomain !==
        "browsergrad.compiler.cpp-cute.browser-extractor-source-set.v1") {
      invalid("$.body.recipe.extractorSource.hashDomain", "unknown extractor source hash domain");
    }
    assertSha256(
      extractorSource.sourceSetSha256,
      "$.body.recipe.extractorSource.sourceSetSha256",
    );
    const extractorFiles = extractorSource.files as readonly {
      readonly path: string;
      readonly sha256: string;
      readonly byteLength: string;
    }[];
    if (extractorFiles.length === 0) {
      invalid("$.body.recipe.extractorSource.files", "extractor source set must not be empty");
    }
    assertSortedUniqueStrings(
      extractorFiles.map((file) => file.path),
      "$.body.recipe.extractorSource.files[*].path",
    );
    for (const [index, file] of extractorFiles.entries()) {
      const path = `$.body.recipe.extractorSource.files[${index}]`;
      assertSafeRelativePath(file.path, `${path}.path`);
      assertSha256(file.sha256, `${path}.sha256`);
      assertWireU64(file.byteLength, `${path}.byteLength`);
      if (file.byteLength === "0") invalid(`${path}.byteLength`, "extractor source file must be nonempty");
    }
    if (body.recipe.parallelJobs !== 4) {
      invalid("$.body.recipe.parallelJobs", "reviewed build parallelism must equal four");
    }
    assertSortedUniqueStrings(body.recipe.environment.map((entry) => entry.name), "$.body.recipe.environment[*].name");
    assertSortedUniqueStrings(body.recipe.prefixMapKinds, "$.body.recipe.prefixMapKinds");
    assertUniqueStrings(body.recipe.stages.map((stage) => stage.stageId), "$.body.recipe.stages[*].stageId");
    for (const [index, stage] of body.recipe.stages.entries()) {
      const path = `$.body.recipe.stages[${index}]`;
      if (stage.ordinal !== index) invalid(`${path}.ordinal`, "stage ordinals must be contiguous and zero-based");
      assertSortedUniqueStrings(stage.definitions.map((definition) => definition.name), `${path}.definitions[*].name`);
      assertSortedUniqueStrings(stage.targets, `${path}.targets`);
      const allowedPlaceholders: ReadonlySet<string> = index === 0
        ? new Set(["@EMSDK@", "@PREFIX_MAP_FLAGS@"])
        : new Set([
            "@BUILD_EVIDENCE@",
            "@EXTRACTOR_SOURCE@",
            "@NATIVE_BUILD@",
            "@PREFIX_MAP_FLAGS@",
          ]);
      for (const [definitionIndex, definition] of stage.definitions.entries()) {
        if (!/^[A-Za-z][A-Za-z0-9_]*$/u.test(definition.name)) {
          invalid(`${path}.definitions[${definitionIndex}].name`, "definition name has an unsafe format");
        }
        assertPlaceholders(definition.value, `${path}.definitions[${definitionIndex}].value`, allowedPlaceholders);
      }
      for (const [flagIndex, flag] of stage.compilerFlags.entries()) {
        assertPlaceholders(flag, `${path}.compilerFlags[${flagIndex}]`, allowedPlaceholders);
      }
      if ("linkerFlags" in stage) {
        for (const [flagIndex, flag] of stage.linkerFlags.entries()) {
          assertPlaceholders(flag, `${path}.linkerFlags[${flagIndex}]`, allowedPlaceholders);
        }
      }
    }

    assertSortedUniqueStrings(
      body.recipe.extractorLinkPolicy.selectedClangLibraries,
      "$.body.recipe.extractorLinkPolicy.selectedClangLibraries",
    );
    assertUniqueStrings(
      body.recipe.extractorLinkPolicy.prohibitedComponents,
      "$.body.recipe.extractorLinkPolicy.prohibitedComponents",
    );
    const allocatorInterception =
      body.recipe.extractorLinkPolicy.allocatorInterceptionPolicy;
    assertSortedUniqueStrings(
      allocatorInterception.exactEntrypoints,
      "$.body.recipe.extractorLinkPolicy.allocatorInterceptionPolicy.exactEntrypoints",
    );
    assertSortedUniqueStrings(
      allocatorInterception.forbiddenEntrypoints,
      "$.body.recipe.extractorLinkPolicy.allocatorInterceptionPolicy.forbiddenEntrypoints",
    );
    if (allocatorInterception.directBypassReferences !==
          "forbidden-outside-BrowserGradCppCuteMetrics.cpp" ||
        allocatorInterception.observedCallGraph !== "detached-evidence-required") {
      invalid(
        "$.body.recipe.extractorLinkPolicy.allocatorInterceptionPolicy",
        "allocator interception must stay source-closed and observed-call-graph blocked",
      );
    }

    const outputs = body.recipe.distributedOutputPlan.outputs;
    const outputPaths = outputs.map((output) => output.path);
    assertSortedUniqueStrings(outputPaths, "$.body.recipe.distributedOutputPlan.outputs[*].path");
    const outputByPath = new Map<string, (typeof outputs)[number]>();
    for (const [index, output] of outputs.entries()) {
      const path = `$.body.recipe.distributedOutputPlan.outputs[${index}]`;
      assertSafeRelativePath(output.path, `${path}.path`);
      if (output.reproducibilityClass !== "deterministic-subject" &&
          output.reproducibilityClass !== "detached-evidence") {
        invalid(`${path}.reproducibilityClass`, "unknown reproducibility class");
      }
      outputByPath.set(output.path, output);
    }
    const detachedOutputs = outputs.filter((output) => output.reproducibilityClass === "detached-evidence");
    if (detachedOutputs.length !== 1 ||
        detachedOutputs[0]?.path !== "assets/browsergrad-cpp-cute/build-provenance.dsse.json" ||
        detachedOutputs[0].role !== "detached-build-provenance") {
      invalid(
        "$.body.recipe.distributedOutputPlan.outputs",
        "only the detached build-provenance envelope may vary across clean builds",
      );
    }
    const runtimeAbiOutput = outputByPath.get(body.runtimeAbiResource.outputPath);
    if (runtimeAbiOutput?.role !== "runtime-abi-manifest" ||
        runtimeAbiOutput.mediaType !== body.runtimeAbiResource.mediaType ||
        runtimeAbiOutput.reproducibilityClass !== "deterministic-subject") {
      invalid(
        "$.body.runtimeAbiResource.outputPath",
        "runtime-ABI resource must bind one exact deterministic distribution output",
      );
    }
    for (const [index, expected] of REQUIRED_BROWSER_ASSET_OUTPUTS.entries()) {
      const output = outputByPath.get(expected.path);
      if (output?.role !== expected.role ||
          output.mediaType !== expected.mediaType ||
          output.reproducibilityClass !== "deterministic-subject") {
        invalid(
          `$.body.recipe.distributedOutputPlan.outputs[asset:${index}]`,
          "every required browser asset must have one exact deterministic distribution output with its build-selected media type",
        );
      }
    }

    assertSortedUniqueStrings(
      body.notices.approvedComponents.map((component) => component.componentId),
      "$.body.notices.approvedComponents[*].componentId",
    );
    const noticeOutputPaths: string[] = [];
    for (const [index, component] of body.notices.approvedComponents.entries()) {
      const path = `$.body.notices.approvedComponents[${index}]`;
      assertSafeRelativePath(component.sourcePath, `${path}.sourcePath`);
      assertSafeRelativePath(component.noticeOutputPath, `${path}.noticeOutputPath`);
      assertSha256(component.noticeSha256, `${path}.noticeSha256`);
      assertWireU64(component.noticeByteLength, `${path}.noticeByteLength`);
      assertSortedUniqueStrings(component.appliesTo, `${path}.appliesTo`);
      noticeOutputPaths.push(component.noticeOutputPath);
      const output = outputByPath.get(component.noticeOutputPath);
      if (output?.role !== "component-license" || output.mediaType !== "text/plain" ||
          output.reproducibilityClass !== "deterministic-subject") {
        invalid(`${path}.noticeOutputPath`, "approved notice must name one deterministic component-license output");
      }
    }
    assertUniqueStrings(noticeOutputPaths, "$.body.notices.approvedComponents[*].noticeOutputPath");
    const componentLicenseOutputs = outputs
      .filter((output) => output.role === "component-license")
      .map((output) => output.path)
      .sort(compareStrings);
    if (!equalStrings([...noticeOutputPaths].sort(compareStrings), componentLicenseOutputs)) {
      invalid(
        "$.body.recipe.distributedOutputPlan.outputs",
        "component-license outputs must exactly equal approved notice output paths",
      );
    }

    assertSortedUniqueStrings(
      body.notices.unresolvedComponents.map((component) => component.componentId),
      "$.body.notices.unresolvedComponents[*].componentId",
    );
    assertSortedUniqueStrings(
      body.unresolvedBuildInputs.map((input) => input.blockerId),
      "$.body.unresolvedBuildInputs[*].blockerId",
    );
    assertSortedUniqueStrings(body.releasePolicy.blockerIds, "$.body.releasePolicy.blockerIds");
    assertSortedUniqueStrings(
      body.releasePolicy.requiredExternalAuthorities,
      "$.body.releasePolicy.requiredExternalAuthorities",
    );
    assertSortedUniqueStrings(
      body.releasePolicy.requiredDetachedEvidence,
      "$.body.releasePolicy.requiredDetachedEvidence",
    );
    verifyReleasePolicy(body, deriveReleaseBlockerIds(body));
  } catch (cause) {
    if (cause instanceof CppCuteBrowserBuildInputLockError) throw cause;
    invalid("$.body", "build-input lock body violates required invariants", { cause });
  }
}

function assertSortedUniqueStrings(values: readonly string[], path: string): void {
  assertUniqueStrings(values, path);
  for (let index = 1; index < values.length; index += 1) {
    if (compareStrings(values[index - 1] ?? "", values[index] ?? "") >= 0) {
      invalid(path, "values must be strictly sorted by canonical alphanumeric token order");
    }
  }
}

function assertUniqueStrings(values: readonly string[], path: string): void {
  const seen = new Set<string>();
  for (const value of values) {
    if (typeof value !== "string" || value.length === 0) invalid(path, "values must be nonempty strings");
    if (seen.has(value)) invalid(path, `duplicate value ${JSON.stringify(value)}`);
    seen.add(value);
  }
}

function compareStrings(left: string, right: string): number {
  const leftTokens = left.toLowerCase().match(/[a-z0-9]+/gu) ?? [];
  const rightTokens = right.toLowerCase().match(/[a-z0-9]+/gu) ?? [];
  const length = Math.min(leftTokens.length, rightTokens.length);
  for (let index = 0; index < length; index += 1) {
    const leftToken = leftTokens[index] ?? "";
    const rightToken = rightTokens[index] ?? "";
    if (leftToken < rightToken) return -1;
    if (leftToken > rightToken) return 1;
  }
  if (leftTokens.length !== rightTokens.length) return leftTokens.length - rightTokens.length;
  return left < right ? -1 : left > right ? 1 : 0;
}

function assertSha1(value: unknown, path: string): void {
  if (typeof value !== "string" || !SHA1_HEX.test(value)) invalid(path, "expected lowercase SHA-1 hex");
}

function assertSha256(value: unknown, path: string): void {
  if (typeof value !== "string" || !SHA256_HEX.test(value)) invalid(path, "expected lowercase SHA-256 hex");
}

function assertSha256Digest(value: unknown, path: string): void {
  if (typeof value !== "string" || !SHA256_DIGEST.test(value)) {
    invalid(path, "expected sha256:<lowercase-hex> OCI digest");
  }
}

function assertWireU64(value: unknown, path: string): void {
  if (typeof value !== "string" || !WIRE_U64.test(value)) invalid(path, "expected canonical WireU64 decimal string");
  if (BigInt(value) > MAX_U64) invalid(path, "WireU64 exceeds unsigned 64-bit range");
}

function assertHttpsUrl(value: unknown, path: string): void {
  if (typeof value !== "string" || value.length > 2_048) invalid(path, "expected bounded HTTPS URL");
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch (cause) {
    invalid(path, "expected valid absolute HTTPS URL", { cause });
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" || parsed.password !== "" || parsed.hash !== "") {
    invalid(path, "URL must use HTTPS without credentials or fragments");
  }
}

function assertSafeRelativePath(value: unknown, path: string): void {
  if (typeof value !== "string" || value.length > 512 || !SAFE_RELATIVE_PATH.test(value)) {
    invalid(path, "expected bounded portable relative path");
  }
  const segments = value.split("/");
  if (segments.some((segment) => segment === "" || segment === "." || segment === "..")) {
    invalid(path, "relative path must not contain empty, dot, or parent segments");
  }
}

function assertPlaceholders(value: unknown, path: string, allowed: ReadonlySet<string>): void {
  if (typeof value !== "string") invalid(path, "recipe value must be a string");
  const placeholders = value.match(PLACEHOLDER) ?? [];
  for (const placeholder of placeholders) {
    if (!allowed.has(placeholder)) invalid(path, `unknown or stage-invalid placeholder ${placeholder}`);
  }
  if (value.replace(PLACEHOLDER, "").includes("@")) invalid(path, "malformed recipe placeholder");
}

function deriveReleaseBlockerIds(body: CppCuteBrowserBuildInputLockBodyV1): readonly string[] {
  const blockers = new Set<string>();
  for (const unresolved of body.unresolvedBuildInputs) blockers.add(unresolved.blockerId);
  for (const unresolved of body.notices.unresolvedComponents) {
    if (unresolved.componentId === "cuda-toolkit-12.6.3-headers") {
      blockers.add("cuda-header-redistribution");
    } else if (unresolved.componentId === "linux-sysroot") {
      blockers.add("linux-sysroot-redistribution");
    } else {
      invalid("$.body.notices.unresolvedComponents", "unknown unresolved license component");
    }
  }
  if (body.notices.fileInventoryPolicy === "exact-distributed-file-to-notice-map-required") {
    blockers.add("distributed-file-license-manifest");
  }
  blockers.add("reproducible-build-evidence");
  blockers.add("observed-wasm-interface-evidence");
  return [...blockers].sort();
}

function verifyReleasePolicy(
  body: CppCuteBrowserBuildInputLockBodyV1,
  derivedBlockerIds: readonly string[],
): void {
  if (body.releasePolicy.decision !== "blocked") {
    invalid("$.body.releasePolicy.decision", "input-only lock must remain blocked for release");
  }
  if (!equalStrings(body.releasePolicy.blockerIds, derivedBlockerIds)) {
    invalid("$.body.releasePolicy.blockerIds", "release blockers must equal blockers derived from unresolved inputs and evidence");
  }
  if (body.scope.outputIdentity !== "not-authorized" ||
      body.scope.reproducibility !== "detached-evidence-required" ||
      body.scope.producerTrust !== "detached-signed-attestation-required") {
    invalid("$.body.scope", "input lock must not claim output, reproducibility, or producer authority");
  }
  if (body.scope.dockerUse !== "pinned-build-time-only" || body.scope.runtimeDocker !== "forbidden") {
    invalid("$.body.scope", "Docker must remain a pinned build-time-only input");
  }
}

function storedLock(prepared: PreparedCppCuteBrowserBuildInputLock): StoredCppCuteBrowserBuildInputLock {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = PREPARED_LOCKS.get(prepared as object);
  if (stored === undefined) unverified();
  return stored;
}

function snapshotBytes(value: unknown): Uint8Array {
  let inspected;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid("$bytes", "build-input lock must be an unshared plain Uint8Array", { cause });
  }
  if (inspected.byteLength === 0) invalid("$bytes", "build-input lock bytes must be nonempty");
  if (inspected.byteLength > CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT) {
    resource("$bytes", `build-input lock exceeds ${CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_BYTE_LIMIT} bytes`);
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid("$bytes", "build-input lock bytes became unreadable while snapshotting", { cause });
  }
}

function normalizeOptions(options: PrepareCppCuteBrowserBuildInputLockOptions): AbortSignal | undefined {
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = Object.getPrototypeOf(options);
    descriptors = Object.getOwnPropertyDescriptors(options);
  } catch (cause) {
    invalid("$options", "options must be an inspectable plain object", { cause });
  }
  if (typeof options !== "object" || options === null || prototype !== Object.prototype) {
    invalid("$options", "options must be a plain object");
  }
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length > 1 || keys.some((key) => key !== "signal")) {
    invalid("$options", "options contain unknown fields");
  }
  const descriptor = descriptors.signal;
  if (descriptor !== undefined && (descriptor.enumerable !== true || !("value" in descriptor))) {
    invalid("$options.signal", "signal must be an enumerable data property");
  }
  const signal = descriptor?.value as unknown;
  if (signal !== undefined && !isAbortSignal(signal)) invalid("$options.signal", "signal must be an AbortSignal");
  return signal;
}

function isAbortSignal(value: unknown): value is AbortSignal {
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined) return false;
  try {
    return typeof ABORT_SIGNAL_ABORTED_GETTER.call(value) === "boolean";
  } catch {
    return false;
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  let aborted: unknown;
  try {
    aborted = ABORT_SIGNAL_ABORTED_GETTER?.call(signal);
  } catch (cause) {
    invalid("$options.signal", "signal is not a readable AbortSignal", { cause });
  }
  if (aborted === true) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-CANCELLED",
      "$options.signal",
      "build-input lock preparation was cancelled",
    );
  }
}

function closedObject(
  value: JsonValue,
  keys: readonly string[],
  path: string,
  requireAll: boolean,
): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) invalid(path, "expected object");
  const object = value as JsonObject;
  for (const key of Object.keys(object)) if (!keys.includes(key)) invalid(path, `unknown field ${key}`);
  if (requireAll) {
    for (const key of keys) {
      if (!Object.prototype.hasOwnProperty.call(object, key)) invalid(`${path}.${key}`, "required field is missing");
    }
  }
  return object;
}

function field(value: JsonObject, key: string, path: string): JsonValue {
  if (!Object.prototype.hasOwnProperty.call(value, key)) invalid(`${path}.${key}`, "required field is missing");
  return value[key] as JsonValue;
}

function boundedPattern(value: JsonValue, path: string, pattern: RegExp): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 2_048 || !pattern.test(value)) {
    invalid(path, "string does not match required closed format");
  }
  return value;
}

function literal<T extends string>(value: JsonValue, expected: T, path: string): asserts value is T {
  if (value !== expected) invalid(path, `must equal ${JSON.stringify(expected)}`);
}

function canonicalResourceBytes(value: JsonValue): Uint8Array {
  try {
    return canonicalJsonBytes(value, {
      limits: CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$", "canonical build-input lock exceeds fixed limits", { cause });
    invalid("$", "build-input lock cannot be canonically encoded", { cause });
  }
}

async function hashJson(value: JsonValue, path: string): Promise<string> {
  try {
    return await hashCanonicalJson(value, {
      limits: CPP_CUTE_BROWSER_BUILD_INPUT_LOCK_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isHashUnavailable(cause)) hashUnavailable(path, cause);
    if (isSchemaResourceLimit(cause)) resource(path, "hash projection exceeds fixed limits", { cause });
    invalid(path, "hash projection is invalid", { cause });
  }
}

async function hashBytes(value: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(value);
  } catch (cause) {
    if (isHashUnavailable(cause)) hashUnavailable(path, cause);
    invalid(path, "SHA-256 calculation failed", { cause });
  }
}

function isHashUnavailable(cause: unknown): boolean {
  return cause instanceof Error && /Web Crypto|crypto\.subtle|SHA-256 unavailable/iu.test(cause.message);
}

function isSchemaResourceLimit(cause: unknown): boolean {
  return cause instanceof SemanticSchemaError &&
    cause.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function equalStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-INVALID", path, message, options);
}

function unsupported(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-UNSUPPORTED-VERSION", path, message);
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-RESOURCE-LIMIT", path, message, options);
}

function hashMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-HASH-MISMATCH", path, message);
}

function hashUnavailable(path: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-HASH-UNAVAILABLE",
    path,
    "SHA-256 is unavailable",
    { cause },
  );
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-UNVERIFIED",
    "$prepared",
    "build-input lock authority was not created by this module instance",
  );
}

function fail(
  code: CppCuteBrowserBuildInputLockErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserBuildInputLockError(code, path, message, options);
}

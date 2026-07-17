import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  assertJsonValue,
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
  encodeWireU64,
  hashCanonicalJson,
  isJsonObject,
  sha256Hex,
  wireIntegerToBigInt,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
  cppCuteDiagnosticNormalizationResourceBytes,
} from "./cpp_cute_diagnostic_normalization.js";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
  cppCuteSemanticAdapterManifestResourceBytes,
} from "./cpp_cute_semantic_adapter_manifest.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type CppCuteFrontendBrowserAssetLimits,
  type CppCuteFrontendCompatibilityProfile,
  type CppCuteFrontendLanguageProfile,
  type CppCuteFrontendTargetProfile,
  type CppCuteFrontendToolchainProfile,
  type CppCuteFrontendVirtualFileSystemProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";

export const CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-asset-manifest";
export const CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR = 1;
export const CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR = 2;
export const CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT = 256 * 1024;

const MANIFEST_ID = /^bg\.cpp\.browser-assets\.sha256\.[0-9a-f]{64}$/u;
const ASSET_ID = /^[a-z][a-z0-9._-]*$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const BUILD_PROVENANCE_ID = /^bg\.build-provenance\.sha256\.[0-9a-f]{64}$/u;
const SAME_ORIGIN_ROOT_RELATIVE_URL = /^\/(?:[A-Za-z0-9._~-]+\/)*[A-Za-z0-9._~-]+$/u;
const MAX_ASSETS = 256;
const HARD_MAX_ASSET_COMPRESSED_BYTES = 1024n * 1024n * 1024n;
const HARD_MAX_ASSET_UNPACKED_BYTES = 4n * 1024n * 1024n * 1024n;
const HARD_MAX_TOTAL_COMPRESSED_BYTES = 2n * 1024n * 1024n * 1024n;
const HARD_MAX_TOTAL_UNPACKED_BYTES = 8n * 1024n * 1024n * 1024n;
const RUNTIME_ABI_RESOURCE_BYTE_LENGTH = cppCuteBrowserRuntimeAbiManifestResourceBytes().byteLength;
const DIAGNOSTIC_NORMALIZATION_RESOURCE_BYTE_LENGTH =
  cppCuteDiagnosticNormalizationResourceBytes().byteLength;
const SEMANTIC_ADAPTER_RESOURCE_BYTE_LENGTH =
  cppCuteSemanticAdapterManifestResourceBytes().byteLength;
const TEXT_ENCODER = new TextEncoder();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;
const PREPARED_MANIFESTS = new WeakMap<object, StoredCppCuteBrowserAssetManifest>();

export const CPP_CUTE_BROWSER_ASSET_MANIFEST_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT,
  maxDepth: 16,
  maxNodes: 8_192,
  maxStringBytes: 128 * 1024,
  maxArrayLength: MAX_ASSETS,
  maxObjectProperties: 64,
  maxRank: 1,
  maxIntegerBits: 64,
  maxArithmeticOperations: 8_192,
});

export interface CppCuteBrowserAssetManifestVersionV1 extends JsonObject {
  readonly major: typeof CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR;
  readonly minor: typeof CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR;
}

/** Requested source/toolchain ABI. This is a binding, not proof of WASM/native parity. */
export interface CppCuteBrowserSourceAbiV1 extends JsonObject {
  readonly profileId: string;
  readonly language: CppCuteFrontendLanguageProfile;
  readonly target: CppCuteFrontendTargetProfile;
  readonly toolchain: CppCuteFrontendToolchainProfile;
  readonly virtualFileSystem: CppCuteFrontendVirtualFileSystemProfile;
  readonly compatibility: CppCuteFrontendCompatibilityProfile;
}

export interface CppCuteBrowserAssetTotalsV1 extends JsonObject {
  readonly compressedByteLength: WireU64;
  readonly unpackedByteLength: WireU64;
  readonly fileContentByteLength: WireU64;
}

interface CppCuteBrowserAssetCommonV1 extends JsonObject {
  readonly assetId: string;
  readonly url: string;
  readonly urlPolicy: "same-origin-root-relative";
  readonly sha256: string;
  readonly byteLength: WireU64;
  readonly unpackedByteLength: WireU64;
  readonly buildProvenanceId: string;
}

export type CppCuteBrowserAssetV1 =
  | (CppCuteBrowserAssetCommonV1 & {
      readonly kind: "clang-extractor-wasm";
      readonly mediaType: "application/wasm";
      readonly compression: "identity";
      readonly sourceAbiSha256: string;
    })
  | (CppCuteBrowserAssetCommonV1 & {
      readonly kind: "compiler-resource-pack";
      readonly mediaType: "application/vnd.browsergrad.vfs-pack.v1";
      readonly compression: "identity";
      readonly includeRootId: string;
      readonly mountedVirtualRoot: string;
      readonly contentSetSha256: string;
      readonly fileContentByteLength: WireU64;
    })
  | (CppCuteBrowserAssetCommonV1 & {
      readonly kind: "dependency-header-pack";
      readonly mediaType: "application/vnd.browsergrad.vfs-pack.v1";
      readonly compression: "identity";
      readonly dependencyId: string;
      readonly includeRootId: string;
      readonly mountedVirtualRoot: string;
      readonly contentSetSha256: string;
      readonly fileContentByteLength: WireU64;
    })
  | (CppCuteBrowserAssetCommonV1 & {
      readonly kind: "semantic-adapter-manifest";
      readonly mediaType: "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json";
      readonly compression: "identity";
    })
  | (CppCuteBrowserAssetCommonV1 & {
      readonly kind: "diagnostic-normalization-manifest";
      readonly mediaType: "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json";
      readonly compression: "identity";
    })
  | (CppCuteBrowserAssetCommonV1 & {
      readonly kind: "runtime-abi-manifest";
      readonly mediaType: "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json";
      readonly compression: "identity";
      readonly runtimeAbiId: "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1";
      readonly runtimeAbiManifestId: typeof CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID;
    });

export interface CppCuteBrowserAssetManifestBodyV1 extends JsonObject {
  readonly profileHash: string;
  readonly sourceAbi: CppCuteBrowserSourceAbiV1;
  readonly sourceAbiSha256: string;
  readonly assetSetSha256: string;
  readonly dependencyIds: readonly string[];
  readonly buildProvenanceIds: readonly string[];
  readonly mountedVirtualRoots: readonly string[];
  readonly assets: readonly CppCuteBrowserAssetV1[];
  readonly totals: CppCuteBrowserAssetTotalsV1;
}

export interface CppCuteBrowserAssetManifestV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA;
  readonly version: CppCuteBrowserAssetManifestVersionV1;
  readonly manifestId: string;
  readonly body: CppCuteBrowserAssetManifestBodyV1;
}

declare const preparedCppCuteBrowserAssetManifestBrand: unique symbol;

/**
 * Opaque authority over exact canonical manifest bytes and profile bindings.
 * It does not claim that any asset was fetched, hashed, unpacked, or executed.
 */
export interface PreparedCppCuteBrowserAssetManifest {
  readonly [preparedCppCuteBrowserAssetManifestBrand]: true;
  readonly manifestId: string;
  readonly manifestSha256: string;
  readonly manifestByteLength: WireU64;
  readonly profileHash: string;
  readonly sourceAbiSha256: string;
  readonly assetSetSha256: string;
  readonly assetCount: number;
  readonly compressedByteLength: WireU64;
  readonly unpackedByteLength: WireU64;
}

export interface PreparedCppCuteBrowserAssetManifestRecord {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly manifest: CppCuteBrowserAssetManifestV1;
}

interface StoredCppCuteBrowserAssetManifest extends PreparedCppCuteBrowserAssetManifestRecord {
  readonly canonicalBytes: Uint8Array;
}

export interface PrepareCppCuteBrowserAssetManifestOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserAssetManifestErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-NONCANONICAL-BYTES"
  | "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-UNVERIFIED";

export class CppCuteBrowserAssetManifestError extends Error {
  constructor(
    readonly code: CppCuteBrowserAssetManifestErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserAssetManifestError";
  }
}

export async function prepareCppCuteBrowserAssetManifest(
  value: unknown,
  profile: PreparedCppCuteFrontendProfile,
  options: PrepareCppCuteBrowserAssetManifestOptions = {},
): Promise<PreparedCppCuteBrowserAssetManifest> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  const expectedSourceAbi = cppCuteBrowserSourceAbi(profile);
  const manifest = parseManifest(value, expectedSourceAbi, profileRecord.profileHash);
  throwIfAborted(signal);

  const sourceAbiSha256 = await hashJson({
    domain: "browsergrad.compiler.cpp-cute.browser-source-abi.v1",
    sourceAbi: expectedSourceAbi,
  }, "$.body.sourceAbiSha256");
  throwIfAborted(signal);
  if (manifest.body.sourceAbiSha256 !== sourceAbiSha256) {
    hashMismatch("$.body.sourceAbiSha256", `sourceAbiSha256 must equal ${sourceAbiSha256}`);
  }
  validateAssetCardinalities(manifest.body.assets);
  const assetSetSha256 = await deriveCppCuteBrowserAssetSetSha256(manifest.body);
  throwIfAborted(signal);
  if (manifest.body.assetSetSha256 !== assetSetSha256) {
    hashMismatch("$.body.assetSetSha256", `assetSetSha256 must equal ${assetSetSha256}`);
  }
  if (assetSetSha256 !== profileRecord.profile.deployment.assetSetSha256) {
    hashMismatch(
      "$.body.assetSetSha256",
      "asset set does not equal browser deployment profile asset-set lock",
    );
  }
  validateAssetProfileClosure(manifest.body, profileRecord.profile, sourceAbiSha256);
  validateResourceAccounting(
    manifest.body.assets,
    profileRecord.profile.deployment.assetLimits,
    manifest.body.totals,
  );

  const manifestId = await deriveCppCuteBrowserAssetManifestId(manifest.body);
  throwIfAborted(signal);
  if (manifest.manifestId !== manifestId) {
    hashMismatch("$.manifestId", `manifestId must equal ${manifestId}`);
  }

  const canonicalBytes = canonicalManifestBytes(manifest);
  if (canonicalBytes.byteLength > CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT) {
    resource("$", `canonical manifest exceeds ${CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT} bytes`);
  }
  const manifestSha256 = await hashBytes(canonicalBytes, "$bytes");
  throwIfAborted(signal);
  const prepared = Object.freeze({
    manifestId,
    manifestSha256,
    manifestByteLength: encodeWireU64(BigInt(canonicalBytes.byteLength)),
    profileHash: manifest.body.profileHash,
    sourceAbiSha256,
    assetSetSha256,
    assetCount: manifest.body.assets.length,
    compressedByteLength: manifest.body.totals.compressedByteLength,
    unpackedByteLength: manifest.body.totals.unpackedByteLength,
  }) as PreparedCppCuteBrowserAssetManifest;
  PREPARED_MANIFESTS.set(prepared, Object.freeze({
    profile,
    manifest,
    canonicalBytes: new Uint8Array(canonicalBytes),
  }));
  return prepared;
}

export async function decodeCppCuteBrowserAssetManifest(
  bytes: Uint8Array,
  profile: PreparedCppCuteFrontendProfile,
  options: PrepareCppCuteBrowserAssetManifestOptions = {},
): Promise<PreparedCppCuteBrowserAssetManifest> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  const snapshot = snapshotBytes(bytes);
  throwIfAborted(signal);
  let decoded: JsonValue;
  try {
    decoded = decodeWireJson(snapshot, { limits: CPP_CUTE_BROWSER_ASSET_MANIFEST_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$bytes", "manifest decoding exceeded fixed resource limits", { cause });
    invalid("$bytes", "manifest bytes are not strict JSON", { cause });
  }
  const prepared = await prepareCppCuteBrowserAssetManifest(
    decoded,
    profile,
    signal === undefined ? {} : { signal },
  );
  const canonical = canonicalCppCuteBrowserAssetManifestBytes(prepared);
  if (!equalBytes(snapshot, canonical)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-NONCANONICAL-BYTES",
      "$bytes",
      "manifest bytes must exactly equal canonical JSON bytes",
    );
  }
  return prepared;
}

export function unwrapPreparedCppCuteBrowserAssetManifest(
  prepared: PreparedCppCuteBrowserAssetManifest,
): PreparedCppCuteBrowserAssetManifestRecord {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = PREPARED_MANIFESTS.get(prepared as object);
  if (stored === undefined) unverified();
  return Object.freeze({ profile: stored.profile, manifest: stored.manifest });
}

export function canonicalCppCuteBrowserAssetManifestBytes(
  prepared: PreparedCppCuteBrowserAssetManifest,
): Uint8Array {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = PREPARED_MANIFESTS.get(prepared as object);
  if (stored === undefined) unverified();
  return new Uint8Array(stored.canonicalBytes);
}

export function cppCuteBrowserSourceAbi(
  profile: PreparedCppCuteFrontendProfile,
): CppCuteBrowserSourceAbiV1 {
  const record = unwrapPreparedCppCuteBrowserFrontendProfile(profile);
  const sourceAbi: CppCuteBrowserSourceAbiV1 = {
    profileId: record.profile.profileId,
    language: record.profile.language,
    target: record.profile.target,
    toolchain: record.profile.toolchain,
    virtualFileSystem: record.profile.virtualFileSystem,
    compatibility: record.profile.compatibility,
  };
  return deepFreezeJson(sourceAbi);
}

/**
 * Hashes exact acquire-and-mount identities. Profile/manifest identity, totals,
 * and policy ceilings are excluded to avoid a profile-manifest hash cycle.
 */
export async function deriveCppCuteBrowserAssetSetSha256(
  body: Pick<
    CppCuteBrowserAssetManifestBodyV1,
    | "sourceAbiSha256"
    | "dependencyIds"
    | "buildProvenanceIds"
    | "mountedVirtualRoots"
    | "assets"
  >,
): Promise<string> {
  try {
    assertJsonValue(body, { limits: CPP_CUTE_BROWSER_ASSET_MANIFEST_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$.body", "asset-set projection exceeds fixed resource limits", { cause });
    invalid("$.body", "asset-set projection must be a canonical JSON tree", { cause });
  }
  return hashJson({
    domain: "browsergrad.compiler.cpp-cute.browser-asset-set.v1",
    sourceAbiSha256: body.sourceAbiSha256,
    dependencyIds: body.dependencyIds,
    buildProvenanceIds: body.buildProvenanceIds,
    mountedVirtualRoots: body.mountedVirtualRoots,
    assets: body.assets,
  }, "$.body.assetSetSha256");
}

/** Deterministic construction helper; returned ID alone grants no authority. */
export async function deriveCppCuteBrowserAssetManifestId(
  body: CppCuteBrowserAssetManifestBodyV1,
): Promise<string> {
  try {
    assertJsonValue(body, { limits: CPP_CUTE_BROWSER_ASSET_MANIFEST_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$.body", "manifest body exceeds fixed resource limits", { cause });
    invalid("$.body", "manifest body must be a canonical JSON tree", { cause });
  }
  const digest = await hashJson({
    domain: "browsergrad.compiler.cpp-cute.browser-asset-manifest-id.v1",
    body,
  }, "$.manifestId");
  return `bg.cpp.browser-assets.sha256.${digest}`;
}

function parseManifest(
  value: unknown,
  expectedSourceAbi: CppCuteBrowserSourceAbiV1,
  expectedProfileHash: string,
): CppCuteBrowserAssetManifestV1 {
  try {
    assertJsonValue(value, { limits: CPP_CUTE_BROWSER_ASSET_MANIFEST_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$", "manifest exceeds fixed resource limits", { cause });
    invalid("$", "manifest must be an accessor-free canonical JSON tree", { cause });
  }
  const object = closedObject(value, ["schema", "version", "manifestId", "body"], "$", true);
  literal(field(object, "schema", "$"), CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA, "$.schema");
  const manifestId = boundedPattern(field(object, "manifestId", "$"), "$.manifestId", MANIFEST_ID);
  const version = parseVersion(field(object, "version", "$"));
  const body = parseBody(field(object, "body", "$"), expectedSourceAbi, expectedProfileHash);
  return deepFreezeJson({
    schema: CPP_CUTE_BROWSER_ASSET_MANIFEST_SCHEMA,
    version,
    manifestId,
    body,
  });
}

function parseVersion(value: JsonValue): CppCuteBrowserAssetManifestVersionV1 {
  const object = closedObject(value, ["major", "minor"], "$.version", true);
  if (object.major !== CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-UNSUPPORTED-VERSION",
      "$.version.major",
      `reader supports manifest major ${CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR}`,
    );
  }
  if (object.minor !== CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-UNSUPPORTED-VERSION",
      "$.version.minor",
      `closed reader supports ${CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR}.${CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR} only`,
    );
  }
  return { major: CPP_CUTE_BROWSER_ASSET_MANIFEST_MAJOR, minor: CPP_CUTE_BROWSER_ASSET_MANIFEST_MINOR };
}

function parseBody(
  value: JsonValue,
  expectedSourceAbi: CppCuteBrowserSourceAbiV1,
  expectedProfileHash: string,
): CppCuteBrowserAssetManifestBodyV1 {
  const path = "$.body";
  const object = closedObject(value, [
    "profileHash",
    "sourceAbi",
    "sourceAbiSha256",
    "assetSetSha256",
    "dependencyIds",
    "buildProvenanceIds",
    "mountedVirtualRoots",
    "assets",
    "totals",
  ], path, true);
  const profileHash = sha256(field(object, "profileHash", path), `${path}.profileHash`);
  if (profileHash !== expectedProfileHash) invalid(`${path}.profileHash`, "manifest names a different prepared profile");
  const sourceAbiValue = field(object, "sourceAbi", path);
  if (!isJsonObject(sourceAbiValue)) invalid(`${path}.sourceAbi`, "expected source ABI object");
  if (canonicalText(sourceAbiValue) !== canonicalText(expectedSourceAbi)) {
    invalid(`${path}.sourceAbi`, "source ABI metadata does not exactly match prepared profile");
  }
  const sourceAbiSha256 = sha256(field(object, "sourceAbiSha256", path), `${path}.sourceAbiSha256`);
  const assetSetSha256 = sha256(field(object, "assetSetSha256", path), `${path}.assetSetSha256`);
  const dependencyIds = sortedIdentifierArray(
    field(object, "dependencyIds", path),
    `${path}.dependencyIds`,
    /^[a-z][a-z0-9._-]*$/u,
  );
  const buildProvenanceIds = sortedIdentifierArray(
    field(object, "buildProvenanceIds", path),
    `${path}.buildProvenanceIds`,
    BUILD_PROVENANCE_ID,
  );
  if (buildProvenanceIds.length === 0) invalid(`${path}.buildProvenanceIds`, "at least one build provenance ID is required");
  const mountedVirtualRoots = sortedStringArray(
    field(object, "mountedVirtualRoots", path),
    `${path}.mountedVirtualRoots`,
  );
  mountedVirtualRoots.forEach((root, index) => virtualPath(root, `${path}.mountedVirtualRoots[${index}]`));
  const assets = array(field(object, "assets", path), `${path}.assets`).map((entry, index) =>
    parseAsset(entry, `${path}.assets[${index}]`));
  if (assets.length === 0) invalid(`${path}.assets`, "asset manifest must not be empty");
  requireSortedUnique(assets.map((asset) => asset.assetId), `${path}.assets`);
  requireUnique(assets.map((asset) => asset.url), `${path}.assets`, "asset URLs must be unique");
  const totals = parseTotals(field(object, "totals", path));

  const referencedBuildIds = [...new Set(assets.map((asset) => asset.buildProvenanceId))].sort();
  if (!equalStrings(referencedBuildIds, buildProvenanceIds)) {
    invalid(`${path}.buildProvenanceIds`, "build provenance IDs must exactly equal asset references");
  }
  const actualRoots = assets.flatMap((asset): string[] =>
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack"
      ? [asset.mountedVirtualRoot]
      : []).sort();
  requireSortedUnique(actualRoots, `${path}.assets`);
  if (!equalStrings(actualRoots, mountedVirtualRoots)) {
    invalid(`${path}.mountedVirtualRoots`, "mounted roots must exactly equal sorted asset mount roots");
  }
  return {
    profileHash,
    sourceAbi: expectedSourceAbi,
    sourceAbiSha256,
    assetSetSha256,
    dependencyIds,
    buildProvenanceIds,
    mountedVirtualRoots,
    assets,
    totals,
  };
}

function parseTotals(value: JsonValue): CppCuteBrowserAssetTotalsV1 {
  const path = "$.body.totals";
  const object = closedObject(
    value,
    ["compressedByteLength", "unpackedByteLength", "fileContentByteLength"],
    path,
    true,
  );
  return {
    compressedByteLength: wirePositive(field(object, "compressedByteLength", path), `${path}.compressedByteLength`),
    unpackedByteLength: wirePositive(field(object, "unpackedByteLength", path), `${path}.unpackedByteLength`),
    fileContentByteLength: wirePositive(
      field(object, "fileContentByteLength", path),
      `${path}.fileContentByteLength`,
    ),
  };
}

function parseAsset(value: JsonValue, path: string): CppCuteBrowserAssetV1 {
  const base = closedObject(value, [
    "assetId", "kind", "url", "urlPolicy", "sha256", "byteLength", "unpackedByteLength",
    "mediaType", "compression", "buildProvenanceId", "sourceAbiSha256", "dependencyId",
    "includeRootId", "mountedVirtualRoot", "contentSetSha256",
    "fileContentByteLength", "runtimeAbiId", "runtimeAbiManifestId",
  ], path, false);
  const kind = boundedString(field(base, "kind", path), `${path}.kind`, 64);
  const common = {
    assetId: boundedPattern(field(base, "assetId", path), `${path}.assetId`, ASSET_ID),
    url: sameOriginRootRelativeUrl(field(base, "url", path), `${path}.url`),
    urlPolicy: exactString(field(base, "urlPolicy", path), "same-origin-root-relative", `${path}.urlPolicy`),
    sha256: sha256(field(base, "sha256", path), `${path}.sha256`),
    byteLength: wirePositive(field(base, "byteLength", path), `${path}.byteLength`),
    unpackedByteLength: wirePositive(field(base, "unpackedByteLength", path), `${path}.unpackedByteLength`),
    buildProvenanceId: boundedPattern(
      field(base, "buildProvenanceId", path),
      `${path}.buildProvenanceId`,
      BUILD_PROVENANCE_ID,
    ),
  };

  if (kind === "clang-extractor-wasm") {
    requireExactFields(base, [
      "assetId", "kind", "url", "urlPolicy", "sha256", "byteLength", "unpackedByteLength",
      "mediaType", "compression", "buildProvenanceId", "sourceAbiSha256",
    ], path);
    exactString(field(base, "mediaType", path), "application/wasm", `${path}.mediaType`);
    exactString(field(base, "compression", path), "identity", `${path}.compression`);
    requireIdentityLength(common, path);
    return {
      ...common,
      kind,
      mediaType: "application/wasm",
      compression: "identity",
      sourceAbiSha256: sha256(field(base, "sourceAbiSha256", path), `${path}.sourceAbiSha256`),
    };
  }
  if (kind === "compiler-resource-pack") {
    requireExactFields(base, [
      "assetId", "kind", "url", "urlPolicy", "sha256", "byteLength", "unpackedByteLength",
      "mediaType", "compression", "buildProvenanceId", "includeRootId", "mountedVirtualRoot",
      "contentSetSha256", "fileContentByteLength",
    ], path);
    exactString(
      field(base, "mediaType", path),
      "application/vnd.browsergrad.vfs-pack.v1",
      `${path}.mediaType`,
    );
    exactString(field(base, "compression", path), "identity", `${path}.compression`);
    requireIdentityLength(common, path);
    const fileContentByteLength = wirePositive(
      field(base, "fileContentByteLength", path),
      `${path}.fileContentByteLength`,
    );
    requirePackLength(common, fileContentByteLength, path);
    return {
      ...common,
      kind,
      mediaType: "application/vnd.browsergrad.vfs-pack.v1",
      compression: "identity",
      includeRootId: boundedPattern(field(base, "includeRootId", path), `${path}.includeRootId`, ASSET_ID),
      mountedVirtualRoot: virtualPathValue(field(base, "mountedVirtualRoot", path), `${path}.mountedVirtualRoot`),
      contentSetSha256: sha256(field(base, "contentSetSha256", path), `${path}.contentSetSha256`),
      fileContentByteLength,
    };
  }
  if (kind === "dependency-header-pack") {
    requireExactFields(base, [
      "assetId", "kind", "url", "urlPolicy", "sha256", "byteLength", "unpackedByteLength",
      "mediaType", "compression", "buildProvenanceId", "dependencyId", "includeRootId",
      "mountedVirtualRoot", "contentSetSha256", "fileContentByteLength",
    ], path);
    exactString(
      field(base, "mediaType", path),
      "application/vnd.browsergrad.vfs-pack.v1",
      `${path}.mediaType`,
    );
    exactString(field(base, "compression", path), "identity", `${path}.compression`);
    requireIdentityLength(common, path);
    const fileContentByteLength = wirePositive(
      field(base, "fileContentByteLength", path),
      `${path}.fileContentByteLength`,
    );
    requirePackLength(common, fileContentByteLength, path);
    return {
      ...common,
      kind,
      mediaType: "application/vnd.browsergrad.vfs-pack.v1",
      compression: "identity",
      dependencyId: boundedPattern(field(base, "dependencyId", path), `${path}.dependencyId`, ASSET_ID),
      includeRootId: boundedPattern(field(base, "includeRootId", path), `${path}.includeRootId`, ASSET_ID),
      mountedVirtualRoot: virtualPathValue(field(base, "mountedVirtualRoot", path), `${path}.mountedVirtualRoot`),
      contentSetSha256: sha256(field(base, "contentSetSha256", path), `${path}.contentSetSha256`),
      fileContentByteLength,
    };
  }
  if (kind === "semantic-adapter-manifest") {
    requireExactFields(base, [
      "assetId", "kind", "url", "urlPolicy", "sha256", "byteLength", "unpackedByteLength",
      "mediaType", "compression", "buildProvenanceId",
    ], path);
    exactString(
      field(base, "mediaType", path),
      "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      `${path}.mediaType`,
    );
    exactString(field(base, "compression", path), "identity", `${path}.compression`);
    requireIdentityLength(common, path);
    return {
      ...common,
      kind,
      mediaType: "application/vnd.browsergrad.cpp-cute.semantic-adapter.v1+json",
      compression: "identity",
    };
  }
  if (kind === "diagnostic-normalization-manifest") {
    requireExactFields(base, [
      "assetId", "kind", "url", "urlPolicy", "sha256", "byteLength", "unpackedByteLength",
      "mediaType", "compression", "buildProvenanceId",
    ], path);
    exactString(
      field(base, "mediaType", path),
      "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json",
      `${path}.mediaType`,
    );
    exactString(field(base, "compression", path), "identity", `${path}.compression`);
    requireIdentityLength(common, path);
    return {
      ...common,
      kind,
      mediaType: "application/vnd.browsergrad.cpp-cute.diagnostic-normalization.v1+json",
      compression: "identity",
    };
  }
  if (kind === "runtime-abi-manifest") {
    requireExactFields(base, [
      "assetId", "kind", "url", "urlPolicy", "sha256", "byteLength", "unpackedByteLength",
      "mediaType", "compression", "buildProvenanceId", "runtimeAbiId", "runtimeAbiManifestId",
    ], path);
    exactString(
      field(base, "mediaType", path),
      "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      `${path}.mediaType`,
    );
    exactString(field(base, "compression", path), "identity", `${path}.compression`);
    requireIdentityLength(common, path);
    return {
      ...common,
      kind,
      mediaType: "application/vnd.browsergrad.cpp-cute.runtime-abi-manifest.v1+json",
      compression: "identity",
      runtimeAbiId: exactString(
        field(base, "runtimeAbiId", path),
        "browsergrad.compiler.cpp-cute.clang-wasm-runtime@1",
        `${path}.runtimeAbiId`,
      ),
      runtimeAbiManifestId: exactString(
        field(base, "runtimeAbiManifestId", path),
        CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID,
        `${path}.runtimeAbiManifestId`,
      ),
    };
  }
  invalid(`${path}.kind`, `unknown browser asset kind ${JSON.stringify(kind)}`);
}

function validateResourceAccounting(
  assets: readonly CppCuteBrowserAssetV1[],
  profileLimits: CppCuteFrontendBrowserAssetLimits,
  totals: CppCuteBrowserAssetTotalsV1,
): void {
  const maxAssets = Math.min(profileLimits.maxAssets, MAX_ASSETS);
  const maxCompressed = minBigInt(
    BigInt(profileLimits.maxAssetCompressedByteLength),
    HARD_MAX_ASSET_COMPRESSED_BYTES,
  );
  const maxUnpacked = minBigInt(
    BigInt(profileLimits.maxAssetUnpackedByteLength),
    HARD_MAX_ASSET_UNPACKED_BYTES,
  );
  const maxTotalCompressed = minBigInt(
    BigInt(profileLimits.maxTotalCompressedByteLength),
    HARD_MAX_TOTAL_COMPRESSED_BYTES,
  );
  const maxTotalUnpacked = minBigInt(
    BigInt(profileLimits.maxTotalUnpackedByteLength),
    HARD_MAX_TOTAL_UNPACKED_BYTES,
  );
  const maxFileContent = BigInt(profileLimits.maxAssetFileContentByteLength);
  const maxTotalFileContent = BigInt(profileLimits.maxTotalFileContentByteLength);
  if (assets.length > maxAssets) resource("$.body.assets", "asset count exceeds browser profile ceiling");
  let compressed = 0n;
  let unpacked = 0n;
  let fileContent = 0n;
  for (const [index, asset] of assets.entries()) {
    const assetCompressed = wireIntegerToBigInt(asset.byteLength);
    const assetUnpacked = wireIntegerToBigInt(asset.unpackedByteLength);
    if (assetCompressed > maxCompressed) resource(`$.body.assets[${index}].byteLength`, "asset exceeds compressed ceiling");
    if (assetUnpacked > maxUnpacked) resource(`$.body.assets[${index}].unpackedByteLength`, "asset exceeds unpacked ceiling");
    compressed += assetCompressed;
    unpacked += assetUnpacked;
    if (asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack") {
      const assetFileContent = wireIntegerToBigInt(asset.fileContentByteLength);
      if (assetFileContent > maxFileContent) {
        resource(`$.body.assets[${index}].fileContentByteLength`, "asset exceeds file-content ceiling");
      }
      fileContent += assetFileContent;
    }
  }
  if (compressed > maxTotalCompressed) {
    resource("$.body.totals.compressedByteLength", "compressed total exceeds browser profile ceiling");
  }
  if (unpacked > maxTotalUnpacked) {
    resource("$.body.totals.unpackedByteLength", "unpacked total exceeds browser profile ceiling");
  }
  if (fileContent > maxTotalFileContent) {
    resource("$.body.totals.fileContentByteLength", "file-content total exceeds browser profile ceiling");
  }
  if (compressed !== wireIntegerToBigInt(totals.compressedByteLength)) {
    invalid("$.body.totals.compressedByteLength", "compressed total does not equal asset byte lengths");
  }
  if (unpacked !== wireIntegerToBigInt(totals.unpackedByteLength)) {
    invalid("$.body.totals.unpackedByteLength", "unpacked total does not equal asset lengths");
  }
  if (fileContent !== wireIntegerToBigInt(totals.fileContentByteLength)) {
    invalid("$.body.totals.fileContentByteLength", "file-content total does not equal VFS pack file lengths");
  }
}

function validateAssetProfileClosure(
  body: CppCuteBrowserAssetManifestBodyV1,
  profile: ReturnType<typeof unwrapPreparedCppCuteBrowserFrontendProfile>["profile"],
  sourceAbiSha256: string,
): void {
  const wasm = body.assets.find((asset) => asset.kind === "clang-extractor-wasm");
  if (wasm?.kind !== "clang-extractor-wasm" ||
      wasm.sourceAbiSha256 !== sourceAbiSha256 ||
      wasm.sha256 !== profile.deployment.extractor.binarySha256) {
    hashMismatch("$.body.assets", "Clang WASM asset must bind exact extractor binary and source ABI hashes");
  }
  const adapter = body.assets.find((asset) => asset.kind === "semantic-adapter-manifest");
  if (adapter?.kind !== "semantic-adapter-manifest" ||
      adapter.sha256 !== CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256 ||
      adapter.sha256 !== profile.deployment.extractor.semanticAdapterManifestSha256 ||
      wireIntegerToBigInt(adapter.byteLength) !== BigInt(SEMANTIC_ADAPTER_RESOURCE_BYTE_LENGTH) ||
      wireIntegerToBigInt(adapter.unpackedByteLength) !== BigInt(SEMANTIC_ADAPTER_RESOURCE_BYTE_LENGTH)) {
    hashMismatch(
      "$.body.assets",
      "semantic-adapter asset must bind exact package canonical policy bytes",
    );
  }
  const diagnosticNormalization = body.assets.find(
    (asset) => asset.kind === "diagnostic-normalization-manifest",
  );
  if (diagnosticNormalization?.kind !== "diagnostic-normalization-manifest" ||
      diagnosticNormalization.sha256 !== CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256 ||
      diagnosticNormalization.sha256 !== profile.language.diagnostics.normalizationManifestSha256 ||
      wireIntegerToBigInt(diagnosticNormalization.byteLength) !==
        BigInt(DIAGNOSTIC_NORMALIZATION_RESOURCE_BYTE_LENGTH) ||
      wireIntegerToBigInt(diagnosticNormalization.unpackedByteLength) !==
        BigInt(DIAGNOSTIC_NORMALIZATION_RESOURCE_BYTE_LENGTH)) {
    hashMismatch(
      "$.body.assets",
      "diagnostic-normalization asset must bind exact package canonical policy bytes",
    );
  }
  const runtimeAbi = body.assets.find((asset) => asset.kind === "runtime-abi-manifest");
  if (runtimeAbi?.kind !== "runtime-abi-manifest" ||
      runtimeAbi.runtimeAbiId !== profile.deployment.compilerRuntime.runtimeAbiId ||
      runtimeAbi.runtimeAbiManifestId !== CPP_CUTE_BROWSER_RUNTIME_ABI_V1_MANIFEST_ID ||
      runtimeAbi.sha256 !== CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256 ||
      runtimeAbi.sha256 !== profile.deployment.compilerRuntime.runtimeAbiManifestSha256 ||
      wireIntegerToBigInt(runtimeAbi.byteLength) !== BigInt(RUNTIME_ABI_RESOURCE_BYTE_LENGTH)) {
    hashMismatch(
      "$.body.assets",
      "runtime-ABI asset must bind exact canonical profile resource identity, hash, and byte length",
    );
  }
  const expectedDependencyIds = profile.toolchain.dependencies.map((dependency) => dependency.dependencyId);
  if (!equalStrings(body.dependencyIds, expectedDependencyIds)) {
    invalid("$.body.dependencyIds", "dependency IDs must exactly equal sorted profile toolchain dependencies");
  }
  const dependencyById = new Map(profile.toolchain.dependencies.map((dependency) => [dependency.dependencyId, dependency]));
  const expectedMounts = profile.virtualFileSystem.includeRoots.filter((root) => root.owner.kind !== "source");
  const actualPacks = body.assets.filter((asset) =>
    asset.kind === "compiler-resource-pack" || asset.kind === "dependency-header-pack");
  if (actualPacks.length !== expectedMounts.length) {
    invalid("$.body.assets", "asset packs must cover every non-source include root exactly once");
  }
  const packByIncludeRoot = new Map(actualPacks.map((asset) => [asset.includeRootId, asset]));
  if (packByIncludeRoot.size !== actualPacks.length) invalid("$.body.assets", "include-root pack bindings must be unique");
  for (const root of expectedMounts) {
    const asset = packByIncludeRoot.get(root.includeRootId);
    if (asset === undefined || asset.mountedVirtualRoot !== root.virtualPath) {
      invalid("$.body.assets", `missing exact pack mount for include root ${JSON.stringify(root.includeRootId)}`);
    }
    if (root.owner.kind === "compiler-resource-directory") {
      if (asset.kind !== "compiler-resource-pack" ||
          asset.contentSetSha256 !== profile.toolchain.compiler.resourceDirectorySha256) {
        hashMismatch("$.body.assets", "compiler resource pack does not bind compiler resource-directory content");
      }
      continue;
    }
    if (root.owner.kind !== "dependency" || asset.kind !== "dependency-header-pack" ||
        asset.dependencyId !== root.owner.dependencyId) {
      invalid("$.body.assets", `dependency pack ownership differs for include root ${JSON.stringify(root.includeRootId)}`);
    }
    const dependency = dependencyById.get(root.owner.dependencyId);
    if (dependency === undefined || asset.contentSetSha256 !== dependency.headerSetSha256) {
      hashMismatch("$.body.assets", "dependency pack does not bind exact profile header set");
    }
  }
}

function validateAssetCardinalities(assets: readonly CppCuteBrowserAssetV1[]): void {
  requireCardinality(assets, "clang-extractor-wasm", 1);
  requireCardinality(assets, "compiler-resource-pack", 1);
  requireCardinality(assets, "runtime-abi-manifest", 1);
  requireCardinality(assets, "semantic-adapter-manifest", 1);
  requireCardinality(assets, "diagnostic-normalization-manifest", 1);
}

function requireCardinality(
  assets: readonly CppCuteBrowserAssetV1[],
  kind: CppCuteBrowserAssetV1["kind"],
  expected: number,
): void {
  if (assets.filter((asset) => asset.kind === kind).length !== expected) {
    invalid("$.body.assets", `manifest must contain exactly ${expected} ${kind} asset`);
  }
}

function canonicalManifestBytes(manifest: CppCuteBrowserAssetManifestV1): Uint8Array {
  try {
    return canonicalJsonBytes(manifest, { limits: CPP_CUTE_BROWSER_ASSET_MANIFEST_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$", "canonical manifest exceeds fixed resource limits", { cause });
    invalid("$", "manifest cannot be canonically encoded", { cause });
  }
}

function snapshotBytes(value: unknown): Uint8Array {
  let inspected;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid("$bytes", "manifest input must be an unshared plain Uint8Array", { cause });
  }
  if (inspected.byteLength === 0) invalid("$bytes", "manifest bytes must be nonempty");
  if (inspected.byteLength > CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT) {
    resource("$bytes", `manifest bytes exceed ${CPP_CUTE_BROWSER_ASSET_MANIFEST_BYTE_LIMIT}`);
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid("$bytes", "manifest bytes became unreadable while snapshotting", { cause });
  }
}

function normalizeOptions(options: PrepareCppCuteBrowserAssetManifestOptions): AbortSignal | undefined {
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
  if (keys.length > 1 || keys.some((key) => key !== "signal")) invalid("$options", "options contain unknown fields");
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
  if (aborted === true) cancelled();
}

function closedObject(
  value: unknown,
  fields: readonly string[],
  path: string,
  requireAll: boolean,
): JsonObject {
  if (!isJsonObject(value as JsonValue)) invalid(path, "expected object");
  const object = value as JsonObject;
  const unknown = Object.keys(object).filter((key) => !fields.includes(key));
  if (unknown.length > 0) invalid(path, `unknown closed-record fields: ${unknown.sort().join(", ")}`);
  if (requireAll) {
    const missing = fields.filter((key) => !Object.prototype.hasOwnProperty.call(object, key));
    if (missing.length > 0) invalid(path, `missing required fields: ${missing.sort().join(", ")}`);
  }
  return object;
}

function requireExactFields(object: JsonObject, fields: readonly string[], path: string): void {
  const keys = Object.keys(object);
  const unknown = keys.filter((key) => !fields.includes(key));
  const missing = fields.filter((key) => !Object.prototype.hasOwnProperty.call(object, key));
  if (unknown.length > 0 || missing.length > 0) {
    invalid(path, "asset fields do not match its closed kind schema");
  }
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  if (!Object.prototype.hasOwnProperty.call(object, name)) invalid(`${path}.${name}`, "required field is missing");
  return object[name] as JsonValue;
}

function array(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  if (value.length > MAX_ASSETS) resource(path, `array length exceeds ${MAX_ASSETS}`);
  return value;
}

function boundedString(value: JsonValue, path: string, maxBytes: number): string {
  if (typeof value !== "string" || value.length === 0 || value.includes("\0") ||
      TEXT_ENCODER.encode(value).byteLength > maxBytes) {
    invalid(path, `expected nonempty NUL-free string no longer than ${maxBytes} UTF-8 bytes`);
  }
  return value;
}

function boundedPattern(value: JsonValue, path: string, pattern: RegExp): string {
  const result = boundedString(value, path, 2_048);
  if (!pattern.test(result)) invalid(path, "string does not match closed canonical format");
  return result;
}

function sha256(value: JsonValue, path: string): string {
  return boundedPattern(value, path, SHA256_HEX);
}

function sameOriginRootRelativeUrl(value: JsonValue, path: string): string {
  const result = boundedString(value, path, 2_048);
  const segments = result.split("/");
  if (!SAME_ORIGIN_ROOT_RELATIVE_URL.test(result) || result.startsWith("//") || result.includes("%") ||
      result.includes("?") || result.includes("#") || result.includes("\\") ||
      segments.some((segment, index) => index > 0 && (segment === "." || segment === ".."))) {
    invalid(path, "URL must be a normalized same-origin root-relative path without authority, query, or fragment");
  }
  return result;
}

function virtualPathValue(value: JsonValue, path: string): string {
  const result = boundedString(value, path, 1_024);
  virtualPath(result, path);
  return result;
}

function virtualPath(value: string, path: string): void {
  if (!value.startsWith("/") || value.includes("\\") || value.includes("\0")) {
    invalid(path, "virtual root must use bounded absolute POSIX syntax");
  }
  if (value === "/") return;
  const segments = value.split("/");
  if (segments.some((segment, index) => index > 0 && (segment.length === 0 || segment === "." || segment === ".."))) {
    invalid(path, "virtual root must be normalized without empty, . or .. segments");
  }
}

function sortedIdentifierArray(value: JsonValue, path: string, pattern: RegExp): readonly string[] {
  const values = sortedStringArray(value, path);
  values.forEach((entry, index) => {
    if (!pattern.test(entry)) invalid(`${path}[${index}]`, "identifier does not match closed canonical format");
  });
  return values;
}

function sortedStringArray(value: JsonValue, path: string): readonly string[] {
  const values = array(value, path).map((entry, index) => boundedString(entry, `${path}[${index}]`, 2_048));
  requireSortedUnique(values, path);
  return values;
}

function requireSortedUnique(values: readonly string[], path: string): void {
  for (let index = 1; index < values.length; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    if (previous === undefined || current === undefined || previous >= current) {
      invalid(path, "entries must be strictly sorted and unique");
    }
  }
}

function requireUnique(values: readonly string[], path: string, message: string): void {
  if (new Set(values).size !== values.length) invalid(path, message);
}

function exactString<T extends string>(value: JsonValue, expected: T, path: string): T {
  if (value !== expected) invalid(path, `must equal ${JSON.stringify(expected)}`);
  return expected;
}

function literal<T extends string>(value: JsonValue, expected: T, path: string): asserts value is T {
  if (value !== expected) invalid(path, `must equal ${JSON.stringify(expected)}`);
}

function requireIdentityLength(
  asset: { readonly byteLength: WireU64; readonly unpackedByteLength: WireU64 },
  path: string,
): void {
  if (asset.byteLength !== asset.unpackedByteLength) {
    invalid(`${path}.unpackedByteLength`, "identity-compressed asset lengths must match");
  }
}

function requirePackLength(
  asset: { readonly byteLength: WireU64; readonly unpackedByteLength: WireU64 },
  fileContentByteLength: WireU64,
  path: string,
): void {
  if (wireIntegerToBigInt(asset.unpackedByteLength) <= wireIntegerToBigInt(fileContentByteLength)) {
    invalid(
      `${path}.fileContentByteLength`,
      "VFS pack bytes must exceed mounted file-content bytes because the pack has a fixed header and index",
    );
  }
}

function wirePositive(value: JsonValue, path: string): WireU64 {
  if (typeof value !== "string" || !/^[1-9][0-9]*$/u.test(value)) invalid(path, "expected positive canonical u64 string");
  const parsed = BigInt(value);
  if (parsed > 18_446_744_073_709_551_615n) resource(path, "u64 exceeds maximum");
  return value as WireU64;
}

function minBigInt(left: bigint, right: bigint): bigint {
  return left < right ? left : right;
}

function equalStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function canonicalText(value: JsonValue): string {
  return new TextDecoder().decode(canonicalJsonBytes(value, {
    limits: CPP_CUTE_BROWSER_ASSET_MANIFEST_DECODE_LIMITS,
  }));
}

async function hashJson(value: JsonValue, path: string): Promise<string> {
  try {
    return await hashCanonicalJson(value, { limits: CPP_CUTE_BROWSER_ASSET_MANIFEST_DECODE_LIMITS });
  } catch (cause) {
    if (isHashUnavailable(cause)) hashUnavailable(path, cause);
    if (isSchemaResourceLimit(cause)) resource(path, "hash projection exceeds fixed resource limits", { cause });
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

function isHashUnavailable(value: unknown): value is SemanticSchemaError {
  return value instanceof SemanticSchemaError && value.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.hashUnavailable;
}

function isSchemaResourceLimit(value: unknown): value is SemanticSchemaError {
  return value instanceof SemanticSchemaError && value.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit;
}

function cancelled(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-CANCELLED",
    "$options.signal",
    "browser asset-manifest preparation was cancelled",
  );
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-UNVERIFIED",
    "$",
    "browser asset operation requires opaque prepared manifest authority",
  );
}

function hashMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-MISMATCH", path, message);
}

function hashUnavailable(path: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-HASH-UNAVAILABLE",
    path,
    "SHA-256 requires Web Crypto SubtleCrypto",
    { cause },
  );
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-RESOURCE-LIMIT", path, message, options);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-ASSETS-INVALID", path, message, options);
}

function fail(
  code: CppCuteBrowserAssetManifestErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserAssetManifestError(code, path, message, options);
}

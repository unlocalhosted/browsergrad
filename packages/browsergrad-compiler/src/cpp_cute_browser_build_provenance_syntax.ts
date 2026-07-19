import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  assertJsonValue,
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
  hashCanonicalJson,
  isJsonObject,
  parseWireU64,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
  unwrapPreparedCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetV1,
  type PreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import {
  unwrapPreparedCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "./cpp_cute_browser_build_lock.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  type VerifiedCppCuteBrowserWorkerBundle,
} from "./cpp_cute_browser_worker_bundle.js";
export { CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE } from "./cpp_cute_browser_assets.js";
export const CPP_CUTE_BROWSER_BUILD_TYPE =
  "https://browsergrad.dev/build-types/cpp-cute-browser-assets/v1";
export const CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE = "application/vnd.in-toto+json";
export const CPP_CUTE_BROWSER_BUILD_IN_TOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1";
export const CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT = 256 * 1024;

export const CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS: DecodeLimits = Object.freeze({
  maxDocumentBytes: CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
  maxDepth: 16,
  maxNodes: 1_024,
  maxStringBytes: 192 * 1024,
  maxArrayLength: 32,
  maxObjectProperties: 64,
  maxRank: 1,
  maxIntegerBits: 64,
  maxArithmeticOperations: 2_048,
});

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const SHA256_KEY_ID = /^sha256:[0-9a-f]{64}$/u;
const BUILD_SUBJECT_ID = /^bg\.cpp\.browser-build-subject\.sha256\.[0-9a-f]{64}$/u;
const BUILD_INPUT_LOCK_ID = /^bg\.cpp\.browser-build-input-lock\.sha256\.[0-9a-f]{64}$/u;
const ASSET_MANIFEST_ID = /^bg\.cpp\.browser-assets\.sha256\.[0-9a-f]{64}$/u;
const WORKER_BUNDLE_ID = /^bg\.cpp\.browser-worker-bundle\.sha256\.[0-9a-f]{64}$/u;
const BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u;

export type CppCuteBrowserBuildSubjectAssetV1 = Omit<CppCuteBrowserAssetV1, "buildSubjectId">;

/**
 * Cycle-free identity projection for one browser compiler build subject.
 *
 * It intentionally excludes the profile hash, manifest identity, asset-set
 * identity, and every build-subject reference. Those identities depend on
 * the build-subject reference and would create a hash cycle if included here.
 */
export interface CppCuteBrowserBuildSubjectProjectionV1 extends JsonObject {
  readonly sourceAbiSha256: string;
  readonly dependencyIds: readonly string[];
  readonly mountedVirtualRoots: readonly string[];
  readonly assets: readonly CppCuteBrowserBuildSubjectAssetV1[];
  readonly buildInputLock: JsonObject & {
    readonly lockId: string;
    readonly resourceSha256: string;
    readonly recipeSha256: string;
    readonly extractorSourceSetSha256: string;
    readonly noticeInventorySha256: string;
  };
  readonly workerBundle: JsonObject & {
    readonly bundleId: string;
    readonly sha256: string;
    readonly byteLength: number;
    readonly factorySha256: string;
    readonly factoryByteLength: number;
  };
}

/** Content identity only. It grants no producer, distribution, or release authority. */
export interface CppCuteBrowserBuildSubjectIdentity {
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
  readonly projection: CppCuteBrowserBuildSubjectProjectionV1;
  readonly grantsProvenanceAuthority: false;
  readonly producerTrusted: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface DeriveCppCuteBrowserBuildSubjectIdentityInput {
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
}

export interface CppCuteBrowserBuildProvenanceSubjectV1 extends JsonObject {
  readonly name: string;
  readonly digest: JsonObject & { readonly sha256: string };
}

export interface CppCuteBrowserBuildSubjectReferenceV1 extends JsonObject {
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
}

export interface CppCuteBrowserBuildProfileReferenceV1 extends JsonObject {
  readonly profileId: string;
  readonly profileHash: string;
  readonly compilationContractHash: string;
}

export interface CppCuteBrowserBuildAssetManifestReferenceV1 extends JsonObject {
  readonly manifestId: string;
  readonly manifestSha256: string;
  readonly manifestByteLength: WireU64;
  readonly assetSetSha256: string;
}

export interface CppCuteBrowserBuildInputLockReferenceV1 extends JsonObject {
  readonly lockId: string;
  readonly resourceSha256: string;
  readonly recipeSha256: string;
}

export interface CppCuteBrowserBuildWorkerBundleReferenceV1 extends JsonObject {
  readonly bundleId: string;
  readonly sha256: string;
  readonly byteLength: number;
  readonly factorySha256: string;
  readonly factoryByteLength: number;
}

export interface CppCuteBrowserBuildAuthorityLimitsV1 extends JsonObject {
  readonly fullDistributedOutputSetReproducible: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface CppCuteBrowserBuildProvenancePredicateV1 extends JsonObject {
  readonly builderId: string;
  readonly buildType: typeof CPP_CUTE_BROWSER_BUILD_TYPE;
  readonly buildSubject: CppCuteBrowserBuildSubjectReferenceV1;
  readonly profile: CppCuteBrowserBuildProfileReferenceV1;
  readonly assetManifest: CppCuteBrowserBuildAssetManifestReferenceV1;
  readonly buildInputLock: CppCuteBrowserBuildInputLockReferenceV1;
  readonly workerBundle: CppCuteBrowserBuildWorkerBundleReferenceV1;
  readonly authorityLimits: CppCuteBrowserBuildAuthorityLimitsV1;
}

export interface CppCuteBrowserBuildProvenanceStatementV1 extends JsonObject {
  readonly _type: typeof CPP_CUTE_BROWSER_BUILD_IN_TOTO_STATEMENT_TYPE;
  readonly subject: readonly [CppCuteBrowserBuildProvenanceSubjectV1];
  readonly predicateType: typeof CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE;
  readonly predicate: CppCuteBrowserBuildProvenancePredicateV1;
}

export interface CppCuteBrowserBuildDsseSignatureV1 extends JsonObject {
  readonly keyid: string;
  readonly sig: string;
}

export interface CppCuteBrowserBuildProvenanceEnvelopeV1 extends JsonObject {
  readonly payloadType: typeof CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE;
  readonly payload: string;
  readonly signatures: readonly [CppCuteBrowserBuildDsseSignatureV1];
}

/**
 * Strictly decoded untrusted syntax. No cryptographic or producer authority is
 * minted by this parser, even when the signature field is well formed.
 */
export interface DecodedUntrustedCppCuteBrowserBuildProvenanceSyntax {
  readonly envelope: CppCuteBrowserBuildProvenanceEnvelopeV1;
  readonly statement: CppCuteBrowserBuildProvenanceStatementV1;
  readonly formatOnly: true;
  readonly signatureVerified: false;
  readonly producerTrusted: false;
  readonly exactAssetBytesVerified: false;
  readonly fullDistributedOutputSetReproducible: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface CppCuteBrowserBuildProvenanceSyntaxOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserBuildProvenanceSyntaxErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-NONCANONICAL"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-RESOURCE-LIMIT";

export class CppCuteBrowserBuildProvenanceSyntaxError extends Error {
  constructor(
    readonly code: CppCuteBrowserBuildProvenanceSyntaxErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserBuildProvenanceSyntaxError";
  }
}

/**
 * Derives a cycle-free content identity from exact opaque prerequisite inputs.
 * The returned identity is deliberately not a provenance or release authority.
 */
export async function deriveCppCuteBrowserBuildSubjectIdentity(
  input: DeriveCppCuteBrowserBuildSubjectIdentityInput,
): Promise<CppCuteBrowserBuildSubjectIdentity> {
  const values = exactInput(input);
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(values.assetManifest);
  unwrapPreparedCppCuteBrowserBuildInputLock(values.buildInputLock);
  const worker = inspectVerifiedCppCuteBrowserWorkerBundle(values.workerBundle);
  const body = manifestRecord.manifest.body;
  const assets = body.assets.map(stripBuildSubjectReference);
  const projection = deepFreezeJson({
    sourceAbiSha256: body.sourceAbiSha256,
    dependencyIds: body.dependencyIds,
    mountedVirtualRoots: body.mountedVirtualRoots,
    assets,
    buildInputLock: {
      lockId: values.buildInputLock.lockId,
      resourceSha256: values.buildInputLock.resourceSha256,
      recipeSha256: values.buildInputLock.recipeSha256,
      extractorSourceSetSha256: values.buildInputLock.extractorSourceSetSha256,
      noticeInventorySha256: values.buildInputLock.noticeInventorySha256,
    },
    workerBundle: {
      bundleId: worker.bundleId,
      sha256: worker.sha256,
      byteLength: worker.byteLength,
      factorySha256: worker.factorySha256,
      factoryByteLength: worker.factoryByteLength,
    },
  }) as CppCuteBrowserBuildSubjectProjectionV1;
  let buildSubjectSha256: string;
  try {
    buildSubjectSha256 = await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.browser-build-subject.v1",
      projection,
    }, { limits: CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$.input", "build-subject projection exceeds fixed limits", { cause });
    invalid("$.input", "build-subject identity could not be derived", { cause });
  }
  return Object.freeze({
    buildSubjectId: `bg.cpp.browser-build-subject.sha256.${buildSubjectSha256}`,
    buildSubjectSha256,
    projection,
    grantsProvenanceAuthority: false,
    producerTrusted: false,
    distributionAuthorized: false,
    releaseReady: false,
  });
}

/**
 * Decodes closed canonical DSSE/in-toto syntax without verifying its signature.
 * This function intentionally has no success path that mints trust authority.
 */
export function decodeUntrustedCppCuteBrowserBuildProvenanceSyntax(
  value: unknown,
  options: CppCuteBrowserBuildProvenanceSyntaxOptions = {},
): DecodedUntrustedCppCuteBrowserBuildProvenanceSyntax {
  throwIfAborted(options.signal);
  let json: JsonValue;
  try {
    assertJsonValue(value, { limits: CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS });
    json = value;
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$", "DSSE envelope exceeds fixed limits", { cause });
    invalid("$", "DSSE envelope must be an accessor-free canonical JSON tree", { cause });
  }
  const object = closedObject(json, ["payloadType", "payload", "signatures"], "$");
  literal(
    field(object, "payloadType", "$"),
    CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE,
    "$.payloadType",
  );
  const payload = boundedString(
    field(object, "payload", "$"),
    "$.payload",
    CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT,
  );
  const payloadBytes = decodeCanonicalBase64(payload, "$.payload");
  if (payloadBytes.byteLength > CPP_CUTE_BROWSER_BUILD_PROVENANCE_BYTE_LIMIT) {
    resource("$.payload", "decoded in-toto payload exceeds fixed byte limit");
  }
  let decoded: JsonValue;
  try {
    decoded = decodeWireJson(payloadBytes, { limits: CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$.payload", "in-toto payload exceeds fixed limits", { cause });
    invalid("$.payload", "payload must decode as strict UTF-8 JSON", { cause });
  }
  const statement = parseStatement(decoded, "$.payload");
  let canonicalPayload: Uint8Array;
  try {
    canonicalPayload = canonicalJsonBytes(statement, { limits: CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) resource("$.payload", "canonical in-toto payload exceeds fixed limits", { cause });
    invalid("$.payload", "in-toto payload cannot be canonically encoded", { cause });
  }
  if (!equalBytes(payloadBytes, canonicalPayload)) {
    noncanonical("$.payload", "in-toto payload must use exact BrowserGrad canonical JSON bytes");
  }
  const signatures = array(field(object, "signatures", "$"), "$.signatures");
  if (signatures.length !== 1 || signatures[0] === undefined) {
    invalid("$.signatures", "syntax v1 requires exactly one DSSE signature record");
  }
  const envelope = deepFreezeJson({
    payloadType: CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE,
    payload,
    signatures: [parseSignature(signatures[0], "$.signatures[0]")],
  }) as CppCuteBrowserBuildProvenanceEnvelopeV1;
  throwIfAborted(options.signal);
  return Object.freeze({
    envelope,
    statement: deepFreezeJson(statement),
    formatOnly: true,
    signatureVerified: false,
    producerTrusted: false,
    exactAssetBytesVerified: false,
    fullDistributedOutputSetReproducible: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    releaseReady: false,
  });
}

export function cppCuteBrowserBuildProvenancePayloadBytes(
  statement: CppCuteBrowserBuildProvenanceStatementV1,
): Uint8Array {
  return canonicalJsonBytes(statement, { limits: CPP_CUTE_BROWSER_BUILD_PROVENANCE_DECODE_LIMITS });
}

export function cppCuteBrowserBuildProvenancePayloadBase64(
  statement: CppCuteBrowserBuildProvenanceStatementV1,
): string {
  return encodeCanonicalBase64(cppCuteBrowserBuildProvenancePayloadBytes(statement));
}

export function cppCuteBrowserBuildProvenanceDsseSigningBytes(
  statement: CppCuteBrowserBuildProvenanceStatementV1,
): Uint8Array {
  const payload = cppCuteBrowserBuildProvenancePayloadBytes(statement);
  const payloadTypeBytes = new TextEncoder().encode(CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE);
  const prefix = new TextEncoder().encode(
    `DSSEv1 ${payloadTypeBytes.byteLength} ${CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE} ${payload.byteLength} `,
  );
  const result = new Uint8Array(prefix.byteLength + payload.byteLength);
  result.set(prefix, 0);
  result.set(payload, prefix.byteLength);
  return result;
}

function parseStatement(value: JsonValue, path: string): CppCuteBrowserBuildProvenanceStatementV1 {
  const object = closedObject(value, ["_type", "subject", "predicateType", "predicate"], path);
  literal(field(object, "_type", path), CPP_CUTE_BROWSER_BUILD_IN_TOTO_STATEMENT_TYPE, `${path}._type`);
  literal(
    field(object, "predicateType", path),
    CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    `${path}.predicateType`,
  );
  const subjects = array(field(object, "subject", path), `${path}.subject`);
  if (subjects.length !== 1 || subjects[0] === undefined) {
    invalid(`${path}.subject`, "syntax v1 requires exactly one in-toto subject");
  }
  const subject = parseSubject(subjects[0], `${path}.subject[0]`);
  const predicate = parsePredicate(field(object, "predicate", path), `${path}.predicate`);
  if (subject.name !== predicate.buildSubject.buildSubjectId ||
      subject.digest.sha256 !== predicate.buildSubject.buildSubjectSha256) {
    invalid(`${path}.subject[0]`, "in-toto subject and predicate build-subject reference must be identical");
  }
  return {
    _type: CPP_CUTE_BROWSER_BUILD_IN_TOTO_STATEMENT_TYPE,
    subject: [subject],
    predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    predicate,
  };
}

function parseSubject(value: JsonValue, path: string): CppCuteBrowserBuildProvenanceSubjectV1 {
  const object = closedObject(value, ["name", "digest"], path);
  const digest = closedObject(field(object, "digest", path), ["sha256"], `${path}.digest`);
  const name = matchingString(field(object, "name", path), `${path}.name`, BUILD_SUBJECT_ID, "build subject ID");
  const sha = sha256(field(digest, "sha256", `${path}.digest`), `${path}.digest.sha256`);
  if (!name.endsWith(sha)) invalid(path, "build subject ID suffix must equal its SHA-256 digest");
  return { name, digest: { sha256: sha } };
}

function parsePredicate(value: JsonValue, path: string): CppCuteBrowserBuildProvenancePredicateV1 {
  const object = closedObject(value, [
    "builderId",
    "buildType",
    "buildSubject",
    "profile",
    "assetManifest",
    "buildInputLock",
    "workerBundle",
    "authorityLimits",
  ], path);
  literal(field(object, "buildType", path), CPP_CUTE_BROWSER_BUILD_TYPE, `${path}.buildType`);
  return {
    builderId: canonicalHttpsIdentifier(field(object, "builderId", path), `${path}.builderId`),
    buildType: CPP_CUTE_BROWSER_BUILD_TYPE,
    buildSubject: parseBuildSubject(field(object, "buildSubject", path), `${path}.buildSubject`),
    profile: parseProfile(field(object, "profile", path), `${path}.profile`),
    assetManifest: parseAssetManifest(field(object, "assetManifest", path), `${path}.assetManifest`),
    buildInputLock: parseBuildInputLock(field(object, "buildInputLock", path), `${path}.buildInputLock`),
    workerBundle: parseWorkerBundle(field(object, "workerBundle", path), `${path}.workerBundle`),
    authorityLimits: parseAuthorityLimits(field(object, "authorityLimits", path), `${path}.authorityLimits`),
  };
}

function parseBuildSubject(value: JsonValue, path: string): CppCuteBrowserBuildSubjectReferenceV1 {
  const object = closedObject(value, ["buildSubjectId", "buildSubjectSha256"], path);
  const buildSubjectId = matchingString(
    field(object, "buildSubjectId", path),
    `${path}.buildSubjectId`,
    BUILD_SUBJECT_ID,
    "build subject ID",
  );
  const buildSubjectSha256 = sha256(
    field(object, "buildSubjectSha256", path),
    `${path}.buildSubjectSha256`,
  );
  if (!buildSubjectId.endsWith(buildSubjectSha256)) {
    invalid(path, "build subject ID suffix must equal buildSubjectSha256");
  }
  return { buildSubjectId, buildSubjectSha256 };
}

function parseProfile(value: JsonValue, path: string): CppCuteBrowserBuildProfileReferenceV1 {
  const object = closedObject(value, ["profileId", "profileHash", "compilationContractHash"], path);
  return {
    profileId: boundedString(field(object, "profileId", path), `${path}.profileId`, 512),
    profileHash: sha256(field(object, "profileHash", path), `${path}.profileHash`),
    compilationContractHash: sha256(
      field(object, "compilationContractHash", path),
      `${path}.compilationContractHash`,
    ),
  };
}

function parseAssetManifest(value: JsonValue, path: string): CppCuteBrowserBuildAssetManifestReferenceV1 {
  const object = closedObject(value, ["manifestId", "manifestSha256", "manifestByteLength", "assetSetSha256"], path);
  return {
    manifestId: matchingString(
      field(object, "manifestId", path),
      `${path}.manifestId`,
      ASSET_MANIFEST_ID,
      "asset manifest ID",
    ),
    manifestSha256: sha256(field(object, "manifestSha256", path), `${path}.manifestSha256`),
    manifestByteLength: parseWireU64(field(object, "manifestByteLength", path), `${path}.manifestByteLength`),
    assetSetSha256: sha256(field(object, "assetSetSha256", path), `${path}.assetSetSha256`),
  };
}

function parseBuildInputLock(value: JsonValue, path: string): CppCuteBrowserBuildInputLockReferenceV1 {
  const object = closedObject(value, ["lockId", "resourceSha256", "recipeSha256"], path);
  return {
    lockId: matchingString(field(object, "lockId", path), `${path}.lockId`, BUILD_INPUT_LOCK_ID, "build-input lock ID"),
    resourceSha256: sha256(field(object, "resourceSha256", path), `${path}.resourceSha256`),
    recipeSha256: sha256(field(object, "recipeSha256", path), `${path}.recipeSha256`),
  };
}

function parseWorkerBundle(value: JsonValue, path: string): CppCuteBrowserBuildWorkerBundleReferenceV1 {
  const object = closedObject(value, ["bundleId", "sha256", "byteLength", "factorySha256", "factoryByteLength"], path);
  return {
    bundleId: matchingString(field(object, "bundleId", path), `${path}.bundleId`, WORKER_BUNDLE_ID, "worker bundle ID"),
    sha256: sha256(field(object, "sha256", path), `${path}.sha256`),
    byteLength: safeByteLength(field(object, "byteLength", path), `${path}.byteLength`),
    factorySha256: sha256(field(object, "factorySha256", path), `${path}.factorySha256`),
    factoryByteLength: safeByteLength(field(object, "factoryByteLength", path), `${path}.factoryByteLength`),
  };
}

function parseAuthorityLimits(value: JsonValue, path: string): CppCuteBrowserBuildAuthorityLimitsV1 {
  const fields = [
    "fullDistributedOutputSetReproducible",
    "licenseReviewComplete",
    "distributionAuthorized",
    "releaseReady",
  ] as const;
  const object = closedObject(value, fields, path);
  for (const fieldName of fields) {
    if (object[fieldName] !== false) invalid(`${path}.${fieldName}`, "syntax v1 requires an explicit false non-authority claim");
  }
  return {
    fullDistributedOutputSetReproducible: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    releaseReady: false,
  };
}

function parseSignature(value: JsonValue, path: string): CppCuteBrowserBuildDsseSignatureV1 {
  const object = closedObject(value, ["keyid", "sig"], path);
  const keyid = matchingString(field(object, "keyid", path), `${path}.keyid`, SHA256_KEY_ID, "signature key ID");
  const sig = boundedString(field(object, "sig", path), `${path}.sig`, 512);
  const signatureBytes = decodeCanonicalBase64(sig, `${path}.sig`);
  if (signatureBytes.byteLength !== 64) invalid(`${path}.sig`, "P-256 signature syntax requires 64-byte IEEE P1363 bytes");
  return { keyid, sig };
}

function stripBuildSubjectReference(asset: CppCuteBrowserAssetV1): CppCuteBrowserBuildSubjectAssetV1 {
  const output: Record<string, JsonValue> = {};
  for (const key of Object.keys(asset)) {
    if (key === "buildSubjectId") continue;
    const value = asset[key];
    if (value !== undefined) output[key] = value;
  }
  return output as CppCuteBrowserBuildSubjectAssetV1;
}

function exactInput(input: DeriveCppCuteBrowserBuildSubjectIdentityInput): DeriveCppCuteBrowserBuildSubjectIdentityInput {
  if (typeof input !== "object" || input === null || Object.getPrototypeOf(input) !== Object.prototype) {
    invalid("$.input", "input must be a plain object");
  }
  const descriptors = Object.getOwnPropertyDescriptors(input);
  const expected = ["assetManifest", "buildInputLock", "workerBundle"];
  const actual = Reflect.ownKeys(descriptors);
  if (actual.some((key) => typeof key !== "string") ||
      actual.length !== expected.length || expected.some((key) => !Object.hasOwn(descriptors, key))) {
    invalid("$.input", "input must contain exactly assetManifest, buildInputLock, and workerBundle");
  }
  for (const key of expected) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor) || descriptor.enumerable !== true) {
      invalid(`$.input.${key}`, "input fields must be enumerable data properties");
    }
  }
  return input;
}

function decodeCanonicalBase64(value: string, path: string): Uint8Array {
  if (!BASE64.test(value)) invalid(path, "expected canonical padded base64");
  let bytes: Uint8Array;
  try {
    bytes = Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
  } catch (cause) {
    invalid(path, "invalid base64", { cause });
  }
  if (encodeCanonicalBase64(bytes) !== value) noncanonical(path, "expected canonical padded base64");
  return bytes;
}

function encodeCanonicalBase64(bytes: Uint8Array): string {
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.byteLength; offset += 32_768) {
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 32_768)));
  }
  return btoa(chunks.join(""));
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const unknown = Object.keys(value).filter((key) => !fields.includes(key));
  const missing = fields.filter((key) => !Object.hasOwn(value, key));
  if (unknown.length > 0) invalid(path, `unknown closed-record fields: ${unknown.sort().join(", ")}`);
  if (missing.length > 0) invalid(path, `missing required fields: ${missing.sort().join(", ")}`);
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(`${path}.${name}`, "field is required");
  return value;
}

function array(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function literal(value: JsonValue, expected: string, path: string): void {
  if (value !== expected) invalid(path, `expected ${expected}`);
}

function boundedString(value: JsonValue, path: string, maximumBytes: number): string {
  if (typeof value !== "string") invalid(path, "expected string");
  const byteLength = new TextEncoder().encode(value).byteLength;
  if (value.length === 0 || value.includes("\0") || byteLength > maximumBytes) {
    invalid(path, `string must be non-empty, NUL-free, and at most ${maximumBytes} UTF-8 bytes`);
  }
  return value;
}

function matchingString(value: JsonValue, path: string, pattern: RegExp, name: string): string {
  const text = boundedString(value, path, 1_024);
  if (!pattern.test(text)) invalid(path, `${name} has invalid syntax`);
  return text;
}

function sha256(value: JsonValue, path: string): string {
  if (typeof value !== "string" || !SHA256_HEX.test(value)) {
    invalid(path, "SHA-256 must be 64 lowercase hexadecimal digits");
  }
  return value;
}

function safeByteLength(value: JsonValue, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    invalid(path, "byte length must be a non-negative safe integer");
  }
  return value;
}

function canonicalHttpsIdentifier(value: JsonValue, path: string): string {
  const identifier = boundedString(value, path, 1_024);
  let parsed: URL;
  try {
    parsed = new URL(identifier);
  } catch (cause) {
    invalid(path, "builder identity must be an absolute HTTPS URL", { cause });
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" || parsed.password !== "" ||
      parsed.search !== "" || parsed.hash !== "" || parsed.pathname === "/" ||
      parsed.pathname.endsWith("/") || `${parsed.origin}${parsed.pathname}` !== identifier) {
    invalid(path, "builder identity must be a canonical credential-free HTTPS URL without query, fragment, or trailing slash");
  }
  return identifier;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function isSchemaResourceLimit(cause: unknown): cause is SemanticSchemaError {
  return cause instanceof SemanticSchemaError && cause.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-CANCELLED",
      "$.options.signal",
      "browser build-provenance syntax decoding was cancelled",
    );
  }
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-RESOURCE-LIMIT", path, message, options);
}

function noncanonical(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-NONCANONICAL", path, message);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SYNTAX-INVALID", path, message, options);
}

function fail(
  code: CppCuteBrowserBuildProvenanceSyntaxErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserBuildProvenanceSyntaxError(code, path, message, options);
}

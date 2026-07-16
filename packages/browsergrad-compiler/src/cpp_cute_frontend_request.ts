import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  assertJsonValue,
  canonicalizeJson,
  deepFreezeJson,
  encodeWireU64,
  hashCanonicalJson,
  isJsonObject,
  parseWireU64,
  resolveDecodeLimits,
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
  CPP_CUTE_FRONTEND_SEMANTIC_EXTRACTION_LIMIT_KEYS,
  unwrapPreparedCppCuteFrontendProfile,
  validateCppCuteVirtualPath,
  type CppCuteFrontendCompilerOption,
  type CppCuteFrontendExtractionLimits,
  type CppCuteFrontendIncludeRoot,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
  CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA,
} from "./cpp_cute_frontend_types.js";

export const CPP_CUTE_FRONTEND_REQUEST_SCHEMA = "browsergrad.compiler.cpp-cute.frontend-request";
export const CPP_CUTE_FRONTEND_REQUEST_MAJOR = 1;
export const CPP_CUTE_FRONTEND_REQUEST_MINOR = 0;

const REQUEST_ID = /^bg\.cpp\.frontend-request\.sha256\.[0-9a-f]{64}$/u;
const ENTRY_REQUEST_ID = /^bg\.cpp\.entry-request\.sha256\.[0-9a-f]{64}$/u;
const FILE_ID = /^bg\.cpp\.file\.sha256\.[0-9a-f]{64}$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const TEXT_ENCODER = new TextEncoder();
const PREPARED_REQUESTS = new WeakMap<object, StoredCppCuteFrontendRequest>();

export interface CppCuteFrontendRequestVersionV1 extends JsonObject {
  readonly major: typeof CPP_CUTE_FRONTEND_REQUEST_MAJOR;
  readonly minor: typeof CPP_CUTE_FRONTEND_REQUEST_MINOR;
}

export interface CppCuteFrontendRequestSourceFileV1 extends JsonObject {
  readonly fileId: string;
  readonly role: "main-source" | "project-header";
  readonly virtualPath: string;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
  /** null for main source; source-owned include root for project headers. */
  readonly includeRootId: string | null;
}

export interface CppCuteFrontendRequestSourceAnchorV1 extends JsonObject {
  readonly virtualPath: string;
  readonly beginByte: WireU64;
  readonly endByte: WireU64;
  readonly tokenSha256: string;
}

export type CppCuteFrontendEntryRequestV1 =
  | (JsonObject & {
      readonly requestId: string;
      readonly kind: "layout";
      readonly declarationKind: "variable";
      readonly anchor: CppCuteFrontendRequestSourceAnchorV1;
    })
  | (JsonObject & {
      readonly requestId: string;
      readonly kind: "view-copy";
      readonly declarationKind: "function";
      readonly anchor: CppCuteFrontendRequestSourceAnchorV1;
    });

export interface CppCuteFrontendExpectedArtifactV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA;
  readonly version: JsonObject & {
    readonly major: typeof CPP_CUTE_FRONTEND_ARTIFACT_MAJOR;
    readonly minor: typeof CPP_CUTE_FRONTEND_ARTIFACT_MINOR;
  };
}

/** Request-owned ceilings may narrow, never widen, profile ceilings. */
export type CppCuteFrontendRequestLimitsV1 = JsonObject & Readonly<Pick<
  CppCuteFrontendExtractionLimits,
  (typeof CPP_CUTE_FRONTEND_SEMANTIC_EXTRACTION_LIMIT_KEYS)[number]
>>;

export interface CppCuteFrontendRequestV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_FRONTEND_REQUEST_SCHEMA;
  readonly version: CppCuteFrontendRequestVersionV1;
  readonly requestId: string;
  /** Producer-neutral compilation identity; deployment profile identity stays detached. */
  readonly compilationContractHash: string;
  readonly mainVirtualPath: string;
  readonly files: readonly CppCuteFrontendRequestSourceFileV1[];
  readonly entryRequests: readonly CppCuteFrontendEntryRequestV1[];
  readonly expectedArtifact: CppCuteFrontendExpectedArtifactV1;
  readonly limits: CppCuteFrontendRequestLimitsV1;
}

export interface CppCuteFrontendRequestBodyV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_FRONTEND_REQUEST_SCHEMA;
  readonly version: CppCuteFrontendRequestVersionV1;
  readonly compilationContractHash: string;
  readonly mainVirtualPath: string;
  readonly files: readonly CppCuteFrontendRequestSourceFileV1[];
  readonly entryRequests: readonly CppCuteFrontendEntryRequestV1[];
  readonly expectedArtifact: CppCuteFrontendExpectedArtifactV1;
  readonly limits: CppCuteFrontendRequestLimitsV1;
}

/** Bytes never enter JSON decoding. Preparation snapshots and copies each exact view. */
export interface CppCuteFrontendSourceSnapshotInput {
  readonly virtualPath: string;
  readonly bytes: Uint8Array;
}

/** Caller-declared source reference. It is not source-acquisition proof. */
export interface CppCuteFrontendDetachedSourceReferenceV1 extends JsonObject {
  readonly statementSha256: string;
}

/** Optional test/release assertions. Never part of normal compilation identity. */
export interface CppCuteFrontendDetachedConformanceV1 extends JsonObject {
  readonly expectedArtifactSha256: string | null;
  readonly expectedOpenedHeaderSetSha256: string | null;
  readonly expectedInputClosureSha256: string | null;
}

export interface CppCuteFrontendDetachedExpectationsV1 extends JsonObject {
  readonly declaredSourceReference: CppCuteFrontendDetachedSourceReferenceV1 | null;
  readonly conformance: CppCuteFrontendDetachedConformanceV1 | null;
}

export interface PrepareCppCuteFrontendRequestOptions {
  readonly detached?: CppCuteFrontendDetachedExpectationsV1;
  readonly limits?: Partial<DecodeLimits>;
  readonly signal?: AbortSignal;
}

declare const preparedCppCuteFrontendRequestBrand: unique symbol;

/** Instance-authorized request. Fields are summaries, not structural authority. */
export interface PreparedCppCuteFrontendRequest {
  readonly [preparedCppCuteFrontendRequestBrand]: true;
  readonly requestId: string;
  readonly requestHash: string;
  readonly profileHash: string;
  readonly compilationContractHash: string;
  readonly mainVirtualPath: string;
  readonly sourceFileCount: number;
  readonly sourceByteLength: WireU64;
  readonly entryRequestId: string;
  readonly declaredSourceReferenceStatementSha256: string | null;
  readonly conformanceAssertionSha256: string | null;
}

export interface PreparedCppCuteFrontendOrderedInputs {
  /** Profile-owned option order. Caller cannot inject compiler argv through common request. */
  readonly compilerOptions: readonly CppCuteFrontendCompilerOption[];
  /** Complete profile-owned available header universe, in resolution order. */
  readonly availableIncludeRoots: readonly CppCuteFrontendIncludeRoot[];
}

export interface PreparedCppCuteFrontendRequestRecord {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly request: CppCuteFrontendRequestV1;
  readonly orderedInputs: PreparedCppCuteFrontendOrderedInputs;
  readonly detached: CppCuteFrontendDetachedExpectationsV1;
}

interface StoredCppCuteFrontendRequest extends PreparedCppCuteFrontendRequestRecord {
  readonly sourceSnapshots: readonly StoredSourceSnapshot[];
}

interface StoredSourceSnapshot {
  readonly virtualPath: string;
  readonly bytes: Uint8Array;
}

export type CppCuteFrontendRequestErrorCode =
  | "BG-COMPILER-CPP-CUTE-REQUEST-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-REQUEST-INVALID"
  | "BG-COMPILER-CPP-CUTE-REQUEST-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-REQUEST-PROFILE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-REQUEST-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-REQUEST-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-REQUEST-UNVERIFIED";

export class CppCuteFrontendRequestError extends Error {
  constructor(
    readonly code: CppCuteFrontendRequestErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteFrontendRequestError";
  }
}

/**
 * Prepares one producer-neutral request plus exact caller-owned source bytes.
 * Repository/revision, container/worker identity, expected output hashes, and
 * observed opened-header closure do not enter request identity.
 */
export async function prepareCppCuteFrontendRequest(
  profile: PreparedCppCuteFrontendProfile,
  value: unknown,
  sourceSnapshots: readonly CppCuteFrontendSourceSnapshotInput[],
  options: PrepareCppCuteFrontendRequestOptions = {},
): Promise<PreparedCppCuteFrontendRequest> {
  const profileRecord = unwrapPreparedCppCuteFrontendProfile(profile);
  const normalizedOptions = normalizeOptions(options);
  throwIfAborted(normalizedOptions.signal);
  let request: CppCuteFrontendRequestV1;
  try {
    assertJsonValue(value, { limits: normalizedOptions.decodeLimits });
    request = parseRequest(value, profileRecord.profile);
    canonicalizeJson(request, { limits: normalizedOptions.decodeLimits });
  } catch (error) {
    translateSchemaError(error, "request is not closed canonical JSON");
  }
  if (request.compilationContractHash !== profile.compilationContractHash) {
    fail(
      "BG-COMPILER-CPP-CUTE-REQUEST-PROFILE-MISMATCH",
      "$.compilationContractHash",
      "request compilation contract differs from exact prepared profile",
    );
  }
  const snapshots = snapshotSourceInputs(sourceSnapshots, request);
  throwIfAborted(normalizedOptions.signal);
  await verifySourceDescriptors(request, snapshots, normalizedOptions.signal);
  await verifyEntryAnchor(request, snapshots, normalizedOptions.signal);
  const expectedRequestHash = await deriveCppCuteFrontendRequestHash(request, {
    limits: normalizedOptions.decodeLimits,
  });
  if (request.requestId !== `bg.cpp.frontend-request.sha256.${expectedRequestHash}`) {
    fail(
      "BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH",
      "$.requestId",
      `requestId must equal bg.cpp.frontend-request.sha256.${expectedRequestHash}`,
    );
  }
  const entry = request.entryRequests[0];
  if (entry === undefined) invalid("$.entryRequests", "prepared request lost entry request");
  const expectedEntryRequestId = await deriveCppCuteFrontendEntryRequestId(entry, {
    limits: normalizedOptions.decodeLimits,
  });
  if (entry.requestId !== expectedEntryRequestId) {
    hashMismatch("$.entryRequests[0].requestId", `entry request ID must equal ${expectedEntryRequestId}`);
  }
  const conformanceAssertionSha256 = normalizedOptions.detached.conformance === null
    ? null
    : await hashJson({
      domain: "browsergrad.compiler.cpp-cute.detached-conformance.v1",
      conformance: normalizedOptions.detached.conformance,
    }, "$.options.detached.conformance");
  throwIfAborted(normalizedOptions.signal);
  const sourceByteLength = request.files.reduce(
    (total, file) => total + wireIntegerToBigInt(file.byteLength),
    0n,
  );
  const prepared = Object.freeze({
    requestId: request.requestId,
    requestHash: expectedRequestHash,
    profileHash: profile.profileHash,
    compilationContractHash: request.compilationContractHash,
    mainVirtualPath: request.mainVirtualPath,
    sourceFileCount: request.files.length,
    sourceByteLength: encodeWireU64(sourceByteLength),
    entryRequestId: expectedEntryRequestId,
    declaredSourceReferenceStatementSha256:
      normalizedOptions.detached.declaredSourceReference?.statementSha256 ?? null,
    conformanceAssertionSha256,
  }) as PreparedCppCuteFrontendRequest;
  const orderedInputs = Object.freeze({
    compilerOptions: profileRecord.profile.language.options,
    availableIncludeRoots: profileRecord.profile.virtualFileSystem.includeRoots,
  });
  const publicRecord = Object.freeze({
    profile,
    request,
    orderedInputs,
    detached: normalizedOptions.detached,
  });
  PREPARED_REQUESTS.set(prepared, Object.freeze({ ...publicRecord, sourceSnapshots: snapshots }));
  return prepared;
}

export function unwrapPreparedCppCuteFrontendRequest(
  prepared: PreparedCppCuteFrontendRequest,
): PreparedCppCuteFrontendRequestRecord {
  return getPreparedRecord(prepared);
}

/** Returns a fresh copy; authoritative snapshot bytes never escape. */
export function copyPreparedCppCuteFrontendSourceBytes(
  prepared: PreparedCppCuteFrontendRequest,
  virtualPath: string,
): Uint8Array {
  const record = getPreparedRecord(prepared);
  const snapshot = record.sourceSnapshots.find((candidate) => candidate.virtualPath === virtualPath);
  if (snapshot === undefined) invalid("$.virtualPath", "source snapshot does not exist");
  return new Uint8Array(snapshot.bytes);
}

/** Returns one ordered fresh copy per request descriptor. */
export function copyPreparedCppCuteFrontendSourceSnapshots(
  prepared: PreparedCppCuteFrontendRequest,
): readonly CppCuteFrontendSourceSnapshotInput[] {
  const record = getPreparedRecord(prepared);
  return Object.freeze(record.sourceSnapshots.map((snapshot) => Object.freeze({
    virtualPath: snapshot.virtualPath,
    bytes: new Uint8Array(snapshot.bytes),
  })));
}

export async function deriveCppCuteFrontendSourceFileId(
  file: Pick<CppCuteFrontendRequestSourceFileV1, "role" | "virtualPath" | "contentSha256" | "byteLength" | "includeRootId">,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const digest = await hashJson({
    domain: "browsergrad.compiler.cpp-cute.source-file.v1",
    file: {
      role: file.role,
      virtualPath: file.virtualPath,
      contentSha256: file.contentSha256,
      byteLength: file.byteLength,
      includeRootId: file.includeRootId,
    },
  }, "$.file", options);
  return `bg.cpp.file.sha256.${digest}`;
}

export async function deriveCppCuteFrontendEntryRequestId(
  request: CppCuteFrontendEntryRequestV1,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const digest = await hashJson({
    domain: "browsergrad.compiler.cpp-cute.entry-request.v1",
    request: {
      kind: request.kind,
      declarationKind: request.declarationKind,
      anchor: request.anchor,
    },
  }, "$.entryRequests", options);
  return `bg.cpp.entry-request.sha256.${digest}`;
}

export async function deriveCppCuteFrontendRequestHash(
  request: CppCuteFrontendRequestV1 | CppCuteFrontendRequestBodyV1,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  return hashJson({
    domain: "browsergrad.compiler.cpp-cute.frontend-request.v1",
    request: {
      schema: request.schema,
      version: request.version,
      compilationContractHash: request.compilationContractHash,
      mainVirtualPath: request.mainVirtualPath,
      files: request.files,
      entryRequests: request.entryRequests,
      expectedArtifact: request.expectedArtifact,
      limits: request.limits,
    },
  }, "$.requestId", options);
}

function parseRequest(
  value: JsonValue,
  profile: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>["profile"],
): CppCuteFrontendRequestV1 {
  const object = closedObject(value, [
    "schema",
    "version",
    "requestId",
    "compilationContractHash",
    "mainVirtualPath",
    "files",
    "entryRequests",
    "expectedArtifact",
    "limits",
  ], "$");
  if (object.schema !== CPP_CUTE_FRONTEND_REQUEST_SCHEMA) {
    invalid("$.schema", `expected ${CPP_CUTE_FRONTEND_REQUEST_SCHEMA}`);
  }
  const versionObject = closedObject(field(object, "version", "$"), ["major", "minor"], "$.version");
  if (versionObject.major !== CPP_CUTE_FRONTEND_REQUEST_MAJOR ||
      versionObject.minor !== CPP_CUTE_FRONTEND_REQUEST_MINOR) {
    fail(
      "BG-COMPILER-CPP-CUTE-REQUEST-UNSUPPORTED-VERSION",
      "$.version",
      `closed request reader supports ${CPP_CUTE_FRONTEND_REQUEST_MAJOR}.${CPP_CUTE_FRONTEND_REQUEST_MINOR} only`,
    );
  }
  const mainVirtualPath = virtualPath(field(object, "mainVirtualPath", "$"), "$.mainVirtualPath");
  if (!profile.virtualFileSystem.sourceRoots.some((root) => belowRoot(mainVirtualPath, root))) {
    invalid("$.mainVirtualPath", "main source escapes profile source roots");
  }
  const files = arrayValue(field(object, "files", "$"), "$.files").map((entry, index) =>
    parseSourceFile(entry, `$.files[${index}]`, profile));
  validateSourceFiles(files, mainVirtualPath, profile.extractionLimits);
  const entryRequests = arrayValue(field(object, "entryRequests", "$"), "$.entryRequests").map((entry, index) =>
    parseEntryRequest(entry, `$.entryRequests[${index}]`, files, mainVirtualPath));
  if (entryRequests.length !== 1) invalid("$.entryRequests", "request requires exactly one supported entry");
  const expectedArtifact = parseExpectedArtifact(field(object, "expectedArtifact", "$"), "$.expectedArtifact");
  const limits = parseRequestLimits(field(object, "limits", "$"), "$.limits", profile.extractionLimits);
  validateRequestSourceLimits(files, limits);
  return deepFreezeJson({
    schema: CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
    version: { major: CPP_CUTE_FRONTEND_REQUEST_MAJOR, minor: CPP_CUTE_FRONTEND_REQUEST_MINOR },
    requestId: stableId(field(object, "requestId", "$"), "$.requestId", REQUEST_ID, "request"),
    compilationContractHash: sha256(field(object, "compilationContractHash", "$"), "$.compilationContractHash"),
    mainVirtualPath,
    files,
    entryRequests,
    expectedArtifact,
    limits,
  });
}

function validateRequestSourceLimits(
  files: readonly CppCuteFrontendRequestSourceFileV1[],
  limits: CppCuteFrontendRequestLimitsV1,
): void {
  if (files.length > limits.maxSourceFiles) {
    resource("$.files", `source file count exceeds request limit ${limits.maxSourceFiles}`);
  }
  const total = files.reduce((sum, file) => sum + wireIntegerToBigInt(file.byteLength), 0n);
  if (total > BigInt(limits.maxSourceBytes)) {
    resource("$.files", `source bytes exceed request limit ${limits.maxSourceBytes}`);
  }
}

function parseSourceFile(
  value: JsonValue,
  path: string,
  profile: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>["profile"],
): CppCuteFrontendRequestSourceFileV1 {
  const object = closedObject(value, [
    "fileId", "role", "virtualPath", "contentSha256", "byteLength", "includeRootId",
  ], path);
  if (object.role !== "main-source" && object.role !== "project-header") {
    invalid(`${path}.role`, "source role must be main-source or project-header");
  }
  const filePath = virtualPath(field(object, "virtualPath", path), `${path}.virtualPath`);
  if (!profile.virtualFileSystem.sourceRoots.some((root) => belowRoot(filePath, root))) {
    invalid(`${path}.virtualPath`, "source file escapes profile source roots");
  }
  const byteLength = parseWireU64(field(object, "byteLength", path), `${path}.byteLength`);
  if (wireIntegerToBigInt(byteLength) === 0n) invalid(`${path}.byteLength`, "source file must be nonempty");
  const rawIncludeRootId = field(object, "includeRootId", path);
  let includeRootId: string | null;
  if (object.role === "main-source") {
    if (rawIncludeRootId !== null) invalid(`${path}.includeRootId`, "main source includeRootId must be null");
    includeRootId = null;
  } else {
    includeRootId = boundedString(rawIncludeRootId, `${path}.includeRootId`, 128);
    const includeRoot = profile.virtualFileSystem.includeRoots.find((root) => root.includeRootId === includeRootId);
    if (includeRoot?.owner.kind !== "source" || !belowRoot(filePath, includeRoot.virtualPath)) {
      invalid(`${path}.includeRootId`, "project header must bind a containing source-owned include root");
    }
  }
  return {
    fileId: stableId(field(object, "fileId", path), `${path}.fileId`, FILE_ID, "source file"),
    role: object.role,
    virtualPath: filePath,
    contentSha256: sha256(field(object, "contentSha256", path), `${path}.contentSha256`),
    byteLength,
    includeRootId,
  };
}

function validateSourceFiles(
  files: readonly CppCuteFrontendRequestSourceFileV1[],
  mainVirtualPath: string,
  profileLimits: CppCuteFrontendExtractionLimits,
): void {
  if (files.length === 0) invalid("$.files", "request requires source files");
  if (files.length > profileLimits.maxSourceFiles) {
    resource("$.files", `source file count exceeds profile limit ${profileLimits.maxSourceFiles}`);
  }
  requireSortedUnique(files, (file) => file.virtualPath, "$.files");
  if (new Set(files.map((file) => file.fileId)).size !== files.length) invalid("$.files", "file IDs must be unique");
  const mains = files.filter((file) => file.role === "main-source");
  if (mains.length !== 1 || mains[0]?.virtualPath !== mainVirtualPath) {
    invalid("$.mainVirtualPath", "mainVirtualPath must identify only main-source file");
  }
  const total = files.reduce((sum, file) => sum + wireIntegerToBigInt(file.byteLength), 0n);
  if (total > BigInt(profileLimits.maxSourceBytes)) {
    resource("$.files", `source bytes exceed profile limit ${profileLimits.maxSourceBytes}`);
  }
}

function parseEntryRequest(
  value: JsonValue,
  path: string,
  files: readonly CppCuteFrontendRequestSourceFileV1[],
  mainVirtualPath: string,
): CppCuteFrontendEntryRequestV1 {
  const object = closedObject(value, ["requestId", "kind", "declarationKind", "anchor"], path);
  const selection = object.kind === "layout" && object.declarationKind === "variable"
    ? { kind: "layout" as const, declarationKind: "variable" as const }
    : object.kind === "view-copy" && object.declarationKind === "function"
      ? { kind: "view-copy" as const, declarationKind: "function" as const }
      : null;
  if (selection === null) {
    invalid(path, "entry kind must map layout to variable or view-copy to function");
  }
  const anchorPath = `${path}.anchor`;
  const anchor = closedObject(field(object, "anchor", path), [
    "virtualPath", "beginByte", "endByte", "tokenSha256",
  ], anchorPath);
  const anchorVirtualPath = virtualPath(field(anchor, "virtualPath", anchorPath), `${anchorPath}.virtualPath`);
  if (anchorVirtualPath !== mainVirtualPath) invalid(`${anchorPath}.virtualPath`, "entry must anchor in main source");
  const file = files.find((candidate) => candidate.virtualPath === anchorVirtualPath);
  if (file === undefined) invalid(`${anchorPath}.virtualPath`, "entry references missing source file");
  const beginByte = parseWireU64(field(anchor, "beginByte", anchorPath), `${anchorPath}.beginByte`);
  const endByte = parseWireU64(field(anchor, "endByte", anchorPath), `${anchorPath}.endByte`);
  const begin = wireIntegerToBigInt(beginByte);
  const end = wireIntegerToBigInt(endByte);
  if (begin >= end || end > wireIntegerToBigInt(file.byteLength) || end - begin > 256n) {
    invalid(anchorPath, "entry anchor must be nonempty in-file range of at most 256 bytes");
  }
  return {
    requestId: stableId(field(object, "requestId", path), `${path}.requestId`, ENTRY_REQUEST_ID, "entry request"),
    ...selection,
    anchor: {
      virtualPath: anchorVirtualPath,
      beginByte,
      endByte,
      tokenSha256: sha256(field(anchor, "tokenSha256", anchorPath), `${anchorPath}.tokenSha256`),
    },
  };
}

function parseExpectedArtifact(value: JsonValue, path: string): CppCuteFrontendExpectedArtifactV1 {
  const object = closedObject(value, ["schema", "version"], path);
  if (object.schema !== CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA) {
    invalid(`${path}.schema`, `expected ${CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA}`);
  }
  const versionPath = `${path}.version`;
  const version = closedObject(field(object, "version", path), ["major", "minor"], versionPath);
  if (version.major !== CPP_CUTE_FRONTEND_ARTIFACT_MAJOR || version.minor !== CPP_CUTE_FRONTEND_ARTIFACT_MINOR) {
    invalid(versionPath, `expected artifact version ${CPP_CUTE_FRONTEND_ARTIFACT_MAJOR}.${CPP_CUTE_FRONTEND_ARTIFACT_MINOR}`);
  }
  return {
    schema: CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA,
    version: { major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR, minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR },
  };
}

function parseRequestLimits(
  value: JsonValue,
  path: string,
  profileLimits: CppCuteFrontendExtractionLimits,
): CppCuteFrontendRequestLimitsV1 {
  const object = closedObject(value, CPP_CUTE_FRONTEND_SEMANTIC_EXTRACTION_LIMIT_KEYS, path);
  const entries = CPP_CUTE_FRONTEND_SEMANTIC_EXTRACTION_LIMIT_KEYS.map((key) => {
    const raw = field(object, key, path);
    if (typeof raw !== "number" || !Number.isSafeInteger(raw) || raw <= 0) {
      invalid(`${path}.${key}`, "limit must be positive safe integer");
    }
    if (raw > profileLimits[key]) {
      resource(`${path}.${key}`, `request limit cannot exceed profile limit ${profileLimits[key]}`);
    }
    return [key, raw] as const;
  });
  return Object.fromEntries(entries) as CppCuteFrontendRequestLimitsV1;
}

function snapshotSourceInputs(
  input: readonly CppCuteFrontendSourceSnapshotInput[],
  request: CppCuteFrontendRequestV1,
): readonly StoredSourceSnapshot[] {
  if (!Array.isArray(input)) invalid("$.sourceSnapshots", "source snapshots must be array");
  if (input.length !== request.files.length) invalid("$.sourceSnapshots", "source snapshot cardinality differs from request files");
  let copiedByteLength = 0;
  const result = input.map((entry, index) => {
    const path = `$.sourceSnapshots[${index}]`;
    const expectedFile = request.files[index];
    if (expectedFile === undefined) invalid(path, "source snapshot has no request descriptor");
    const descriptors = plainDataRecord(entry, path, ["virtualPath", "bytes"]);
    const rawPath = descriptorValue(descriptors, "virtualPath", path);
    if (typeof rawPath !== "string") invalid(`${path}.virtualPath`, "expected string");
    const filePath = virtualPath(rawPath, `${path}.virtualPath`);
    if (filePath !== expectedFile.virtualPath) {
      invalid(`${path}.virtualPath`, "snapshot path differs from sorted request file");
    }
    const rawBytes = descriptorValue(descriptors, "bytes", path);
    let inspection: ReturnType<typeof inspectUnsharedPlainUint8Array>;
    try {
      inspection = inspectUnsharedPlainUint8Array(rawBytes);
    } catch (error) {
      invalid(`${path}.bytes`, "bytes must be exact unshared plain Uint8Array", { cause: error });
    }
    const expectedByteLength = Number(wireIntegerToBigInt(expectedFile.byteLength));
    if (inspection.byteLength !== expectedByteLength) {
      hashMismatch(`$.files[${index}].byteLength`, "source descriptor byte length differs from snapshot");
    }
    copiedByteLength += inspection.byteLength;
    if (copiedByteLength > request.limits.maxSourceBytes) {
      resource("$.sourceSnapshots", `source snapshots exceed request limit ${request.limits.maxSourceBytes}`);
    }
    let copy: Uint8Array;
    try {
      copy = copyInspectedUnsharedUint8Array(rawBytes, inspection);
    } catch (error) {
      invalid(`${path}.bytes`, "bytes changed after exact inspection", { cause: error });
    }
    return Object.freeze({ virtualPath: filePath, bytes: copy });
  });
  requireSortedUnique(result, (entry) => entry.virtualPath, "$.sourceSnapshots");
  return Object.freeze(result);
}

async function verifySourceDescriptors(
  request: CppCuteFrontendRequestV1,
  snapshots: readonly StoredSourceSnapshot[],
  signal: AbortSignal | undefined,
): Promise<void> {
  for (const [index, file] of request.files.entries()) {
    throwIfAborted(signal);
    const snapshot = snapshots[index];
    if (snapshot?.virtualPath !== file.virtualPath) {
      invalid(`$.sourceSnapshots[${index}].virtualPath`, "snapshot path differs from sorted request file");
    }
    if (snapshot.bytes.byteLength !== Number(wireIntegerToBigInt(file.byteLength))) {
      hashMismatch(`$.files[${index}].byteLength`, "source descriptor byte length differs from snapshot");
    }
    const contentSha256 = await hashBytes(snapshot.bytes, `$.files[${index}].contentSha256`);
    if (file.contentSha256 !== contentSha256) {
      hashMismatch(`$.files[${index}].contentSha256`, "source descriptor hash differs from snapshot");
    }
    const fileId = await deriveCppCuteFrontendSourceFileId(file);
    if (file.fileId !== fileId) hashMismatch(`$.files[${index}].fileId`, `file ID must equal ${fileId}`);
  }
}

async function verifyEntryAnchor(
  request: CppCuteFrontendRequestV1,
  snapshots: readonly StoredSourceSnapshot[],
  signal: AbortSignal | undefined,
): Promise<void> {
  throwIfAborted(signal);
  const entry = request.entryRequests[0];
  if (entry === undefined) invalid("$.entryRequests", "entry missing");
  const snapshot = snapshots.find((candidate) => candidate.virtualPath === entry.anchor.virtualPath);
  if (snapshot === undefined) invalid("$.entryRequests[0].anchor.virtualPath", "anchor snapshot missing");
  const start = Number(wireIntegerToBigInt(entry.anchor.beginByte));
  const end = Number(wireIntegerToBigInt(entry.anchor.endByte));
  const tokenSha256 = await hashBytes(snapshot.bytes.subarray(start, end), "$.entryRequests[0].anchor.tokenSha256");
  if (entry.anchor.tokenSha256 !== tokenSha256) {
    hashMismatch("$.entryRequests[0].anchor.tokenSha256", "anchor token hash differs from source snapshot");
  }
}

function normalizeOptions(options: PrepareCppCuteFrontendRequestOptions): {
  readonly decodeLimits: DecodeLimits;
  readonly signal: AbortSignal | undefined;
  readonly detached: CppCuteFrontendDetachedExpectationsV1;
} {
  const descriptors = plainDataRecord(options, "$.options", ["detached", "limits", "signal"]);
  const rawLimits = optionalDescriptorValue(descriptors, "limits");
  let decodeLimits: DecodeLimits;
  try {
    decodeLimits = resolveDecodeLimits(rawLimits as Partial<DecodeLimits> | undefined);
  } catch (error) {
    resource("$.options.limits", "invalid JSON decode limits", { cause: error });
  }
  const rawSignal = optionalDescriptorValue(descriptors, "signal");
  if (rawSignal !== undefined && !(rawSignal instanceof AbortSignal)) {
    invalid("$.options.signal", "signal must be AbortSignal");
  }
  const rawDetached = optionalDescriptorValue(descriptors, "detached");
  const detached = rawDetached === undefined
    ? deepFreezeJson({ declaredSourceReference: null, conformance: null })
    : parseDetached(rawDetached);
  return Object.freeze({ decodeLimits, signal: rawSignal as AbortSignal | undefined, detached });
}

function parseDetached(value: unknown): CppCuteFrontendDetachedExpectationsV1 {
  try {
    assertJsonValue(value);
  } catch (error) {
    invalid("$.options.detached", "detached expectations must be closed JSON", { cause: error });
  }
  const object = closedObject(value as JsonValue, ["declaredSourceReference", "conformance"], "$.options.detached");
  const rawSourceReference = field(object, "declaredSourceReference", "$.options.detached");
  const declaredSourceReference = rawSourceReference === null ? null : (() => {
    const record = closedObject(
      rawSourceReference,
      ["statementSha256"],
      "$.options.detached.declaredSourceReference",
    );
    return {
      statementSha256: sha256(
        field(record, "statementSha256", "$.options.detached.declaredSourceReference"),
        "$.options.detached.declaredSourceReference.statementSha256",
      ),
    };
  })();
  const rawConformance = field(object, "conformance", "$.options.detached");
  const conformance = rawConformance === null ? null : (() => {
    const path = "$.options.detached.conformance";
    const record = closedObject(rawConformance, [
      "expectedArtifactSha256", "expectedOpenedHeaderSetSha256", "expectedInputClosureSha256",
    ], path);
    return {
      expectedArtifactSha256: nullableSha256(field(record, "expectedArtifactSha256", path), `${path}.expectedArtifactSha256`),
      expectedOpenedHeaderSetSha256: nullableSha256(
        field(record, "expectedOpenedHeaderSetSha256", path),
        `${path}.expectedOpenedHeaderSetSha256`,
      ),
      expectedInputClosureSha256: nullableSha256(
        field(record, "expectedInputClosureSha256", path),
        `${path}.expectedInputClosureSha256`,
      ),
    };
  })();
  return deepFreezeJson({ declaredSourceReference, conformance });
}

function plainDataRecord(
  value: unknown,
  path: string,
  allowedFields: readonly string[],
): PropertyDescriptorMap {
  if (typeof value !== "object" || value === null || Array.isArray(value)) invalid(path, "expected plain object");
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) invalid(path, "expected plain object");
  const descriptors = Object.getOwnPropertyDescriptors(value);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== "string" || !allowedFields.includes(key)) invalid(path, "object contains unknown fields");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "fields must be enumerable data properties without accessors");
    }
  }
  return descriptors;
}

function descriptorValue(descriptors: PropertyDescriptorMap, key: string, path: string): unknown {
  const descriptor = descriptors[key];
  if (descriptor === undefined || !("value" in descriptor)) invalid(`${path}.${key}`, "field is required");
  return descriptor.value;
}

function optionalDescriptorValue(descriptors: PropertyDescriptorMap, key: string): unknown {
  const descriptor = descriptors[key];
  return descriptor !== undefined && "value" in descriptor ? descriptor.value : undefined;
}

function getPreparedRecord(prepared: PreparedCppCuteFrontendRequest): StoredCppCuteFrontendRequest {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const record = PREPARED_REQUESTS.get(prepared as object);
  if (record === undefined) unverified();
  return record;
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

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function virtualPath(value: JsonValue, path: string): string {
  const result = boundedString(value, path, 1_024);
  try {
    validateCppCuteVirtualPath(result, path);
  } catch (error) {
    invalid(path, "invalid normalized absolute virtual path", { cause: error });
  }
  return result;
}

function stableId(value: JsonValue, path: string, pattern: RegExp, name: string): string {
  const result = stringValue(value, path);
  if (!pattern.test(result)) invalid(path, `${name} ID has invalid syntax`);
  return result;
}

function nullableSha256(value: JsonValue, path: string): string | null {
  return value === null ? null : sha256(value, path);
}

function sha256(value: JsonValue, path: string): string {
  const result = stringValue(value, path);
  if (!SHA256_HEX.test(result)) invalid(path, "SHA-256 must be 64 lowercase hexadecimal digits");
  return result;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(path, "expected string");
  return value;
}

function boundedString(value: JsonValue, path: string, maxBytes: number): string {
  const result = stringValue(value, path);
  if (result.length === 0 || result.includes("\0") || TEXT_ENCODER.encode(result).byteLength > maxBytes) {
    invalid(path, `string must be nonempty, NUL-free, and at most ${maxBytes} UTF-8 bytes`);
  }
  return result;
}

function requireSortedUnique<T>(values: readonly T[], key: (value: T) => string, path: string): void {
  for (let index = 1; index < values.length; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    if (previous === undefined || current === undefined || key(previous).localeCompare(key(current)) >= 0) {
      invalid(path, "set-like records must be strictly sorted and unique");
    }
  }
}

function belowRoot(filePath: string, root: string): boolean {
  return root === "/" ? filePath !== "/" : filePath.startsWith(`${root}/`);
}

async function hashBytes(bytes: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(bytes);
  } catch (error) {
    fail("BG-COMPILER-CPP-CUTE-REQUEST-HASH-UNAVAILABLE", path, "SHA-256 unavailable", { cause: error });
  }
}

async function hashJson(
  value: JsonValue,
  path: string,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  try {
    return await hashCanonicalJson(value, options);
  } catch (error) {
    if (error instanceof SemanticSchemaError && error.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit) {
      resource(path, "canonical hashing exceeded limits", { cause: error });
    }
    fail("BG-COMPILER-CPP-CUTE-REQUEST-HASH-UNAVAILABLE", path, "canonical SHA-256 unavailable", { cause: error });
  }
}

function translateSchemaError(error: unknown, message: string): never {
  if (error instanceof CppCuteFrontendRequestError) throw error;
  if (error instanceof SemanticSchemaError && error.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit) {
    resource(error.diagnostic.path ?? "$", error.message, { cause: error });
  }
  invalid("$", message, { cause: error });
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) cancelled("$.signal", "request preparation cancelled");
}

function unverified(): never {
  fail("BG-COMPILER-CPP-CUTE-REQUEST-UNVERIFIED", "$", "expected exact prepared frontend request authority");
}

function cancelled(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-REQUEST-CANCELLED", path, message);
}

function hashMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-REQUEST-HASH-MISMATCH", path, message);
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-REQUEST-RESOURCE-LIMIT", path, message, options);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-REQUEST-INVALID", path, message, options);
}

function fail(
  code: CppCuteFrontendRequestErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteFrontendRequestError(code, path, message, options);
}

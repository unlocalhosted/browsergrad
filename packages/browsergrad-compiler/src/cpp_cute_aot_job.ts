import {
  assertJsonValue,
  canonicalizeJson,
  deepFreezeJson,
  encodeWireU64,
  hashCanonicalJson,
  isJsonObject,
  parseWireU64,
  resolveDecodeLimits,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  wireIntegerToBigInt,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  deriveCppCuteStableId,
} from "./cpp_cute_frontend_artifact.js";
import {
  unwrapPreparedCppCuteAotFrontendProfile,
  validateCppCuteVirtualPath,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
  CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA,
} from "./cpp_cute_frontend_types.js";

export const CPP_CUTE_AOT_JOB_SCHEMA = "browsergrad.compiler.cpp-cute.aot-job";
export const CPP_CUTE_AOT_JOB_MAJOR = 2;
export const CPP_CUTE_AOT_JOB_MINOR = 0;

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const JOB_ID = /^bg\.cpp\.aot-job\.sha256\.[0-9a-f]{64}$/u;
const REQUEST_ID = /^bg\.cpp\.entry-request\.sha256\.[0-9a-f]{64}$/u;
const FILE_ID = /^bg\.cpp\.file\.sha256\.[0-9a-f]{64}$/u;
const PREPARED_JOBS = new WeakMap<object, PreparedCppCuteAotJobRecord>();

export interface CppCuteAotJobVersionV2 extends JsonObject {
  readonly major: typeof CPP_CUTE_AOT_JOB_MAJOR;
  readonly minor: typeof CPP_CUTE_AOT_JOB_MINOR;
}

export interface CppCuteAotSourceFileV2 extends JsonObject {
  readonly fileId: string;
  readonly role: "main-source" | "project-header";
  readonly virtualPath: string;
  readonly contentSha256: string;
  readonly byteLength: WireU64;
}

export interface CppCuteAotSourceAnchorV2 extends JsonObject {
  readonly virtualPath: string;
  readonly beginByte: WireU64;
  readonly endByte: WireU64;
  readonly tokenSha256: string;
}

export interface CppCuteAotEntryRequestV2 extends JsonObject {
  readonly requestId: string;
  readonly kind: "layout";
  readonly declarationKind: "variable";
  readonly anchor: CppCuteAotSourceAnchorV2;
}

export interface CppCuteAotSourceRevisionV2 extends JsonObject {
  readonly algorithm: "git-sha1" | "git-sha256";
  readonly value: string;
}

export interface CppCuteAotSourceIdentityV2 extends JsonObject {
  readonly repository: string;
  readonly revision: CppCuteAotSourceRevisionV2;
}

export interface CppCuteAotExpectedOutputV2 extends JsonObject {
  readonly schema: typeof CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA;
  readonly version: {
    readonly major: typeof CPP_CUTE_FRONTEND_ARTIFACT_MAJOR;
    readonly minor: typeof CPP_CUTE_FRONTEND_ARTIFACT_MINOR;
  };
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly inputClosureSha256: string;
}

export interface CppCuteAotJobV2 extends JsonObject {
  readonly schema: typeof CPP_CUTE_AOT_JOB_SCHEMA;
  readonly version: CppCuteAotJobVersionV2;
  readonly jobId: string;
  readonly profileHash: string;
  readonly source: CppCuteAotSourceIdentityV2;
  readonly mainVirtualPath: string;
  readonly files: readonly CppCuteAotSourceFileV2[];
  readonly entryRequests: readonly CppCuteAotEntryRequestV2[];
  readonly expectedOutput: CppCuteAotExpectedOutputV2;
}

export interface CppCuteAotJobBodyV2 extends JsonObject {
  readonly schema: typeof CPP_CUTE_AOT_JOB_SCHEMA;
  readonly version: CppCuteAotJobVersionV2;
  readonly profileHash: string;
  readonly source: CppCuteAotSourceIdentityV2;
  readonly mainVirtualPath: string;
  readonly files: readonly CppCuteAotSourceFileV2[];
  readonly entryRequests: readonly CppCuteAotEntryRequestV2[];
  readonly expectedOutput: CppCuteAotExpectedOutputV2;
}

declare const preparedCppCuteAotJobBrand: unique symbol;

export interface PreparedCppCuteAotJob {
  readonly [preparedCppCuteAotJobBrand]: true;
  readonly jobId: string;
  readonly profileHash: string;
  readonly mainVirtualPath: string;
  readonly sourceFileCount: number;
  readonly sourceBytes: WireU64;
  readonly entryRequestId: string;
}

export interface PreparedCppCuteAotJobRecord {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly job: CppCuteAotJobV2;
}

export interface PrepareCppCuteAotJobOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly signal?: AbortSignal;
}

export type CppCuteAotJobErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-JOB-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-JOB-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-JOB-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-AOT-JOB-PROFILE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-JOB-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-JOB-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-AOT-JOB-UNVERIFIED";

export class CppCuteAotJobError extends Error {
  constructor(
    readonly code: CppCuteAotJobErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteAotJobError";
  }
}

/**
 * Verifies one immutable request for the pinned AOT frontend. The request may
 * select declarations, but cannot supply compiler flags, commands, host paths,
 * environment variables, include roots, or output destinations.
 */
export async function prepareCppCuteAotJob(
  profile: PreparedCppCuteFrontendProfile,
  value: unknown,
  options: PrepareCppCuteAotJobOptions = {},
): Promise<PreparedCppCuteAotJob> {
  const profileRecord = unwrapPreparedCppCuteAotFrontendProfile(profile);
  const limits = normalizeOptions(options);
  throwIfAborted(options.signal);
  let job: CppCuteAotJobV2;
  try {
    assertJsonValue(value, { limits });
    job = parseJob(
      value,
      profileRecord.profile.virtualFileSystem.sourceRoots,
      profile.extractionLimits,
    );
    canonicalizeJson(job, { limits });
  } catch (error) {
    if (error instanceof CppCuteAotJobError) throw error;
    if (error instanceof SemanticSchemaError && error.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit) {
      resource(error.diagnostic.path ?? "$", error.message, { cause: error });
    }
    fail("BG-COMPILER-CPP-CUTE-AOT-JOB-INVALID", "$", "AOT job is not closed canonical JSON", { cause: error });
  }
  if (job.profileHash !== profile.profileHash) {
    fail(
      "BG-COMPILER-CPP-CUTE-AOT-JOB-PROFILE-MISMATCH",
      "$.profileHash",
      "AOT job profile hash differs from the exact prepared frontend profile",
    );
  }
  if (job.expectedOutput.headerSetSha256 !== profile.expectedHeaderSetSha256) {
    fail(
      "BG-COMPILER-CPP-CUTE-AOT-JOB-PROFILE-MISMATCH",
      "$.expectedOutput.headerSetSha256",
      "requested header set differs from the exact prepared frontend profile",
    );
  }
  for (const [index, file] of job.files.entries()) {
    const expectedFileId = await deriveCppCuteAotSourceFileId(file, { limits });
    if (file.fileId !== expectedFileId) {
      fail(
        "BG-COMPILER-CPP-CUTE-AOT-JOB-HASH-MISMATCH",
        `$.files[${index}].fileId`,
        `source file ID must equal ${expectedFileId}`,
      );
    }
  }
  const request = job.entryRequests[0];
  if (request === undefined) {
    fail("BG-COMPILER-CPP-CUTE-AOT-JOB-INVALID", "$", "verified AOT job lost its entry request");
  }
  const expectedRequestId = await deriveCppCuteAotEntryRequestId(request, { limits });
  if (request.requestId !== expectedRequestId) {
    fail(
      "BG-COMPILER-CPP-CUTE-AOT-JOB-HASH-MISMATCH",
      "$.entryRequests[0].requestId",
      `entry request ID must equal ${expectedRequestId}`,
    );
  }
  const expectedJobId = await deriveCppCuteAotJobId(job, { limits });
  if (job.jobId !== expectedJobId) {
    fail("BG-COMPILER-CPP-CUTE-AOT-JOB-HASH-MISMATCH", "$.jobId", `job ID must equal ${expectedJobId}`);
  }
  throwIfAborted(options.signal);
  const sourceBytes = job.files.reduce((total, file) => total + wireIntegerToBigInt(file.byteLength), 0n);
  const prepared = Object.freeze({
    jobId: expectedJobId,
    profileHash: job.profileHash,
    mainVirtualPath: job.mainVirtualPath,
    sourceFileCount: job.files.length,
    sourceBytes: encodeWireU64(sourceBytes),
    entryRequestId: expectedRequestId,
  }) as PreparedCppCuteAotJob;
  PREPARED_JOBS.set(prepared, Object.freeze({ profile, job }));
  return prepared;
}

export function unwrapPreparedCppCuteAotJob(prepared: PreparedCppCuteAotJob): PreparedCppCuteAotJobRecord {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const record = PREPARED_JOBS.get(prepared as object);
  if (record === undefined) unverified();
  return record;
}

export async function deriveCppCuteAotEntryRequestId(
  request: CppCuteAotEntryRequestV2,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const digest = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.aot-entry-request.v2",
    request: {
      kind: request.kind,
      declarationKind: request.declarationKind,
      anchor: request.anchor,
    },
  }, options);
  return `bg.cpp.entry-request.sha256.${digest}`;
}

export async function deriveCppCuteAotSourceFileId(
  file: CppCuteAotSourceFileV2,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  return deriveCppCuteStableId("file", {
    role: file.role,
    virtualPath: file.virtualPath,
    contentSha256: file.contentSha256,
    byteLength: file.byteLength,
  }, options);
}

export async function deriveCppCuteAotJobId(
  job: CppCuteAotJobV2 | CppCuteAotJobBodyV2,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const digest = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.aot-job.v2",
    job: {
      schema: job.schema,
      version: job.version,
      profileHash: job.profileHash,
      source: job.source,
      mainVirtualPath: job.mainVirtualPath,
      files: job.files,
      entryRequests: job.entryRequests,
      expectedOutput: job.expectedOutput,
    },
  }, options);
  return `bg.cpp.aot-job.sha256.${digest}`;
}

function parseJob(
  value: JsonValue,
  sourceRoots: readonly string[],
  extractionLimits: PreparedCppCuteFrontendProfile["extractionLimits"],
): CppCuteAotJobV2 {
  const object = closedObject(value, [
    "schema",
    "version",
    "jobId",
    "profileHash",
    "source",
    "mainVirtualPath",
    "files",
    "entryRequests",
    "expectedOutput",
  ], "$");
  if (object.schema !== CPP_CUTE_AOT_JOB_SCHEMA) invalid("$.schema", `expected ${CPP_CUTE_AOT_JOB_SCHEMA}`);
  const version = parseVersion(field(object, "version", "$"), "$.version");
  const profileHash = sha256(field(object, "profileHash", "$"), "$.profileHash");
  const source = parseSourceIdentity(field(object, "source", "$"), "$.source");
  const mainVirtualPath = virtualPath(field(object, "mainVirtualPath", "$"), "$.mainVirtualPath");
  const files = arrayValue(field(object, "files", "$"), "$.files").map((entry, index) => (
    parseFile(entry, `$.files[${index}]`, sourceRoots)
  ));
  if (files.length === 0) invalid("$.files", "AOT job requires at least one source file");
  if (files.length > extractionLimits.maxSourceFiles) {
    resource("$.files", `source file count exceeds profile limit ${extractionLimits.maxSourceFiles}`);
  }
  requireSortedUnique(files, (file) => file.virtualPath, "$.files");
  if (new Set(files.map((file) => file.fileId)).size !== files.length) {
    invalid("$.files", "source file IDs must be unique");
  }
  const totalBytes = files.reduce((total, file) => total + wireIntegerToBigInt(file.byteLength), 0n);
  if (totalBytes > BigInt(extractionLimits.maxSourceBytes)) {
    resource("$.files", `source bytes exceed profile limit ${extractionLimits.maxSourceBytes}`);
  }
  const mains = files.filter((file) => file.role === "main-source");
  if (mains.length !== 1 || mains[0]?.virtualPath !== mainVirtualPath) {
    invalid("$.mainVirtualPath", "mainVirtualPath must identify the job's only main-source file");
  }
  const entryRequests = arrayValue(field(object, "entryRequests", "$"), "$.entryRequests").map((entry, index) => (
    parseEntryRequest(entry, `$.entryRequests[${index}]`, files, mainVirtualPath)
  ));
  if (entryRequests.length !== 1) {
    invalid("$.entryRequests", "AOT layout-tracer profile requires exactly one explicit declaration request");
  }
  const expectedOutput = parseExpectedOutput(field(object, "expectedOutput", "$"), "$.expectedOutput");
  return deepFreezeJson({
    schema: CPP_CUTE_AOT_JOB_SCHEMA,
    version,
    jobId: stableId(field(object, "jobId", "$"), "$.jobId", JOB_ID, "AOT job"),
    profileHash,
    source,
    mainVirtualPath,
    files,
    entryRequests,
    expectedOutput,
  });
}

function parseVersion(value: JsonValue, path: string): CppCuteAotJobVersionV2 {
  const object = closedObject(value, ["major", "minor"], path);
  if (object.major !== CPP_CUTE_AOT_JOB_MAJOR || object.minor !== CPP_CUTE_AOT_JOB_MINOR) {
    fail(
      "BG-COMPILER-CPP-CUTE-AOT-JOB-UNSUPPORTED-VERSION",
      path,
      `closed AOT job reader supports ${CPP_CUTE_AOT_JOB_MAJOR}.${CPP_CUTE_AOT_JOB_MINOR} only`,
    );
  }
  return { major: CPP_CUTE_AOT_JOB_MAJOR, minor: CPP_CUTE_AOT_JOB_MINOR };
}

function parseFile(
  value: JsonValue,
  path: string,
  sourceRoots: readonly string[],
): CppCuteAotSourceFileV2 {
  const object = closedObject(value, ["fileId", "role", "virtualPath", "contentSha256", "byteLength"], path);
  if (object.role !== "main-source" && object.role !== "project-header") {
    invalid(`${path}.role`, "source role must be main-source or project-header");
  }
  const filePath = virtualPath(field(object, "virtualPath", path), `${path}.virtualPath`);
  if (!sourceRoots.some((root) => isVirtualPathBelow(filePath, root))) {
    invalid(`${path}.virtualPath`, "source file escapes every profile source root");
  }
  const byteLength = parseWireU64(field(object, "byteLength", path), `${path}.byteLength`);
  if (wireIntegerToBigInt(byteLength) === 0n) invalid(`${path}.byteLength`, "source files must be nonempty");
  return {
    fileId: stableId(field(object, "fileId", path), `${path}.fileId`, FILE_ID, "source file"),
    role: object.role,
    virtualPath: filePath,
    contentSha256: sha256(field(object, "contentSha256", path), `${path}.contentSha256`),
    byteLength,
  };
}

function parseEntryRequest(
  value: JsonValue,
  path: string,
  files: readonly CppCuteAotSourceFileV2[],
  mainVirtualPath: string,
): CppCuteAotEntryRequestV2 {
  const object = closedObject(value, ["requestId", "kind", "declarationKind", "anchor"], path);
  if (object.kind !== "layout" || object.declarationKind !== "variable") {
    invalid(path, "AOT job v2 selects one layout variable declaration only");
  }
  const anchorPath = `${path}.anchor`;
  const anchorObject = closedObject(field(object, "anchor", path), [
    "virtualPath",
    "beginByte",
    "endByte",
    "tokenSha256",
  ], anchorPath);
  const anchorVirtualPath = virtualPath(field(anchorObject, "virtualPath", anchorPath), `${anchorPath}.virtualPath`);
  if (anchorVirtualPath !== mainVirtualPath) {
    invalid(`${anchorPath}.virtualPath`, "initial layout request must anchor in the main source file");
  }
  const file = files.find((candidate) => candidate.virtualPath === anchorVirtualPath);
  if (file === undefined) invalid(`${anchorPath}.virtualPath`, "entry anchor references a missing source file");
  const beginByte = parseWireU64(field(anchorObject, "beginByte", anchorPath), `${anchorPath}.beginByte`);
  const endByte = parseWireU64(field(anchorObject, "endByte", anchorPath), `${anchorPath}.endByte`);
  const begin = wireIntegerToBigInt(beginByte);
  const end = wireIntegerToBigInt(endByte);
  if (begin >= end || end > wireIntegerToBigInt(file.byteLength) || end - begin > 256n) {
    invalid(anchorPath, "entry anchor must be a nonempty in-file token range of at most 256 bytes");
  }
  return {
    requestId: stableId(
      field(object, "requestId", path),
      `${path}.requestId`,
      REQUEST_ID,
      "entry request",
    ),
    kind: "layout",
    declarationKind: "variable",
    anchor: {
      virtualPath: anchorVirtualPath,
      beginByte,
      endByte,
      tokenSha256: sha256(field(anchorObject, "tokenSha256", anchorPath), `${anchorPath}.tokenSha256`),
    },
  };
}

function parseSourceIdentity(value: JsonValue, path: string): CppCuteAotSourceIdentityV2 {
  const object = closedObject(value, ["repository", "revision"], path);
  const repository = boundedString(field(object, "repository", path), `${path}.repository`, 1_024);
  let parsed: URL;
  try {
    parsed = new URL(repository);
  } catch (error) {
    invalid(`${path}.repository`, "source repository must be an absolute URL", { cause: error });
  }
  if (
    parsed.protocol !== "https:"
    || parsed.username !== ""
    || parsed.password !== ""
    || parsed.search !== ""
    || parsed.hash !== ""
    || parsed.pathname === "/"
    || parsed.pathname.endsWith("/")
    || `${parsed.origin}${parsed.pathname}` !== repository
  ) {
    invalid(`${path}.repository`, "source repository must be canonical credential-free HTTPS without query, fragment, or trailing slash");
  }
  const revisionPath = `${path}.revision`;
  const revision = closedObject(field(object, "revision", path), ["algorithm", "value"], revisionPath);
  if (revision.algorithm !== "git-sha1" && revision.algorithm !== "git-sha256") {
    invalid(`${revisionPath}.algorithm`, "source revision algorithm must be git-sha1 or git-sha256");
  }
  const revisionValue = stringValue(field(revision, "value", revisionPath), `${revisionPath}.value`);
  const digits = revision.algorithm === "git-sha1" ? 40 : 64;
  if (!new RegExp(`^[0-9a-f]{${digits}}$`, "u").test(revisionValue)) {
    invalid(`${revisionPath}.value`, `source revision must contain ${digits} lowercase hexadecimal digits`);
  }
  return {
    repository,
    revision: { algorithm: revision.algorithm, value: revisionValue },
  };
}

function parseExpectedOutput(value: JsonValue, path: string): CppCuteAotExpectedOutputV2 {
  const object = closedObject(value, [
    "schema",
    "version",
    "sourceSetSha256",
    "headerSetSha256",
    "inputClosureSha256",
  ], path);
  if (object.schema !== CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA) {
    invalid(`${path}.schema`, `expected output schema must be ${CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA}`);
  }
  const versionPath = `${path}.version`;
  const version = closedObject(field(object, "version", path), ["major", "minor"], versionPath);
  if (version.major !== CPP_CUTE_FRONTEND_ARTIFACT_MAJOR || version.minor !== CPP_CUTE_FRONTEND_ARTIFACT_MINOR) {
    invalid(versionPath, `expected output version must be ${CPP_CUTE_FRONTEND_ARTIFACT_MAJOR}.${CPP_CUTE_FRONTEND_ARTIFACT_MINOR}`);
  }
  return {
    schema: CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA,
    version: { major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR, minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR },
    sourceSetSha256: sha256(field(object, "sourceSetSha256", path), `${path}.sourceSetSha256`),
    headerSetSha256: sha256(field(object, "headerSetSha256", path), `${path}.headerSetSha256`),
    inputClosureSha256: sha256(field(object, "inputClosureSha256", path), `${path}.inputClosureSha256`),
  };
}

function normalizeOptions(options: PrepareCppCuteAotJobOptions): DecodeLimits {
  if (typeof options !== "object" || options === null || Array.isArray(options)) invalid("$options", "options must be a plain object");
  const prototype = Object.getPrototypeOf(options);
  if (prototype !== Object.prototype && prototype !== null) invalid("$options", "options must be a plain object");
  const descriptors = Object.getOwnPropertyDescriptors(options);
  for (const key of Reflect.ownKeys(options)) {
    if (typeof key !== "string" || (key !== "limits" && key !== "signal")) invalid("$options", "options contain unknown fields");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`$options.${key}`, "options require enumerable data properties without accessors");
    }
  }
  try {
    return resolveDecodeLimits(options.limits);
  } catch (error) {
    resource("$options.limits", "invalid semantic decode limits", { cause: error });
  }
}

function isVirtualPathBelow(filePath: string, root: string): boolean {
  return root === "/" ? filePath !== "/" : filePath.startsWith(`${root}/`);
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

function sha256(value: JsonValue, path: string): string {
  const result = stringValue(value, path);
  if (!SHA256_HEX.test(result)) invalid(path, "SHA-256 must be 64 lowercase hexadecimal digits");
  return result;
}

function boundedString(value: JsonValue, path: string, maximumBytes: number): string {
  const result = stringValue(value, path);
  if (result.length === 0 || result.includes("\0") || new TextEncoder().encode(result).byteLength > maximumBytes) {
    invalid(path, `string must be nonempty, NUL-free, and at most ${maximumBytes} UTF-8 bytes`);
  }
  return result;
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

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(path, "expected string");
  return value;
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

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) fail("BG-COMPILER-CPP-CUTE-AOT-JOB-CANCELLED", "$.signal", "AOT job preparation was aborted");
}

function unverified(): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-JOB-UNVERIFIED", "$", "expected an instance-authorized prepared C++/CuTe AOT job");
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-JOB-RESOURCE-LIMIT", path, message, options);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-JOB-INVALID", path, message, options);
}

function fail(
  code: CppCuteAotJobErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteAotJobError(code, path, message, options);
}

import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  assertJsonValue,
  canonicalizeJson,
  deepFreezeJson,
  hashCanonicalJson,
  isJsonObject,
  resolveDecodeLimits,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyPreparedCppCuteFrontendSourceSnapshots,
  unwrapPreparedCppCuteFrontendRequest,
  type CppCuteFrontendSourceSnapshotInput,
  type PreparedCppCuteFrontendRequest,
} from "./cpp_cute_frontend_request.js";
import {
  unwrapPreparedCppCuteAotFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";

export const CPP_CUTE_AOT_RUN_METADATA_SCHEMA = "browsergrad.compiler.cpp-cute.aot-run-metadata";
export const CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA =
  "browsergrad.compiler.cpp-cute.git-source-reference";
export const CPP_CUTE_AOT_RUN_METADATA_MAJOR = 1;
export const CPP_CUTE_AOT_RUN_METADATA_MINOR = 0;

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const RUN_METADATA_ID = /^bg\.cpp\.aot-run-metadata\.sha256\.[0-9a-f]{64}$/u;
const PREPARED_METADATA = new WeakMap<object, StoredCppCuteAotRunMetadata>();

export interface CppCuteGitRevisionV1 extends JsonObject {
  readonly algorithm: "git-sha1" | "git-sha256";
  readonly value: string;
}

export interface CppCuteGitSourceReferenceStatementV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA;
  readonly version: JsonObject & {
    readonly major: 1;
    readonly minor: 0;
  };
  readonly repository: string;
  readonly revision: CppCuteGitRevisionV1;
}

export interface CppCuteAotRunMetadataBodyV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_AOT_RUN_METADATA_SCHEMA;
  readonly version: JsonObject & {
    readonly major: typeof CPP_CUTE_AOT_RUN_METADATA_MAJOR;
    readonly minor: typeof CPP_CUTE_AOT_RUN_METADATA_MINOR;
  };
  readonly profileHash: string;
  readonly requestId: string;
  readonly declaredSourceReference: JsonObject & {
    readonly statementSha256: string;
    readonly statement: CppCuteGitSourceReferenceStatementV1;
  };
}

export interface CppCuteAotRunMetadataV1 extends CppCuteAotRunMetadataBodyV1 {
  readonly runMetadataId: string;
}

export interface PrepareCppCuteAotRunMetadataOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly signal?: AbortSignal;
}

declare const preparedCppCuteAotRunMetadataBrand: unique symbol;

/** AOT-only execution metadata composed around producer-neutral request authority. */
export interface PreparedCppCuteAotRunMetadata {
  readonly [preparedCppCuteAotRunMetadataBrand]: true;
  readonly runMetadataId: string;
  readonly metadataHash: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly declaredSourceReferenceStatementSha256: string;
  readonly sourceFileCount: number;
}

export interface PreparedCppCuteAotRunMetadataRecord {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly metadata: CppCuteAotRunMetadataV1;
}

type StoredCppCuteAotRunMetadata = PreparedCppCuteAotRunMetadataRecord;

export type CppCuteAotRunMetadataErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-PROFILE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-REQUEST-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-SOURCE-REFERENCE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-UNVERIFIED";

export class CppCuteAotRunMetadataError extends Error {
  constructor(
    readonly code: CppCuteAotRunMetadataErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteAotRunMetadataError";
  }
}

export async function prepareCppCuteAotRunMetadata(
  request: PreparedCppCuteFrontendRequest,
  value: unknown,
  options: PrepareCppCuteAotRunMetadataOptions = {},
): Promise<PreparedCppCuteAotRunMetadata> {
  const normalizedOptions = normalizeOptions(options);
  throwIfAborted(normalizedOptions.signal);
  const requestRecord = unwrapPreparedCppCuteFrontendRequest(request);
  const profile = requestRecord.profile;
  unwrapPreparedCppCuteAotFrontendProfile(profile);
  const metadata = parseMetadata(value, normalizedOptions.limits);
  if (metadata.profileHash !== profile.profileHash) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-PROFILE-MISMATCH",
      "$.profileHash",
      "run metadata profile hash differs from exact prepared AOT profile",
    );
  }
  if (metadata.requestId !== request.requestId) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-REQUEST-MISMATCH",
      "$.requestId",
      "run metadata request ID differs from exact prepared request",
    );
  }
  const declaredSourceReferenceStatementSha256 = await deriveCppCuteGitSourceReferenceStatementSha256(
    metadata.declaredSourceReference.statement,
    { limits: normalizedOptions.limits },
  );
  throwIfAborted(normalizedOptions.signal);
  if (metadata.declaredSourceReference.statementSha256 !== declaredSourceReferenceStatementSha256) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-HASH-MISMATCH",
      "$.declaredSourceReference.statementSha256",
      "declared source-reference statement hash differs from exact canonical statement",
    );
  }
  if (requestRecord.detached.declaredSourceReference?.statementSha256 !== declaredSourceReferenceStatementSha256) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-SOURCE-REFERENCE-MISMATCH",
      "$.declaredSourceReference.statementSha256",
      "declared source reference differs from prepared request detached reference",
    );
  }
  const metadataHash = await deriveCppCuteAotRunMetadataHash(metadata, {
    limits: normalizedOptions.limits,
  });
  const runMetadataId = `bg.cpp.aot-run-metadata.sha256.${metadataHash}`;
  if (metadata.runMetadataId !== runMetadataId) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-HASH-MISMATCH",
      "$.runMetadataId",
      `runMetadataId must equal ${runMetadataId}`,
    );
  }
  throwIfAborted(normalizedOptions.signal);
  const prepared = Object.freeze({
    runMetadataId,
    metadataHash,
    profileHash: profile.profileHash,
    requestId: request.requestId,
    declaredSourceReferenceStatementSha256,
    sourceFileCount: request.sourceFileCount,
  }) as PreparedCppCuteAotRunMetadata;
  PREPARED_METADATA.set(prepared, Object.freeze({
    profile,
    request,
    metadata,
  }));
  return prepared;
}

export function unwrapPreparedCppCuteAotRunMetadata(
  prepared: PreparedCppCuteAotRunMetadata,
): PreparedCppCuteAotRunMetadataRecord {
  const stored = storedMetadata(prepared);
  return Object.freeze({ profile: stored.profile, request: stored.request, metadata: stored.metadata });
}

/** Fresh copies sourced only from the opaque producer-neutral request. */
export function copyPreparedCppCuteAotRunSourceSnapshots(
  prepared: PreparedCppCuteAotRunMetadata,
): readonly CppCuteFrontendSourceSnapshotInput[] {
  const stored = storedMetadata(prepared);
  return copyPreparedCppCuteFrontendSourceSnapshots(stored.request);
}

export async function deriveCppCuteGitSourceReferenceStatementSha256(
  statement: CppCuteGitSourceReferenceStatementV1,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  return hashJson({
    domain: "browsergrad.compiler.cpp-cute.git-source-reference.v1",
    statement,
  }, "$.declaredSourceReference.statement", options.limits);
}

export async function deriveCppCuteAotRunMetadataHash(
  metadata: CppCuteAotRunMetadataV1 | CppCuteAotRunMetadataBodyV1,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  return hashJson({
    domain: "browsergrad.compiler.cpp-cute.aot-run-metadata.v1",
    metadata: {
      schema: metadata.schema,
      version: metadata.version,
      profileHash: metadata.profileHash,
      requestId: metadata.requestId,
      declaredSourceReference: metadata.declaredSourceReference,
    },
  }, "$.runMetadataId", options.limits);
}

function parseMetadata(value: unknown, limits: DecodeLimits): CppCuteAotRunMetadataV1 {
  let json: JsonValue;
  try {
    assertJsonValue(value, { limits });
    json = value as JsonValue;
    canonicalizeJson(json, { limits });
  } catch (error) {
    translateSchemaError(error, "AOT run metadata must be closed canonical JSON");
  }
  const object = closedObject(json, [
    "schema", "version", "runMetadataId", "profileHash", "requestId", "declaredSourceReference",
  ], "$");
  if (object.schema !== CPP_CUTE_AOT_RUN_METADATA_SCHEMA) {
    invalid("$.schema", `expected ${CPP_CUTE_AOT_RUN_METADATA_SCHEMA}`);
  }
  const version = closedObject(field(object, "version", "$"), ["major", "minor"], "$.version");
  if (version.major !== CPP_CUTE_AOT_RUN_METADATA_MAJOR || version.minor !== CPP_CUTE_AOT_RUN_METADATA_MINOR) {
    fail(
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-UNSUPPORTED-VERSION",
      "$.version",
      `closed metadata reader supports ${CPP_CUTE_AOT_RUN_METADATA_MAJOR}.${CPP_CUTE_AOT_RUN_METADATA_MINOR} only`,
    );
  }
  const sourcePath = "$.declaredSourceReference";
  const declaredSourceReference = closedObject(
    field(object, "declaredSourceReference", "$"),
    ["statementSha256", "statement"],
    sourcePath,
  );
  return deepFreezeJson({
    schema: CPP_CUTE_AOT_RUN_METADATA_SCHEMA,
    version: { major: CPP_CUTE_AOT_RUN_METADATA_MAJOR, minor: CPP_CUTE_AOT_RUN_METADATA_MINOR },
    runMetadataId: patterned(
      field(object, "runMetadataId", "$"),
      "$.runMetadataId",
      RUN_METADATA_ID,
      "run metadata ID",
    ),
    profileHash: sha256(field(object, "profileHash", "$"), "$.profileHash"),
    requestId: boundedString(field(object, "requestId", "$"), "$.requestId", 512),
    declaredSourceReference: {
      statementSha256: sha256(field(declaredSourceReference, "statementSha256", sourcePath), `${sourcePath}.statementSha256`),
      statement: parseSourceReferenceStatement(
        field(declaredSourceReference, "statement", sourcePath),
        `${sourcePath}.statement`,
      ),
    },
  });
}

function parseSourceReferenceStatement(
  value: JsonValue,
  path: string,
): CppCuteGitSourceReferenceStatementV1 {
  const object = closedObject(value, ["schema", "version", "repository", "revision"], path);
  if (object.schema !== CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA) {
    invalid(`${path}.schema`, `expected ${CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA}`);
  }
  const version = closedObject(field(object, "version", path), ["major", "minor"], `${path}.version`);
  if (version.major !== 1 || version.minor !== 0) invalid(`${path}.version`, "expected source reference 1.0");
  const revisionPath = `${path}.revision`;
  const revision = closedObject(field(object, "revision", path), ["algorithm", "value"], revisionPath);
  if (revision.algorithm !== "git-sha1" && revision.algorithm !== "git-sha256") {
    invalid(`${revisionPath}.algorithm`, "revision algorithm must be git-sha1 or git-sha256");
  }
  const revisionValue = stringValue(field(revision, "value", revisionPath), `${revisionPath}.value`);
  const expected = revision.algorithm === "git-sha1" ? /^[0-9a-f]{40}$/u : SHA256_HEX;
  if (!expected.test(revisionValue)) invalid(`${revisionPath}.value`, `invalid ${revision.algorithm} revision digest`);
  return {
    schema: CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA,
    version: { major: 1, minor: 0 },
    repository: canonicalRepository(field(object, "repository", path), `${path}.repository`),
    revision: { algorithm: revision.algorithm, value: revisionValue },
  };
}

function normalizeOptions(options: PrepareCppCuteAotRunMetadataOptions): {
  readonly limits: DecodeLimits;
  readonly signal: AbortSignal | undefined;
} {
  const descriptors = plainDataRecord(options, "$.options", ["limits", "signal"]);
  const rawLimits = optionalDescriptorValue(descriptors, "limits");
  let limits: DecodeLimits;
  try {
    limits = resolveDecodeLimits(rawLimits as Partial<DecodeLimits> | undefined);
  } catch (error) {
    resource("$.options.limits", "invalid semantic decode limits", { cause: error });
  }
  const rawSignal = optionalDescriptorValue(descriptors, "signal");
  if (rawSignal !== undefined && !(rawSignal instanceof AbortSignal)) {
    invalid("$.options.signal", "signal must be AbortSignal");
  }
  return Object.freeze({ limits, signal: rawSignal as AbortSignal | undefined });
}

function plainDataRecord(
  value: unknown,
  path: string,
  allowedKeys: readonly string[],
): PropertyDescriptorMap {
  if (typeof value !== "object" || value === null || Array.isArray(value)) invalid(path, "expected plain object");
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) invalid(path, "expected plain object");
  const descriptors = Object.getOwnPropertyDescriptors(value);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== "string" || !allowedKeys.includes(key)) invalid(path, "unknown option field");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "expected enumerable data property");
    }
  }
  return descriptors;
}

function optionalDescriptorValue(descriptors: PropertyDescriptorMap, name: string): unknown {
  const descriptor = descriptors[name];
  return descriptor !== undefined && "value" in descriptor ? descriptor.value : undefined;
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

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(path, "expected string");
  return value;
}

function boundedString(value: JsonValue, path: string, maximumBytes: number): string {
  const text = stringValue(value, path);
  if (text.length === 0 || text.includes("\0") || new TextEncoder().encode(text).byteLength > maximumBytes) {
    invalid(path, `string must be nonempty, NUL-free, and at most ${maximumBytes} UTF-8 bytes`);
  }
  return text;
}

function patterned(value: JsonValue, path: string, pattern: RegExp, name: string): string {
  const text = boundedString(value, path, 512);
  if (!pattern.test(text)) invalid(path, `${name} has invalid syntax`);
  return text;
}

function sha256(value: JsonValue, path: string): string {
  const text = stringValue(value, path);
  if (!SHA256_HEX.test(text)) invalid(path, "SHA-256 must be 64 lowercase hexadecimal digits");
  return text;
}

function canonicalRepository(value: JsonValue, path: string): string {
  const repository = boundedString(value, path, 1_024);
  let parsed: URL;
  try {
    parsed = new URL(repository);
  } catch (cause) {
    invalid(path, "source repository must be an absolute URL", { cause });
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" || parsed.password !== "" || parsed.search !== "" ||
      parsed.hash !== "" || parsed.pathname === "/" || parsed.pathname.endsWith("/") ||
      `${parsed.origin}${parsed.pathname}` !== repository) {
    invalid(path, "source repository must be canonical credential-free HTTPS without query, fragment, or trailing slash");
  }
  return repository;
}

async function hashJson(value: JsonValue, path: string, limits?: Partial<DecodeLimits>): Promise<string> {
  try {
    return await hashCanonicalJson(value, limits === undefined ? {} : { limits });
  } catch (error) {
    if (error instanceof SemanticSchemaError && error.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit) {
      resource(error.diagnostic.path ?? path, "canonical hash exceeded fixed resource limits", { cause: error });
    }
    invalid(path, "canonical hash input is invalid", { cause: error });
  }
}

function translateSchemaError(error: unknown, message: string): never {
  if (error instanceof CppCuteAotRunMetadataError) throw error;
  if (error instanceof SemanticSchemaError && error.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit) {
    resource(error.diagnostic.path ?? "$", message, { cause: error });
  }
  invalid("$", message, { cause: error });
}

function storedMetadata(prepared: PreparedCppCuteAotRunMetadata): StoredCppCuteAotRunMetadata {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = PREPARED_METADATA.get(prepared as object);
  if (stored === undefined) unverified();
  return stored;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    fail(
      "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-CANCELLED",
      "$.signal",
      "AOT run metadata preparation was aborted",
    );
  }
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-UNVERIFIED",
    "$",
    "operation requires opaque prepared AOT run metadata",
  );
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-RESOURCE-LIMIT", path, message, options);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-INVALID", path, message, options);
}

function mismatch(
  code: Extract<CppCuteAotRunMetadataErrorCode,
    | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-PROFILE-MISMATCH"
    | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-REQUEST-MISMATCH"
    | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-SOURCE-REFERENCE-MISMATCH"
    | "BG-COMPILER-CPP-CUTE-AOT-RUN-METADATA-HASH-MISMATCH">,
  path: string,
  message: string,
): never {
  fail(code, path, message);
}

function fail(
  code: CppCuteAotRunMetadataErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteAotRunMetadataError(code, path, message, options);
}

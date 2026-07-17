import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
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
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE,
  type CppCuteDiagnosticNormalizationV1Resource,
} from "./resources/cpp_cute_diagnostic_normalization_v1.js";
import type {
  CppCuteFrontendDiagnosticV3,
  CppCuteSemanticPassRecordV1,
} from "./cpp_cute_frontend_types.js";

export const CPP_CUTE_DIAGNOSTIC_NORMALIZATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.diagnostic-normalization";
export const CPP_CUTE_DIAGNOSTIC_NORMALIZATION_MAJOR = 1;
export const CPP_CUTE_DIAGNOSTIC_NORMALIZATION_MINOR = 0;
export const CPP_CUTE_DIAGNOSTIC_NORMALIZATION_BYTE_LIMIT = 32 * 1024;
export const CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_MANIFEST_ID =
  "bg.cpp.diagnostic-normalization.sha256.77434d0a71c7ada7ee6345d0460184b3611c824f0e51ffc3b7a9cccb4d0875d8";
export const CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256 =
  "805db4268cb63090d53bbb16940ba4362234f89a1059aa8f25df5591867f6467";

const MANIFEST_ID = /^bg\.cpp\.diagnostic-normalization\.sha256\.[0-9a-f]{64}$/u;
const PREPARED_NORMALIZATIONS =
  new WeakMap<object, StoredCppCuteDiagnosticNormalization>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export const CPP_CUTE_DIAGNOSTIC_NORMALIZATION_DECODE_LIMITS: DecodeLimits =
  Object.freeze({
    maxDocumentBytes: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_BYTE_LIMIT,
    maxDepth: 12,
    maxNodes: 2_048,
    maxStringBytes: 24 * 1024,
    maxArrayLength: 64,
    maxObjectProperties: 32,
    maxRank: 1,
    maxIntegerBits: 64,
    maxArithmeticOperations: 4_096,
  });

export type CppCuteDiagnosticNormalizationV1 =
  CppCuteDiagnosticNormalizationV1Resource;
export type CppCuteDiagnosticNormalizationBodyV1 =
  CppCuteDiagnosticNormalizationV1["body"];
export type CppCuteDiagnosticProducerStage =
  CppCuteDiagnosticNormalizationBodyV1["stageMappings"][number]["producerStage"];
export type CppCuteDiagnosticArtifactPhase =
  CppCuteDiagnosticNormalizationBodyV1["stageMappings"][number]["artifactPhase"];
export type CppCuteDiagnosticClangLevel =
  CppCuteDiagnosticNormalizationBodyV1["severityMappings"][number]["clangLevel"];
export type CppCuteDiagnosticCategory =
  CppCuteDiagnosticNormalizationBodyV1["categoryMappings"][number]["category"];
export type CppCuteSerializedDiagnosticIdMaterial = Pick<
  CppCuteFrontendDiagnosticV3,
  | "phase"
  | "severity"
  | "code"
  | "renderedMessage"
  | "location"
  | "subject"
  | "parentDiagnosticId"
>;

export interface CppCuteDiagnosticIdInput {
  readonly compilationContractHash: string;
  readonly ownerPassId: CppCuteSemanticPassRecordV1["passId"];
  readonly diagnostic: CppCuteSerializedDiagnosticIdMaterial;
}

declare const preparedCppCuteDiagnosticNormalizationBrand: unique symbol;

/**
 * Opaque authority over the exact canonical normalization design resource.
 * It grants no Clang invocation, diagnostic-normalization, artifact-production,
 * native-producer, or Wasm-producer authority.
 */
export interface PreparedCppCuteDiagnosticNormalization {
  readonly [preparedCppCuteDiagnosticNormalizationBrand]: true;
  readonly manifestId: typeof CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_MANIFEST_ID;
  readonly policyId: "browsergrad.compiler.cpp-cute.clang-diagnostic-normalization@1";
  readonly clangVersion: "22.1.8";
  readonly resourceSha256: typeof CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256;
  readonly resourceByteLength: number;
  readonly designAuthority: true;
  readonly clangInvocationAuthorized: false;
  readonly diagnosticNormalizationPerformed: false;
  readonly artifactProductionAuthorized: false;
}

export interface PreparedCppCuteDiagnosticNormalizationRecord {
  readonly normalization: CppCuteDiagnosticNormalizationV1;
}

interface StoredCppCuteDiagnosticNormalization
  extends PreparedCppCuteDiagnosticNormalizationRecord {
  readonly bytes: Uint8Array;
}

export interface DecodeCppCuteDiagnosticNormalizationOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteDiagnosticNormalizationErrorCode =
  | "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID"
  | "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-NONCANONICAL-BYTES"
  | "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-UNVERIFIED";

export class CppCuteDiagnosticNormalizationError extends Error {
  constructor(
    readonly code: CppCuteDiagnosticNormalizationErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteDiagnosticNormalizationError";
  }
}

const BUILTIN_RESOURCE_BYTES = canonicalResourceBytes(
  CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE,
);

/** Returns a disposable copy of the exact canonical v1 design resource. */
export function cppCuteDiagnosticNormalizationResourceBytes(): Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

/**
 * Snapshots hostile bytes, strict-decodes canonical JSON under fixed ceilings,
 * and verifies both the content-derived manifest ID and pinned byte identity.
 */
export async function decodeCppCuteDiagnosticNormalization(
  bytes: Uint8Array,
  options: DecodeCppCuteDiagnosticNormalizationOptions = {},
): Promise<PreparedCppCuteDiagnosticNormalization> {
  const signal = normalizeOptions(options);
  const snapshot = snapshotBytes(bytes);
  throwIfAborted(signal);

  let value: JsonValue;
  try {
    value = decodeWireJson(snapshot, {
      limits: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$bytes", "diagnostic-normalization decoding exceeded fixed resource limits", { cause });
    }
    invalid("$bytes", "diagnostic-normalization bytes are not bounded strict JSON", { cause });
  }

  const normalization = parseNormalization(value);
  const canonical = canonicalResourceBytes(normalization);
  if (!equalBytes(snapshot, canonical)) {
    fail(
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-NONCANONICAL-BYTES",
      "$bytes",
      "diagnostic-normalization bytes must exactly equal canonical JSON bytes",
    );
  }

  throwIfAborted(signal);
  const expectedManifestId = await deriveCppCuteDiagnosticNormalizationManifestId(
    normalization.body,
  );
  if (normalization.manifestId !== expectedManifestId) {
    hashMismatch("$.manifestId", `manifest ID must equal ${expectedManifestId}`);
  }

  throwIfAborted(signal);
  const resourceSha256 = await hashBytes(snapshot, "$bytes");
  if (
    resourceSha256 !== CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256
    || !equalBytes(snapshot, BUILTIN_RESOURCE_BYTES)
  ) {
    hashMismatch("$bytes", "resource SHA-256 does not equal the pinned v1 normalization identity");
  }

  throwIfAborted(signal);
  const frozen = deepFreezeJson(normalization) as CppCuteDiagnosticNormalizationV1;
  const prepared = Object.freeze({
    manifestId: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_MANIFEST_ID,
    policyId: frozen.body.policyId,
    clangVersion: frozen.body.clang.version,
    resourceSha256: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE_SHA256,
    resourceByteLength: snapshot.byteLength,
    designAuthority: true,
    clangInvocationAuthorized: false,
    diagnosticNormalizationPerformed: false,
    artifactProductionAuthorized: false,
  }) as PreparedCppCuteDiagnosticNormalization;
  PREPARED_NORMALIZATIONS.set(prepared, Object.freeze({
    normalization: frozen,
    bytes: new Uint8Array(snapshot),
  }));
  return prepared;
}

export async function deriveCppCuteDiagnosticNormalizationManifestId(
  body: CppCuteDiagnosticNormalizationBodyV1,
): Promise<string> {
  const digest = await hashJson({
    domain: "browsergrad.compiler.cpp-cute.diagnostic-normalization-manifest-id.v1",
    body,
  }, "$.manifestId");
  return `bg.cpp.diagnostic-normalization.sha256.${digest}`;
}

/**
 * Recomputes a diagnostic ID from fields that are all present in canonical
 * Artifact V3 plus the diagnostic's serialized semantic-pass ownership.
 * This calculation grants no artifact or normalization authority.
 */
export async function deriveCppCuteNormalizedDiagnosticId(
  prepared: PreparedCppCuteDiagnosticNormalization,
  input: CppCuteDiagnosticIdInput,
): Promise<string> {
  const stable = storedNormalization(prepared)
    .normalization.body.stableDiagnosticId;
  if (!/^[0-9a-f]{64}$/u.test(input.compilationContractHash)) {
    invalid(
      "$diagnosticId.compilationContractHash",
      "compilation contract hash must be exactly 64 lowercase hexadecimal characters",
    );
  }
  if (input.ownerPassId !== "cuda-device-sema" && input.ownerPassId !== "cuda-host-sema") {
    invalid(
      "$diagnosticId.ownerPassId",
      "owner pass ID must name one exact Artifact V3 semantic pass",
    );
  }
  const diagnostic = input.diagnostic;
  const digest = await hashJson({
    domain: stable.domain,
    compilationContractHash: input.compilationContractHash,
    ownerPassId: input.ownerPassId,
    diagnostic: {
      phase: diagnostic.phase,
      severity: diagnostic.severity,
      code: diagnostic.code,
      renderedMessage: diagnostic.renderedMessage,
      location: diagnostic.location,
      subject: diagnostic.subject,
      parentDiagnosticId: diagnostic.parentDiagnosticId,
    },
  }, "$diagnosticId");
  return `${stable.outputPrefix}${digest}`;
}

export function unwrapPreparedCppCuteDiagnosticNormalization(
  prepared: PreparedCppCuteDiagnosticNormalization,
): PreparedCppCuteDiagnosticNormalizationRecord {
  return Object.freeze({ normalization: storedNormalization(prepared).normalization });
}

export function canonicalCppCuteDiagnosticNormalizationBytes(
  prepared: PreparedCppCuteDiagnosticNormalization,
): Uint8Array {
  return new Uint8Array(storedNormalization(prepared).bytes);
}

export function cppCuteDiagnosticNormalizationStageRule(
  prepared: PreparedCppCuteDiagnosticNormalization,
  producerStage: CppCuteDiagnosticProducerStage,
): CppCuteDiagnosticNormalizationBodyV1["stageMappings"][number] {
  const rules = storedNormalization(prepared).normalization.body.stageMappings;
  const rule = rules.find((candidate) => candidate.producerStage === producerStage);
  if (rule === undefined) unverified("unknown producer-stage rule");
  return rule;
}

export function cppCuteDiagnosticNormalizationSeverityRule(
  prepared: PreparedCppCuteDiagnosticNormalization,
  clangLevel: CppCuteDiagnosticClangLevel,
): CppCuteDiagnosticNormalizationBodyV1["severityMappings"][number] {
  const rules = storedNormalization(prepared).normalization.body.severityMappings;
  const rule = rules.find((candidate) => candidate.clangLevel === clangLevel);
  if (rule === undefined) unverified("unknown Clang-level rule");
  return rule;
}

export function cppCuteDiagnosticNormalizationCategoryRule(
  prepared: PreparedCppCuteDiagnosticNormalization,
  category: CppCuteDiagnosticCategory,
): CppCuteDiagnosticNormalizationBodyV1["categoryMappings"][number] {
  const rules = storedNormalization(prepared).normalization.body.categoryMappings;
  const rule = rules.find((candidate) => candidate.category === category);
  if (rule === undefined) unverified("unknown diagnostic-category rule");
  return rule;
}

/** Materializes the closed public code for one exact Clang uint32 ID. */
export function cppCuteClangDiagnosticCode(
  prepared: PreparedCppCuteDiagnosticNormalization,
  rawDiagnosticId: number,
): string {
  const classification = storedNormalization(prepared)
    .normalization.body.classification;
  if (
    !Number.isSafeInteger(rawDiagnosticId)
    || rawDiagnosticId < classification.rawDiagnosticIdRange.minimum
    || rawDiagnosticId > classification.rawDiagnosticIdRange.maximum
  ) {
    invalid(
      "$rawDiagnosticId",
      "Clang diagnostic ID must be a canonical unsigned 32-bit integer",
    );
  }
  return `clang:diag-${rawDiagnosticId}`;
}

function parseNormalization(value: JsonValue): CppCuteDiagnosticNormalizationV1 {
  const object = closedObject(value, ["schema", "version", "manifestId", "body"], "$", true);
  literal(
    field(object, "schema", "$"),
    CPP_CUTE_DIAGNOSTIC_NORMALIZATION_SCHEMA,
    "$.schema",
  );
  const version = closedObject(
    field(object, "version", "$"),
    ["major", "minor"],
    "$.version",
    true,
  );
  if (version.major !== CPP_CUTE_DIAGNOSTIC_NORMALIZATION_MAJOR) {
    unsupported(
      "$.version.major",
      `reader supports major ${CPP_CUTE_DIAGNOSTIC_NORMALIZATION_MAJOR}`,
    );
  }
  if (version.minor !== CPP_CUTE_DIAGNOSTIC_NORMALIZATION_MINOR) {
    unsupported(
      "$.version.minor",
      `closed reader supports ${CPP_CUTE_DIAGNOSTIC_NORMALIZATION_MAJOR}.${CPP_CUTE_DIAGNOSTIC_NORMALIZATION_MINOR} only`,
    );
  }
  boundedPattern(field(object, "manifestId", "$"), "$.manifestId", MANIFEST_ID);
  assertExactJson(
    field(object, "body", "$"),
    CPP_CUTE_DIAGNOSTIC_NORMALIZATION_V1_RESOURCE.body,
    "$.body",
  );
  return value as CppCuteDiagnosticNormalizationV1;
}

function assertExactJson(actual: JsonValue, expected: JsonValue, path: string): void {
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) {
      invalid(path, "array must equal the exact ordered v1 inventory");
    }
    for (let index = 0; index < expected.length; index += 1) {
      assertExactJson(actual[index] as JsonValue, expected[index] as JsonValue, `${path}[${index}]`);
    }
    return;
  }
  if (isJsonObject(expected)) {
    if (!isJsonObject(actual)) invalid(path, "expected object");
    const expectedKeys = Object.keys(expected);
    const actualKeys = Object.keys(actual);
    for (const key of actualKeys) {
      if (!expectedKeys.includes(key)) invalid(path, `unknown field ${key}`);
    }
    for (const key of expectedKeys) {
      if (!Object.prototype.hasOwnProperty.call(actual, key)) {
        invalid(`${path}.${key}`, "required field is missing");
      }
      assertExactJson(
        actual[key] as JsonValue,
        expected[key] as JsonValue,
        `${path}.${key}`,
      );
    }
    return;
  }
  if (actual !== expected) invalid(path, `must equal ${JSON.stringify(expected)}`);
}

function storedNormalization(
  prepared: PreparedCppCuteDiagnosticNormalization,
): StoredCppCuteDiagnosticNormalization {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = PREPARED_NORMALIZATIONS.get(prepared as object);
  if (stored === undefined) unverified();
  return stored;
}

function snapshotBytes(value: unknown): Uint8Array {
  let inspected;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid("$bytes", "diagnostic-normalization input must be an unshared plain Uint8Array", { cause });
  }
  if (inspected.byteLength === 0) {
    invalid("$bytes", "diagnostic-normalization bytes must be nonempty");
  }
  if (inspected.byteLength > CPP_CUTE_DIAGNOSTIC_NORMALIZATION_BYTE_LIMIT) {
    resource(
      "$bytes",
      `diagnostic-normalization exceeds ${CPP_CUTE_DIAGNOSTIC_NORMALIZATION_BYTE_LIMIT} bytes`,
    );
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid("$bytes", "diagnostic-normalization bytes became unreadable while snapshotting", { cause });
  }
}

function normalizeOptions(
  options: DecodeCppCuteDiagnosticNormalizationOptions,
): AbortSignal | undefined {
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
  if (signal !== undefined && !isAbortSignal(signal)) {
    invalid("$options.signal", "signal must be an AbortSignal");
  }
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
      "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-CANCELLED",
      "$options.signal",
      "diagnostic-normalization preparation was cancelled",
    );
  }
}

function closedObject(
  value: JsonValue,
  keys: readonly string[],
  path: string,
  requireAll: boolean,
): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  for (const key of Object.keys(value)) {
    if (!keys.includes(key)) invalid(path, `unknown field ${key}`);
  }
  if (requireAll) {
    for (const key of keys) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) {
        invalid(`${path}.${key}`, "required field is missing");
      }
    }
  }
  return value;
}

function isJsonObject(value: JsonValue): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function field(value: JsonObject, key: string, path: string): JsonValue {
  if (!Object.prototype.hasOwnProperty.call(value, key)) {
    invalid(`${path}.${key}`, "required field is missing");
  }
  return value[key] as JsonValue;
}

function literal<T extends string>(
  value: JsonValue,
  expected: T,
  path: string,
): asserts value is T {
  if (value !== expected) invalid(path, `must equal ${JSON.stringify(expected)}`);
}

function boundedPattern(value: JsonValue, path: string, pattern: RegExp): string {
  if (
    typeof value !== "string"
    || value.length === 0
    || value.length > 128
    || !pattern.test(value)
  ) {
    invalid(path, "string does not match required closed format");
  }
  return value;
}

function canonicalResourceBytes(value: JsonValue): Uint8Array {
  try {
    return canonicalJsonBytes(value, {
      limits: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$", "canonical diagnostic-normalization exceeds fixed resource limits", { cause });
    }
    invalid("$", "diagnostic-normalization cannot be canonically encoded", { cause });
  }
}

async function hashJson(value: JsonValue, path: string): Promise<string> {
  try {
    return await hashCanonicalJson(value, {
      limits: CPP_CUTE_DIAGNOSTIC_NORMALIZATION_DECODE_LIMITS,
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
  return cause instanceof Error
    && /Web Crypto|crypto\.subtle|SHA-256 unavailable/iu.test(cause.message);
}

function isSchemaResourceLimit(cause: unknown): boolean {
  return cause instanceof SemanticSchemaError
    && cause.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-INVALID", path, message, options);
}

function unsupported(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-UNSUPPORTED-VERSION", path, message);
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-RESOURCE-LIMIT", path, message, options);
}

function hashMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-HASH-MISMATCH", path, message);
}

function hashUnavailable(path: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-HASH-UNAVAILABLE",
    path,
    "SHA-256 is unavailable",
    { cause },
  );
}

function unverified(message = "prepared diagnostic-normalization authority is not recognized"): never {
  fail("BG-COMPILER-CPP-CUTE-DIAGNOSTIC-NORMALIZATION-UNVERIFIED", "$prepared", message);
}

function fail(
  code: CppCuteDiagnosticNormalizationErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteDiagnosticNormalizationError(code, path, message, options);
}

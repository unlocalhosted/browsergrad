import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  assertJsonValue,
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
  hashCanonicalJson,
  isJsonObject,
  sha256Hex,
  type DecodeLimits,
  type JsonObject,
  type JsonValue,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";

export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE =
  "https://browsergrad.dev/attestations/cpp-cute-header-distribution-approval/v1";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-distribution-approval-policy";
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR = 1;
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR = 0;
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_BYTE_LIMIT =
  64 * 1024;
export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_ID_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-distribution-approval-policy.v1";

const POLICY_ID =
  /^bg\.cpp\.browser-distribution-approval-policy\.sha256\.[0-9a-f]{64}$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const KEY_ID = /^sha256:[0-9a-f]{64}$/u;
const MAX_REVIEWERS = 64;
const MAX_KEYS = 256;
const TEXT_ENCODER = new TextEncoder();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export const CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_DECODE_LIMITS:
DecodeLimits = Object.freeze({
  maxDocumentBytes:
    CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_BYTE_LIMIT,
  maxDepth: 8,
  maxNodes: 1_024,
  maxStringBytes: 32 * 1024,
  maxArrayLength: MAX_KEYS,
  maxObjectProperties: 16,
  maxRank: 1,
  maxIntegerBits: 32,
  maxArithmeticOperations: 2_048,
});

export interface CppCuteBrowserDistributionApprovalPolicyVersionV1 extends
  JsonObject {
  readonly major:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR;
  readonly minor:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR;
}

export interface CppCuteBrowserDistributionApprovalPolicyV1 extends
  JsonObject {
  readonly schema:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA;
  readonly version: CppCuteBrowserDistributionApprovalPolicyVersionV1;
  readonly policyId: string;
  readonly predicateType:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE;
  readonly trustStoreSha256: string;
  readonly reviewerIds: readonly string[];
  readonly keyIds: readonly string[];
}

export interface CppCuteBrowserDistributionApprovalPolicyProjectionV1 extends
  JsonObject {
  readonly schema:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA;
  readonly version: CppCuteBrowserDistributionApprovalPolicyVersionV1;
  readonly predicateType:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE;
  readonly trustStoreSha256: string;
  readonly reviewerIds: readonly string[];
  readonly keyIds: readonly string[];
}

declare const admittedPolicyBrand: unique symbol;

/**
 * Explicit host root-of-trust admission for external redistribution review.
 * It proves exact policy bytes only; it cannot approve distribution.
 */
export interface AdmittedCppCuteBrowserDistributionApprovalPolicy {
  readonly [admittedPolicyBrand]: true;
  readonly authority:
    "host-admitted-browser-distribution-approval-policy";
  readonly policyId: string;
  readonly policySha256: string;
  readonly policyByteLength: number;
  readonly policyVersion: "1.0";
  readonly predicateType:
    typeof CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE;
  readonly trustStoreSha256: string;
  readonly reviewerIds: readonly string[];
  readonly keyIds: readonly string[];
  readonly hostOnly: true;
  readonly workerTransferable: false;
  readonly externalReviewVerified: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly releaseReady: false;
}

export interface AdmittedCppCuteBrowserDistributionApprovalPolicyRecord {
  readonly policy: CppCuteBrowserDistributionApprovalPolicyV1;
}

export interface AdmitCppCuteBrowserDistributionApprovalPolicyOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserDistributionApprovalPolicyErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-ID-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-NONCANONICAL"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-UNVERIFIED";

export class CppCuteBrowserDistributionApprovalPolicyError extends Error {
  constructor(
    readonly code: CppCuteBrowserDistributionApprovalPolicyErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserDistributionApprovalPolicyError";
  }
}

const ADMITTED_POLICIES = new WeakMap<
  object,
  AdmittedCppCuteBrowserDistributionApprovalPolicyRecord
>();
const ADMITTED_POLICY_BYTES = new WeakMap<object, Uint8Array>();

export async function admitCppCuteBrowserDistributionApprovalPolicy(
  bytes: Uint8Array,
  options: AdmitCppCuteBrowserDistributionApprovalPolicyOptions = {},
): Promise<AdmittedCppCuteBrowserDistributionApprovalPolicy> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  let snapshot: Uint8Array;
  try {
    const inspection = inspectUnsharedPlainUint8Array(bytes);
    if (inspection.byteLength === 0 ||
        inspection.byteLength >
          CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_BYTE_LIMIT) {
      resource(
        "$.bytes",
        "distribution approval policy exceeds the fixed byte limit",
      );
    }
    snapshot = copyInspectedUnsharedUint8Array(bytes, inspection);
  } catch (cause) {
    if (cause instanceof CppCuteBrowserDistributionApprovalPolicyError) {
      throw cause;
    }
    invalid(
      "$.bytes",
      "distribution approval policy must be an unshared plain Uint8Array",
      { cause },
    );
  }

  let decoded: JsonValue;
  try {
    decoded = decodeWireJson(snapshot, {
      limits: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource(
        "$.bytes",
        "distribution approval policy exceeds fixed decode limits",
        { cause },
      );
    }
    invalid(
      "$.bytes",
      "distribution approval policy must be strict UTF-8 JSON",
      { cause },
    );
  }
  let canonical: Uint8Array;
  try {
    canonical = canonicalJsonBytes(decoded, {
      limits: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource(
        "$.bytes",
        "distribution approval policy exceeds fixed canonical limits",
        { cause },
      );
    }
    invalid(
      "$.bytes",
      "distribution approval policy cannot be canonically encoded",
      { cause },
    );
  }
  if (!equalBytes(snapshot, canonical)) {
    noncanonical(
      "$.bytes",
      "distribution approval policy bytes must use exact canonical JSON",
    );
  }

  const policy = parsePolicy(decoded);
  const expectedPolicyId = await policyId(policyProjection(policy));
  if (policy.policyId !== expectedPolicyId) {
    mismatch(
      "$.policyId",
      "distribution approval policy ID differs from its canonical projection",
    );
  }
  const policySha256 = await hashBytes(snapshot, "$.bytes");
  throwIfAborted(signal);
  const admitted = Object.freeze({
    authority: "host-admitted-browser-distribution-approval-policy",
    policyId: policy.policyId,
    policySha256,
    policyByteLength: snapshot.byteLength,
    policyVersion: "1.0",
    predicateType: policy.predicateType,
    trustStoreSha256: policy.trustStoreSha256,
    reviewerIds: policy.reviewerIds,
    keyIds: policy.keyIds,
    hostOnly: true,
    workerTransferable: false,
    externalReviewVerified: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    releaseReady: false,
  }) as AdmittedCppCuteBrowserDistributionApprovalPolicy;
  ADMITTED_POLICIES.set(admitted, Object.freeze({ policy }));
  ADMITTED_POLICY_BYTES.set(admitted, snapshot);
  return admitted;
}

export function copyAdmittedCppCuteBrowserDistributionApprovalPolicyBytes(
  admitted: AdmittedCppCuteBrowserDistributionApprovalPolicy,
): Uint8Array {
  unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy(admitted);
  const bytes = ADMITTED_POLICY_BYTES.get(admitted as object);
  if (bytes === undefined) unverified();
  return new Uint8Array(bytes);
}

export function unwrapAdmittedCppCuteBrowserDistributionApprovalPolicy(
  admitted: AdmittedCppCuteBrowserDistributionApprovalPolicy,
): AdmittedCppCuteBrowserDistributionApprovalPolicyRecord {
  if (typeof admitted !== "object" || admitted === null) unverified();
  const record = ADMITTED_POLICIES.get(admitted as object);
  if (record === undefined) unverified();
  return record;
}

export async function deriveCppCuteBrowserDistributionApprovalPolicyId(
  projection: CppCuteBrowserDistributionApprovalPolicyProjectionV1,
): Promise<string> {
  try {
    assertJsonValue(projection, {
      limits: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource(
        "$.projection",
        "distribution approval policy projection exceeds fixed limits",
        { cause },
      );
    }
    invalid(
      "$.projection",
      "distribution approval policy projection must be an accessor-free JSON tree",
      { cause },
    );
  }
  return await policyId(parseProjection(projection as JsonValue));
}

function parsePolicy(
  value: JsonValue,
): CppCuteBrowserDistributionApprovalPolicyV1 {
  const object = closedObject(value, [
    "schema",
    "version",
    "policyId",
    "predicateType",
    "trustStoreSha256",
    "reviewerIds",
    "keyIds",
  ], "$");
  const projection = parseProjectionObject(object, "$");
  return deepFreezeJson({
    ...projection,
    policyId: pattern(field(object, "policyId", "$"), POLICY_ID, "$.policyId"),
  }) as CppCuteBrowserDistributionApprovalPolicyV1;
}

function parseProjection(
  value: JsonValue,
): CppCuteBrowserDistributionApprovalPolicyProjectionV1 {
  const object = closedObject(value, [
    "schema",
    "version",
    "predicateType",
    "trustStoreSha256",
    "reviewerIds",
    "keyIds",
  ], "$.projection");
  return deepFreezeJson(
    parseProjectionObject(object, "$.projection"),
  ) as CppCuteBrowserDistributionApprovalPolicyProjectionV1;
}

function parseProjectionObject(
  object: JsonObject,
  path: string,
): CppCuteBrowserDistributionApprovalPolicyProjectionV1 {
  exact(
    field(object, "schema", path),
    CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
    `${path}.schema`,
  );
  const versionPath = `${path}.version`;
  const version = closedObject(
    field(object, "version", path),
    ["major", "minor"],
    versionPath,
  );
  const major = integer(field(version, "major", versionPath),
    `${versionPath}.major`);
  const minor = integer(field(version, "minor", versionPath),
    `${versionPath}.minor`);
  if (major !== CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR ||
      minor !== CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR) {
    unsupported(
      versionPath,
      "reader supports closed distribution approval policy version 1.0 only",
    );
  }
  exact(
    field(object, "predicateType", path),
    CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
    `${path}.predicateType`,
  );
  const reviewerIds = sortedUniqueStrings(
    field(object, "reviewerIds", path),
    `${path}.reviewerIds`,
    MAX_REVIEWERS,
    canonicalHttpsIdentifier,
  );
  const keyIds = sortedUniqueStrings(
    field(object, "keyIds", path),
    `${path}.keyIds`,
    MAX_KEYS,
    (entry, entryPath) => pattern(entry, KEY_ID, entryPath),
  );
  if (reviewerIds.length === 0) {
    invalid(`${path}.reviewerIds`, "policy requires at least one reviewer");
  }
  if (keyIds.length === 0) {
    invalid(`${path}.keyIds`, "policy requires at least one key");
  }
  return {
    schema: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MAJOR,
      minor: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_MINOR,
    },
    predicateType: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_PREDICATE_TYPE,
    trustStoreSha256: pattern(
      field(object, "trustStoreSha256", path),
      SHA256_HEX,
      `${path}.trustStoreSha256`,
    ),
    reviewerIds,
    keyIds,
  };
}

function policyProjection(
  policy: CppCuteBrowserDistributionApprovalPolicyV1,
): CppCuteBrowserDistributionApprovalPolicyProjectionV1 {
  return {
    schema: policy.schema,
    version: policy.version,
    predicateType: policy.predicateType,
    trustStoreSha256: policy.trustStoreSha256,
    reviewerIds: policy.reviewerIds,
    keyIds: policy.keyIds,
  };
}

async function policyId(
  projection: CppCuteBrowserDistributionApprovalPolicyProjectionV1,
): Promise<string> {
  try {
    const digest = await hashCanonicalJson({
      domain: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_ID_DOMAIN,
      policy: projection,
    }, {
      limits: CPP_CUTE_BROWSER_DISTRIBUTION_APPROVAL_POLICY_DECODE_LIMITS,
    });
    return `bg.cpp.browser-distribution-approval-policy.sha256.${digest}`;
  } catch (cause) {
    if (isHashUnavailable(cause)) hashUnavailable("$.policyId", cause);
    if (isSchemaResourceLimit(cause)) {
      resource(
        "$.policyId",
        "distribution approval policy identity exceeds fixed limits",
        { cause },
      );
    }
    invalid(
      "$.policyId",
      "distribution approval policy identity could not be derived",
      { cause },
    );
  }
}

function normalizeOptions(
  options: AdmitCppCuteBrowserDistributionApprovalPolicyOptions,
): AbortSignal | undefined {
  try {
    if (typeof options !== "object" || options === null ||
        Object.getPrototypeOf(options) !== Object.prototype) {
      invalid("$.options", "options must be a plain data record");
    }
    const descriptors = Object.getOwnPropertyDescriptors(options);
    const keys = Reflect.ownKeys(descriptors);
    if (keys.some((key) => typeof key !== "string" || key !== "signal")) {
      invalid("$.options", "options contains unknown fields");
    }
    const signalDescriptor = descriptors.signal;
    if (signalDescriptor === undefined) return undefined;
    if (!("value" in signalDescriptor) ||
        signalDescriptor.enumerable !== true ||
        typeof AbortSignal === "undefined" ||
        signalDescriptor.value instanceof AbortSignal === false) {
      invalid(
        "$.options.signal",
        "signal must be an enumerable AbortSignal data property",
      );
    }
    return signalDescriptor.value as AbortSignal;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserDistributionApprovalPolicyError) {
      throw cause;
    }
    invalid(
      "$.options",
      "options could not be inspected as a plain data record",
      { cause },
    );
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      Reflect.apply(ABORT_SIGNAL_ABORTED_GETTER, signal, []) === true) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-CANCELLED",
      "$.options.signal",
      "distribution approval policy admission was cancelled",
    );
  }
}

function closedObject(
  value: JsonValue,
  fields: readonly string[],
  path: string,
): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const unknown = Object.keys(value).filter((key) => !fields.includes(key));
  const missing = fields.filter((key) => !Object.hasOwn(value, key));
  if (unknown.length !== 0) {
    invalid(path, `unknown fields: ${unknown.sort().join(", ")}`);
  }
  if (missing.length !== 0) {
    invalid(path, `missing fields: ${missing.sort().join(", ")}`);
  }
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(`${path}.${name}`, "field is required");
  return value;
}

function exact(value: JsonValue, expected: string, path: string): void {
  if (value !== expected) invalid(path, `expected ${JSON.stringify(expected)}`);
}

function integer(value: JsonValue, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    invalid(path, "expected non-negative safe integer");
  }
  return value;
}

function pattern(value: JsonValue, expression: RegExp, path: string): string {
  if (typeof value !== "string" || !expression.test(value)) {
    invalid(path, "value does not match the required closed identifier syntax");
  }
  return value;
}

function sortedUniqueStrings(
  value: JsonValue,
  path: string,
  maximum: number,
  validate: (entry: JsonValue, path: string) => string,
): readonly string[] {
  if (!Array.isArray(value) || value.length > maximum) {
    resource(path, `expected at most ${maximum} entries`);
  }
  const entries = value.map(
    (entry, index) => validate(entry, `${path}[${index}]`),
  );
  for (let index = 1; index < entries.length; index += 1) {
    if (entries[index - 1]! >= entries[index]!) {
      invalid(path, "entries must be strictly sorted and unique");
    }
  }
  return Object.freeze(entries);
}

function canonicalHttpsIdentifier(
  value: JsonValue,
  path: string,
): string {
  if (typeof value !== "string" || value.length === 0 ||
      TEXT_ENCODER.encode(value).byteLength > 1_024) {
    invalid(path, "reviewer identity must be a bounded string");
  }
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch (cause) {
    invalid(path, "reviewer identity must be an absolute HTTPS URL", { cause });
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" ||
      parsed.password !== "" || parsed.search !== "" || parsed.hash !== "" ||
      parsed.pathname === "/" || parsed.pathname.endsWith("/") ||
      `${parsed.origin}${parsed.pathname}` !== value) {
    invalid(
      path,
      "reviewer identity must be a canonical credential-free HTTPS URL without query, fragment, or trailing slash",
    );
  }
  return value;
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
  return cause instanceof Error &&
    /Web Crypto|crypto\.subtle|SHA-256 unavailable/iu.test(cause.message);
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

function invalid(
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-INVALID",
    path,
    message,
    options,
  );
}

function resource(
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-RESOURCE-LIMIT",
    path,
    message,
    options,
  );
}

function noncanonical(path: string, message: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-NONCANONICAL",
    path,
    message,
  );
}

function mismatch(path: string, message: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-ID-MISMATCH",
    path,
    message,
  );
}

function unsupported(path: string, message: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-UNSUPPORTED-VERSION",
    path,
    message,
  );
}

function hashUnavailable(path: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-HASH-UNAVAILABLE",
    path,
    "SHA-256 is unavailable for distribution approval policy admission",
    { cause },
  );
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-DISTRIBUTION-APPROVAL-POLICY-UNVERIFIED",
    "$",
    "distribution approval policy must come from the opaque host admission authority",
  );
}

function fail(
  code: CppCuteBrowserDistributionApprovalPolicyErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserDistributionApprovalPolicyError(
    code,
    path,
    message,
    options,
  );
}

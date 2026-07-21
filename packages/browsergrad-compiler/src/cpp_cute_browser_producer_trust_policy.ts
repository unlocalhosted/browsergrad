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
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
} from "./cpp_cute_browser_assets.js";

export const CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-producer-trust-policy";
export const CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR = 1;
export const CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR = 0;
export const CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT = 64 * 1024;
export const CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_ID_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-producer-trust-policy.v1";

const POLICY_ID =
  /^bg\.cpp\.browser-producer-trust-policy\.sha256\.[0-9a-f]{64}$/u;
const SHA256_HEX = /^[0-9a-f]{64}$/u;
const KEY_ID = /^sha256:[0-9a-f]{64}$/u;
const MAX_BUILDERS = 64;
const MAX_KEYS = 256;
const TEXT_ENCODER = new TextEncoder();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export const CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_DECODE_LIMITS: DecodeLimits =
  Object.freeze({
    maxDocumentBytes: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT,
    maxDepth: 8,
    maxNodes: 1_024,
    maxStringBytes: 32 * 1024,
    maxArrayLength: MAX_KEYS,
    maxObjectProperties: 16,
    maxRank: 1,
    maxIntegerBits: 32,
    maxArithmeticOperations: 2_048,
  });

export interface CppCuteBrowserProducerTrustPolicyVersionV1 extends JsonObject {
  readonly major: typeof CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR;
  readonly minor: typeof CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR;
}

export interface CppCuteBrowserProducerTrustPolicyV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_SCHEMA;
  readonly version: CppCuteBrowserProducerTrustPolicyVersionV1;
  readonly policyId: string;
  readonly predicateType: typeof CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE;
  readonly trustStoreSha256: string;
  readonly builderIds: readonly string[];
  readonly keyIds: readonly string[];
}

export type CppCuteBrowserProducerTrustPolicyProjectionV1 = Omit<
  CppCuteBrowserProducerTrustPolicyV1,
  "policyId"
>;

declare const admittedPolicyBrand: unique symbol;

/**
 * Explicit host root-of-trust admission. User source and the compiler Worker
 * never receive this authority. Admission proves only the exact policy bytes;
 * producer trust still requires a separately verified signature binding.
 */
export interface AdmittedCppCuteBrowserProducerTrustPolicy {
  readonly [admittedPolicyBrand]: true;
  readonly authority: "host-admitted-browser-producer-trust-policy";
  readonly policyId: string;
  readonly policySha256: string;
  readonly policyByteLength: number;
  readonly policyVersion: "1.0";
  readonly predicateType: typeof CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE;
  readonly trustStoreSha256: string;
  readonly builderIds: readonly string[];
  readonly keyIds: readonly string[];
  readonly hostOnly: true;
  readonly workerTransferable: false;
  readonly producerTrusted: false;
  readonly releaseReady: false;
}

export interface AdmittedCppCuteBrowserProducerTrustPolicyRecord {
  readonly policy: CppCuteBrowserProducerTrustPolicyV1;
}

export interface AdmitCppCuteBrowserProducerTrustPolicyOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserProducerTrustPolicyErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-ID-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-NONCANONICAL"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-UNVERIFIED";

export class CppCuteBrowserProducerTrustPolicyError extends Error {
  constructor(
    readonly code: CppCuteBrowserProducerTrustPolicyErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserProducerTrustPolicyError";
  }
}

const ADMITTED_POLICIES = new WeakMap<
  object,
  AdmittedCppCuteBrowserProducerTrustPolicyRecord
>();
const ADMITTED_POLICY_BYTES = new WeakMap<object, Uint8Array>();

export async function admitCppCuteBrowserProducerTrustPolicy(
  bytes: Uint8Array,
  options: AdmitCppCuteBrowserProducerTrustPolicyOptions = {},
): Promise<AdmittedCppCuteBrowserProducerTrustPolicy> {
  const signal = normalizeOptions(options);
  throwIfAborted(signal);
  let snapshot: Uint8Array;
  try {
    const inspection = inspectUnsharedPlainUint8Array(bytes);
    if (inspection.byteLength === 0 ||
        inspection.byteLength > CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_BYTE_LIMIT) {
      resource("$.bytes", "producer trust policy exceeds the fixed byte limit");
    }
    snapshot = copyInspectedUnsharedUint8Array(bytes, inspection);
  } catch (cause) {
    if (cause instanceof CppCuteBrowserProducerTrustPolicyError) throw cause;
    invalid("$.bytes", "producer trust policy must be an unshared plain Uint8Array", {
      cause,
    });
  }

  let decoded: JsonValue;
  try {
    decoded = decodeWireJson(snapshot, {
      limits: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$.bytes", "producer trust policy exceeds fixed decode limits", { cause });
    }
    invalid("$.bytes", "producer trust policy must be strict UTF-8 JSON", { cause });
  }
  let canonical: Uint8Array;
  try {
    canonical = canonicalJsonBytes(decoded, {
      limits: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$.bytes", "producer trust policy exceeds fixed canonical limits", {
        cause,
      });
    }
    invalid("$.bytes", "producer trust policy cannot be canonically encoded", {
      cause,
    });
  }
  if (!equalBytes(snapshot, canonical)) {
    noncanonical("$.bytes", "producer trust policy bytes must use exact canonical JSON");
  }

  const policy = parsePolicy(decoded);
  const projection = policyProjection(policy);
  const expectedPolicyId = await policyId(projection);
  if (policy.policyId !== expectedPolicyId) {
    mismatch("$.policyId", "producer trust policy ID differs from its canonical projection");
  }
  const policySha256 = await hashBytes(snapshot, "$.bytes");
  throwIfAborted(signal);
  const admitted = Object.freeze({
    authority: "host-admitted-browser-producer-trust-policy",
    policyId: policy.policyId,
    policySha256,
    policyByteLength: snapshot.byteLength,
    policyVersion: "1.0",
    predicateType: policy.predicateType,
    trustStoreSha256: policy.trustStoreSha256,
    builderIds: policy.builderIds,
    keyIds: policy.keyIds,
    hostOnly: true,
    workerTransferable: false,
    producerTrusted: false,
    releaseReady: false,
  }) as AdmittedCppCuteBrowserProducerTrustPolicy;
  ADMITTED_POLICIES.set(admitted, Object.freeze({
    policy,
  }));
  ADMITTED_POLICY_BYTES.set(admitted, snapshot);
  return admitted;
}

export function copyAdmittedCppCuteBrowserProducerTrustPolicyBytes(
  admitted: AdmittedCppCuteBrowserProducerTrustPolicy,
): Uint8Array {
  unwrapAdmittedCppCuteBrowserProducerTrustPolicy(admitted);
  const bytes = ADMITTED_POLICY_BYTES.get(admitted as object);
  if (bytes === undefined) unverified();
  return new Uint8Array(bytes);
}

export function unwrapAdmittedCppCuteBrowserProducerTrustPolicy(
  admitted: AdmittedCppCuteBrowserProducerTrustPolicy,
): AdmittedCppCuteBrowserProducerTrustPolicyRecord {
  if (typeof admitted !== "object" || admitted === null) unverified();
  const record = ADMITTED_POLICIES.get(admitted as object);
  if (record === undefined) unverified();
  return record;
}

export async function deriveCppCuteBrowserProducerTrustPolicyId(
  projection: CppCuteBrowserProducerTrustPolicyProjectionV1,
): Promise<string> {
  try {
    assertJsonValue(projection, {
      limits: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$.projection", "producer trust policy projection exceeds fixed limits", {
        cause,
      });
    }
    invalid("$.projection", "producer trust policy projection must be an accessor-free JSON tree", {
      cause,
    });
  }
  const parsed = parseProjection(projection as JsonValue, "$.projection");
  return await policyId(parsed);
}

function parsePolicy(value: JsonValue): CppCuteBrowserProducerTrustPolicyV1 {
  const object = closedObject(value, [
    "schema",
    "version",
    "policyId",
    "predicateType",
    "trustStoreSha256",
    "builderIds",
    "keyIds",
  ], "$", true);
  const projection = parseProjectionObject(object, "$", true);
  return deepFreezeJson({
    ...projection,
    policyId: pattern(field(object, "policyId", "$"), POLICY_ID, "$.policyId"),
  }) as CppCuteBrowserProducerTrustPolicyV1;
}

function parseProjection(
  value: JsonValue,
  path: string,
): CppCuteBrowserProducerTrustPolicyProjectionV1 {
  const object = closedObject(value, [
    "schema",
    "version",
    "predicateType",
    "trustStoreSha256",
    "builderIds",
    "keyIds",
  ], path, true);
  return deepFreezeJson(parseProjectionObject(object, path, false)) as
    CppCuteBrowserProducerTrustPolicyProjectionV1;
}

function parseProjectionObject(
  object: JsonObject,
  path: string,
  allowPolicyId: boolean,
): CppCuteBrowserProducerTrustPolicyProjectionV1 {
  exact(field(object, "schema", path), CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_SCHEMA,
    `${path}.schema`);
  const versionPath = `${path}.version`;
  const version = closedObject(field(object, "version", path), ["major", "minor"],
    versionPath, true);
  const major = integer(field(version, "major", versionPath), `${versionPath}.major`);
  const minor = integer(field(version, "minor", versionPath), `${versionPath}.minor`);
  if (major !== CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR ||
      minor !== CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR) {
    unsupported(versionPath, "reader supports closed producer trust policy version 1.0 only");
  }
  exact(
    field(object, "predicateType", path),
    CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    `${path}.predicateType`,
  );
  const builderIds = sortedUniqueStrings(
    field(object, "builderIds", path),
    `${path}.builderIds`,
    MAX_BUILDERS,
    canonicalHttpsIdentifier,
  );
  const keyIds = sortedUniqueStrings(
    field(object, "keyIds", path),
    `${path}.keyIds`,
    MAX_KEYS,
    (entry, entryPath) => pattern(entry, KEY_ID, entryPath),
  );
  if (builderIds.length === 0) invalid(`${path}.builderIds`, "policy requires at least one builder");
  if (keyIds.length === 0) invalid(`${path}.keyIds`, "policy requires at least one key");
  if (!allowPolicyId && Object.hasOwn(object, "policyId")) {
    invalid(path, "policy projection must not contain policyId");
  }
  return {
    schema: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_SCHEMA,
    version: {
      major: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR,
      minor: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR,
    },
    predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    trustStoreSha256: pattern(
      field(object, "trustStoreSha256", path),
      SHA256_HEX,
      `${path}.trustStoreSha256`,
    ),
    builderIds,
    keyIds,
  };
}

function policyProjection(
  policy: CppCuteBrowserProducerTrustPolicyV1,
): CppCuteBrowserProducerTrustPolicyProjectionV1 {
  return {
    schema: policy.schema,
    version: policy.version,
    predicateType: policy.predicateType,
    trustStoreSha256: policy.trustStoreSha256,
    builderIds: policy.builderIds,
    keyIds: policy.keyIds,
  };
}

async function policyId(
  projection: CppCuteBrowserProducerTrustPolicyProjectionV1,
): Promise<string> {
  try {
    const digest = await hashCanonicalJson({
      domain: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_ID_DOMAIN,
      policy: projection,
    }, { limits: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_DECODE_LIMITS });
    return `bg.cpp.browser-producer-trust-policy.sha256.${digest}`;
  } catch (cause) {
    if (isHashUnavailable(cause)) hashUnavailable("$.policyId", cause);
    if (isSchemaResourceLimit(cause)) {
      resource("$.policyId", "producer trust policy identity exceeds fixed limits", {
        cause,
      });
    }
    invalid("$.policyId", "producer trust policy identity could not be derived", {
      cause,
    });
  }
}

function normalizeOptions(
  options: AdmitCppCuteBrowserProducerTrustPolicyOptions,
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
    if (!("value" in signalDescriptor) || signalDescriptor.enumerable !== true ||
        typeof AbortSignal === "undefined" ||
        signalDescriptor.value instanceof AbortSignal === false) {
      invalid("$.options.signal", "signal must be an enumerable AbortSignal data property");
    }
    return signalDescriptor.value as AbortSignal;
  } catch (cause) {
    if (cause instanceof CppCuteBrowserProducerTrustPolicyError) throw cause;
    invalid("$.options", "options could not be inspected as a plain data record", { cause });
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      Reflect.apply(ABORT_SIGNAL_ABORTED_GETTER, signal, []) === true) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-CANCELLED",
      "$.options.signal",
      "producer trust policy admission was cancelled",
    );
  }
}

function closedObject(
  value: JsonValue,
  fields: readonly string[],
  path: string,
  allowPolicyId: boolean,
): JsonObject {
  if (!isJsonObject(value)) invalid(path, "expected object");
  const allowed = allowPolicyId ? fields : fields.filter((fieldName) => fieldName !== "policyId");
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  const missing = allowed.filter((key) => !Object.hasOwn(value, key));
  if (unknown.length !== 0) invalid(path, `unknown fields: ${unknown.sort().join(", ")}`);
  if (missing.length !== 0) invalid(path, `missing fields: ${missing.sort().join(", ")}`);
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
  const entries = value.map((entry, index) => validate(entry, `${path}[${index}]`));
  for (let index = 1; index < entries.length; index += 1) {
    if (entries[index - 1]! >= entries[index]!) {
      invalid(path, "entries must be strictly sorted and unique");
    }
  }
  return Object.freeze(entries);
}

function canonicalHttpsIdentifier(value: JsonValue, path: string): string {
  if (typeof value !== "string" || value.length === 0 ||
      TEXT_ENCODER.encode(value).byteLength > 1_024) {
    invalid(path, "builder identity must be a bounded string");
  }
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch (cause) {
    invalid(path, "builder identity must be an absolute HTTPS URL", { cause });
  }
  if (parsed.protocol !== "https:" || parsed.username !== "" ||
      parsed.password !== "" || parsed.search !== "" || parsed.hash !== "" ||
      parsed.pathname === "/" || parsed.pathname.endsWith("/") ||
      `${parsed.origin}${parsed.pathname}` !== value) {
    invalid(
      path,
      "builder identity must be a canonical credential-free HTTPS URL without query, fragment, or trailing slash",
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

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-INVALID", path, message, options);
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-RESOURCE-LIMIT", path, message, options);
}

function noncanonical(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-NONCANONICAL", path, message);
}

function mismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-ID-MISMATCH", path, message);
}

function unsupported(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-UNSUPPORTED-VERSION", path, message);
}

function hashUnavailable(path: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-HASH-UNAVAILABLE",
    path,
    "SHA-256 is unavailable for producer trust policy admission",
    { cause },
  );
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCER-TRUST-POLICY-UNVERIFIED",
    "$",
    "producer trust policy must come from the opaque host admission authority",
  );
}

function fail(
  code: CppCuteBrowserProducerTrustPolicyErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserProducerTrustPolicyError(code, path, message, options);
}

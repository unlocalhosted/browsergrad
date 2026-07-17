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
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE,
  type CppCuteSemanticAdapterManifestV1Resource,
} from "./resources/cpp_cute_semantic_adapter_manifest_v1.js";

export const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_SCHEMA =
  "browsergrad.compiler.cpp-cute.semantic-adapter-manifest";
export const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_MAJOR = 1;
export const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_MINOR = 0;
export const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_BYTE_LIMIT = 16 * 1024;
export const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_MANIFEST_ID =
  "bg.cpp.semantic-adapter.sha256.77d2e58c18d0df8e8a8aef7fa5742f8e9ae82912e692efab5507cf14112bceb0";
export const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256 =
  "e5aa795c4feebd523ed72b95be03b102d497f2e0313ee9c99fadf1309cde6150";

const MANIFEST_ID = /^bg\.cpp\.semantic-adapter\.sha256\.[0-9a-f]{64}$/u;
const PREPARED_MANIFESTS = new WeakMap<object, StoredCppCuteSemanticAdapterManifest>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export const CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_DECODE_LIMITS: DecodeLimits =
  Object.freeze({
    maxDocumentBytes: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_BYTE_LIMIT,
    maxDepth: 12,
    maxNodes: 512,
    maxStringBytes: 8 * 1024,
    maxArrayLength: 32,
    maxObjectProperties: 16,
    maxRank: 1,
    maxIntegerBits: 64,
    maxArithmeticOperations: 1_024,
  });

export type CppCuteSemanticAdapterManifestV1 =
  CppCuteSemanticAdapterManifestV1Resource;
export type CppCuteSemanticAdapterManifestBodyV1 =
  CppCuteSemanticAdapterManifestV1["body"];
export type CppCuteSemanticAdapterWarningPolicyEntryV1 =
  CppCuteSemanticAdapterManifestBodyV1["warningPolicyRegistry"]["entries"][number];
export type CppCuteSemanticAdapterWarningPolicyId =
  CppCuteSemanticAdapterWarningPolicyEntryV1["policyId"];
export type CppCuteSemanticAdapterWarningDisposition =
  keyof CppCuteSemanticAdapterWarningPolicyEntryV1["argv"];

declare const preparedCppCuteSemanticAdapterManifestBrand: unique symbol;

/**
 * Opaque authority over the exact canonical policy resource. It proves no
 * Clang invocation and no native/Wasm producer conformance.
 */
export interface PreparedCppCuteSemanticAdapterManifest {
  readonly [preparedCppCuteSemanticAdapterManifestBrand]: true;
  readonly manifestId: typeof CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_MANIFEST_ID;
  readonly semanticAdapterId: "browsergrad.compiler.cpp-cute.clang-semantic-adapter@1";
  readonly clangVersion: "22.1.8";
  readonly resourceSha256: typeof CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256;
  readonly resourceByteLength: number;
  readonly designAuthority: true;
  readonly clangInvocationAuthorized: false;
}

export interface PreparedCppCuteSemanticAdapterManifestRecord {
  readonly manifest: CppCuteSemanticAdapterManifestV1;
}

interface StoredCppCuteSemanticAdapterManifest
  extends PreparedCppCuteSemanticAdapterManifestRecord {
  readonly bytes: Uint8Array;
}

export interface DecodeCppCuteSemanticAdapterManifestOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteSemanticAdapterManifestErrorCode =
  | "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-INVALID"
  | "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-HASH-UNAVAILABLE"
  | "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-NONCANONICAL-BYTES"
  | "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-UNVERIFIED";

export class CppCuteSemanticAdapterManifestError extends Error {
  constructor(
    readonly code: CppCuteSemanticAdapterManifestErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteSemanticAdapterManifestError";
  }
}

const BUILTIN_RESOURCE_BYTES = canonicalResourceBytes(
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE,
);

/** Returns a disposable copy of the exact canonical v1 policy resource. */
export function cppCuteSemanticAdapterManifestResourceBytes(): Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

/**
 * Snapshots hostile bytes, strict-decodes canonical JSON under fixed ceilings,
 * then verifies the content-derived manifest ID and pinned resource hash.
 */
export async function decodeCppCuteSemanticAdapterManifest(
  bytes: Uint8Array,
  options: DecodeCppCuteSemanticAdapterManifestOptions = {},
): Promise<PreparedCppCuteSemanticAdapterManifest> {
  const signal = normalizeOptions(options);
  const snapshot = snapshotBytes(bytes);
  throwIfAborted(signal);
  let value: JsonValue;
  try {
    value = decodeWireJson(snapshot, {
      limits: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$bytes", "semantic-adapter decoding exceeded fixed resource limits", { cause });
    }
    invalid("$bytes", "semantic-adapter bytes are not bounded strict JSON", { cause });
  }
  const manifest = parseManifest(value);
  const canonical = canonicalResourceBytes(manifest);
  if (!equalBytes(snapshot, canonical)) {
    fail(
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-NONCANONICAL-BYTES",
      "$bytes",
      "semantic-adapter bytes must exactly equal canonical JSON bytes",
    );
  }
  throwIfAborted(signal);
  const expectedManifestId = await deriveCppCuteSemanticAdapterManifestId(manifest.body);
  if (manifest.manifestId !== expectedManifestId) {
    hashMismatch("$.manifestId", `manifest ID must equal ${expectedManifestId}`);
  }
  throwIfAborted(signal);
  const resourceSha256 = await hashBytes(snapshot, "$bytes");
  if (resourceSha256 !== CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256 ||
      !equalBytes(snapshot, BUILTIN_RESOURCE_BYTES)) {
    hashMismatch("$bytes", "resource SHA-256 does not equal the pinned v1 policy identity");
  }
  throwIfAborted(signal);
  const frozen = deepFreezeJson(manifest) as CppCuteSemanticAdapterManifestV1;
  const prepared = Object.freeze({
    manifestId: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_MANIFEST_ID,
    semanticAdapterId: frozen.body.semanticAdapterId,
    clangVersion: frozen.body.clang.version,
    resourceSha256: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
    resourceByteLength: snapshot.byteLength,
    designAuthority: true,
    clangInvocationAuthorized: false,
  }) as PreparedCppCuteSemanticAdapterManifest;
  PREPARED_MANIFESTS.set(prepared, Object.freeze({
    manifest: frozen,
    bytes: new Uint8Array(snapshot),
  }));
  return prepared;
}

export async function deriveCppCuteSemanticAdapterManifestId(
  body: CppCuteSemanticAdapterManifestBodyV1,
): Promise<string> {
  const digest = await hashJson({
    domain: "browsergrad.compiler.cpp-cute.semantic-adapter-manifest-id.v1",
    body,
  }, "$.manifestId");
  return `bg.cpp.semantic-adapter.sha256.${digest}`;
}

export function unwrapPreparedCppCuteSemanticAdapterManifest(
  prepared: PreparedCppCuteSemanticAdapterManifest,
): PreparedCppCuteSemanticAdapterManifestRecord {
  return Object.freeze({ manifest: storedManifest(prepared).manifest });
}

export function canonicalCppCuteSemanticAdapterManifestBytes(
  prepared: PreparedCppCuteSemanticAdapterManifest,
): Uint8Array {
  return new Uint8Array(storedManifest(prepared).bytes);
}

export function cppCuteSemanticAdapterWarningArgv(
  prepared: PreparedCppCuteSemanticAdapterManifest,
  policyId: CppCuteSemanticAdapterWarningPolicyId,
  disposition: CppCuteSemanticAdapterWarningDisposition,
): readonly string[] {
  const entries = storedManifest(prepared).manifest.body.warningPolicyRegistry.entries;
  const entry = entries.find((candidate) => candidate.policyId === policyId);
  if (entry === undefined) unverified("unknown warning policy ID");
  return Object.freeze([...entry.argv[disposition]]);
}

function parseManifest(value: JsonValue): CppCuteSemanticAdapterManifestV1 {
  const object = closedObject(value, ["schema", "version", "manifestId", "body"], "$", true);
  literal(field(object, "schema", "$"), CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_SCHEMA, "$.schema");
  const version = closedObject(field(object, "version", "$"), ["major", "minor"], "$.version", true);
  if (version.major !== CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_MAJOR) {
    unsupported("$.version.major", `reader supports major ${CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_MAJOR}`);
  }
  if (version.minor !== CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_MINOR) {
    unsupported(
      "$.version.minor",
      `closed reader supports ${CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_MAJOR}.${CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_MINOR} only`,
    );
  }
  boundedPattern(field(object, "manifestId", "$"), "$.manifestId", MANIFEST_ID);
  parseBody(field(object, "body", "$"));
  return value as CppCuteSemanticAdapterManifestV1;
}

function parseBody(value: JsonValue): void {
  const expected = CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE.body;
  const body = closedObject(
    value,
    ["semanticAdapterId", "clang", "temporalMacros", "warningPolicyRegistry"],
    "$.body",
    true,
  );
  literal(field(body, "semanticAdapterId", "$.body"), expected.semanticAdapterId, "$.body.semanticAdapterId");

  const clang = closedObject(field(body, "clang", "$.body"), ["compilerId", "version"], "$.body.clang", true);
  literal(field(clang, "compilerId", "$.body.clang"), expected.clang.compilerId, "$.body.clang.compilerId");
  literal(field(clang, "version", "$.body.clang"), expected.clang.version, "$.body.clang.version");

  const temporal = closedObject(
    field(body, "temporalMacros", "$.body"),
    [
      "policyId",
      "mode",
      "macroNames",
      "consultation",
      "mutation",
      "enforcement",
      "diagnosticCodes",
      "defenseInDepthArgv",
    ],
    "$.body.temporalMacros",
    true,
  );
  literal(field(temporal, "policyId", "$.body.temporalMacros"), expected.temporalMacros.policyId, "$.body.temporalMacros.policyId");
  literal(field(temporal, "mode", "$.body.temporalMacros"), expected.temporalMacros.mode, "$.body.temporalMacros.mode");
  exactStrings(field(temporal, "macroNames", "$.body.temporalMacros"), expected.temporalMacros.macroNames, "$.body.temporalMacros.macroNames");
  literal(field(temporal, "consultation", "$.body.temporalMacros"), expected.temporalMacros.consultation, "$.body.temporalMacros.consultation");
  literal(field(temporal, "mutation", "$.body.temporalMacros"), expected.temporalMacros.mutation, "$.body.temporalMacros.mutation");
  literal(field(temporal, "enforcement", "$.body.temporalMacros"), expected.temporalMacros.enforcement, "$.body.temporalMacros.enforcement");
  const diagnosticCodes = closedObject(
    field(temporal, "diagnosticCodes", "$.body.temporalMacros"),
    ["consultation", "mutation"],
    "$.body.temporalMacros.diagnosticCodes",
    true,
  );
  literal(
    field(diagnosticCodes, "consultation", "$.body.temporalMacros.diagnosticCodes"),
    expected.temporalMacros.diagnosticCodes.consultation,
    "$.body.temporalMacros.diagnosticCodes.consultation",
  );
  literal(
    field(diagnosticCodes, "mutation", "$.body.temporalMacros.diagnosticCodes"),
    expected.temporalMacros.diagnosticCodes.mutation,
    "$.body.temporalMacros.diagnosticCodes.mutation",
  );
  exactStrings(
    field(temporal, "defenseInDepthArgv", "$.body.temporalMacros"),
    expected.temporalMacros.defenseInDepthArgv,
    "$.body.temporalMacros.defenseInDepthArgv",
  );

  const registry = closedObject(
    field(body, "warningPolicyRegistry", "$.body"),
    [
      "registryId",
      "compilerBaseline",
      "unknownPolicy",
      "reservedClangDiagnosticGroups",
      "entries",
    ],
    "$.body.warningPolicyRegistry",
    true,
  );
  literal(field(registry, "registryId", "$.body.warningPolicyRegistry"), expected.warningPolicyRegistry.registryId, "$.body.warningPolicyRegistry.registryId");
  literal(field(registry, "compilerBaseline", "$.body.warningPolicyRegistry"), expected.warningPolicyRegistry.compilerBaseline, "$.body.warningPolicyRegistry.compilerBaseline");
  literal(field(registry, "unknownPolicy", "$.body.warningPolicyRegistry"), expected.warningPolicyRegistry.unknownPolicy, "$.body.warningPolicyRegistry.unknownPolicy");
  exactStrings(
    field(registry, "reservedClangDiagnosticGroups", "$.body.warningPolicyRegistry"),
    expected.warningPolicyRegistry.reservedClangDiagnosticGroups,
    "$.body.warningPolicyRegistry.reservedClangDiagnosticGroups",
  );
  const entriesValue = field(registry, "entries", "$.body.warningPolicyRegistry");
  if (!Array.isArray(entriesValue) || entriesValue.length !== expected.warningPolicyRegistry.entries.length) {
    invalid("$.body.warningPolicyRegistry.entries", "entries must equal the exact ordered v1 registry");
  }
  for (let index = 0; index < expected.warningPolicyRegistry.entries.length; index += 1) {
    const expectedEntry = expected.warningPolicyRegistry.entries[index];
    const path = `$.body.warningPolicyRegistry.entries[${index}]`;
    if (expectedEntry === undefined) invalid(path, "missing built-in warning policy entry");
    const entry = closedObject(entriesValue[index] as JsonValue, ["policyId", "clangDiagnosticGroup", "argv"], path, true);
    literal(field(entry, "policyId", path), expectedEntry.policyId, `${path}.policyId`);
    literal(field(entry, "clangDiagnosticGroup", path), expectedEntry.clangDiagnosticGroup, `${path}.clangDiagnosticGroup`);
    const argv = closedObject(field(entry, "argv", path), ["ignore", "warn", "error"], `${path}.argv`, true);
    exactStrings(field(argv, "ignore", `${path}.argv`), expectedEntry.argv.ignore, `${path}.argv.ignore`);
    exactStrings(field(argv, "warn", `${path}.argv`), expectedEntry.argv.warn, `${path}.argv.warn`);
    exactStrings(field(argv, "error", `${path}.argv`), expectedEntry.argv.error, `${path}.argv.error`);
  }
}

function storedManifest(
  prepared: PreparedCppCuteSemanticAdapterManifest,
): StoredCppCuteSemanticAdapterManifest {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const stored = PREPARED_MANIFESTS.get(prepared as object);
  if (stored === undefined) unverified();
  return stored;
}

function snapshotBytes(value: unknown): Uint8Array {
  let inspected;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid("$bytes", "semantic-adapter input must be an unshared plain Uint8Array", { cause });
  }
  if (inspected.byteLength === 0) invalid("$bytes", "semantic-adapter bytes must be nonempty");
  if (inspected.byteLength > CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_BYTE_LIMIT) {
    resource("$bytes", `semantic-adapter exceeds ${CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_BYTE_LIMIT} bytes`);
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid("$bytes", "semantic-adapter bytes became unreadable while snapshotting", { cause });
  }
}

function normalizeOptions(
  options: DecodeCppCuteSemanticAdapterManifestOptions,
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
      "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-CANCELLED",
      "$options.signal",
      "semantic-adapter manifest preparation was cancelled",
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
  for (const key of Object.keys(object)) {
    if (!keys.includes(key)) invalid(path, `unknown field ${key}`);
  }
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

function literal<T extends string>(value: JsonValue, expected: T, path: string): asserts value is T {
  if (value !== expected) invalid(path, `must equal ${JSON.stringify(expected)}`);
}

function boundedPattern(value: JsonValue, path: string, pattern: RegExp): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 128 || !pattern.test(value)) {
    invalid(path, "string does not match required closed format");
  }
  return value;
}

function exactStrings(value: JsonValue, expected: readonly string[], path: string): void {
  if (!Array.isArray(value) || value.length !== expected.length ||
      value.some((entry, index) => entry !== expected[index])) {
    invalid(path, "values must equal the exact ordered v1 inventory");
  }
}

function canonicalResourceBytes(value: JsonValue): Uint8Array {
  try {
    return canonicalJsonBytes(value, {
      limits: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_DECODE_LIMITS,
    });
  } catch (cause) {
    if (isSchemaResourceLimit(cause)) {
      resource("$", "canonical semantic-adapter exceeds fixed resource limits", { cause });
    }
    invalid("$", "semantic-adapter cannot be canonically encoded", { cause });
  }
}

async function hashJson(value: JsonValue, path: string): Promise<string> {
  try {
    return await hashCanonicalJson(value, {
      limits: CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_DECODE_LIMITS,
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

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-INVALID", path, message, options);
}

function unsupported(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-UNSUPPORTED-VERSION", path, message);
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-RESOURCE-LIMIT", path, message, options);
}

function hashMismatch(path: string, message: string): never {
  fail("BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-HASH-MISMATCH", path, message);
}

function hashUnavailable(path: string, cause: unknown): never {
  fail(
    "BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-HASH-UNAVAILABLE",
    path,
    "SHA-256 is unavailable",
    { cause },
  );
}

function unverified(message = "prepared semantic-adapter authority is not recognized"): never {
  fail("BG-COMPILER-CPP-CUTE-SEMANTIC-ADAPTER-UNVERIFIED", "$prepared", message);
}

function fail(
  code: CppCuteSemanticAdapterManifestErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteSemanticAdapterManifestError(code, path, message, options);
}

import { canonicalJsonBytes } from "./canonical-json.js";
import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "./diagnostics.js";
import { assertJsonValue, deepFreezeJson, isJsonObject, type JsonObject, type JsonValue } from "./json.js";
import type { DecodeLimits } from "./limits.js";

export interface WireVersion extends JsonObject {
  readonly major: number;
  readonly minor: number;
}

export interface WireProducer extends JsonObject {
  readonly id: string;
  readonly version: string;
}

export interface WireEnvelope<T extends JsonValue> extends JsonObject {
  readonly schema: string;
  readonly version: WireVersion;
  readonly producer: WireProducer;
  readonly artifactId: string;
  readonly payload: T;
  readonly requiredExtensions: readonly string[];
  readonly optionalMetadata?: JsonObject;
}

export interface EnvelopeValidationOptions<T extends JsonValue> {
  readonly schema: string;
  readonly supportedMajor: number;
  readonly supportedMinor: number;
  readonly knownRequiredExtensions?: ReadonlySet<string>;
  readonly validatePayload?: (value: JsonValue, path: string) => T;
  readonly limits?: Partial<DecodeLimits>;
}

export interface ArtifactVerificationOptions<T extends JsonValue>
  extends EnvelopeValidationOptions<T> {
  /**
   * Trusted schema verifier. It performs raw validation, normalization,
   * deterministic ID remapping, reference checks, and normalized semantic
   * verification before returning the payload.
   */
  readonly validatePayload: (value: JsonValue, path: string) => T;
}

declare const verifiedArtifactBrand: unique symbol;

export interface VerifiedArtifact<T extends JsonValue> {
  readonly [verifiedArtifactBrand]: T;
}

interface VerifiedEnvelopeRecord {
  readonly envelope: WireEnvelope<JsonValue>;
  readonly authority: object;
}

const VERIFIED_ENVELOPES = new WeakMap<object, VerifiedEnvelopeRecord>();

class VerifiedArtifactValue {
  constructor(envelope: WireEnvelope<JsonValue>, authority: object) {
    VERIFIED_ENVELOPES.set(this, { envelope, authority });
    Object.freeze(this);
  }
}

const ENVELOPE_FIELDS = new Set([
  "schema",
  "version",
  "producer",
  "artifactId",
  "payload",
  "requiredExtensions",
  "optionalMetadata",
]);

export function validateWireEnvelope<T extends JsonValue = JsonValue>(
  value: unknown,
  options: EnvelopeValidationOptions<T>,
): WireEnvelope<T> {
  assertJsonValue(value, options.limits === undefined ? {} : { limits: options.limits });
  if (!isJsonObject(value)) invalid("$", "wire envelope must be an object");
  rejectUnknownFields(value, ENVELOPE_FIELDS, "$", "wire envelope");
  if (value.schema !== options.schema) invalid("$.schema", `expected schema ${options.schema}`);
  const versionValue = value.version;
  if (versionValue === undefined) invalid("$.version", "version is required");
  const version = validateVersion(versionValue, options);
  const producerValue = value.producer;
  if (producerValue === undefined) invalid("$.producer", "producer is required");
  const producer = validateProducer(producerValue);
  if (typeof value.artifactId !== "string" || value.artifactId.length === 0) invalid("$.artifactId", "artifactId must be a non-empty string");
  if (!Array.isArray(value.requiredExtensions)) invalid("$.requiredExtensions", "requiredExtensions must be an array");
  const requiredExtensions = value.requiredExtensions.map((extension, index) => {
    if (typeof extension !== "string" || !/^[a-z][a-z0-9.-]*:[a-z][a-z0-9._-]*(?:@[0-9]+)?$/u.test(extension)) {
      invalid(`$.requiredExtensions[${index}]`, "required extension ID must be namespaced and versioned only by an optional integer major");
    }
    return extension;
  });
  if (new Set(requiredExtensions).size !== requiredExtensions.length) invalid("$.requiredExtensions", "required extension IDs must be unique");
  const known = options.knownRequiredExtensions ?? new Set<string>();
  for (const extension of requiredExtensions) {
    if (!known.has(extension)) {
      throw schemaError(
        SCHEMA_DIAGNOSTIC_CODES.unknownRequiredExtension,
        `unknown required extension ${extension}`,
        { path: "$.requiredExtensions" },
      );
    }
  }
  const payloadValue = value.payload;
  if (payloadValue === undefined) invalid("$.payload", "payload is required");
  const payload = options.validatePayload === undefined
    ? payloadValue as T
    : options.validatePayload(payloadValue, "$.payload");
  assertJsonValue(payload, options.limits === undefined ? {} : { limits: options.limits });
  const optionalMetadata = value.optionalMetadata;
  if (optionalMetadata !== undefined && !isJsonObject(optionalMetadata)) invalid("$.optionalMetadata", "optionalMetadata must be a JSON object");

  const normalized: WireEnvelope<T> = {
    schema: value.schema,
    version,
    producer,
    artifactId: value.artifactId,
    payload,
    requiredExtensions: [...requiredExtensions].sort(),
    ...(optionalMetadata === undefined ? {} : { optionalMetadata }),
  };
  return deepFreezeJson(normalized);
}

export function verifyWireArtifact<T extends JsonValue>(
  value: unknown,
  options: ArtifactVerificationOptions<T>,
  authority: object,
): VerifiedArtifact<T> {
  return new VerifiedArtifactValue(validateWireEnvelope(value, options), authority) as unknown as VerifiedArtifact<T>;
}

/**
 * Copies one verifier-issued artifact into its complete canonical wire form.
 *
 * The opaque in-memory authority is deliberately not transferable between
 * realms. A receiver must decode and verify these bytes again with its own
 * schema authority before semantic preparation or execution.
 */
export function copyVerifiedArtifactWireBytes(
  artifact: VerifiedArtifact<JsonValue>,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Uint8Array {
  return canonicalJsonBytes(unwrapVerifiedArtifact(artifact), options);
}

/** @internal Package hashing and verified evaluators only. */
export function unwrapVerifiedArtifact<T extends JsonValue>(
  artifact: VerifiedArtifact<T>,
  authority?: object,
): WireEnvelope<T> {
  if (typeof artifact !== "object" || artifact === null) unverifiedArtifact();
  const record = VERIFIED_ENVELOPES.get(artifact as object);
  if (record === undefined || (authority !== undefined && record.authority !== authority)) unverifiedArtifact();
  return record.envelope as WireEnvelope<T>;
}

function unverifiedArtifact(): never {
  throw schemaError(
    SCHEMA_DIAGNOSTIC_CODES.unverifiedArtifact,
    "semantic artifact operation requires an opaque verified artifact",
    { path: "$" },
  );
}

function validateVersion(value: JsonValue, options: EnvelopeValidationOptions<JsonValue>): WireVersion {
  if (!isJsonObject(value)) invalid("$.version", "version must be an object");
  rejectUnknownFields(value, new Set(["major", "minor"]), "$.version", "version");
  if (!isNonNegativeSafeInteger(value.major) || !isNonNegativeSafeInteger(value.minor)) invalid("$.version", "major and minor must be non-negative safe integers");
  if (value.major !== options.supportedMajor) {
    throw schemaError(
      SCHEMA_DIAGNOSTIC_CODES.unsupportedVersion,
      `unsupported schema major ${value.major}; reader supports ${options.supportedMajor}`,
      { path: "$.version.major" },
    );
  }
  // A newer minor is readable because minor additions are restricted to open,
  // losslessly preserved metadata/extension bags. Closed records stay closed.
  return Object.freeze({ major: value.major, minor: value.minor });
}

function validateProducer(value: JsonValue): WireProducer {
  if (!isJsonObject(value)) invalid("$.producer", "producer must be an object");
  rejectUnknownFields(value, new Set(["id", "version"]), "$.producer", "producer");
  if (typeof value.id !== "string" || value.id.length === 0) invalid("$.producer.id", "producer id must be a non-empty string");
  if (typeof value.version !== "string" || value.version.length === 0) invalid("$.producer.version", "producer version must be a non-empty string");
  return Object.freeze({ id: value.id, version: value.version });
}

function rejectUnknownFields(value: JsonObject, allowed: ReadonlySet<string>, path: string, name: string): void {
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) invalid(path, `${name} has unknown closed-record fields: ${unknown.sort().join(", ")}`);
}

function isNonNegativeSafeInteger(value: JsonValue | undefined): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function invalid(path: string, message: string): never {
  throw schemaError(SCHEMA_DIAGNOSTIC_CODES.invalidEnvelope, message, { path });
}

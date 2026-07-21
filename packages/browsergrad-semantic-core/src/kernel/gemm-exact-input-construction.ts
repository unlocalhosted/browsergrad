import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "../schema/diagnostics.js";
import type { WireProducer } from "../schema/envelope.js";
import { isJsonObject, parseWireJson, type JsonObject, type JsonValue } from "../schema/json.js";
import { DEFAULT_DECODE_LIMITS } from "../schema/limits.js";
import type { VerifiedLogicalGemmTileArtifact } from "./gemm-tile-artifact.js";
import {
  LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA,
  evaluateLogicalGemmExactF32Inputs,
  verifyEvaluatedLogicalGemmExactF32InputCertificate,
  type CertifyLogicalGemmExactF32InputsRequest,
  type VerifiedLogicalGemmExactF32InputCertificate,
} from "./gemm-exact-input-artifact.js";

const DEFAULT_PRODUCER = Object.freeze({
  id: "browsergrad.semantic-core.logical-gemm-exact-f32-input-construction",
  version: "1",
});

export interface LogicalGemmExactF32InputCertificateConstructionOptions {
  readonly producer?: WireProducer;
  readonly artifactId?: string;
}

export interface ConstructedLogicalGemmExactF32InputCertificate {
  readonly certificate: VerifiedLogicalGemmExactF32InputCertificate;
  readonly logicalGemmSemanticHash: string;
  readonly specializationHash: string;
}

/**
 * Constructs an authority-bound certificate from a synchronous snapshot of
 * concrete host allocation bytes. The certificate retains its own private
 * snapshots, so later caller mutation cannot change what it authorizes.
 */
export async function createVerifiedLogicalGemmExactF32InputCertificate(
  layout: VerifiedLayoutArtifact,
  logicalGemm: VerifiedLogicalGemmTileArtifact,
  request: CertifyLogicalGemmExactF32InputsRequest,
  options: LogicalGemmExactF32InputCertificateConstructionOptions = {},
): Promise<ConstructedLogicalGemmExactF32InputCertificate> {
  const normalizedOptions = normalizeOptions(options);
  const evaluation = await evaluateLogicalGemmExactF32Inputs(layout, logicalGemm, request);
  const certificate = verifyEvaluatedLogicalGemmExactF32InputCertificate({
    schema: LOGICAL_GEMM_EXACT_F32_INPUT_CERTIFICATE_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: normalizedOptions.producer,
    artifactId: normalizedOptions.artifactId,
    requiredExtensions: [],
    payload: evaluation.payload,
  }, evaluation);
  return Object.freeze({
    certificate,
    logicalGemmSemanticHash: evaluation.payload.logicalGemmSemanticHash,
    specializationHash: evaluation.payload.specializationHash,
  });
}

interface NormalizedOptions {
  readonly producer: WireProducer;
  readonly artifactId: string;
}

function normalizeOptions(options: LogicalGemmExactF32InputCertificateConstructionOptions): NormalizedOptions {
  const snapshot = snapshotJson(options);
  const object = closedRecord(snapshot, ["producer", "artifactId"], [], "$options");
  return Object.freeze({
    producer: object.producer === undefined ? DEFAULT_PRODUCER : parseProducer(object.producer),
    artifactId: object.artifactId === undefined
      ? "logical-gemm-exact-f32-input"
      : nonemptyString(object.artifactId, "$options.artifactId"),
  });
}

function parseProducer(value: JsonValue): WireProducer {
  const object = closedRecord(value, ["id", "version"], ["id", "version"], "$options.producer");
  return {
    id: nonemptyString(object.id, "$options.producer.id"),
    version: nonemptyString(object.version, "$options.producer.version"),
  };
}

function snapshotJson(value: unknown): JsonValue {
  return parseWireJson(canonicalizeJson(value, { limits: DEFAULT_DECODE_LIMITS }), { limits: DEFAULT_DECODE_LIMITS });
}

function closedRecord(
  value: JsonValue | undefined,
  allowedFields: readonly string[],
  requiredFields: readonly string[],
  path: string,
): JsonObject {
  if (value === undefined || !isJsonObject(value)) constructionError(path, "expected a plain JSON object");
  const allowed = new Set(allowedFields);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) constructionError(path, `unknown fields: ${unknown.sort().join(", ")}`);
  for (const fieldName of requiredFields) {
    if (value[fieldName] === undefined) constructionError(`${path}.${fieldName}`, "required field is missing");
  }
  return value;
}

function nonemptyString(value: JsonValue | undefined, path: string): string {
  if (typeof value !== "string" || value.length === 0) constructionError(path, "expected a non-empty string");
  return value;
}

function constructionError(path: string, message: string): never {
  throw schemaError(
    SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
    `logical GEMM exact f32 input certificate construction request ${message}`,
    { path },
  );
}

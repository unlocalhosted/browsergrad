import type { VerifiedLogicalGemmTileArtifact } from "../kernel/gemm-tile-artifact.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "../schema/diagnostics.js";
import type { WireProducer } from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import { parseWireU64, type WireU64 } from "../schema/integers.js";
import { isJsonObject, parseWireJson, type JsonObject, type JsonValue } from "../schema/json.js";
import { DEFAULT_DECODE_LIMITS, resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import {
  LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
  verifyLogicalGemmTileScheduleArtifact,
  type VerifiedLogicalGemmTileScheduleArtifact,
} from "./gemm-tile-artifact.js";

const DEFAULT_PRODUCER = Object.freeze({
  id: "browsergrad.semantic-core.logical-gemm-tile-schedule-construction",
  version: "1",
});

export interface CreateVerifiedLogicalGemmTileScheduleRequest {
  readonly physicalTile: {
    readonly m: WireU64;
    readonly n: WireU64;
    readonly k: WireU64;
  };
}

export interface LogicalGemmTileScheduleConstructionOptions {
  readonly producer?: WireProducer;
  readonly artifactId?: string;
  readonly limits?: Partial<DecodeLimits>;
}

export interface ConstructedLogicalGemmTileSchedule {
  readonly artifact: VerifiedLogicalGemmTileScheduleArtifact;
  readonly logicalGemmSemanticHash: string;
  readonly scheduleSemanticHash: string;
}

/**
 * Constructs the initial scalar, single-buffered cooperative GEMM schedule.
 * Workgroup mapping, staging, barriers, vectors, and masks are derived from the
 * selected physical tile instead of accepted as caller-defined semantics.
 */
export async function createVerifiedLogicalGemmTileSchedule(
  logicalGemm: VerifiedLogicalGemmTileArtifact,
  request: CreateVerifiedLogicalGemmTileScheduleRequest,
  options: LogicalGemmTileScheduleConstructionOptions = {},
): Promise<ConstructedLogicalGemmTileSchedule> {
  const normalizedOptions = normalizeOptions(options);
  const snapshot = snapshotJson(request, normalizedOptions.limits);
  const object = closedRecord(snapshot, ["physicalTile"], ["physicalTile"], "$");
  const tile = closedRecord(object.physicalTile, ["m", "n", "k"], ["m", "n", "k"], "$.physicalTile");
  const m = parseWireU64(tile.m, "$.physicalTile.m");
  const n = parseWireU64(tile.n, "$.physicalTile.n");
  const k = parseWireU64(tile.k, "$.physicalTile.k");
  const logicalGemmSemanticHash = await hashSemanticArtifact(logicalGemm, { limits: normalizedOptions.limits });
  const artifact = await verifyLogicalGemmTileScheduleArtifact({
    schema: LOGICAL_GEMM_TILE_SCHEDULE_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: normalizedOptions.producer,
    artifactId: normalizedOptions.artifactId,
    requiredExtensions: [],
    payload: {
      logicalGemmSemanticHash,
      schedule: {
        kind: "logical-gemm-tile-schedule",
        version: { major: 1, minor: 0 },
        physicalTile: { m, n, k },
        workgroup: {
          size: { x: n, y: m, z: "1" },
          x: "physical-tile-column",
          y: "physical-tile-row",
          z: "singleton",
        },
        invocation: {
          output: "one-element",
          localX: "output-column",
          localY: "output-row",
          localZ: "unused",
        },
        staging: { space: "workgroup", lhs: "cooperative", rhs: "cooperative", buffering: "single" },
        participation: { workgroup: "all-invocations", boundaryLanes: "participate", earlyExit: "forbidden" },
        uniformity: { barrierControl: "workgroup-uniform", activeMaskScope: "memory-effects-only" },
        vectorization: { lhsLoad: "1", rhsLoad: "1", destinationStore: "1" },
        barriers: {
          afterCooperativeLoad: { scope: "workgroup", memory: "workgroup", semantics: "acquire-release" },
          beforeStagingReuse: { scope: "workgroup", memory: "workgroup", semantics: "acquire-release" },
        },
        masks: { lhsLoad: "zero-fill", rhsLoad: "zero-fill", destinationStore: "suppress" },
      },
    },
  }, { logicalGemm, limits: normalizedOptions.limits });
  return Object.freeze({
    artifact,
    logicalGemmSemanticHash,
    scheduleSemanticHash: await hashSemanticArtifact(artifact, { limits: normalizedOptions.limits }),
  });
}

interface NormalizedOptions {
  readonly producer: WireProducer;
  readonly artifactId: string;
  readonly limits: DecodeLimits;
}

function normalizeOptions(options: LogicalGemmTileScheduleConstructionOptions): NormalizedOptions {
  const snapshot = snapshotJson(options, DEFAULT_DECODE_LIMITS);
  const object = closedRecord(snapshot, ["producer", "artifactId", "limits"], [], "$options");
  const rawLimits = object.limits;
  if (rawLimits !== undefined) closedRecord(rawLimits, Object.keys(DEFAULT_DECODE_LIMITS), [], "$options.limits");
  const limits = resolveDecodeLimits(rawLimits === undefined ? {} : rawLimits as Partial<DecodeLimits>);
  const producer = object.producer === undefined ? DEFAULT_PRODUCER : parseProducer(object.producer);
  return Object.freeze({
    producer,
    artifactId: object.artifactId === undefined ? "logical-gemm-tile-schedule" : nonemptyString(object.artifactId, "$options.artifactId"),
    limits,
  });
}

function parseProducer(value: JsonValue): WireProducer {
  const object = closedRecord(value, ["id", "version"], ["id", "version"], "$options.producer");
  return {
    id: nonemptyString(object.id, "$options.producer.id"),
    version: nonemptyString(object.version, "$options.producer.version"),
  };
}

function snapshotJson(value: unknown, limits: Partial<DecodeLimits>): JsonValue {
  return parseWireJson(canonicalizeJson(value, { limits }), { limits });
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
  for (const field of requiredFields) if (value[field] === undefined) constructionError(`${path}.${field}`, "required field is missing");
  return value;
}

function nonemptyString(value: JsonValue | undefined, path: string): string {
  if (typeof value !== "string" || value.length === 0) constructionError(path, "expected a non-empty string");
  return value;
}

function constructionError(path: string, message: string): never {
  throw schemaError(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue, `logical GEMM tile schedule construction request ${message}`, { path });
}

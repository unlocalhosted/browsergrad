import type { VerifiedAttentionForwardArtifact } from "../kernel/attention-forward-artifact.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { SCHEMA_DIAGNOSTIC_CODES, schemaError } from "../schema/diagnostics.js";
import type { WireProducer } from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import { parseWireU64, type WireU64 } from "../schema/integers.js";
import { isJsonObject, parseWireJson, type JsonObject, type JsonValue } from "../schema/json.js";
import { DEFAULT_DECODE_LIMITS, resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import {
  ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
  verifyAttentionOnlineKvTileScheduleArtifact,
  type VerifiedAttentionOnlineKvTileScheduleArtifact,
} from "./attention-online-kv-tile-artifact.js";

const DEFAULT_PRODUCER = Object.freeze({
  id: "browsergrad.semantic-core.attention-online-kv-tile-schedule-construction",
  version: "1",
});

export interface CreateVerifiedAttentionOnlineKvTileScheduleRequest {
  readonly physicalTile: {
    readonly queryRows: WireU64;
    readonly keyRows: WireU64;
  };
}

export interface AttentionOnlineKvTileScheduleConstructionOptions {
  readonly producer?: WireProducer;
  readonly artifactId?: string;
  readonly limits?: Partial<DecodeLimits>;
}

export interface ConstructedAttentionOnlineKvTileSchedule {
  readonly artifact: VerifiedAttentionOnlineKvTileScheduleArtifact;
  readonly attentionForwardSemanticHash: string;
  readonly scheduleSemanticHash: string;
}

/**
 * Constructs the initial scalar, single-buffered online K/V tile schedule.
 * Mapping, staging, recurrence, barriers, vectors, and masks are derived from
 * the requested physical tile rather than accepted as caller-authored facts.
 */
export async function createVerifiedAttentionOnlineKvTileSchedule(
  attentionForward: VerifiedAttentionForwardArtifact,
  request: CreateVerifiedAttentionOnlineKvTileScheduleRequest,
  options: AttentionOnlineKvTileScheduleConstructionOptions = {},
): Promise<ConstructedAttentionOnlineKvTileSchedule> {
  const normalizedOptions = normalizeOptions(options);
  const snapshot = snapshotJson(request, normalizedOptions.limits);
  const object = closedRecord(snapshot, ["physicalTile"], ["physicalTile"], "$");
  const tile = closedRecord(
    object.physicalTile,
    ["queryRows", "keyRows"],
    ["queryRows", "keyRows"],
    "$.physicalTile",
  );
  const queryRows = parseWireU64(tile.queryRows, "$.physicalTile.queryRows");
  const keyRows = parseWireU64(tile.keyRows, "$.physicalTile.keyRows");
  const attentionForwardSemanticHash = await hashSemanticArtifact(
    attentionForward,
    { limits: normalizedOptions.limits },
  );
  const artifact = await verifyAttentionOnlineKvTileScheduleArtifact({
    schema: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: normalizedOptions.producer,
    artifactId: normalizedOptions.artifactId,
    requiredExtensions: [],
    payload: {
      attentionForwardSemanticHash,
      schedule: {
        kind: "attention-online-kv-tile-schedule",
        version: { major: 1, minor: 0 },
        physicalTile: { queryRows, keyRows },
        workgroup: {
          size: { x: queryRows, y: "1", z: "1" },
          dispatchX: "query-tile",
          dispatchY: "head",
          dispatchZ: "batch",
        },
        invocation: {
          localX: "query-row-within-tile",
          localY: "unused",
          localZ: "unused",
          privateQuery: "one-logical-query-row",
          privateOutput: "one-logical-output-row",
        },
        traversal: {
          keyTiles: "increasing-key-index",
          keysWithinTile: "increasing-key-index",
          coverage: "complete-logical-key-range",
          tail: "masked-final-tile",
        },
        staging: {
          space: "workgroup",
          key: "cooperative",
          value: "cooperative",
          layout: "key-major-contiguous-depth",
          buffering: "single",
        },
        onlineSoftmax: {
          state: "running-maximum-denominator-and-weighted-value",
          tileScores: "scaled-query-key-dot-products",
          tileMaximum: "maximum-over-valid-tile-scores",
          tileReductionOrder: "increasing-key-index",
          update: "rescale-prior-state-then-accumulate-current-tile",
          priorRescale: "exp-previous-maximum-minus-new-maximum",
          currentWeight: "exp-score-minus-new-maximum",
          finalize: "divide-weighted-value-by-denominator-after-all-key-tiles",
        },
        participation: {
          workgroup: "all-invocations",
          boundaryQueryLanes: "participate",
          earlyExit: "forbidden",
        },
        uniformity: {
          barrierControl: "workgroup-uniform",
          activeMaskScope: "memory-effects-and-online-state-only",
        },
        vectorization: { keyLoad: "1", valueLoad: "1", destinationStore: "1" },
        barriers: {
          afterCooperativeLoad: workgroupBarrier(),
          beforeStagingReuse: workgroupBarrier(),
        },
        masks: {
          queryLane: "suppress-logical-state-and-store",
          keyLoad: "zero-fill",
          valueLoad: "zero-fill",
          invalidKeyScore: "exclude-before-online-state-update",
          logicalMask: "exclude-before-online-state-update",
          destinationStore: "suppress",
        },
      },
    },
  }, { attentionForward, limits: normalizedOptions.limits });
  return Object.freeze({
    artifact,
    attentionForwardSemanticHash,
    scheduleSemanticHash: await hashSemanticArtifact(artifact, { limits: normalizedOptions.limits }),
  });
}

function workgroupBarrier() {
  return {
    scope: "workgroup" as const,
    memory: "workgroup" as const,
    semantics: "acquire-release" as const,
  };
}

interface NormalizedOptions {
  readonly producer: WireProducer;
  readonly artifactId: string;
  readonly limits: DecodeLimits;
}

function normalizeOptions(options: AttentionOnlineKvTileScheduleConstructionOptions): NormalizedOptions {
  const snapshot = snapshotJson(options, DEFAULT_DECODE_LIMITS);
  const object = closedRecord(snapshot, ["producer", "artifactId", "limits"], [], "$options");
  const rawLimits = object.limits;
  if (rawLimits !== undefined) {
    closedRecord(rawLimits, Object.keys(DEFAULT_DECODE_LIMITS), [], "$options.limits");
  }
  const limits = resolveDecodeLimits(rawLimits === undefined ? {} : rawLimits as Partial<DecodeLimits>);
  const producer = object.producer === undefined ? DEFAULT_PRODUCER : parseProducer(object.producer);
  return Object.freeze({
    producer,
    artifactId: object.artifactId === undefined
      ? "attention-online-kv-tile-schedule"
      : nonemptyString(object.artifactId, "$options.artifactId"),
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
  for (const field of requiredFields) {
    if (value[field] === undefined) constructionError(`${path}.${field}`, "required field is missing");
  }
  return value;
}

function nonemptyString(value: JsonValue | undefined, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    constructionError(path, "expected a non-empty string");
  }
  return value;
}

function constructionError(path: string, message: string): never {
  throw schemaError(
    SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue,
    `attention online K/V tile schedule construction request ${message}`,
    { path },
  );
}

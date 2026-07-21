import {
  attentionForwardArtifactPayload,
  type VerifiedAttentionForwardArtifact,
} from "../kernel/attention-forward-artifact.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { SCHEDULE_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import {
  unwrapVerifiedArtifact,
  validateWireEnvelope,
  verifyWireArtifact,
  type VerifiedArtifact,
  type WireEnvelope,
} from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import { parseWireU64, wireIntegerToBigInt } from "../schema/integers.js";
import { decodeWireJson, isJsonObject, type JsonObject, type JsonValue } from "../schema/json.js";
import { resolveDecodeLimits, type DecodeLimits } from "../schema/limits.js";
import {
  INITIAL_ATTENTION_ONLINE_KV_MAX_TILE_ROWS,
  type AttentionOnlineKvTileSchedule,
  type AttentionOnlineKvWorkgroupBarrier,
} from "./attention-online-kv-tile-model.js";

export const ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA =
  "browsergrad.schedule.attention-online-kv-tile";
export const ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_MAJOR = 1;
export const ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_MINOR = 0;

const AUTHORITY = Object.freeze({
  schema: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
  major: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_MAJOR,
});

export type AttentionOnlineKvTileScheduleArtifactPayloadV1 = JsonObject & {
  readonly attentionForwardSemanticHash: string;
  readonly schedule: AttentionOnlineKvTileSchedule;
};

export type VerifiedAttentionOnlineKvTileScheduleArtifact =
  VerifiedArtifact<AttentionOnlineKvTileScheduleArtifactPayloadV1>;

export interface AttentionOnlineKvTileScheduleArtifactVerificationOptions {
  readonly attentionForward: VerifiedAttentionForwardArtifact;
  readonly limits?: Partial<DecodeLimits>;
}

export async function verifyAttentionOnlineKvTileScheduleArtifact(
  value: unknown,
  options: AttentionOnlineKvTileScheduleArtifactVerificationOptions,
): Promise<VerifiedAttentionOnlineKvTileScheduleArtifact> {
  const limits = resolveDecodeLimits(options.limits);
  attentionForwardArtifactPayload(options.attentionForward);
  const envelope = validateWireEnvelope(value, {
    schema: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
    supportedMajor: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_MAJOR,
    supportedMinor: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
  });
  const normalized = parsePayload(envelope.payload);
  const attentionForwardSemanticHash = await hashSemanticArtifact(
    options.attentionForward,
    { limits },
  );
  if (normalized.attentionForwardSemanticHash !== attentionForwardSemanticHash) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch,
      "$.payload.attentionForwardSemanticHash",
      "attention schedule does not reference the supplied verified attention-forward semantics",
    );
  }
  verifyCompatibility(normalized.schedule);
  const normalizedEnvelope: WireEnvelope<JsonValue> = {
    ...envelope,
    payload: normalized as unknown as JsonValue,
  };
  canonicalizeJson(normalizedEnvelope, { limits });
  return verifyWireArtifact(normalizedEnvelope, {
    schema: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA,
    supportedMajor: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_MAJOR,
    supportedMinor: ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
    validatePayload: (payload) => payload,
  }, AUTHORITY) as VerifiedAttentionOnlineKvTileScheduleArtifact;
}

export async function decodeAttentionOnlineKvTileScheduleArtifact(
  bytes: Uint8Array,
  options: AttentionOnlineKvTileScheduleArtifactVerificationOptions,
): Promise<VerifiedAttentionOnlineKvTileScheduleArtifact> {
  return verifyAttentionOnlineKvTileScheduleArtifact(
    decodeWireJson(bytes, options.limits === undefined ? {} : { limits: options.limits }),
    options,
  );
}

export function attentionOnlineKvTileScheduleArtifactPayload(
  artifact: VerifiedAttentionOnlineKvTileScheduleArtifact,
): AttentionOnlineKvTileScheduleArtifactPayloadV1 {
  const envelope = unwrapVerifiedArtifact(artifact, AUTHORITY);
  if (envelope.schema !== ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_SCHEMA
    || envelope.version.major !== ATTENTION_ONLINE_KV_TILE_SCHEDULE_ARTIFACT_MAJOR) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact,
      "$",
      "verified artifact is not a browsergrad.schedule.attention-online-kv-tile@1 artifact",
    );
  }
  return envelope.payload;
}

function parsePayload(value: JsonValue): AttentionOnlineKvTileScheduleArtifactPayloadV1 {
  const object = closedObject(value, ["attentionForwardSemanticHash", "schedule"], "$.payload");
  const attentionForwardSemanticHash = stringValue(
    field(object, "attentionForwardSemanticHash", "$.payload"),
    "$.payload.attentionForwardSemanticHash",
  );
  if (!/^[0-9a-f]{64}$/u.test(attentionForwardSemanticHash)) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact,
      "$.payload.attentionForwardSemanticHash",
      "attention-forward semantic hash must be 64 lowercase hexadecimal digits",
    );
  }
  return {
    attentionForwardSemanticHash,
    schedule: parseSchedule(field(object, "schedule", "$.payload"), "$.payload.schedule"),
  } as AttentionOnlineKvTileScheduleArtifactPayloadV1;
}

function parseSchedule(value: JsonValue, path: string): AttentionOnlineKvTileSchedule {
  const object = closedObject(value, [
    "kind", "version", "physicalTile", "workgroup", "invocation", "traversal",
    "staging", "onlineSoftmax", "participation", "uniformity", "vectorization",
    "barriers", "masks",
  ], path);
  if (object.kind !== "attention-online-kv-tile-schedule") {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact,
      `${path}.kind`,
      "expected attention-online-kv-tile-schedule",
    );
  }
  const version = closedObject(field(object, "version", path), ["major", "minor"], `${path}.version`);
  if (version.major !== 1 || version.minor !== 0) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.version`,
      "attention online K/V tile schedule reader supports version 1.0 only",
    );
  }
  const physicalTile = parsePhysicalTile(field(object, "physicalTile", path), `${path}.physicalTile`);
  const workgroup = closedObject(
    field(object, "workgroup", path),
    ["size", "dispatchX", "dispatchY", "dispatchZ"],
    `${path}.workgroup`,
  );
  const workgroupSize = parseExtentXyz(field(workgroup, "size", `${path}.workgroup`), `${path}.workgroup.size`);
  requireExactObject(workgroup, {
    dispatchX: "query-tile",
    dispatchY: "head",
    dispatchZ: "batch",
  }, `${path}.workgroup`);
  const invocation = closedObject(
    field(object, "invocation", path),
    ["localX", "localY", "localZ", "privateQuery", "privateOutput"],
    `${path}.invocation`,
  );
  requireExactObject(invocation, {
    localX: "query-row-within-tile",
    localY: "unused",
    localZ: "unused",
    privateQuery: "one-logical-query-row",
    privateOutput: "one-logical-output-row",
  }, `${path}.invocation`);
  const traversal = closedObject(
    field(object, "traversal", path),
    ["keyTiles", "keysWithinTile", "coverage", "tail"],
    `${path}.traversal`,
  );
  requireExactObject(traversal, {
    keyTiles: "increasing-key-index",
    keysWithinTile: "increasing-key-index",
    coverage: "complete-logical-key-range",
    tail: "masked-final-tile",
  }, `${path}.traversal`);
  const staging = closedObject(
    field(object, "staging", path),
    ["space", "key", "value", "layout", "buffering"],
    `${path}.staging`,
  );
  requireExactObject(staging, {
    space: "workgroup",
    key: "cooperative",
    value: "cooperative",
    layout: "key-major-contiguous-depth",
    buffering: "single",
  }, `${path}.staging`);
  const onlineSoftmax = closedObject(
    field(object, "onlineSoftmax", path),
    [
      "state", "tileScores", "tileMaximum", "tileReductionOrder", "update",
      "priorRescale", "currentWeight", "finalize",
    ],
    `${path}.onlineSoftmax`,
  );
  requireExactObject(onlineSoftmax, {
    state: "running-maximum-denominator-and-weighted-value",
    tileScores: "scaled-query-key-dot-products",
    tileMaximum: "maximum-over-valid-tile-scores",
    tileReductionOrder: "increasing-key-index",
    update: "rescale-prior-state-then-accumulate-current-tile",
    priorRescale: "exp-previous-maximum-minus-new-maximum",
    currentWeight: "exp-score-minus-new-maximum",
    finalize: "divide-weighted-value-by-denominator-after-all-key-tiles",
  }, `${path}.onlineSoftmax`);
  const participation = closedObject(
    field(object, "participation", path),
    ["workgroup", "boundaryQueryLanes", "earlyExit"],
    `${path}.participation`,
  );
  requireExactObject(participation, {
    workgroup: "all-invocations",
    boundaryQueryLanes: "participate",
    earlyExit: "forbidden",
  }, `${path}.participation`);
  const uniformity = closedObject(
    field(object, "uniformity", path),
    ["barrierControl", "activeMaskScope"],
    `${path}.uniformity`,
  );
  requireExactObject(uniformity, {
    barrierControl: "workgroup-uniform",
    activeMaskScope: "memory-effects-and-online-state-only",
  }, `${path}.uniformity`);
  const vectorization = closedObject(
    field(object, "vectorization", path),
    ["keyLoad", "valueLoad", "destinationStore"],
    `${path}.vectorization`,
  );
  const keyLoad = positiveWire(field(vectorization, "keyLoad", `${path}.vectorization`), `${path}.vectorization.keyLoad`);
  const valueLoad = positiveWire(field(vectorization, "valueLoad", `${path}.vectorization`), `${path}.vectorization.valueLoad`);
  const destinationStore = positiveWire(
    field(vectorization, "destinationStore", `${path}.vectorization`),
    `${path}.vectorization.destinationStore`,
  );
  const barriers = closedObject(
    field(object, "barriers", path),
    ["afterCooperativeLoad", "beforeStagingReuse"],
    `${path}.barriers`,
  );
  const afterCooperativeLoad = parseBarrier(
    field(barriers, "afterCooperativeLoad", `${path}.barriers`),
    `${path}.barriers.afterCooperativeLoad`,
  );
  const beforeStagingReuse = parseBarrier(
    field(barriers, "beforeStagingReuse", `${path}.barriers`),
    `${path}.barriers.beforeStagingReuse`,
  );
  const masks = closedObject(
    field(object, "masks", path),
    ["queryLane", "keyLoad", "valueLoad", "invalidKeyScore", "logicalMask", "destinationStore"],
    `${path}.masks`,
  );
  requireExactObject(masks, {
    queryLane: "suppress-logical-state-and-store",
    keyLoad: "zero-fill",
    valueLoad: "zero-fill",
    invalidKeyScore: "exclude-before-online-state-update",
    logicalMask: "exclude-before-online-state-update",
    destinationStore: "suppress",
  }, `${path}.masks`);

  return {
    kind: "attention-online-kv-tile-schedule",
    version: { major: 1, minor: 0 },
    physicalTile,
    workgroup: {
      size: workgroupSize,
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
    vectorization: { keyLoad, valueLoad, destinationStore },
    barriers: { afterCooperativeLoad, beforeStagingReuse },
    masks: {
      queryLane: "suppress-logical-state-and-store",
      keyLoad: "zero-fill",
      valueLoad: "zero-fill",
      invalidKeyScore: "exclude-before-online-state-update",
      logicalMask: "exclude-before-online-state-update",
      destinationStore: "suppress",
    },
  };
}

function verifyCompatibility(schedule: AttentionOnlineKvTileSchedule): void {
  for (const name of ["queryRows", "keyRows"] as const) {
    if (wireIntegerToBigInt(schedule.physicalTile[name])
      > INITIAL_ATTENTION_ONLINE_KV_MAX_TILE_ROWS) {
      invalid(
        SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
        `$.payload.schedule.physicalTile.${name}`,
        `attention tile rows must not exceed ${INITIAL_ATTENTION_ONLINE_KV_MAX_TILE_ROWS}`,
      );
    }
  }
  if (schedule.workgroup.size.x !== schedule.physicalTile.queryRows
    || schedule.workgroup.size.y !== "1"
    || schedule.workgroup.size.z !== "1") {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.schedule.workgroup.size",
      "one-query-row-per-invocation schedule requires queryRows by 1 by 1",
    );
  }
  if (schedule.vectorization.keyLoad !== "1"
    || schedule.vectorization.valueLoad !== "1"
    || schedule.vectorization.destinationStore !== "1") {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.schedule.vectorization",
      "attention online K/V tile schedule v1 supports scalar memory vectors only",
    );
  }
}

function parsePhysicalTile(
  value: JsonValue,
  path: string,
): AttentionOnlineKvTileSchedule["physicalTile"] {
  const object = closedObject(value, ["queryRows", "keyRows"], path);
  return {
    queryRows: positiveWire(field(object, "queryRows", path), `${path}.queryRows`),
    keyRows: positiveWire(field(object, "keyRows", path), `${path}.keyRows`),
  };
}

function parseExtentXyz(
  value: JsonValue,
  path: string,
): AttentionOnlineKvTileSchedule["workgroup"]["size"] {
  const object = closedObject(value, ["x", "y", "z"], path);
  return {
    x: positiveWire(field(object, "x", path), `${path}.x`),
    y: positiveWire(field(object, "y", path), `${path}.y`),
    z: positiveWire(field(object, "z", path), `${path}.z`),
  };
}

function parseBarrier(value: JsonValue, path: string): AttentionOnlineKvWorkgroupBarrier {
  const object = closedObject(value, ["scope", "memory", "semantics"], path);
  requireExactObject(object, {
    scope: "workgroup",
    memory: "workgroup",
    semantics: "acquire-release",
  }, path);
  return { scope: "workgroup", memory: "workgroup", semantics: "acquire-release" };
}

function positiveWire(value: JsonValue, path: string) {
  const parsed = parseWireU64(value, path);
  if (wireIntegerToBigInt(parsed) === 0n) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
      path,
      "attention schedule extents and widths must be positive",
    );
  }
  return parsed;
}

function requireExactObject(
  object: JsonObject,
  expected: Readonly<Record<string, string>>,
  path: string,
): void {
  for (const [name, literal] of Object.entries(expected)) {
    if (object[name] !== literal) {
      invalid(
        SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.${name}`,
        `attention online K/V tile schedule v1 requires ${name}=${literal}`,
      );
    }
  }
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, path, "expected object");
  const allowed = new Set(fields);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.unknownField,
      path,
      `unknown fields: ${unknown.sort().join(", ")}`,
    );
  }
  for (const name of fields) {
    if (value[name] === undefined) {
      invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
    }
  }
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) {
    invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  }
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(SCHEDULE_DIAGNOSTIC_CODES.invalidArtifact, path, "expected string");
  return value;
}

function invalid(code: `BG-SCHEDULE-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

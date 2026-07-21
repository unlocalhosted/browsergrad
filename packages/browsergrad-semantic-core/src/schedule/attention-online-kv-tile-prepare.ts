import type { VerifiedAttentionForwardArtifact } from "../kernel/attention-forward-artifact.js";
import {
  requirePreparedAttentionForwardSpecialization,
  type PreparedAttentionForwardSpecialization,
} from "../kernel/attention-forward-prepare.js";
import { SCHEDULE_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { hashNamedComponents, hashSemanticArtifact } from "../schema/hash.js";
import { encodeWireU64 } from "../schema/integers.js";
import type { DecodeLimits } from "../schema/limits.js";
import {
  attentionOnlineKvTileScheduleArtifactPayload,
  type VerifiedAttentionOnlineKvTileScheduleArtifact,
} from "./attention-online-kv-tile-artifact.js";
import type { AttentionOnlineKvTileSchedule } from "./attention-online-kv-tile-model.js";

const DEFAULT_MAX_WORKGROUP_INVOCATIONS = 256;
const MAX_CONFIGURABLE_WORKGROUP_INVOCATIONS = 1_024;
const DEFAULT_MAX_STAGING_BYTES = 65_536;
const MAX_CONFIGURABLE_STAGING_BYTES = 16_777_216;
const DEFAULT_MAX_PRIVATE_ELEMENTS = 512;
const MAX_CONFIGURABLE_PRIVATE_ELEMENTS = 16_384;
const DEFAULT_MAX_DISPATCH_WORKGROUPS = 16_777_216;
const MAX_CONFIGURABLE_DISPATCH_WORKGROUPS = 1_073_741_824;
const DEFAULT_MAX_KEY_TILES = 1_048_576;
const MAX_CONFIGURABLE_KEY_TILES = 16_777_216;
const PREPARED_ATTENTION_ONLINE_KV_TILE_SCHEDULES = new WeakSet<object>();

export interface PrepareAttentionOnlineKvTileScheduleRequest {
  readonly evaluationLimits?: Partial<DecodeLimits>;
  readonly maxWorkgroupInvocations?: number;
  readonly maxStagingBytes?: number;
  readonly maxPrivateElementsPerInvocation?: number;
  readonly maxDispatchWorkgroups?: number;
  readonly maxKeyTiles?: number;
}

export interface PreparedAttentionOnlineKvTileSchedule {
  readonly logical: PreparedAttentionForwardSpecialization;
  readonly schedule: AttentionOnlineKvTileSchedule;
  readonly attentionForwardSemanticHash: string;
  readonly scheduleSemanticHash: string;
  readonly scheduleSpecializationHash: string;
  readonly queryRows: bigint;
  readonly keyRows: bigint;
  readonly workgroupSizeX: bigint;
  readonly workgroupSizeY: bigint;
  readonly workgroupSizeZ: bigint;
  readonly workgroupInvocations: bigint;
  readonly keyStagingElements: bigint;
  readonly valueStagingElements: bigint;
  readonly aggregateStagingElements: bigint;
  readonly aggregateStagingBytes: bigint;
  readonly queryPrivateElements: bigint;
  readonly outputPrivateElements: bigint;
  readonly privateElementsPerInvocation: bigint;
  readonly keyTiles: bigint;
  readonly dispatchX: bigint;
  readonly dispatchY: bigint;
  readonly dispatchZ: bigint;
  readonly dispatchWorkgroups: bigint;
}

/**
 * Composes one authorized logical attention specialization with one exact
 * physical online K/V-tile schedule. It derives only backend-neutral staging,
 * private-state, traversal, and launch geometry.
 */
export async function prepareAttentionOnlineKvTileSchedule(
  logical: PreparedAttentionForwardSpecialization,
  logicalArtifact: VerifiedAttentionForwardArtifact,
  scheduleArtifact: VerifiedAttentionOnlineKvTileScheduleArtifact,
  request: PrepareAttentionOnlineKvTileScheduleRequest = {},
): Promise<PreparedAttentionOnlineKvTileSchedule> {
  requirePreparedAttentionForwardSpecialization(logical);
  const hashOptions = request.evaluationLimits === undefined
    ? {}
    : { limits: request.evaluationLimits };
  const attentionForwardSemanticHash = await hashSemanticArtifact(logicalArtifact, hashOptions);
  if (logical.kernelSemanticHash !== attentionForwardSemanticHash) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch,
      "$.logical",
      "prepared attention specialization does not belong to the supplied logical artifact",
    );
  }
  const payload = attentionOnlineKvTileScheduleArtifactPayload(scheduleArtifact);
  if (payload.attentionForwardSemanticHash !== attentionForwardSemanticHash) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch,
      "$.schedule.attentionForwardSemanticHash",
      "verified schedule does not belong to the supplied attention-forward artifact",
    );
  }

  const queryRows = BigInt(payload.schedule.physicalTile.queryRows);
  const keyRows = BigInt(payload.schedule.physicalTile.keyRows);
  const workgroupSizeX = BigInt(payload.schedule.workgroup.size.x);
  const workgroupSizeY = BigInt(payload.schedule.workgroup.size.y);
  const workgroupSizeZ = BigInt(payload.schedule.workgroup.size.z);
  const workgroupInvocations = workgroupSizeX * workgroupSizeY * workgroupSizeZ;
  bounded(
    workgroupInvocations,
    budget(
      request.maxWorkgroupInvocations,
      DEFAULT_MAX_WORKGROUP_INVOCATIONS,
      MAX_CONFIGURABLE_WORKGROUP_INVOCATIONS,
      "maxWorkgroupInvocations",
    ),
    "$.maxWorkgroupInvocations",
    "attention schedule workgroup invocations",
  );

  const keyStagingElements = keyRows * logical.queryDepth;
  const valueStagingElements = keyRows * logical.valueDepth;
  const aggregateStagingElements = keyStagingElements + valueStagingElements;
  const aggregateStagingBytes = aggregateStagingElements * 4n;
  bounded(
    aggregateStagingBytes,
    budget(
      request.maxStagingBytes,
      DEFAULT_MAX_STAGING_BYTES,
      MAX_CONFIGURABLE_STAGING_BYTES,
      "maxStagingBytes",
    ),
    "$.maxStagingBytes",
    "attention schedule workgroup staging bytes",
  );

  const queryPrivateElements = logical.queryDepth;
  const outputPrivateElements = logical.valueDepth;
  const privateElementsPerInvocation = queryPrivateElements + outputPrivateElements;
  bounded(
    privateElementsPerInvocation,
    budget(
      request.maxPrivateElementsPerInvocation,
      DEFAULT_MAX_PRIVATE_ELEMENTS,
      MAX_CONFIGURABLE_PRIVATE_ELEMENTS,
      "maxPrivateElementsPerInvocation",
    ),
    "$.maxPrivateElementsPerInvocation",
    "attention schedule private elements per invocation",
  );

  const keyTiles = ceilDiv(logical.keyLength, keyRows);
  bounded(
    keyTiles,
    budget(
      request.maxKeyTiles,
      DEFAULT_MAX_KEY_TILES,
      MAX_CONFIGURABLE_KEY_TILES,
      "maxKeyTiles",
    ),
    "$.maxKeyTiles",
    "attention schedule key tiles",
  );
  const dispatchX = ceilDiv(logical.queryLength, queryRows);
  const dispatchY = logical.heads;
  const dispatchZ = logical.batch;
  const dispatchWorkgroups = dispatchX * dispatchY * dispatchZ;
  bounded(
    dispatchWorkgroups,
    budget(
      request.maxDispatchWorkgroups,
      DEFAULT_MAX_DISPATCH_WORKGROUPS,
      MAX_CONFIGURABLE_DISPATCH_WORKGROUPS,
      "maxDispatchWorkgroups",
    ),
    "$.maxDispatchWorkgroups",
    "attention schedule dispatch workgroups",
  );

  const scheduleSemanticHash = await hashSemanticArtifact(scheduleArtifact, hashOptions);
  const scheduleSpecializationHash = await hashNamedComponents({
    profile: "browsergrad.attention-forward.schedule.online-kv-tile-scalar@1",
    logicalSpecialization: logical.specializationHash,
    attentionForward: attentionForwardSemanticHash,
    schedule: scheduleSemanticHash,
    resolved: {
      physicalTile: {
        queryRows: encodeWireU64(queryRows),
        keyRows: encodeWireU64(keyRows),
      },
      workgroupSize: {
        x: encodeWireU64(workgroupSizeX),
        y: encodeWireU64(workgroupSizeY),
        z: encodeWireU64(workgroupSizeZ),
      },
      staging: {
        keyElements: encodeWireU64(keyStagingElements),
        valueElements: encodeWireU64(valueStagingElements),
        aggregateBytes: encodeWireU64(aggregateStagingBytes),
      },
      privateElements: {
        query: encodeWireU64(queryPrivateElements),
        output: encodeWireU64(outputPrivateElements),
      },
      keyTiles: encodeWireU64(keyTiles),
      dispatch: {
        x: encodeWireU64(dispatchX),
        y: encodeWireU64(dispatchY),
        z: encodeWireU64(dispatchZ),
      },
    },
  }, hashOptions);
  const prepared = Object.freeze({
    logical,
    schedule: payload.schedule,
    attentionForwardSemanticHash,
    scheduleSemanticHash,
    scheduleSpecializationHash,
    queryRows,
    keyRows,
    workgroupSizeX,
    workgroupSizeY,
    workgroupSizeZ,
    workgroupInvocations,
    keyStagingElements,
    valueStagingElements,
    aggregateStagingElements,
    aggregateStagingBytes,
    queryPrivateElements,
    outputPrivateElements,
    privateElementsPerInvocation,
    keyTiles,
    dispatchX,
    dispatchY,
    dispatchZ,
    dispatchWorkgroups,
  });
  PREPARED_ATTENTION_ONLINE_KV_TILE_SCHEDULES.add(prepared);
  return prepared;
}

/** @internal Authority check for backend preparation in the same module graph. */
export function requirePreparedAttentionOnlineKvTileSchedule(
  prepared: PreparedAttentionOnlineKvTileSchedule,
): void {
  if (!PREPARED_ATTENTION_ONLINE_KV_TILE_SCHEDULES.has(prepared as object)) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.schedule",
      "backend preparation requires an exact attention schedule specialization from this module instance",
    );
  }
}

function ceilDiv(value: bigint, divisor: bigint): bigint {
  return (value + divisor - 1n) / divisor;
}

function budget(value: number | undefined, fallback: number, maximum: number, name: string): bigint {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > maximum) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.resourceLimit,
      `$.${name}`,
      `${name} must be a positive safe integer no greater than ${maximum}`,
    );
  }
  return BigInt(resolved);
}

function bounded(value: bigint, maximum: bigint, path: string, label: string): void {
  if (value > maximum) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.resourceLimit,
      path,
      `${label} require ${value}; limit is ${maximum}`,
    );
  }
}

function invalid(code: `BG-SCHEDULE-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

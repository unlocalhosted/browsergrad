import type { VerifiedLogicalGemmTileArtifact } from "../kernel/gemm-tile-artifact.js";
import {
  requirePreparedLogicalGemmTileSpecialization,
  type PreparedLogicalGemmTileSpecialization,
} from "../kernel/gemm-tile-prepare.js";
import { SCHEDULE_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { hashNamedComponents, hashSemanticArtifact } from "../schema/hash.js";
import { encodeWireU64 } from "../schema/integers.js";
import type { DecodeLimits } from "../schema/limits.js";
import {
  logicalGemmTileScheduleArtifactPayload,
  type VerifiedLogicalGemmTileScheduleArtifact,
} from "./gemm-tile-artifact.js";
import type { LogicalGemmTileSchedule } from "./gemm-tile-model.js";

const DEFAULT_MAX_WORKGROUP_INVOCATIONS = 1_024;
const MAX_CONFIGURABLE_WORKGROUP_INVOCATIONS = 65_536;
const DEFAULT_MAX_STAGING_ELEMENTS = 65_536;
const MAX_CONFIGURABLE_STAGING_ELEMENTS = 16_777_216;
const DEFAULT_MAX_DISPATCH_WORKGROUPS = 16_777_216;
const MAX_CONFIGURABLE_DISPATCH_WORKGROUPS = 1_073_741_824;
const PREPARED_LOGICAL_GEMM_TILE_SCHEDULES = new WeakSet<object>();

export interface PrepareLogicalGemmTileScheduleRequest {
  readonly evaluationLimits?: Partial<DecodeLimits>;
  readonly maxWorkgroupInvocations?: number;
  readonly maxStagingElements?: number;
  readonly maxDispatchWorkgroups?: number;
}

export interface PreparedLogicalGemmTileSchedule {
  readonly logical: PreparedLogicalGemmTileSpecialization;
  readonly schedule: LogicalGemmTileSchedule;
  readonly logicalGemmSemanticHash: string;
  readonly scheduleSemanticHash: string;
  readonly scheduleSpecializationHash: string;
  readonly physicalM: bigint;
  readonly physicalN: bigint;
  readonly physicalK: bigint;
  readonly workgroupSizeX: bigint;
  readonly workgroupSizeY: bigint;
  readonly workgroupSizeZ: bigint;
  readonly workgroupInvocations: bigint;
  readonly lhsStagingElements: bigint;
  readonly rhsStagingElements: bigint;
  readonly aggregateStagingElements: bigint;
  readonly dispatchX: bigint;
  readonly dispatchY: bigint;
  readonly dispatchZ: bigint;
  readonly dispatchWorkgroups: bigint;
}

/**
 * Composes one authorized logical GEMM specialization with one exact verified
 * physical schedule. It derives only backend-neutral launch and staging
 * geometry; numerical preservation and device legality remain separate.
 */
export async function prepareLogicalGemmTileSchedule(
  logical: PreparedLogicalGemmTileSpecialization,
  logicalArtifact: VerifiedLogicalGemmTileArtifact,
  scheduleArtifact: VerifiedLogicalGemmTileScheduleArtifact,
  request: PrepareLogicalGemmTileScheduleRequest = {},
): Promise<PreparedLogicalGemmTileSchedule> {
  requirePreparedLogicalGemmTileSpecialization(logical);
  const hashOptions = request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits };
  const logicalGemmSemanticHash = await hashSemanticArtifact(logicalArtifact, hashOptions);
  if (logical.kernelSemanticHash !== logicalGemmSemanticHash) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch,
      "$.logical",
      "prepared logical GEMM specialization does not belong to the supplied verified logical artifact",
    );
  }
  const payload = logicalGemmTileScheduleArtifactPayload(scheduleArtifact);
  if (payload.logicalGemmSemanticHash !== logicalGemmSemanticHash) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.kernelHashMismatch,
      "$.schedule.logicalGemmSemanticHash",
      "verified schedule does not belong to the supplied logical GEMM artifact",
    );
  }

  const physicalM = BigInt(payload.schedule.physicalTile.m);
  const physicalN = BigInt(payload.schedule.physicalTile.n);
  const physicalK = BigInt(payload.schedule.physicalTile.k);
  const workgroupSizeX = BigInt(payload.schedule.workgroup.size.x);
  const workgroupSizeY = BigInt(payload.schedule.workgroup.size.y);
  const workgroupSizeZ = BigInt(payload.schedule.workgroup.size.z);
  const workgroupInvocations = workgroupSizeX * workgroupSizeY * workgroupSizeZ;
  const maxWorkgroupInvocations = budget(
    request.maxWorkgroupInvocations,
    DEFAULT_MAX_WORKGROUP_INVOCATIONS,
    MAX_CONFIGURABLE_WORKGROUP_INVOCATIONS,
    "maxWorkgroupInvocations",
  );
  bounded(
    workgroupInvocations,
    maxWorkgroupInvocations,
    "$.maxWorkgroupInvocations",
    "schedule workgroup invocations",
  );

  const lhsStagingElements = physicalM * physicalK;
  const rhsStagingElements = physicalK * physicalN;
  const aggregateStagingElements = lhsStagingElements + rhsStagingElements;
  const maxStagingElements = budget(
    request.maxStagingElements,
    DEFAULT_MAX_STAGING_ELEMENTS,
    MAX_CONFIGURABLE_STAGING_ELEMENTS,
    "maxStagingElements",
  );
  bounded(
    aggregateStagingElements,
    maxStagingElements,
    "$.maxStagingElements",
    "schedule staging elements",
  );

  const dispatchX = ceilDiv(logical.n, physicalN);
  const dispatchY = ceilDiv(logical.m, physicalM);
  const dispatchZ = 1n;
  const dispatchWorkgroups = dispatchX * dispatchY;
  const maxDispatchWorkgroups = budget(
    request.maxDispatchWorkgroups,
    DEFAULT_MAX_DISPATCH_WORKGROUPS,
    MAX_CONFIGURABLE_DISPATCH_WORKGROUPS,
    "maxDispatchWorkgroups",
  );
  bounded(
    dispatchWorkgroups,
    maxDispatchWorkgroups,
    "$.maxDispatchWorkgroups",
    "schedule dispatch workgroups",
  );

  const scheduleSemanticHash = await hashSemanticArtifact(scheduleArtifact, hashOptions);
  const scheduleSpecializationHash = await hashNamedComponents({
    profile: "browsergrad.logical-gemm-tile.schedule.scalar-cooperative@1",
    logicalSpecialization: logical.specializationHash,
    logicalGemm: logicalGemmSemanticHash,
    schedule: scheduleSemanticHash,
    resolved: {
      physicalTile: {
        m: encodeWireU64(physicalM),
        n: encodeWireU64(physicalN),
        k: encodeWireU64(physicalK),
      },
      workgroupSize: {
        x: encodeWireU64(workgroupSizeX),
        y: encodeWireU64(workgroupSizeY),
        z: encodeWireU64(workgroupSizeZ),
      },
      stagingElements: {
        lhs: encodeWireU64(lhsStagingElements),
        rhs: encodeWireU64(rhsStagingElements),
      },
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
    logicalGemmSemanticHash,
    scheduleSemanticHash,
    scheduleSpecializationHash,
    physicalM,
    physicalN,
    physicalK,
    workgroupSizeX,
    workgroupSizeY,
    workgroupSizeZ,
    workgroupInvocations,
    lhsStagingElements,
    rhsStagingElements,
    aggregateStagingElements,
    dispatchX,
    dispatchY,
    dispatchZ,
    dispatchWorkgroups,
  });
  PREPARED_LOGICAL_GEMM_TILE_SCHEDULES.add(prepared);
  return prepared;
}

/** @internal Authority check for backend preparation in the same module graph. */
export function requirePreparedLogicalGemmTileSchedule(
  prepared: PreparedLogicalGemmTileSchedule,
): void {
  if (!PREPARED_LOGICAL_GEMM_TILE_SCHEDULES.has(prepared as object)) {
    invalid(
      SCHEDULE_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.schedule",
      "backend preparation requires an exact schedule specialization produced by this semantic-core module instance",
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
    invalid(SCHEDULE_DIAGNOSTIC_CODES.resourceLimit, path, `${label} require ${value}; limit is ${maximum}`);
  }
}

function invalid(code: `BG-SCHEDULE-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { prepareViewAccessor, type PreparedViewAccessor } from "../layout/prepare.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { hashNamedComponents, hashSemanticArtifact } from "../schema/hash.js";
import { encodeWireU64, type WireI64 } from "../schema/integers.js";
import type { JsonObject } from "../schema/json.js";
import type { DecodeLimits } from "../schema/limits.js";
import {
  attentionForwardArtifactPayload,
  type VerifiedAttentionForwardArtifact,
} from "./attention-forward-artifact.js";
import type { AttentionForwardOperation } from "./attention-forward-model.js";
import {
  ensureKernelPreparationActive,
  kernelMonotonicNow,
  normalizeKernelBindings,
  proveDenseRowMajorAccessor,
  resolveKernelBudget,
  type KernelPreparationControl,
} from "./preparation.js";

const DEFAULT_MAX_ELEMENTS = 1_000_000;
const MAX_CONFIGURABLE_ELEMENTS = 16_777_216;
const DEFAULT_MAX_ALLOCATION_BYTES = 67_108_864;
const MAX_CONFIGURABLE_ALLOCATION_BYTES = 1_073_741_824;
const DEFAULT_MAX_SCALAR_OPERATIONS = 25_000_000;
const MAX_CONFIGURABLE_SCALAR_OPERATIONS = 250_000_000;
const DEFAULT_MAX_EVALUATION_STEPS = 25_000_000;
const MAX_CONFIGURABLE_EVALUATION_STEPS = 250_000_000;
const DEFAULT_MAX_PREPARATION_MS = 5_000;
const MAX_CONFIGURABLE_PREPARATION_MS = 60_000;
const PREPARED_ATTENTION_FORWARD_SPECIALIZATIONS = new WeakSet<object>();

export interface PrepareAttentionForwardSpecializationRequest {
  readonly operationId: string;
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly evaluationLimits?: Partial<DecodeLimits>;
  readonly maxElements?: number;
  readonly maxAllocationBytes?: number;
  readonly maxScalarOperations?: number;
  readonly maxEvaluationSteps?: number;
  readonly maxPreparationMs?: number;
  readonly signal?: AbortSignal;
}

export interface PreparedAttentionForwardSpecialization {
  readonly operation: AttentionForwardOperation;
  readonly bindings: Readonly<Record<string, WireI64>>;
  readonly query: PreparedViewAccessor;
  readonly key: PreparedViewAccessor;
  readonly value: PreparedViewAccessor;
  readonly destination: PreparedViewAccessor;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly specializationHash: string;
  readonly batch: bigint;
  readonly heads: bigint;
  readonly queryLength: bigint;
  readonly keyLength: bigint;
  readonly queryDepth: bigint;
  readonly valueDepth: bigint;
  readonly queryElements: bigint;
  readonly keyElements: bigint;
  readonly valueElements: bigint;
  readonly outputElements: bigint;
  readonly aggregateAllocationBytes: bigint;
  readonly scoreMultiplyAdds: bigint;
  readonly weightedValueMultiplyAdds: bigint;
  readonly maximumScoreElements: bigint;
  readonly scalarOperations: bigint;
}

/**
 * Resolves the initial dense attention-forward profile without selecting a
 * physical schedule. Address proof, resource bounds, and hashes are shared by
 * the CPU reference and later schedule/backend specializations.
 */
export async function prepareAttentionForwardSpecialization(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedAttentionForwardArtifact,
  request: PrepareAttentionForwardSpecializationRequest,
): Promise<PreparedAttentionForwardSpecialization> {
  const startedAt = kernelMonotonicNow();
  const kernel = attentionForwardArtifactPayload(kernelArtifact);
  const hashOptions = request.evaluationLimits === undefined
    ? {}
    : { limits: request.evaluationLimits };
  const layoutSemanticHash = await hashSemanticArtifact(layoutArtifact, hashOptions);
  if (kernel.layoutSemanticHash !== layoutSemanticHash) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch,
      "$.layout",
      "attention-forward specialization received a different layout artifact",
    );
  }
  if (kernel.operation.operationId !== request.operationId) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.danglingReference,
      "$.operationId",
      `unknown verified attention-forward operation ${request.operationId}`,
    );
  }
  const bindings = normalizeKernelBindings(request.bindings ?? {});
  const accessorRequest = {
    bindings,
    ...(request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits }),
  };
  const query = prepareViewAccessor(layoutArtifact, {
    viewId: kernel.operation.query.viewId,
    ...accessorRequest,
  });
  const key = prepareViewAccessor(layoutArtifact, {
    viewId: kernel.operation.key.viewId,
    ...accessorRequest,
  });
  const value = prepareViewAccessor(layoutArtifact, {
    viewId: kernel.operation.value.viewId,
    ...accessorRequest,
  });
  const destination = prepareViewAccessor(layoutArtifact, {
    viewId: kernel.operation.destination.viewId,
    ...accessorRequest,
  });
  ensurePreparedContract(query, key, value, destination);

  const [batch, heads, queryLength, queryDepth] = query.logicalShape as readonly [
    bigint, bigint, bigint, bigint,
  ];
  const keyLength = key.logicalShape[2] as bigint;
  const valueDepth = value.logicalShape[3] as bigint;
  const queryElements = batch * heads * queryLength * queryDepth;
  const keyElements = batch * heads * keyLength * queryDepth;
  const valueElements = batch * heads * keyLength * valueDepth;
  const outputElements = batch * heads * queryLength * valueDepth;
  const aggregateElements = queryElements + keyElements + valueElements + outputElements;
  const maxElements = resolveKernelBudget(
    request.maxElements,
    DEFAULT_MAX_ELEMENTS,
    MAX_CONFIGURABLE_ELEMENTS,
    "maxElements",
  );
  bounded(aggregateElements, maxElements, "$.maxElements", "attention aggregate elements");
  const aggregateAllocationBytes = query.allocationByteLength
    + key.allocationByteLength
    + value.allocationByteLength
    + destination.allocationByteLength;
  const maxAllocationBytes = resolveKernelBudget(
    request.maxAllocationBytes,
    DEFAULT_MAX_ALLOCATION_BYTES,
    MAX_CONFIGURABLE_ALLOCATION_BYTES,
    "maxAllocationBytes",
  );
  bounded(
    aggregateAllocationBytes,
    maxAllocationBytes,
    "$.maxAllocationBytes",
    "attention aggregate allocation bytes",
  );

  const maximumScoreElements = batch * heads * queryLength * keyLength;
  const scoreMultiplyAdds = maximumScoreElements * queryDepth;
  const weightedValueMultiplyAdds = maximumScoreElements * valueDepth;
  const scalarOperations = (scoreMultiplyAdds * 2n)
    + (weightedValueMultiplyAdds * 2n)
    + (maximumScoreElements * 8n)
    + outputElements;
  const maxScalarOperations = resolveKernelBudget(
    request.maxScalarOperations,
    DEFAULT_MAX_SCALAR_OPERATIONS,
    MAX_CONFIGURABLE_SCALAR_OPERATIONS,
    "maxScalarOperations",
  );
  bounded(
    scalarOperations,
    maxScalarOperations,
    "$.maxScalarOperations",
    "attention scalar operations",
  );

  const evaluationSteps = (queryElements * BigInt(query.evaluationStepsPerAccess))
    + (keyElements * BigInt(key.evaluationStepsPerAccess))
    + (valueElements * BigInt(value.evaluationStepsPerAccess))
    + (outputElements * BigInt(destination.evaluationStepsPerAccess));
  const maxEvaluationSteps = resolveKernelBudget(
    request.maxEvaluationSteps,
    DEFAULT_MAX_EVALUATION_STEPS,
    MAX_CONFIGURABLE_EVALUATION_STEPS,
    "maxEvaluationSteps",
  );
  bounded(
    evaluationSteps,
    maxEvaluationSteps,
    "$.maxEvaluationSteps",
    "attention address-proof evaluation steps",
  );
  const maxPreparationMs = resolveKernelBudget(
    request.maxPreparationMs,
    DEFAULT_MAX_PREPARATION_MS,
    MAX_CONFIGURABLE_PREPARATION_MS,
    "maxPreparationMs",
  );
  const control: KernelPreparationControl = {
    startedAt,
    maxPreparationMs,
    ...(request.signal === undefined ? {} : { signal: request.signal }),
  };
  ensureKernelPreparationActive(control);
  await proveDenseRowMajorAccessor(query, query.logicalShape, control, "query");
  await proveDenseRowMajorAccessor(key, key.logicalShape, control, "key");
  await proveDenseRowMajorAccessor(value, value.logicalShape, control, "value");
  await proveDenseRowMajorAccessor(destination, destination.logicalShape, control, "destination");

  const kernelSemanticHash = await hashSemanticArtifact(kernelArtifact, hashOptions);
  ensureKernelPreparationActive(control);
  const specializationHash = await hashNamedComponents({
    profile: "browsergrad.attention-forward.dense-rank4-f32@1",
    layout: layoutSemanticHash,
    kernel: kernelSemanticHash,
    operation: kernel.operation.operationId,
    bindings: bindings as unknown as JsonObject,
    resolved: {
      batch: encodeWireU64(batch),
      heads: encodeWireU64(heads),
      queryLength: encodeWireU64(queryLength),
      keyLength: encodeWireU64(keyLength),
      queryDepth: encodeWireU64(queryDepth),
      valueDepth: encodeWireU64(valueDepth),
      queryByteOffset: encodeWireU64(query.viewByteOffset),
      keyByteOffset: encodeWireU64(key.viewByteOffset),
      valueByteOffset: encodeWireU64(value.viewByteOffset),
      destinationByteOffset: encodeWireU64(destination.viewByteOffset),
      queryAllocationBytes: encodeWireU64(query.allocationByteLength),
      keyAllocationBytes: encodeWireU64(key.allocationByteLength),
      valueAllocationBytes: encodeWireU64(value.allocationByteLength),
      destinationAllocationBytes: encodeWireU64(destination.allocationByteLength),
    },
  }, hashOptions);
  ensureKernelPreparationActive(control);
  const prepared = Object.freeze({
    operation: kernel.operation,
    bindings,
    query,
    key,
    value,
    destination,
    layoutSemanticHash,
    kernelSemanticHash,
    specializationHash,
    batch,
    heads,
    queryLength,
    keyLength,
    queryDepth,
    valueDepth,
    queryElements,
    keyElements,
    valueElements,
    outputElements,
    aggregateAllocationBytes,
    scoreMultiplyAdds,
    weightedValueMultiplyAdds,
    maximumScoreElements,
    scalarOperations,
  });
  PREPARED_ATTENTION_FORWARD_SPECIALIZATIONS.add(prepared);
  return prepared;
}

/** @internal Shared authority check for CPU and schedule composition. */
export function requirePreparedAttentionForwardSpecialization(
  prepared: PreparedAttentionForwardSpecialization,
): void {
  if (!PREPARED_ATTENTION_FORWARD_SPECIALIZATIONS.has(prepared as object)) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidBinding,
      "$.logical",
      "attention execution requires an exact specialization produced by this module instance",
    );
  }
}

function ensurePreparedContract(
  query: PreparedViewAccessor,
  key: PreparedViewAccessor,
  value: PreparedViewAccessor,
  destination: PreparedViewAccessor,
): void {
  const roles = [["query", query], ["key", key], ["value", value], ["destination", destination]] as const;
  for (const [role, accessor] of roles) {
    if (accessor.dtype !== "f32" || accessor.dtypeBytes !== 4 || accessor.logicalShape.length !== 4) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
        `$.${role}`,
        "attention-forward specialization requires rank-4 f32 views",
      );
    }
    if (accessor.memorySpace.kind !== "global") {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
        `$.${role}`,
        "attention-forward specialization requires global-memory views",
      );
    }
    if (!accessor.fullySpecialized) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.invalidBinding,
        `$.${role}`,
        "attention-forward requires every dimension binding",
      );
    }
    if (accessor.allocationByteLength > BigInt(Number.MAX_SAFE_INTEGER)) {
      invalid(
        KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
        `$.${role}`,
        "CPU attention allocation lengths must fit exact JavaScript indexes",
      );
    }
  }
  if (query.logicalShape[0] !== key.logicalShape[0]
    || query.logicalShape[0] !== value.logicalShape[0]
    || query.logicalShape[0] !== destination.logicalShape[0]
    || query.logicalShape[1] !== key.logicalShape[1]
    || query.logicalShape[1] !== value.logicalShape[1]
    || query.logicalShape[1] !== destination.logicalShape[1]
    || query.logicalShape[3] !== key.logicalShape[3]
    || key.logicalShape[2] !== value.logicalShape[2]
    || query.logicalShape[2] !== destination.logicalShape[2]
    || value.logicalShape[3] !== destination.logicalShape[3]) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.shapeMismatch,
      "$.operation",
      "resolved Q/K/V/output attention dimensions must match",
    );
  }
  const allocationIds = roles.map(([, accessor]) => accessor.allocationId);
  const aliasIds = roles.map(([, accessor]) => accessor.aliasSetId);
  if (new Set(allocationIds).size !== 4 || new Set(aliasIds).size !== 4) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.aliasConflict,
      "$.operation.overlap",
      "attention-forward requires pairwise-disjoint allocations and alias sets",
    );
  }
}

function bounded(value: bigint, maximum: number, path: string, label: string): void {
  if (value > BigInt(maximum)) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.resourceLimit,
      path,
      `${label} require ${value}; limit is ${maximum}`,
    );
  }
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

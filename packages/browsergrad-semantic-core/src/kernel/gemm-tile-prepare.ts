import { prepareViewAccessor, type PreparedViewAccessor } from "../layout/prepare.js";
import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { hashNamedComponents, hashSemanticArtifact } from "../schema/hash.js";
import { encodeWireU64, parseWireI64, type WireI64 } from "../schema/integers.js";
import type { JsonObject } from "../schema/json.js";
import type { DecodeLimits } from "../schema/limits.js";
import {
  logicalGemmTileArtifactPayload,
  type VerifiedLogicalGemmTileArtifact,
} from "./gemm-tile-artifact.js";
import type { LogicalGemmTileOperation } from "./gemm-tile-model.js";

const DEFAULT_MAX_ELEMENTS = 1_000_000;
const MAX_CONFIGURABLE_ELEMENTS = 16_777_216;
const DEFAULT_MAX_MULTIPLY_ADDS = 25_000_000;
const MAX_CONFIGURABLE_MULTIPLY_ADDS = 250_000_000;
const DEFAULT_MAX_EVALUATION_STEPS = 25_000_000;
const MAX_CONFIGURABLE_EVALUATION_STEPS = 250_000_000;
const DEFAULT_MAX_PREPARATION_MS = 5_000;
const MAX_CONFIGURABLE_PREPARATION_MS = 60_000;
const YIELD_INTERVAL_MS = 16;
const PREPARED_LOGICAL_GEMM_TILES = new WeakSet<object>();

export interface PrepareLogicalGemmTileSpecializationRequest {
  readonly operationId: string;
  readonly bindings?: Readonly<Record<string, WireI64>>;
  readonly evaluationLimits?: Partial<DecodeLimits>;
  readonly maxElements?: number;
  readonly maxMultiplyAdds?: number;
  readonly maxEvaluationSteps?: number;
  readonly maxPreparationMs?: number;
  readonly signal?: AbortSignal;
}

export interface PreparedLogicalGemmTileSpecialization {
  readonly operation: LogicalGemmTileOperation;
  readonly bindings: Readonly<Record<string, WireI64>>;
  readonly lhs: PreparedViewAccessor;
  readonly rhs: PreparedViewAccessor;
  readonly destination: PreparedViewAccessor;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly specializationHash: string;
  readonly m: bigint;
  readonly n: bigint;
  readonly k: bigint;
  readonly tileM: bigint;
  readonly tileN: bigint;
  readonly tileK: bigint;
  readonly outputElements: bigint;
  readonly multiplyAdds: bigint;
}

/**
 * Resolves the initial logical GEMM tile without selecting any physical
 * schedule. The proof validates all dense rank-2 f32 addresses once and keeps
 * workgroup, staging, vectorization, backend, and WGSL facts out of the result.
 */
export async function prepareLogicalGemmTileSpecialization(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedLogicalGemmTileArtifact,
  request: PrepareLogicalGemmTileSpecializationRequest,
): Promise<PreparedLogicalGemmTileSpecialization> {
  const preparationStartedAt = monotonicNow();
  const kernel = logicalGemmTileArtifactPayload(kernelArtifact);
  const hashOptions = request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits };
  const layoutSemanticHash = await hashSemanticArtifact(layoutArtifact, hashOptions);
  if (kernel.layoutSemanticHash !== layoutSemanticHash) {
    invalid(KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch, "$.layout", "logical GEMM tile specialization received a different layout artifact");
  }
  if (kernel.operation.operationId !== request.operationId) {
    invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, "$.operationId", `unknown verified logical GEMM tile operation ${request.operationId}`);
  }
  const bindings = normalizeBindings(request.bindings ?? {});
  const accessorRequest = {
    bindings,
    ...(request.evaluationLimits === undefined ? {} : { limits: request.evaluationLimits }),
  };
  const lhs = prepareViewAccessor(layoutArtifact, { viewId: kernel.operation.lhs.viewId, ...accessorRequest });
  const rhs = prepareViewAccessor(layoutArtifact, { viewId: kernel.operation.rhs.viewId, ...accessorRequest });
  const destination = prepareViewAccessor(layoutArtifact, { viewId: kernel.operation.destination.viewId, ...accessorRequest });
  ensurePreparedContract(lhs, rhs, destination);

  const [m, k] = lhs.logicalShape as readonly [bigint, bigint];
  const [, n] = rhs.logicalShape as readonly [bigint, bigint];
  const tileM = BigInt(kernel.operation.logicalTile.m);
  const tileN = BigInt(kernel.operation.logicalTile.n);
  const tileK = BigInt(kernel.operation.logicalTile.k);
  const lhsElements = m * k;
  const rhsElements = k * n;
  const outputElements = m * n;
  const aggregateElements = lhsElements + rhsElements + outputElements;
  const maxElements = resolveBudget(request.maxElements, DEFAULT_MAX_ELEMENTS, MAX_CONFIGURABLE_ELEMENTS, "maxElements");
  if (aggregateElements > BigInt(maxElements)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.maxElements", `logical GEMM tile preparation requires ${aggregateElements} aggregate elements; limit is ${maxElements}`);
  }
  const multiplyAdds = outputElements * k;
  const maxMultiplyAdds = resolveBudget(request.maxMultiplyAdds, DEFAULT_MAX_MULTIPLY_ADDS, MAX_CONFIGURABLE_MULTIPLY_ADDS, "maxMultiplyAdds");
  if (multiplyAdds > BigInt(maxMultiplyAdds)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.maxMultiplyAdds", `logical GEMM tile requires ${multiplyAdds} multiply-adds; limit is ${maxMultiplyAdds}`);
  }
  const evaluationSteps = (lhsElements * BigInt(lhs.evaluationStepsPerAccess))
    + (rhsElements * BigInt(rhs.evaluationStepsPerAccess))
    + (outputElements * BigInt(destination.evaluationStepsPerAccess));
  const maxEvaluationSteps = resolveBudget(
    request.maxEvaluationSteps,
    DEFAULT_MAX_EVALUATION_STEPS,
    MAX_CONFIGURABLE_EVALUATION_STEPS,
    "maxEvaluationSteps",
  );
  if (evaluationSteps > BigInt(maxEvaluationSteps)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.maxEvaluationSteps", `logical GEMM address proof requires ${evaluationSteps} evaluation steps; limit is ${maxEvaluationSteps}`);
  }
  const maxPreparationMs = resolveBudget(
    request.maxPreparationMs,
    DEFAULT_MAX_PREPARATION_MS,
    MAX_CONFIGURABLE_PREPARATION_MS,
    "maxPreparationMs",
  );
  ensurePreparationActive(preparationStartedAt, maxPreparationMs, request.signal);
  await proveDenseAccessor(lhs, [m, k], preparationStartedAt, maxPreparationMs, request.signal, "lhs");
  await proveDenseAccessor(rhs, [k, n], preparationStartedAt, maxPreparationMs, request.signal, "rhs");
  await proveDenseAccessor(destination, [m, n], preparationStartedAt, maxPreparationMs, request.signal, "destination");

  const kernelSemanticHash = await hashSemanticArtifact(kernelArtifact, hashOptions);
  ensurePreparationActive(preparationStartedAt, maxPreparationMs, request.signal);
  const specializationHash = await hashNamedComponents({
    profile: "browsergrad.logical-gemm-tile.dense-rank2-f32@1",
    layout: layoutSemanticHash,
    kernel: kernelSemanticHash,
    operation: kernel.operation.operationId,
    bindings: bindings as unknown as JsonObject,
    resolved: {
      m: encodeWireU64(m),
      n: encodeWireU64(n),
      k: encodeWireU64(k),
      tileM: kernel.operation.logicalTile.m,
      tileN: kernel.operation.logicalTile.n,
      tileK: kernel.operation.logicalTile.k,
      lhsByteOffset: encodeWireU64(lhs.viewByteOffset),
      rhsByteOffset: encodeWireU64(rhs.viewByteOffset),
      destinationByteOffset: encodeWireU64(destination.viewByteOffset),
      lhsAllocationBytes: encodeWireU64(lhs.allocationByteLength),
      rhsAllocationBytes: encodeWireU64(rhs.allocationByteLength),
      destinationAllocationBytes: encodeWireU64(destination.allocationByteLength),
    },
  }, hashOptions);
  ensurePreparationActive(preparationStartedAt, maxPreparationMs, request.signal);
  const prepared = Object.freeze({
    operation: kernel.operation,
    bindings,
    lhs,
    rhs,
    destination,
    layoutSemanticHash,
    kernelSemanticHash,
    specializationHash,
    m,
    n,
    k,
    tileM,
    tileN,
    tileK,
    outputElements,
    multiplyAdds,
  });
  PREPARED_LOGICAL_GEMM_TILES.add(prepared);
  return prepared;
}

/** @internal Shared authority check for semantic-core schedule composition. */
export function requirePreparedLogicalGemmTileSpecialization(
  prepared: PreparedLogicalGemmTileSpecialization,
): void {
  if (!PREPARED_LOGICAL_GEMM_TILES.has(prepared as object)) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidBinding,
      "$.logical",
      "logical GEMM schedule requires an exact specialization produced by this semantic-core module instance",
    );
  }
}

function ensurePreparedContract(
  lhs: PreparedViewAccessor,
  rhs: PreparedViewAccessor,
  destination: PreparedViewAccessor,
): void {
  for (const [role, accessor] of [["lhs", lhs], ["rhs", rhs], ["destination", destination]] as const) {
    if (accessor.dtype !== "f32" || accessor.dtypeBytes !== 4 || accessor.logicalShape.length !== 2) {
      invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.${role}`, "logical GEMM tile specialization requires rank-2 f32 views");
    }
    if (accessor.memorySpace.kind !== "global") {
      invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.${role}`, "logical GEMM tile specialization requires global-memory views");
    }
    if (!accessor.fullySpecialized) invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, `$.${role}`, "logical GEMM tile requires every dimension binding");
  }
  if (lhs.logicalShape[0] !== destination.logicalShape[0] || lhs.logicalShape[1] !== rhs.logicalShape[0] || rhs.logicalShape[1] !== destination.logicalShape[1]) {
    invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, "$.operation", "resolved lhs, rhs, and destination GEMM dimensions must match");
  }
  if (new Set([lhs.allocationId, rhs.allocationId, destination.allocationId]).size !== 3
    || new Set([lhs.aliasSetId, rhs.aliasSetId, destination.aliasSetId]).size !== 3) {
    invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, "$.operation.overlap", "logical GEMM tile requires pairwise-disjoint allocations and alias sets");
  }
}

async function proveDenseAccessor(
  accessor: PreparedViewAccessor,
  shape: readonly [bigint, bigint],
  startedAt: number,
  maxPreparationMs: number,
  signal: AbortSignal | undefined,
  role: string,
): Promise<void> {
  let yieldAt = startedAt + YIELD_INTERVAL_MS;
  let linear = 0n;
  for (let row = 0n; row < shape[0]; row += 1n) {
    for (let column = 0n; column < shape[1]; column += 1n) {
      if ((linear & 1023n) === 0n) {
        ensurePreparationActive(startedAt, maxPreparationMs, signal);
        const now = monotonicNow();
        if (now >= yieldAt) {
          await yieldToMainThread();
          ensurePreparationActive(startedAt, maxPreparationMs, signal);
          yieldAt = monotonicNow() + YIELD_INTERVAL_MS;
        }
      }
      const access = accessor.access([row, column]);
      if (!access.accessInBounds) invalid(KERNEL_DIAGNOSTIC_CODES.invalidAccess, `$.${role}[${row},${column}]`, "dense logical GEMM coordinate is not a valid allocation access");
      const expected = accessor.viewByteOffset + (linear * 4n);
      if (access.rootByteStart !== expected) {
        invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.${role}[${row},${column}]`, "initial logical GEMM tile profile requires dense row-major views");
      }
      if (access.rootByteStart > BigInt(Number.MAX_SAFE_INTEGER)) {
        invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `$.${role}[${row},${column}]`, "CPU-addressable logical GEMM offsets must fit exact JavaScript indexes");
      }
      linear += 1n;
    }
  }
  ensurePreparationActive(startedAt, maxPreparationMs, signal);
}

function normalizeBindings(bindings: Readonly<Record<string, WireI64>>): Readonly<Record<string, WireI64>> {
  if (typeof bindings !== "object" || bindings === null || Array.isArray(bindings)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "bindings must be a plain data object");
  }
  const prototype = Object.getPrototypeOf(bindings);
  if (prototype !== Object.prototype && prototype !== null) invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "bindings must be a plain data object");
  const descriptors = Object.getOwnPropertyDescriptors(bindings);
  const result = Object.create(null) as Record<string, WireI64>;
  for (const key of Reflect.ownKeys(bindings)) {
    if (typeof key !== "string") invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, "$.bindings", "binding keys must be strings");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, `$.bindings.${key}`, "bindings must use enumerable data properties without accessors");
    }
    result[key] = parseWireI64(descriptor.value, `$.bindings.${key}`);
  }
  return Object.freeze(result);
}

function resolveBudget(value: number | undefined, fallback: number, maximum: number, name: string): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved <= 0 || resolved > maximum) {
    resource(`$.${name}`, `${name} must be a positive safe integer no greater than ${maximum}`);
  }
  return resolved;
}

function monotonicNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function isAborted(signal: AbortSignal | undefined): boolean {
  return signal?.aborted === true;
}

function ensurePreparationActive(startedAt: number, maxPreparationMs: number, signal: AbortSignal | undefined): void {
  if (isAborted(signal)) resource("$.signal", "logical GEMM tile preparation was aborted");
  if (monotonicNow() - startedAt > maxPreparationMs) {
    resource("$.maxPreparationMs", `logical GEMM tile preparation exceeded ${maxPreparationMs} ms`);
  }
}

async function yieldToMainThread(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function resource(path: string, message: string): never {
  invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, path, message);
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

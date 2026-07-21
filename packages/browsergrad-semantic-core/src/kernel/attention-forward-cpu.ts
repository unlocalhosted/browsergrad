import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { encodeWireU64, type WireU64 } from "../schema/integers.js";
import type { VerifiedAttentionForwardArtifact } from "./attention-forward-artifact.js";
import {
  INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY,
} from "./attention-forward-model.js";
import {
  prepareAttentionForwardSpecialization,
  type PreparedAttentionForwardSpecialization,
  type PrepareAttentionForwardSpecializationRequest,
} from "./attention-forward-prepare.js";
import {
  captureExactUint8Bindings,
  copyNativeUint8,
  nativeRangesOverlap,
  nativeUint8Slots,
  requireExactNativeByteLength,
  requireNativeAlignment,
  type NativeUint8Slots,
} from "./native-buffer.js";
import { kernelMonotonicNow, resolveKernelBudget } from "./preparation.js";

const DEFAULT_MAX_EXECUTION_MS = 5_000;
const MAX_CONFIGURABLE_EXECUTION_MS = 60_000;
const YIELD_INTERVAL_MS = 16;
const REFLECT_APPLY = Reflect.apply;
const ARRAY_BUFFER_CONSTRUCTOR = ArrayBuffer;
const DATA_VIEW_CONSTRUCTOR = DataView;
const FLOAT32_ARRAY_CONSTRUCTOR = Float32Array;
const DATA_VIEW_GET_FLOAT32 = DataView.prototype.getFloat32;
const DATA_VIEW_SET_FLOAT32 = DataView.prototype.setFloat32;
const DATA_VIEW_SET_UINT32 = DataView.prototype.setUint32;
const MATH_FROUND = Math.fround;
const MATH_EXP = Math.exp;
const MATH_MAX = Math.max;
const MATH_ABS = Math.abs;
const NUMBER_IS_FINITE = Number.isFinite;
const NUMBER_PARSE_INT = Number.parseInt;
const NUMBER_CONSTRUCTOR = Number;
const NUMBER_MAX_VALUE = Number.MAX_VALUE;
const NUMBER_MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;

export interface PrepareAttentionForwardCpuRequest
  extends PrepareAttentionForwardSpecializationRequest {}

export interface AttentionForwardCpuBuffers {
  readonly query: Uint8Array;
  readonly key: Uint8Array;
  readonly value: Uint8Array;
  readonly destination: Uint8Array;
}

export interface AttentionForwardCpuExecutionOptions {
  readonly maxExecutionMs?: number;
  readonly signal?: AbortSignal;
}

export interface AttentionForwardCpuTrace {
  readonly operationId: string;
  readonly specializationHash: string;
  readonly referenceProfileId: "browsergrad.attention-forward.cpu-stable-softmax-f32@1";
  readonly comparisonPolicyId: "browsergrad.attention-forward.f32-abs-relative@1";
  readonly mask: "none" | "causal-upper-left";
  readonly batch: WireU64;
  readonly heads: WireU64;
  readonly queryLength: WireU64;
  readonly keyLength: WireU64;
  readonly queryDepth: WireU64;
  readonly valueDepth: WireU64;
  readonly validScoreElements: WireU64;
  readonly scoreMultiplyAdds: WireU64;
  readonly weightedValueMultiplyAdds: WireU64;
  readonly outputElements: WireU64;
  readonly bytesRead: WireU64;
  readonly bytesWritten: WireU64;
}

export interface AttentionForwardComparisonTrace {
  readonly comparisonPolicyId: "browsergrad.attention-forward.f32-abs-relative@1";
  readonly passed: boolean;
  readonly comparedElements: WireU64;
  readonly mismatchCount: WireU64;
  readonly firstMismatchIndex: WireU64 | null;
  readonly maxAbsoluteError: number;
  readonly maxRelativeError: number;
}

export interface PreparedAttentionForwardCpu {
  readonly operationId: string;
  readonly specializationHash: string;
  readonly batch: bigint;
  readonly heads: bigint;
  readonly queryLength: bigint;
  readonly keyLength: bigint;
  readonly queryDepth: bigint;
  readonly valueDepth: bigint;
  readonly execute: (
    buffers: AttentionForwardCpuBuffers,
    options?: AttentionForwardCpuExecutionOptions,
  ) => Promise<AttentionForwardCpuTrace>;
  readonly compare: (
    actual: Uint8Array,
    expected: Uint8Array,
  ) => AttentionForwardComparisonTrace;
}

/**
 * Prepares a schedule-independent stable-softmax CPU oracle. Inputs are copied
 * into private fixed buffers before the first yield and destination writes are
 * delayed until all domain and numerical checks pass.
 */
export async function prepareAttentionForwardCpu(
  layoutArtifact: VerifiedLayoutArtifact,
  kernelArtifact: VerifiedAttentionForwardArtifact,
  request: PrepareAttentionForwardCpuRequest,
): Promise<PreparedAttentionForwardCpu> {
  const prepared = await prepareAttentionForwardSpecialization(
    layoutArtifact,
    kernelArtifact,
    request,
  );
  const scale = decodeF32Bits(prepared.operation.scale.value.bits);
  const outputLength = safeElementCount(prepared.outputElements, "$.destination");
  const scoreLength = safeElementCount(prepared.keyLength, "$.keyLength");

  const execute = async (
    buffers: AttentionForwardCpuBuffers,
    options: AttentionForwardCpuExecutionOptions = {},
  ): Promise<AttentionForwardCpuTrace> => {
    const startedAt = kernelMonotonicNow();
    const maxExecutionMs = resolveKernelBudget(
      options.maxExecutionMs,
      DEFAULT_MAX_EXECUTION_MS,
      MAX_CONFIGURABLE_EXECUTION_MS,
      "maxExecutionMs",
    );
    const admitted = validateBuffers(buffers, prepared);
    const querySnapshot = copyNativeUint8(admitted.bindings.query, admitted.slots.query);
    const keySnapshot = copyNativeUint8(admitted.bindings.key, admitted.slots.key);
    const valueSnapshot = copyNativeUint8(admitted.bindings.value, admitted.slots.value);
    const queryView = new DATA_VIEW_CONSTRUCTOR(querySnapshot.buffer);
    const keyView = new DATA_VIEW_CONSTRUCTOR(keySnapshot.buffer);
    const valueView = new DATA_VIEW_CONSTRUCTOR(valueSnapshot.buffer);
    await validateFiniteInput(
      queryView,
      prepared.query.viewByteOffset,
      prepared.queryElements,
      "query",
      startedAt,
      maxExecutionMs,
      options.signal,
    );
    await validateFiniteInput(
      keyView,
      prepared.key.viewByteOffset,
      prepared.keyElements,
      "key",
      startedAt,
      maxExecutionMs,
      options.signal,
    );
    await validateFiniteInput(
      valueView,
      prepared.value.viewByteOffset,
      prepared.valueElements,
      "value",
      startedAt,
      maxExecutionMs,
      options.signal,
    );

    const output = new FLOAT32_ARRAY_CONSTRUCTOR(outputLength);
    const scores = new FLOAT32_ARRAY_CONSTRUCTOR(scoreLength);
    let yieldAt = startedAt + YIELD_INTERVAL_MS;
    let validScoreElements = 0n;
    let scoreMultiplyAdds = 0n;
    let weightedValueMultiplyAdds = 0n;
    for (let batch = 0n; batch < prepared.batch; batch += 1n) {
      for (let head = 0n; head < prepared.heads; head += 1n) {
        for (let queryIndex = 0n; queryIndex < prepared.queryLength; queryIndex += 1n) {
          ensureExecutionActive(startedAt, maxExecutionMs, options.signal);
          const validKeys = validKeyCount(prepared, queryIndex);
          let maximum = -Infinity;
          for (let keyIndex = 0n; keyIndex < validKeys; keyIndex += 1n) {
            let score = MATH_FROUND(0);
            for (let depth = 0n; depth < prepared.queryDepth; depth += 1n) {
              const queryValue = readF32(queryView,
                denseOffset(
                  prepared.query.viewByteOffset,
                  [batch, head, queryIndex, depth],
                  [prepared.batch, prepared.heads, prepared.queryLength, prepared.queryDepth],
                ),
              );
              const keyValue = readF32(keyView,
                denseOffset(
                  prepared.key.viewByteOffset,
                  [batch, head, keyIndex, depth],
                  [prepared.batch, prepared.heads, prepared.keyLength, prepared.queryDepth],
                ),
              );
              score = MATH_FROUND(score + MATH_FROUND(queryValue * keyValue));
              scoreMultiplyAdds += 1n;
            }
            score = MATH_FROUND(score * scale);
            if (!NUMBER_IS_FINITE(score)) {
              domain("$.buffers", `scaled score [${batch},${head},${queryIndex},${keyIndex}] is not finite`);
            }
            scores[NUMBER_CONSTRUCTOR(keyIndex)] = score;
            maximum = MATH_MAX(maximum, score);
            validScoreElements += 1n;
          }
          if (!NUMBER_IS_FINITE(maximum)) {
            domain("$.buffers", `attention row [${batch},${head},${queryIndex}] has no finite score maximum`);
          }
          let denominator = MATH_FROUND(0);
          for (let keyIndex = 0n; keyIndex < validKeys; keyIndex += 1n) {
            const weight = MATH_FROUND(MATH_EXP(MATH_FROUND(
              (scores[NUMBER_CONSTRUCTOR(keyIndex)] as number) - maximum,
            )));
            if (!NUMBER_IS_FINITE(weight)) {
              domain("$.buffers", `attention exponential [${batch},${head},${queryIndex},${keyIndex}] is not finite`);
            }
            scores[NUMBER_CONSTRUCTOR(keyIndex)] = weight;
            denominator = MATH_FROUND(denominator + weight);
          }
          if (!NUMBER_IS_FINITE(denominator) || denominator <= 0) {
            domain("$.buffers", `attention denominator [${batch},${head},${queryIndex}] is not positive finite`);
          }
          for (let valueIndex = 0n; valueIndex < prepared.valueDepth; valueIndex += 1n) {
            let accumulator = MATH_FROUND(0);
            for (let keyIndex = 0n; keyIndex < validKeys; keyIndex += 1n) {
              const probability = MATH_FROUND(
                (scores[NUMBER_CONSTRUCTOR(keyIndex)] as number) / denominator,
              );
              const inputValue = readF32(valueView,
                denseOffset(
                  prepared.value.viewByteOffset,
                  [batch, head, keyIndex, valueIndex],
                  [prepared.batch, prepared.heads, prepared.keyLength, prepared.valueDepth],
                ),
              );
              accumulator = MATH_FROUND(
                accumulator + MATH_FROUND(probability * inputValue),
              );
              weightedValueMultiplyAdds += 1n;
            }
            if (!NUMBER_IS_FINITE(accumulator)) {
              domain("$.buffers", `attention output [${batch},${head},${queryIndex},${valueIndex}] is not finite`);
            }
            output[denseElementIndex(
              [batch, head, queryIndex, valueIndex],
              [prepared.batch, prepared.heads, prepared.queryLength, prepared.valueDepth],
            )] = accumulator;
          }
          const now = kernelMonotonicNow();
          if (now >= yieldAt) {
            await yieldToMainThread();
            ensureExecutionActive(startedAt, maxExecutionMs, options.signal);
            yieldAt = kernelMonotonicNow() + YIELD_INTERVAL_MS;
          }
        }
      }
    }
    ensureExecutionActive(startedAt, maxExecutionMs, options.signal);
    const destinationSlots = nativeUint8Slots(
      admitted.bindings.destination,
      "$.buffers.destination",
    );
    requireUnchangedSlots(destinationSlots, admitted.slots.destination, "$.buffers.destination");
    const destinationView = new DATA_VIEW_CONSTRUCTOR(
      destinationSlots.buffer,
      destinationSlots.byteOffset,
      destinationSlots.byteLength,
    );
    for (let index = 0; index < output.length; index += 1) {
      writeF32(destinationView,
        safeIndex(prepared.destination.viewByteOffset + (BigInt(index) * 4n), "$.destination"),
        output[index] as number,
      );
    }
    return Object.freeze({
      operationId: prepared.operation.operationId,
      specializationHash: prepared.specializationHash,
      referenceProfileId: "browsergrad.attention-forward.cpu-stable-softmax-f32@1",
      comparisonPolicyId: INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY.policyId,
      mask: prepared.operation.mask.kind === "causal" ? "causal-upper-left" : "none",
      batch: encodeWireU64(prepared.batch),
      heads: encodeWireU64(prepared.heads),
      queryLength: encodeWireU64(prepared.queryLength),
      keyLength: encodeWireU64(prepared.keyLength),
      queryDepth: encodeWireU64(prepared.queryDepth),
      valueDepth: encodeWireU64(prepared.valueDepth),
      validScoreElements: encodeWireU64(validScoreElements),
      scoreMultiplyAdds: encodeWireU64(scoreMultiplyAdds),
      weightedValueMultiplyAdds: encodeWireU64(weightedValueMultiplyAdds),
      outputElements: encodeWireU64(prepared.outputElements),
      bytesRead: encodeWireU64((scoreMultiplyAdds * 8n) + (weightedValueMultiplyAdds * 4n)),
      bytesWritten: encodeWireU64(prepared.outputElements * 4n),
    });
  };

  const compare = (actual: Uint8Array, expected: Uint8Array): AttentionForwardComparisonTrace => {
    const captured = captureExactUint8Bindings(
      { actual, expected },
      ["actual", "expected"] as const,
      "$.comparison",
    );
    const actualSlots = nativeUint8Slots(captured.actual, "$.comparison.actual");
    const expectedSlots = nativeUint8Slots(captured.expected, "$.comparison.expected");
    requireExactNativeByteLength(
      actualSlots,
      prepared.destination.allocationByteLength,
      "$.comparison.actual",
    );
    requireExactNativeByteLength(
      expectedSlots,
      prepared.destination.allocationByteLength,
      "$.comparison.expected",
    );
    requireNativeAlignment(
      actualSlots,
      prepared.destination.allocationAlignmentBytes,
      "$.comparison.actual",
    );
    requireNativeAlignment(
      expectedSlots,
      prepared.destination.allocationAlignmentBytes,
      "$.comparison.expected",
    );
    const actualView = new DATA_VIEW_CONSTRUCTOR(
      actualSlots.buffer,
      actualSlots.byteOffset,
      actualSlots.byteLength,
    );
    const expectedView = new DATA_VIEW_CONSTRUCTOR(
      expectedSlots.buffer,
      expectedSlots.byteOffset,
      expectedSlots.byteLength,
    );
    let mismatchCount = 0n;
    let firstMismatchIndex: bigint | null = null;
    let maxAbsoluteError = 0;
    let maxRelativeError = 0;
    for (let index = 0n; index < prepared.outputElements; index += 1n) {
      const offset = safeIndex(prepared.destination.viewByteOffset + (index * 4n), "$.comparison");
      const actualValue = readF32(actualView, offset);
      const expectedValue = readF32(expectedView, offset);
      const finite = NUMBER_IS_FINITE(actualValue) && NUMBER_IS_FINITE(expectedValue);
      const absoluteError = finite ? MATH_ABS(actualValue - expectedValue) : NUMBER_MAX_VALUE;
      const relativeError = finite && expectedValue !== 0
        ? absoluteError / MATH_ABS(expectedValue)
        : (absoluteError === 0 ? 0 : NUMBER_MAX_VALUE);
      maxAbsoluteError = MATH_MAX(maxAbsoluteError, absoluteError);
      maxRelativeError = MATH_MAX(maxRelativeError, relativeError);
      const withinAbsolute = absoluteError <= INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY.absoluteTolerance;
      const withinRelative = relativeError <= INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY.relativeTolerance;
      if (!finite || (!withinAbsolute && !withinRelative)) {
        mismatchCount += 1n;
        if (firstMismatchIndex === null) firstMismatchIndex = index;
      }
    }
    return Object.freeze({
      comparisonPolicyId: INITIAL_ATTENTION_FORWARD_COMPARISON_POLICY.policyId,
      passed: mismatchCount === 0n,
      comparedElements: encodeWireU64(prepared.outputElements),
      mismatchCount: encodeWireU64(mismatchCount),
      firstMismatchIndex: firstMismatchIndex === null ? null : encodeWireU64(firstMismatchIndex),
      maxAbsoluteError,
      maxRelativeError,
    });
  };

  return Object.freeze({
    operationId: prepared.operation.operationId,
    specializationHash: prepared.specializationHash,
    batch: prepared.batch,
    heads: prepared.heads,
    queryLength: prepared.queryLength,
    keyLength: prepared.keyLength,
    queryDepth: prepared.queryDepth,
    valueDepth: prepared.valueDepth,
    execute,
    compare,
  });
}

interface AdmittedBuffers {
  readonly bindings: Readonly<Record<"query" | "key" | "value" | "destination", Uint8Array>>;
  readonly slots: Readonly<Record<"query" | "key" | "value" | "destination", NativeUint8Slots>>;
}

function validateBuffers(
  buffers: AttentionForwardCpuBuffers,
  prepared: PreparedAttentionForwardSpecialization,
): AdmittedBuffers {
  const bindings = captureExactUint8Bindings(
    buffers,
    ["query", "key", "value", "destination"] as const,
    "$.buffers",
  );
  const slots = Object.freeze({
    query: nativeUint8Slots(bindings.query, "$.buffers.query"),
    key: nativeUint8Slots(bindings.key, "$.buffers.key"),
    value: nativeUint8Slots(bindings.value, "$.buffers.value"),
    destination: nativeUint8Slots(bindings.destination, "$.buffers.destination"),
  });
  for (const role of ["query", "key", "value", "destination"] as const) {
    requireExactNativeByteLength(
      slots[role],
      prepared[role].allocationByteLength,
      `$.buffers.${role}`,
    );
    requireNativeAlignment(
      slots[role],
      prepared[role].allocationAlignmentBytes,
      `$.buffers.${role}`,
    );
  }
  const roles = ["query", "key", "value", "destination"] as const;
  for (let left = 0; left < roles.length; left += 1) {
    for (let right = left + 1; right < roles.length; right += 1) {
      const leftRole = roles[left] as typeof roles[number];
      const rightRole = roles[right] as typeof roles[number];
      if (nativeRangesOverlap(slots[leftRole], slots[rightRole])) {
        invalid(
          KERNEL_DIAGNOSTIC_CODES.aliasConflict,
          "$.buffers",
          `attention bindings ${leftRole} and ${rightRole} must not overlap`,
        );
      }
    }
  }
  return Object.freeze({ bindings, slots });
}

async function validateFiniteInput(
  view: DataView,
  base: bigint,
  elements: bigint,
  role: string,
  startedAt: number,
  maxExecutionMs: number,
  signal: AbortSignal | undefined,
): Promise<void> {
  let yieldAt = startedAt + YIELD_INTERVAL_MS;
  for (let index = 0n; index < elements; index += 1n) {
    if ((index & 4095n) === 0n) {
      ensureExecutionActive(startedAt, maxExecutionMs, signal);
      const now = kernelMonotonicNow();
      if (now >= yieldAt) {
        await yieldToMainThread();
        ensureExecutionActive(startedAt, maxExecutionMs, signal);
        yieldAt = kernelMonotonicNow() + YIELD_INTERVAL_MS;
      }
    }
    const value = readF32(view, safeIndex(base + (index * 4n), `$.buffers.${role}`));
    if (!NUMBER_IS_FINITE(value)) {
      domain(`$.buffers.${role}`, `${role} element ${index} is not finite f32`);
    }
  }
  ensureExecutionActive(startedAt, maxExecutionMs, signal);
}

function validKeyCount(
  prepared: PreparedAttentionForwardSpecialization,
  queryIndex: bigint,
): bigint {
  if (prepared.operation.mask.kind === "none") return prepared.keyLength;
  const causalEnd = queryIndex + 1n;
  return causalEnd < prepared.keyLength ? causalEnd : prepared.keyLength;
}

function denseOffset(base: bigint, coordinates: readonly bigint[], shape: readonly bigint[]): number {
  return safeIndex(base + (BigInt(denseElementIndex(coordinates, shape)) * 4n), "$.buffers");
}

function denseElementIndex(coordinates: readonly bigint[], shape: readonly bigint[]): number {
  let linear = 0n;
  for (let axis = 0; axis < shape.length; axis += 1) {
    const coordinate = coordinates[axis];
    const extent = shape[axis];
    if (coordinate === undefined || extent === undefined) {
      throw new Error("internal: attention coordinate axis disappeared");
    }
    linear = (linear * extent) + coordinate;
  }
  return safeIndex(linear, "$.coordinates");
}

function decodeF32Bits(bits: string): number {
  const buffer = new ARRAY_BUFFER_CONSTRUCTOR(4);
  const view = new DATA_VIEW_CONSTRUCTOR(buffer);
  REFLECT_APPLY(DATA_VIEW_SET_UINT32, view, [0, NUMBER_PARSE_INT(bits, 16), false]);
  return REFLECT_APPLY(DATA_VIEW_GET_FLOAT32, view, [0, false]) as number;
}

function safeElementCount(value: bigint, path: string): number {
  if (value < 0n || value > BigInt(NUMBER_MAX_SAFE_INTEGER)) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.unsupportedProfile,
      path,
      "attention scratch element count must fit an exact JavaScript index",
    );
  }
  return NUMBER_CONSTRUCTOR(value);
}

function safeIndex(value: bigint, path: string): number {
  if (value < 0n || value > BigInt(NUMBER_MAX_SAFE_INTEGER)) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidAccess,
      path,
      "byte address cannot be represented as a JavaScript buffer index",
    );
  }
  return NUMBER_CONSTRUCTOR(value);
}

function readF32(view: DataView, offset: number): number {
  return REFLECT_APPLY(DATA_VIEW_GET_FLOAT32, view, [offset, true]) as number;
}

function writeF32(view: DataView, offset: number, value: number): void {
  REFLECT_APPLY(DATA_VIEW_SET_FLOAT32, view, [offset, value, true]);
}

function requireUnchangedSlots(
  actual: NativeUint8Slots,
  expected: NativeUint8Slots,
  path: string,
): void {
  if (actual.buffer !== expected.buffer
    || actual.byteOffset !== expected.byteOffset
    || actual.byteLength !== expected.byteLength) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.invalidBinding,
      path,
      "destination binding changed while attention execution was in progress",
    );
  }
}

function ensureExecutionActive(
  startedAt: number,
  maxExecutionMs: number,
  signal: AbortSignal | undefined,
): void {
  if (signal?.aborted === true) {
    invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, "$.signal", "attention CPU execution was aborted");
  }
  if (kernelMonotonicNow() - startedAt > maxExecutionMs) {
    invalid(
      KERNEL_DIAGNOSTIC_CODES.resourceLimit,
      "$.maxExecutionMs",
      `attention CPU execution exceeded ${maxExecutionMs} ms`,
    );
  }
}

async function yieldToMainThread(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function domain(path: string, message: string): never {
  invalid(KERNEL_DIAGNOSTIC_CODES.invalidBinding, path, message);
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}

import {
  createVerifiedDensePermutationViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  parseWireI64,
  wireIntegerToBigInt,
  type WireI64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  prepareSemanticViewCopyWgsl,
  type PreparedSemanticViewCopyWgsl,
} from "./semantic_view_copy.js";
import {
  normalizeTensorGpuPlan,
  runTensorGpuPlanResidentWithPreparedSemanticViewCopies,
  type TensorGpuPlan,
  type TensorPlanInput,
  type TensorPlanResidentResult,
  type TensorPlanRunResult,
  type TensorPlanStep,
} from "./tensor_plan.js";
import { materializeFloat32, releaseDirectBuffer } from "./runner.js";
import { KernelError, type KernelDevice } from "./types.js";
import {
  issueWithWebGpuErrorScopes,
} from "./webgpu_error_scope.js";

export const TENSOR_PLAN_SEMANTIC_REQUEST_SCHEMA =
  "browsergrad.jit.tensor-plan-semantic-requests";
export const TENSOR_PLAN_SEMANTIC_REQUEST_VERSION = Object.freeze({
  major: 1,
  minor: 0,
} as const);
export const DENSE_PERMUTATION_VIEW_COPY_REQUEST =
  "dense-permutation-view-copy";

const MAX_REQUESTS = 4_096;
const MAX_JSON_CODE_UNITS = 512 * 1_024;
const PREPARED_REQUESTS = new WeakMap<
  object,
  ReadonlyMap<number, PreparedSemanticViewCopyWgsl>
>();

type JsonRecord = Record<string, unknown>;

interface DensePermutationRequest {
  readonly kind: typeof DENSE_PERMUTATION_VIEW_COPY_REQUEST;
  readonly valueId: number;
  readonly inputShape: readonly WireI64[];
  readonly axes: readonly number[];
  readonly dtype: "f32";
}

export interface PreparedTensorPlanSemanticRequest {
  readonly kind: typeof DENSE_PERMUTATION_VIEW_COPY_REQUEST;
  /** Plan-local routing identity. Excluded from all semantic hashes. */
  readonly valueId: number;
  readonly inputShape: readonly WireI64[];
  readonly axes: readonly number[];
  readonly dtype: "f32";
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly semanticSpecializationHash: string;
  readonly wgslModuleHash: string;
  readonly backendProfile: PreparedSemanticViewCopyWgsl["backendProfile"];
  readonly backendVersion: PreparedSemanticViewCopyWgsl["backendVersion"];
  readonly workgroupSize: number;
  readonly logicalInvocationCount: readonly [number, number, number];
  /** Planned workgroups. Actual submitted workgroups come from execution profiles. */
  readonly plannedWorkgroupCount: readonly [number, number, number];
}

export interface PreparedTensorPlanSemanticRequests {
  readonly schema: typeof TENSOR_PLAN_SEMANTIC_REQUEST_SCHEMA;
  readonly version: typeof TENSOR_PLAN_SEMANTIC_REQUEST_VERSION;
  readonly requests: readonly PreparedTensorPlanSemanticRequest[];
}

export interface TensorPlanResidentSemanticResult extends TensorPlanResidentResult {
  /** Exact authority-bound preparation consumed by this execution. */
  readonly semanticPreparation: PreparedTensorPlanSemanticRequests;
}

interface PreparedPermutation {
  readonly artifacts: VerifiedViewCopyArtifacts;
  readonly wgsl: PreparedSemanticViewCopyWgsl;
}

/**
 * Verify JIT-emitted construction requests, build canonical artifacts, and
 * prepare their exact WGSL beside the frozen scheduling plan.
 *
 * Plan fields are used only to correlate and reject transport drift. Shape,
 * layout, allocation, effects, aliases, IDs, and hashes are constructed solely
 * from each closed semantic request by browsergrad-semantic-core.
 */
export async function prepareTensorPlanSemanticRequests(
  rawPlan: TensorGpuPlan | unknown,
  rawRequests: unknown,
): Promise<PreparedTensorPlanSemanticRequests> {
  const plan = normalizeTensorGpuPlan(rawPlan);
  const requests = parseRequestEnvelope(rawRequests);
  const permutationSteps = plan.steps.filter((step) => step.op === "PERMUTE");
  if (requests.length !== permutationSteps.length) {
    semanticError(
      "$.requests",
      `expected exactly ${permutationSteps.length} request(s) for plan PERMUTE steps, got ${requests.length}`,
    );
  }

  const stepsByValueId = indexPlanSteps(plan);
  const semanticCache = new Map<string, Promise<PreparedPermutation>>();
  const preparedByValueId = new Map<number, PreparedSemanticViewCopyWgsl>();
  const preparedRequests: PreparedTensorPlanSemanticRequest[] = [];

  for (let index = 0; index < requests.length; index += 1) {
    const request = requests[index] as DensePermutationRequest;
    const expectedStep = permutationSteps[index] as TensorPlanStep;
    const path = `$.requests[${index}]`;
    if (request.valueId !== expectedStep.valueId) {
      semanticError(
        `${path}.valueId`,
        `request order/valueId must match plan PERMUTE value ${expectedStep.valueId}`,
      );
    }
    verifyPlanProjection(request, expectedStep, stepsByValueId, path);

    const semanticKey = JSON.stringify({
      inputShape: request.inputShape,
      axes: request.axes,
      dtype: request.dtype,
    });
    let preparation = semanticCache.get(semanticKey);
    if (preparation === undefined) {
      preparation = preparePermutation(request);
      semanticCache.set(semanticKey, preparation);
    }
    const { artifacts, wgsl } = await preparation;
    const logicalInvocationCount = Object.freeze([
      wgsl.launch.dispatchCount[0],
      wgsl.launch.dispatchCount[1],
      wgsl.launch.dispatchCount[2],
    ] as const);
    const plannedWorkgroupCount = Object.freeze([
      Math.max(Math.ceil(logicalInvocationCount[0] / wgsl.program.workgroupSize[0]), 1),
      Math.max(Math.ceil(logicalInvocationCount[1] / wgsl.program.workgroupSize[1]), 1),
      Math.max(Math.ceil(logicalInvocationCount[2] / wgsl.program.workgroupSize[2]), 1),
    ] as const);
    preparedByValueId.set(request.valueId, wgsl);
    preparedRequests.push(Object.freeze({
      kind: request.kind,
      valueId: request.valueId,
      inputShape: request.inputShape,
      axes: request.axes,
      dtype: request.dtype,
      layoutSemanticHash: artifacts.layoutSemanticHash,
      kernelSemanticHash: artifacts.kernelSemanticHash,
      semanticSpecializationHash: wgsl.semantic.specializationHash,
      wgslModuleHash: wgsl.wgslModuleHash,
      backendProfile: wgsl.backendProfile,
      backendVersion: wgsl.backendVersion,
      workgroupSize: wgsl.workgroupSize,
      logicalInvocationCount,
      plannedWorkgroupCount,
    }));
  }

  const prepared = Object.freeze({
    schema: TENSOR_PLAN_SEMANTIC_REQUEST_SCHEMA,
    version: TENSOR_PLAN_SEMANTIC_REQUEST_VERSION,
    requests: Object.freeze(preparedRequests),
  });
  PREPARED_REQUESTS.set(prepared, preparedByValueId);
  return prepared;
}

export async function runTensorGpuPlanSemantic(
  device: KernelDevice,
  rawPlan: TensorGpuPlan | unknown,
  rawRequests: unknown,
  inputs: readonly TensorPlanInput[],
): Promise<TensorPlanRunResult> {
  const resident = await runTensorGpuPlanResidentSemantic(
    device,
    rawPlan,
    rawRequests,
    inputs,
  );
  try {
    const data = await materializeFloat32(device, resident.buffer, resident.byteLength);
    const result = Object.freeze({
      data,
      shape: resident.shape,
      peakLiveBytes: resident.peakLiveBytes,
      materializedValueId: resident.residentValueId,
      earlyReleasedBuffers: resident.earlyReleasedBuffers,
      earlyReleasedBytes: resident.earlyReleasedBytes,
      profiles: resident.profiles,
    });
    releaseDirectBuffer(device, resident.buffer, resident.byteLength);
    return result;
  } catch (error) {
    resident.buffer.destroy();
    device.clearCache();
    await Promise.allSettled(resident.profiles);
    throw error;
  }
}

export async function runTensorGpuPlanResidentSemantic(
  device: KernelDevice,
  rawPlan: TensorGpuPlan | unknown,
  rawRequests: unknown,
  inputs: readonly TensorPlanInput[],
): Promise<TensorPlanResidentSemanticResult> {
  const prepared = await prepareTensorPlanSemanticRequests(rawPlan, rawRequests);
  const viewCopies = requirePreparedMap(prepared);
  let result: TensorPlanResidentResult;
  try {
    result = await issueWithWebGpuErrorScopes(
      device.gpu,
      "$.tensorPlan.semanticDispatch",
      () => runTensorGpuPlanResidentWithPreparedSemanticViewCopies(
        device,
        rawPlan,
        inputs,
        viewCopies,
      ),
      {
        cleanup: (failed) => {
          failed.buffer.destroy();
          void Promise.allSettled(failed.profiles);
        },
      },
    );
  } catch (error) {
    // A late scope/device failure invalidates both the produced root and any
    // pipeline/output-pool state touched by the issue phase.
    device.clearCache();
    throw error;
  }
  return Object.freeze({
    ...result,
    semanticPreparation: prepared,
  });
}

export function assertPreparedTensorPlanSemanticRequests(
  prepared: PreparedTensorPlanSemanticRequests,
): void {
  requirePreparedMap(prepared);
}

export function preparedSemanticViewCopyForValue(
  prepared: PreparedTensorPlanSemanticRequests,
  valueId: number,
): PreparedSemanticViewCopyWgsl | undefined {
  const requests = PREPARED_REQUESTS.get(prepared as object);
  if (requests === undefined) {
    throw new KernelError(
      "tensor plan semantic requests were not prepared by this module instance",
    );
  }
  return requests.get(valueId);
}

function requirePreparedMap(
  prepared: PreparedTensorPlanSemanticRequests,
): ReadonlyMap<number, PreparedSemanticViewCopyWgsl> {
  const requests = PREPARED_REQUESTS.get(prepared as object);
  if (requests === undefined) {
    throw new KernelError(
      "tensor plan semantic requests were not prepared by this module instance",
    );
  }
  return requests;
}

async function preparePermutation(
  request: DensePermutationRequest,
): Promise<PreparedPermutation> {
  const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: request.inputShape,
    axes: request.axes,
    dtype: request.dtype,
  });
  const wgsl = await prepareSemanticViewCopyWgsl(
    artifacts.layout,
    artifacts.kernel,
    { operationId: artifacts.operationId },
  );
  return Object.freeze({ artifacts, wgsl });
}

function indexPlanSteps(plan: TensorGpuPlan): ReadonlyMap<number, TensorPlanStep> {
  const steps = new Map<number, TensorPlanStep>();
  for (let index = 0; index < plan.steps.length; index += 1) {
    const step = plan.steps[index] as TensorPlanStep;
    if (!Number.isSafeInteger(step.valueId) || step.valueId < 0) {
      semanticError(
        `$.plan.steps[${index}].valueId`,
        "plan valueId must be a non-negative safe integer",
      );
    }
    if (steps.has(step.valueId)) {
      semanticError(
        `$.plan.steps[${index}].valueId`,
        `plan value ${step.valueId} is produced more than once`,
      );
    }
    steps.set(step.valueId, step);
  }
  return steps;
}

function verifyPlanProjection(
  request: DensePermutationRequest,
  step: TensorPlanStep,
  stepsByValueId: ReadonlyMap<number, TensorPlanStep>,
  path: string,
): void {
  if (step.op !== "PERMUTE") {
    semanticError(`${path}.valueId`, `plan value ${step.valueId} is not PERMUTE`);
  }
  if (step.inputIds.length !== 1) {
    semanticError(`${path}.valueId`, "plan PERMUTE must have exactly one input");
  }
  const source = stepsByValueId.get(step.inputIds[0] as number);
  if (source === undefined) {
    semanticError(`${path}.valueId`, "plan PERMUTE source is missing");
  }
  if (source.dtype !== "float32" || step.dtype !== "float32") {
    semanticError(`${path}.dtype`, "plan projection must remain float32");
  }
  if (step.arg !== null && step.arg !== undefined) {
    semanticError(
      `${path}.valueId`,
      "semantic-route PERMUTE must erase legacy arg meaning from the scheduling plan",
    );
  }
  compareShape(request.inputShape, source.shape, `${path}.inputShape`);
  const expectedOutput = request.axes.map((axis) => request.inputShape[axis] as WireI64);
  compareShape(expectedOutput, step.shape, `${path}.axes`);
}

function compareShape(
  semanticShape: readonly WireI64[],
  planShape: readonly number[],
  path: string,
): void {
  if (semanticShape.length !== planShape.length) {
    semanticError(path, "semantic request and plan projection ranks differ");
  }
  for (let axis = 0; axis < semanticShape.length; axis += 1) {
    const planExtent = planShape[axis];
    if (!Number.isSafeInteger(planExtent) || (planExtent as number) <= 0) {
      semanticError(path, `plan extent ${axis} must be a positive safe integer`);
    }
    if (wireIntegerToBigInt(semanticShape[axis] as WireI64) !== BigInt(planExtent as number)) {
      semanticError(path, `semantic request and plan projection differ at axis ${axis}`);
    }
  }
}

function parseRequestEnvelope(raw: unknown): readonly DensePermutationRequest[] {
  let value = raw;
  if (typeof raw === "string") {
    if (raw.length > MAX_JSON_CODE_UNITS) {
      semanticError("$", `JSON request envelope exceeds ${MAX_JSON_CODE_UNITS} code units`);
    }
    try {
      value = JSON.parse(raw) as unknown;
    } catch {
      throw new KernelError("tensor plan semantic requests must be valid JSON");
    }
  }
  const envelope = expectClosedRecord(
    value,
    ["schema", "version", "requests"],
    "$",
  );
  if (envelope.schema !== TENSOR_PLAN_SEMANTIC_REQUEST_SCHEMA) {
    semanticError("$.schema", `expected ${TENSOR_PLAN_SEMANTIC_REQUEST_SCHEMA}`);
  }
  const version = expectClosedRecord(
    envelope.version,
    ["major", "minor"],
    "$.version",
  );
  if (
    version.major !== TENSOR_PLAN_SEMANTIC_REQUEST_VERSION.major
    || version.minor !== TENSOR_PLAN_SEMANTIC_REQUEST_VERSION.minor
  ) {
    semanticError("$.version", "only semantic request version 1.0 is supported");
  }
  if (!Array.isArray(envelope.requests)) {
    semanticError("$.requests", "requests must be an array");
  }
  if (envelope.requests.length > MAX_REQUESTS) {
    semanticError("$.requests", `request count exceeds ${MAX_REQUESTS}`);
  }
  return Object.freeze(envelope.requests.map((request, index) =>
    parseDensePermutationRequest(request, index)));
}

function parseDensePermutationRequest(
  raw: unknown,
  index: number,
): DensePermutationRequest {
  const path = `$.requests[${index}]`;
  const request = expectClosedRecord(
    raw,
    ["kind", "valueId", "inputShape", "axes", "dtype"],
    path,
  );
  if (request.kind !== DENSE_PERMUTATION_VIEW_COPY_REQUEST) {
    semanticError(`${path}.kind`, `expected ${DENSE_PERMUTATION_VIEW_COPY_REQUEST}`);
  }
  const valueId = expectSafeInteger(request.valueId, `${path}.valueId`);
  if (valueId < 0) semanticError(`${path}.valueId`, "valueId must be non-negative");
  if (request.dtype !== "f32") {
    semanticError(`${path}.dtype`, "initial semantic permutation request requires f32");
  }
  if (!Array.isArray(request.inputShape)) {
    semanticError(`${path}.inputShape`, "inputShape must be an array");
  }
  if (request.inputShape.length !== 2 && request.inputShape.length !== 3) {
    semanticError(`${path}.inputShape`, "initial semantic permutation request requires rank 2 or 3");
  }
  const inputShape = Object.freeze(request.inputShape.map((extent, axis) => {
    let parsed: WireI64;
    try {
      parsed = parseWireI64(extent, `${path}.inputShape[${axis}]`);
    } catch (error) {
      throw new KernelError(
        `tensor plan semantic requests ${path}.inputShape[${axis}]: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    const integer = wireIntegerToBigInt(parsed);
    if (integer <= 0n || integer > BigInt(Number.MAX_SAFE_INTEGER)) {
      semanticError(
        `${path}.inputShape[${axis}]`,
        "extent must be positive and exactly representable by the frozen plan",
      );
    }
    return parsed;
  }));
  if (!Array.isArray(request.axes) || request.axes.length !== inputShape.length) {
    semanticError(`${path}.axes`, "axes must contain exactly one entry per input rank");
  }
  const seen = new Set<number>();
  const axes = Object.freeze(request.axes.map((axis, position) => {
    const value = expectSafeInteger(axis, `${path}.axes[${position}]`);
    if (value < 0 || value >= inputShape.length || seen.has(value)) {
      semanticError(`${path}.axes`, "axes must be an exact non-negative permutation");
    }
    seen.add(value);
    return value;
  }));
  return Object.freeze({
    kind: DENSE_PERMUTATION_VIEW_COPY_REQUEST,
    valueId,
    inputShape,
    axes,
    dtype: "f32",
  });
}

function expectClosedRecord(
  raw: unknown,
  expectedKeys: readonly string[],
  path: string,
): JsonRecord {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    semanticError(path, "expected an object");
  }
  const record = raw as JsonRecord;
  const keys = Reflect.ownKeys(record);
  if (keys.some((key) => typeof key !== "string")) {
    semanticError(path, "symbol keys are not valid JSON fields");
  }
  const actual = (keys as string[]).sort();
  const expected = [...expectedKeys].sort();
  if (
    actual.length !== expected.length
    || actual.some((key, index) => key !== expected[index])
  ) {
    semanticError(path, `expected closed fields ${expected.join(", ")}`);
  }
  return record;
}

function expectSafeInteger(raw: unknown, path: string): number {
  if (typeof raw !== "number" || !Number.isSafeInteger(raw)) {
    semanticError(path, "expected a safe integer");
  }
  return raw;
}

function semanticError(path: string, message: string): never {
  throw new KernelError(`tensor plan semantic requests ${path}: ${message}`);
}

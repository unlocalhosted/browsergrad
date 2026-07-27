import {
  kernelArtifactPayload,
  type VerifiedKernelArtifact,
} from "../kernel/artifact.js";
import {
  prepareViewCopyCpu,
  type PreparedViewCopyCpu,
} from "../kernel/cpu.js";
import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { SemanticSchemaError } from "../schema/diagnostics.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import {
  encodeWireU64,
  parseWireU64,
  wireIntegerToBigInt,
  type WireU64,
} from "../schema/integers.js";
import {
  hostGraphArtifactPayload,
  prepareHostGraphProgram,
  type VerifiedHostGraphArtifact,
} from "./artifact.js";
import type {
  HostGraphAllReduceNode,
  HostGraphCopyNode,
  HostGraphDispatchNode,
  HostGraphMaterializeNode,
  HostGraphResource,
} from "./model.js";

export const HOST_GRAPH_CPU_PROFILE =
  "browsergrad.host-graph.cpu-reference@1" as const;
export const HOST_GRAPH_CPU_MAX_WORKING_BYTES = 268_435_456;
export const HOST_GRAPH_CPU_MAX_ELEMENT_OPERATIONS = 16_777_216;
export const HOST_GRAPH_CPU_MAX_PREPARATION_MS = 60_000;
export const HOST_GRAPH_CPU_MAX_EXECUTION_MS = 60_000;

const DEFAULT_MAX_WORKING_BYTES = 67_108_864;
const DEFAULT_MAX_ELEMENT_OPERATIONS = 4_194_304;
const DEFAULT_MAX_PREPARATION_MS = 5_000;
const DEFAULT_MAX_EXECUTION_MS = 5_000;
const MAX_VIEW_COPY_ELEMENTS = 1_000_000;
const MAX_ARTIFACTS = 256;
const YIELD_INTERVAL_MS = 16;
const UINT8_ARRAY_CONSTRUCTOR = Uint8Array;
const DATA_VIEW_CONSTRUCTOR = DataView;
const ABORT_SIGNAL_CONSTRUCTOR =
  typeof AbortSignal === "undefined" ? undefined : AbortSignal;
const DIRECT_UINT8_PROTOTYPE = Uint8Array.prototype;
const TYPED_ARRAY_PROTOTYPE =
  Object.getPrototypeOf(DIRECT_UINT8_PROTOTYPE) as object;
const TYPED_ARRAY_BUFFER_GETTER = requiredGetter(
  TYPED_ARRAY_PROTOTYPE,
  "buffer",
);
const TYPED_ARRAY_BYTE_LENGTH_GETTER = requiredGetter(
  TYPED_ARRAY_PROTOTYPE,
  "byteLength",
);
const TYPED_ARRAY_BYTE_OFFSET_GETTER = requiredGetter(
  TYPED_ARRAY_PROTOTYPE,
  "byteOffset",
);
const SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER =
  typeof SharedArrayBuffer === "undefined"
    ? undefined
    : Object.getOwnPropertyDescriptor(
      SharedArrayBuffer.prototype,
      "byteLength",
    )?.get;
const ABORT_SIGNAL_ABORTED_GETTER =
  ABORT_SIGNAL_CONSTRUCTOR === undefined
    ? undefined
    : Object.getOwnPropertyDescriptor(
      ABORT_SIGNAL_CONSTRUCTOR.prototype,
      "aborted",
    )?.get;
const UINT8_SET = Uint8Array.prototype.set;
const DATA_VIEW_GET_FLOAT32 = DataView.prototype.getFloat32;
const DATA_VIEW_SET_FLOAT32 = DataView.prototype.setFloat32;
const DATA_VIEW_GET_INT32 = DataView.prototype.getInt32;
const DATA_VIEW_SET_INT32 = DataView.prototype.setInt32;
const DATA_VIEW_GET_UINT32 = DataView.prototype.getUint32;
const DATA_VIEW_SET_UINT32 = DataView.prototype.setUint32;
const REFLECT_APPLY = Reflect.apply;
const MATH_FROUND = Math.fround;
const MATH_MIN = Math.min;
const MATH_MAX = Math.max;
const NUMBER_IS_FINITE = Number.isFinite;
const DATE_NOW = Date.now;
const PERFORMANCE_NOW = globalThis.performance?.now.bind(
  globalThis.performance,
);
const SET_TIMEOUT = globalThis.setTimeout.bind(globalThis);

export type HostGraphCpuErrorCode =
  | "BG-GRAPH-CPU-INVALID-AUTHORITY"
  | "BG-GRAPH-CPU-INVALID-BINDING"
  | "BG-GRAPH-CPU-UNSUPPORTED-PROFILE"
  | "BG-GRAPH-CPU-RESOURCE-LIMIT"
  | "BG-GRAPH-CPU-NUMERICAL-DOMAIN"
  | "BG-GRAPH-CPU-ABORTED"
  | "BG-GRAPH-CPU-TIMEOUT"
  | "BG-GRAPH-CPU-INTERNAL";

export class HostGraphCpuError extends Error {
  constructor(
    readonly code: HostGraphCpuErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "HostGraphCpuError";
  }
}

export interface HostGraphCpuPreparationOptions {
  readonly kernelArtifacts: readonly VerifiedKernelArtifact[];
  readonly layoutArtifacts: readonly VerifiedLayoutArtifact[];
  readonly maxWorkingBytes?: number;
  readonly maxElementOperations?: number;
  readonly maxPreparationMs?: number;
  readonly maxExecutionMs?: number;
  readonly signal?: AbortSignal;
}

export interface HostGraphCpuInputBinding {
  readonly rank: WireU64;
  readonly resourceId: string;
  readonly bytes: Uint8Array;
}

export interface HostGraphCpuExecutionRequest {
  readonly inputs: readonly HostGraphCpuInputBinding[];
  readonly signal?: AbortSignal;
}

export interface HostGraphCpuOutputBinding {
  readonly rank: WireU64;
  readonly resourceId: string;
  readonly bytes: Uint8Array;
}

export interface HostGraphCpuExecutionResult {
  readonly profile: typeof HOST_GRAPH_CPU_PROFILE;
  readonly graphSemanticHash: string;
  readonly failureModel: "fail-stop-no-partial-output-commit";
  readonly executedNodeIds: readonly string[];
  readonly elementOperations: WireU64;
  readonly outputs: readonly HostGraphCpuOutputBinding[];
}

export interface PreparedHostGraphCpu {
  readonly profile: typeof HOST_GRAPH_CPU_PROFILE;
  readonly graphSemanticHash: string;
  readonly rankCount: bigint;
  readonly inputResourceIds: readonly string[];
  readonly outputResourceIds: readonly string[];
  readonly elementOperations: bigint;
  readonly execute: (
    request: HostGraphCpuExecutionRequest,
  ) => Promise<HostGraphCpuExecutionResult>;
}

interface NormalizedPreparationOptions {
  readonly kernelArtifacts: readonly VerifiedKernelArtifact[];
  readonly layoutArtifacts: readonly VerifiedLayoutArtifact[];
  readonly maxWorkingBytes: number;
  readonly maxElementOperations: number;
  readonly maxPreparationMs: number;
  readonly maxExecutionMs: number;
  readonly signal?: AbortSignal;
}

interface SemanticCatalogEntry {
  readonly kernel: VerifiedKernelArtifact;
  readonly layout: VerifiedLayoutArtifact;
}

interface DispatchPlan {
  readonly kind: "dispatch";
  readonly nodeId: string;
  readonly sourceResourceId: string;
  readonly destinationResourceId: string;
  readonly prepared: PreparedViewCopyCpu;
  readonly elementOperations: bigint;
}

interface CollectivePlan {
  readonly kind: "all-reduce";
  readonly nodeId: string;
  readonly resourceId: string;
  readonly reduction: HostGraphAllReduceNode["reduction"];
  readonly dtype: HostGraphAllReduceNode["dtype"];
  readonly participants: readonly number[];
  readonly elementCount: number;
  readonly elementOperations: bigint;
}

interface CopyPlan {
  readonly kind: "copy";
  readonly nodeId: string;
  readonly sourceResourceId: string;
  readonly destinationResourceId: string;
  readonly byteLength: number;
  readonly elementOperations: bigint;
}

interface MaterializePlan {
  readonly kind: "materialize";
  readonly nodeId: string;
  readonly resourceId: string;
  readonly elementOperations: 0n;
}

type CpuNodePlan =
  | DispatchPlan
  | CollectivePlan
  | CopyPlan
  | MaterializePlan;

interface NativeUint8Slots {
  readonly buffer: ArrayBufferLike;
  readonly byteOffset: number;
  readonly byteLength: number;
}

interface AdmittedInput {
  readonly rank: number;
  readonly resourceId: string;
  readonly bytes: Uint8Array;
}

type RankResources = readonly ReadonlyMap<string, Uint8Array>[];

/**
 * Prepares a bounded CPU reference for one exact verified host graph. The
 * returned executor snapshots every caller input and executes only against
 * private rank-local resources. Output buffers become observable only after
 * every node succeeds.
 */
export async function prepareHostGraphCpu(
  artifact: VerifiedHostGraphArtifact,
  options: HostGraphCpuPreparationOptions,
): Promise<PreparedHostGraphCpu> {
  const startedAt = monotonicNow();
  const normalized = normalizePreparationOptions(options);
  let payload: ReturnType<typeof hostGraphArtifactPayload>;
  try {
    payload = hostGraphArtifactPayload(artifact);
  } catch (cause) {
    fail(
      "BG-GRAPH-CPU-INVALID-AUTHORITY",
      "$.artifact",
      "CPU preparation requires an exact verifier-issued host graph artifact",
      cause,
    );
  }
  ensurePreparationActive(startedAt, normalized);
  const preparedGraph = await prepareHostGraphProgram(artifact);
  const rankCount = safeNumber(preparedGraph.rankCount, "$.rankCount");
  const resources = new Map(
    payload.program.resources.map((resource) => [
      resource.resourceId,
      resource,
    ]),
  );
  const resourceBytes = payload.program.resources.reduce(
    (total, resource) =>
      total +
      (wireIntegerToBigInt(resource.byteLength) * preparedGraph.rankCount),
    0n,
  );
  const largestCollectiveScratch = payload.program.nodes.reduce(
    (largest, node) => {
      if (node.kind !== "all-reduce") return largest;
      const resource = resources.get(node.resourceId);
      if (resource === undefined) return largest;
      const byteLength = wireIntegerToBigInt(resource.byteLength);
      return byteLength > largest ? byteLength : largest;
    },
    0n,
  );
  const peakWorkingBytes = resourceBytes + largestCollectiveScratch;
  if (peakWorkingBytes > BigInt(normalized.maxWorkingBytes)) {
    fail(
      "BG-GRAPH-CPU-RESOURCE-LIMIT",
      "$.maxWorkingBytes",
      `rank-local resources plus collective scratch require ${
        peakWorkingBytes
      } bytes; CPU limit is ${normalized.maxWorkingBytes}`,
    );
  }
  const catalog = await buildSemanticCatalog(normalized, startedAt);
  const nodes = new Map(
    payload.program.nodes.map((node) => [node.nodeId, node]),
  );
  const plans: CpuNodePlan[] = [];
  let elementOperations = 0n;
  for (const nodeId of preparedGraph.topologicalNodeIds) {
    ensurePreparationActive(startedAt, normalized);
    const node = nodes.get(nodeId);
    if (node === undefined) {
      fail(
        "BG-GRAPH-CPU-INTERNAL",
        "$.artifact",
        `prepared topological node ${nodeId} disappeared`,
      );
    }
    const plan = node.kind === "dispatch"
      ? await prepareDispatchPlan(
        node,
        catalog,
        rankCount,
        normalized,
        startedAt,
      )
      : node.kind === "all-reduce"
        ? prepareCollectivePlan(node, resources)
        : node.kind === "copy"
          ? prepareCopyPlan(node, resources, rankCount)
          : prepareMaterializePlan(node, resources);
    elementOperations += plan.elementOperations;
    if (elementOperations > BigInt(normalized.maxElementOperations)) {
      fail(
        "BG-GRAPH-CPU-RESOURCE-LIMIT",
        "$.maxElementOperations",
        `graph requires ${elementOperations} element operations; CPU limit is ${
          normalized.maxElementOperations
        }`,
      );
    }
    plans.push(plan);
  }
  ensurePreparationActive(startedAt, normalized);
  const inputResources = Object.freeze(
    payload.program.resources
      .filter((resource) => resource.role === "input"),
  );
  const outputResources = Object.freeze(
    preparedGraph.outputResourceIds.map((resourceId) => {
      const resource = resources.get(resourceId);
      if (resource === undefined || resource.role !== "output") {
        fail(
          "BG-GRAPH-CPU-INTERNAL",
          "$.artifact",
          `prepared output resource ${resourceId} disappeared`,
        );
      }
      return resource;
    }),
  );
  const frozenPlans = Object.freeze(plans);
  const executedNodeIds = Object.freeze(frozenPlans.map((plan) => plan.nodeId));

  const execute = async (
    request: HostGraphCpuExecutionRequest,
  ): Promise<HostGraphCpuExecutionResult> => {
    const executionStartedAt = monotonicNow();
    const captured = captureExecutionRequest(
      request,
      rankCount,
      inputResources,
    );
    ensureExecutionActive(
      executionStartedAt,
      normalized.maxExecutionMs,
      captured.signal,
    );
    const rankResources = materializeRankResources(
      rankCount,
      payload.program.resources,
      captured.inputs,
    );
    for (const plan of frozenPlans) {
      ensureExecutionActive(
        executionStartedAt,
        normalized.maxExecutionMs,
        captured.signal,
      );
      if (plan.kind === "dispatch") {
        executeDispatch(
          plan,
          rankResources,
          executionStartedAt,
          normalized.maxExecutionMs,
          captured.signal,
        );
      } else if (plan.kind === "all-reduce") {
        await executeAllReduce(
          plan,
          rankResources,
          executionStartedAt,
          normalized.maxExecutionMs,
          captured.signal,
        );
      } else if (plan.kind === "copy") {
        executeCopy(
          plan,
          rankResources,
          executionStartedAt,
          normalized.maxExecutionMs,
          captured.signal,
        );
      } else {
        ensureExecutionActive(
          executionStartedAt,
          normalized.maxExecutionMs,
          captured.signal,
        );
      }
    }
    ensureExecutionActive(
      executionStartedAt,
      normalized.maxExecutionMs,
      captured.signal,
    );
    const outputs = Object.freeze(Array.from(
      { length: rankCount },
      (_, rank) => outputResources.map((resource) => {
        const bytes = rankResources[rank]?.get(resource.resourceId);
        if (bytes === undefined) {
          fail(
            "BG-GRAPH-CPU-INTERNAL",
            "$.outputs",
            `rank ${rank} output ${resource.resourceId} disappeared`,
          );
        }
        return Object.freeze({
          rank: encodeWireU64(BigInt(rank)),
          resourceId: resource.resourceId,
          bytes,
        });
      }),
    ).flat());
    return Object.freeze({
      profile: HOST_GRAPH_CPU_PROFILE,
      graphSemanticHash: preparedGraph.graphSemanticHash,
      failureModel: "fail-stop-no-partial-output-commit",
      executedNodeIds,
      elementOperations: encodeWireU64(elementOperations),
      outputs,
    });
  };

  return Object.freeze({
    profile: HOST_GRAPH_CPU_PROFILE,
    graphSemanticHash: preparedGraph.graphSemanticHash,
    rankCount: preparedGraph.rankCount,
    inputResourceIds: Object.freeze(
      inputResources.map((resource) => resource.resourceId),
    ),
    outputResourceIds: Object.freeze(
      outputResources.map((resource) => resource.resourceId),
    ),
    elementOperations,
    execute,
  });
}

async function prepareDispatchPlan(
  node: HostGraphDispatchNode,
  catalog: ReadonlyMap<string, SemanticCatalogEntry>,
  rankCount: number,
  options: NormalizedPreparationOptions,
  startedAt: number,
): Promise<DispatchPlan> {
  const semantic = catalog.get(node.semanticArtifactHash);
  if (semantic === undefined) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      `$.nodes.${node.nodeId}.semanticArtifactHash`,
      "CPU preparation did not receive the exact semantic artifact referenced by the graph",
    );
  }
  const operation = kernelArtifactPayload(semantic.kernel).operations.find(
    (candidate) => candidate.operationId === node.entrypointId,
  );
  if (operation === undefined) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      `$.nodes.${node.nodeId}.entrypointId`,
      "CPU preparation could not resolve the verified graph entrypoint",
    );
  }
  const bindings = new Map(
    node.bindings.map((binding) => [
      binding.semanticResourceId,
      binding.graphResourceId,
    ]),
  );
  const sourceResourceId = bindings.get(operation.source.viewId);
  const destinationResourceId = bindings.get(operation.destination.viewId);
  if (sourceResourceId === undefined || destinationResourceId === undefined) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      `$.nodes.${node.nodeId}.bindings`,
      "CPU preparation could not resolve derived source/destination resources",
    );
  }
  const remainingMs = remainingPreparationMs(startedAt, options);
  let prepared: PreparedViewCopyCpu;
  try {
    prepared = await prepareViewCopyCpu(
      semantic.layout,
      semantic.kernel,
      {
        operationId: node.entrypointId,
        bindings: node.dimensionBindings,
        maxElements: Math.min(
          MAX_VIEW_COPY_ELEMENTS,
          options.maxElementOperations,
        ),
        maxPreparationMs: remainingMs,
        ...(options.signal === undefined ? {} : { signal: options.signal }),
      },
    );
  } catch (cause) {
    translateSemanticFailure(
      cause,
      `$.nodes.${node.nodeId}`,
      "view-copy CPU preparation failed",
    );
  }
  return Object.freeze({
    kind: "dispatch",
    nodeId: node.nodeId,
    sourceResourceId,
    destinationResourceId,
    prepared,
    elementOperations: prepared.elementCount * BigInt(rankCount),
  });
}

function prepareCollectivePlan(
  node: HostGraphAllReduceNode,
  resources: ReadonlyMap<string, HostGraphResource>,
): CollectivePlan {
  const resource = resources.get(node.resourceId);
  if (resource === undefined) {
    fail(
      "BG-GRAPH-CPU-INTERNAL",
      `$.nodes.${node.nodeId}.resourceId`,
      "verified collective resource disappeared",
    );
  }
  const byteLength = safeNumber(
    wireIntegerToBigInt(resource.byteLength),
    `$.resources.${resource.resourceId}.byteLength`,
  );
  if (byteLength % 4 !== 0) {
    fail(
      "BG-GRAPH-CPU-UNSUPPORTED-PROFILE",
      `$.nodes.${node.nodeId}.resourceId`,
      "CPU all-reduce requires a whole number of 32-bit elements",
    );
  }
  const participants = Object.freeze(node.participants.map((participant) =>
    safeNumber(
      wireIntegerToBigInt(participant),
      `$.nodes.${node.nodeId}.participants`,
    )));
  const elementCount = byteLength / 4;
  return Object.freeze({
    kind: "all-reduce",
    nodeId: node.nodeId,
    resourceId: node.resourceId,
    reduction: node.reduction,
    dtype: node.dtype,
    participants,
    elementCount,
    elementOperations: BigInt(elementCount * participants.length),
  });
}

function prepareCopyPlan(
  node: HostGraphCopyNode,
  resources: ReadonlyMap<string, HostGraphResource>,
  rankCount: number,
): CopyPlan {
  const source = resources.get(node.sourceResourceId);
  const destination = resources.get(node.destinationResourceId);
  if (source === undefined || destination === undefined) {
    fail(
      "BG-GRAPH-CPU-INTERNAL",
      `$.nodes.${node.nodeId}`,
      "verified copy resource disappeared",
    );
  }
  const byteLength = safeNumber(
    wireIntegerToBigInt(source.byteLength),
    `$.resources.${source.resourceId}.byteLength`,
  );
  return Object.freeze({
    kind: "copy",
    nodeId: node.nodeId,
    sourceResourceId: node.sourceResourceId,
    destinationResourceId: node.destinationResourceId,
    byteLength,
    elementOperations: BigInt(byteLength) * BigInt(rankCount),
  });
}

function prepareMaterializePlan(
  node: HostGraphMaterializeNode,
  resources: ReadonlyMap<string, HostGraphResource>,
): MaterializePlan {
  const resource = resources.get(node.resourceId);
  if (resource === undefined || resource.role !== "output") {
    fail(
      "BG-GRAPH-CPU-INTERNAL",
      `$.nodes.${node.nodeId}.resourceId`,
      "verified materialization output disappeared",
    );
  }
  return Object.freeze({
    kind: "materialize",
    nodeId: node.nodeId,
    resourceId: node.resourceId,
    elementOperations: 0n,
  });
}

function executeDispatch(
  plan: DispatchPlan,
  rankResources: RankResources,
  startedAt: number,
  maxExecutionMs: number,
  signal: AbortSignal | undefined,
): void {
  for (const [rank, resources] of rankResources.entries()) {
    ensureExecutionActive(startedAt, maxExecutionMs, signal);
    const source = resources.get(plan.sourceResourceId);
    const destination = resources.get(plan.destinationResourceId);
    if (source === undefined || destination === undefined) {
      fail(
        "BG-GRAPH-CPU-INTERNAL",
        `$.nodes.${plan.nodeId}`,
        `rank ${rank} dispatch resources disappeared`,
      );
    }
    try {
      plan.prepared.execute({ source, destination });
    } catch (cause) {
      translateSemanticFailure(
        cause,
        `$.nodes.${plan.nodeId}`,
        `rank ${rank} view-copy execution failed`,
      );
    }
  }
  ensureExecutionActive(startedAt, maxExecutionMs, signal);
}

function executeCopy(
  plan: CopyPlan,
  rankResources: RankResources,
  startedAt: number,
  maxExecutionMs: number,
  signal: AbortSignal | undefined,
): void {
  for (const [rank, resources] of rankResources.entries()) {
    ensureExecutionActive(startedAt, maxExecutionMs, signal);
    const source = resources.get(plan.sourceResourceId);
    const destination = resources.get(plan.destinationResourceId);
    if (source === undefined || destination === undefined) {
      fail(
        "BG-GRAPH-CPU-INTERNAL",
        `$.nodes.${plan.nodeId}`,
        `rank ${rank} copy resources disappeared`,
      );
    }
    if (
      source.byteLength !== plan.byteLength ||
      destination.byteLength !== plan.byteLength
    ) {
      fail(
        "BG-GRAPH-CPU-INTERNAL",
        `$.nodes.${plan.nodeId}`,
        `rank ${rank} copy allocation length diverged after verification`,
      );
    }
    REFLECT_APPLY(UINT8_SET, destination, [source]);
  }
  ensureExecutionActive(startedAt, maxExecutionMs, signal);
}

async function executeAllReduce(
  plan: CollectivePlan,
  rankResources: RankResources,
  startedAt: number,
  maxExecutionMs: number,
  signal: AbortSignal | undefined,
): Promise<void> {
  const participantViews = plan.participants.map((rank) => {
    const bytes = rankResources[rank]?.get(plan.resourceId);
    if (bytes === undefined) {
      fail(
        "BG-GRAPH-CPU-INTERNAL",
        `$.nodes.${plan.nodeId}`,
        `rank ${rank} collective resource disappeared`,
      );
    }
    const slots = nativeUint8Slots(bytes, `$.nodes.${plan.nodeId}`);
    return new DATA_VIEW_CONSTRUCTOR(
      slots.buffer,
      slots.byteOffset,
      slots.byteLength,
    );
  });
  const reduced = allocateBytes(
    plan.elementCount * 4,
    `$.nodes.${plan.nodeId}.scratch`,
  );
  const reducedSlots = nativeUint8Slots(
    reduced,
    `$.nodes.${plan.nodeId}.scratch`,
  );
  const reducedView = new DATA_VIEW_CONSTRUCTOR(
    reducedSlots.buffer,
    reducedSlots.byteOffset,
    reducedSlots.byteLength,
  );
  let yieldAt = monotonicNow() + YIELD_INTERVAL_MS;
  for (let index = 0; index < plan.elementCount; index += 1) {
    if ((index & 4095) === 0) {
      ensureExecutionActive(startedAt, maxExecutionMs, signal);
      if (monotonicNow() >= yieldAt) {
        await yieldToMainThread();
        ensureExecutionActive(startedAt, maxExecutionMs, signal);
        yieldAt = monotonicNow() + YIELD_INTERVAL_MS;
      }
    }
    const byteOffset = index * 4;
    if (plan.dtype === "f32") {
      reduceF32(plan, participantViews, reducedView, byteOffset, index);
    } else if (plan.dtype === "i32") {
      reduceI32(plan, participantViews, reducedView, byteOffset);
    } else {
      reduceU32(plan, participantViews, reducedView, byteOffset);
    }
  }
  ensureExecutionActive(startedAt, maxExecutionMs, signal);
  for (const rank of plan.participants) {
    const destination = rankResources[rank]?.get(plan.resourceId);
    if (destination === undefined) {
      fail(
        "BG-GRAPH-CPU-INTERNAL",
        `$.nodes.${plan.nodeId}`,
        `rank ${rank} collective destination disappeared`,
      );
    }
    REFLECT_APPLY(UINT8_SET, destination, [reduced]);
  }
}

function reduceF32(
  plan: CollectivePlan,
  participants: readonly DataView[],
  destination: DataView,
  byteOffset: number,
  elementIndex: number,
): void {
  let accumulator = readF32(participants[0], byteOffset);
  requireFiniteCollectiveF32(
    accumulator,
    plan,
    plan.participants[0] as number,
    elementIndex,
  );
  for (let index = 1; index < participants.length; index += 1) {
    const value = readF32(participants[index], byteOffset);
    requireFiniteCollectiveF32(
      value,
      plan,
      plan.participants[index] as number,
      elementIndex,
    );
    accumulator = plan.reduction === "sum"
      ? MATH_FROUND(accumulator + value)
      : plan.reduction === "min"
        ? MATH_MIN(accumulator, value)
        : MATH_MAX(accumulator, value);
    if (!NUMBER_IS_FINITE(accumulator)) {
      fail(
        "BG-GRAPH-CPU-NUMERICAL-DOMAIN",
        `$.nodes.${plan.nodeId}.resourceId`,
        `f32 ${plan.reduction} overflowed at element ${elementIndex}`,
      );
    }
  }
  REFLECT_APPLY(DATA_VIEW_SET_FLOAT32, destination, [
    byteOffset,
    accumulator,
    true,
  ]);
}

function reduceI32(
  plan: CollectivePlan,
  participants: readonly DataView[],
  destination: DataView,
  byteOffset: number,
): void {
  let accumulator = readI32(participants[0], byteOffset);
  for (let index = 1; index < participants.length; index += 1) {
    const value = readI32(participants[index], byteOffset);
    accumulator = plan.reduction === "sum"
      ? (accumulator + value) | 0
      : plan.reduction === "min"
        ? MATH_MIN(accumulator, value)
        : MATH_MAX(accumulator, value);
  }
  REFLECT_APPLY(DATA_VIEW_SET_INT32, destination, [
    byteOffset,
    accumulator,
    true,
  ]);
}

function reduceU32(
  plan: CollectivePlan,
  participants: readonly DataView[],
  destination: DataView,
  byteOffset: number,
): void {
  let accumulator = readU32(participants[0], byteOffset);
  for (let index = 1; index < participants.length; index += 1) {
    const value = readU32(participants[index], byteOffset);
    accumulator = plan.reduction === "sum"
      ? (accumulator + value) >>> 0
      : plan.reduction === "min"
        ? MATH_MIN(accumulator, value)
        : MATH_MAX(accumulator, value);
  }
  REFLECT_APPLY(DATA_VIEW_SET_UINT32, destination, [
    byteOffset,
    accumulator,
    true,
  ]);
}

function requireFiniteCollectiveF32(
  value: number,
  plan: CollectivePlan,
  rank: number,
  elementIndex: number,
): void {
  if (!NUMBER_IS_FINITE(value)) {
    fail(
      "BG-GRAPH-CPU-NUMERICAL-DOMAIN",
      `$.nodes.${plan.nodeId}.resourceId`,
      `rank ${rank} f32 element ${elementIndex} is not finite`,
    );
  }
}

function captureExecutionRequest(
  request: HostGraphCpuExecutionRequest,
  rankCount: number,
  inputResources: readonly HostGraphResource[],
): {
  readonly inputs: readonly AdmittedInput[];
  readonly signal?: AbortSignal;
} {
  const captured = inspectPlainObject(
    request,
    ["inputs", "signal"],
    ["inputs"],
    "$.request",
  );
  const signal = captured.signal === undefined
    ? undefined
    : requireAbortSignal(captured.signal, "$.request.signal");
  const inputResourceMap = new Map(
    inputResources.map((resource) => [resource.resourceId, resource]),
  );
  const expectedBindings = rankCount * inputResources.length;
  const values = snapshotDenseArray(
    captured.inputs,
    "$.request.inputs",
    expectedBindings,
  );
  if (values.length !== expectedBindings) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      "$.request.inputs",
      `expected exactly ${expectedBindings} rank-local input bindings`,
    );
  }
  const admitted: AdmittedInput[] = [];
  const seen = new Set<string>();
  for (const [index, value] of values.entries()) {
    const path = `$.request.inputs[${index}]`;
    const binding = inspectPlainObject(
      value,
      ["rank", "resourceId", "bytes"],
      ["rank", "resourceId", "bytes"],
      path,
    );
    const rank = parseRank(binding.rank, `${path}.rank`, rankCount);
    const resourceId = stringValue(binding.resourceId, `${path}.resourceId`);
    const resource = inputResourceMap.get(resourceId);
    if (resource === undefined) {
      fail(
        "BG-GRAPH-CPU-INVALID-BINDING",
        `${path}.resourceId`,
        `resource ${resourceId} is not a graph input`,
      );
    }
    const key = `${rank}\0${resourceId}`;
    if (seen.has(key)) {
      fail(
        "BG-GRAPH-CPU-INVALID-BINDING",
        path,
        `duplicate rank ${rank} input ${resourceId}`,
      );
    }
    seen.add(key);
    const bytes = snapshotInputBytes(binding.bytes, resource, `${path}.bytes`);
    admitted.push(Object.freeze({ rank, resourceId, bytes }));
  }
  return Object.freeze({
    inputs: Object.freeze(admitted),
    ...(signal === undefined ? {} : { signal }),
  });
}

function materializeRankResources(
  rankCount: number,
  resources: readonly HostGraphResource[],
  inputs: readonly AdmittedInput[],
): RankResources {
  const inputMap = new Map(
    inputs.map((input) => [
      `${input.rank}\0${input.resourceId}`,
      input.bytes,
    ]),
  );
  return Object.freeze(Array.from({ length: rankCount }, (_, rank) => {
    const rankResources = new Map<string, Uint8Array>();
    for (const resource of resources) {
      const input = inputMap.get(`${rank}\0${resource.resourceId}`);
      const bytes = input ?? allocateBytes(
        safeNumber(
          wireIntegerToBigInt(resource.byteLength),
          `$.resources.${resource.resourceId}.byteLength`,
        ),
        `$.resources.${resource.resourceId}`,
      );
      rankResources.set(resource.resourceId, bytes);
    }
    return rankResources as ReadonlyMap<string, Uint8Array>;
  }));
}

async function buildSemanticCatalog(
  options: NormalizedPreparationOptions,
  startedAt: number,
): Promise<ReadonlyMap<string, SemanticCatalogEntry>> {
  const layouts = new Map<string, VerifiedLayoutArtifact>();
  for (const layout of options.layoutArtifacts) {
    ensurePreparationActive(startedAt, options);
    let hash: string;
    try {
      hash = await hashSemanticArtifact(layout);
    } catch (cause) {
      fail(
        "BG-GRAPH-CPU-INVALID-AUTHORITY",
        "$.options.layoutArtifacts",
        "CPU preparation received a non-verifier-issued layout artifact",
        cause,
      );
    }
    if (layouts.has(hash)) {
      fail(
        "BG-GRAPH-CPU-INVALID-BINDING",
        "$.options.layoutArtifacts",
        `duplicate layout semantic hash ${hash}`,
      );
    }
    layouts.set(hash, layout);
  }
  const catalog = new Map<string, SemanticCatalogEntry>();
  for (const kernel of options.kernelArtifacts) {
    ensurePreparationActive(startedAt, options);
    let hash: string;
    let layoutHash: string;
    try {
      hash = await hashSemanticArtifact(kernel);
      layoutHash = kernelArtifactPayload(kernel).layoutSemanticHash;
    } catch (cause) {
      fail(
        "BG-GRAPH-CPU-INVALID-AUTHORITY",
        "$.options.kernelArtifacts",
        "CPU preparation received a non-verifier-issued kernel artifact",
        cause,
      );
    }
    if (catalog.has(hash)) {
      fail(
        "BG-GRAPH-CPU-INVALID-BINDING",
        "$.options.kernelArtifacts",
        `duplicate kernel semantic hash ${hash}`,
      );
    }
    const layout = layouts.get(layoutHash);
    if (layout === undefined) {
      fail(
        "BG-GRAPH-CPU-INVALID-BINDING",
        "$.options.layoutArtifacts",
        `kernel ${hash} has no supplied verified layout ${layoutHash}`,
      );
    }
    catalog.set(hash, Object.freeze({ kernel, layout }));
  }
  return catalog;
}

function normalizePreparationOptions(
  options: HostGraphCpuPreparationOptions,
): NormalizedPreparationOptions {
  const captured = inspectPlainObject(
    options,
    [
      "kernelArtifacts",
      "layoutArtifacts",
      "maxWorkingBytes",
      "maxElementOperations",
      "maxPreparationMs",
      "maxExecutionMs",
      "signal",
    ],
    ["kernelArtifacts", "layoutArtifacts"],
    "$.options",
  );
  const signal = captured.signal === undefined
    ? undefined
    : requireAbortSignal(captured.signal, "$.options.signal");
  return Object.freeze({
    kernelArtifacts: snapshotDenseArray<VerifiedKernelArtifact>(
      captured.kernelArtifacts,
      "$.options.kernelArtifacts",
      MAX_ARTIFACTS,
    ),
    layoutArtifacts: snapshotDenseArray<VerifiedLayoutArtifact>(
      captured.layoutArtifacts,
      "$.options.layoutArtifacts",
      MAX_ARTIFACTS,
    ),
    maxWorkingBytes: positiveBudget(
      captured.maxWorkingBytes,
      DEFAULT_MAX_WORKING_BYTES,
      HOST_GRAPH_CPU_MAX_WORKING_BYTES,
      "$.options.maxWorkingBytes",
    ),
    maxElementOperations: positiveBudget(
      captured.maxElementOperations,
      DEFAULT_MAX_ELEMENT_OPERATIONS,
      HOST_GRAPH_CPU_MAX_ELEMENT_OPERATIONS,
      "$.options.maxElementOperations",
    ),
    maxPreparationMs: positiveBudget(
      captured.maxPreparationMs,
      DEFAULT_MAX_PREPARATION_MS,
      HOST_GRAPH_CPU_MAX_PREPARATION_MS,
      "$.options.maxPreparationMs",
    ),
    maxExecutionMs: positiveBudget(
      captured.maxExecutionMs,
      DEFAULT_MAX_EXECUTION_MS,
      HOST_GRAPH_CPU_MAX_EXECUTION_MS,
      "$.options.maxExecutionMs",
    ),
    ...(signal === undefined ? {} : { signal }),
  });
}

function inspectPlainObject(
  value: unknown,
  allowedFields: readonly string[],
  requiredFields: readonly string[],
  path: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null ||
      Object.getPrototypeOf(value) !== Object.prototype) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "expected a direct plain data object",
    );
  }
  const allowed = new Set(allowedFields);
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const keys = Reflect.ownKeys(descriptors);
  if (keys.some((key) => typeof key !== "string" || !allowed.has(key))) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "object contains unknown or symbolic fields",
    );
  }
  const result: Record<string, unknown> = Object.create(null) as
    Record<string, unknown>;
  for (const key of keys) {
    if (typeof key !== "string") continue;
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor) ||
        descriptor.enumerable !== true) {
      fail(
        "BG-GRAPH-CPU-INVALID-BINDING",
        `${path}.${key}`,
        "fields must be enumerable data properties without accessors",
      );
    }
    result[key] = descriptor.value;
  }
  for (const key of requiredFields) {
    if (!(key in result)) {
      fail(
        "BG-GRAPH-CPU-INVALID-BINDING",
        `${path}.${key}`,
        "required field is missing",
      );
    }
  }
  return result;
}

function snapshotDenseArray<T>(
  value: unknown,
  path: string,
  maximumLength: number,
): readonly T[] {
  if (!Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Array.prototype ||
      value.length > maximumLength) {
    fail(
      "BG-GRAPH-CPU-RESOURCE-LIMIT",
      path,
      `expected a direct array with at most ${maximumLength} entries`,
    );
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length !== value.length + 1 ||
      keys.some((key) =>
        key !== "length" &&
        (typeof key !== "string" ||
         !/^(?:0|[1-9][0-9]*)$/u.test(key) ||
         Number(key) >= value.length))) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "array must be dense and contain no named or symbolic properties",
    );
  }
  return Object.freeze(Array.from({ length: value.length }, (_, index) => {
    const descriptor = descriptors[String(index)];
    if (descriptor === undefined || !("value" in descriptor) ||
        descriptor.enumerable !== true) {
      fail(
        "BG-GRAPH-CPU-INVALID-BINDING",
        `${path}[${index}]`,
        "array entries must be enumerable data properties",
      );
    }
    return descriptor.value as T;
  }));
}

function snapshotInputBytes(
  value: unknown,
  resource: HostGraphResource,
  path: string,
): Uint8Array {
  if (!(value instanceof UINT8_ARRAY_CONSTRUCTOR) ||
      Object.getPrototypeOf(value) !== DIRECT_UINT8_PROTOTYPE) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "input bytes must be a direct Uint8Array",
    );
  }
  const slots = nativeUint8Slots(value, path);
  if (isSharedArrayBuffer(slots.buffer)) {
    fail(
      "BG-GRAPH-CPU-UNSUPPORTED-PROFILE",
      path,
      "shared input memory requires an explicit synchronization profile",
    );
  }
  const expectedLength = wireIntegerToBigInt(resource.byteLength);
  if (BigInt(slots.byteLength) !== expectedLength) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      `input length ${slots.byteLength} does not equal graph resource length ${expectedLength}`,
    );
  }
  if (slots.byteOffset % resource.alignmentBytes !== 0) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      `input byte offset does not satisfy ${resource.alignmentBytes}-byte alignment`,
    );
  }
  const snapshot = allocateBytes(slots.byteLength, path);
  REFLECT_APPLY(UINT8_SET, snapshot, [value]);
  return snapshot;
}

function nativeUint8Slots(value: Uint8Array, path: string): NativeUint8Slots {
  try {
    return Object.freeze({
      buffer: TYPED_ARRAY_BUFFER_GETTER.call(value) as ArrayBufferLike,
      byteOffset: TYPED_ARRAY_BYTE_OFFSET_GETTER.call(value) as number,
      byteLength: TYPED_ARRAY_BYTE_LENGTH_GETTER.call(value) as number,
    });
  } catch (cause) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "input does not expose native typed-array internal slots",
      cause,
    );
  }
}

function isSharedArrayBuffer(buffer: ArrayBufferLike): boolean {
  if (SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) return false;
  try {
    SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER.call(buffer);
    return true;
  } catch {
    return false;
  }
}

function allocateBytes(byteLength: number, path: string): Uint8Array {
  try {
    return new UINT8_ARRAY_CONSTRUCTOR(byteLength);
  } catch (cause) {
    fail(
      "BG-GRAPH-CPU-RESOURCE-LIMIT",
      path,
      `unable to allocate ${byteLength} private CPU bytes`,
      cause,
    );
  }
}

function parseRank(value: unknown, path: string, rankCount: number): number {
  let rank: bigint;
  try {
    rank = wireIntegerToBigInt(parseWireU64(value, path));
  } catch (cause) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "rank must be a canonical wire u64",
      cause,
    );
  }
  if (rank >= BigInt(rankCount)) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      `rank ${rank} is outside graph rank count ${rankCount}`,
    );
  }
  return Number(rank);
}

function requireAbortSignal(value: unknown, path: string): AbortSignal {
  if (ABORT_SIGNAL_CONSTRUCTOR === undefined ||
      ABORT_SIGNAL_ABORTED_GETTER === undefined ||
      typeof value !== "object" ||
      value === null ||
      Object.getPrototypeOf(value) !== ABORT_SIGNAL_CONSTRUCTOR.prototype) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "signal must be a direct native AbortSignal",
    );
  }
  try {
    ABORT_SIGNAL_ABORTED_GETTER.call(value);
  } catch (cause) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "signal does not expose native AbortSignal state",
      cause,
    );
  }
  return value as AbortSignal;
}

function ensurePreparationActive(
  startedAt: number,
  options: NormalizedPreparationOptions,
): void {
  if (options.signal !== undefined && signalAborted(options.signal)) {
    fail(
      "BG-GRAPH-CPU-ABORTED",
      "$.options.signal",
      "host graph CPU preparation was aborted",
    );
  }
  if (monotonicNow() - startedAt > options.maxPreparationMs) {
    fail(
      "BG-GRAPH-CPU-TIMEOUT",
      "$.options.maxPreparationMs",
      `host graph CPU preparation exceeded ${options.maxPreparationMs} ms`,
    );
  }
}

function ensureExecutionActive(
  startedAt: number,
  maxExecutionMs: number,
  signal: AbortSignal | undefined,
): void {
  if (signal !== undefined && signalAborted(signal)) {
    fail(
      "BG-GRAPH-CPU-ABORTED",
      "$.request.signal",
      "host graph CPU execution was aborted",
    );
  }
  if (monotonicNow() - startedAt > maxExecutionMs) {
    fail(
      "BG-GRAPH-CPU-TIMEOUT",
      "$.maxExecutionMs",
      `host graph CPU execution exceeded ${maxExecutionMs} ms`,
    );
  }
}

function signalAborted(signal: AbortSignal): boolean {
  try {
    return ABORT_SIGNAL_ABORTED_GETTER?.call(signal) === true;
  } catch (cause) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      "$.signal",
      "signal lost native AbortSignal state",
      cause,
    );
  }
}

function remainingPreparationMs(
  startedAt: number,
  options: NormalizedPreparationOptions,
): number {
  const remaining = Math.floor(
    options.maxPreparationMs - (monotonicNow() - startedAt),
  );
  if (remaining <= 0) {
    fail(
      "BG-GRAPH-CPU-TIMEOUT",
      "$.options.maxPreparationMs",
      `host graph CPU preparation exceeded ${options.maxPreparationMs} ms`,
    );
  }
  return remaining;
}

function positiveBudget(
  value: unknown,
  fallback: number,
  maximum: number,
  path: string,
): number {
  const resolved = value ?? fallback;
  if (typeof resolved !== "number" ||
      !Number.isSafeInteger(resolved) ||
      resolved <= 0 ||
      resolved > maximum) {
    fail(
      "BG-GRAPH-CPU-RESOURCE-LIMIT",
      path,
      `budget must be a positive safe integer no greater than ${maximum}`,
    );
  }
  return resolved;
}

function stringValue(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 128) {
    fail(
      "BG-GRAPH-CPU-INVALID-BINDING",
      path,
      "expected a bounded non-empty string",
    );
  }
  return value;
}

function safeNumber(value: bigint, path: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    fail(
      "BG-GRAPH-CPU-UNSUPPORTED-PROFILE",
      path,
      "value does not fit an exact JavaScript index",
    );
  }
  return Number(value);
}

function readF32(view: DataView | undefined, byteOffset: number): number {
  if (view === undefined) {
    fail(
      "BG-GRAPH-CPU-INTERNAL",
      "$.collective",
      "collective participant view disappeared",
    );
  }
  return REFLECT_APPLY(DATA_VIEW_GET_FLOAT32, view, [
    byteOffset,
    true,
  ]) as number;
}

function readI32(view: DataView | undefined, byteOffset: number): number {
  if (view === undefined) {
    fail(
      "BG-GRAPH-CPU-INTERNAL",
      "$.collective",
      "collective participant view disappeared",
    );
  }
  return REFLECT_APPLY(DATA_VIEW_GET_INT32, view, [
    byteOffset,
    true,
  ]) as number;
}

function readU32(view: DataView | undefined, byteOffset: number): number {
  if (view === undefined) {
    fail(
      "BG-GRAPH-CPU-INTERNAL",
      "$.collective",
      "collective participant view disappeared",
    );
  }
  return REFLECT_APPLY(DATA_VIEW_GET_UINT32, view, [
    byteOffset,
    true,
  ]) as number;
}

function translateSemanticFailure(
  cause: unknown,
  path: string,
  message: string,
): never {
  if (cause instanceof HostGraphCpuError) throw cause;
  if (cause instanceof SemanticSchemaError) {
    const semanticPath = cause.diagnostic.path ?? path;
    const code = cause.diagnostic.code.endsWith("-RESOURCE-LIMIT")
      ? "BG-GRAPH-CPU-RESOURCE-LIMIT"
      : cause.diagnostic.code.endsWith("-UNSUPPORTED-PROFILE")
        ? "BG-GRAPH-CPU-UNSUPPORTED-PROFILE"
        : "BG-GRAPH-CPU-INVALID-BINDING";
    fail(code, semanticPath, `${message}: ${cause.message}`, cause);
  }
  fail("BG-GRAPH-CPU-INTERNAL", path, message, cause);
}

function monotonicNow(): number {
  return PERFORMANCE_NOW?.() ?? DATE_NOW();
}

async function yieldToMainThread(): Promise<void> {
  await new Promise<void>((resolve) => SET_TIMEOUT(resolve, 0));
}

function requiredGetter(
  target: object,
  name: string,
): (this: unknown) => unknown {
  const getter = Object.getOwnPropertyDescriptor(target, name)?.get;
  if (getter === undefined) {
    throw new Error(`internal: missing typed-array ${name} getter`);
  }
  return getter;
}

function fail(
  code: HostGraphCpuErrorCode,
  path: string,
  message: string,
  cause?: unknown,
): never {
  throw new HostGraphCpuError(
    code,
    path,
    message,
    cause === undefined ? undefined : { cause },
  );
}

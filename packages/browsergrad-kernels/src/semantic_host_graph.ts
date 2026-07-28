import {
  HOST_GRAPH_FAILURE_MODEL,
  hostGraphArtifactPayload,
  prepareHostGraphProgram,
  type HostGraphAllReduceNode,
  type HostGraphCopyNode,
  type HostGraphDispatchNode,
  type HostGraphMaterializeNode,
  type HostGraphRepeatBodyNode,
  type HostGraphRepeatCompletion,
  type HostGraphRepeatNode,
  type HostGraphResource,
  type VerifiedHostGraphArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import type {
  VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  kernelArtifactPayload,
  type VerifiedKernelArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  encodeWireU64,
  hashNamedComponents,
  hashSemanticArtifact,
  SemanticSchemaError,
  wireIntegerToBigInt,
  type JsonObject,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  prepareHostGraphCollectiveWgsl,
  prepareHostGraphReplicationWgsl,
} from "./semantic_host_graph_wgsl.js";
import {
  prepareSemanticViewCopyWgsl,
  SemanticViewCopyWebGpuError,
  type PreparedSemanticViewCopyWgsl,
} from "./semantic_view_copy.js";
import type { KernelDevice } from "./types.js";
import {
  clearWgslPipelineCache,
  prepareWgslKernelProgramSequence,
  WgslPipelineCreationError,
  WgslShaderCreationError,
  type WgslKernelRunResult,
  type WgslKernelSequenceStep,
  type WgslStorageBufferMetadata,
  type WgslTypedArray,
} from "./wgsl_program.js";
import {
  issueAsyncWithWebGpuErrorScopes,
  ScopedWebGpuIssueError,
} from "./webgpu_error_scope.js";

export const SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE =
  "browsergrad.host-graph.webgpu@1" as const;
export const SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION = "1.4.0" as const;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_EXPANDED_STEPS = 16_384;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_WORKING_BYTES = 1_073_741_824;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_PREPARATION_MS = 300_000;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_EXECUTION_MS = 300_000;

const DEFAULT_WORKGROUP_SIZE = 64;
const MAX_WORKGROUP_SIZE = 256;
const DEFAULT_MAX_WGSL_BYTES = 64 * 1024;
const MAX_WGSL_BYTES = 1024 * 1024;
const DEFAULT_MAX_EXPANDED_STEPS = 4_096;
const DEFAULT_MAX_WORKING_BYTES = 256 * 1024 * 1024;
const DEFAULT_MAX_PREPARATION_MS = 30_000;
const DEFAULT_MAX_EXECUTION_MS = 30_000;
const MAX_VIEW_COPY_ELEMENTS = 16_777_216;
const NUMERICAL_STATUS_STORAGE = "bg_graph_numerical_status";
const PREPARED = new WeakMap<object, PreparedState>();
const ACTIVE_DEVICES = new WeakSet<GPUDevice>();
const LOST_DEVICES = new WeakSet<GPUDevice>();
const WATCHED_DEVICES = new WeakSet<GPUDevice>();

const UINT8_ARRAY_PROTOTYPE = Uint8Array.prototype;
const TYPED_ARRAY_PROTOTYPE =
  Object.getPrototypeOf(UINT8_ARRAY_PROTOTYPE) as object;
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
const ABORT_SIGNAL_CONSTRUCTOR =
  typeof AbortSignal === "undefined" ? undefined : AbortSignal;
const ABORT_SIGNAL_ABORTED_GETTER =
  ABORT_SIGNAL_CONSTRUCTOR === undefined
    ? undefined
    : Object.getOwnPropertyDescriptor(
      ABORT_SIGNAL_CONSTRUCTOR.prototype,
      "aborted",
    )?.get;
const UINT8_SET = Uint8Array.prototype.set;
const REFLECT_APPLY = Reflect.apply;
const DATE_NOW = Date.now;
const PERFORMANCE_NOW = globalThis.performance?.now.bind(
  globalThis.performance,
);
const HOST_IS_LITTLE_ENDIAN =
  new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;

export type SemanticHostGraphWebGpuErrorCode =
  | "BG-WEBGPU-GRAPH-INVALID-AUTHORITY"
  | "BG-WEBGPU-GRAPH-INVALID-BINDING"
  | "BG-WEBGPU-GRAPH-UNVERIFIED-PREPARED"
  | "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE"
  | "BG-WEBGPU-GRAPH-RESOURCE-LIMIT"
  | "BG-WEBGPU-GRAPH-DEVICE-LIMIT"
  | "BG-WEBGPU-GRAPH-SHADER"
  | "BG-WEBGPU-GRAPH-PIPELINE"
  | "BG-WEBGPU-GRAPH-VALIDATION"
  | "BG-WEBGPU-GRAPH-OUT-OF-MEMORY"
  | "BG-WEBGPU-GRAPH-NUMERICAL-DOMAIN"
  | "BG-WEBGPU-GRAPH-DEVICE-LOST"
  | "BG-WEBGPU-GRAPH-CANCELLED"
  | "BG-WEBGPU-GRAPH-TIMEOUT"
  | "BG-WEBGPU-GRAPH-EXECUTION"
  | "BG-WEBGPU-GRAPH-INTERNAL";

export class SemanticHostGraphWebGpuError extends Error {
  constructor(
    readonly code: SemanticHostGraphWebGpuErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SemanticHostGraphWebGpuError";
  }
}

export interface PrepareSemanticHostGraphWebGpuOptions {
  readonly kernelArtifacts: readonly VerifiedKernelArtifact[];
  readonly layoutArtifacts: readonly VerifiedLayoutArtifact[];
  readonly workgroupSize?: number;
  readonly maxWgslBytes?: number;
  readonly maxExpandedSteps?: number;
  /** Bounds private host snapshots/results plus GPU upload/readback storage. */
  readonly maxTransientWorkingSetBytes?: number;
  readonly maxPreparationMs?: number;
  readonly signal?: AbortSignal;
}

export interface SemanticHostGraphWebGpuInputBinding {
  readonly rank: WireU64;
  readonly resourceId: string;
  readonly bytes: Uint8Array;
}

export interface SemanticHostGraphWebGpuExecutionRequest {
  readonly inputs: readonly SemanticHostGraphWebGpuInputBinding[];
}

export interface SemanticHostGraphWebGpuRunOptions {
  /** Stops a queued submission or suppresses a result from submitted work. */
  readonly signal?: AbortSignal;
  /** Caller-visible budget; timed-out work owns background cleanup. */
  readonly timeoutMs?: number;
}

export interface SemanticHostGraphWebGpuOutputBinding {
  readonly rank: WireU64;
  readonly resourceId: string;
  readonly bytes: Uint8Array;
}

export interface SemanticHostGraphWebGpuDeviceFacts {
  readonly features: readonly string[];
  readonly limits: Readonly<{
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupsPerDimension: number;
    maxComputeInvocationsPerWorkgroup: number;
    maxComputeWorkgroupSizeX: number;
    maxBindingsPerBindGroup: number;
    maxStorageBuffersPerShaderStage: number;
  }>;
}

export interface PreparedSemanticHostGraphWebGpu {
  readonly profile: typeof SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE;
  readonly backendVersion:
    typeof SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION;
  readonly graphSemanticHash: string;
  readonly rankCount: WireU64;
  readonly nodeCount: number;
  readonly expandedNodeCount: number;
  readonly inputResourceIds: readonly string[];
  readonly outputResourceIds: readonly string[];
  readonly expandedStepCount: number;
  readonly dispatchStepCount: number;
  readonly copyStepCount: number;
  readonly materializationCount: number;
  readonly eventCount: number;
  readonly eventIds: readonly string[];
  readonly repeatCount: number;
  readonly repeatIterationCount: number;
  readonly collectiveReductionStepCount: number;
  readonly collectiveReplicationStepCount: number;
  readonly wgslModuleHashes: readonly string[];
  readonly plannedTransientGpuBytes: WireU64;
  readonly plannedTransientHostBytes: WireU64;
  readonly plannedTransientWorkingSetBytes: WireU64;
  readonly maxTransientWorkingSetBytes: WireU64;
}

export interface SemanticHostGraphWebGpuTrace {
  readonly profile: typeof SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE;
  readonly backendVersion:
    typeof SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION;
  readonly graphSemanticHash: string;
  readonly backendSpecializationHash: string;
  readonly failureModel: typeof HOST_GRAPH_FAILURE_MODEL;
  readonly executedNodeIds: readonly string[];
  readonly expandedStepCount: number;
  readonly dispatchStepCount: number;
  readonly copyStepCount: number;
  readonly materializationCount: number;
  readonly completedEventIds: readonly string[];
  readonly completedRepeats: readonly HostGraphRepeatCompletion[];
  readonly collectiveReductionStepCount: number;
  readonly collectiveReplicationStepCount: number;
  readonly wgslModuleHashes: readonly string[];
  readonly plannedTransientGpuBytes: WireU64;
  readonly plannedTransientHostBytes: WireU64;
  readonly plannedTransientWorkingSetBytes: WireU64;
  readonly maxTransientWorkingSetBytes: WireU64;
  readonly submitted: boolean;
  readonly device: SemanticHostGraphWebGpuDeviceFacts;
}

export interface SemanticHostGraphWebGpuExecutionResult {
  readonly outputs: readonly SemanticHostGraphWebGpuOutputBinding[];
  readonly trace: SemanticHostGraphWebGpuTrace;
}

interface NormalizedPreparationOptions {
  readonly kernelArtifacts: readonly VerifiedKernelArtifact[];
  readonly layoutArtifacts: readonly VerifiedLayoutArtifact[];
  readonly workgroupSize: number;
  readonly maxWgslBytes: number;
  readonly maxExpandedSteps: number;
  readonly maxTransientWorkingSetBytes: number;
  readonly maxPreparationMs: number;
  readonly signal?: AbortSignal;
}

interface SemanticCatalogEntry {
  readonly kernel: VerifiedKernelArtifact;
  readonly layout: VerifiedLayoutArtifact;
}

interface ResourcePlan {
  readonly resource: HostGraphResource;
  readonly byteLength: number;
  readonly storageNames: readonly string[];
}

interface PreparedState {
  readonly artifact: VerifiedHostGraphArtifact;
  readonly rankCount: number;
  readonly resources: readonly ResourcePlan[];
  readonly resourcesById: ReadonlyMap<string, ResourcePlan>;
  readonly inputs: readonly ResourcePlan[];
  readonly outputs: readonly ResourcePlan[];
  readonly steps: readonly WgslKernelSequenceStep[];
  readonly storageMetadata:
    Readonly<Record<string, WgslStorageBufferMetadata>>;
  readonly boundStorageNames: ReadonlySet<string>;
  readonly readbackStorageNames: readonly string[];
  readonly usesNumericalStatus: boolean;
  readonly topologicalNodeIds: readonly string[];
  readonly eventIds: readonly string[];
  readonly repeats: readonly HostGraphRepeatCompletion[];
  readonly maximumBoundAllocationBytes: bigint;
  readonly workgroupSize: number;
}

interface MutablePreparationCounts {
  dispatch: number;
  copy: number;
  materialization: number;
  event: number;
  reduction: number;
  replication: number;
}

interface PreparedRepeatBodyTemplate {
  readonly steps: readonly WgslKernelSequenceStep[];
  readonly counts: Readonly<MutablePreparationCounts>;
}

interface CapturedInput {
  readonly rank: number;
  readonly resourceId: string;
  readonly bytes: Uint8Array;
}

interface NativeUint8Slots {
  readonly buffer: ArrayBufferLike;
  readonly byteOffset: number;
  readonly byteLength: number;
}

/**
 * Lower one exact verifier-issued host graph to a bounded sequence of
 * canonical WebGPU programs. The returned summary is intentionally opaque:
 * execution accepts only the exact object issued by this module instance.
 */
export async function prepareSemanticHostGraphWebGpu(
  artifact: VerifiedHostGraphArtifact,
  options: PrepareSemanticHostGraphWebGpuOptions,
): Promise<PreparedSemanticHostGraphWebGpu> {
  const startedAt = monotonicNow();
  const normalized = normalizePreparationOptions(options);
  let payload: ReturnType<typeof hostGraphArtifactPayload>;
  try {
    payload = hostGraphArtifactPayload(artifact);
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-AUTHORITY",
      "$.artifact",
      "WebGPU preparation requires an exact verifier-issued host graph",
      cause,
    );
  }
  ensurePreparationActive(startedAt, normalized);
  let preparedGraph: Awaited<ReturnType<typeof prepareHostGraphProgram>>;
  try {
    preparedGraph = await prepareHostGraphProgram(artifact);
  } catch (cause) {
    translatePreparationFailure(
      cause,
      "$.artifact",
      "host graph preparation failed",
    );
  }
  const rankCount = safeNumber(preparedGraph.rankCount, "$.rankCount");
  const resources = Object.freeze(payload.program.resources.map(
    (resource, index): ResourcePlan => Object.freeze({
      resource,
      byteLength: safeNumber(
        wireIntegerToBigInt(resource.byteLength),
        `$.resources[${index}].byteLength`,
      ),
      storageNames: Object.freeze(Array.from(
        { length: rankCount },
        (_, rank) => `bg_graph_rank_${rank}_resource_${index}`,
      )),
    }),
  ));
  const resourcesById = new Map(
    resources.map((resource) => [
      resource.resource.resourceId,
      resource,
    ]),
  );
  const catalog = await buildSemanticCatalog(
    normalized,
    startedAt,
  );
  const nodes = new Map(
    payload.program.nodes.map((node) => [node.nodeId, node]),
  );
  const steps: WgslKernelSequenceStep[] = [];
  const counts = emptyPreparationCounts();
  const boundStorageNames = new Set<string>();
  const storageMetadata: Record<string, WgslStorageBufferMetadata> = {};
  const moduleHashes: string[] = [];
  let usesNumericalStatus = false;

  for (const nodeId of preparedGraph.topologicalNodeIds) {
    ensurePreparationActive(startedAt, normalized);
    const node = nodes.get(nodeId);
    if (node === undefined) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        "$.artifact",
        `prepared topological node ${nodeId} disappeared`,
      );
    }
    if (
      node.kind === "dispatch" ||
      node.kind === "all-reduce" ||
      node.kind === "copy"
    ) {
      const nodeUsesNumericalStatus = await appendExecutableNodeSteps(
        node,
        rankCount,
        catalog,
        resourcesById,
        normalized,
        startedAt,
        steps,
        counts,
        boundStorageNames,
        storageMetadata,
        moduleHashes,
      );
      usesNumericalStatus ||= nodeUsesNumericalStatus;
    } else if (node.kind === "materialize") {
      appendMaterialization(
        node,
        resourcesById,
        boundStorageNames,
        counts,
      );
    } else if (node.kind === "event") {
      appendEvent(counts);
    } else {
      const repeatUsesNumericalStatus = await appendRepeatSteps(
        node,
        rankCount,
        catalog,
        resourcesById,
        normalized,
        startedAt,
        steps,
        counts,
        boundStorageNames,
        storageMetadata,
        moduleHashes,
      );
      usesNumericalStatus ||= repeatUsesNumericalStatus;
    }
    if (steps.length > normalized.maxExpandedSteps) {
      fail(
        "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
        "$.maxExpandedSteps",
        `graph expands to more than ${normalized.maxExpandedSteps} WebGPU steps`,
      );
    }
  }
  ensurePreparationActive(startedAt, normalized);
  if (usesNumericalStatus) {
    boundStorageNames.add(NUMERICAL_STATUS_STORAGE);
    storageMetadata[NUMERICAL_STATUS_STORAGE] =
      Object.freeze({ valueType: "u32" });
  }
  const inputs = Object.freeze(
    resources.filter(({ resource }) => resource.role === "input"),
  );
  const outputResourceIds = new Set(preparedGraph.outputResourceIds);
  const outputs = Object.freeze(
    resources.filter(({ resource }) =>
      outputResourceIds.has(resource.resourceId)),
  );
  if (outputs.length !== preparedGraph.outputResourceIds.length) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.artifact",
      "prepared output resources disappeared",
    );
  }
  const readbackStorageNames = Object.freeze([
    ...outputs.flatMap((resource) => resource.storageNames)
      .filter((name) => boundStorageNames.has(name)),
    ...(usesNumericalStatus ? [NUMERICAL_STATUS_STORAGE] : []),
  ]);
  const inputBytes = inputs.reduce(
    (total, resource) =>
      total + (BigInt(resource.byteLength) * preparedGraph.rankCount),
    0n,
  );
  const outputBytes = outputs.reduce(
    (total, resource) =>
      total + (BigInt(resource.byteLength) * preparedGraph.rankCount),
    0n,
  );
  const boundResourceBytes = resources.reduce(
    (total, resource) =>
      total + (BigInt(resource.byteLength) * BigInt(
        resource.storageNames.filter((name) =>
          boundStorageNames.has(name)).length,
      )),
    0n,
  );
  const boundOutputBytes = outputs.reduce(
    (total, resource) =>
      total + (BigInt(resource.byteLength) * BigInt(
        resource.storageNames.filter((name) =>
          boundStorageNames.has(name)).length,
      )),
    0n,
  );
  const statusBytes = usesNumericalStatus ? 4n : 0n;
  const plannedTransientGpuBytes =
    boundResourceBytes + boundOutputBytes + (statusBytes * 2n);
  const plannedTransientHostBytes =
    inputBytes + boundResourceBytes + boundOutputBytes +
    outputBytes + statusBytes;
  const plannedTransientWorkingSetBytes =
    plannedTransientGpuBytes + plannedTransientHostBytes;
  if (
    plannedTransientWorkingSetBytes >
    BigInt(normalized.maxTransientWorkingSetBytes)
  ) {
    fail(
      "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      "$.maxTransientWorkingSetBytes",
      `planned private host and GPU storage requires ${
        plannedTransientWorkingSetBytes
      } bytes; limit is ${normalized.maxTransientWorkingSetBytes}`,
    );
  }
  const maximumBoundAllocationBytes = resources.reduce(
    (largest, resource) => {
      if (
        resource.storageNames.some((name) => boundStorageNames.has(name)) &&
        BigInt(resource.byteLength) > largest
      ) {
        return BigInt(resource.byteLength);
      }
      return largest;
    },
    0n,
  );
  const publicModuleHashes = Object.freeze(unique(moduleHashes));
  const repeats = Object.freeze(preparedGraph.topologicalNodeIds.flatMap(
    (nodeId): readonly HostGraphRepeatCompletion[] => {
      const node = nodes.get(nodeId);
      return node?.kind === "repeat"
        ? [Object.freeze({
            nodeId: node.nodeId,
            iterationCount: node.iterationCount,
            bodyNodeIds: Object.freeze(
              node.body.map((bodyNode) => bodyNode.nodeId),
            ),
          })]
        : [];
    },
  ));
  const prepared = Object.freeze({
    profile: SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE,
    backendVersion: SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION,
    graphSemanticHash: preparedGraph.graphSemanticHash,
    rankCount: encodeWireU64(preparedGraph.rankCount),
    nodeCount: preparedGraph.nodeCount,
    expandedNodeCount: preparedGraph.expandedNodeCount,
    inputResourceIds: Object.freeze(
      inputs.map(({ resource }) => resource.resourceId),
    ),
    outputResourceIds: Object.freeze(
      outputs.map(({ resource }) => resource.resourceId),
    ),
    expandedStepCount: steps.length,
    dispatchStepCount: counts.dispatch,
    copyStepCount: counts.copy,
    materializationCount: counts.materialization,
    eventCount: counts.event,
    eventIds: Object.freeze([...preparedGraph.eventIds]),
    repeatCount: preparedGraph.repeatCount,
    repeatIterationCount: preparedGraph.repeatIterationCount,
    collectiveReductionStepCount: counts.reduction,
    collectiveReplicationStepCount: counts.replication,
    wgslModuleHashes: publicModuleHashes,
    plannedTransientGpuBytes: encodeWireU64(plannedTransientGpuBytes),
    plannedTransientHostBytes: encodeWireU64(plannedTransientHostBytes),
    plannedTransientWorkingSetBytes: encodeWireU64(
      plannedTransientWorkingSetBytes,
    ),
    maxTransientWorkingSetBytes: encodeWireU64(
      BigInt(normalized.maxTransientWorkingSetBytes),
    ),
  });
  PREPARED.set(prepared, Object.freeze({
    artifact,
    rankCount,
    resources,
    resourcesById,
    inputs,
    outputs,
    steps: Object.freeze([...steps]),
    storageMetadata: Object.freeze({ ...storageMetadata }),
    boundStorageNames,
    readbackStorageNames,
    usesNumericalStatus,
    topologicalNodeIds: Object.freeze([
      ...preparedGraph.topologicalNodeIds,
    ]),
    eventIds: Object.freeze([...preparedGraph.eventIds]),
    repeats,
    maximumBoundAllocationBytes,
    workgroupSize: normalized.workgroupSize,
  }));
  return prepared;
}

async function appendExecutableNodeSteps(
  node: HostGraphRepeatBodyNode,
  rankCount: number,
  catalog: ReadonlyMap<string, SemanticCatalogEntry>,
  resourcesById: ReadonlyMap<string, ResourcePlan>,
  options: NormalizedPreparationOptions,
  startedAt: number,
  steps: WgslKernelSequenceStep[],
  counts: MutablePreparationCounts,
  boundStorageNames: Set<string>,
  storageMetadata: Record<string, WgslStorageBufferMetadata>,
  moduleHashes: string[],
): Promise<boolean> {
  if (node.kind === "dispatch") {
    await appendDispatchSteps(
      node,
      rankCount,
      catalog,
      resourcesById,
      options,
      startedAt,
      steps,
      counts,
      boundStorageNames,
      storageMetadata,
      moduleHashes,
    );
    return false;
  }
  if (node.kind === "all-reduce") {
    return appendCollectiveSteps(
      node,
      resourcesById,
      options,
      steps,
      counts,
      boundStorageNames,
      storageMetadata,
      moduleHashes,
    );
  }
  await appendCopySteps(
    node,
    resourcesById,
    options,
    steps,
    counts,
    boundStorageNames,
    storageMetadata,
    moduleHashes,
  );
  return false;
}

async function appendRepeatSteps(
  node: HostGraphRepeatNode,
  rankCount: number,
  catalog: ReadonlyMap<string, SemanticCatalogEntry>,
  resourcesById: ReadonlyMap<string, ResourcePlan>,
  options: NormalizedPreparationOptions,
  startedAt: number,
  steps: WgslKernelSequenceStep[],
  counts: MutablePreparationCounts,
  boundStorageNames: Set<string>,
  storageMetadata: Record<string, WgslStorageBufferMetadata>,
  moduleHashes: string[],
): Promise<boolean> {
  const iterationCount = safeNumber(
    wireIntegerToBigInt(node.iterationCount),
    `$.nodes.${node.nodeId}.iterationCount`,
  );
  const templates: PreparedRepeatBodyTemplate[] = [];
  let usesNumericalStatus = false;
  for (const bodyNode of node.body) {
    ensurePreparationActive(startedAt, options);
    const templateSteps: WgslKernelSequenceStep[] = [];
    const templateCounts = emptyPreparationCounts();
    const bodyUsesNumericalStatus = await appendExecutableNodeSteps(
      bodyNode,
      rankCount,
      catalog,
      resourcesById,
      options,
      startedAt,
      templateSteps,
      templateCounts,
      boundStorageNames,
      storageMetadata,
      moduleHashes,
    );
    usesNumericalStatus ||= bodyUsesNumericalStatus;
    templates.push(Object.freeze({
      steps: Object.freeze(templateSteps),
      counts: Object.freeze(templateCounts),
    }));
  }
  for (let iteration = 0; iteration < iterationCount; iteration += 1) {
    ensurePreparationActive(startedAt, options);
    for (const template of templates) {
      ensurePreparationActive(startedAt, options);
      if (
        steps.length + template.steps.length >
        options.maxExpandedSteps
      ) {
        fail(
          "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
          "$.maxExpandedSteps",
          `graph expands to more than ${options.maxExpandedSteps} WebGPU steps`,
        );
      }
      steps.push(...template.steps);
      addPreparationCounts(counts, template.counts);
    }
  }
  return usesNumericalStatus;
}

function emptyPreparationCounts(): MutablePreparationCounts {
  return {
    dispatch: 0,
    copy: 0,
    materialization: 0,
    event: 0,
    reduction: 0,
    replication: 0,
  };
}

function addPreparationCounts(
  destination: MutablePreparationCounts,
  source: Readonly<MutablePreparationCounts>,
): void {
  destination.dispatch += source.dispatch;
  destination.copy += source.copy;
  destination.materialization += source.materialization;
  destination.event += source.event;
  destination.reduction += source.reduction;
  destination.replication += source.replication;
}

/**
 * Execute against private rank-local buffers and publish fresh output copies
 * only after every dispatch, collective, readback, and numerical check passes.
 */
export async function runSemanticHostGraphWebGpu(
  device: KernelDevice,
  prepared: PreparedSemanticHostGraphWebGpu,
  request: SemanticHostGraphWebGpuExecutionRequest,
  options: SemanticHostGraphWebGpuRunOptions = {},
): Promise<SemanticHostGraphWebGpuExecutionResult> {
  const state = PREPARED.get(prepared as object);
  if (state === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-UNVERIFIED-PREPARED",
      "$.prepared",
      "prepared graph was not issued by this module instance",
    );
  }
  const capturedOptions = captureRunOptions(options);
  throwIfCancelled(capturedOptions.signal);
  const capturedInputs = captureExecutionRequest(request, state);
  if (!HOST_IS_LITTLE_ENDIAN) {
    fail(
      "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      "$.environment.byteOrder",
      "bit-exact word32 graph bindings require a little-endian JavaScript host",
    );
  }
  const gpu = captureGpuDevice(device);
  watchDeviceLoss(device, gpu);
  if (LOST_DEVICES.has(gpu)) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LOST",
      "$.device",
      "WebGPU device was previously lost",
    );
  }
  const deviceFacts = readAndVerifyDeviceFacts(gpu, state);
  const timeoutMs = resolvePositiveInteger(
    capturedOptions.timeoutMs,
    DEFAULT_MAX_EXECUTION_MS,
    SEMANTIC_HOST_GRAPH_WEBGPU_MAX_EXECUTION_MS,
    "$.options.timeoutMs",
  );
  const materialized = materializePrivateBuffers(state, capturedInputs);

  if (state.steps.length === 0) {
    const backendSpecializationHash = await hashBackendSpecialization(
      prepared,
      state,
      deviceFacts,
    );
    throwIfCancelled(capturedOptions.signal);
    return Object.freeze({
      outputs: collectOutputs(state, materialized.buffers, {}),
      trace: createTrace(
        prepared,
        state,
        backendSpecializationHash,
        deviceFacts,
        false,
      ),
    });
  }
  if (ACTIVE_DEVICES.has(gpu)) {
    fail(
      "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      "$.device.inFlight",
      "only one semantic host graph may be active per GPUDevice",
    );
  }
  ACTIVE_DEVICES.add(gpu);
  const execution = executeGraph(
    device,
    gpu,
    state,
    materialized.buffers,
  ).finally(() => {
    ACTIVE_DEVICES.delete(gpu);
  });
  const [result, backendSpecializationHash] = await Promise.all([
    awaitBoundedExecution(
      execution,
      capturedOptions.signal,
      timeoutMs,
    ),
    hashBackendSpecialization(prepared, state, deviceFacts),
  ]);
  if (state.usesNumericalStatus) {
    const status = result.buffers[NUMERICAL_STATUS_STORAGE];
    if (!(status instanceof Uint32Array) || status.length !== 1) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        "$.result.numericalStatus",
        "WebGPU graph returned an invalid numerical status allocation",
      );
    }
    if (status[0] !== 0) {
      fail(
        "BG-WEBGPU-GRAPH-NUMERICAL-DOMAIN",
        "$.result.numericalStatus",
        "f32 collective received a non-finite operand or produced overflow",
      );
    }
  }
  throwIfCancelled(capturedOptions.signal);
  return Object.freeze({
    outputs: collectOutputs(state, materialized.buffers, result.buffers),
    trace: createTrace(
      prepared,
      state,
      backendSpecializationHash,
      deviceFacts,
      true,
    ),
  });
}

async function appendDispatchSteps(
  node: HostGraphDispatchNode,
  rankCount: number,
  catalog: ReadonlyMap<string, SemanticCatalogEntry>,
  resourcesById: ReadonlyMap<string, ResourcePlan>,
  options: NormalizedPreparationOptions,
  startedAt: number,
  steps: WgslKernelSequenceStep[],
  counts: MutablePreparationCounts,
  boundStorageNames: Set<string>,
  storageMetadata: Record<string, WgslStorageBufferMetadata>,
  moduleHashes: string[],
): Promise<void> {
  const semantic = catalog.get(node.semanticArtifactHash);
  if (semantic === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      `$.nodes.${node.nodeId}.semanticArtifactHash`,
      "WebGPU preparation did not receive the referenced semantic artifact",
    );
  }
  const operation = kernelArtifactPayload(semantic.kernel).operations.find(
    (candidate) => candidate.operationId === node.entrypointId,
  );
  if (operation === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      `$.nodes.${node.nodeId}.entrypointId`,
      "WebGPU preparation could not resolve the verified entrypoint",
    );
  }
  const bindings = new Map(node.bindings.map((binding) => [
    binding.semanticResourceId,
    binding.graphResourceId,
  ]));
  const source = resourceBinding(
    bindings,
    resourcesById,
    operation.source.viewId,
    node.nodeId,
  );
  const destination = resourceBinding(
    bindings,
    resourcesById,
    operation.destination.viewId,
    node.nodeId,
  );
  let prepared: PreparedSemanticViewCopyWgsl;
  try {
    prepared = await prepareSemanticViewCopyWgsl(
      semantic.layout,
      semantic.kernel,
      {
        operationId: node.entrypointId,
        bindings: node.dimensionBindings,
        workgroupSize: options.workgroupSize,
        maxWgslBytes: options.maxWgslBytes,
        maxElements: MAX_VIEW_COPY_ELEMENTS,
        maxPreparationMs: Math.min(
          60_000,
          remainingPreparationMs(startedAt, options),
        ),
        maxTransientWorkingSetBytes:
          options.maxTransientWorkingSetBytes,
        ...(options.signal === undefined
          ? {}
          : { signal: options.signal }),
      },
    );
  } catch (cause) {
    translatePreparationFailure(
      cause,
      `$.nodes.${node.nodeId}`,
      "view-copy WebGPU preparation failed",
    );
  }
  if (prepared.semantic.elementCount === 0n) return;
  for (let rank = 0; rank < rankCount; rank += 1) {
    const sourceName = source.storageNames[rank] as string;
    const destinationName = destination.storageNames[rank] as string;
    steps.push(Object.freeze({
      program: prepared.program,
      launch: prepared.launch,
      storageAliases: Object.freeze({
        source_words: sourceName,
        destination_words: destinationName,
      }),
    }));
    addResourceMetadata(
      source,
      sourceName,
      boundStorageNames,
      storageMetadata,
    );
    addResourceMetadata(
      destination,
      destinationName,
      boundStorageNames,
      storageMetadata,
    );
    counts.dispatch += 1;
  }
  moduleHashes.push(prepared.wgslModuleHash);
}

async function appendCollectiveSteps(
  node: HostGraphAllReduceNode,
  resourcesById: ReadonlyMap<string, ResourcePlan>,
  options: NormalizedPreparationOptions,
  steps: WgslKernelSequenceStep[],
  counts: MutablePreparationCounts,
  boundStorageNames: Set<string>,
  storageMetadata: Record<string, WgslStorageBufferMetadata>,
  moduleHashes: string[],
): Promise<boolean> {
  const resource = resourcesById.get(node.resourceId);
  if (resource === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}.resourceId`,
      "verified collective resource disappeared",
    );
  }
  if (resource.byteLength % 4 !== 0) {
    fail(
      "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      `$.nodes.${node.nodeId}.resourceId`,
      "WebGPU all-reduce requires whole 32-bit elements",
    );
  }
  const elementCount = resource.byteLength / 4;
  if (elementCount === 0) return false;
  const participants = node.participants.map((rank, index) =>
    safeNumber(
      wireIntegerToBigInt(rank),
      `$.nodes.${node.nodeId}.participants[${index}]`,
    ));
  const accumulatorName =
    resource.storageNames[participants[0] as number] as string;
  const collective = await prepareHostGraphCollectiveWgsl(
    node.dtype,
    node.reduction,
    options.workgroupSize,
  );
  const replication = await prepareHostGraphReplicationWgsl(
    options.workgroupSize,
  );
  for (let index = 1; index < participants.length; index += 1) {
    const operandName =
      resource.storageNames[participants[index] as number] as string;
    steps.push(Object.freeze({
      program: collective.program,
      launch: frozenLaunch(elementCount),
      storageAliases: Object.freeze({
        accumulator: accumulatorName,
        operand: operandName,
        ...(collective.usesNumericalStatus
          ? { numerical_status: NUMERICAL_STATUS_STORAGE }
          : {}),
      }),
    }));
    addResourceMetadata(
      resource,
      accumulatorName,
      boundStorageNames,
      storageMetadata,
    );
    addResourceMetadata(
      resource,
      operandName,
      boundStorageNames,
      storageMetadata,
    );
    counts.reduction += 1;
  }
  for (let index = 1; index < participants.length; index += 1) {
    const destinationName =
      resource.storageNames[participants[index] as number] as string;
    steps.push(Object.freeze({
      program: replication.program,
      launch: frozenLaunch(elementCount),
      storageAliases: Object.freeze({
        source_words: accumulatorName,
        destination_words: destinationName,
      }),
    }));
    counts.replication += 1;
  }
  moduleHashes.push(collective.moduleHash, replication.moduleHash);
  return collective.usesNumericalStatus;
}

async function appendCopySteps(
  node: HostGraphCopyNode,
  resourcesById: ReadonlyMap<string, ResourcePlan>,
  options: NormalizedPreparationOptions,
  steps: WgslKernelSequenceStep[],
  counts: MutablePreparationCounts,
  boundStorageNames: Set<string>,
  storageMetadata: Record<string, WgslStorageBufferMetadata>,
  moduleHashes: string[],
): Promise<void> {
  const source = resourcesById.get(node.sourceResourceId);
  const destination = resourcesById.get(node.destinationResourceId);
  if (source === undefined || destination === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}`,
      "verified copy resource disappeared",
    );
  }
  if (
    source.byteLength !== destination.byteLength ||
    source.resource.dtype !== destination.resource.dtype
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}`,
      "verified copy resource contract diverged",
    );
  }
  if (source.byteLength === 0) return;
  if (source.byteLength % 4 !== 0) {
    fail(
      "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      `$.nodes.${node.nodeId}`,
      "portable WebGPU copy requires a whole number of 32-bit words",
    );
  }
  const replication = await prepareHostGraphReplicationWgsl(
    options.workgroupSize,
  );
  const elementCount = source.byteLength / 4;
  for (let rank = 0; rank < source.storageNames.length; rank += 1) {
    const sourceName = source.storageNames[rank] as string;
    const destinationName = destination.storageNames[rank] as string;
    steps.push(Object.freeze({
      program: replication.program,
      launch: frozenLaunch(elementCount),
      storageAliases: Object.freeze({
        source_words: sourceName,
        destination_words: destinationName,
      }),
    }));
    addRawResourceMetadata(
      sourceName,
      boundStorageNames,
      storageMetadata,
    );
    addRawResourceMetadata(
      destinationName,
      boundStorageNames,
      storageMetadata,
    );
    counts.copy += 1;
  }
  moduleHashes.push(replication.moduleHash);
}

function appendMaterialization(
  node: HostGraphMaterializeNode,
  resourcesById: ReadonlyMap<string, ResourcePlan>,
  boundStorageNames: ReadonlySet<string>,
  counts: MutablePreparationCounts,
): void {
  const resource = resourcesById.get(node.resourceId);
  if (resource === undefined || resource.resource.role !== "output") {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}.resourceId`,
      "verified materialization output disappeared",
    );
  }
  if (
    resource.byteLength > 0 &&
    resource.storageNames.some((name) => !boundStorageNames.has(name))
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}.resourceId`,
      "materialized output has no complete portable WebGPU producer",
    );
  }
  counts.materialization += 1;
}

function appendEvent(counts: MutablePreparationCounts): void {
  counts.event += 1;
}

function resourceBinding(
  bindings: ReadonlyMap<string, string>,
  resources: ReadonlyMap<string, ResourcePlan>,
  semanticResourceId: string,
  nodeId: string,
): ResourcePlan {
  const graphResourceId = bindings.get(semanticResourceId);
  const resource = graphResourceId === undefined
    ? undefined
    : resources.get(graphResourceId);
  if (resource === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      `$.nodes.${nodeId}.bindings`,
      `could not resolve semantic resource ${semanticResourceId}`,
    );
  }
  return resource;
}

function addResourceMetadata(
  resource: ResourcePlan,
  storageName: string,
  names: Set<string>,
  metadata: Record<string, WgslStorageBufferMetadata>,
): void {
  if (
    resource.resource.dtype !== "f32" &&
    resource.resource.dtype !== "i32" &&
    resource.resource.dtype !== "u32"
  ) {
    fail(
      "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      `$.resources.${resource.resource.resourceId}.dtype`,
      `portable host-graph storage does not support ${
        resource.resource.dtype
      }`,
    );
  }
  if (resource.byteLength === 0 || resource.byteLength % 4 !== 0) {
    fail(
      "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      `$.resources.${resource.resource.resourceId}.byteLength`,
      "bound portable WebGPU storage must contain whole 32-bit words",
    );
  }
  names.add(storageName);
  metadata[storageName] = Object.freeze({
    valueType: "u32",
    compatibleValueTypes: Object.freeze([resource.resource.dtype]),
  });
}

function addRawResourceMetadata(
  storageName: string,
  names: Set<string>,
  metadata: Record<string, WgslStorageBufferMetadata>,
): void {
  names.add(storageName);
  metadata[storageName] ??= Object.freeze({ valueType: "u32" });
}

function frozenLaunch(
  elementCount: number,
): Readonly<{ readonly dispatchCount: readonly [number, number, number] }> {
  return Object.freeze({
    dispatchCount: Object.freeze([elementCount, 1, 1] as const),
  });
}

async function buildSemanticCatalog(
  options: NormalizedPreparationOptions,
  startedAt: number,
): Promise<ReadonlyMap<string, SemanticCatalogEntry>> {
  const layouts = new Map<string, VerifiedLayoutArtifact>();
  for (const [index, layout] of options.layoutArtifacts.entries()) {
    ensurePreparationActive(startedAt, options);
    let hash: string;
    try {
      hash = await hashSemanticArtifact(layout);
    } catch (cause) {
      translatePreparationFailure(
        cause,
        `$.layoutArtifacts[${index}]`,
        "layout artifact is not verifier-issued",
      );
    }
    if (layouts.has(hash)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `$.layoutArtifacts[${index}]`,
        `duplicate layout semantic hash ${hash}`,
      );
    }
    layouts.set(hash, layout);
  }
  const catalog = new Map<string, SemanticCatalogEntry>();
  for (const [index, kernel] of options.kernelArtifacts.entries()) {
    ensurePreparationActive(startedAt, options);
    let hash: string;
    let layoutHash: string;
    try {
      hash = await hashSemanticArtifact(kernel);
      layoutHash = kernelArtifactPayload(kernel).layoutSemanticHash;
    } catch (cause) {
      translatePreparationFailure(
        cause,
        `$.kernelArtifacts[${index}]`,
        "kernel artifact is not verifier-issued",
      );
    }
    const layout = layouts.get(layoutHash);
    if (layout === undefined) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `$.kernelArtifacts[${index}]`,
        `missing exact layout artifact ${layoutHash}`,
      );
    }
    if (catalog.has(hash)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `$.kernelArtifacts[${index}]`,
        `duplicate kernel semantic hash ${hash}`,
      );
    }
    catalog.set(hash, Object.freeze({ kernel, layout }));
  }
  return catalog;
}

async function executeGraph(
  device: KernelDevice,
  gpu: GPUDevice,
  state: PreparedState,
  buffers: Readonly<Record<string, WgslTypedArray>>,
): Promise<WgslKernelRunResult> {
  let sequence: Awaited<
    ReturnType<typeof prepareWgslKernelProgramSequence>
  >;
  try {
    sequence = await issueAsyncWithWebGpuErrorScopes(
      gpu,
      "$.pipeline",
      () => prepareWgslKernelProgramSequence(
        device,
        state.steps,
        {
          buffers,
          storageMetadata: state.storageMetadata,
          readback: state.readbackStorageNames,
        },
      ),
      { cleanup: (prepared) => prepared.destroy() },
    );
  } catch (cause) {
    translateExecutionFailure(cause, "$.pipeline");
  }
  try {
    return await issueAsyncWithWebGpuErrorScopes(
      gpu,
      "$.dispatch",
      () => sequence.run({
        readback: state.readbackStorageNames,
        awaitCompletion: true,
      }),
    );
  } catch (cause) {
    translateExecutionFailure(cause, "$.dispatch");
  } finally {
    sequence.destroy();
  }
}

function materializePrivateBuffers(
  state: PreparedState,
  inputs: readonly CapturedInput[],
): {
  readonly buffers: Readonly<Record<string, WgslTypedArray>>;
} {
  const inputMap = new Map(inputs.map((input) => [
    `${input.rank}\0${input.resourceId}`,
    input.bytes,
  ]));
  const buffers: Record<string, WgslTypedArray> = {};
  for (const resource of state.resources) {
    for (let rank = 0; rank < state.rankCount; rank += 1) {
      const storageName = resource.storageNames[rank] as string;
      if (!state.boundStorageNames.has(storageName)) continue;
      const words = new Uint32Array(resource.byteLength / 4);
      const input = inputMap.get(
        `${rank}\0${resource.resource.resourceId}`,
      );
      if (input !== undefined) {
        REFLECT_APPLY(UINT8_SET, new Uint8Array(words.buffer), [input]);
      }
      buffers[storageName] = words;
    }
  }
  if (state.usesNumericalStatus) {
    buffers[NUMERICAL_STATUS_STORAGE] = new Uint32Array(1);
  }
  return Object.freeze({ buffers: Object.freeze(buffers) });
}

function collectOutputs(
  state: PreparedState,
  initial: Readonly<Record<string, WgslTypedArray>>,
  readbacks: Readonly<Record<string, WgslTypedArray>>,
): readonly SemanticHostGraphWebGpuOutputBinding[] {
  const outputs: SemanticHostGraphWebGpuOutputBinding[] = [];
  for (let rank = 0; rank < state.rankCount; rank += 1) {
    for (const resource of state.outputs) {
      const storageName = resource.storageNames[rank] as string;
      const value = readbacks[storageName] ?? initial[storageName];
      const bytes = value === undefined
        ? new Uint8Array(resource.byteLength)
        : copyArrayBytes(value, resource.byteLength, storageName);
      outputs.push(Object.freeze({
        rank: encodeWireU64(BigInt(rank)),
        resourceId: resource.resource.resourceId,
        bytes,
      }));
    }
  }
  return Object.freeze(outputs);
}

function copyArrayBytes(
  value: WgslTypedArray,
  expectedBytes: number,
  storageName: string,
): Uint8Array {
  if (value.byteLength !== expectedBytes) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.result",
      `readback ${storageName} has ${value.byteLength} bytes; expected ${
        expectedBytes
      }`,
    );
  }
  const bytes = new Uint8Array(expectedBytes);
  REFLECT_APPLY(UINT8_SET, bytes, [
    new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
  ]);
  return bytes;
}

function captureExecutionRequest(
  request: SemanticHostGraphWebGpuExecutionRequest,
  state: PreparedState,
): readonly CapturedInput[] {
  const object = inspectPlainObject(
    request,
    ["inputs"],
    ["inputs"],
    "$.request",
  );
  const expected = state.rankCount * state.inputs.length;
  const values = snapshotDenseArray(
    object.inputs,
    "$.request.inputs",
    expected,
  );
  if (values.length !== expected) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      "$.request.inputs",
      `expected exactly ${expected} rank-local input bindings`,
    );
  }
  const resources = new Map(state.inputs.map((resource) => [
    resource.resource.resourceId,
    resource,
  ]));
  const seen = new Set<string>();
  const inputs: CapturedInput[] = [];
  for (const [index, value] of values.entries()) {
    const path = `$.request.inputs[${index}]`;
    const binding = inspectPlainObject(
      value,
      ["rank", "resourceId", "bytes"],
      ["rank", "resourceId", "bytes"],
      path,
    );
    const rank = parseRank(binding.rank, `${path}.rank`, state.rankCount);
    const resourceId = stringValue(
      binding.resourceId,
      `${path}.resourceId`,
    );
    const resource = resources.get(resourceId);
    if (resource === undefined) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `${path}.resourceId`,
        `resource ${resourceId} is not a graph input`,
      );
    }
    const key = `${rank}\0${resourceId}`;
    if (seen.has(key)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        path,
        `duplicate rank ${rank} input ${resourceId}`,
      );
    }
    seen.add(key);
    inputs.push(Object.freeze({
      rank,
      resourceId,
      bytes: snapshotInputBytes(binding.bytes, resource, `${path}.bytes`),
    }));
  }
  return Object.freeze(inputs);
}

function snapshotInputBytes(
  value: unknown,
  resource: ResourcePlan,
  path: string,
): Uint8Array {
  let isDirectUint8Array: boolean;
  try {
    isDirectUint8Array =
      value instanceof Uint8Array &&
      Object.getPrototypeOf(value) === UINT8_ARRAY_PROTOTYPE;
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "input byte reflection failed",
      cause,
    );
  }
  if (!isDirectUint8Array) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "input bytes must be a direct Uint8Array",
    );
  }
  let slots: NativeUint8Slots;
  try {
    slots = {
      buffer: REFLECT_APPLY(
        TYPED_ARRAY_BUFFER_GETTER,
        value,
        [],
      ) as ArrayBufferLike,
      byteLength: REFLECT_APPLY(
        TYPED_ARRAY_BYTE_LENGTH_GETTER,
        value,
        [],
      ) as number,
      byteOffset: REFLECT_APPLY(
        TYPED_ARRAY_BYTE_OFFSET_GETTER,
        value,
        [],
      ) as number,
    };
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "input bytes do not expose native typed-array slots",
      cause,
    );
  }
  if (isSharedArrayBuffer(slots.buffer)) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "shared input bytes require an explicit synchronization contract",
    );
  }
  if (slots.byteLength !== resource.byteLength) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      `input has ${slots.byteLength} bytes; expected ${resource.byteLength}`,
    );
  }
  if (slots.byteOffset % resource.resource.alignmentBytes !== 0) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      `input offset must satisfy ${
        resource.resource.alignmentBytes
      }-byte alignment`,
    );
  }
  const snapshot = new Uint8Array(slots.byteLength);
  REFLECT_APPLY(UINT8_SET, snapshot, [
    new Uint8Array(slots.buffer, slots.byteOffset, slots.byteLength),
  ]);
  return snapshot;
}

function normalizePreparationOptions(
  options: PrepareSemanticHostGraphWebGpuOptions,
): NormalizedPreparationOptions {
  const object = inspectPlainObject(
    options,
    [
      "kernelArtifacts",
      "layoutArtifacts",
      "workgroupSize",
      "maxWgslBytes",
      "maxExpandedSteps",
      "maxTransientWorkingSetBytes",
      "maxPreparationMs",
      "signal",
    ],
    ["kernelArtifacts", "layoutArtifacts"],
    "$.options",
  );
  const kernelArtifacts = snapshotDenseArray(
    object.kernelArtifacts,
    "$.options.kernelArtifacts",
    256,
  ) as readonly VerifiedKernelArtifact[];
  const layoutArtifacts = snapshotDenseArray(
    object.layoutArtifacts,
    "$.options.layoutArtifacts",
    256,
  ) as readonly VerifiedLayoutArtifact[];
  const signal = object.signal === undefined
    ? undefined
    : requireAbortSignal(object.signal, "$.options.signal");
  return Object.freeze({
    kernelArtifacts,
    layoutArtifacts,
    workgroupSize: resolvePositiveInteger(
      numberOrUndefined(object.workgroupSize, "$.options.workgroupSize"),
      DEFAULT_WORKGROUP_SIZE,
      MAX_WORKGROUP_SIZE,
      "$.options.workgroupSize",
    ),
    maxWgslBytes: resolvePositiveInteger(
      numberOrUndefined(object.maxWgslBytes, "$.options.maxWgslBytes"),
      DEFAULT_MAX_WGSL_BYTES,
      MAX_WGSL_BYTES,
      "$.options.maxWgslBytes",
    ),
    maxExpandedSteps: resolvePositiveInteger(
      numberOrUndefined(
        object.maxExpandedSteps,
        "$.options.maxExpandedSteps",
      ),
      DEFAULT_MAX_EXPANDED_STEPS,
      SEMANTIC_HOST_GRAPH_WEBGPU_MAX_EXPANDED_STEPS,
      "$.options.maxExpandedSteps",
    ),
    maxTransientWorkingSetBytes: resolvePositiveInteger(
      numberOrUndefined(
        object.maxTransientWorkingSetBytes,
        "$.options.maxTransientWorkingSetBytes",
      ),
      DEFAULT_MAX_WORKING_BYTES,
      SEMANTIC_HOST_GRAPH_WEBGPU_MAX_WORKING_BYTES,
      "$.options.maxTransientWorkingSetBytes",
    ),
    maxPreparationMs: resolvePositiveInteger(
      numberOrUndefined(
        object.maxPreparationMs,
        "$.options.maxPreparationMs",
      ),
      DEFAULT_MAX_PREPARATION_MS,
      SEMANTIC_HOST_GRAPH_WEBGPU_MAX_PREPARATION_MS,
      "$.options.maxPreparationMs",
    ),
    ...(signal === undefined ? {} : { signal }),
  });
}

function captureRunOptions(
  options: SemanticHostGraphWebGpuRunOptions,
): {
  readonly signal?: AbortSignal;
  readonly timeoutMs?: number;
} {
  const object = inspectPlainObject(
    options,
    ["signal", "timeoutMs"],
    [],
    "$.options",
  );
  const signal = object.signal === undefined
    ? undefined
    : requireAbortSignal(object.signal, "$.options.signal");
  const timeoutMs = numberOrUndefined(
    object.timeoutMs,
    "$.options.timeoutMs",
  );
  return Object.freeze({
    ...(signal === undefined ? {} : { signal }),
    ...(timeoutMs === undefined ? {} : { timeoutMs }),
  });
}

function readAndVerifyDeviceFacts(
  gpu: GPUDevice,
  state: PreparedState,
): SemanticHostGraphWebGpuDeviceFacts {
  let limits: SemanticHostGraphWebGpuDeviceFacts["limits"];
  let features: readonly string[];
  try {
    limits = Object.freeze({
      maxBufferSize: gpu.limits.maxBufferSize,
      maxStorageBufferBindingSize:
        gpu.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension:
        gpu.limits.maxComputeWorkgroupsPerDimension,
      maxComputeInvocationsPerWorkgroup:
        gpu.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: gpu.limits.maxComputeWorkgroupSizeX,
      maxBindingsPerBindGroup: gpu.limits.maxBindingsPerBindGroup,
      maxStorageBuffersPerShaderStage:
        gpu.limits.maxStorageBuffersPerShaderStage,
    });
    features = Object.freeze([...gpu.features].map(String).sort());
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      "$.device",
      "kernel device does not expose native WebGPU device facts",
      cause,
    );
  }
  if (state.maximumBoundAllocationBytes > BigInt(limits.maxBufferSize)) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      "$.device.limits.maxBufferSize",
      `allocation requires ${state.maximumBoundAllocationBytes} bytes; limit is ${
        limits.maxBufferSize
      }`,
    );
  }
  if (
    state.maximumBoundAllocationBytes >
    BigInt(limits.maxStorageBufferBindingSize)
  ) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      "$.device.limits.maxStorageBufferBindingSize",
      `allocation requires ${state.maximumBoundAllocationBytes} bytes; limit is ${
        limits.maxStorageBufferBindingSize
      }`,
    );
  }
  if (
    state.workgroupSize > limits.maxComputeInvocationsPerWorkgroup ||
    state.workgroupSize > limits.maxComputeWorkgroupSizeX
  ) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      "$.workgroupSize",
      `workgroup size ${state.workgroupSize} exceeds device limits`,
    );
  }
  const requiredBindings = state.usesNumericalStatus ? 3 : 2;
  if (
    limits.maxBindingsPerBindGroup < requiredBindings ||
    limits.maxStorageBuffersPerShaderStage < requiredBindings
  ) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      "$.device.limits",
      `device cannot bind the required ${requiredBindings} storage buffers`,
    );
  }
  for (const [index, step] of state.steps.entries()) {
    const workgroups = divideRoundUp(
      BigInt(step.launch.dispatchCount[0]),
      BigInt(step.program.workgroupSize[0]),
    );
    if (workgroups > BigInt(limits.maxComputeWorkgroupsPerDimension)) {
      fail(
        "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
        `$.steps[${index}].launch`,
        `dispatch requires ${workgroups} workgroups; limit is ${
          limits.maxComputeWorkgroupsPerDimension
        }`,
      );
    }
  }
  return Object.freeze({ features, limits });
}

async function hashBackendSpecialization(
  prepared: PreparedSemanticHostGraphWebGpu,
  state: PreparedState,
  device: SemanticHostGraphWebGpuDeviceFacts,
): Promise<string> {
  return hashNamedComponents({
    profile: prepared.profile,
    backendVersion: prepared.backendVersion,
    graph: prepared.graphSemanticHash,
    steps: prepared.expandedStepCount,
    wgslModules: prepared.wgslModuleHashes,
    workgroupSize: state.workgroupSize,
    selectedFeatures: [],
    limits: device.limits as unknown as JsonObject,
  });
}

function createTrace(
  prepared: PreparedSemanticHostGraphWebGpu,
  state: PreparedState,
  backendSpecializationHash: string,
  device: SemanticHostGraphWebGpuDeviceFacts,
  submitted: boolean,
): SemanticHostGraphWebGpuTrace {
  return Object.freeze({
    profile: prepared.profile,
    backendVersion: prepared.backendVersion,
    graphSemanticHash: prepared.graphSemanticHash,
    backendSpecializationHash,
    failureModel: HOST_GRAPH_FAILURE_MODEL,
    executedNodeIds: state.topologicalNodeIds,
    expandedStepCount: prepared.expandedStepCount,
    dispatchStepCount: prepared.dispatchStepCount,
    copyStepCount: prepared.copyStepCount,
    materializationCount: prepared.materializationCount,
    completedEventIds: state.eventIds,
    completedRepeats: state.repeats,
    collectiveReductionStepCount:
      prepared.collectiveReductionStepCount,
    collectiveReplicationStepCount:
      prepared.collectiveReplicationStepCount,
    wgslModuleHashes: prepared.wgslModuleHashes,
    plannedTransientGpuBytes: prepared.plannedTransientGpuBytes,
    plannedTransientHostBytes: prepared.plannedTransientHostBytes,
    plannedTransientWorkingSetBytes:
      prepared.plannedTransientWorkingSetBytes,
    maxTransientWorkingSetBytes:
      prepared.maxTransientWorkingSetBytes,
    submitted,
    device,
  });
}

function ensurePreparationActive(
  startedAt: number,
  options: NormalizedPreparationOptions,
): void {
  throwIfCancelled(options.signal);
  if (monotonicNow() - startedAt > options.maxPreparationMs) {
    fail(
      "BG-WEBGPU-GRAPH-TIMEOUT",
      "$.options.maxPreparationMs",
      `WebGPU graph preparation exceeded ${options.maxPreparationMs}ms`,
    );
  }
}

function remainingPreparationMs(
  startedAt: number,
  options: NormalizedPreparationOptions,
): number {
  ensurePreparationActive(startedAt, options);
  return Math.max(
    1,
    Math.floor(options.maxPreparationMs - (monotonicNow() - startedAt)),
  );
}

async function awaitBoundedExecution<T>(
  execution: Promise<T>,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<T> {
  throwIfCancelled(signal);
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let abortHandler: (() => void) | undefined;
  const timeoutPromise = new Promise<never>((_resolve, reject) => {
    timeout = setTimeout(() => reject(new SemanticHostGraphWebGpuError(
      "BG-WEBGPU-GRAPH-TIMEOUT",
      "$.options.timeoutMs",
      `WebGPU graph did not finish within ${timeoutMs}ms; submitted work will clean up without publishing a stale result`,
    )), timeoutMs);
  });
  const abortPromise = signal === undefined
    ? new Promise<never>(() => undefined)
    : new Promise<never>((_resolve, reject) => {
        abortHandler = () => reject(cancelledError());
        signal.addEventListener("abort", abortHandler, { once: true });
      });
  try {
    return await Promise.race([
      execution,
      timeoutPromise,
      abortPromise,
    ]);
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
    if (signal !== undefined && abortHandler !== undefined) {
      signal.removeEventListener("abort", abortHandler);
    }
  }
}

function translatePreparationFailure(
  cause: unknown,
  path: string,
  context: string,
): never {
  if (cause instanceof SemanticHostGraphWebGpuError) throw cause;
  if (cause instanceof SemanticViewCopyWebGpuError) {
    const code = cause.code.includes("RESOURCE-LIMIT")
      ? "BG-WEBGPU-GRAPH-RESOURCE-LIMIT"
      : cause.code.includes("UNSUPPORTED-PROFILE")
        ? "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE"
        : "BG-WEBGPU-GRAPH-INVALID-BINDING";
    throw new SemanticHostGraphWebGpuError(
      code,
      `${path}${cause.path.startsWith("$") ? cause.path.slice(1) : ""}`,
      `${context}: ${cause.message}`,
      { cause },
    );
  }
  if (cause instanceof SemanticSchemaError) {
    const code = cause.diagnostic.code.endsWith("RESOURCE-LIMIT")
      ? "BG-WEBGPU-GRAPH-RESOURCE-LIMIT"
      : cause.diagnostic.code.endsWith("UNSUPPORTED-PROFILE")
        ? "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE"
        : "BG-WEBGPU-GRAPH-INVALID-AUTHORITY";
    throw new SemanticHostGraphWebGpuError(
      code,
      path,
      `${context}: ${cause.message}`,
      { cause },
    );
  }
  throw new SemanticHostGraphWebGpuError(
    "BG-WEBGPU-GRAPH-INTERNAL",
    path,
    `${context}: ${message(cause)}`,
    { cause },
  );
}

function translateExecutionFailure(cause: unknown, path: string): never {
  if (cause instanceof SemanticHostGraphWebGpuError) throw cause;
  let current: unknown = cause;
  const visited = new Set<unknown>();
  while (
    current instanceof Error &&
    !visited.has(current)
  ) {
    visited.add(current);
    if (current instanceof WgslShaderCreationError) {
      throw new SemanticHostGraphWebGpuError(
        "BG-WEBGPU-GRAPH-SHADER",
        "$.shaderModule",
        current.message,
        { cause },
      );
    }
    if (current instanceof WgslPipelineCreationError) {
      throw new SemanticHostGraphWebGpuError(
        "BG-WEBGPU-GRAPH-PIPELINE",
        "$.pipeline",
        current.message,
        { cause },
      );
    }
    current = current.cause;
  }
  if (cause instanceof ScopedWebGpuIssueError) {
    const code = cause.kind === "validation"
      ? "BG-WEBGPU-GRAPH-VALIDATION"
      : cause.kind === "out-of-memory"
        ? "BG-WEBGPU-GRAPH-OUT-OF-MEMORY"
        : cause.kind === "device-lost"
          ? "BG-WEBGPU-GRAPH-DEVICE-LOST"
          : cause.kind === "internal" ||
              cause.kind === "error-scope"
            ? "BG-WEBGPU-GRAPH-INTERNAL"
            : "BG-WEBGPU-GRAPH-EXECUTION";
    throw new SemanticHostGraphWebGpuError(
      code,
      cause.path,
      cause.message,
      { cause },
    );
  }
  throw new SemanticHostGraphWebGpuError(
    "BG-WEBGPU-GRAPH-EXECUTION",
    path,
    message(cause),
    { cause },
  );
}

function captureGpuDevice(device: KernelDevice): GPUDevice {
  let gpu: GPUDevice;
  try {
    gpu = device.gpu;
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      "$.device",
      "kernel device does not expose a GPUDevice",
      cause,
    );
  }
  if (gpu === null || typeof gpu !== "object") {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      "$.device",
      "kernel device does not expose a GPUDevice",
    );
  }
  return gpu;
}

function watchDeviceLoss(device: KernelDevice, gpu: GPUDevice): void {
  if (WATCHED_DEVICES.has(gpu)) return;
  let lost: Promise<GPUDeviceLostInfo>;
  try {
    lost = gpu.lost;
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      "$.device.lost",
      "GPUDevice does not expose its loss promise",
      cause,
    );
  }
  WATCHED_DEVICES.add(gpu);
  void lost.then(
    () => invalidateLostDevice(device, gpu),
    () => invalidateLostDevice(device, gpu),
  ).catch(() => undefined);
}

function invalidateLostDevice(device: KernelDevice, gpu: GPUDevice): void {
  LOST_DEVICES.add(gpu);
  try {
    clearWgslPipelineCache(device);
  } catch {
    // Device loss remains authoritative even if cache teardown is unavailable.
  }
  try {
    device.clearCache();
  } catch {
    // A lost device is unusable; cache cleanup must not create an unhandled task.
  }
}

function inspectPlainObject(
  value: unknown,
  allowed: readonly string[],
  required: readonly string[],
  path: string,
): Readonly<Record<string, unknown>> {
  let isArray: boolean;
  try {
    isArray = Array.isArray(value);
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "object reflection failed",
      cause,
    );
  }
  if (value === null || typeof value !== "object" || isArray) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "value must be a plain object",
    );
  }
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = Object.getPrototypeOf(value);
    descriptors = Object.getOwnPropertyDescriptors(value);
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "object reflection failed",
      cause,
    );
  }
  if (prototype !== Object.prototype && prototype !== null) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "value must have Object.prototype or null prototype",
    );
  }
  const allowedSet = new Set(allowed);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== "string" || !allowedSet.has(key)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        path,
        `unknown field ${String(key)}`,
      );
    }
    const descriptor = descriptors[key] as PropertyDescriptor;
    if (!("value" in descriptor)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `${path}.${key}`,
        "accessor properties are not accepted",
      );
    }
  }
  for (const key of required) {
    if (!Object.prototype.hasOwnProperty.call(descriptors, key)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `${path}.${key}`,
        "required field is missing",
      );
    }
  }
  return Object.freeze(Object.fromEntries(
    Object.entries(descriptors).map(([key, descriptor]) => [
      key,
      descriptor.value,
    ]),
  ));
}

function snapshotDenseArray(
  value: unknown,
  path: string,
  maximum: number,
): readonly unknown[] {
  let isDirectArray: boolean;
  try {
    isDirectArray =
      Array.isArray(value) &&
      Object.getPrototypeOf(value) === Array.prototype;
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "array reflection failed",
      cause,
    );
  }
  if (!isDirectArray) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "value must be a direct dense Array",
    );
  }
  let descriptors: PropertyDescriptorMap;
  try {
    descriptors = Object.getOwnPropertyDescriptors(
      value,
    ) as unknown as PropertyDescriptorMap;
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "array reflection failed",
      cause,
    );
  }
  const lengthDescriptor = descriptors.length;
  const length = lengthDescriptor?.value;
  if (
    !Number.isSafeInteger(length) ||
    length < 0 ||
    length > maximum
  ) {
    fail(
      "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      path,
      `array length must not exceed ${maximum}`,
    );
  }
  const result: unknown[] = [];
  for (let index = 0; index < length; index += 1) {
    const descriptor = descriptors[String(index)];
    if (descriptor === undefined || !("value" in descriptor)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `${path}[${index}]`,
        "array must be dense and contain data properties",
      );
    }
    result.push(descriptor.value);
  }
  const expectedKeys = new Set([
    "length",
    ...Array.from({ length }, (_, index) => String(index)),
  ]);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== "string" || !expectedKeys.has(key)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        path,
        "array must not contain extra properties",
      );
    }
  }
  return Object.freeze(result);
}

function requireAbortSignal(value: unknown, path: string): AbortSignal {
  let isNativeAbortSignal: boolean;
  try {
    isNativeAbortSignal =
      ABORT_SIGNAL_CONSTRUCTOR !== undefined &&
      value instanceof ABORT_SIGNAL_CONSTRUCTOR &&
      Object.getPrototypeOf(value) === ABORT_SIGNAL_CONSTRUCTOR.prototype;
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "signal reflection failed",
      cause,
    );
  }
  if (!isNativeAbortSignal) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "signal must be a native AbortSignal",
    );
  }
  try {
    REFLECT_APPLY(ABORT_SIGNAL_ABORTED_GETTER as () => boolean, value, []);
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "signal does not expose native AbortSignal slots",
      cause,
    );
  }
  return value as AbortSignal;
}

function throwIfCancelled(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  let aborted: boolean;
  try {
    aborted = REFLECT_APPLY(
      ABORT_SIGNAL_ABORTED_GETTER as () => boolean,
      signal,
      [],
    ) as boolean;
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      "$.signal",
      "signal does not expose native AbortSignal slots",
      cause,
    );
  }
  if (aborted) throw cancelledError();
}

function cancelledError(): SemanticHostGraphWebGpuError {
  return new SemanticHostGraphWebGpuError(
    "BG-WEBGPU-GRAPH-CANCELLED",
    "$.signal",
    "WebGPU graph was cancelled; submitted work will not publish a stale result",
  );
}

function parseRank(
  value: unknown,
  path: string,
  rankCount: number,
): number {
  let rank: bigint;
  try {
    rank = wireIntegerToBigInt(value as WireU64);
  } catch (cause) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "rank must be a canonical unsigned wire integer",
      cause,
    );
  }
  if (rank >= BigInt(rankCount)) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      `rank ${rank} is outside graph rank count ${rankCount}`,
    );
  }
  return Number(rank);
}

function numberOrUndefined(value: unknown, path: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "number") {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "value must be a number",
    );
  }
  return value;
}

function stringValue(value: unknown, path: string): string {
  if (typeof value !== "string") {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      path,
      "value must be a string",
    );
  }
  return value;
}

function resolvePositiveInteger(
  value: number | undefined,
  fallback: number,
  maximum: number,
  path: string,
): number {
  const resolved = value ?? fallback;
  if (
    !Number.isSafeInteger(resolved) ||
    resolved <= 0 ||
    resolved > maximum
  ) {
    fail(
      "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      path,
      `value must be a positive safe integer no greater than ${maximum}`,
    );
  }
  return resolved;
}

function safeNumber(value: bigint, path: string): number {
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    fail(
      "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      path,
      "value exceeds JavaScript safe integer range",
    );
  }
  return Number(value);
}

function divideRoundUp(value: bigint, divisor: bigint): bigint {
  return value === 0n ? 0n : ((value - 1n) / divisor) + 1n;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function isSharedArrayBuffer(value: ArrayBufferLike): boolean {
  if (SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER === undefined) return false;
  try {
    REFLECT_APPLY(
      SHARED_ARRAY_BUFFER_BYTE_LENGTH_GETTER,
      value,
      [],
    );
    return true;
  } catch {
    return false;
  }
}

function requiredGetter(
  target: object,
  property: string,
): (this: unknown) => unknown {
  const getter = Object.getOwnPropertyDescriptor(target, property)?.get;
  if (getter === undefined) {
    throw new Error(`missing intrinsic getter ${property}`);
  }
  return getter;
}

function monotonicNow(): number {
  return PERFORMANCE_NOW?.() ?? DATE_NOW();
}

function message(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}

function fail(
  code: SemanticHostGraphWebGpuErrorCode,
  path: string,
  messageText: string,
  cause?: unknown,
): never {
  throw new SemanticHostGraphWebGpuError(
    code,
    path,
    messageText,
    cause === undefined ? undefined : { cause },
  );
}

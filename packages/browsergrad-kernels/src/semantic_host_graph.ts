import {
  HOST_GRAPH_FAILURE_MODEL,
  hostGraphArtifactPayload,
  prepareHostGraphProgram,
  type HostGraphAllReduceNode,
  type HostGraphConditionalCompletion,
  type HostGraphConditionalNode,
  type HostGraphCopyNode,
  type HostGraphDynamicDispatchCompletion,
  type HostGraphDynamicDispatchNode,
  type HostGraphDispatchNode,
  type HostGraphExecutableNode,
  type HostGraphMaterializeNode,
  type HostGraphNode,
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
  parseWireU64,
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
import {
  prepareSemanticViewCopyDynamicWgsl,
  type PreparedSemanticViewCopyDynamicWgsl,
} from "./semantic_view_copy_internal.js";
import type { KernelDevice } from "./types.js";
import {
  clearWgslPipelineCache,
  createWgslStorageBuffer,
  destroyWgslStorageBuffer,
  prepareWgslKernelPipelineSet,
  prepareWgslKernelProgramSequence,
  readWgslStorageBuffer,
  WgslPipelineCreationError,
  WgslPipelineSetResourceLimitError,
  WgslShaderCreationError,
  WGSL_PREPARED_PIPELINE_SET_MAX_PIPELINES,
  type WgslKernelRunResult,
  type WgslKernelPipelineAlternative,
  type WgslKernelSequenceStep,
  type WgslPreparedPipelineSet,
  type WgslResidentBuffer,
  type WgslStorageBufferMetadata,
  type WgslTypedArray,
} from "./wgsl_program.js";
import {
  issueAsyncWithWebGpuErrorScopes,
  issueWithWebGpuErrorScopes,
  ScopedWebGpuIssueError,
} from "./webgpu_error_scope.js";

export const SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE =
  "browsergrad.host-graph.webgpu@1" as const;
export const SEMANTIC_HOST_GRAPH_WEBGPU_PIPELINE_PROFILE =
  "browsergrad.host-graph.webgpu-pipeline@1" as const;
export const SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION = "1.24.0" as const;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_EXPANDED_STEPS = 16_384;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_WORKING_BYTES = 1_073_741_824;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_PREPARATION_MS = 300_000;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_EXECUTION_MS = 300_000;
export const SEMANTIC_HOST_GRAPH_WEBGPU_MAX_PIPELINES =
  WGSL_PREPARED_PIPELINE_SET_MAX_PIPELINES;

const DEFAULT_WORKGROUP_SIZE = 64;
const MAX_WORKGROUP_SIZE = 256;
const DEFAULT_MAX_WGSL_BYTES = 64 * 1024;
const MAX_WGSL_BYTES = 1024 * 1024;
const DEFAULT_MAX_EXPANDED_STEPS = 4_096;
const DEFAULT_MAX_WORKING_BYTES = 256 * 1024 * 1024;
const DEFAULT_MAX_PREPARATION_MS = 30_000;
const DEFAULT_MAX_EXECUTION_MS = 30_000;
const MAX_VIEW_COPY_ELEMENTS = 16_777_216;
const MAX_U32 = 0xffff_ffffn;
const NUMERICAL_STATUS_STORAGE = "bg_graph_numerical_status";
const PREPARED = new WeakMap<object, PreparedState>();
const PREPARED_PIPELINES =
  new WeakMap<object, PreparedPipelineState>();
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
const DATA_VIEW_CONSTRUCTOR = DataView;
const DATA_VIEW_GET_UINT32 = DataView.prototype.getUint32;
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
  | "BG-WEBGPU-GRAPH-UNVERIFIED-PIPELINE"
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

export interface SemanticHostGraphWebGpuControlBinding {
  readonly controlId: string;
  readonly value: WireU64;
}

export interface SemanticHostGraphWebGpuExecutionRequest {
  readonly inputs: readonly SemanticHostGraphWebGpuInputBinding[];
  readonly controls?: readonly SemanticHostGraphWebGpuControlBinding[];
}

export interface SemanticHostGraphWebGpuRunOptions {
  /** Stops a queued submission or suppresses a result from submitted work. */
  readonly signal?: AbortSignal;
  /** Caller-visible budget; timed-out work owns background cleanup. */
  readonly timeoutMs?: number;
}

export interface PrepareSemanticHostGraphWebGpuPipelineOptions {
  readonly signal?: AbortSignal;
  readonly timeoutMs?: number;
  readonly maxPipelineCount?: number;
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
    maxUniformBuffersPerShaderStage: number;
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
  readonly runtimeRepeatCount: number;
  readonly resourceRepeatCount: number;
  readonly dynamicDispatchCount: number;
  readonly resourceDynamicDispatchCount: number;
  readonly conditionalCount: number;
  readonly conditionalNodeIds: readonly string[];
  readonly resourceConditionalCount: number;
  readonly midGraphFeedbackCount: number;
  readonly runtimeControlIds: readonly string[];
  readonly collectiveReductionStepCount: number;
  readonly collectiveReplicationStepCount: number;
  readonly wgslModuleHashes: readonly string[];
  readonly plannedTransientGpuBytes: WireU64;
  readonly plannedTransientHostBytes: WireU64;
  readonly plannedTransientWorkingSetBytes: WireU64;
  readonly maxTransientWorkingSetBytes: WireU64;
}

export interface PreparedSemanticHostGraphWebGpuPipeline {
  readonly profile:
    typeof SEMANTIC_HOST_GRAPH_WEBGPU_PIPELINE_PROFILE;
  readonly backendVersion:
    typeof SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION;
  readonly graphSemanticHash: string;
  readonly pipelineIdentityHash: string;
  readonly stepCount: number;
  readonly pipelineCount: number;
  readonly maxPipelineCount: number;
  readonly numericalPolicies: readonly string[];
  readonly device: SemanticHostGraphWebGpuDeviceFacts;
}

export interface SemanticHostGraphWebGpuTrace {
  readonly profile: typeof SEMANTIC_HOST_GRAPH_WEBGPU_PROFILE;
  readonly backendVersion:
    typeof SEMANTIC_HOST_GRAPH_WEBGPU_BACKEND_VERSION;
  readonly graphSemanticHash: string;
  readonly pipelineIdentityHash: string;
  readonly backendSpecializationHash: string;
  readonly failureModel: typeof HOST_GRAPH_FAILURE_MODEL;
  readonly executedNodeIds: readonly string[];
  readonly expandedStepCount: number;
  readonly dispatchStepCount: number;
  readonly copyStepCount: number;
  readonly materializationCount: number;
  readonly completedEventIds: readonly string[];
  readonly completedRepeats: readonly HostGraphRepeatCompletion[];
  readonly completedDynamicDispatches:
    readonly HostGraphDynamicDispatchCompletion[];
  readonly completedConditionals: readonly HostGraphConditionalCompletion[];
  readonly midGraphFeedbackCount: number;
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
  readonly deviceAdmissionSteps: readonly WgslKernelSequenceStep[];
  readonly pipelineAlternatives:
    readonly WgslKernelPipelineAlternative[];
  readonly conditionals: readonly PreparedConditionalPlan[];
  readonly resourceConditional?: PreparedConditionalPlan;
  readonly runtimeRepeats: readonly PreparedRuntimeRepeatPlan[];
  readonly resourceRepeat?: PreparedResourceRepeatPlan;
  readonly runtimeRepeatLimits: ReadonlyMap<string, number>;
  readonly dynamicDispatches: readonly PreparedDynamicDispatchPlan[];
  readonly resourceDynamicDispatch?: PreparedDynamicDispatchPlan;
  readonly dynamicDispatchLimits: ReadonlyMap<string, number>;
  readonly runtimeControlIds: readonly string[];
  readonly maximumCounts: Readonly<MutablePreparationCounts>;
  readonly storageMetadata:
    Readonly<Record<string, WgslStorageBufferMetadata>>;
  readonly boundStorageNames: ReadonlySet<string>;
  readonly readbackStorageNames: readonly string[];
  readonly usesNumericalStatus: boolean;
  readonly topologicalNodeIds: readonly string[];
  readonly eventIds: readonly string[];
  readonly fixedRepeats: readonly HostGraphRepeatCompletion[];
  readonly maximumBoundAllocationBytes: bigint;
  readonly workgroupSize: number;
  readonly numericalPolicies: readonly string[];
}

interface PreparedPipelineState {
  readonly prepared: PreparedSemanticHostGraphWebGpu;
  readonly state: PreparedState;
  readonly device: KernelDevice;
  readonly gpu: GPUDevice;
  readonly deviceFacts: SemanticHostGraphWebGpuDeviceFacts;
  readonly pipelineSet: WgslPreparedPipelineSet;
  readonly pipelineIdentityHash: string;
  destroyed: boolean;
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

interface PreparedRuntimeRepeatPlan {
  readonly nodeId: string;
  readonly controlId: string;
  readonly maxIterationCount: number;
  readonly startStepIndex: number;
  readonly stepCountPerIteration: number;
  readonly countsPerIteration: Readonly<MutablePreparationCounts>;
  readonly bodyNodeIds: readonly string[];
}

interface PreparedResourceRepeatPlan {
  readonly nodeId: string;
  readonly resourceId: string;
  readonly rank: number;
  readonly maxIterationCount: number;
  readonly startStepIndex: number;
  readonly stepCountPerIteration: number;
  readonly countsPerIteration: Readonly<MutablePreparationCounts>;
  readonly bodyNodeIds: readonly string[];
}

interface PreparedDynamicDispatchPlanBase {
  readonly nodeId: string;
  readonly startStepIndex: number;
  readonly stepCount: number;
  readonly dynamicUniformName: string;
  readonly dynamicUniformHostBytes: 4 | 16 | 32;
}

interface PreparedLinearDynamicDispatchPlan
  extends PreparedDynamicDispatchPlanBase {
  readonly kind: "linear-prefix";
  readonly selector:
    | Readonly<{
        readonly kind: "runtime-control";
        readonly controlId: string;
      }>
    | Readonly<{
        readonly kind: "resource";
        readonly resourceId: string;
        readonly rank: number;
      }>;
  readonly maxElementCount: number;
}

interface PreparedRectangularDynamicDispatchPlan
  extends PreparedDynamicDispatchPlanBase {
  readonly kind: "rectangular-prefix";
  readonly selector:
    | Readonly<{
        readonly kind: "runtime-control";
        readonly controls: readonly Readonly<{
          readonly axis: number;
          readonly controlId: string;
          readonly maxExtent: number;
        }>[];
      }>
    | Readonly<{
        readonly kind: "resource";
        readonly sources: readonly Readonly<{
          readonly axis: number;
          readonly resourceId: string;
          readonly rank: number;
          readonly maxExtent: number;
        }>[];
      }>;
  readonly maxExtents: readonly number[];
  readonly maxElementCount: number;
}

type PreparedDynamicDispatchPlan =
  | PreparedLinearDynamicDispatchPlan
  | PreparedRectangularDynamicDispatchPlan;

type DynamicLaunchSelection =
  | Readonly<{
      readonly kind: "linear-prefix";
      readonly elementCount: number;
    }>
  | Readonly<{
      readonly kind: "rectangular-prefix";
      readonly logicalExtents: readonly number[];
      readonly elementCount: number;
    }>;

interface PreparedConditionalBranch {
  readonly steps: readonly WgslKernelSequenceStep[];
  readonly counts: Readonly<MutablePreparationCounts>;
  readonly bodyNodeIds: readonly string[];
}

interface PreparedConditionalPlan {
  readonly nodeId: string;
  readonly predicate:
    | Readonly<{
        kind: "input";
        resourceId: string;
        rank: number;
      }>
    | Readonly<{
        kind: "runtime-control";
        controlId: string;
      }>
    | Readonly<{
        kind: "resource";
        resourceId: string;
        rank: number;
      }>;
  readonly startStepIndex: number;
  readonly stepCount: number;
  readonly thenBranch: PreparedConditionalBranch;
  readonly elseBranch: PreparedConditionalBranch;
}

interface SelectedExecution {
  readonly steps: readonly WgslKernelSequenceStep[];
  readonly pipelineStepIndices: readonly number[];
  readonly dispatchStepCount: number;
  readonly copyStepCount: number;
  readonly collectiveReductionStepCount: number;
  readonly collectiveReplicationStepCount: number;
  readonly completedRepeats: readonly HostGraphRepeatCompletion[];
  readonly completedDynamicDispatches:
    readonly HostGraphDynamicDispatchCompletion[];
  readonly completedConditionals: readonly HostGraphConditionalCompletion[];
  readonly resourceConditional?: PreparedConditionalPlan;
  readonly resourceRepeat?: PreparedResourceRepeatPlan;
  readonly resourceDynamicDispatch?: PreparedDynamicDispatchPlan;
}

interface ExecutedGraph {
  readonly result: WgslKernelRunResult;
  readonly selectedExecution: SelectedExecution;
}

interface CapturedInput {
  readonly rank: number;
  readonly resourceId: string;
  readonly bytes: Uint8Array;
}

interface CapturedControl {
  readonly controlId: string;
  readonly value: number;
}

interface CapturedExecutionRequest {
  readonly inputs: readonly CapturedInput[];
  readonly controls: readonly CapturedControl[];
}

interface NativeUint8Slots {
  readonly buffer: ArrayBufferLike;
  readonly byteOffset: number;
  readonly byteLength: number;
}

function collectNodeNumericalPolicies(
  node: HostGraphNode,
): readonly string[] {
  if (node.kind === "all-reduce") return [node.numericalPolicy];
  if (node.kind === "repeat") {
    return node.body.flatMap((bodyNode) =>
      bodyNode.kind === "all-reduce"
        ? [bodyNode.numericalPolicy]
        : []
    );
  }
  if (node.kind === "conditional") {
    return [...node.thenBody, ...node.elseBody].flatMap((bodyNode) =>
      bodyNode.kind === "all-reduce"
        ? [bodyNode.numericalPolicy]
        : []
    );
  }
  return [];
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
  const conditionals: PreparedConditionalPlan[] = [];
  const runtimeRepeats: PreparedRuntimeRepeatPlan[] = [];
  const resourceRepeats: PreparedResourceRepeatPlan[] = [];
  const dynamicDispatches: PreparedDynamicDispatchPlan[] = [];
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
    } else if (node.kind === "dynamic-dispatch") {
      await appendDynamicDispatchSteps(
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
        dynamicDispatches,
      );
    } else if (node.kind === "materialize") {
      appendMaterialization(
        node,
        resourcesById,
        boundStorageNames,
        counts,
      );
    } else if (node.kind === "event") {
      appendEvent(counts);
    } else if (node.kind === "repeat") {
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
        runtimeRepeats,
        resourceRepeats,
      );
      usesNumericalStatus ||= repeatUsesNumericalStatus;
    } else {
      const conditionalUsesNumericalStatus = await appendConditionalSteps(
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
        conditionals,
      );
      usesNumericalStatus ||= conditionalUsesNumericalStatus;
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
  const resourceConditionals = conditionals.filter((conditional) =>
    conditional.predicate.kind === "resource");
  if (
    resourceConditionals.length !==
    preparedGraph.resourceConditionalCount
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.artifact",
      "prepared resource conditional count diverged",
    );
  }
  const resourceConditional = resourceConditionals[0];
  if (resourceRepeats.length !== preparedGraph.resourceRepeatCount) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.artifact",
      "prepared resource repeat count diverged",
    );
  }
  const resourceRepeat = resourceRepeats[0];
  const resourceDynamicDispatches = dynamicDispatches.filter((
    dispatch,
  ) => dispatch.selector.kind === "resource");
  if (
    resourceDynamicDispatches.length !==
    preparedGraph.resourceDynamicDispatchCount
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.artifact",
      "prepared resource dynamic dispatch count diverged",
    );
  }
  const resourceDynamicDispatch = resourceDynamicDispatches[0];
  if (
    [
      resourceConditional,
      resourceRepeat,
      resourceDynamicDispatch,
    ].filter((item) => item !== undefined).length > 1
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.artifact",
      "verified graph contains more than one resource feedback node",
    );
  }
  const feedbackBytes = resourceDynamicDispatch?.kind ===
      "rectangular-prefix"
    ? BigInt(resourceDynamicDispatch.selector.kind === "resource"
      ? resourceDynamicDispatch.selector.sources.length * 4
      : 0)
    : resourceConditional === undefined &&
        resourceRepeat === undefined &&
        resourceDynamicDispatch === undefined
      ? 0n
      : 4n;
  const dynamicUniformGpuBytes =
    dynamicDispatches.reduce(
      (total, dispatch) =>
        total +
        (BigInt(dispatch.stepCount) *
          alignedDynamicUniformGpuBytes(dispatch.dynamicUniformHostBytes)),
      0n,
    );
  const dynamicUniformHostBytes =
    dynamicDispatches.reduce(
      (total, dispatch) =>
        total +
        (BigInt(dispatch.stepCount) *
          BigInt(dispatch.dynamicUniformHostBytes)),
      0n,
    );
  const plannedTransientGpuBytes =
    boundResourceBytes + boundOutputBytes + (statusBytes * 2n) +
    feedbackBytes + dynamicUniformGpuBytes;
  const plannedTransientHostBytes =
    inputBytes + boundResourceBytes + boundOutputBytes +
    outputBytes + statusBytes + feedbackBytes +
    dynamicUniformHostBytes;
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
  const conditionalNodeIds = Object.freeze(
    conditionals.map((conditional) => conditional.nodeId),
  );
  const deviceAdmissionSteps = Object.freeze([
    ...steps,
    ...conditionals.flatMap((conditional) =>
      conditional.elseBranch.steps),
  ]);
  const pipelineAlternatives = Object.freeze(
    conditionals.flatMap((conditional) =>
      conditional.elseBranch.steps.map((step, index) =>
        Object.freeze({
          stepIndex: conditional.startStepIndex + index,
          program: step.program,
        })
      )
    ),
  );
  const numericalPolicies = Object.freeze(unique(
    payload.program.nodes.flatMap(collectNodeNumericalPolicies),
  ).sort());
  const fixedRepeats = Object.freeze(preparedGraph.topologicalNodeIds.flatMap(
    (nodeId): readonly HostGraphRepeatCompletion[] => {
      const node = nodes.get(nodeId);
      return node?.kind === "repeat" &&
          node.mode === "fixed-count-sequential"
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
    runtimeRepeatCount: preparedGraph.runtimeRepeatCount,
    resourceRepeatCount: preparedGraph.resourceRepeatCount,
    dynamicDispatchCount: preparedGraph.dynamicDispatchCount,
    resourceDynamicDispatchCount:
      preparedGraph.resourceDynamicDispatchCount,
    conditionalCount: preparedGraph.conditionalCount,
    conditionalNodeIds,
    resourceConditionalCount: preparedGraph.resourceConditionalCount,
    midGraphFeedbackCount:
      preparedGraph.resourceConditionalCount +
      preparedGraph.resourceRepeatCount +
      preparedGraph.resourceDynamicDispatchCount,
    runtimeControlIds: Object.freeze([
      ...preparedGraph.runtimeControlIds,
    ]),
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
    deviceAdmissionSteps,
    pipelineAlternatives,
    conditionals: Object.freeze([...conditionals]),
    ...(resourceConditional === undefined
      ? {}
      : { resourceConditional }),
    ...(resourceRepeat === undefined ? {} : { resourceRepeat }),
    ...(resourceDynamicDispatch === undefined
      ? {}
      : { resourceDynamicDispatch }),
    runtimeRepeats: Object.freeze([...runtimeRepeats]),
    runtimeRepeatLimits: runtimeRepeatControlLimits(runtimeRepeats),
    dynamicDispatches: Object.freeze([...dynamicDispatches]),
    dynamicDispatchLimits:
      dynamicDispatchControlLimits(dynamicDispatches),
    runtimeControlIds: Object.freeze([
      ...preparedGraph.runtimeControlIds,
    ]),
    storageMetadata: Object.freeze({ ...storageMetadata }),
    boundStorageNames,
    readbackStorageNames,
    usesNumericalStatus,
    topologicalNodeIds: Object.freeze([
      ...preparedGraph.topologicalNodeIds,
    ]),
    eventIds: Object.freeze([...preparedGraph.eventIds]),
    fixedRepeats,
    maximumCounts: Object.freeze({ ...counts }),
    maximumBoundAllocationBytes,
    workgroupSize: normalized.workgroupSize,
    numericalPolicies,
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
  runtimeRepeats: PreparedRuntimeRepeatPlan[],
  resourceRepeats: PreparedResourceRepeatPlan[],
): Promise<boolean> {
  const iterationCount = safeNumber(
    wireIntegerToBigInt(
      node.mode === "fixed-count-sequential"
        ? node.iterationCount
        : node.maxIterationCount,
    ),
    node.mode === "fixed-count-sequential"
      ? `$.nodes.${node.nodeId}.iterationCount`
      : `$.nodes.${node.nodeId}.maxIterationCount`,
  );
  const startStepIndex = steps.length;
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
  if (node.mode !== "fixed-count-sequential") {
    const countsPerIteration = emptyPreparationCounts();
    for (const template of templates) {
      addPreparationCounts(countsPerIteration, template.counts);
    }
    const shared = {
      nodeId: node.nodeId,
      maxIterationCount: iterationCount,
      startStepIndex,
      stepCountPerIteration: templates.reduce(
        (total, template) => total + template.steps.length,
        0,
      ),
      countsPerIteration: Object.freeze(countsPerIteration),
      bodyNodeIds: Object.freeze(
        node.body.map((bodyNode) => bodyNode.nodeId),
      ),
    };
    if (node.mode === "runtime-u32-count-sequential") {
      runtimeRepeats.push(Object.freeze({
        ...shared,
        controlId: node.iterationControl.controlId,
      }));
    } else {
      resourceRepeats.push(Object.freeze({
        ...shared,
        resourceId: node.iterationSource.resourceId,
        rank: safeNumber(
          wireIntegerToBigInt(node.iterationSource.rank),
          `$.nodes.${node.nodeId}.iterationSource.rank`,
        ),
      }));
    }
  }
  return usesNumericalStatus;
}

async function appendConditionalSteps(
  node: HostGraphConditionalNode,
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
  conditionals: PreparedConditionalPlan[],
): Promise<boolean> {
  const thenPrepared = await prepareConditionalBranch(
    node.thenBody,
    rankCount,
    catalog,
    resourcesById,
    options,
    startedAt,
    boundStorageNames,
    storageMetadata,
    moduleHashes,
  );
  const elsePrepared = await prepareConditionalBranch(
    node.elseBody,
    rankCount,
    catalog,
    resourcesById,
    options,
    startedAt,
    boundStorageNames,
    storageMetadata,
    moduleHashes,
  );
  requireEqualConditionalBranchShape(
    node,
    thenPrepared.branch,
    elsePrepared.branch,
  );
  if (
    steps.length + thenPrepared.branch.steps.length >
    options.maxExpandedSteps
  ) {
    fail(
      "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      "$.maxExpandedSteps",
      `graph expands to more than ${options.maxExpandedSteps} WebGPU steps`,
    );
  }
  const startStepIndex = steps.length;
  if (
    node.mode === "resource-u32-branch-sequential" &&
    startStepIndex === 0
  ) {
    fail(
      "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      `$.nodes.${node.nodeId}.predicate`,
      "resource conditional requires at least one lowered producer step before feedback",
    );
  }
  steps.push(...thenPrepared.branch.steps);
  addPreparationCounts(counts, thenPrepared.branch.counts);
  conditionals.push(Object.freeze({
    nodeId: node.nodeId,
    predicate: node.mode === "runtime-u32-branch-sequential"
      ? Object.freeze({
          kind: "runtime-control" as const,
          controlId: node.predicate.controlId,
        })
      : Object.freeze({
          kind: node.mode === "input-u32-branch-sequential"
            ? "input" as const
            : "resource" as const,
          resourceId: node.predicate.resourceId,
          rank: safeNumber(
            wireIntegerToBigInt(node.predicate.rank),
            `$.nodes.${node.nodeId}.predicate.rank`,
          ),
        }),
    startStepIndex,
    stepCount: thenPrepared.branch.steps.length,
    thenBranch: thenPrepared.branch,
    elseBranch: elsePrepared.branch,
  }));
  return thenPrepared.usesNumericalStatus ||
    elsePrepared.usesNumericalStatus;
}

async function prepareConditionalBranch(
  body: readonly HostGraphExecutableNode[],
  rankCount: number,
  catalog: ReadonlyMap<string, SemanticCatalogEntry>,
  resourcesById: ReadonlyMap<string, ResourcePlan>,
  options: NormalizedPreparationOptions,
  startedAt: number,
  boundStorageNames: Set<string>,
  storageMetadata: Record<string, WgslStorageBufferMetadata>,
  moduleHashes: string[],
): Promise<Readonly<{
  branch: PreparedConditionalBranch;
  usesNumericalStatus: boolean;
}>> {
  const steps: WgslKernelSequenceStep[] = [];
  const counts = emptyPreparationCounts();
  let usesNumericalStatus = false;
  for (const bodyNode of body) {
    ensurePreparationActive(startedAt, options);
    const bodyUsesNumericalStatus = await appendExecutableNodeSteps(
      bodyNode,
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
    usesNumericalStatus ||= bodyUsesNumericalStatus;
  }
  return Object.freeze({
    branch: Object.freeze({
      steps: Object.freeze(steps),
      counts: Object.freeze(counts),
      bodyNodeIds: Object.freeze(body.map((bodyNode) => bodyNode.nodeId)),
    }),
    usesNumericalStatus,
  });
}

function requireEqualConditionalBranchShape(
  node: HostGraphConditionalNode,
  thenBranch: PreparedConditionalBranch,
  elseBranch: PreparedConditionalBranch,
): void {
  if (
    !equalPreparationCounts(thenBranch.counts, elseBranch.counts) ||
    thenBranch.steps.length !== elseBranch.steps.length ||
    thenBranch.steps.some((step, index) =>
      !equalStepExecutionShape(step, elseBranch.steps[index]))
  ) {
    fail(
      "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      `$.nodes.${node.nodeId}`,
      "WebGPU conditional branches require equal lowered step and execution shapes",
    );
  }
}

function equalPreparationCounts(
  left: Readonly<MutablePreparationCounts>,
  right: Readonly<MutablePreparationCounts>,
): boolean {
  return left.dispatch === right.dispatch &&
    left.copy === right.copy &&
    left.materialization === right.materialization &&
    left.event === right.event &&
    left.reduction === right.reduction &&
    left.replication === right.replication;
}

function equalStepExecutionShape(
  left: WgslKernelSequenceStep,
  right: WgslKernelSequenceStep | undefined,
): boolean {
  return right !== undefined &&
    left.launch.dispatchCount.every((value, index) =>
      value === right.launch.dispatchCount[index]) &&
    left.program.workgroupSize.every((value, index) =>
      value === right.program.workgroupSize[index]) &&
    left.program.bindings.length === right.program.bindings.length &&
    left.program.bindings.every((binding, index) => {
      const other = right.program.bindings[index];
      return other !== undefined &&
        binding.kind === other.kind &&
        binding.binding === other.binding;
    });
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
 * Bind an exact verifier-issued graph to one admitted GPUDevice and compile
 * every pipeline reachable through its bounded control-flow envelope.
 */
export async function prepareSemanticHostGraphWebGpuPipeline(
  device: KernelDevice,
  prepared: PreparedSemanticHostGraphWebGpu,
  options: PrepareSemanticHostGraphWebGpuPipelineOptions = {},
): Promise<PreparedSemanticHostGraphWebGpuPipeline> {
  const state = PREPARED.get(prepared as object);
  if (state === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-UNVERIFIED-PREPARED",
      "$.prepared",
      "prepared graph was not issued by this module instance",
    );
  }
  const capturedOptions = capturePipelineOptions(options);
  throwIfCancelled(capturedOptions.signal);
  return preparePipelineAuthority(
    device,
    prepared,
    state,
    capturedOptions,
  );
}

export function destroySemanticHostGraphWebGpuPipeline(
  preparedPipeline: PreparedSemanticHostGraphWebGpuPipeline,
): void {
  const state = PREPARED_PIPELINES.get(preparedPipeline as object);
  if (state === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-UNVERIFIED-PIPELINE",
      "$.preparedPipeline",
      "prepared pipeline was not issued by this module instance",
    );
  }
  if (state.destroyed) return;
  state.destroyed = true;
  state.pipelineSet.destroy();
}

/**
 * Convenience execution path. It captures caller-owned inputs before device
 * access, prepares an ephemeral pipeline authority through the same public
 * contract, and then delegates to the authority-bound executor.
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
  const captured = captureExecutionRequest(request, state);
  const selectedExecution = selectConditionalExecution(
    state,
    captured,
  );
  const preparedPipeline = await preparePipelineAuthority(
    device,
    prepared,
    state,
    capturedOptions,
  );
  const pipelineState = PREPARED_PIPELINES.get(
    preparedPipeline as object,
  )!;
  try {
    return await executeCapturedHostGraph(
      prepared,
      pipelineState,
      captured,
      selectedExecution,
      capturedOptions,
    );
  } finally {
    destroySemanticHostGraphWebGpuPipeline(preparedPipeline);
  }
}

/**
 * Execute against private rank-local buffers using an explicit pipeline
 * authority. Fresh outputs are published only after every dispatch,
 * collective, readback, and numerical check passes.
 */
export async function runSemanticHostGraphWebGpuPipeline(
  preparedPipeline: PreparedSemanticHostGraphWebGpuPipeline,
  request: SemanticHostGraphWebGpuExecutionRequest,
  options: SemanticHostGraphWebGpuRunOptions = {},
): Promise<SemanticHostGraphWebGpuExecutionResult> {
  const pipelineState = PREPARED_PIPELINES.get(
    preparedPipeline as object,
  );
  if (pipelineState === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-UNVERIFIED-PIPELINE",
      "$.preparedPipeline",
      "prepared pipeline was not issued by this module instance",
    );
  }
  if (pipelineState.destroyed) {
    fail(
      "BG-WEBGPU-GRAPH-UNVERIFIED-PIPELINE",
      "$.preparedPipeline",
      "prepared pipeline authority has been destroyed",
    );
  }
  const capturedOptions = captureRunOptions(options);
  throwIfCancelled(capturedOptions.signal);
  const captured = captureExecutionRequest(
    request,
    pipelineState.state,
  );
  const selectedExecution = selectConditionalExecution(
    pipelineState.state,
    captured,
  );
  return executeCapturedHostGraph(
    pipelineState.prepared,
    pipelineState,
    captured,
    selectedExecution,
    capturedOptions,
  );
}

async function preparePipelineAuthority(
  device: KernelDevice,
  prepared: PreparedSemanticHostGraphWebGpu,
  state: PreparedState,
  options: {
    readonly signal?: AbortSignal;
    readonly timeoutMs?: number;
    readonly maxPipelineCount?: number;
  },
): Promise<PreparedSemanticHostGraphWebGpuPipeline> {
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
    options.timeoutMs,
    DEFAULT_MAX_PREPARATION_MS,
    SEMANTIC_HOST_GRAPH_WEBGPU_MAX_PREPARATION_MS,
    "$.options.timeoutMs",
  );
  const maxPipelineCount = resolvePositiveInteger(
    options.maxPipelineCount,
    SEMANTIC_HOST_GRAPH_WEBGPU_MAX_PIPELINES,
    SEMANTIC_HOST_GRAPH_WEBGPU_MAX_PIPELINES,
    "$.options.maxPipelineCount",
  );
  let pipelineIdentityHash: string;
  try {
    pipelineIdentityHash = await hashPipelineIdentity(
      prepared,
      state,
      deviceFacts,
      maxPipelineCount,
    );
    throwIfCancelled(options.signal);
  } catch (cause) {
    translateExecutionFailure(cause, "$.pipeline.identity");
  }
  const preparation = issueAsyncWithWebGpuErrorScopes(
    gpu,
    "$.pipeline",
    () => prepareWgslKernelPipelineSet(
      device,
      state.steps,
      state.pipelineAlternatives,
      pipelineIdentityHash,
      maxPipelineCount,
    ),
    { cleanup: (pipelineSet) => pipelineSet.destroy() },
  );
  let pipelineSet: WgslPreparedPipelineSet;
  try {
    pipelineSet = await awaitBoundedExecution(
      preparation,
      options.signal,
      timeoutMs,
    );
  } catch (cause) {
    void preparation.then(
      (latePipelineSet) => latePipelineSet.destroy(),
      () => undefined,
    );
    translateExecutionFailure(cause, "$.pipeline");
  }
  try {
    throwIfCancelled(options.signal);
    const publicPipeline = Object.freeze({
      profile: SEMANTIC_HOST_GRAPH_WEBGPU_PIPELINE_PROFILE,
      backendVersion: prepared.backendVersion,
      graphSemanticHash: prepared.graphSemanticHash,
      pipelineIdentityHash,
      stepCount: state.steps.length,
      pipelineCount: pipelineSet.pipelineCount,
      maxPipelineCount,
      numericalPolicies: state.numericalPolicies,
      device: deviceFacts,
    });
    PREPARED_PIPELINES.set(publicPipeline, {
      prepared,
      state,
      device,
      gpu,
      deviceFacts,
      pipelineSet,
      pipelineIdentityHash,
      destroyed: false,
    });
    return publicPipeline;
  } catch (cause) {
    pipelineSet.destroy();
    translateExecutionFailure(cause, "$.pipeline");
  }
}

async function executeCapturedHostGraph(
  prepared: PreparedSemanticHostGraphWebGpu,
  pipelineState: PreparedPipelineState,
  captured: CapturedExecutionRequest,
  selectedExecution: SelectedExecution,
  options: {
    readonly signal?: AbortSignal;
    readonly timeoutMs?: number;
  },
): Promise<SemanticHostGraphWebGpuExecutionResult> {
  if (pipelineState.destroyed) {
    fail(
      "BG-WEBGPU-GRAPH-UNVERIFIED-PIPELINE",
      "$.preparedPipeline",
      "prepared pipeline authority has been destroyed",
    );
  }
  const {
    device,
    gpu,
    deviceFacts,
    pipelineIdentityHash,
    pipelineSet,
    state,
  } = pipelineState;
  if (LOST_DEVICES.has(gpu)) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LOST",
      "$.device",
      "WebGPU device was previously lost",
    );
  }
  const timeoutMs = resolvePositiveInteger(
    options.timeoutMs,
    DEFAULT_MAX_EXECUTION_MS,
    SEMANTIC_HOST_GRAPH_WEBGPU_MAX_EXECUTION_MS,
    "$.options.timeoutMs",
  );
  const materialized = materializePrivateBuffers(state, captured.inputs);

  if (selectedExecution.steps.length === 0) {
    const backendSpecializationHash = await hashBackendSpecialization(
      pipelineIdentityHash,
      selectedExecution,
    );
    throwIfCancelled(options.signal);
    return Object.freeze({
      outputs: collectOutputs(state, materialized.buffers, {}),
      trace: createTrace(
        prepared,
        state,
        pipelineIdentityHash,
        backendSpecializationHash,
        deviceFacts,
        false,
        selectedExecution,
        selectedExecution.completedConditionals,
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
  const execution = (
    selectedExecution.resourceConditional === undefined &&
      selectedExecution.resourceRepeat === undefined &&
      selectedExecution.resourceDynamicDispatch === undefined
      ? executeGraph(
        device,
        gpu,
        state,
        selectedExecution.steps,
        selectedExecution.pipelineStepIndices,
        materialized.buffers,
        pipelineSet,
      ).then((result) => Object.freeze({
        result,
        selectedExecution,
      }))
      : executeGraphWithResourceFeedback(
        device,
        gpu,
        state,
        selectedExecution,
        materialized.buffers,
        pipelineSet,
        options.signal,
      )
  ).finally(() => {
    ACTIVE_DEVICES.delete(gpu);
  });
  const outcome = await awaitBoundedExecution(
    execution,
    options.signal,
    timeoutMs,
  );
  const { result } = outcome;
  const backendSpecializationHash = await hashBackendSpecialization(
    pipelineIdentityHash,
    outcome.selectedExecution,
  );
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
  throwIfCancelled(options.signal);
  return Object.freeze({
    outputs: collectOutputs(state, materialized.buffers, result.buffers),
    trace: createTrace(
      prepared,
      state,
      pipelineIdentityHash,
      backendSpecializationHash,
      deviceFacts,
      true,
      outcome.selectedExecution,
      outcome.selectedExecution.completedConditionals,
    ),
  });
}

function selectConditionalExecution(
  state: PreparedState,
  captured: CapturedExecutionRequest,
): SelectedExecution {
  const inputMap = new Map(captured.inputs.map((input) => [
    `${input.rank}\0${input.resourceId}`,
    input.bytes,
  ]));
  const controlMap = new Map(captured.controls.map((control) => [
    control.controlId,
    control.value,
  ]));
  const steps = [...state.steps];
  const completedConditionals: HostGraphConditionalCompletion[] = [];
  for (const conditional of state.conditionals) {
    if (conditional.predicate.kind === "resource") continue;
    const predicateValue = conditional.predicate.kind === "input"
      ? readCapturedInputPredicate(
          conditional.nodeId,
          conditional.predicate,
          inputMap,
        )
      : controlMap.get(conditional.predicate.controlId);
    if (predicateValue === undefined) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${conditional.nodeId}.predicate`,
        "verified runtime conditional control disappeared",
      );
    }
    const selectedBranch = predicateValue === 0
      ? "else" as const
      : "then" as const;
    const branch = selectedBranch === "then"
      ? conditional.thenBranch
      : conditional.elseBranch;
    steps.splice(
      conditional.startStepIndex,
      conditional.stepCount,
      ...branch.steps,
    );
    completedConditionals.push(Object.freeze({
      nodeId: conditional.nodeId,
      selectedBranch,
      bodyNodeIds: branch.bodyNodeIds,
    }));
  }
  const selectedSteps: WgslKernelSequenceStep[] = [];
  const pipelineStepIndices: number[] = [];
  const repeatCompletions = new Map(
    state.fixedRepeats.map((completion) => [
      completion.nodeId,
      completion,
    ]),
  );
  const selectedCounts = { ...state.maximumCounts };
  let cursor = 0;
  for (const repeat of state.runtimeRepeats) {
    const iterationCount = controlMap.get(repeat.controlId);
    if (iterationCount === undefined) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${repeat.nodeId}.iterationControl`,
        "verified runtime repeat control disappeared",
      );
    }
    if (iterationCount > repeat.maxIterationCount) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${repeat.nodeId}.iterationControl`,
        "admitted runtime repeat control exceeds its verified bound",
      );
    }
    appendSelectedStepRange(
      steps,
      cursor,
      repeat.startStepIndex,
      selectedSteps,
      pipelineStepIndices,
    );
    const selectedRepeatEnd = repeat.startStepIndex +
      (iterationCount * repeat.stepCountPerIteration);
    appendSelectedStepRange(
      steps,
      repeat.startStepIndex,
      selectedRepeatEnd,
      selectedSteps,
      pipelineStepIndices,
    );
    cursor = repeat.startStepIndex +
      (repeat.maxIterationCount * repeat.stepCountPerIteration);
    const skippedIterations = repeat.maxIterationCount - iterationCount;
    selectedCounts.dispatch -=
      repeat.countsPerIteration.dispatch * skippedIterations;
    selectedCounts.copy -=
      repeat.countsPerIteration.copy * skippedIterations;
    selectedCounts.reduction -=
      repeat.countsPerIteration.reduction * skippedIterations;
    selectedCounts.replication -=
      repeat.countsPerIteration.replication * skippedIterations;
    repeatCompletions.set(repeat.nodeId, Object.freeze({
      nodeId: repeat.nodeId,
      iterationCount: encodeWireU64(BigInt(iterationCount)),
      bodyNodeIds: repeat.bodyNodeIds,
    }));
  }
  appendSelectedStepRange(
    steps,
    cursor,
    steps.length,
    selectedSteps,
    pipelineStepIndices,
  );
  const selectedIndexByPipelineIndex = new Map(
    pipelineStepIndices.map((pipelineIndex, selectedIndex) => [
      pipelineIndex,
      selectedIndex,
    ]),
  );
  const dynamicCompletions = new Map<
    string,
    HostGraphDynamicDispatchCompletion
  >();
  for (const dispatch of state.dynamicDispatches) {
    if (dispatch.selector.kind === "resource") continue;
    const selection = selectRuntimeDynamicLaunch(dispatch, controlMap);
    for (
      let pipelineIndex = dispatch.startStepIndex;
      pipelineIndex < dispatch.startStepIndex + dispatch.stepCount;
      pipelineIndex += 1
    ) {
      const selectedIndex = selectedIndexByPipelineIndex.get(
        pipelineIndex,
      );
      const step = selectedIndex === undefined
        ? undefined
        : selectedSteps[selectedIndex];
      if (selectedIndex === undefined || step === undefined) {
        fail(
          "BG-WEBGPU-GRAPH-INTERNAL",
          `$.nodes.${dispatch.nodeId}`,
          "dynamic dispatch step disappeared during runtime selection",
        );
      }
      selectedSteps[selectedIndex] = dynamicDispatchStep(
        step,
        dispatch,
        selection,
      );
    }
    dynamicCompletions.set(
      dispatch.nodeId,
      dynamicDispatchCompletion(dispatch.nodeId, selection),
    );
  }
  const completedRepeats = Object.freeze(
    state.topologicalNodeIds.flatMap((nodeId) => {
      const completion = repeatCompletions.get(nodeId);
      return completion === undefined ? [] : [completion];
    }),
  );
  const completedDynamicDispatches = Object.freeze(
    state.topologicalNodeIds.flatMap((nodeId) => {
      const completion = dynamicCompletions.get(nodeId);
      return completion === undefined ? [] : [completion];
    }),
  );
  return Object.freeze({
    steps: Object.freeze(selectedSteps),
    pipelineStepIndices: Object.freeze(pipelineStepIndices),
    dispatchStepCount: selectedCounts.dispatch,
    copyStepCount: selectedCounts.copy,
    collectiveReductionStepCount: selectedCounts.reduction,
    collectiveReplicationStepCount: selectedCounts.replication,
    completedRepeats,
    completedDynamicDispatches,
    completedConditionals: Object.freeze(completedConditionals),
    ...(state.resourceConditional === undefined
      ? {}
      : { resourceConditional: state.resourceConditional }),
    ...(state.resourceRepeat === undefined
      ? {}
      : { resourceRepeat: state.resourceRepeat }),
    ...(state.resourceDynamicDispatch === undefined
      ? {}
      : { resourceDynamicDispatch: state.resourceDynamicDispatch }),
  });
}

function appendSelectedStepRange(
  source: readonly WgslKernelSequenceStep[],
  start: number,
  end: number,
  selectedSteps: WgslKernelSequenceStep[],
  pipelineStepIndices: number[],
): void {
  for (let index = start; index < end; index += 1) {
    const step = source[index];
    if (step === undefined) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        "$.steps",
        "runtime control selected a step outside the prepared maximum graph",
      );
    }
    selectedSteps.push(step);
    pipelineStepIndices.push(index);
  }
}

function readCapturedInputPredicate(
  nodeId: string,
  predicate: Readonly<{
    kind: "input";
    resourceId: string;
    rank: number;
  }>,
  inputs: ReadonlyMap<string, Uint8Array>,
): number {
  const bytes = inputs.get(`${predicate.rank}\0${predicate.resourceId}`);
  if (bytes === undefined || bytes.byteLength !== 4) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${nodeId}.predicate`,
      "verified conditional predicate input disappeared",
    );
  }
  const view = new DATA_VIEW_CONSTRUCTOR(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength,
  );
  return REFLECT_APPLY(DATA_VIEW_GET_UINT32, view, [0, true]);
}

async function appendDispatchSteps(
  node: HostGraphDispatchNode | HostGraphDynamicDispatchNode,
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
  dynamicSelection?: DynamicLaunchSelection,
): Promise<PreparedSemanticViewCopyDynamicWgsl | undefined> {
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
  let prepared:
    | PreparedSemanticViewCopyWgsl
    | PreparedSemanticViewCopyDynamicWgsl;
  try {
    const request = {
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
    };
    prepared = node.kind === "dynamic-dispatch"
      ? await prepareSemanticViewCopyDynamicWgsl(
          semantic.layout,
          semantic.kernel,
          request,
          node.mode === "runtime-u32-rectangular-prefix" ||
              node.mode === "resource-u32-rectangular-prefix"
            ? "rectangular-prefix"
            : "linear-prefix",
        )
      : await prepareSemanticViewCopyWgsl(
          semantic.layout,
          semantic.kernel,
          request,
        );
  } catch (cause) {
    translatePreparationFailure(
      cause,
      `$.nodes.${node.nodeId}`,
      "view-copy WebGPU preparation failed",
    );
  }
  if (prepared.semantic.elementCount === 0n) return undefined;
  const selectedElementCount = dynamicSelection?.elementCount ??
    safeNumber(
      prepared.semantic.elementCount,
      `$.nodes.${node.nodeId}.launch`,
    );
  if (
    selectedElementCount <= 0 ||
    BigInt(selectedElementCount) > prepared.semantic.elementCount
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}.launch`,
      "verified dispatch launch is outside its prepared semantic domain",
    );
  }
  const launch = dynamicSelection === undefined
    ? prepared.launch
    : dynamicLaunch(dynamicSelection);
  const dynamicPrepared = node.kind === "dynamic-dispatch"
    ? preparedDynamicViewCopy(prepared, node.nodeId)
    : undefined;
  const dynamicUniform = dynamicPrepared === undefined
    ? undefined
    : dynamicUniformBinding(
        dynamicPrepared,
        dynamicSelection ??
          fail(
            "BG-WEBGPU-GRAPH-INTERNAL",
            `$.nodes.${node.nodeId}.launch`,
            "dynamic dispatch launch selection disappeared",
          ),
      );
  for (let rank = 0; rank < rankCount; rank += 1) {
    const sourceName = source.storageNames[rank] as string;
    const destinationName = destination.storageNames[rank] as string;
    steps.push(Object.freeze({
      program: prepared.program,
      launch,
      storageAliases: Object.freeze({
        source_words: sourceName,
        destination_words: destinationName,
      }),
      ...(dynamicUniform === undefined
        ? {}
        : { uniforms: dynamicUniform }),
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
  return dynamicPrepared;
}

async function appendDynamicDispatchSteps(
  node: HostGraphDynamicDispatchNode,
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
  dynamicDispatches: PreparedDynamicDispatchPlan[],
): Promise<void> {
  const maxSelection: DynamicLaunchSelection =
    node.mode === "runtime-u32-rectangular-prefix" ||
      node.mode === "resource-u32-rectangular-prefix"
      ? rectangularDynamicSelection(
          node.maxExtents.map((extent, axis) =>
            safeNumber(
              wireIntegerToBigInt(extent),
              `$.nodes.${node.nodeId}.maxExtents[${axis}]`,
            )),
          `$.nodes.${node.nodeId}.maxExtents`,
        )
      : Object.freeze({
          kind: "linear-prefix" as const,
          elementCount: safeNumber(
            wireIntegerToBigInt(node.maxElementCount),
            `$.nodes.${node.nodeId}.maxElementCount`,
          ),
        });
  const startStepIndex = steps.length;
  const dynamicPrepared = await appendDispatchSteps(
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
    maxSelection,
  );
  const stepCount = steps.length - startStepIndex;
  if (stepCount !== rankCount) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}`,
      "dynamic dispatch did not lower to one exact step per rank",
    );
  }
  if (dynamicPrepared === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}`,
      "dynamic dispatch did not lower a runtime launch uniform",
    );
  }
  const common = {
    nodeId: node.nodeId,
    startStepIndex,
    stepCount,
    dynamicUniformName: dynamicPrepared.dynamicUniformName,
    dynamicUniformHostBytes: dynamicPrepared.dynamicUniformByteLength,
  };
  if (
    node.mode === "runtime-u32-rectangular-prefix" ||
    node.mode === "resource-u32-rectangular-prefix"
  ) {
    if (
      dynamicPrepared.dynamicLaunchMode !== "rectangular-prefix" ||
      maxSelection.kind !== "rectangular-prefix"
    ) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${node.nodeId}`,
        "rectangular dynamic dispatch received a linear prepared program",
      );
    }
    dynamicDispatches.push(Object.freeze({
      ...common,
      kind: "rectangular-prefix",
      selector: node.mode === "runtime-u32-rectangular-prefix"
        ? Object.freeze({
            kind: "runtime-control" as const,
            controls: Object.freeze(node.launchControls.map((control) =>
              Object.freeze({
                axis: control.axis,
                controlId: control.controlId,
                maxExtent:
                  maxSelection.logicalExtents[control.axis] as number,
              }))),
          })
        : Object.freeze({
            kind: "resource" as const,
            sources: Object.freeze(node.launchSources.map((source, index) =>
              Object.freeze({
                axis: source.axis,
                resourceId: source.resourceId,
                rank: safeNumber(
                  wireIntegerToBigInt(source.rank),
                  `$.nodes.${node.nodeId}.launchSources[${index}].rank`,
                ),
                maxExtent:
                  maxSelection.logicalExtents[source.axis] as number,
              }))),
          }),
      maxExtents: maxSelection.logicalExtents,
      maxElementCount: maxSelection.elementCount,
    }));
    return;
  }
  if (
    dynamicPrepared.dynamicLaunchMode !== "linear-prefix" ||
    maxSelection.kind !== "linear-prefix"
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${node.nodeId}`,
      "linear dynamic dispatch received a rectangular prepared program",
    );
  }
  dynamicDispatches.push(Object.freeze({
    ...common,
    kind: "linear-prefix",
    selector: node.mode === "runtime-u32-prefix-elements"
      ? Object.freeze({
          kind: "runtime-control" as const,
          controlId: node.launchControl.controlId,
        })
      : Object.freeze({
          kind: "resource" as const,
          resourceId: node.launchSource.resourceId,
          rank: safeNumber(
            wireIntegerToBigInt(node.launchSource.rank),
            `$.nodes.${node.nodeId}.launchSource.rank`,
          ),
        }),
    maxElementCount: maxSelection.elementCount,
  }));
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

function preparedDynamicViewCopy(
  prepared:
    | PreparedSemanticViewCopyWgsl
    | PreparedSemanticViewCopyDynamicWgsl,
  nodeId: string,
): PreparedSemanticViewCopyDynamicWgsl {
  if (!("dynamicLaunchMode" in prepared)) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${nodeId}`,
      "dynamic dispatch received a static view-copy program",
    );
  }
  return prepared;
}

function dynamicDispatchStep(
  step: WgslKernelSequenceStep,
  plan: PreparedDynamicDispatchPlan,
  selection: DynamicLaunchSelection,
): WgslKernelSequenceStep {
  if (
    (plan.kind === "linear-prefix") !==
      (selection.kind === "linear-prefix")
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${plan.nodeId}.launch`,
      "dynamic dispatch launch kind diverged from its prepared program",
    );
  }
  return Object.freeze({
    ...step,
    launch: dynamicLaunch(selection),
    uniforms: dynamicUniformBindingFromName(
      plan.dynamicUniformName,
      selection,
      step.uniforms,
    ),
  });
}

function dynamicUniformBinding(
  prepared: PreparedSemanticViewCopyDynamicWgsl,
  selection: DynamicLaunchSelection,
): Readonly<Record<string, ArrayBuffer | ArrayBufferView>> {
  if (
    (prepared.dynamicLaunchMode === "linear-prefix") !==
      (selection.kind === "linear-prefix")
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.dynamicDispatch.launch",
      "dynamic launch selection does not match its prepared WGSL mode",
    );
  }
  return dynamicUniformBindingFromName(
    prepared.dynamicUniformName,
    selection,
  );
}

function dynamicUniformBindingFromName(
  uniformName: string,
  selection: DynamicLaunchSelection,
  existing: WgslKernelSequenceStep["uniforms"] = {},
): Readonly<Record<string, ArrayBuffer | ArrayBufferView>> {
  const value = selection.kind === "linear-prefix"
    ? new Uint32Array([selection.elementCount])
    : new Uint32Array([
        ...selection.logicalExtents,
        ...Array.from(
          {
            length:
              rectangularDynamicUniformWordLength(
                selection.logicalExtents.length,
              ) - selection.logicalExtents.length,
          },
          () => 0,
        ),
      ]);
  return Object.freeze({
    ...existing,
    [uniformName]: value,
  });
}

function rectangularDynamicUniformWordLength(rank: number): 4 | 8 {
  return rank <= 4 ? 4 : 8;
}

function alignedDynamicUniformGpuBytes(hostBytes: 4 | 16 | 32): bigint {
  return hostBytes <= 16 ? 16n : 32n;
}

function dynamicLaunch(
  selection: DynamicLaunchSelection,
): Readonly<{ readonly dispatchCount: readonly [number, number, number] }> {
  if (selection.kind === "linear-prefix") {
    return frozenLaunch(selection.elementCount);
  }
  const extents = selection.logicalExtents;
  if (extents.length < 2 || extents.length > 7) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.dynamicDispatch.logicalExtents",
      "rectangular dynamic launch rank must be between two and seven",
    );
  }
  const lastAxis = extents.length - 1;
  const leadingPlane = extents
    .slice(0, -2)
    .reduce((product, extent) => product * extent, 1);
  return Object.freeze({
    dispatchCount: Object.freeze([
      extents[lastAxis] as number,
      extents[lastAxis - 1] as number,
      leadingPlane,
    ] as const),
  });
}

function rectangularDynamicSelection(
  extents: readonly number[],
  path: string,
): Extract<
  DynamicLaunchSelection,
  { readonly kind: "rectangular-prefix" }
> {
  if (extents.length < 2 || extents.length > 7) {
    fail(
      "BG-WEBGPU-GRAPH-UNSUPPORTED-PROFILE",
      path,
      "portable rectangular dynamic launch supports ranks two through seven",
    );
  }
  let elementCount = 1n;
  for (const [axis, extent] of extents.entries()) {
    if (!Number.isSafeInteger(extent) || extent <= 0) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `${path}[${axis}]`,
        "rectangular dynamic extent must be a positive exact integer",
      );
    }
    elementCount *= BigInt(extent);
  }
  if (elementCount > BigInt(MAX_VIEW_COPY_ELEMENTS)) {
    fail(
      "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
      path,
      `rectangular dynamic launch requires ${elementCount} elements; portable limit is ${MAX_VIEW_COPY_ELEMENTS}`,
    );
  }
  return Object.freeze({
    kind: "rectangular-prefix",
    logicalExtents: Object.freeze([...extents]),
    elementCount: Number(elementCount),
  });
}

function selectRuntimeDynamicLaunch(
  dispatch: PreparedDynamicDispatchPlan,
  controls: ReadonlyMap<string, number>,
): DynamicLaunchSelection {
  if (dispatch.kind === "linear-prefix") {
    if (dispatch.selector.kind !== "runtime-control") {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${dispatch.nodeId}.launchSource`,
        "resource dynamic dispatch entered request-time selection",
      );
    }
    const elementCount = controls.get(dispatch.selector.controlId);
    if (
      elementCount === undefined ||
      elementCount <= 0 ||
      elementCount > dispatch.maxElementCount
    ) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${dispatch.nodeId}.launchControl`,
        "admitted dynamic dispatch control is outside its verified bound",
      );
    }
    return Object.freeze({
      kind: "linear-prefix",
      elementCount,
    });
  }
  if (dispatch.selector.kind !== "runtime-control") {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      `$.nodes.${dispatch.nodeId}.launchSources`,
      "resource rectangular dynamic dispatch entered request-time selection",
    );
  }
  const extents = dispatch.selector.controls.map((control) => {
    const extent = controls.get(control.controlId);
    if (
      extent === undefined ||
      extent <= 0 ||
      extent > control.maxExtent
    ) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${dispatch.nodeId}.launchControls[${control.axis}]`,
        "admitted rectangular dispatch extent is outside its verified bound",
      );
    }
    return extent;
  });
  return rectangularDynamicSelection(
    extents,
    `$.nodes.${dispatch.nodeId}.launchControls`,
  );
}

function dynamicDispatchCompletion(
  nodeId: string,
  selection: DynamicLaunchSelection,
): HostGraphDynamicDispatchCompletion {
  return Object.freeze({
    nodeId,
    ...(selection.kind === "rectangular-prefix"
      ? {
          logicalExtents: Object.freeze(
            selection.logicalExtents.map((extent) =>
              encodeWireU64(BigInt(extent))),
          ),
        }
      : {}),
    elementCount: encodeWireU64(BigInt(selection.elementCount)),
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
  steps: readonly WgslKernelSequenceStep[],
  pipelineStepIndices: readonly number[],
  buffers: Readonly<Record<string, WgslTypedArray>>,
  pipelineSet: WgslPreparedPipelineSet,
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
        steps,
        {
          buffers,
          storageMetadata: state.storageMetadata,
          readback: state.readbackStorageNames,
        },
        pipelineSet,
        0,
        pipelineStepIndices,
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

async function executeGraphWithResourceFeedback(
  device: KernelDevice,
  gpu: GPUDevice,
  state: PreparedState,
  selectedExecution: SelectedExecution,
  buffers: Readonly<Record<string, WgslTypedArray>>,
  pipelineSet: WgslPreparedPipelineSet,
  signal: AbortSignal | undefined,
): Promise<ExecutedGraph> {
  if (selectedExecution.resourceDynamicDispatch !== undefined) {
    return executeGraphWithResourceDynamicDispatchFeedback(
      device,
      gpu,
      state,
      selectedExecution,
      buffers,
      pipelineSet,
      signal,
    );
  }
  if (selectedExecution.resourceRepeat !== undefined) {
    return executeGraphWithResourceRepeatFeedback(
      device,
      gpu,
      state,
      selectedExecution,
      buffers,
      pipelineSet,
      signal,
    );
  }
  const conditional = selectedExecution.resourceConditional;
  if (
    conditional === undefined ||
    conditional.predicate.kind !== "resource"
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.feedback",
      "resource feedback execution lost its verified conditional",
    );
  }
  const residents = await createResidentGraphBuffers(
    device,
    gpu,
    buffers,
  );
  try {
    const predicatePlan = state.resourcesById.get(
      conditional.predicate.resourceId,
    );
    const predicateStorageName = predicatePlan?.storageNames[
      conditional.predicate.rank
    ];
    if (
      predicatePlan === undefined ||
      predicateStorageName === undefined ||
      predicatePlan.byteLength !== 4 ||
      residents[predicateStorageName] === undefined
    ) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${conditional.nodeId}.predicate`,
        "verified resource conditional storage disappeared",
      );
    }
    const selectedConditionalIndex =
      selectedExecution.pipelineStepIndices.indexOf(
        conditional.startStepIndex,
      );
    if (selectedConditionalIndex < 0) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${conditional.nodeId}.predicate`,
        "verified resource conditional step disappeared after runtime selection",
      );
    }
    const prefixSteps = selectedExecution.steps.slice(
      0,
      selectedConditionalIndex,
    );
    const prefixPipelineStepIndices =
      selectedExecution.pipelineStepIndices.slice(
        0,
        selectedConditionalIndex,
      );
    if (prefixSteps.length === 0) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${conditional.nodeId}.predicate`,
        "verified resource conditional producer disappeared",
      );
    }
    const prefix = await executeResidentGraphStage(
      device,
      gpu,
      state,
      prefixSteps,
      residents,
      pipelineSet,
      prefixPipelineStepIndices,
      [predicateStorageName],
      "$.feedback.prefix",
    );
    throwIfCancelled(signal);
    const predicateValue = prefix.buffers[predicateStorageName];
    if (!(predicateValue instanceof Uint32Array) ||
        predicateValue.length !== 1) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${conditional.nodeId}.predicate`,
        "resource conditional feedback returned an invalid u32 allocation",
      );
    }
    const selectedBranch = predicateValue[0] === 0
      ? "else" as const
      : "then" as const;
    const branch = selectedBranch === "then"
      ? conditional.thenBranch
      : conditional.elseBranch;
    const steps = [...selectedExecution.steps];
    steps.splice(
      selectedConditionalIndex,
      conditional.stepCount,
      ...branch.steps,
    );
    const completion = Object.freeze({
      nodeId: conditional.nodeId,
      selectedBranch,
      bodyNodeIds: branch.bodyNodeIds,
    });
    const completedByNodeId = new Map(
      selectedExecution.completedConditionals.map((item) => [
        item.nodeId,
        item,
      ]),
    );
    completedByNodeId.set(conditional.nodeId, completion);
    const completedConditionals = Object.freeze(
      state.conditionals.map((item) => {
        const completed = completedByNodeId.get(item.nodeId);
        if (completed === undefined) {
          fail(
            "BG-WEBGPU-GRAPH-INTERNAL",
            `$.nodes.${item.nodeId}`,
            "conditional completion disappeared during feedback staging",
          );
        }
        return completed;
      }),
    );
    const finalSelection = Object.freeze({
      steps: Object.freeze(steps),
      pipelineStepIndices: selectedExecution.pipelineStepIndices,
      dispatchStepCount: selectedExecution.dispatchStepCount,
      copyStepCount: selectedExecution.copyStepCount,
      collectiveReductionStepCount:
        selectedExecution.collectiveReductionStepCount,
      collectiveReplicationStepCount:
        selectedExecution.collectiveReplicationStepCount,
      completedRepeats: selectedExecution.completedRepeats,
      completedDynamicDispatches:
        selectedExecution.completedDynamicDispatches,
      completedConditionals,
      resourceConditional: conditional,
    });
    throwIfCancelled(signal);
    const result = await executeResidentGraphStage(
      device,
      gpu,
      state,
      finalSelection.steps.slice(selectedConditionalIndex),
      residents,
      pipelineSet,
      finalSelection.pipelineStepIndices.slice(selectedConditionalIndex),
      state.readbackStorageNames,
      "$.feedback.suffix",
    );
    return Object.freeze({ result, selectedExecution: finalSelection });
  } finally {
    destroyResidentGraphBuffers(residents);
  }
}

async function executeGraphWithResourceDynamicDispatchFeedback(
  device: KernelDevice,
  gpu: GPUDevice,
  state: PreparedState,
  selectedExecution: SelectedExecution,
  buffers: Readonly<Record<string, WgslTypedArray>>,
  pipelineSet: WgslPreparedPipelineSet,
  signal: AbortSignal | undefined,
): Promise<ExecutedGraph> {
  const dispatch = selectedExecution.resourceDynamicDispatch;
  if (
    dispatch === undefined ||
    dispatch.selector.kind !== "resource"
  ) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.feedback",
      "resource feedback execution lost its verified dynamic dispatch",
    );
  }
  const residents = await createResidentGraphBuffers(
    device,
    gpu,
    buffers,
  );
  try {
    const feedbackSources = dispatch.kind === "linear-prefix"
      ? [Object.freeze({
          axis: 0,
          resourceId: dispatch.selector.resourceId,
          rank: dispatch.selector.rank,
          maxExtent: dispatch.maxElementCount,
        })]
      : dispatch.selector.sources;
    const sourceStorageNames = feedbackSources.map((source, index) => {
      const sourcePlan = state.resourcesById.get(source.resourceId);
      const sourceStorageName = sourcePlan?.storageNames[source.rank];
      if (
        sourcePlan === undefined ||
        sourceStorageName === undefined ||
        sourcePlan.byteLength !== 4 ||
        residents[sourceStorageName] === undefined
      ) {
        fail(
          "BG-WEBGPU-GRAPH-INTERNAL",
          dispatch.kind === "linear-prefix"
            ? `$.nodes.${dispatch.nodeId}.launchSource`
            : `$.nodes.${dispatch.nodeId}.launchSources[${index}]`,
          "verified resource dynamic dispatch source storage disappeared",
        );
      }
      return sourceStorageName;
    });
    const selectedDispatchIndex =
      selectedExecution.pipelineStepIndices.indexOf(
        dispatch.startStepIndex,
      );
    if (selectedDispatchIndex < 0) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${dispatch.nodeId}.launchSource`,
        "verified resource dynamic dispatch step disappeared after runtime selection",
      );
    }
    const prefixSteps = selectedExecution.steps.slice(
      0,
      selectedDispatchIndex,
    );
    const prefixPipelineStepIndices =
      selectedExecution.pipelineStepIndices.slice(
        0,
        selectedDispatchIndex,
      );
    if (prefixSteps.length === 0) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${dispatch.nodeId}.launchSource`,
        "verified resource dynamic dispatch producer disappeared",
      );
    }
    const prefix = await executeResidentGraphStage(
      device,
      gpu,
      state,
      prefixSteps,
      residents,
      pipelineSet,
      prefixPipelineStepIndices,
      sourceStorageNames,
      "$.feedback.prefix",
    );
    throwIfCancelled(signal);
    const producedExtents = sourceStorageNames.map((storageName, index) => {
      const sourceValue = prefix.buffers[storageName];
      if (!(sourceValue instanceof Uint32Array) || sourceValue.length !== 1) {
        fail(
          "BG-WEBGPU-GRAPH-INTERNAL",
          dispatch.kind === "linear-prefix"
            ? `$.nodes.${dispatch.nodeId}.launchSource`
            : `$.nodes.${dispatch.nodeId}.launchSources[${index}]`,
          "resource dynamic dispatch feedback returned an invalid u32 allocation",
        );
      }
      const value = sourceValue[0] as number;
      const maximum = feedbackSources[index]?.maxExtent;
      if (
        maximum === undefined ||
        value <= 0 ||
        value > maximum
      ) {
        fail(
          "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
          dispatch.kind === "linear-prefix"
            ? `$.nodes.${dispatch.nodeId}.launchSource`
            : `$.nodes.${dispatch.nodeId}.launchSources[${index}]`,
          dispatch.kind === "linear-prefix"
            ? `produced resource dynamic dispatch count ${value} must be between one and its artifact bound ${maximum}`
            : `produced rectangular dynamic dispatch extent ${value} must be between one and its artifact bound ${maximum}`,
        );
      }
      return value;
    });
    const selection: DynamicLaunchSelection =
      dispatch.kind === "linear-prefix"
        ? Object.freeze({
            kind: "linear-prefix" as const,
            elementCount: producedExtents[0] as number,
          })
        : rectangularDynamicSelection(
            producedExtents,
            `$.nodes.${dispatch.nodeId}.launchSources`,
          );
    const steps = [...selectedExecution.steps];
    for (
      let offset = 0;
      offset < dispatch.stepCount;
      offset += 1
    ) {
      const selectedIndex = selectedDispatchIndex + offset;
      const step = steps[selectedIndex];
      const pipelineIndex =
        selectedExecution.pipelineStepIndices[selectedIndex];
      if (
        step === undefined ||
        pipelineIndex !== dispatch.startStepIndex + offset
      ) {
        fail(
          "BG-WEBGPU-GRAPH-INTERNAL",
          `$.nodes.${dispatch.nodeId}`,
          "resource dynamic dispatch rank step disappeared during feedback staging",
        );
      }
      steps[selectedIndex] = dynamicDispatchStep(
        step,
        dispatch,
        selection,
      );
    }
    const completion = dynamicDispatchCompletion(
      dispatch.nodeId,
      selection,
    );
    const completionsByNodeId = new Map(
      selectedExecution.completedDynamicDispatches.map((item) => [
        item.nodeId,
        item,
      ]),
    );
    completionsByNodeId.set(dispatch.nodeId, completion);
    const completedDynamicDispatches = Object.freeze(
      state.topologicalNodeIds.flatMap((nodeId) => {
        const item = completionsByNodeId.get(nodeId);
        return item === undefined ? [] : [item];
      }),
    );
    const finalSelection = Object.freeze({
      steps: Object.freeze(steps),
      pipelineStepIndices: selectedExecution.pipelineStepIndices,
      dispatchStepCount: selectedExecution.dispatchStepCount,
      copyStepCount: selectedExecution.copyStepCount,
      collectiveReductionStepCount:
        selectedExecution.collectiveReductionStepCount,
      collectiveReplicationStepCount:
        selectedExecution.collectiveReplicationStepCount,
      completedRepeats: selectedExecution.completedRepeats,
      completedDynamicDispatches,
      completedConditionals: selectedExecution.completedConditionals,
      resourceDynamicDispatch: dispatch,
    });
    throwIfCancelled(signal);
    const result = await executeResidentGraphStage(
      device,
      gpu,
      state,
      finalSelection.steps.slice(selectedDispatchIndex),
      residents,
      pipelineSet,
      finalSelection.pipelineStepIndices.slice(selectedDispatchIndex),
      state.readbackStorageNames,
      "$.feedback.suffix",
    );
    return Object.freeze({ result, selectedExecution: finalSelection });
  } finally {
    destroyResidentGraphBuffers(residents);
  }
}

async function executeGraphWithResourceRepeatFeedback(
  device: KernelDevice,
  gpu: GPUDevice,
  state: PreparedState,
  selectedExecution: SelectedExecution,
  buffers: Readonly<Record<string, WgslTypedArray>>,
  pipelineSet: WgslPreparedPipelineSet,
  signal: AbortSignal | undefined,
): Promise<ExecutedGraph> {
  const repeat = selectedExecution.resourceRepeat;
  if (repeat === undefined) {
    fail(
      "BG-WEBGPU-GRAPH-INTERNAL",
      "$.feedback",
      "resource feedback execution lost its verified repeat",
    );
  }
  const residents = await createResidentGraphBuffers(
    device,
    gpu,
    buffers,
  );
  try {
    const sourcePlan = state.resourcesById.get(repeat.resourceId);
    const sourceStorageName = sourcePlan?.storageNames[repeat.rank];
    if (
      sourcePlan === undefined ||
      sourceStorageName === undefined ||
      sourcePlan.byteLength !== 4 ||
      residents[sourceStorageName] === undefined
    ) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${repeat.nodeId}.iterationSource`,
        "verified resource repeat source storage disappeared",
      );
    }
    const selectedRepeatIndex =
      selectedExecution.pipelineStepIndices.indexOf(
        repeat.startStepIndex,
      );
    if (selectedRepeatIndex < 0) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${repeat.nodeId}.iterationSource`,
        "verified resource repeat step disappeared after runtime selection",
      );
    }
    const prefixSteps = selectedExecution.steps.slice(
      0,
      selectedRepeatIndex,
    );
    const prefixPipelineStepIndices =
      selectedExecution.pipelineStepIndices.slice(
        0,
        selectedRepeatIndex,
      );
    if (prefixSteps.length === 0) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${repeat.nodeId}.iterationSource`,
        "verified resource repeat producer disappeared",
      );
    }
    const prefix = await executeResidentGraphStage(
      device,
      gpu,
      state,
      prefixSteps,
      residents,
      pipelineSet,
      prefixPipelineStepIndices,
      [sourceStorageName],
      "$.feedback.prefix",
    );
    throwIfCancelled(signal);
    const sourceValue = prefix.buffers[sourceStorageName];
    if (!(sourceValue instanceof Uint32Array) || sourceValue.length !== 1) {
      fail(
        "BG-WEBGPU-GRAPH-INTERNAL",
        `$.nodes.${repeat.nodeId}.iterationSource`,
        "resource repeat feedback returned an invalid u32 allocation",
      );
    }
    const iterationCount = sourceValue[0] as number;
    if (iterationCount > repeat.maxIterationCount) {
      fail(
        "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
        `$.nodes.${repeat.nodeId}.iterationSource`,
        `produced resource repeat count ${iterationCount} exceeds its artifact bound ${repeat.maxIterationCount}`,
      );
    }
    const maximumRepeatEnd = repeat.startStepIndex +
      (repeat.maxIterationCount * repeat.stepCountPerIteration);
    const selectedRepeatEnd = selectedRepeatIndex +
      (iterationCount * repeat.stepCountPerIteration);
    let afterMaximumIndex = selectedExecution.pipelineStepIndices.findIndex(
      (pipelineIndex, selectedIndex) =>
        selectedIndex >= selectedRepeatIndex &&
        pipelineIndex >= maximumRepeatEnd,
    );
    if (afterMaximumIndex < 0) {
      afterMaximumIndex = selectedExecution.steps.length;
    }
    const steps = Object.freeze([
      ...selectedExecution.steps.slice(0, selectedRepeatEnd),
      ...selectedExecution.steps.slice(afterMaximumIndex),
    ]);
    const pipelineStepIndices = Object.freeze([
      ...selectedExecution.pipelineStepIndices.slice(0, selectedRepeatEnd),
      ...selectedExecution.pipelineStepIndices.slice(afterMaximumIndex),
    ]);
    const skippedIterations =
      repeat.maxIterationCount - iterationCount;
    const completion = Object.freeze({
      nodeId: repeat.nodeId,
      iterationCount: encodeWireU64(BigInt(iterationCount)),
      bodyNodeIds: repeat.bodyNodeIds,
    });
    const completionsByNodeId = new Map(
      selectedExecution.completedRepeats.map((item) => [
        item.nodeId,
        item,
      ]),
    );
    completionsByNodeId.set(repeat.nodeId, completion);
    const completedRepeats = Object.freeze(
      state.topologicalNodeIds.flatMap((nodeId) => {
        const item = completionsByNodeId.get(nodeId);
        return item === undefined ? [] : [item];
      }),
    );
    const finalSelection = Object.freeze({
      steps,
      pipelineStepIndices,
      dispatchStepCount:
        selectedExecution.dispatchStepCount -
        (repeat.countsPerIteration.dispatch * skippedIterations),
      copyStepCount:
        selectedExecution.copyStepCount -
        (repeat.countsPerIteration.copy * skippedIterations),
      collectiveReductionStepCount:
        selectedExecution.collectiveReductionStepCount -
        (repeat.countsPerIteration.reduction * skippedIterations),
      collectiveReplicationStepCount:
        selectedExecution.collectiveReplicationStepCount -
        (repeat.countsPerIteration.replication * skippedIterations),
      completedRepeats,
      completedDynamicDispatches:
        selectedExecution.completedDynamicDispatches,
      completedConditionals: selectedExecution.completedConditionals,
      resourceRepeat: repeat,
    });
    throwIfCancelled(signal);
    const suffixSteps = finalSelection.steps.slice(selectedRepeatIndex);
    const suffixPipelineStepIndices =
      finalSelection.pipelineStepIndices.slice(selectedRepeatIndex);
    const result = suffixSteps.length === 0
      ? await readResidentGraphBuffers(
        device,
        gpu,
        residents,
        state.readbackStorageNames,
        "$.feedback.suffix",
      )
      : await executeResidentGraphStage(
        device,
        gpu,
        state,
        suffixSteps,
        residents,
        pipelineSet,
        suffixPipelineStepIndices,
        state.readbackStorageNames,
        "$.feedback.suffix",
      );
    return Object.freeze({ result, selectedExecution: finalSelection });
  } finally {
    destroyResidentGraphBuffers(residents);
  }
}

async function createResidentGraphBuffers(
  device: KernelDevice,
  gpu: GPUDevice,
  buffers: Readonly<Record<string, WgslTypedArray>>,
): Promise<Readonly<Record<string, WgslResidentBuffer>>> {
  try {
    return await issueWithWebGpuErrorScopes(
      gpu,
      "$.feedback.storage",
      () => {
        const residents: Record<string, WgslResidentBuffer> = {};
        try {
          for (const [name, data] of Object.entries(buffers)) {
            residents[name] = createWgslStorageBuffer(device, {
              valueType: "u32",
              byteLength: data.byteLength,
              data: data as Uint32Array,
              label: `bg-host-graph-${name}`,
            });
          }
          return Object.freeze(residents);
        } catch (cause) {
          destroyResidentGraphBuffers(residents);
          throw cause;
        }
      },
      { cleanup: destroyResidentGraphBuffers },
    );
  } catch (cause) {
    translateExecutionFailure(cause, "$.feedback.storage");
  }
}

function destroyResidentGraphBuffers(
  residents: Readonly<Record<string, WgslResidentBuffer>>,
): void {
  for (const resident of Object.values(residents)) {
    destroyWgslStorageBuffer(resident);
  }
}

async function executeResidentGraphStage(
  device: KernelDevice,
  gpu: GPUDevice,
  state: PreparedState,
  steps: readonly WgslKernelSequenceStep[],
  residentBuffers: Readonly<Record<string, WgslResidentBuffer>>,
  pipelineSet: WgslPreparedPipelineSet,
  pipelineStepIndices: readonly number[],
  readback: readonly string[],
  path: string,
): Promise<WgslKernelRunResult> {
  let sequence: Awaited<
    ReturnType<typeof prepareWgslKernelProgramSequence>
  >;
  try {
    sequence = await issueAsyncWithWebGpuErrorScopes(
      gpu,
      `${path}.pipeline`,
      () => prepareWgslKernelProgramSequence(
        device,
        steps,
        {
          buffers: {},
          residentBuffers,
          storageMetadata: state.storageMetadata,
          readback,
        },
        pipelineSet,
        0,
        pipelineStepIndices,
      ),
      { cleanup: (prepared) => prepared.destroy() },
    );
  } catch (cause) {
    translateExecutionFailure(cause, `${path}.pipeline`);
  }
  try {
    return await issueAsyncWithWebGpuErrorScopes(
      gpu,
      `${path}.dispatch`,
      () => sequence.run({
        readback,
        awaitCompletion: true,
      }),
    );
  } catch (cause) {
    translateExecutionFailure(cause, `${path}.dispatch`);
  } finally {
    sequence.destroy();
  }
}

async function readResidentGraphBuffers(
  device: KernelDevice,
  gpu: GPUDevice,
  residentBuffers: Readonly<Record<string, WgslResidentBuffer>>,
  readback: readonly string[],
  path: string,
): Promise<WgslKernelRunResult> {
  try {
    return await issueAsyncWithWebGpuErrorScopes(
      gpu,
      `${path}.readback`,
      async () => {
        const buffers: Record<string, WgslTypedArray> = {};
        for (const name of readback) {
          const resident = residentBuffers[name];
          if (resident === undefined) {
            fail(
              "BG-WEBGPU-GRAPH-INTERNAL",
              `${path}.readback`,
              `resident readback ${name} disappeared`,
            );
          }
          buffers[name] = await readWgslStorageBuffer(device, resident);
        }
        return Object.freeze({ buffers: Object.freeze(buffers) });
      },
    );
  } catch (cause) {
    translateExecutionFailure(cause, `${path}.readback`);
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
): CapturedExecutionRequest {
  const object = inspectPlainObject(
    request,
    ["inputs", "controls"],
    ["inputs"],
    "$.request",
  );
  const controls = captureRuntimeControls(
    object.controls,
    state.runtimeControlIds,
    state.runtimeRepeatLimits,
    state.dynamicDispatchLimits,
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
  return Object.freeze({
    inputs: Object.freeze(inputs),
    controls,
  });
}

function captureRuntimeControls(
  value: unknown,
  runtimeControlIds: readonly string[],
  runtimeRepeatLimits: ReadonlyMap<string, number>,
  dynamicDispatchLimits: ReadonlyMap<string, number>,
): readonly CapturedControl[] {
  const controlValues = value === undefined
    ? []
    : snapshotDenseArray(
        value,
        "$.request.controls",
        runtimeControlIds.length + 1,
      );
  const expectedControls = new Set(runtimeControlIds);
  const seenControls = new Set<string>();
  const controls: CapturedControl[] = [];
  for (const [index, value] of controlValues.entries()) {
    const path = `$.request.controls[${index}]`;
    const binding = inspectPlainObject(
      value,
      ["controlId", "value"],
      ["controlId", "value"],
      path,
    );
    const controlId = stringValue(
      binding.controlId,
      `${path}.controlId`,
    );
    if (!expectedControls.has(controlId)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `${path}.controlId`,
        `control ${controlId} is not required by the graph`,
      );
    }
    if (seenControls.has(controlId)) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        path,
        `duplicate runtime control ${controlId}`,
      );
    }
    seenControls.add(controlId);
    const controlValue = wireIntegerToBigInt(
      parseWireU64(binding.value, `${path}.value`),
    );
    if (controlValue > MAX_U32) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `${path}.value`,
        "runtime u32 control exceeds 4294967295",
      );
    }
    const repeatLimit = runtimeRepeatLimits.get(controlId);
    if (
      repeatLimit !== undefined &&
      controlValue > BigInt(repeatLimit)
    ) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `${path}.value`,
        `runtime repeat control ${controlId} exceeds its artifact bound ${repeatLimit}`,
      );
    }
    const dynamicDispatchLimit = dynamicDispatchLimits.get(controlId);
    if (
      dynamicDispatchLimit !== undefined &&
      (controlValue === 0n ||
        controlValue > BigInt(dynamicDispatchLimit))
    ) {
      fail(
        "BG-WEBGPU-GRAPH-INVALID-BINDING",
        `${path}.value`,
        `dynamic dispatch control ${controlId} must be between 1 and its artifact bound ${dynamicDispatchLimit}`,
      );
    }
    controls.push(Object.freeze({
      controlId,
      value: Number(controlValue),
    }));
  }
  if (controls.length !== runtimeControlIds.length) {
    fail(
      "BG-WEBGPU-GRAPH-INVALID-BINDING",
      "$.request.controls",
      `expected exactly ${runtimeControlIds.length} runtime control bindings`,
    );
  }
  return Object.freeze(controls);
}

function runtimeRepeatControlLimits(
  repeats: readonly PreparedRuntimeRepeatPlan[],
): ReadonlyMap<string, number> {
  const limits = new Map<string, number>();
  for (const repeat of repeats) {
    const existing = limits.get(repeat.controlId);
    limits.set(
      repeat.controlId,
      existing === undefined
        ? repeat.maxIterationCount
        : Math.min(existing, repeat.maxIterationCount),
    );
  }
  return limits;
}

function dynamicDispatchControlLimits(
  dispatches: readonly PreparedDynamicDispatchPlan[],
): ReadonlyMap<string, number> {
  const limits = new Map<string, number>();
  for (const dispatch of dispatches) {
    const controls =
      dispatch.kind === "rectangular-prefix" &&
        dispatch.selector.kind === "runtime-control"
      ? dispatch.selector.controls.map((control) => ({
          controlId: control.controlId,
          limit: control.maxExtent,
        }))
      : dispatch.kind === "linear-prefix" &&
          dispatch.selector.kind === "runtime-control"
        ? [{
            controlId: dispatch.selector.controlId,
            limit: dispatch.maxElementCount,
          }]
        : [];
    for (const control of controls) {
      const existing = limits.get(control.controlId);
      limits.set(
        control.controlId,
        existing === undefined
          ? control.limit
          : Math.min(existing, control.limit),
      );
    }
  }
  return limits;
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

function capturePipelineOptions(
  options: PrepareSemanticHostGraphWebGpuPipelineOptions,
): {
  readonly signal?: AbortSignal;
  readonly timeoutMs?: number;
  readonly maxPipelineCount?: number;
} {
  const object = inspectPlainObject(
    options,
    ["signal", "timeoutMs", "maxPipelineCount"],
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
  const maxPipelineCount = numberOrUndefined(
    object.maxPipelineCount,
    "$.options.maxPipelineCount",
  );
  return Object.freeze({
    ...(signal === undefined ? {} : { signal }),
    ...(timeoutMs === undefined ? {} : { timeoutMs }),
    ...(maxPipelineCount === undefined ? {} : { maxPipelineCount }),
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
      maxUniformBuffersPerShaderStage:
        gpu.limits.maxUniformBuffersPerShaderStage,
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
  const requiredBindings = Math.max(
    0,
    ...state.deviceAdmissionSteps.map((step) =>
      step.program.bindings.length),
  );
  const requiredStorageBindings = Math.max(
    0,
    ...state.deviceAdmissionSteps.map((step) =>
      step.program.bindings.filter((binding) =>
        binding.kind === "storage").length),
  );
  const requiredUniformBindings = Math.max(
    0,
    ...state.deviceAdmissionSteps.map((step) =>
      step.program.bindings.filter((binding) =>
        binding.kind === "uniform").length),
  );
  if (limits.maxBindingsPerBindGroup < requiredBindings) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      "$.device.limits.maxBindingsPerBindGroup",
      `device cannot bind the required ${requiredBindings} graph resources`,
    );
  }
  if (
    limits.maxStorageBuffersPerShaderStage < requiredStorageBindings
  ) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      "$.device.limits.maxStorageBuffersPerShaderStage",
      `device cannot bind the required ${requiredStorageBindings} storage buffers`,
    );
  }
  if (
    limits.maxUniformBuffersPerShaderStage < requiredUniformBindings
  ) {
    fail(
      "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
      "$.device.limits.maxUniformBuffersPerShaderStage",
      `device cannot bind the required ${requiredUniformBindings} uniform buffers`,
    );
  }
  for (const [index, step] of state.deviceAdmissionSteps.entries()) {
    for (const axis of [0, 1, 2] as const) {
      const workgroups = divideRoundUp(
        BigInt(step.launch.dispatchCount[axis]),
        BigInt(step.program.workgroupSize[axis]),
      );
      if (workgroups > BigInt(limits.maxComputeWorkgroupsPerDimension)) {
        fail(
          "BG-WEBGPU-GRAPH-DEVICE-LIMIT",
          `$.steps[${index}].launch[${axis}]`,
          `dispatch axis ${axis} requires ${workgroups} workgroups; limit is ${
            limits.maxComputeWorkgroupsPerDimension
          }`,
        );
      }
    }
  }
  return Object.freeze({ features, limits });
}

async function hashPipelineIdentity(
  prepared: PreparedSemanticHostGraphWebGpu,
  state: PreparedState,
  device: SemanticHostGraphWebGpuDeviceFacts,
  maxPipelineCount: number,
): Promise<string> {
  return hashNamedComponents({
    profile: SEMANTIC_HOST_GRAPH_WEBGPU_PIPELINE_PROFILE,
    backendVersion: prepared.backendVersion,
    graph: prepared.graphSemanticHash,
    steps: state.steps.length,
    alternativePrograms: state.pipelineAlternatives.length,
    maxPipelineCount,
    wgslModules: prepared.wgslModuleHashes,
    workgroupSize: state.workgroupSize,
    selectedFeatures: device.features,
    limits: device.limits as unknown as JsonObject,
    numericalPolicies: state.numericalPolicies,
    midGraphFeedbackCount: prepared.midGraphFeedbackCount,
  });
}

async function hashBackendSpecialization(
  pipelineIdentityHash: string,
  selectedExecution: SelectedExecution,
): Promise<string> {
  return hashNamedComponents({
    pipelineAuthority: pipelineIdentityHash,
    selectedPipelineStepIndices: selectedExecution.pipelineStepIndices,
    completedRepeats: selectedExecution.completedRepeats,
    completedDynamicDispatches:
      selectedExecution.completedDynamicDispatches,
    selectedConditionals: selectedExecution.completedConditionals,
  });
}

function createTrace(
  prepared: PreparedSemanticHostGraphWebGpu,
  state: PreparedState,
  pipelineIdentityHash: string,
  backendSpecializationHash: string,
  device: SemanticHostGraphWebGpuDeviceFacts,
  submitted: boolean,
  selectedExecution: SelectedExecution,
  completedConditionals: readonly HostGraphConditionalCompletion[],
): SemanticHostGraphWebGpuTrace {
  return Object.freeze({
    profile: prepared.profile,
    backendVersion: prepared.backendVersion,
    graphSemanticHash: prepared.graphSemanticHash,
    pipelineIdentityHash,
    backendSpecializationHash,
    failureModel: HOST_GRAPH_FAILURE_MODEL,
    executedNodeIds: state.topologicalNodeIds,
    expandedStepCount: selectedExecution.steps.length,
    dispatchStepCount: selectedExecution.dispatchStepCount,
    copyStepCount: selectedExecution.copyStepCount,
    materializationCount: prepared.materializationCount,
    completedEventIds: state.eventIds,
    completedRepeats: selectedExecution.completedRepeats,
    completedDynamicDispatches:
      selectedExecution.completedDynamicDispatches,
    completedConditionals,
    midGraphFeedbackCount: prepared.midGraphFeedbackCount,
    collectiveReductionStepCount:
      selectedExecution.collectiveReductionStepCount,
    collectiveReplicationStepCount:
      selectedExecution.collectiveReplicationStepCount,
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
    if (current instanceof WgslPipelineSetResourceLimitError) {
      throw new SemanticHostGraphWebGpuError(
        "BG-WEBGPU-GRAPH-RESOURCE-LIMIT",
        "$.options.maxPipelineCount",
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

import type { JsonObject } from "../schema/json.js";
import type { WireI64, WireU64 } from "../schema/integers.js";
import type { BuiltinDTypeId } from "../layout/dtype.js";

export const HOST_GRAPH_FAILURE_MODEL =
  "fail-stop-no-partial-output-commit" as const;

export type HostGraphResourceRole = "input" | "temporary" | "output";
export type HostGraphCollectiveReduction = "sum" | "min" | "max";
export type HostGraphCollectiveDType = "f32" | "i32" | "u32";
export type HostGraphCollectiveNumericalPolicy =
  | "rank-order-f32"
  | "rank-order-wrapping-32"
  | "exact-32-bit";

export interface HostGraphResource extends JsonObject {
  readonly resourceId: string;
  readonly role: HostGraphResourceRole;
  readonly multiplicity: "per-rank";
  readonly initialization: "external-input" | "zero-fill";
  readonly dtype: BuiltinDTypeId;
  readonly byteLength: WireU64;
  readonly alignmentBytes: number;
}

export interface HostGraphDispatchResourceBinding extends JsonObject {
  readonly semanticResourceId: string;
  readonly graphResourceId: string;
}

export interface HostGraphDispatchNode extends JsonObject {
  readonly nodeId: string;
  readonly kind: "dispatch";
  readonly dependsOn: readonly string[];
  readonly semanticArtifactHash: string;
  readonly entrypointId: string;
  readonly dimensionBindings: Readonly<Record<string, WireI64>>;
  readonly bindings: readonly HostGraphDispatchResourceBinding[];
}

export interface HostGraphAllReduceNode extends JsonObject {
  readonly nodeId: string;
  readonly kind: "all-reduce";
  readonly dependsOn: readonly string[];
  readonly resourceId: string;
  readonly reduction: HostGraphCollectiveReduction;
  readonly dtype: HostGraphCollectiveDType;
  readonly numericalPolicy: HostGraphCollectiveNumericalPolicy;
  readonly participants: readonly WireU64[];
  readonly result: "replicated-to-all-participants";
}

export interface HostGraphCopyNode extends JsonObject {
  readonly nodeId: string;
  readonly kind: "copy";
  readonly dependsOn: readonly string[];
  readonly sourceResourceId: string;
  readonly destinationResourceId: string;
  readonly mode: "whole-allocation-bytes-per-rank";
}

export interface HostGraphMaterializeNode extends JsonObject {
  readonly nodeId: string;
  readonly kind: "materialize";
  readonly dependsOn: readonly string[];
  readonly resourceId: string;
  readonly mode: "host-readback-after-graph-success";
}

export interface HostGraphEventNode extends JsonObject {
  readonly nodeId: string;
  readonly kind: "event";
  readonly dependsOn: readonly string[];
  readonly eventId: string;
  readonly mode: "completion-after-dependencies";
}

export type HostGraphExecutableNode =
  | HostGraphDispatchNode
  | HostGraphAllReduceNode
  | HostGraphCopyNode;

export type HostGraphRepeatBodyNode = HostGraphExecutableNode;

interface HostGraphRepeatNodeBase extends JsonObject {
  readonly nodeId: string;
  readonly kind: "repeat";
  readonly dependsOn: readonly string[];
  readonly body: readonly HostGraphRepeatBodyNode[];
}

export interface HostGraphFixedRepeatNode extends HostGraphRepeatNodeBase {
  readonly iterationCount: WireU64;
  readonly mode: "fixed-count-sequential";
}

export interface HostGraphRuntimeRepeatControl extends JsonObject {
  readonly controlId: string;
  readonly mode: "u32-count";
}

export interface HostGraphRuntimeControlRepeatNode
  extends HostGraphRepeatNodeBase {
  readonly iterationControl: HostGraphRuntimeRepeatControl;
  readonly maxIterationCount: WireU64;
  readonly mode: "runtime-u32-count-sequential";
}

export type HostGraphRepeatNode =
  | HostGraphFixedRepeatNode
  | HostGraphRuntimeControlRepeatNode;

export interface HostGraphRepeatCompletion extends JsonObject {
  readonly nodeId: string;
  readonly iterationCount: WireU64;
  readonly bodyNodeIds: readonly string[];
}

export type HostGraphConditionalBodyNode = HostGraphExecutableNode;

export interface HostGraphInputPredicate extends JsonObject {
  readonly resourceId: string;
  readonly rank: WireU64;
  readonly mode: "u32-nonzero";
}

export interface HostGraphRuntimeControlPredicate extends JsonObject {
  readonly controlId: string;
  readonly mode: "u32-nonzero";
}

export interface HostGraphResourcePredicate extends JsonObject {
  readonly resourceId: string;
  readonly rank: WireU64;
  readonly mode: "u32-nonzero";
}

interface HostGraphConditionalNodeBase extends JsonObject {
  readonly nodeId: string;
  readonly kind: "conditional";
  readonly dependsOn: readonly string[];
  readonly thenBody: readonly HostGraphConditionalBodyNode[];
  readonly elseBody: readonly HostGraphConditionalBodyNode[];
}

export interface HostGraphInputConditionalNode
  extends HostGraphConditionalNodeBase {
  readonly predicate: HostGraphInputPredicate;
  readonly mode: "input-u32-branch-sequential";
}

export interface HostGraphRuntimeControlConditionalNode
  extends HostGraphConditionalNodeBase {
  readonly predicate: HostGraphRuntimeControlPredicate;
  readonly mode: "runtime-u32-branch-sequential";
}

export interface HostGraphResourceConditionalNode
  extends HostGraphConditionalNodeBase {
  readonly predicate: HostGraphResourcePredicate;
  readonly mode: "resource-u32-branch-sequential";
}

export type HostGraphConditionalNode =
  | HostGraphInputConditionalNode
  | HostGraphRuntimeControlConditionalNode
  | HostGraphResourceConditionalNode;

export interface HostGraphConditionalCompletion extends JsonObject {
  readonly nodeId: string;
  readonly selectedBranch: "then" | "else";
  readonly bodyNodeIds: readonly string[];
}

export type HostGraphNode =
  | HostGraphDispatchNode
  | HostGraphAllReduceNode
  | HostGraphCopyNode
  | HostGraphMaterializeNode
  | HostGraphEventNode
  | HostGraphRepeatNode
  | HostGraphConditionalNode;

/**
 * Backend-neutral host graph meaning. Transport, topology, scheduling,
 * retries, and execution evidence belong to separately verified adapters.
 */
export interface HostGraphProgram extends JsonObject {
  readonly kind: "host-graph";
  readonly version: {
    readonly major: 1;
    readonly minor: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
  };
  readonly failureModel: typeof HOST_GRAPH_FAILURE_MODEL;
  readonly rankCount: WireU64;
  readonly resources: readonly HostGraphResource[];
  readonly nodes: readonly HostGraphNode[];
}

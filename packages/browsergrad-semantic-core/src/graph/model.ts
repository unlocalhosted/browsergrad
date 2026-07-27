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

export type HostGraphNode =
  | HostGraphDispatchNode
  | HostGraphAllReduceNode
  | HostGraphCopyNode;

/**
 * Backend-neutral host graph meaning. Transport, topology, scheduling,
 * retries, and execution evidence belong to separately verified adapters.
 */
export interface HostGraphProgram extends JsonObject {
  readonly kind: "host-graph";
  readonly version: {
    readonly major: 1;
    readonly minor: 0 | 1;
  };
  readonly failureModel: typeof HOST_GRAPH_FAILURE_MODEL;
  readonly rankCount: WireU64;
  readonly resources: readonly HostGraphResource[];
  readonly nodes: readonly HostGraphNode[];
}

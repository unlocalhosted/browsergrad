import {
  cudaLiteTruthy as truthy,
} from "./cuda_lite_values.js";
import { cudaSyncthreadsPredicateReduction } from "./cuda_sync_calls.js";
import {
  cooperativeReductionOperationForName,
  type CooperativeReductionOperation,
} from "./cooperative_reduction.js";
import {
  createBuiltinSemanticSymbolId,
  semanticIdsEqual,
} from "./semantic_ids.js";
import {
  semanticCooperativeGroupInfo,
  semanticCooperativeGroupRankParamName,
  semanticCooperativeGroupSizeParamName,
} from "./semantic_cooperative_groups.js";
import {
  isSemanticFloatVectorType,
  semanticExpressionValueType,
} from "./semantic_vector_intrinsics.js";
import type {
  CompiledCudaLiteKernel,
  CudaLiteScalarType,
  SourceSpan,
} from "./types.js";
import type {
  SemanticExpression,
  SemanticKernelIrOperation,
} from "./semantic_ir_types.js";
import type { ReferenceVector3 } from "./reference_vectors.js";

export type SemanticReferenceCollectiveValue = number | ReferenceVector3 | number[];

export interface SemanticReferenceCollectiveContext {
  readonly compiled: CompiledCudaLiteKernel;
  readonly locals: Map<string, SemanticReferenceCollectiveValue>;
  readonly blockIdx: ReferenceVector3;
  readonly threadIdx: ReferenceVector3;
  readonly blockDim: ReferenceVector3;
  readonly gridDim: ReferenceVector3;
  readonly blockContexts: readonly SemanticReferenceCollectiveContext[];
  readonly activeCollectiveContexts?: readonly SemanticReferenceCollectiveContext[];
}

export interface SemanticReferenceCollectiveExecutorAdapter {
  readonly evalNumber: (expression: SemanticExpression, context: SemanticReferenceCollectiveContext) => number;
  readonly evalExpression: (expression: SemanticExpression, context: SemanticReferenceCollectiveContext) => SemanticReferenceCollectiveValue;
  readonly runFunction: (
    fn: CompiledCudaLiteKernel["kernelIr"]["functions"][number],
    args: readonly SemanticExpression[],
    context: SemanticReferenceCollectiveContext,
    span: SourceSpan,
  ) => SemanticReferenceCollectiveValue | undefined;
  readonly fail: (message: string, span: SourceSpan) => Error;
}

/**
 * Executes CUDA cooperative-group semantics through a narrow reference-runtime
 * adapter. The adapter owns expression/function evaluation; this module owns
 * collective topology, lane selection, and reduction protocol.
 */
export function createSemanticReferenceCollectiveExecutor(
  adapter: SemanticReferenceCollectiveExecutorAdapter,
) {
  const cooperativeContexts = (
    name: string,
    context: SemanticReferenceCollectiveContext,
  ): readonly SemanticReferenceCollectiveContext[] => {
    const group = semanticCooperativeGroupInfo(context.compiled.kernelIr, name);
    if (!group) return [];
    const parent = group.partitionParent
      ? semanticCooperativeGroupInfo(context.compiled.kernelIr, group.partitionParent)
      : undefined;
    const tileSize = group.tileSize ?? parent?.tileSize ?? 32;
    const rank = localLinearRank(context);
    const base = Math.floor(rank / tileSize) * tileSize;
    const peers = (group.partitioned ? context.blockContexts : context.activeCollectiveContexts ?? context.blockContexts).filter((peer) => {
      const peerRank = localLinearRank(peer);
      return peerRank >= base && peerRank < base + tileSize;
    });
    if (!group.partitioned || !group.partitionPredicate) return peers;
    const selected = partitionMembership(name, group.partitionPredicate, context);
    return peers.filter((peer) => partitionMembership(name, group.partitionPredicate!, peer) === selected);
  };

  const partitionMembership = (
    name: string,
    predicate: SemanticExpression,
    context: SemanticReferenceCollectiveContext,
  ): number => {
    const membershipName = partitionMembershipName(name);
    const existing = context.locals.get(membershipName);
    if (typeof existing === "number") return existing;
    const membership = truthy(adapter.evalNumber(predicate, context)) ? 1 : 0;
    context.locals.set(membershipName, membership);
    return membership;
  };

  const cooperativeGroupValue = (
    name: string,
    property: "thread_rank" | "size" | "meta_group_rank" | "meta_group_size",
    context: SemanticReferenceCollectiveContext,
  ): number => {
    const group = semanticCooperativeGroupInfo(context.compiled.kernelIr, name);
    if (!group) throw adapter.fail(`unknown cooperative group '${name}'`, context.compiled.kernelIr.span);
    const workgroupSize = context.blockDim.x * context.blockDim.y * context.blockDim.z;
    const localRank = localLinearRank(context);
    if (property === "meta_group_rank") return group.kind === "tile" ? Math.floor(localRank / (group.tileSize ?? 32)) : 0;
    if (property === "meta_group_size") return group.kind === "tile" ? Math.ceil(workgroupSize / (group.tileSize ?? 32)) : 1;
    const localName = property === "thread_rank"
      ? semanticCooperativeGroupRankParamName(name)
      : semanticCooperativeGroupSizeParamName(name);
    const local = context.locals.get(localName);
    if (typeof local === "number") return local;
    if (group.partitioned && (property === "thread_rank" || property === "size")) {
      const peers = cooperativeContexts(name, context);
      if (property === "size") return peers.length;
      return peers.findIndex((peer) => localLinearRank(peer) === localRank);
    }
    if (property === "thread_rank") {
      if (group.kind === "grid") {
        const blockRank = context.blockIdx.x + context.gridDim.x * (context.blockIdx.y + context.gridDim.y * context.blockIdx.z);
        return localRank + workgroupSize * blockRank;
      }
      if (group.kind === "tile") return localRank % (group.tileSize ?? 32);
      return localRank;
    }
    if (group.kind === "grid") return workgroupSize * context.gridDim.x * context.gridDim.y * context.gridDim.z;
    if (group.kind === "tile") return group.tileSize ?? 32;
    return workgroupSize;
  };

  const evaluateCooperativeGroupCall = (
    expression: Extract<SemanticExpression, { readonly kind: "call" }>,
    context: SemanticReferenceCollectiveContext,
  ): number => {
    if (expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol") {
      throw adapter.fail("semantic reference cooperative-group call requires symbol receiver", expression.span);
    }
    if (expression.callee.property === "ballot" || expression.callee.property === "any" || expression.callee.property === "all") {
      const value = expression.args[0];
      if (!value) throw adapter.fail("semantic cooperative-group vote requires a predicate", expression.span);
      const groupName = expression.callee.object.name;
      const peers = cooperativeContexts(groupName, context);
      if (expression.callee.property === "any") return Number(peers.some((peer) => truthy(adapter.evalNumber(value, peer))));
      if (expression.callee.property === "all") return Number(peers.every((peer) => truthy(adapter.evalNumber(value, peer))));
      return peers.reduce((mask, peer) => truthy(adapter.evalNumber(value, peer))
        ? mask | (1 << cooperativeGroupValue(groupName, "thread_rank", peer))
        : mask, 0) >>> 0;
    }
    if (
      expression.callee.property !== "thread_rank" &&
      expression.callee.property !== "size" &&
      expression.callee.property !== "meta_group_rank" &&
      expression.callee.property !== "meta_group_size"
    ) {
      throw adapter.fail("semantic reference cooperative-group call requires rank, size, or meta-group topology", expression.span);
    }
    return cooperativeGroupValue(expression.callee.object.name, expression.callee.property, context);
  };

  const evaluateCoalescedGroupCall = (
    expression: Extract<SemanticExpression, { readonly kind: "call" }>,
    context: SemanticReferenceCollectiveContext,
  ): number => {
    if (expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol") {
      throw adapter.fail("semantic coalesced-group call requires group receiver", expression.span);
    }
    const peers = cooperativeContexts(expression.callee.object.name, context);
    if (expression.callee.property === "size") return peers.length;
    if (expression.callee.property === "thread_rank") {
      const rank = localLinearRank(context);
      return peers.findIndex((peer) => localLinearRank(peer) === rank);
    }
    if (!expression.args[0]) throw adapter.fail("semantic coalesced-group vote or shuffle requires a value", expression.span);
    if (expression.callee.property === "ballot") {
      return peers.reduce((mask, peer) => truthy(adapter.evalNumber(expression.args[0]!, peer))
        ? mask | (1 << (localLinearRank(peer) % 32))
        : mask, 0) >>> 0;
    }
    if (expression.callee.property === "any") return Number(peers.some((peer) => truthy(adapter.evalNumber(expression.args[0]!, peer))));
    if (expression.callee.property === "all") return Number(peers.every((peer) => truthy(adapter.evalNumber(expression.args[0]!, peer))));
    const sourceLane = expression.args[1] ? Math.trunc(adapter.evalNumber(expression.args[1], context)) : 0;
    const base = Math.floor(localLinearRank(context) / 32) * 32;
    const source = peers.find((peer) => localLinearRank(peer) === base + sourceLane) ?? context;
    return adapter.evalNumber(expression.args[0], source);
  };

  const evaluateCooperativeReduceCall = (
    expression: Extract<SemanticExpression, { readonly kind: "call" }>,
    context: SemanticReferenceCollectiveContext,
  ): SemanticReferenceCollectiveValue => {
    const [groupArg, valueArg] = expression.args;
    if (groupArg?.kind !== "symbol" || !valueArg) {
      throw adapter.fail("semantic cooperative reduce requires group and value", expression.span);
    }
    const valueType = semanticExpressionValueType(valueArg);
    if (valueType !== undefined && isSemanticFloatVectorType(valueType)) {
      const reducerArg = expression.args[2];
      const reducer = reducerArg?.kind === "symbol"
        ? context.compiled.kernelIr.functions.find((fn) => semanticIdsEqual(fn.id, reducerArg.id))
        : undefined;
      if (!reducer) throw adapter.fail("semantic vector cooperative reduce requires resolved reducer", expression.span);
      const values = cooperativeContexts(groupArg.name, context)
        .map((peer) => adapter.evalExpression(valueArg, peer));
      const first = values.shift();
      if (!Array.isArray(first)) throw adapter.fail("semantic vector cooperative reduce requires vector values", valueArg.span);
      return values.reduce<SemanticReferenceCollectiveValue>((left, right) => {
        if (!Array.isArray(left) || !Array.isArray(right)) throw adapter.fail("semantic vector cooperative reduce requires vector values", valueArg.span);
        return evaluateVectorReducer(reducer, left, right, valueType, context, expression.span);
      }, first);
    }
    const operation = cooperativeReductionOperation(expression.args[2]);
    if (operation === undefined) throw adapter.fail("semantic cooperative reduce requires a supported reducer", expression.span);
    const values = cooperativeContexts(groupArg.name, context).map((peer) => adapter.evalNumber(valueArg, peer));
    const first = values.shift();
    if (first === undefined) throw adapter.fail("semantic cooperative reduce has no active lanes", expression.span);
    return values.reduce((left, right) => combineReduction(operation, left, right), first);
  };

  const evaluateVectorReducer = (
    reducer: CompiledCudaLiteKernel["kernelIr"]["functions"][number],
    left: readonly number[],
    right: readonly number[],
    valueType: CudaLiteScalarType,
    context: SemanticReferenceCollectiveContext,
    span: SourceSpan,
  ): SemanticReferenceCollectiveValue => {
    const vectorExpression = (value: readonly number[]): SemanticExpression => ({
      kind: "call",
      callee: { kind: "symbol", id: createBuiltinSemanticSymbolId(`make_${valueType}`), name: `make_${valueType}`, addressSpace: "builtin", span },
      args: value.map((lane) => ({ kind: "literal", literalKind: "number", value: lane, valueType: "float", span })),
      valueType,
      span,
    });
    const { activeCollectiveContexts: _activeCollectiveContexts, ...reducerContext } = context;
    const returnValue = adapter.runFunction(reducer, [vectorExpression(left), vectorExpression(right)], reducerContext, span);
    if (!Array.isArray(returnValue)) throw adapter.fail(`semantic vector reducer '${reducer.name}' did not return vector`, reducer.span);
    return returnValue;
  };

  const evaluateCooperativeScanCall = (
    expression: Extract<SemanticExpression, { readonly kind: "call" }>,
    context: SemanticReferenceCollectiveContext,
  ): number => {
    const [groupArg, valueArg] = expression.args;
    if (groupArg?.kind !== "symbol" || !valueArg || expression.callee.kind !== "symbol") {
      throw adapter.fail("semantic cooperative scan requires group and value", expression.span);
    }
    const rank = localLinearRank(context);
    const peers = cooperativeContexts(groupArg.name, context)
      .filter((peer) => localLinearRank(peer) <= rank);
    if (expression.callee.name.endsWith("::exclusive_scan")) peers.pop();
    return peers.reduce((sum, peer) => sum + adapter.evalNumber(valueArg, peer), 0);
  };

  const evaluateSyncthreadsPredicateCall = (
    expression: Extract<SemanticExpression, { readonly kind: "call" }>,
    context: SemanticReferenceCollectiveContext,
  ): number => {
    if (expression.callee.kind !== "symbol" || !expression.args[0]) {
      throw adapter.fail("semantic syncthreads predicate requires one predicate", expression.span);
    }
    const reduction = cudaSyncthreadsPredicateReduction(expression.callee.name);
    if (reduction === undefined) throw adapter.fail("unknown semantic syncthreads predicate", expression.span);
    const values: number[] = (context.activeCollectiveContexts ?? context.blockContexts)
      .map((peer) => truthy(adapter.evalNumber(expression.args[0]!, peer)) ? 1 : 0);
    if (reduction === "count") return values.reduce((sum, value) => sum + value, 0);
    if (reduction === "and") return Number(values.every(Boolean));
    return Number(values.some(Boolean));
  };

  const recordPartitionMembership = (
    declaration: Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }> ["declaration"],
    context: SemanticReferenceCollectiveContext,
  ): void => {
    if (!declaration.partitionPredicate) return;
    context.locals.set(
      partitionMembershipName(declaration.name),
      truthy(adapter.evalNumber(declaration.partitionPredicate, context)) ? 1 : 0,
    );
  };

  return {
    cooperativeGroupValue,
    evaluateCooperativeGroupCall,
    evaluateCoalescedGroupCall,
    evaluateCooperativeReduceCall,
    evaluateCooperativeScanCall,
    evaluateSyncthreadsPredicateCall,
    recordPartitionMembership,
  };
}

function cooperativeReductionOperation(
  expression: SemanticExpression | undefined,
): CooperativeReductionOperation | undefined {
  if (expression === undefined) return "add";
  if (expression.kind === "symbol") return cooperativeReductionOperationForName(expression.name);
  return expression.kind === "call" && expression.callee.kind === "symbol"
    ? cooperativeReductionOperationForName(expression.callee.name)
    : undefined;
}

function combineReduction(
  operation: CooperativeReductionOperation,
  left: number,
  right: number,
): number {
  if (operation === "min") return Math.min(left, right);
  if (operation === "max") return Math.max(left, right);
  return left + right;
}

function localLinearRank(context: SemanticReferenceCollectiveContext): number {
  return context.threadIdx.x + context.blockDim.x * (context.threadIdx.y + context.blockDim.y * context.threadIdx.z);
}

function partitionMembershipName(name: string): string {
  return `${name}__bg_partition_membership`;
}

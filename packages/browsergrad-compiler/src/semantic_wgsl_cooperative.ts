import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import type { CudaLiteScalarType, SourceSpan } from "./types.js";
import { semanticIdsEqual } from "./semantic_ids.js";
import { semanticExpressionChildren, semanticOperationExpressions } from "./semantic_ir_walk.js";
import { semanticExpressionValueType } from "./semantic_vector_intrinsics.js";
import { wgslValueScalar, wgslValueType } from "./semantic_wgsl_types.js";
import { cudaArithmeticReduceOpForCall, isCudaWarpReduceCallName, isCudaWarpSumCallName } from "./cuda_subgroup_calls.js";
import { isCudaCompatSubgroupReduceCallName } from "./cuda_subgroup_calls.js";
import { semanticBallotHelper } from "./semantic_wgsl_subgroups.js";
import {
  semanticCooperativeGroupInfo,
  semanticCooperativeGroupRankParamName,
  semanticCooperativeGroupSizeParamName,
} from "./semantic_cooperative_groups.js";
import { cooperativeReductionOperationForName, type CooperativeReductionOperation } from "./cooperative_reduction.js";
import {
  convertTypedWgslExpression,
  createTypedWgslCall,
  createTypedWgslIdentifier,
  createTypedWgslLiteral,
  createTypedWgslMemberAccess,
  emitTypedWgslBinary,
  emitTypedWgslSelect,
  emitTypedWgslUnary,
  type TypedWgslExpression,
} from "./typed_wgsl_expression.js";

export interface SemanticCooperativeReduceHelper {
  readonly name: string;
  readonly scratchName: string;
  readonly valueType: Exclude<ReturnType<typeof semanticExpressionValueType>, undefined | "void">;
  readonly tileSize: number;
  readonly masked: boolean;
  readonly partitioned: boolean;
  readonly operation: "add" | "min" | "max";
}

export interface SemanticCooperativeScanHelper {
  readonly name: string;
  readonly scratchName: string;
  readonly valueType: "float" | "double" | "half" | "int" | "uint";
  readonly tileSize: number;
  readonly inclusive: boolean;
}

export interface SemanticCooperativeVectorReduceHelper {
  readonly name: string;
  readonly scratchName: string;
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly tileSize: number;
  readonly reducerName: string;
}

export interface SemanticCooperativePartitionMaskDeclaration {
  readonly trueMaskName: string;
  readonly trueMaskValue: TypedWgslExpression;
  readonly name: string;
  readonly value: TypedWgslExpression;
}

export function semanticCooperativePartitionDeclarations(
  ir: SemanticKernelIrModule,
): readonly Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }>[] {
  return [...ir.operations, ...ir.functions.flatMap((fn) => fn.body)]
    .flatMap(semanticOperationsDeep)
    .filter((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }> =>
      operation.kind === "cooperative-group-declare" && operation.declaration.partitionPredicate !== undefined,
    );
}

export function semanticCooperativePartitionMaskName(groupName: string): string {
  return `${groupName}__bg_partition_mask`;
}

export function emitSemanticTypedCooperativePartitionMaskDeclaration(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }>,
  ir: SemanticKernelIrModule,
  predicate: TypedWgslExpression,
  resolveName: (name: string) => string = (name) => name,
): SemanticCooperativePartitionMaskDeclaration | undefined {
  const group = semanticCooperativeGroupInfo(ir, operation.declaration.name);
  if (!group?.partitioned || !group.partitionPredicate) return undefined;
  const trueMaskName = resolveName(`${operation.declaration.name}__bg_partition_true_mask`);
  const trueMaskValue = emitSemanticTypedCooperativePartitionBallotMask(group, ir, predicate, operation.span);
  const trueMask = createTypedWgslIdentifier(trueMaskName, "u32", operation.span);
  const tileMask = semanticTypedCooperativePartitionLowMask(group, ir, operation.span);
  return {
    trueMaskName,
    trueMaskValue,
    name: resolveName(semanticCooperativePartitionMaskName(operation.declaration.name)),
    value: emitTypedWgslSelect(
      emitTypedWgslBinary("&", emitTypedWgslUnary("~", trueMask, operation.span), tileMask, operation.span),
      trueMask,
      predicate,
      operation.span,
    ),
  };
}

export function emitSemanticTypedCooperativePartitionBallotMask(
  group: NonNullable<ReturnType<typeof semanticCooperativeGroupInfo>>,
  ir: SemanticKernelIrModule,
  predicate: TypedWgslExpression,
  span: SourceSpan,
): TypedWgslExpression {
  const localRank = semanticTypedCooperativeLocalLinearRank(ir, span);
  const tileSize = semanticCooperativePartitionTileSize(group, ir);
  const tileLane = emitTypedWgslBinary("%", localRank, semanticTypedU32(tileSize, span), span);
  const warpLane = emitTypedWgslBinary("%", localRank, semanticTypedU32(32, span), span);
  const tileBase = emitTypedWgslBinary("-", warpLane, tileLane, span);
  const tileMask = semanticTypedCooperativePartitionLowMask(group, ir, span);
  const activeMask = emitTypedWgslBinary("<<", tileMask, tileBase, span);
  const ballot = createTypedWgslCall(
    semanticBallotHelper().name,
    [predicate, activeMask, createTypedWgslIdentifier("local_id", "vec3<u32>", span)],
    "u32",
    span,
  );
  return emitTypedWgslBinary(">>", ballot, tileBase, span);
}

export function emitSemanticTypedCooperativeGroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  activeFunction?: string,
): TypedWgslExpression | undefined {
  if (!semanticWgslCooperativeGroupCallSupported(expression, ir) || expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol") return undefined;
  const span = expression.span;
  const groupName = expression.callee.object.name;
  const group = semanticCooperativeGroupInfo(ir, groupName)!;
  const resultType = expression.valueType === "uint" ? "u32" : "i32";
  const groupParam = activeFunction === undefined ? undefined : ir.functions.find((fn) => fn.name === activeFunction)?.params
    .find((param) => param.name === groupName && param.cooperativeGroupKind !== undefined);
  if (groupParam && (expression.callee.property === "thread_rank" || expression.callee.property === "size")) {
    const suffix = expression.callee.property === "thread_rank" ? semanticCooperativeGroupRankParamName(groupParam.name) : semanticCooperativeGroupSizeParamName(groupParam.name);
    return convertTypedWgslExpression(createTypedWgslIdentifier(suffix, "i32", span), resultType);
  }
  const localRank = semanticTypedCooperativeLocalLinearRank(ir, span);
  const asResult = (value: TypedWgslExpression): TypedWgslExpression => convertTypedWgslExpression(value, resultType);
  if (group.partitioned && (expression.callee.property === "thread_rank" || expression.callee.property === "size")) {
    const mask = createTypedWgslIdentifier(semanticCooperativePartitionMaskName(groupName), "u32", span);
    if (expression.callee.property === "size") return asResult(createTypedWgslCall("countOneBits", [mask], "u32", span));
    const lane = emitTypedWgslBinary("%", localRank, semanticTypedU32(semanticCooperativePartitionTileSize(group, ir), span), span);
    const lower = emitTypedWgslBinary("-", emitTypedWgslBinary("<<", semanticTypedU32(1, span), lane, span), semanticTypedU32(1, span), span);
    return asResult(createTypedWgslCall("countOneBits", [emitTypedWgslBinary("&", mask, lower, span)], "u32", span));
  }
  if (expression.callee.property === "thread_rank") {
    if (group.kind === "grid") return asResult(semanticTypedCooperativeGlobalLinearRank(ir, span));
    if (group.kind === "tile") return asResult(emitTypedWgslBinary("%", localRank, semanticTypedU32(group.tileSize ?? 32, span), span));
    return asResult(localRank);
  }
  if (expression.callee.property === "size") {
    if (group.kind === "grid") return asResult(semanticTypedCooperativeGridThreadCount(ir, span));
    return asResult(semanticTypedU32(group.kind === "tile" ? group.tileSize ?? 32 : semanticCooperativeWorkgroupSize(ir), span));
  }
  if (expression.callee.property === "meta_group_rank") {
    return asResult(group.kind === "tile" ? emitTypedWgslBinary("/", localRank, semanticTypedU32(group.tileSize ?? 32, span), span) : semanticTypedU32(0, span));
  }
  if (expression.callee.property === "meta_group_size") {
    return asResult(semanticTypedU32(group.kind === "tile" ? Math.ceil(semanticCooperativeWorkgroupSize(ir) / (group.tileSize ?? 32)) : 1, span));
  }
  return undefined;
}

export function semanticWgslCooperativeGroupCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol" || expression.args.length !== 0) return false;
  if (
    expression.callee.property !== "thread_rank" &&
    expression.callee.property !== "size" &&
    expression.callee.property !== "meta_group_rank" &&
    expression.callee.property !== "meta_group_size"
  ) return false;
  const group = semanticCooperativeGroupInfo(ir, expression.callee.object.name);
  return group !== undefined && group.kind !== "coalesced";
}

export function semanticWgslCooperativeReduceCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  valueSupported: (value: SemanticExpression) => boolean,
): boolean {
  const helper = semanticCooperativeReduceHelperFor(ir, expression);
  const value = semanticCooperativeReduceValue(expression);
  return helper !== undefined && value !== undefined && valueSupported(value);
}

export function semanticWgslCooperativeScanCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  valueSupported: (value: SemanticExpression) => boolean,
): boolean {
  const helper = semanticCooperativeScanHelperFor(ir, expression);
  const value = expression.args[1];
  return helper !== undefined && value !== undefined && valueSupported(value);
}

export function emitSemanticCooperativeGroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  activeFunction?: string,
  partitionScratchName?: string,
): string | undefined {
  if (expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol") return undefined;
  const groupName = expression.callee.object.name;
  const group = semanticCooperativeGroupInfo(ir, groupName);
  if (!group) return undefined;
  const groupParam = activeFunction === undefined
    ? undefined
    : ir.functions.find((fn) => fn.name === activeFunction)?.params
      .find((param) => param.name === groupName && param.cooperativeGroupKind !== undefined);
  if (group.partitioned && partitionScratchName !== undefined) {
    const tileSize = group.tileSize ?? 32;
    const workgroupSize = semanticCooperativeWorkgroupSize(ir);
    const lane = `((${semanticCooperativeLocalLinearRank(ir)}) % ${tileSize}u)`;
    const base = `((${semanticCooperativeLocalLinearRank(ir)}) - ${lane})`;
    const terms = Array.from({ length: tileSize }, (_, index) =>
      `select(0u, ${partitionScratchName}[${base} + ${index}u], ${index}u < ${lane} && (${base} + ${index}u) < ${workgroupSize}u)`);
    const sizeTerms = Array.from({ length: tileSize }, (_, index) =>
      `select(0u, ${partitionScratchName}[${base} + ${index}u], (${base} + ${index}u) < ${workgroupSize}u)`);
    if (expression.callee.property === "thread_rank") {
      const rank = terms.length === 0 ? "0u" : `(${terms.join(" + ")})`;
      return expression.valueType === "uint" ? rank : `i32(${rank})`;
    }
    if (expression.callee.property === "size") {
      const size = sizeTerms.length === 0 ? "0u" : `(${sizeTerms.join(" + ")})`;
      return expression.valueType === "uint" ? size : `i32(${size})`;
    }
  }
  if (expression.callee.property === "thread_rank") {
    if (groupParam) return semanticCooperativeGroupRankParamName(groupParam.name);
    const rank = group.kind === "grid"
      ? semanticCooperativeGlobalLinearRank(ir)
      : group.kind === "tile"
      ? `i32((${semanticCooperativeLocalLinearRank(ir)}) % ${group.tileSize ?? 32}u)`
      : `i32(${semanticCooperativeLocalLinearRank(ir)})`;
    return expression.valueType === "uint" ? `u32(${rank})` : rank;
  }
  if (expression.callee.property === "size") {
    if (groupParam) return semanticCooperativeGroupSizeParamName(groupParam.name);
    const size = group.kind === "grid"
      ? semanticCooperativeGridThreadCount(ir)
      : group.kind === "tile"
      ? String(group.tileSize ?? 32)
      : String(semanticCooperativeWorkgroupSize(ir));
    return expression.valueType === "uint" ? `u32(${size})` : size;
  }
  if (expression.callee.property === "meta_group_rank") {
    const rank = group.kind === "tile"
      ? `i32((${semanticCooperativeLocalLinearRank(ir)}) / ${group.tileSize ?? 32}u)`
      : "0";
    return expression.valueType === "uint" ? `u32(${rank})` : rank;
  }
  if (expression.callee.property === "meta_group_size") {
    const size = group.kind === "tile"
      ? Math.ceil(semanticCooperativeWorkgroupSize(ir) / (group.tileSize ?? 32))
      : 1;
    return expression.valueType === "uint" ? `${size}u` : String(size);
  }
  return undefined;
}

export function emitSemanticCooperativeReduceCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  value: string,
  mask?: string,
): string | undefined {
  const helper = semanticCooperativeReduceHelperFor(ir, expression);
  if (helper === undefined) return undefined;
  if (helper.masked && mask === undefined) return undefined;
  return `${helper.name}(${value}${helper.masked ? `, ${mask}` : ""}, local_id)`;
}

export function semanticCooperativeReduceHelperFor(
  ir: SemanticKernelIrModule,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): SemanticCooperativeReduceHelper | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace === "function") return undefined;
  if (isCudaCompatSubgroupReduceCallName(expression.callee.name)) return undefined;
  const arithmeticOperation = cudaArithmeticReduceOpForCall(expression.callee.name);
  if (ir.subgroupMode === "scalar" && arithmeticOperation !== undefined) return undefined;
  const logicalWarpOperation = isCudaWarpSumCallName(expression.callee.name) ? "add" : arithmeticOperation;
  if (logicalWarpOperation) {
    const valueArg = semanticCooperativeReduceValue(expression);
    const valueType = valueArg ? semanticExpressionValueType(valueArg) : undefined;
    if (valueType !== "float" && valueType !== "double" && valueType !== "half" && valueType !== "int" && valueType !== "uint") return undefined;
    const typeName = wgslValueScalar(valueType).replace(/[^A-Za-z0-9_]/gu, "_");
    const masked = expression.args.length === 2;
    return {
      name: `bg_semantic_warp_reduce_${logicalWarpOperation === "add" ? "sum" : logicalWarpOperation}_${typeName}_32${masked ? "_masked" : ""}`,
      scratchName: `bg_semantic_warp_reduce_${logicalWarpOperation === "add" ? "sum" : logicalWarpOperation}_${typeName}_32${masked ? "_masked" : ""}_scratch`,
      valueType,
      tileSize: 32,
      masked,
      partitioned: false,
      operation: logicalWarpOperation,
    };
  }
  if (!isCooperativeReduceName(expression.callee.name)) return undefined;
  const [groupArg, valueArg, operationArg] = expression.args;
  const operation = semanticCooperativeReductionOperation(operationArg);
  if (!groupArg || groupArg.kind !== "symbol" || !valueArg || operation === undefined) return undefined;
  const group = semanticCooperativeGroupInfo(ir, groupArg.name);
  if (!group || (group.kind !== "tile" && group.kind !== "thread" && group.kind !== "binary")) return undefined;
  const valueType = semanticExpressionValueType(valueArg);
  if (valueType !== "float" && valueType !== "double" && valueType !== "half" && valueType !== "int" && valueType !== "uint") return undefined;
  const parent = group.partitionParent ? semanticCooperativeGroupInfo(ir, group.partitionParent) : undefined;
  const tileSize = group.tileSize ?? parent?.tileSize ?? 32;
  const typeName = wgslValueScalar(valueType).replace(/[^A-Za-z0-9_]/gu, "_");
  const operationName = operation === "add" ? "" : `_${operation}`;
  return {
    name: `bg_semantic_cg_reduce${operationName}_${typeName}_${tileSize}`,
    scratchName: `bg_semantic_cg_reduce${operationName}_${typeName}_${tileSize}_scratch`,
    valueType,
    tileSize,
    masked: group.partitioned === true || group.kind === "binary",
    partitioned: group.partitioned === true || group.kind === "binary",
    operation,
  };
}

export function semanticCooperativeReduceValue(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): SemanticExpression | undefined {
  if (expression.callee.kind === "symbol" &&
    (isCudaWarpReduceCallName(expression.callee.name) || cudaArithmeticReduceOpForCall(expression.callee.name) === "add")) {
    return expression.args.at(-1);
  }
  return expression.args[1];
}

export function semanticCooperativeReduceHelpers(
  ir: SemanticKernelIrModule,
): readonly SemanticCooperativeReduceHelper[] {
  const helpers = new Map<string, SemanticCooperativeReduceHelper>();
  for (const expression of semanticModuleExpressions(ir)) {
    if (expression.kind !== "call") continue;
    const helper = semanticCooperativeReduceHelperFor(ir, expression);
    if (helper) helpers.set(helper.name, helper);
  }
  return [...helpers.values()];
}

export function semanticCooperativeScanHelperFor(
  ir: SemanticKernelIrModule,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): SemanticCooperativeScanHelper | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace === "function") return undefined;
  const inclusive = expression.callee.name === "cg::inclusive_scan" || expression.callee.name === "cooperative_groups::inclusive_scan";
  const exclusive = expression.callee.name === "cg::exclusive_scan" || expression.callee.name === "cooperative_groups::exclusive_scan";
  if (!inclusive && !exclusive) return undefined;
  const [groupArg, valueArg, operationArg] = expression.args;
  if (groupArg?.kind !== "symbol" || !valueArg || operationArg !== undefined && !isPlusOperation(operationArg)) return undefined;
  const group = semanticCooperativeGroupInfo(ir, groupArg.name);
  if (!group || group.kind === "grid" || group.kind === "coalesced" || group.partitioned) return undefined;
  const valueType = semanticExpressionValueType(valueArg);
  if (valueType !== "float" && valueType !== "double" && valueType !== "half" && valueType !== "int" && valueType !== "uint") return undefined;
  const tileSize = Math.min(group.tileSize ?? semanticCooperativeWorkgroupSize(ir), semanticCooperativeWorkgroupSize(ir));
  const typeName = wgslValueScalar(valueType).replace(/[^A-Za-z0-9_]/gu, "_");
  const name = `bg_semantic_cg_${inclusive ? "inclusive" : "exclusive"}_scan_sum_${typeName}_${tileSize}`;
  return { name, scratchName: `${name}_scratch`, valueType, tileSize, inclusive };
}

export function semanticCooperativeScanHelpers(
  ir: SemanticKernelIrModule,
): readonly SemanticCooperativeScanHelper[] {
  const helpers = new Map<string, SemanticCooperativeScanHelper>();
  for (const expression of semanticModuleExpressions(ir)) {
    if (expression.kind !== "call") continue;
    const helper = semanticCooperativeScanHelperFor(ir, expression);
    if (helper) helpers.set(helper.name, helper);
  }
  return [...helpers.values()];
}

export function semanticCooperativeVectorReduceHelperFor(
  ir: SemanticKernelIrModule,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): SemanticCooperativeVectorReduceHelper | undefined {
  if (expression.callee.kind !== "symbol" || !isCooperativeReduceName(expression.callee.name)) return undefined;
  const [groupArg, valueArg, reducerArg] = expression.args;
  if (groupArg?.kind !== "symbol" || !valueArg || reducerArg?.kind !== "symbol" || reducerArg.addressSpace !== "function") return undefined;
  const group = semanticCooperativeGroupInfo(ir, groupArg.name);
  if (!group || group.kind !== "tile" || group.partitioned) return undefined;
  const valueType = semanticExpressionValueType(valueArg);
  if (valueType !== "float2" && valueType !== "float3" && valueType !== "float4") return undefined;
  const reducer = ir.functions.find((fn) => semanticIdsEqual(fn.id, reducerArg.id));
  if (!reducer || reducer.returnType !== valueType || reducer.params.length !== 2 || reducer.params.some((param) => param.valueType !== valueType || param.pointer)) return undefined;
  const tileSize = group.tileSize ?? 32;
  const name = `bg_semantic_cg_reduce_${reducer.name}_${valueType}_${tileSize}`;
  return { name, scratchName: `${name}_scratch`, valueType, tileSize, reducerName: reducer.name };
}

export function semanticCooperativeVectorReduceHelpers(
  ir: SemanticKernelIrModule,
): readonly SemanticCooperativeVectorReduceHelper[] {
  const helpers = new Map<string, SemanticCooperativeVectorReduceHelper>();
  for (const expression of semanticModuleExpressions(ir)) {
    if (expression.kind !== "call") continue;
    const helper = semanticCooperativeVectorReduceHelperFor(ir, expression);
    if (helper) helpers.set(helper.name, helper);
  }
  return [...helpers.values()];
}

export function emitSemanticCooperativeVectorReduceHelper(
  helper: SemanticCooperativeVectorReduceHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueType(helper.valueType);
  const workgroupSize = semanticCooperativeWorkgroupSize(ir);
  const start = Math.max(1, Math.floor(Math.min(helper.tileSize, workgroupSize) / 2));
  return [
    `fn ${helper.name}(value_arg: ${type}, local_id: vec3<u32>, workgroup_id: vec3<u32>, num_workgroups: vec3<u32>) -> ${type} {`,
    `  let rank: u32 = ${semanticCooperativeLocalLinearRank(ir)};`,
    `  let width: u32 = min(${helper.tileSize}u, ${workgroupSize}u);`,
    "  let lane: u32 = rank % width;",
    "  let base: u32 = rank - lane;",
    `  ${helper.scratchName}[rank] = value_arg;`,
    "  workgroupBarrier();",
    `  var stride: u32 = ${start}u;`,
    "  while (stride > 0u) {",
    `    if (lane < stride && (lane + stride) < width && (rank + stride) < ${workgroupSize}u) {`,
    `      ${helper.scratchName}[rank] = ${helper.reducerName}(${helper.scratchName}[rank], ${helper.scratchName}[rank + stride], local_id, workgroup_id, num_workgroups);`,
    "    }",
    "    workgroupBarrier();",
    "    stride = stride / 2u;",
    "  }",
    `  let result: ${type} = ${helper.scratchName}[base];`,
    "  workgroupBarrier();",
    "  return result;",
    "}",
  ];
}

export function emitSemanticCooperativeScanHelper(
  helper: SemanticCooperativeScanHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = semanticCooperativeWorkgroupSize(ir);
  const zero = type === "u32" ? "0u" : type === "i32" ? "0" : type === "f16" ? "f16(0.0)" : "0.0";
  return [
    `fn ${helper.name}(value_arg: ${type}, local_id: vec3<u32>) -> ${type} {`,
    `  let rank: u32 = ${semanticCooperativeLocalLinearRank(ir)};`,
    `  let width: u32 = min(${helper.tileSize}u, ${workgroupSize}u);`,
    "  let lane: u32 = rank % width;",
    `  ${helper.scratchName}[rank] = value_arg;`,
    "  workgroupBarrier();",
    "  var stride: u32 = 1u;",
    "  while (stride < width) {",
    `    var addend: ${type} = ${zero};`,
    "    if (lane >= stride) {",
    `      addend = ${helper.scratchName}[rank - stride];`,
    "    }",
    "    workgroupBarrier();",
    "    if (lane >= stride) {",
    `      ${helper.scratchName}[rank] = ${helper.scratchName}[rank] + addend;`,
    "    }",
    "    workgroupBarrier();",
    "    stride = stride * 2u;",
    "  }",
    `  var result: ${type} = ${zero};`,
    ...(helper.inclusive
      ? [`  result = ${helper.scratchName}[rank];`]
      : ["  if (lane > 0u) {", `    result = ${helper.scratchName}[rank - 1u];`, "  }"]),
    "  workgroupBarrier();",
    "  return result;",
    "}",
  ];
}

export function emitSemanticCooperativeReduceHelper(
  helper: SemanticCooperativeReduceHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = ir.workgroupSize[0] * ir.workgroupSize[1] * ir.workgroupSize[2];
  const start = Math.max(1, Math.floor(Math.min(helper.tileSize, workgroupSize) / 2));
  return [
    `fn ${helper.name}(value_arg: ${type}${helper.masked ? ", mask_arg: u32" : ""}, local_id: vec3<u32>) -> ${type} {`,
    `  let rank: u32 = ${semanticCooperativeLocalLinearRank(ir)};`,
    `  let width: u32 = min(${helper.tileSize}u, ${workgroupSize}u);`,
    "  let lane: u32 = rank % width;",
    "  let base: u32 = rank - lane;",
    `  ${helper.scratchName}[rank] = ${helper.masked ? "select(" + identityForReduction(type, helper.operation) + ", value_arg, (mask_arg & (1u << lane)) != 0u)" : "value_arg"};`,
    "  workgroupBarrier();",
    `  var stride: u32 = ${start}u;`,
    "  while (stride > 0u) {",
    `    if (lane < stride && (lane + stride) < width && (rank + stride) < ${workgroupSize}u) {`,
    `      ${helper.scratchName}[rank] = ${combineReductionValues(helper.scratchName, helper.operation)};`,
    "    }",
    "    workgroupBarrier();",
    "    stride = stride / 2u;",
    "  }",
    `  let result: ${type} = ${helper.scratchName}[base];`,
    "  workgroupBarrier();",
    "  return result;",
    "}",
  ];
}

function identityForReduction(type: ReturnType<typeof wgslValueScalar>, operation: "add" | "min" | "max"): string {
  if (operation === "min") {
    if (type === "u32") return "0xffffffffu";
    if (type === "i32") return "2147483647";
    if (type === "f16") return "f16(65504.0)";
    return "3.402823466e+38";
  }
  if (operation === "max") {
    if (type === "u32") return "0u";
    if (type === "i32") return "(-2147483647 - 1)";
    if (type === "f16") return "f16(-65504.0)";
    return "-3.402823466e+38";
  }
  if (type === "u32") return "0u";
  if (type === "i32") return "0";
  if (type === "f16") return "f16(0.0)";
  return "0.0";
}

function combineReductionValues(scratchName: string, operation: "add" | "min" | "max"): string {
  const left = `${scratchName}[rank]`;
  const right = `${scratchName}[rank + stride]`;
  return operation === "add" ? `${left} + ${right}` : `${operation}(${left}, ${right})`;
}

function semanticModuleExpressions(ir: SemanticKernelIrModule): readonly SemanticExpression[] {
  return [...ir.operations, ...ir.functions.flatMap((fn) => fn.body)].flatMap(semanticOperationExpressionsDeep);
}

function semanticOperationExpressionsDeep(operation: SemanticKernelIrOperation): readonly SemanticExpression[] {
  const own = semanticOperationExpressions(operation).flatMap(semanticExpressionDeep);
  if (operation.kind === "branch") return [...own, ...operation.consequent.flatMap(semanticOperationExpressionsDeep), ...operation.alternate.flatMap(semanticOperationExpressionsDeep)];
  if (operation.kind === "loop") return [...own, ...operation.body.flatMap(semanticOperationExpressionsDeep)];
  if (operation.kind === "block") return [...own, ...operation.body.flatMap(semanticOperationExpressionsDeep)];
  return own;
}

function semanticExpressionDeep(expression: SemanticExpression): readonly SemanticExpression[] {
  return [expression, ...semanticExpressionChildren(expression).flatMap(semanticExpressionDeep)];
}

function isCooperativeReduceName(name: string): boolean {
  return name === "cg::reduce" || name === "cooperative_groups::reduce";
}

function isPlusOperation(expression: SemanticExpression | undefined): boolean {
  return semanticCooperativeReductionOperation(expression) === "add";
}

function semanticCooperativeReductionOperation(
  expression: SemanticExpression | undefined,
): CooperativeReductionOperation | undefined {
  if (expression?.kind === "symbol" && expression.addressSpace === "builtin") {
    return cooperativeReductionOperationForName(expression.name);
  }
  return expression?.kind === "call" && expression.callee.kind === "symbol"
    ? cooperativeReductionOperationForName(expression.callee.name)
    : undefined;
}

function semanticCooperativeWorkgroupSize(ir: SemanticKernelIrModule): number {
  return ir.workgroupSize[0] * ir.workgroupSize[1] * ir.workgroupSize[2];
}

function semanticCooperativeLocalLinearRank(ir: SemanticKernelIrModule): string {
  const [x, y, z] = ir.workgroupSize;
  if (y === 1 && z === 1) return "local_id.x";
  if (z === 1) return `local_id.x + local_id.y * ${x}u`;
  return `local_id.x + local_id.y * ${x}u + local_id.z * ${x * y}u`;
}

function semanticCooperativeGlobalLinearRank(ir: SemanticKernelIrModule): string {
  return `(i32(${semanticCooperativeLocalLinearRank(ir)}) + i32(${semanticCooperativeWorkgroupSize(ir)}u * (workgroup_id.x + workgroup_id.y * num_workgroups.x + workgroup_id.z * num_workgroups.x * num_workgroups.y)))`;
}

function semanticCooperativeGridThreadCount(ir: SemanticKernelIrModule): string {
  return `i32(${semanticCooperativeWorkgroupSize(ir)}u * num_workgroups.x * num_workgroups.y * num_workgroups.z)`;
}

function semanticOperationsDeep(operation: SemanticKernelIrOperation): readonly SemanticKernelIrOperation[] {
  if (operation.kind === "branch") return [operation, ...operation.consequent.flatMap(semanticOperationsDeep), ...operation.alternate.flatMap(semanticOperationsDeep)];
  if (operation.kind === "loop" || operation.kind === "block") return [operation, ...operation.body.flatMap(semanticOperationsDeep)];
  return [operation];
}

function semanticCooperativePartitionTileSize(
  group: NonNullable<ReturnType<typeof semanticCooperativeGroupInfo>>,
  ir: SemanticKernelIrModule,
): number {
  const parent = group.partitionParent ? semanticCooperativeGroupInfo(ir, group.partitionParent) : undefined;
  return Math.max(1, Math.min(32, group.tileSize ?? parent?.tileSize ?? semanticCooperativeWorkgroupSize(ir)));
}

function semanticTypedCooperativePartitionLowMask(
  group: NonNullable<ReturnType<typeof semanticCooperativeGroupInfo>>,
  ir: SemanticKernelIrModule,
  span: SourceSpan,
): TypedWgslExpression {
  const tileSize = semanticCooperativePartitionTileSize(group, ir);
  return createTypedWgslLiteral(tileSize === 32 ? "0xffffffffu" : `${2 ** tileSize - 1}u`, "u32", span);
}

function semanticTypedU32(value: number, span: SourceSpan): TypedWgslExpression {
  return createTypedWgslLiteral(`${value}u`, "u32", span);
}

function semanticTypedCooperativeLocalLinearRank(ir: SemanticKernelIrModule, span: SourceSpan): TypedWgslExpression {
  const [x, y, z] = ir.workgroupSize;
  const localId = createTypedWgslIdentifier("local_id", "vec3<u32>", span);
  const lane = (field: "x" | "y" | "z"): TypedWgslExpression => createTypedWgslMemberAccess(localId, field, "u32", span);
  let rank = lane("x");
  if (y !== 1 || z !== 1) rank = emitTypedWgslBinary("+", rank, emitTypedWgslBinary("*", lane("y"), semanticTypedU32(x, span), span), span);
  if (z !== 1) rank = emitTypedWgslBinary("+", rank, emitTypedWgslBinary("*", lane("z"), semanticTypedU32(x * y, span), span), span);
  return rank;
}

function semanticTypedCooperativeGlobalLinearRank(ir: SemanticKernelIrModule, span: SourceSpan): TypedWgslExpression {
  const workgroupId = createTypedWgslIdentifier("workgroup_id", "vec3<u32>", span);
  const numWorkgroups = createTypedWgslIdentifier("num_workgroups", "vec3<u32>", span);
  const member = (object: TypedWgslExpression, field: "x" | "y" | "z"): TypedWgslExpression => createTypedWgslMemberAccess(object, field, "u32", span);
  const product = (left: TypedWgslExpression, right: TypedWgslExpression): TypedWgslExpression => emitTypedWgslBinary("*", left, right, span);
  const sum = (left: TypedWgslExpression, right: TypedWgslExpression): TypedWgslExpression => emitTypedWgslBinary("+", left, right, span);
  const flatWorkgroup = sum(member(workgroupId, "x"), sum(product(member(workgroupId, "y"), member(numWorkgroups, "x")), product(member(workgroupId, "z"), product(member(numWorkgroups, "x"), member(numWorkgroups, "y")))));
  return sum(semanticTypedCooperativeLocalLinearRank(ir, span), product(semanticTypedU32(semanticCooperativeWorkgroupSize(ir), span), flatWorkgroup));
}

function semanticTypedCooperativeGridThreadCount(ir: SemanticKernelIrModule, span: SourceSpan): TypedWgslExpression {
  const numWorkgroups = createTypedWgslIdentifier("num_workgroups", "vec3<u32>", span);
  return (["x", "y", "z"] as const).reduce<TypedWgslExpression>(
    (value, field) => emitTypedWgslBinary("*", value, createTypedWgslMemberAccess(numWorkgroups, field, "u32", span), span),
    semanticTypedU32(semanticCooperativeWorkgroupSize(ir), span),
  );
}

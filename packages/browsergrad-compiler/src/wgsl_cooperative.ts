import { expressionName } from "./analyzer.js";
import { isCudaVectorType, type CudaLiteVectorType } from "./vector_types.js";
import {
  type EmitContext,
  type ScalarWarpReduceHelper,
  type ScalarWarpShuffleHelper,
  type VectorCooperativeReduceHelper,
} from "./wgsl_context.js";
import { safeWgslIdentifier } from "./wgsl_names.js";
import { wgslScalar, zeroValue } from "./wgsl_storage.js";
import {
  CudaLiteCompilerError,
  type CudaLiteCallExpression,
  type CudaLiteCooperativeGroupDecl,
  type CudaLiteExpression,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type SourceSpan,
} from "./types.js";

export interface WgslCooperativeCallbacks {
  emitExpression(expression: CudaLiteExpression): string;
  emitTruthinessExpression(expression: CudaLiteExpression): string;
  emitExpressionAsValueType(expression: CudaLiteExpression, valueType: CudaLiteScalarType): string;
  emitExpressionAsWgslScalar(expression: CudaLiteExpression, wgslType: "f32" | "f16" | "i32" | "u32"): string;
  expressionValueType(expression: CudaLiteExpression): CudaLiteScalarType | undefined;
}

export function emitCooperativeGroupCall(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string | undefined {
  const namespaceCall = emitCooperativeNamespaceCall(expression, context, callbacks);
  if (namespaceCall !== undefined) return namespaceCall;
  const callee = expression.callee;
  if (callee.kind !== "member" || callee.object.kind !== "identifier") return undefined;
  const group = context.cooperativeGroupFor(callee.object.name);
  if (!group) return undefined;
  if (callee.property === "sync") {
    return group.groupKind === "grid" ? "0" : "workgroupBarrier()";
  }
  if (callee.property === "size") {
    const partitionSize = emitCooperativePartitionSize(group, context, callbacks);
    if (partitionSize) return partitionSize;
    if (group.groupKind === "thread" && group.dynamicTileSizeName) return `i32(${group.dynamicTileSizeName})`;
    if (group.groupKind === "tile") return String(group.tileSize ?? 32);
    return String(context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2]);
  }
  if (callee.property === "thread_rank") {
    const partitionRank = emitCooperativePartitionRank(group, context, callbacks);
    if (partitionRank) return partitionRank;
    const localRank = emitLocalLinearRank(context);
    if (group.groupKind === "thread" && group.dynamicTileSizeName) return `(${localRank} % i32(${group.dynamicTileSizeName}))`;
    if (group.groupKind === "tile") return `(${localRank} % ${group.tileSize ?? 32})`;
    return localRank;
  }
  if (callee.property === "meta_group_size") {
    if (group.groupKind === "thread" && group.dynamicTileSizeName) {
      const blockSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
      return `i32((${blockSize}u + ${group.dynamicTileSizeName} - 1u) / ${group.dynamicTileSizeName})`;
    }
    if (group.groupKind !== "tile") return "1";
    const blockSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
    return String(Math.ceil(blockSize / (group.tileSize ?? 32)));
  }
  if (callee.property === "meta_group_rank") {
    if (group.groupKind === "thread" && group.dynamicTileSizeName) return `(${emitLocalLinearRank(context)} / i32(${group.dynamicTileSizeName}))`;
    if (group.groupKind !== "tile") return "0";
    return `(${emitLocalLinearRank(context)} / ${group.tileSize ?? 32})`;
  }
  if (callee.property === "shfl" || callee.property === "shfl_down" || callee.property === "shfl_up" || callee.property === "shfl_xor") {
    const valueExpression = expression.args[0];
    const offsetExpression = expression.args[1];
    if (!valueExpression) return "0";
    const value = callbacks.emitExpression(valueExpression);
    if (context.subgroupMode === "scalar") return value;
    const op = callee.property === "shfl"
      ? "sync"
      : callee.property === "shfl_up"
      ? "up"
      : callee.property === "shfl_xor"
        ? "xor"
        : "down";
    const valueType = scalarWarpValueType(valueExpression, callbacks, "cooperative group shuffle");
    const helper = registerScalarWarpShuffleHelper(context, op, valueType, maxCooperativeGroupTileSize(group, context));
    const offset = offsetExpression ? callbacks.emitExpressionAsWgslScalar(offsetExpression, "u32") : "0u";
    return `${helper.name}(${callbacks.emitExpressionAsValueType(valueExpression, valueType)}, ${offset}, ${emitCooperativeGroupTileSizeValue(group, context)}, local_id)`;
  }
  if (callee.property === "ballot") {
    const predicate = expression.args[0] ? callbacks.emitTruthinessExpression(expression.args[0]) : "false";
    if (context.subgroupMode === "scalar") return `select(0u, 1u, ${predicate})`;
    return `subgroupBallot(${predicate}).x`;
  }
  if (callee.property === "any" || callee.property === "all") {
    const predicate = expression.args[0] ? callbacks.emitTruthinessExpression(expression.args[0]) : "false";
    if (context.subgroupMode === "scalar") return predicate;
    return `${callee.property === "any" ? "subgroupAny" : "subgroupAll"}(${predicate})`;
  }
  return undefined;
}

export function emitScalarWarpReduceCall(
  op: ScalarWarpReduceHelper["op"],
  expression: CudaLiteCallExpression,
  context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string {
  const valueExpression = scalarWarpReductionValueExpression(expression);
  if (!valueExpression) return "0";
  const valueType = scalarWarpValueType(valueExpression, callbacks, "warp reduction");
  const helper = registerScalarWarpReduceHelper(context, op, valueType, 32);
  return `${helper.name}(${callbacks.emitExpressionAsValueType(valueExpression, valueType)}, 32u, local_id)`;
}

export function emitScalarWarpShuffleCall(
  op: ScalarWarpShuffleHelper["op"],
  expression: CudaLiteCallExpression,
  context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string {
  const valueExpression = expression.args[1];
  if (!valueExpression) return "0";
  const valueType = scalarWarpValueType(valueExpression, callbacks, "warp shuffle");
  const helper = registerScalarWarpShuffleHelper(context, op, valueType, 32);
  const index = expression.args[2] ? callbacks.emitExpressionAsWgslScalar(expression.args[2], "u32") : "0u";
  const width = expression.args[3] ? callbacks.emitExpressionAsWgslScalar(expression.args[3], "u32") : "32u";
  return `${helper.name}(${callbacks.emitExpressionAsValueType(valueExpression, valueType)}, ${index}, ${width}, local_id)`;
}

export function emitScalarWarpReduceHelper(
  helper: ScalarWarpReduceHelper,
  context: EmitContext,
): string[] {
  if (helper.partitioned) return emitScalarWarpPartitionReduceHelper(helper, context);
  const type = wgslScalar(helper.valueType);
  const workgroupSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
  const start = Math.max(1, Math.floor(Math.min(helper.tileSize, workgroupSize) / 2));
  const scratch = scalarWarpReduceScratchName(helper);
  return [
    `fn ${helper.name}(value_arg: ${type}, width_arg: u32, local_id: vec3<u32>) -> ${type} {`,
    `  let bg_linear_rank: u32 = u32(${emitLocalLinearRank(context)});`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_tile_lane: u32 = bg_linear_rank % bg_width;",
    "  let bg_tile_base: u32 = bg_linear_rank - bg_tile_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    `  var bg_stride: u32 = ${start}u;`,
    "  while (bg_stride > 0u) {",
    `    if (bg_stride < bg_width && bg_tile_lane < bg_stride && (bg_tile_lane + bg_stride) < bg_width && (bg_linear_rank + bg_stride) < ${workgroupSize}u) {`,
    `      ${scratch}[bg_linear_rank] = ${emitScalarWarpReduceStep(helper, `${scratch}[bg_linear_rank]`, `${scratch}[bg_linear_rank + bg_stride]`)};`,
    "    }",
    "    workgroupBarrier();",
    "    bg_stride = bg_stride / 2u;",
    "  }",
    `  let bg_result: ${type} = ${scratch}[bg_tile_base];`,
    "  workgroupBarrier();",
    "  return bg_result;",
    "}",
  ];
}

function emitScalarWarpPartitionReduceHelper(
  helper: ScalarWarpReduceHelper,
  context: EmitContext,
): string[] {
  const type = wgslScalar(helper.valueType);
  const zero = zeroValue(helper.valueType);
  const workgroupSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
  const start = Math.max(1, Math.floor(Math.min(helper.tileSize, workgroupSize) / 2));
  const scratch = scalarWarpReduceScratchName(helper);
  const reduceLoop = (predicate: string, strideName: string): string[] => [
    `  ${scratch}[bg_linear_rank] = select(${zero}, value_arg, ${predicate});`,
    "  workgroupBarrier();",
    `  var ${strideName}: u32 = ${start}u;`,
    `  while (${strideName} > 0u) {`,
    `    if (${strideName} < bg_width && bg_tile_lane < ${strideName} && (bg_tile_lane + ${strideName}) < bg_width && (bg_linear_rank + ${strideName}) < ${workgroupSize}u) {`,
    `      ${scratch}[bg_linear_rank] = ${emitScalarWarpReduceStep(helper, `${scratch}[bg_linear_rank]`, `${scratch}[bg_linear_rank + ${strideName}]`)};`,
    "    }",
    "    workgroupBarrier();",
    `    ${strideName} = ${strideName} / 2u;`,
    "  }",
  ];
  return [
    `fn ${helper.name}(value_arg: ${type}, predicate_arg: bool, width_arg: u32, local_id: vec3<u32>) -> ${type} {`,
    `  let bg_linear_rank: u32 = u32(${emitLocalLinearRank(context)});`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_tile_lane: u32 = bg_linear_rank % bg_width;",
    "  let bg_tile_base: u32 = bg_linear_rank - bg_tile_lane;",
    ...reduceLoop("predicate_arg", "bg_true_stride"),
    `  let bg_true_result: ${type} = ${scratch}[bg_tile_base];`,
    "  workgroupBarrier();",
    ...reduceLoop("!predicate_arg", "bg_false_stride"),
    `  let bg_false_result: ${type} = ${scratch}[bg_tile_base];`,
    "  workgroupBarrier();",
    "  return select(bg_false_result, bg_true_result, predicate_arg);",
    "}",
  ];
}

export function emitScalarWarpReduceWorkgroupStorage(
  helper: ScalarWarpReduceHelper,
  context: EmitContext,
): string[] {
  const type = wgslScalar(helper.valueType);
  const workgroupSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
  return [`var<workgroup> ${scalarWarpReduceScratchName(helper)}: array<${type}, ${workgroupSize}>;`];
}

export function emitScalarWarpShuffleHelper(
  helper: ScalarWarpShuffleHelper,
  context: EmitContext,
): string[] {
  const type = wgslScalar(helper.valueType);
  const workgroupSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
  const scratch = scalarWarpShuffleScratchName(helper);
  return [
    `fn ${helper.name}(value_arg: ${type}, index_arg: u32, width_arg: u32, local_id: vec3<u32>) -> ${type} {`,
    `  let bg_linear_rank: u32 = u32(${emitLocalLinearRank(context)});`,
    `  let bg_tile_lane: u32 = bg_linear_rank % ${helper.tileSize}u;`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_logical_lane: u32 = bg_tile_lane % bg_width;",
    "  let bg_group_base: u32 = bg_linear_rank - bg_logical_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    ...emitScalarWarpShuffleSourceLines(helper, workgroupSize),
    `  let bg_result: ${type} = ${scratch}[bg_source_rank];`,
    "  workgroupBarrier();",
    "  return bg_result;",
    "}",
  ];
}

export function emitScalarWarpShuffleWorkgroupStorage(
  helper: ScalarWarpShuffleHelper,
  context: EmitContext,
): string[] {
  const type = wgslScalar(helper.valueType);
  const workgroupSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
  return [`var<workgroup> ${scalarWarpShuffleScratchName(helper)}: array<${type}, ${workgroupSize}>;`];
}

export function emitVectorCooperativeReduceHelper(
  helper: VectorCooperativeReduceHelper,
  context: EmitContext,
): string[] {
  const type = wgslScalar(helper.valueType);
  const workgroupSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
  const start = Math.max(1, Math.floor(Math.min(helper.tileSize, workgroupSize) / 2));
  const scratch = vectorCooperativeReduceScratchName(helper);
  return [
    `fn ${helper.name}(value_arg: ${type}, local_id: vec3<u32>, workgroup_id: vec3<u32>, num_workgroups: vec3<u32>) -> ${type} {`,
    `  let bg_linear_rank: u32 = u32(${emitLocalLinearRank(context)});`,
    `  let bg_tile_lane: u32 = bg_linear_rank % ${helper.tileSize}u;`,
    "  let bg_tile_base: u32 = bg_linear_rank - bg_tile_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    `  var bg_stride: u32 = ${start}u;`,
    "  while (bg_stride > 0u) {",
    `    if (bg_tile_lane < bg_stride && (bg_linear_rank + bg_stride) < ${workgroupSize}u) {`,
    `      ${scratch}[bg_linear_rank] = ${context.nameFor(helper.opName)}(${scratch}[bg_linear_rank], ${scratch}[bg_linear_rank + bg_stride], local_id, workgroup_id, num_workgroups);`,
    "    }",
    "    workgroupBarrier();",
    "    bg_stride = bg_stride / 2u;",
    "  }",
    `  let bg_result: ${type} = ${scratch}[bg_tile_base];`,
    "  workgroupBarrier();",
    "  return bg_result;",
    "}",
  ];
}

export function emitVectorCooperativeReduceWorkgroupStorage(
  helper: VectorCooperativeReduceHelper,
  context: EmitContext,
): string[] {
  const type = wgslScalar(helper.valueType);
  const workgroupSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
  return [`var<workgroup> ${vectorCooperativeReduceScratchName(helper)}: array<${type}, ${workgroupSize}>;`];
}

export function cooperativeGroupForParam(param: CudaLiteParam, context: EmitContext): CudaLiteCooperativeGroupDecl {
  return {
    kind: "cooperative-group",
    groupKind: param.cooperativeGroupKind ?? "block",
    name: param.name,
    ...(param.tileSize === undefined ? {} : { tileSize: param.tileSize }),
    ...(param.cooperativeGroupKind === "thread" ? { dynamicTileSizeName: context.nameFor(`${param.name}_tile_size`) } : {}),
    span: param.span,
  };
}

export function cooperativeReduceDeviceFunctionName(expression: CudaLiteExpression): string | undefined {
  if (expression.kind !== "call") return undefined;
  const name = expressionName(expression.callee);
  if (!name?.endsWith("::reduce")) return undefined;
  const op = expression.args[2];
  if (op === undefined) return undefined;
  return cooperativeReductionOpName(op);
}

export function emitLocalLinearRank(context: EmitContext): string {
  const [, y, z] = context.ir.workgroupSize;
  if (y === 1 && z === 1) return "i32(local_id.x)";
  if (z === 1) return `i32(local_id.x + local_id.y * ${context.ir.workgroupSize[0]}u)`;
  return `i32(local_id.x + local_id.y * ${context.ir.workgroupSize[0]}u + local_id.z * ${context.ir.workgroupSize[0] * context.ir.workgroupSize[1]}u)`;
}

function emitCooperativeNamespaceCall(
  expression: CudaLiteCallExpression,
  context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string | undefined {
  const name = expressionName(expression.callee);
  if (!name?.endsWith("::sync") && !name?.endsWith("::reduce")) return undefined;
  const groupArg = expression.args[0];
  if (groupArg?.kind !== "identifier") return undefined;
  const group = context.cooperativeGroupFor(groupArg.name);
  if (!group && name.endsWith("::sync")) return "workgroupBarrier()";
  if (!group) return undefined;
  if (name.endsWith("::sync")) return group.groupKind === "grid" ? "0" : "workgroupBarrier()";
  const vectorReduce = emitVectorCooperativeNamespaceReduce(expression, group, context, callbacks);
  if (vectorReduce !== undefined) return vectorReduce;
  const value = expression.args[1] ? callbacks.emitExpression(expression.args[1]) : "0";
  const op = cooperativeReductionOpName(expression.args[2]);
  if (context.subgroupMode === "scalar") return value;
  const partitionPredicate = emitCooperativePartitionPredicate(group, context, callbacks);
  if (partitionPredicate) {
    const valueType = scalarWarpValueType(expression.args[1]!, callbacks, "cooperative group reduce");
    const helper = registerScalarWarpPartitionReduceHelper(context, op?.endsWith("::greater") ? "max" : "sum", valueType, maxCooperativeGroupTileSize(group, context));
    return `${helper.name}(${value}, ${partitionPredicate}, ${emitCooperativeGroupTileSizeValue(group, context)}, local_id)`;
  }
  const helper = registerScalarWarpReduceHelper(context, op?.endsWith("::greater") ? "max" : "sum", scalarWarpValueType(expression.args[1]!, callbacks, "cooperative group reduce"), maxCooperativeGroupTileSize(group, context));
  return `${helper.name}(${value}, ${emitCooperativeGroupTileSizeValue(group, context)}, local_id)`;
}

function scalarWarpReductionValueExpression(expression: CudaLiteCallExpression): CudaLiteExpression | undefined {
  const name = expressionName(expression.callee);
  if (name === "__reduce_add_sync") return expression.args[1];
  return expression.args.length === 2 ? expression.args[1] : expression.args[0];
}

function scalarWarpValueType(
  expression: CudaLiteExpression,
  callbacks: WgslCooperativeCallbacks,
  label: string,
): Exclude<CudaLiteScalarType, "void"> {
  const valueType = callbacks.expressionValueType(expression);
  if (!valueType || valueType === "void" || isCudaVectorType(valueType)) {
    throw featureError("unsupported-subgroup", `${label} expects a scalar value`);
  }
  return valueType as Exclude<CudaLiteScalarType, "void">;
}

function registerScalarWarpReduceHelper(
  context: EmitContext,
  op: ScalarWarpReduceHelper["op"],
  valueType: Exclude<CudaLiteScalarType, "void">,
  tileSize: number,
): ScalarWarpReduceHelper {
  const key = `${op}:${valueType}:${tileSize}`;
  const existing = context.scalarWarpReduceHelpers.get(key);
  if (existing) return existing;
  const helper = {
    key,
    name: `bg_warp_reduce_${op}_${safeWgslIdentifier(valueType)}_${tileSize}`,
    op,
    valueType,
    tileSize,
  };
  context.scalarWarpReduceHelpers.set(key, helper);
  return helper;
}

function registerScalarWarpPartitionReduceHelper(
  context: EmitContext,
  op: ScalarWarpReduceHelper["op"],
  valueType: Exclude<CudaLiteScalarType, "void">,
  tileSize: number,
): ScalarWarpReduceHelper {
  const key = `partition:${op}:${valueType}:${tileSize}`;
  const existing = context.scalarWarpReduceHelpers.get(key);
  if (existing) return existing;
  const helper = {
    key,
    name: `bg_warp_partition_reduce_${op}_${safeWgslIdentifier(valueType)}_${tileSize}`,
    op,
    valueType,
    tileSize,
    partitioned: true,
  };
  context.scalarWarpReduceHelpers.set(key, helper);
  return helper;
}

function emitScalarWarpReduceStep(helper: ScalarWarpReduceHelper, left: string, right: string): string {
  switch (helper.op) {
    case "sum":
      return `(${left} + ${right})`;
    case "max":
      return `max(${left}, ${right})`;
    case "min":
      return `min(${left}, ${right})`;
  }
}

function scalarWarpReduceScratchName(helper: ScalarWarpReduceHelper): string {
  return `${helper.name}_scratch`;
}

function registerScalarWarpShuffleHelper(
  context: EmitContext,
  op: ScalarWarpShuffleHelper["op"],
  valueType: Exclude<CudaLiteScalarType, "void">,
  tileSize: number,
): ScalarWarpShuffleHelper {
  const key = `${op}:${valueType}:${tileSize}`;
  const existing = context.scalarWarpShuffleHelpers.get(key);
  if (existing) return existing;
  const helper = {
    key,
    name: `bg_warp_shuffle_${op}_${safeWgslIdentifier(valueType)}_${tileSize}`,
    op,
    valueType,
    tileSize,
  };
  context.scalarWarpShuffleHelpers.set(key, helper);
  return helper;
}

function emitScalarWarpShuffleSourceLines(helper: ScalarWarpShuffleHelper, workgroupSize: number): string[] {
  switch (helper.op) {
    case "sync":
      return [
        "  let bg_source_lane: u32 = index_arg % bg_width;",
        "  let bg_source_candidate: u32 = bg_group_base + bg_source_lane;",
        `  let bg_source_rank: u32 = select(bg_linear_rank, bg_source_candidate, bg_source_candidate < ${workgroupSize}u);`,
      ];
    case "down":
      return [
        "  let bg_source_lane: u32 = bg_logical_lane + index_arg;",
        "  let bg_source_candidate: u32 = bg_linear_rank + index_arg;",
        `  let bg_source_rank: u32 = select(bg_linear_rank, bg_source_candidate, bg_source_lane < bg_width && bg_source_candidate < ${workgroupSize}u);`,
      ];
    case "up":
      return [
        "  let bg_source_candidate: u32 = bg_linear_rank - min(index_arg, bg_linear_rank);",
        "  let bg_source_rank: u32 = select(bg_linear_rank, bg_source_candidate, bg_logical_lane >= index_arg);",
      ];
    case "xor":
      return [
        "  let bg_source_lane: u32 = bg_logical_lane ^ index_arg;",
        "  let bg_source_candidate: u32 = bg_group_base + bg_source_lane;",
        `  let bg_source_rank: u32 = select(bg_linear_rank, bg_source_candidate, bg_source_lane < bg_width && bg_source_candidate < ${workgroupSize}u);`,
      ];
  }
}

function scalarWarpShuffleScratchName(helper: ScalarWarpShuffleHelper): string {
  return `${helper.name}_scratch`;
}

function maxCooperativeGroupTileSize(group: CudaLiteCooperativeGroupDecl, context: EmitContext): number {
  const workgroupSize = context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2];
  if (group.groupKind === "tile") return Math.min(group.tileSize ?? 32, workgroupSize);
  if (group.groupKind === "thread" && group.dynamicTileSizeName) return workgroupSize;
  if (group.groupKind === "block") return workgroupSize;
  return workgroupSize;
}

function emitCooperativeGroupTileSizeValue(group: CudaLiteCooperativeGroupDecl, context: EmitContext): string {
  const workgroupSize = `${context.ir.workgroupSize[0] * context.ir.workgroupSize[1] * context.ir.workgroupSize[2]}u`;
  if (group.groupKind === "tile") return `${group.tileSize ?? 32}u`;
  if (group.groupKind === "thread" && group.dynamicTileSizeName) return group.dynamicTileSizeName;
  return workgroupSize;
}

function emitCooperativePartitionPredicate(
  group: CudaLiteCooperativeGroupDecl,
  _context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string | undefined {
  if (!group.partitionPredicate) return undefined;
  return callbacks.emitTruthinessExpression(group.partitionPredicate);
}

function emitCooperativePartitionMask(
  group: CudaLiteCooperativeGroupDecl,
  context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string | undefined {
  const predicate = emitCooperativePartitionPredicate(group, context, callbacks);
  if (!predicate) return undefined;
  if (context.subgroupMode === "scalar") return `select(0u, 1u, ${predicate})`;
  const ballot = `subgroupBallot(${predicate}).x`;
  const tileMask = emitCooperativeTileLaneMask(group, context);
  return tileMask ? `(${ballot} & ${tileMask})` : ballot;
}

function emitCooperativeTileLaneMask(group: CudaLiteCooperativeGroupDecl, context: EmitContext): string | undefined {
  if (group.groupKind !== "tile") return undefined;
  const tileSize = group.tileSize ?? 32;
  if (tileSize >= 32) return "0xffffffffu";
  const lane = `u32(${emitLocalLinearRank(context)} % 32)`;
  const tileBase = `((${lane} / ${tileSize}u) * ${tileSize}u)`;
  return `(((1u << ${tileSize}u) - 1u) << ${tileBase})`;
}

function emitCooperativePartitionSize(
  group: CudaLiteCooperativeGroupDecl,
  context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string | undefined {
  const mask = emitCooperativePartitionMask(group, context, callbacks);
  return mask ? `i32(countOneBits(${mask}))` : undefined;
}

function emitCooperativePartitionRank(
  group: CudaLiteCooperativeGroupDecl,
  context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string | undefined {
  const mask = emitCooperativePartitionMask(group, context, callbacks);
  if (!mask) return undefined;
  const tileSize = group.tileSize ?? 32;
  const lane = `u32(${emitLocalLinearRank(context)} % ${tileSize})`;
  return `i32(countOneBits(${mask} & ((1u << ${lane}) - 1u)))`;
}

function emitVectorCooperativeNamespaceReduce(
  expression: CudaLiteCallExpression,
  group: CudaLiteCooperativeGroupDecl,
  context: EmitContext,
  callbacks: WgslCooperativeCallbacks,
): string | undefined {
  const valueExpression = expression.args[1];
  const opExpression = expression.args[2];
  if (!valueExpression) return undefined;
  const valueType = callbacks.expressionValueType(valueExpression);
  if (!isCudaVectorType(valueType)) return undefined;
  if (context.subgroupMode === "scalar") return callbacks.emitExpression(valueExpression);
  const opName = opExpression ? expressionName(opExpression) : undefined;
  const op = opName ? context.deviceFunctionFor(opName) : undefined;
  if (!op || op.returnType !== valueType) {
    throw featureError("unsupported-cooperative-groups", "vector cg::reduce expects a device helper returning the same CUDA vector type");
  }
  const tileSize = group.groupKind === "tile" ? group.tileSize ?? 32 : 32;
  const helper = registerVectorCooperativeReduceHelper(context, op.name, valueType, tileSize);
  return `${helper.name}(${callbacks.emitExpression(valueExpression)}, local_id, workgroup_id, num_workgroups)`;
}

function registerVectorCooperativeReduceHelper(
  context: EmitContext,
  opName: string,
  valueType: CudaLiteVectorType,
  tileSize: number,
): VectorCooperativeReduceHelper {
  const key = `${opName}:${valueType}:${tileSize}`;
  const existing = context.vectorCooperativeReduceHelpers.get(key);
  if (existing) return existing;
  const helper = {
    key,
    name: `bg_cg_reduce_${safeWgslIdentifier(opName)}_${valueType}_${tileSize}`,
    opName,
    valueType,
    tileSize,
  };
  context.vectorCooperativeReduceHelpers.set(key, helper);
  return helper;
}

function vectorCooperativeReduceScratchName(helper: VectorCooperativeReduceHelper): string {
  return `${helper.name}_scratch`;
}

function cooperativeReductionOpName(expression: CudaLiteExpression | undefined): string | undefined {
  if (expression?.kind === "call") return expressionName(expression.callee);
  return expression === undefined ? undefined : expressionName(expression);
}

function featureError(code: string, message: string): CudaLiteCompilerError {
  const span: SourceSpan = { start: 0, end: 0, line: 1, column: 1 };
  return new CudaLiteCompilerError(message, [{ code, severity: "error", message, span }]);
}

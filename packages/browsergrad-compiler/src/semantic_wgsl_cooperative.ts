import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import { semanticExpressionChildren, semanticOperationExpressions } from "./semantic_ir_walk.js";
import { semanticExpressionValueType } from "./semantic_vector_intrinsics.js";
import { wgslValueScalar } from "./semantic_wgsl_types.js";

export interface SemanticCooperativeReduceHelper {
  readonly name: string;
  readonly scratchName: string;
  readonly valueType: Exclude<ReturnType<typeof semanticExpressionValueType>, undefined | "void">;
  readonly tileSize: number;
}

export function semanticWgslCooperativeGroupCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): boolean {
  if (expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol" || expression.args.length !== 0) return false;
  if (expression.callee.property !== "thread_rank" && expression.callee.property !== "size") return false;
  return semanticCooperativeGroupInfo(ir, expression.callee.object.name) !== undefined;
}

export function semanticWgslCooperativeReduceCallSupported(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  valueSupported: (value: SemanticExpression) => boolean,
): boolean {
  const helper = semanticCooperativeReduceHelperFor(ir, expression);
  const value = expression.args[1];
  return helper !== undefined && value !== undefined && valueSupported(value);
}

export function emitSemanticCooperativeGroupCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
): string | undefined {
  if (expression.callee.kind !== "member" || expression.callee.object.kind !== "symbol") return undefined;
  const group = semanticCooperativeGroupInfo(ir, expression.callee.object.name);
  if (!group) return undefined;
  if (expression.callee.property === "thread_rank") {
    const rank = group.kind === "grid"
      ? semanticCooperativeGlobalLinearRank(ir)
      : group.kind === "tile"
      ? `i32(${semanticCooperativeLocalLinearRank(ir)} % ${group.tileSize ?? 32}u)`
      : `i32(${semanticCooperativeLocalLinearRank(ir)})`;
    return expression.valueType === "uint" ? `u32(${rank})` : rank;
  }
  if (expression.callee.property === "size") {
    const size = group.kind === "grid"
      ? semanticCooperativeGridThreadCount(ir)
      : group.kind === "tile"
      ? String(group.tileSize ?? 32)
      : String(semanticCooperativeWorkgroupSize(ir));
    return expression.valueType === "uint" ? `u32(${size})` : size;
  }
  return undefined;
}

export function emitSemanticCooperativeReduceCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  ir: SemanticKernelIrModule,
  value: string,
): string | undefined {
  const helper = semanticCooperativeReduceHelperFor(ir, expression);
  return helper === undefined ? undefined : `${helper.name}(${value}, local_id)`;
}

export function semanticCooperativeReduceHelperFor(
  ir: SemanticKernelIrModule,
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): SemanticCooperativeReduceHelper | undefined {
  if (expression.callee.kind !== "symbol" || !isCooperativeReduceName(expression.callee.name)) return undefined;
  const [groupArg, valueArg, operationArg] = expression.args;
  if (!groupArg || groupArg.kind !== "symbol" || !valueArg || !isPlusOperation(operationArg)) return undefined;
  const group = semanticCooperativeGroupInfo(ir, groupArg.name);
  if (!group || (group.kind !== "tile" && group.kind !== "thread")) return undefined;
  const valueType = semanticExpressionValueType(valueArg);
  if (valueType !== "float" && valueType !== "double" && valueType !== "int" && valueType !== "uint") return undefined;
  const tileSize = group.tileSize ?? 32;
  const typeName = wgslValueScalar(valueType).replace(/[^A-Za-z0-9_]/gu, "_");
  return {
    name: `bg_semantic_cg_reduce_${typeName}_${tileSize}`,
    scratchName: `bg_semantic_cg_reduce_${typeName}_${tileSize}_scratch`,
    valueType,
    tileSize,
  };
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

export function emitSemanticCooperativeReduceHelper(
  helper: SemanticCooperativeReduceHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = ir.workgroupSize[0] * ir.workgroupSize[1] * ir.workgroupSize[2];
  const start = Math.max(1, Math.floor(Math.min(helper.tileSize, workgroupSize) / 2));
  return [
    `fn ${helper.name}(value_arg: ${type}, local_id: vec3<u32>) -> ${type} {`,
    `  let rank: u32 = ${semanticCooperativeLocalLinearRank(ir)};`,
    `  let width: u32 = min(${helper.tileSize}u, ${workgroupSize}u);`,
    "  let lane: u32 = rank % width;",
    "  let base: u32 = rank - lane;",
    `  ${helper.scratchName}[rank] = value_arg;`,
    "  workgroupBarrier();",
    `  var stride: u32 = ${start}u;`,
    "  while (stride > 0u) {",
    `    if (lane < stride && (lane + stride) < width && (rank + stride) < ${workgroupSize}u) {`,
    `      ${helper.scratchName}[rank] = ${helper.scratchName}[rank] + ${helper.scratchName}[rank + stride];`,
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

function semanticCooperativeGroupInfo(
  ir: SemanticKernelIrModule,
  name: string,
): { readonly kind: "thread" | "block" | "grid" | "tile" | "coalesced" | "binary"; readonly tileSize?: number } | undefined {
  const declaration = [...ir.operations, ...ir.functions.flatMap((fn) => fn.body)].flatMap(semanticOperationsDeep).find((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "cooperative-group-declare" }> =>
    operation.kind === "cooperative-group-declare" && operation.declaration.name === name,
  );
  if (declaration) return { kind: declaration.declaration.groupKind, ...(declaration.declaration.tileSize === undefined ? {} : { tileSize: declaration.declaration.tileSize }) };
  const param = ir.functions.flatMap((fn) => fn.params).find((item) => item.name === name && item.cooperativeGroupKind !== undefined);
  return param?.cooperativeGroupKind === undefined ? undefined : { kind: param.cooperativeGroupKind, ...(param.tileSize === undefined ? {} : { tileSize: param.tileSize }) };
}

function semanticOperationsDeep(operation: SemanticKernelIrOperation): readonly SemanticKernelIrOperation[] {
  if (operation.kind === "branch") return [operation, ...operation.consequent.flatMap(semanticOperationsDeep), ...operation.alternate.flatMap(semanticOperationsDeep)];
  if (operation.kind === "loop") return [operation, ...operation.body.flatMap(semanticOperationsDeep)];
  if (operation.kind === "block") return [operation, ...operation.body.flatMap(semanticOperationsDeep)];
  return [operation];
}

function isCooperativeReduceName(name: string): boolean {
  return name === "cg::reduce" || name === "cooperative_groups::reduce";
}

function isPlusOperation(expression: SemanticExpression | undefined): boolean {
  return expression?.kind === "call" && expression.callee.kind === "symbol" && expression.callee.name.endsWith("::plus");
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

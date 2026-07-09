import type { CudaLiteScalarType } from "./types.js";
import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import {
  isSemanticKernelIrOperation,
  semanticExpressionChildren,
} from "./semantic_ir_walk.js";
import {
  cudaBitwiseReduceOpForCall,
  cudaShuffleOpForCall,
  isCudaLegacyShuffleCallName as legacyShuffleCall,
  isCudaLegacyVoteCallName as legacyVoteCall,
  type CudaBitwiseReduceOp,
  type CudaShuffleOp,
} from "./cuda_subgroup_calls.js";
import { safeWgslIdentifier } from "./wgsl_names.js";
import { wgslValueScalar } from "./semantic_wgsl_types.js";
import { semanticExpressionValueType } from "./semantic_vector_intrinsics.js";

export type SemanticShuffleOp = CudaShuffleOp;

export interface SemanticWarpShuffleHelper {
  readonly key: string;
  readonly name: string;
  readonly op: SemanticShuffleOp;
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly tileSize: number;
}

export interface SemanticMatchAnyHelper {
  readonly key: string;
  readonly name: string;
  readonly valueType: Exclude<CudaLiteScalarType, "void">;
  readonly tileSize: number;
}

export type SemanticBitwiseReduceOp = CudaBitwiseReduceOp;

export interface SemanticBitwiseReduceHelper {
  readonly key: string;
  readonly name: string;
  readonly op: SemanticBitwiseReduceOp;
  readonly valueType: "int" | "uint";
  readonly tileSize: number;
}

export function semanticWarpShuffleHelpers(ir: SemanticKernelIrModule): readonly SemanticWarpShuffleHelper[] {
  const helpers = new Map<string, SemanticWarpShuffleHelper>();
  collectSemanticWarpShuffleHelpers(ir.operations, helpers);
  for (const fn of ir.functions) collectSemanticWarpShuffleHelpers(fn.body, helpers);
  return [...helpers.values()];
}

function collectSemanticWarpShuffleHelpers(
  operations: readonly SemanticKernelIrOperation[],
  helpers: Map<string, SemanticWarpShuffleHelper>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.init) collectSemanticWarpShuffleExpressionHelpers(operation.init, helpers);
    if (operation.kind === "store") {
      collectSemanticWarpShuffleExpressionHelpers(operation.value, helpers);
      operation.target.indices.forEach((index) => collectSemanticWarpShuffleExpressionHelpers(index, helpers));
    }
    if (operation.kind === "atomic") {
      operation.args.forEach((arg) => collectSemanticWarpShuffleExpressionHelpers(arg, helpers));
      operation.target?.indices.forEach((index) => collectSemanticWarpShuffleExpressionHelpers(index, helpers));
    }
    if (operation.kind === "call") operation.args.forEach((arg) => collectSemanticWarpShuffleExpressionHelpers(arg, helpers));
    if (operation.kind === "expression") collectSemanticWarpShuffleExpressionHelpers(operation.expression, helpers);
    if (operation.kind === "branch") {
      collectSemanticWarpShuffleExpressionHelpers(operation.condition, helpers);
      collectSemanticWarpShuffleHelpers(operation.consequent, helpers);
      collectSemanticWarpShuffleHelpers(operation.alternate, helpers);
    }
    if (operation.kind === "loop") {
      if (operation.init !== undefined) {
        if (isSemanticKernelIrOperation(operation.init)) collectSemanticWarpShuffleHelpers([operation.init], helpers);
        else collectSemanticWarpShuffleExpressionHelpers(operation.init, helpers);
      }
      if (operation.condition) collectSemanticWarpShuffleExpressionHelpers(operation.condition, helpers);
      if (operation.update) collectSemanticWarpShuffleExpressionHelpers(operation.update, helpers);
      collectSemanticWarpShuffleHelpers(operation.body, helpers);
    }
    if (operation.kind === "return" && operation.value) collectSemanticWarpShuffleExpressionHelpers(operation.value, helpers);
    if (operation.kind === "block") collectSemanticWarpShuffleHelpers(operation.body, helpers);
  }
}

function collectSemanticWarpShuffleExpressionHelpers(
  expression: SemanticExpression,
  helpers: Map<string, SemanticWarpShuffleHelper>,
): void {
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    const op = semanticShuffleOpForCall(expression.callee.name);
    const value = expression.args[legacyShuffleCall(expression.callee.name) ? 0 : 1];
    const valueType = value ? semanticExpressionValueType(value) : undefined;
    if (op && valueType && valueType !== "void") {
      const helper = semanticWarpShuffleHelper(op, valueType, 32);
      helpers.set(helper.key, helper);
    }
  }
  for (const child of semanticExpressionChildren(expression)) {
    collectSemanticWarpShuffleExpressionHelpers(child, helpers);
  }
}

export function semanticMatchAnyHelpers(ir: SemanticKernelIrModule): readonly SemanticMatchAnyHelper[] {
  const helpers = new Map<string, SemanticMatchAnyHelper>();
  collectSemanticMatchAnyHelpers(ir.operations, helpers);
  for (const fn of ir.functions) collectSemanticMatchAnyHelpers(fn.body, helpers);
  return [...helpers.values()];
}

function collectSemanticMatchAnyHelpers(
  operations: readonly SemanticKernelIrOperation[],
  helpers: Map<string, SemanticMatchAnyHelper>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.init) collectSemanticMatchAnyExpressionHelpers(operation.init, helpers);
    if (operation.kind === "store") {
      collectSemanticMatchAnyExpressionHelpers(operation.value, helpers);
      operation.target.indices.forEach((index) => collectSemanticMatchAnyExpressionHelpers(index, helpers));
    }
    if (operation.kind === "atomic") {
      operation.args.forEach((arg) => collectSemanticMatchAnyExpressionHelpers(arg, helpers));
      operation.target?.indices.forEach((index) => collectSemanticMatchAnyExpressionHelpers(index, helpers));
    }
    if (operation.kind === "call") operation.args.forEach((arg) => collectSemanticMatchAnyExpressionHelpers(arg, helpers));
    if (operation.kind === "expression") collectSemanticMatchAnyExpressionHelpers(operation.expression, helpers);
    if (operation.kind === "branch") {
      collectSemanticMatchAnyExpressionHelpers(operation.condition, helpers);
      collectSemanticMatchAnyHelpers(operation.consequent, helpers);
      collectSemanticMatchAnyHelpers(operation.alternate, helpers);
    }
    if (operation.kind === "loop") {
      if (operation.init !== undefined) {
        if (isSemanticKernelIrOperation(operation.init)) collectSemanticMatchAnyHelpers([operation.init], helpers);
        else collectSemanticMatchAnyExpressionHelpers(operation.init, helpers);
      }
      if (operation.condition) collectSemanticMatchAnyExpressionHelpers(operation.condition, helpers);
      if (operation.update) collectSemanticMatchAnyExpressionHelpers(operation.update, helpers);
      collectSemanticMatchAnyHelpers(operation.body, helpers);
    }
    if (operation.kind === "return" && operation.value) collectSemanticMatchAnyExpressionHelpers(operation.value, helpers);
    if (operation.kind === "block") collectSemanticMatchAnyHelpers(operation.body, helpers);
  }
}

function collectSemanticMatchAnyExpressionHelpers(
  expression: SemanticExpression,
  helpers: Map<string, SemanticMatchAnyHelper>,
): void {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && expression.callee.name === "__match_any_sync") {
    const value = expression.args[1];
    const valueType = value ? semanticExpressionValueType(value) : undefined;
    if (valueType && valueType !== "void") {
      const helper = semanticMatchAnyHelper(valueType, 32);
      helpers.set(helper.key, helper);
    }
  }
  for (const child of semanticExpressionChildren(expression)) {
    collectSemanticMatchAnyExpressionHelpers(child, helpers);
  }
}

export function semanticMatchAnyHelper(
  valueType: Exclude<CudaLiteScalarType, "void">,
  tileSize: number,
): SemanticMatchAnyHelper {
  const key = `${valueType}:${tileSize}`;
  return {
    key,
    name: `bg_semantic_match_any_${safeWgslIdentifier(valueType)}_${tileSize}`,
    valueType,
    tileSize,
  };
}

export function semanticMatchAnyScratchName(helper: SemanticMatchAnyHelper): string {
  return `${helper.name}_scratch`;
}

export function emitSemanticMatchAnyHelper(
  helper: SemanticMatchAnyHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = semanticWorkgroupSize(ir);
  const scratch = semanticMatchAnyScratchName(helper);
  return [
    `fn ${helper.name}(value_arg: ${type}, width_arg: u32, local_id: vec3<u32>) -> u32 {`,
    `  let bg_linear_rank: u32 = ${semanticLocalLinearRank(ir)};`,
    `  let bg_tile_lane: u32 = bg_linear_rank % ${helper.tileSize}u;`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_logical_lane: u32 = bg_tile_lane % bg_width;",
    "  let bg_group_base: u32 = bg_linear_rank - bg_logical_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    "  var bg_mask: u32 = 0u;",
    "  var bg_lane: u32 = 0u;",
    "  while (bg_lane < bg_width) {",
    "    let bg_source_rank: u32 = bg_group_base + bg_lane;",
    `    if (bg_source_rank < ${workgroupSize}u && ${scratch}[bg_source_rank] == value_arg) {`,
    "      bg_mask = bg_mask | (1u << bg_lane);",
    "    }",
    "    bg_lane = bg_lane + 1u;",
    "  }",
    "  workgroupBarrier();",
    "  return bg_mask;",
    "}",
  ];
}

export function semanticBitwiseReduceHelpers(ir: SemanticKernelIrModule): readonly SemanticBitwiseReduceHelper[] {
  const helpers = new Map<string, SemanticBitwiseReduceHelper>();
  collectSemanticBitwiseReduceHelpers(ir.operations, helpers);
  for (const fn of ir.functions) collectSemanticBitwiseReduceHelpers(fn.body, helpers);
  return [...helpers.values()];
}

function collectSemanticBitwiseReduceHelpers(
  operations: readonly SemanticKernelIrOperation[],
  helpers: Map<string, SemanticBitwiseReduceHelper>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.init) collectSemanticBitwiseReduceExpressionHelpers(operation.init, helpers);
    if (operation.kind === "store") {
      collectSemanticBitwiseReduceExpressionHelpers(operation.value, helpers);
      operation.target.indices.forEach((index) => collectSemanticBitwiseReduceExpressionHelpers(index, helpers));
    }
    if (operation.kind === "atomic") {
      operation.args.forEach((arg) => collectSemanticBitwiseReduceExpressionHelpers(arg, helpers));
      operation.target?.indices.forEach((index) => collectSemanticBitwiseReduceExpressionHelpers(index, helpers));
    }
    if (operation.kind === "call") operation.args.forEach((arg) => collectSemanticBitwiseReduceExpressionHelpers(arg, helpers));
    if (operation.kind === "expression") collectSemanticBitwiseReduceExpressionHelpers(operation.expression, helpers);
    if (operation.kind === "branch") {
      collectSemanticBitwiseReduceExpressionHelpers(operation.condition, helpers);
      collectSemanticBitwiseReduceHelpers(operation.consequent, helpers);
      collectSemanticBitwiseReduceHelpers(operation.alternate, helpers);
    }
    if (operation.kind === "loop") {
      if (operation.init !== undefined) {
        if (isSemanticKernelIrOperation(operation.init)) collectSemanticBitwiseReduceHelpers([operation.init], helpers);
        else collectSemanticBitwiseReduceExpressionHelpers(operation.init, helpers);
      }
      if (operation.condition) collectSemanticBitwiseReduceExpressionHelpers(operation.condition, helpers);
      if (operation.update) collectSemanticBitwiseReduceExpressionHelpers(operation.update, helpers);
      collectSemanticBitwiseReduceHelpers(operation.body, helpers);
    }
    if (operation.kind === "return" && operation.value) collectSemanticBitwiseReduceExpressionHelpers(operation.value, helpers);
    if (operation.kind === "block") collectSemanticBitwiseReduceHelpers(operation.body, helpers);
  }
}

function collectSemanticBitwiseReduceExpressionHelpers(
  expression: SemanticExpression,
  helpers: Map<string, SemanticBitwiseReduceHelper>,
): void {
  if (expression.kind === "call" && expression.callee.kind === "symbol") {
    const op = semanticBitwiseReduceOpForCall(expression.callee.name);
    const value = expression.args[1];
    const valueType = value ? semanticExpressionValueType(value) : undefined;
    if (op && (valueType === "int" || valueType === "uint")) {
      const helper = semanticBitwiseReduceHelper(op, valueType, 32);
      helpers.set(helper.key, helper);
    }
  }
  for (const child of semanticExpressionChildren(expression)) {
    collectSemanticBitwiseReduceExpressionHelpers(child, helpers);
  }
}

export function semanticBitwiseReduceOpForCall(name: string): SemanticBitwiseReduceOp | undefined {
  return cudaBitwiseReduceOpForCall(name);
}

export function semanticBitwiseReduceHelper(
  op: SemanticBitwiseReduceOp,
  valueType: "int" | "uint",
  tileSize: number,
): SemanticBitwiseReduceHelper {
  const key = `${op}:${valueType}:${tileSize}`;
  return {
    key,
    name: `bg_semantic_reduce_${op}_${safeWgslIdentifier(valueType)}_${tileSize}`,
    op,
    valueType,
    tileSize,
  };
}

export function semanticBitwiseReduceScratchName(helper: SemanticBitwiseReduceHelper): string {
  return `${helper.name}_scratch`;
}

export function emitSemanticBitwiseReduceHelper(
  helper: SemanticBitwiseReduceHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = semanticWorkgroupSize(ir);
  const start = Math.max(1, Math.floor(Math.min(helper.tileSize, workgroupSize) / 2));
  const scratch = semanticBitwiseReduceScratchName(helper);
  const operator = helper.op === "and" ? "&" : helper.op === "or" ? "|" : "^";
  return [
    `fn ${helper.name}(value_arg: ${type}, width_arg: u32, local_id: vec3<u32>) -> ${type} {`,
    `  let bg_linear_rank: u32 = ${semanticLocalLinearRank(ir)};`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_tile_lane: u32 = bg_linear_rank % bg_width;",
    "  let bg_tile_base: u32 = bg_linear_rank - bg_tile_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    `  var bg_stride: u32 = ${start}u;`,
    "  while (bg_stride > 0u) {",
    `    if (bg_stride < bg_width && bg_tile_lane < bg_stride && (bg_tile_lane + bg_stride) < bg_width && (bg_linear_rank + bg_stride) < ${workgroupSize}u) {`,
    `      ${scratch}[bg_linear_rank] = ${scratch}[bg_linear_rank] ${operator} ${scratch}[bg_linear_rank + bg_stride];`,
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

export function semanticShuffleOpForCall(name: string): SemanticShuffleOp | undefined {
  return cudaShuffleOpForCall(name);
}

export function semanticWarpShuffleHelper(
  op: SemanticShuffleOp,
  valueType: Exclude<CudaLiteScalarType, "void">,
  tileSize: number,
): SemanticWarpShuffleHelper {
  const key = `${op}:${valueType}:${tileSize}`;
  return {
    key,
    name: `bg_semantic_warp_shuffle_${op}_${safeWgslIdentifier(valueType)}_${tileSize}`,
    op,
    valueType,
    tileSize,
  };
}

export function semanticWarpShuffleScratchName(helper: SemanticWarpShuffleHelper): string {
  return `${helper.name}_scratch`;
}

export function emitSemanticWarpShuffleHelper(
  helper: SemanticWarpShuffleHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const type = wgslValueScalar(helper.valueType);
  const workgroupSize = semanticWorkgroupSize(ir);
  const scratch = semanticWarpShuffleScratchName(helper);
  return [
    `fn ${helper.name}(value_arg: ${type}, index_arg: u32, width_arg: u32, local_id: vec3<u32>) -> ${type} {`,
    `  let bg_linear_rank: u32 = ${semanticLocalLinearRank(ir)};`,
    `  let bg_tile_lane: u32 = bg_linear_rank % ${helper.tileSize}u;`,
    `  let bg_width: u32 = clamp(width_arg, 1u, ${helper.tileSize}u);`,
    "  let bg_logical_lane: u32 = bg_tile_lane % bg_width;",
    "  let bg_group_base: u32 = bg_linear_rank - bg_logical_lane;",
    `  ${scratch}[bg_linear_rank] = value_arg;`,
    "  workgroupBarrier();",
    ...emitSemanticWarpShuffleSourceLines(helper, workgroupSize),
    `  let bg_result: ${type} = ${scratch}[bg_source_rank];`,
    "  workgroupBarrier();",
    "  return bg_result;",
    "}",
  ];
}

function emitSemanticWarpShuffleSourceLines(helper: SemanticWarpShuffleHelper, workgroupSize: number): readonly string[] {
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

export function semanticWorkgroupSize(ir: SemanticKernelIrModule): number {
  return ir.workgroupSize[0] * ir.workgroupSize[1] * ir.workgroupSize[2];
}

function semanticLocalLinearRank(ir: SemanticKernelIrModule): string {
  return `(local_id.x + local_id.y * ${ir.workgroupSize[0]}u + local_id.z * ${ir.workgroupSize[0] * ir.workgroupSize[1]}u)`;
}

export { legacyShuffleCall, legacyVoteCall };

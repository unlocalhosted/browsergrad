import { walkSemanticOperations } from "./semantic_ir.js";
import type { SemanticExpression, SemanticKernelIrModule } from "./semantic_ir_types.js";

import {
  cudaSyncthreadsPredicateReduction,
  type CudaSyncthreadsPredicateReduction,
} from "./cuda_sync_calls.js";

export interface SemanticSyncthreadsPredicateHelper {
  readonly name: string;
  readonly scratchName: string;
  readonly reduction: CudaSyncthreadsPredicateReduction;
}

export function semanticSyncthreadsPredicateHelperFor(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
): SemanticSyncthreadsPredicateHelper | undefined {
  if (expression.callee.kind !== "symbol" || expression.callee.addressSpace === "function" || expression.args.length !== 1) return undefined;
  const reduction = cudaSyncthreadsPredicateReduction(expression.callee.name);
  if (reduction === undefined) return undefined;
  const name = `bg_semantic_syncthreads_${reduction}`;
  return { name, scratchName: `${name}_scratch`, reduction };
}

export function semanticSyncthreadsPredicateHelpers(
  ir: SemanticKernelIrModule,
): readonly SemanticSyncthreadsPredicateHelper[] {
  const helpers = new Map<string, SemanticSyncthreadsPredicateHelper>();
  const visit = (expression: SemanticExpression): void => {
    if (expression.kind !== "call") return;
    const helper = semanticSyncthreadsPredicateHelperFor(expression);
    if (helper) helpers.set(helper.name, helper);
  };
  walkSemanticOperations(ir.operations, visit);
  for (const fn of ir.functions) walkSemanticOperations(fn.body, visit);
  return [...helpers.values()];
}

export function emitSemanticSyncthreadsPredicateHelper(
  helper: SemanticSyncthreadsPredicateHelper,
  ir: SemanticKernelIrModule,
): readonly string[] {
  const [x, y, z] = ir.workgroupSize;
  const rank = y === 1 && z === 1
    ? "local_id.x"
    : z === 1
      ? `local_id.x + local_id.y * ${x}u`
      : `local_id.x + local_id.y * ${x}u + local_id.z * ${x * y}u`;
  const workgroupSize = x * y * z;
  const initial = helper.reduction === "and" ? "1u" : "0u";
  const combine = helper.reduction === "count"
    ? `result = result + ${helper.scratchName}[index];`
    : helper.reduction === "and"
      ? `result = result & ${helper.scratchName}[index];`
      : `result = result | ${helper.scratchName}[index];`;
  return [
    `fn ${helper.name}(predicate_arg: bool, local_id: vec3<u32>) -> i32 {`,
    `  let rank: u32 = ${rank};`,
    `  ${helper.scratchName}[rank] = select(0u, 1u, predicate_arg);`,
    "  workgroupBarrier();",
    "  if (rank == 0u) {",
    `    var result: u32 = ${initial};`,
    `    for (var index: u32 = 0u; index < ${workgroupSize}u; index = index + 1u) {`,
    `      ${combine}`,
    "    }",
    `    ${helper.scratchName}[0] = result;`,
    "  }",
    "  workgroupBarrier();",
    `  let result: i32 = i32(${helper.scratchName}[0]);`,
    "  workgroupBarrier();",
    "  return result;",
    "}",
  ];
}

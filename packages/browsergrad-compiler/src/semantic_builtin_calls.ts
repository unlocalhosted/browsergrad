import { CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES } from "./cuda_pointer_calls.js";
import {
  CUDA_SUBGROUP_CALL_NAMES,
  cudaBitwiseReduceOpForCall,
  cudaShuffleOpForCall,
  isCudaLegacyShuffleCallName,
  isCudaLegacyVoteCallName,
  isCudaWarpSumCallName,
} from "./cuda_subgroup_calls.js";
import type {
  SemanticAddressSpace,
  SemanticExpression,
} from "./semantic_ir.js";

export const SEMANTIC_LOCAL_ARRAY_FILL_CALLS = new Set(["fill_1D_regs", "fill_2D_regs", "fill_3D_regs"]);

export const SEMANTIC_NOOP_CALLS = new Set([
  "__nanosleep",
  "__prof_trigger",
  "__trap",
  "cudaGraphSetConditional",
]);

export const SEMANTIC_ADDRESS_PREDICATE_CALLS: ReadonlySet<string> = new Set(CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES);

export const SEMANTIC_SUBGROUP_CALLS: ReadonlySet<string> = new Set(CUDA_SUBGROUP_CALL_NAMES);

type SemanticBuiltinExpressionSupported = (expression: SemanticExpression) => boolean;

export function semanticAssertCallSupported(args: readonly SemanticExpression[], expressionSupported: SemanticBuiltinExpressionSupported): boolean {
  return args.length === 1 && args[0] !== undefined && expressionSupported(args[0]);
}

export function semanticNoopCallSupported(callee: string, args: readonly SemanticExpression[], expressionSupported: SemanticBuiltinExpressionSupported): boolean {
  return SEMANTIC_NOOP_CALLS.has(callee) && args.every(expressionSupported);
}

export function semanticPrintfCallSupported(args: readonly SemanticExpression[], expressionSupported: SemanticBuiltinExpressionSupported): boolean {
  const [format, ...rest] = args;
  return format?.kind === "literal" &&
    format.literalKind === "string" &&
    rest.every(expressionSupported);
}

type SemanticSubgroupScalarArgRule = "all" | readonly number[];

interface SemanticSubgroupCallShape {
  readonly minArgs: number;
  readonly maxArgs: number;
  readonly scalarArgRule: SemanticSubgroupScalarArgRule;
}

export function semanticSubgroupCallShape(name: string | undefined): SemanticSubgroupCallShape | undefined {
  if (name === undefined || !SEMANTIC_SUBGROUP_CALLS.has(name)) return undefined;
  if (name === "__activemask") return { minArgs: 0, maxArgs: 0, scalarArgRule: [] };
  if (isCudaLegacyVoteCallName(name)) return { minArgs: 1, maxArgs: 1, scalarArgRule: [0] };
  if (isCudaLegacyShuffleCallName(name)) return { minArgs: 2, maxArgs: 3, scalarArgRule: "all" };
  if (isCudaWarpSumCallName(name)) return { minArgs: 1, maxArgs: 2, scalarArgRule: "all" };
  if (cudaBitwiseReduceOpForCall(name)) return { minArgs: 2, maxArgs: 2, scalarArgRule: "all" };
  if (cudaShuffleOpForCall(name)) return { minArgs: 3, maxArgs: 4, scalarArgRule: "all" };
  return { minArgs: 2, maxArgs: 2, scalarArgRule: [1] };
}

export function semanticSubgroupScalarArguments(
  name: string | undefined,
  args: readonly SemanticExpression[],
): readonly SemanticExpression[] | undefined {
  const shape = semanticSubgroupCallShape(name);
  if (!shape || args.length < shape.minArgs || args.length > shape.maxArgs) return undefined;
  if (shape.scalarArgRule === "all") return args;
  return shape.scalarArgRule.map((index) => args[index]).filter((arg): arg is SemanticExpression => arg !== undefined);
}

export function semanticAddressPredicateAddressSpace(expression: SemanticExpression | undefined): SemanticAddressSpace | undefined {
  if (!expression) return undefined;
  if (expression.kind === "symbol") return expression.addressSpace;
  if (expression.kind === "index") return expression.addressSpace;
  if (expression.kind === "member") return semanticAddressPredicateAddressSpace(expression.object);
  if (expression.kind === "cast" && expression.pointer) return semanticAddressPredicateAddressSpace(expression.expression);
  if (expression.kind === "unary" && expression.operator === "&") return semanticAddressPredicateAddressSpace(expression.argument);
  if (expression.kind === "conditional") {
    const consequent = semanticAddressPredicateAddressSpace(expression.consequent);
    const alternate = semanticAddressPredicateAddressSpace(expression.alternate);
    return consequent === alternate ? consequent : undefined;
  }
  return undefined;
}

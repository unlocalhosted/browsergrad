import { CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES } from "./cuda_pointer_calls.js";
import { CUDA_SUBGROUP_CALL_NAMES } from "./cuda_subgroup_calls.js";
import type {
  SemanticAddressSpace,
  SemanticExpression,
} from "./semantic_ir.js";

export const SEMANTIC_LOCAL_ARRAY_FILL_CALLS = new Set(["fill_1D_regs", "fill_2D_regs", "fill_3D_regs"]);

export const SEMANTIC_NOOP_CALLS = new Set([
  "__nanosleep",
  "__prof_trigger",
  "__trap",
]);

export const SEMANTIC_ADDRESS_PREDICATE_CALLS: ReadonlySet<string> = new Set(CUDA_ADDRESS_SPACE_PREDICATE_CALL_NAMES);

export const SEMANTIC_SUBGROUP_CALLS: ReadonlySet<string> = new Set(CUDA_SUBGROUP_CALL_NAMES);

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

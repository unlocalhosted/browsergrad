import { LAYOUT_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import type { DecodeLimits } from "../schema/limits.js";
import {
  createDimEvaluationBudget,
  euclideanModulo,
  evaluateDimExprWithBudget,
  type DimEvaluation,
  type DimEvaluationBudget,
  type DimEvaluationEnvironment,
  type DimExpr,
} from "./dim-expr.js";

export type ShapeConstraint =
  | { readonly kind: "equal"; readonly lhs: DimExpr; readonly rhs: DimExpr }
  | { readonly kind: "lessEqual"; readonly lhs: DimExpr; readonly rhs: DimExpr }
  | { readonly kind: "nonNegative"; readonly value: DimExpr }
  | { readonly kind: "positive"; readonly value: DimExpr }
  | { readonly kind: "divisible"; readonly value: DimExpr; readonly divisor: DimExpr };

export type ConstraintEvaluation =
  | { readonly kind: "satisfied" }
  | { readonly kind: "violated"; readonly constraintIndex: number }
  | { readonly kind: "unresolved"; readonly symbols: readonly string[] };

export function evaluateConstraintSet(
  constraints: readonly ShapeConstraint[],
  environment: DimEvaluationEnvironment = {},
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): ConstraintEvaluation {
  const budget = createDimEvaluationBudget(environment, options.limits);
  const missing = new Set<string>();
  for (const [index, constraint] of constraints.entries()) {
    const result = evaluateConstraint(constraint, budget, `$.constraints[${index}]`);
    if (result.kind === "unresolved") {
      for (const symbol of result.symbols) missing.add(symbol);
    } else if (!result.value) {
      return { kind: "violated", constraintIndex: index };
    }
  }
  return missing.size > 0
    ? { kind: "unresolved", symbols: [...missing].sort() }
    : { kind: "satisfied" };
}

function evaluateConstraint(
  constraint: ShapeConstraint,
  budget: DimEvaluationBudget,
  path: string,
): { readonly kind: "resolved"; readonly value: boolean } | { readonly kind: "unresolved"; readonly symbols: readonly string[] } {
  switch (constraint.kind) {
    case "equal":
    case "lessEqual": {
      const lhs = evaluateDimExprWithBudget(constraint.lhs, budget, `${path}.lhs`);
      const rhs = evaluateDimExprWithBudget(constraint.rhs, budget, `${path}.rhs`);
      const unresolved = mergeUnresolved([lhs, rhs]);
      if (unresolved !== undefined) return unresolved;
      const lhsValue = requireResolved(lhs);
      const rhsValue = requireResolved(rhs);
      return { kind: "resolved", value: constraint.kind === "equal" ? lhsValue === rhsValue : lhsValue <= rhsValue };
    }
    case "nonNegative":
    case "positive": {
      const value = evaluateDimExprWithBudget(constraint.value, budget, `${path}.value`);
      if (value.kind === "unresolved") return value;
      return { kind: "resolved", value: constraint.kind === "nonNegative" ? value.value >= 0n : value.value > 0n };
    }
    case "divisible": {
      const value = evaluateDimExprWithBudget(constraint.value, budget, `${path}.value`);
      const divisor = evaluateDimExprWithBudget(constraint.divisor, budget, `${path}.divisor`);
      if (divisor.kind === "resolved" && divisor.value <= 0n) {
        throw new SemanticSchemaError({
          code: LAYOUT_DIAGNOSTIC_CODES.nonpositiveDivisor,
          stage: "verification",
          severity: "error",
          message: "divisible constraint divisor must be strictly positive",
          path: `${path}.divisor`,
        });
      }
      const unresolved = mergeUnresolved([value, divisor]);
      if (unresolved !== undefined) return unresolved;
      const valueResult = requireResolved(value);
      const divisorResult = requireResolved(divisor);
      return { kind: "resolved", value: euclideanModulo(valueResult, divisorResult) === 0n };
    }
  }
}

function mergeUnresolved(values: readonly DimEvaluation[]): { readonly kind: "unresolved"; readonly symbols: readonly string[] } | undefined {
  const symbols = [...new Set(values.flatMap((value) => value.kind === "unresolved" ? value.symbols : []))].sort();
  return symbols.length === 0 ? undefined : { kind: "unresolved", symbols };
}

function requireResolved(value: DimEvaluation): bigint {
  if (value.kind !== "resolved") throw new Error("internal: expected resolved dimension evaluation");
  return value.value;
}

import { expressionName } from "./analyzer.js";
import type { CompiledKernelInput, CudaLiteExpression } from "./types.js";

export type HostEvalValue = number | readonly [number, number, number];

export function evaluatePointerArgument(
  expression: CudaLiteExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): { readonly root: string; readonly offset: number } | undefined {
  if (expression.kind === "identifier") return { root: expression.name, offset: 0 };
  if (expression.kind !== "binary" || (expression.operator !== "+" && expression.operator !== "-")) return undefined;
  if (expression.left.kind !== "identifier") return undefined;
  const offset = evaluateHostNumber(expression.right, env, input);
  if (offset === undefined) return undefined;
  return {
    root: expression.left.name,
    offset: Math.trunc(offset) * (expression.operator === "-" ? -1 : 1),
  };
}

export function evaluateVectorExpressions(
  expressions: readonly CudaLiteExpression[],
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): readonly [number, number, number] | undefined {
  const x = expressions[0] ? evaluateHostNumber(expressions[0], env, input) : 1;
  const y = expressions[1] ? evaluateHostNumber(expressions[1], env, input) : 1;
  const z = expressions[2] ? evaluateHostNumber(expressions[2], env, input) : 1;
  if (x === undefined || y === undefined || z === undefined) return undefined;
  return [Math.max(1, Math.trunc(x)), Math.max(1, Math.trunc(y)), Math.max(1, Math.trunc(z))];
}

export function evaluateHostNumber(
  expression: CudaLiteExpression,
  env: ReadonlyMap<string, HostEvalValue>,
  input: CompiledKernelInput,
): number | undefined {
  switch (expression.kind) {
    case "number":
      return expression.value;
    case "identifier": {
      const local = env.get(expression.name);
      if (typeof local === "number") return local;
      return input.scalars?.[expression.name];
    }
    case "cast":
      return evaluateHostNumber(expression.expression, env, input);
    case "call":
      if (expressionName(expression.callee) === "sizeof" && expression.args[0]?.kind === "identifier") {
        return sizeofType(expression.args[0].name);
      }
      return undefined;
    case "member": {
      if (expression.object.kind !== "identifier") return undefined;
      const vector = env.get(expression.object.name);
      if (!isHostVector(vector)) return undefined;
      return expression.property === "x" ? vector[0] : expression.property === "y" ? vector[1] : expression.property === "z" ? vector[2] : undefined;
    }
    case "unary": {
      const value = evaluateHostNumber(expression.argument, env, input);
      if (value === undefined) return undefined;
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      if (expression.operator === "!") return value === 0 ? 1 : 0;
      return undefined;
    }
    case "binary": {
      const left = evaluateHostNumber(expression.left, env, input);
      const right = evaluateHostNumber(expression.right, env, input);
      if (left === undefined || right === undefined) return undefined;
      return evaluateHostBinary(expression.operator, left, right);
    }
    case "conditional": {
      const condition = evaluateHostNumber(expression.condition, env, input);
      if (condition === undefined) return undefined;
      return evaluateHostNumber(condition !== 0 ? expression.consequent : expression.alternate, env, input);
    }
    default:
      return undefined;
  }
}

export function isHostVector(value: HostEvalValue | undefined): value is readonly [number, number, number] {
  return Array.isArray(value) && value.length === 3;
}

export function isSingleInvocationGuard(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "binary") return false;
  const left = threadIdxXGuardSide(expression.left);
  const right = literalGuardSide(expression.right);
  if (left && right !== undefined) return guardAllowsOnlyThreadZero(expression.operator, right);
  const flippedLeft = threadIdxXGuardSide(expression.right);
  const flippedRight = literalGuardSide(expression.left);
  if (flippedLeft && flippedRight !== undefined) return guardAllowsOnlyThreadZero(flipComparison(expression.operator), flippedRight);
  if (expression.operator === "&&") return isSingleInvocationGuard(expression.left) || isSingleInvocationGuard(expression.right);
  return false;
}

function evaluateHostBinary(operator: string, left: number, right: number): number | undefined {
  switch (operator) {
    case "+": return left + right;
    case "-": return left - right;
    case "*": return left * right;
    case "/": return left / right;
    case "%": return left % right;
    case "<<": return Math.trunc(left) << Math.trunc(right);
    case ">>": return Math.trunc(left) >> Math.trunc(right);
    case "&": return Math.trunc(left) & Math.trunc(right);
    case "^": return Math.trunc(left) ^ Math.trunc(right);
    case "|": return Math.trunc(left) | Math.trunc(right);
    case "<": return left < right ? 1 : 0;
    case "<=": return left <= right ? 1 : 0;
    case ">": return left > right ? 1 : 0;
    case ">=": return left >= right ? 1 : 0;
    case "==": return left === right ? 1 : 0;
    case "!=": return left !== right ? 1 : 0;
    case "&&": return left !== 0 && right !== 0 ? 1 : 0;
    case "||": return left !== 0 || right !== 0 ? 1 : 0;
    default: return undefined;
  }
}

function threadIdxXGuardSide(expression: CudaLiteExpression): boolean {
  return expression.kind === "member" &&
    expression.property === "x" &&
    expression.object.kind === "identifier" &&
    expression.object.name === "threadIdx";
}

function literalGuardSide(expression: CudaLiteExpression): number | undefined {
  return expression.kind === "number" ? expression.value : undefined;
}

function guardAllowsOnlyThreadZero(operator: string, literal: number): boolean {
  switch (operator) {
    case "==": return literal === 0;
    case "<": return literal <= 1;
    case "<=": return literal < 1;
    default: return false;
  }
}

function flipComparison(operator: string): string {
  switch (operator) {
    case "<": return ">";
    case "<=": return ">=";
    case ">": return "<";
    case ">=": return "<=";
    default: return operator;
  }
}

function sizeofType(typeName: string): number {
  switch (typeName) {
    case "half": return 2;
    case "float":
    case "int":
    case "uint":
    case "unsigned":
    case "size_t":
    default:
      return 4;
  }
}

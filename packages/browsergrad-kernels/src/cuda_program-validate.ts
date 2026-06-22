import { KernelError } from "./types.js";
import type { Cuda1DCondition, Cuda1DExpression, Cuda1DStatement } from "./cuda_program-types.js";

export function cloneStatement(statement: Cuda1DStatement): Cuda1DStatement {
  switch (statement.op) {
    case "write":
      return {
        op: "write",
        index: cloneExpression(statement.index),
        value: cloneExpression(statement.value),
      };
    case "if":
      return {
        op: "if",
        condition: cloneCondition(statement.condition),
        body: statement.body.map(cloneStatement),
      };
  }
}

export function cloneCondition(condition: Cuda1DCondition): Cuda1DCondition {
  return {
    op: condition.op,
    left: cloneExpression(condition.left),
    right: cloneExpression(condition.right),
  };
}

export function cloneExpression(expression: Cuda1DExpression): Cuda1DExpression {
  switch (expression.op) {
    case "literal":
      return { op: "literal", value: expression.value };
    case "threadId":
    case "inputLength":
    case "outputLength":
      return { op: expression.op };
    case "param":
      return { op: "param", name: expression.name };
    case "read":
      return { op: "read", index: cloneExpression(expression.index) };
    case "outputRead":
      return { op: "outputRead", index: cloneExpression(expression.index) };
    case "add":
    case "sub":
    case "mul":
    case "div":
      return {
        op: expression.op,
        left: cloneExpression(expression.left),
        right: cloneExpression(expression.right),
      };
  }
}

export function validateParameters(
  parameters: Readonly<Record<string, number>>,
): Readonly<Record<string, number>> {
  const out: Record<string, number> = {};
  for (const [name, value] of Object.entries(parameters).sort(([a], [b]) =>
    a.localeCompare(b),
  )) {
    validateIdentifier(name, "parameter name");
    out[name] = validateFiniteNumber(value, `parameters.${name}`);
  }
  return out;
}

export function readParameter(
  parameters: Readonly<Record<string, number>>,
  name: string,
): number {
  validateIdentifier(name, "parameter name");
  const value = parameters[name];
  if (value === undefined) {
    throw new KernelError(`missing Cuda1DProgram parameter: ${name}`);
  }
  return value;
}

export function materializeVector(
  values: readonly number[] | undefined,
  length: number,
  name: string,
): number[] {
  if (values === undefined) return Array.from({ length }, () => 0);
  if (values.length !== length) {
    throw new KernelError(`${name} length must be ${length}`);
  }
  return values.map((value, index) =>
    validateFiniteNumber(value, `${name}[${index}]`),
  );
}

export function validateIdentifier(value: string, name: string): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new KernelError(`${name} must be a valid WGSL identifier`);
  }
  return value;
}

export function validatePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new KernelError(`${name} must be a positive integer`);
  }
  return value;
}

export function validateNonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new KernelError(`${name} must be a non-negative integer`);
  }
  return value;
}

export function validateInteger(value: number, name: string): number {
  if (!Number.isInteger(value)) {
    throw new KernelError(`${name} must be an integer`);
  }
  return value;
}

export function validateFiniteNumber(value: number, name: string): number {
  if (!Number.isFinite(value)) {
    throw new KernelError(`${name} must be finite`);
  }
  return value;
}

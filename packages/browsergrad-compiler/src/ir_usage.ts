import { expressionName } from "./analyzer.js";
import type { CudaLiteExpression, CudaLiteStatement } from "./types.js";

export function statementsUseCall(statements: readonly CudaLiteStatement[], names: ReadonlySet<string>): boolean {
  for (const statement of statements) {
    if (statement.kind === "expr" && expressionUsesCall(statement.expression, names)) return true;
    if (statement.kind === "var" && statement.init && expressionUsesCall(statement.init, names)) return true;
    if (statement.kind === "if" && (
      expressionUsesCall(statement.condition, names) ||
      statementsUseCall(statement.consequent, names) ||
      (statement.alternate ? statementsUseCall(statement.alternate, names) : false)
    )) return true;
    if (statement.kind === "for" && (
      (statement.init?.kind === "var" && statement.init.init ? expressionUsesCall(statement.init.init, names) : false) ||
      (statement.init && statement.init.kind !== "var" ? expressionUsesCall(statement.init, names) : false) ||
      (statement.condition ? expressionUsesCall(statement.condition, names) : false) ||
      (statement.update ? expressionUsesCall(statement.update, names) : false) ||
      statementsUseCall(statement.body, names)
    )) return true;
    if ((statement.kind === "while" || statement.kind === "do-while") && (
      expressionUsesCall(statement.condition, names) ||
      statementsUseCall(statement.body, names)
    )) return true;
    if (statement.kind === "block" && statementsUseCall(statement.body, names)) return true;
    if (statement.kind === "return" && statement.value && expressionUsesCall(statement.value, names)) return true;
  }
  return false;
}

export function statementsUseIdentifier(statements: readonly CudaLiteStatement[], names: ReadonlySet<string>): boolean {
  for (const statement of statements) {
    if (statement.kind === "expr" && expressionUsesIdentifier(statement.expression, names)) return true;
    if (statement.kind === "var" && statement.init && expressionUsesIdentifier(statement.init, names)) return true;
    if (statement.kind === "if" && (
      expressionUsesIdentifier(statement.condition, names) ||
      statementsUseIdentifier(statement.consequent, names) ||
      (statement.alternate ? statementsUseIdentifier(statement.alternate, names) : false)
    )) return true;
    if (statement.kind === "for" && (
      (statement.init?.kind === "var" && statement.init.init ? expressionUsesIdentifier(statement.init.init, names) : false) ||
      (statement.init && statement.init.kind !== "var" ? expressionUsesIdentifier(statement.init, names) : false) ||
      (statement.condition ? expressionUsesIdentifier(statement.condition, names) : false) ||
      (statement.update ? expressionUsesIdentifier(statement.update, names) : false) ||
      statementsUseIdentifier(statement.body, names)
    )) return true;
    if ((statement.kind === "while" || statement.kind === "do-while") && (
      expressionUsesIdentifier(statement.condition, names) ||
      statementsUseIdentifier(statement.body, names)
    )) return true;
    if (statement.kind === "block" && statementsUseIdentifier(statement.body, names)) return true;
    if (statement.kind === "return" && statement.value && expressionUsesIdentifier(statement.value, names)) return true;
    if (statement.kind === "kernel-launch" && (
      statement.grid.some((expression) => expressionUsesIdentifier(expression, names)) ||
      statement.block.some((expression) => expressionUsesIdentifier(expression, names)) ||
      statement.args.some((expression) => expressionUsesIdentifier(expression, names))
    )) return true;
    if (statement.kind === "dim3" && statement.args.some((expression) => expressionUsesIdentifier(expression, names))) return true;
    if (statement.kind === "asm" && (
      (statement.output ? expressionUsesIdentifier(statement.output, names) : false) ||
      statement.inputs.some((expression) => expressionUsesIdentifier(expression, names))
    )) return true;
  }
  return false;
}

function expressionUsesCall(expression: CudaLiteExpression, names: ReadonlySet<string>): boolean {
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    if (name && names.has(name)) return true;
    return expression.args.some((arg) => expressionUsesCall(arg, names)) || expressionUsesCall(expression.callee, names);
  }
  if (expression.kind === "cast") return expressionUsesCall(expression.expression, names);
  if (expression.kind === "member") return expressionUsesCall(expression.object, names);
  if (expression.kind === "index") return expressionUsesCall(expression.target, names) || expressionUsesCall(expression.index, names);
  if (expression.kind === "unary" || expression.kind === "update") return expressionUsesCall(expression.argument, names);
  if (expression.kind === "binary") return expressionUsesCall(expression.left, names) || expressionUsesCall(expression.right, names);
  if (expression.kind === "conditional") {
    return expressionUsesCall(expression.condition, names) ||
      expressionUsesCall(expression.consequent, names) ||
      expressionUsesCall(expression.alternate, names);
  }
  if (expression.kind === "assignment") return expressionUsesCall(expression.left, names) || expressionUsesCall(expression.right, names);
  if (expression.kind === "sequence") return expression.expressions.some((item) => expressionUsesCall(item, names));
  return false;
}

function expressionUsesIdentifier(expression: CudaLiteExpression, names: ReadonlySet<string>): boolean {
  if (expression.kind === "identifier") return names.has(expression.name);
  if (expression.kind === "cast") return expressionUsesIdentifier(expression.expression, names);
  if (expression.kind === "member") return expressionUsesIdentifier(expression.object, names);
  if (expression.kind === "index") return expressionUsesIdentifier(expression.target, names) || expressionUsesIdentifier(expression.index, names);
  if (expression.kind === "unary" || expression.kind === "update") return expressionUsesIdentifier(expression.argument, names);
  if (expression.kind === "binary") return expressionUsesIdentifier(expression.left, names) || expressionUsesIdentifier(expression.right, names);
  if (expression.kind === "conditional") {
    return expressionUsesIdentifier(expression.condition, names) ||
      expressionUsesIdentifier(expression.consequent, names) ||
      expressionUsesIdentifier(expression.alternate, names);
  }
  if (expression.kind === "assignment") return expressionUsesIdentifier(expression.left, names) || expressionUsesIdentifier(expression.right, names);
  if (expression.kind === "sequence") return expression.expressions.some((item) => expressionUsesIdentifier(item, names));
  if (expression.kind === "call") {
    return expressionUsesIdentifier(expression.callee, names) ||
      expression.args.some((arg) => expressionUsesIdentifier(arg, names));
  }
  return false;
}

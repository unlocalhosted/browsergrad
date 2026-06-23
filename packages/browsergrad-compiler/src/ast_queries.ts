import type { CudaLiteExpression, CudaLiteStatement } from "./types.js";

export function collectExternalDevicePoolNames(
  statements: readonly CudaLiteStatement[],
  excluded: ReadonlySet<string> = new Set(),
): readonly string[] {
  const pools = new Set<string>();
  walkCudaLiteExpressions(statements, (expression) => {
    if (expression.kind !== "call") return;
    const callName = expressionName(expression.callee);
    if (callName !== "deviceAllocate" && callName !== "streamOrderedAllocate") return;
    if (expression.args.length !== 2) return;
    const pool = expression.args[0];
    if (pool?.kind !== "unary" || pool.operator !== "&" || pool.argument.kind !== "identifier") return;
    if (!excluded.has(pool.argument.name)) pools.add(pool.argument.name);
  });
  return [...pools].sort();
}

export function collectKernelLaunchCallees(statements: readonly CudaLiteStatement[]): readonly string[] {
  const out = new Set<string>();
  walkCudaLiteStatements(statements, (statement) => {
    if (statement.kind === "kernel-launch") out.add(statement.callee);
  });
  return [...out].sort();
}

export function walkCudaLiteExpressions(
  statements: readonly CudaLiteStatement[],
  visitExpression: (expression: CudaLiteExpression) => void,
): void {
  for (const statement of statements) {
    if (statement.kind === "block") walkCudaLiteExpressions(statement.body, visitExpression);
    if (statement.kind === "var" && statement.init) walkExpression(statement.init, visitExpression);
    if (statement.kind === "dim3") {
      for (const arg of statement.args) walkExpression(arg, visitExpression);
    }
    if (statement.kind === "kernel-launch") {
      for (const arg of [...statement.grid, ...statement.block, ...statement.args]) {
        walkExpression(arg, visitExpression);
      }
    }
    if (statement.kind === "asm") {
      if (statement.output) walkExpression(statement.output, visitExpression);
      for (const input of statement.inputs) walkExpression(input, visitExpression);
    }
    if (statement.kind === "expr") walkExpression(statement.expression, visitExpression);
    if (statement.kind === "if") {
      walkExpression(statement.condition, visitExpression);
      walkCudaLiteExpressions(statement.consequent, visitExpression);
      if (statement.alternate) walkCudaLiteExpressions(statement.alternate, visitExpression);
    }
    if (statement.kind === "for") {
      if (statement.init?.kind === "var" && statement.init.init) walkExpression(statement.init.init, visitExpression);
      else if (statement.init && statement.init.kind !== "var") walkExpression(statement.init, visitExpression);
      if (statement.condition) walkExpression(statement.condition, visitExpression);
      if (statement.update) walkExpression(statement.update, visitExpression);
      walkCudaLiteExpressions(statement.body, visitExpression);
    }
    if (statement.kind === "return" && statement.value) walkExpression(statement.value, visitExpression);
  }
}

export function walkCudaLiteStatements(
  statements: readonly CudaLiteStatement[],
  visitStatement: (statement: CudaLiteStatement) => void,
): void {
  for (const statement of statements) {
    visitStatement(statement);
    if (statement.kind === "if") {
      walkCudaLiteStatements(statement.consequent, visitStatement);
      if (statement.alternate) walkCudaLiteStatements(statement.alternate, visitStatement);
    }
    if (statement.kind === "for") walkCudaLiteStatements(statement.body, visitStatement);
    if (statement.kind === "block") walkCudaLiteStatements(statement.body, visitStatement);
  }
}

function walkExpression(
  expression: CudaLiteExpression,
  visit: (expression: CudaLiteExpression) => void,
): void {
  visit(expression);
  if (expression.kind === "call") {
    walkExpression(expression.callee, visit);
    for (const arg of expression.args) walkExpression(arg, visit);
  } else if (expression.kind === "initializer") {
    for (const element of expression.elements) walkExpression(element, visit);
  } else if (expression.kind === "cast") {
    walkExpression(expression.expression, visit);
  } else if (expression.kind === "member") {
    walkExpression(expression.object, visit);
  } else if (expression.kind === "index") {
    walkExpression(expression.target, visit);
    walkExpression(expression.index, visit);
  } else if (expression.kind === "unary" || expression.kind === "update") {
    walkExpression(expression.argument, visit);
  } else if (expression.kind === "binary") {
    walkExpression(expression.left, visit);
    walkExpression(expression.right, visit);
  } else if (expression.kind === "conditional") {
    walkExpression(expression.condition, visit);
    walkExpression(expression.consequent, visit);
    walkExpression(expression.alternate, visit);
  } else if (expression.kind === "assignment") {
    walkExpression(expression.left, visit);
    walkExpression(expression.right, visit);
  } else if (expression.kind === "sequence") {
    for (const item of expression.expressions) walkExpression(item, visit);
  }
}

function expressionName(expression: CudaLiteExpression): string | undefined {
  return expression.kind === "identifier" ? expression.name : undefined;
}

import { CUDA_NAMED_CONSTANTS } from "./named_constants.js";
import {
  CudaLiteCompilerError,
  type CompiledCudaLiteKernel,
  type CudaLiteDiagnostic,
  type CudaLiteExpression,
  type CudaLiteKernel,
  type CudaLiteParam,
  type CudaLiteScalarType,
  type CudaLiteStatement,
  type SourceSpan,
} from "./types.js";

export interface CudaTrapLaunchPrecondition {
  readonly condition: CudaLiteExpression;
  readonly trapSpans: readonly SourceSpan[];
  readonly span: SourceSpan;
}

export function collectCudaTrapLaunchPreconditions(kernel: CudaLiteKernel): readonly CudaTrapLaunchPrecondition[] {
  const scalarParams = new Set(kernel.params.filter(isScalarLaunchParam).map((param) => param.name));
  const preconditions: CudaTrapLaunchPrecondition[] = [];

  const visitStatements = (statements: readonly CudaLiteStatement[]): void => {
    for (const statement of statements) {
      if (statement.kind === "if") {
        if ((statement.alternate === undefined || statement.alternate.length === 0) &&
          isScalarLaunchPreconditionExpression(statement.condition, scalarParams)) {
          const trapSpans = collectTrapCallSpans(statement.consequent);
          if (trapSpans.length > 0) {
            preconditions.push({
              condition: statement.condition,
              trapSpans,
              span: statement.span,
            });
          }
        }
        visitStatements(statement.consequent);
        if (statement.alternate) visitStatements(statement.alternate);
        continue;
      }
      if (statement.kind === "block") visitStatements(statement.body);
      else if (statement.kind === "for" || statement.kind === "while" || statement.kind === "do-while") visitStatements(statement.body);
    }
  };

  visitStatements(kernel.body);
  return preconditions;
}

export function collectCudaAllowedTrapCallSpanStarts(kernel: CudaLiteKernel): ReadonlySet<number> {
  const starts = new Set<number>();
  for (const precondition of collectCudaTrapLaunchPreconditions(kernel)) {
    for (const span of precondition.trapSpans) starts.add(span.start);
  }
  return starts;
}

export function createCudaTrapLaunchPreconditionDiagnostics(
  compiled: Pick<CompiledCudaLiteKernel, "analysis">,
  scalars: Readonly<Record<string, number>>,
): readonly CudaLiteDiagnostic[] {
  const diagnostics: CudaLiteDiagnostic[] = [];
  for (const precondition of collectCudaTrapLaunchPreconditions(compiled.analysis.kernel)) {
    const value = evalLaunchPreconditionExpression(precondition.condition, scalars);
    if (value === undefined || !Number.isFinite(value)) {
      diagnostics.push(diagnostic(
        "cuda-launch-precondition-unknown",
        "CUDA trap launch precondition could not be evaluated before WebGPU execution",
        precondition.condition.span,
      ));
      continue;
    }
    if (truthy(value)) {
      diagnostics.push(diagnostic(
        "cuda-launch-precondition-failed",
        "CUDA trap launch precondition failed; guarded __trap would execute",
        precondition.condition.span,
      ));
    }
  }
  return diagnostics;
}

export function assertCudaTrapLaunchPreconditions(
  compiled: Pick<CompiledCudaLiteKernel, "analysis">,
  scalars: Readonly<Record<string, number>>,
): void {
  const diagnostics = createCudaTrapLaunchPreconditionDiagnostics(compiled, scalars);
  if (diagnostics.length === 0) return;
  throw new CudaLiteCompilerError(diagnostics[0]?.message ?? "CUDA trap launch precondition failed", diagnostics);
}

function collectTrapCallSpans(statements: readonly CudaLiteStatement[]): SourceSpan[] {
  const spans: SourceSpan[] = [];
  const visitExpression = (expression: CudaLiteExpression): void => {
    if (expression.kind === "call" && expressionName(expression.callee) === "__trap") spans.push(expression.span);
    for (const child of expressionChildren(expression)) visitExpression(child);
  };
  const visitStatements = (body: readonly CudaLiteStatement[]): void => {
    for (const statement of body) {
      if (statement.kind === "expr") visitExpression(statement.expression);
      else if (statement.kind === "var" && statement.init) visitExpression(statement.init);
      else if (statement.kind === "dim3") statement.args.forEach(visitExpression);
      else if (statement.kind === "kernel-launch") {
        statement.grid.forEach(visitExpression);
        statement.block.forEach(visitExpression);
        statement.args.forEach(visitExpression);
      } else if (statement.kind === "asm") {
        if (statement.output) visitExpression(statement.output);
        statement.outputs?.forEach(visitExpression);
        statement.inputs.forEach(visitExpression);
      } else if (statement.kind === "if") {
        visitExpression(statement.condition);
        visitStatements(statement.consequent);
        if (statement.alternate) visitStatements(statement.alternate);
      } else if (statement.kind === "for") {
        if (statement.init && statement.init.kind !== "var") visitExpression(statement.init);
        else if (statement.init?.init) visitExpression(statement.init.init);
        if (statement.condition) visitExpression(statement.condition);
        if (statement.update) visitExpression(statement.update);
        visitStatements(statement.body);
      } else if (statement.kind === "while" || statement.kind === "do-while") {
        visitExpression(statement.condition);
        visitStatements(statement.body);
      } else if (statement.kind === "block") visitStatements(statement.body);
      else if (statement.kind === "return" && statement.value) visitExpression(statement.value);
      else if (statement.kind === "cooperative-group" && statement.partitionPredicate) visitExpression(statement.partitionPredicate);
    }
  };
  visitStatements(statements);
  return spans;
}

function isScalarLaunchPreconditionExpression(expression: CudaLiteExpression, scalarParams: ReadonlySet<string>): boolean {
  switch (expression.kind) {
    case "number":
      return true;
    case "identifier":
      return scalarParams.has(expression.name) || CUDA_NAMED_CONSTANTS.has(expression.name);
    case "cast":
      return !expression.pointer && isScalarLaunchPreconditionExpression(expression.expression, scalarParams);
    case "unary":
      return (expression.operator === "-" || expression.operator === "+" || expression.operator === "!" || expression.operator === "~") &&
        isScalarLaunchPreconditionExpression(expression.argument, scalarParams);
    case "binary":
      return isScalarLaunchPreconditionExpression(expression.left, scalarParams) &&
        isScalarLaunchPreconditionExpression(expression.right, scalarParams);
    case "conditional":
      return isScalarLaunchPreconditionExpression(expression.condition, scalarParams) &&
        isScalarLaunchPreconditionExpression(expression.consequent, scalarParams) &&
        isScalarLaunchPreconditionExpression(expression.alternate, scalarParams);
    default:
      return false;
  }
}

function evalLaunchPreconditionExpression(
  expression: CudaLiteExpression,
  scalars: Readonly<Record<string, number>>,
): number | undefined {
  switch (expression.kind) {
    case "number":
      return expression.value;
    case "identifier":
      return scalars[expression.name] ?? CUDA_NAMED_CONSTANTS.get(expression.name)?.value;
    case "cast":
      return evalLaunchPreconditionExpression(expression.expression, scalars);
    case "unary": {
      const value = evalLaunchPreconditionExpression(expression.argument, scalars);
      if (value === undefined) return undefined;
      if (expression.operator === "-") return -value;
      if (expression.operator === "+") return value;
      if (expression.operator === "!") return truthy(value) ? 0 : 1;
      if (expression.operator === "~") return ~Math.trunc(value);
      return undefined;
    }
    case "binary":
      return evalBinaryExpression(expression.operator, expression.left, expression.right, scalars);
    case "conditional": {
      const condition = evalLaunchPreconditionExpression(expression.condition, scalars);
      if (condition === undefined) return undefined;
      return evalLaunchPreconditionExpression(truthy(condition) ? expression.consequent : expression.alternate, scalars);
    }
    default:
      return undefined;
  }
}

function evalBinaryExpression(
  operator: Extract<CudaLiteExpression, { kind: "binary" }>["operator"],
  leftExpression: CudaLiteExpression,
  rightExpression: CudaLiteExpression,
  scalars: Readonly<Record<string, number>>,
): number | undefined {
  const left = evalLaunchPreconditionExpression(leftExpression, scalars);
  if (left === undefined) return undefined;
  if (operator === "&&") {
    if (!truthy(left)) return 0;
    const right = evalLaunchPreconditionExpression(rightExpression, scalars);
    return right === undefined ? undefined : truthy(right) ? 1 : 0;
  }
  if (operator === "||") {
    if (truthy(left)) return 1;
    const right = evalLaunchPreconditionExpression(rightExpression, scalars);
    return right === undefined ? undefined : truthy(right) ? 1 : 0;
  }
  const right = evalLaunchPreconditionExpression(rightExpression, scalars);
  if (right === undefined) return undefined;
  switch (operator) {
    case "+": return left + right;
    case "-": return left - right;
    case "*": return left * right;
    case "/": return right === 0 ? undefined : left / right;
    case "%": return right === 0 ? undefined : left % right;
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
  }
}

function expressionChildren(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  switch (expression.kind) {
    case "initializer":
      return expression.elements;
    case "cast":
      return [expression.expression];
    case "member":
      return [expression.object];
    case "index":
      return [expression.target, expression.index];
    case "call":
      return [expression.callee, ...expression.args];
    case "unary":
      return [expression.argument];
    case "binary":
      return [expression.left, expression.right];
    case "conditional":
      return [expression.condition, expression.consequent, expression.alternate];
    case "sequence":
      return expression.expressions;
    case "assignment":
      return [expression.left, expression.right];
    case "update":
      return [expression.argument];
    default:
      return [];
  }
}

function expressionName(expression: CudaLiteExpression): string | undefined {
  return expression.kind === "identifier" ? expression.name : undefined;
}

function isScalarLaunchParam(param: CudaLiteParam): boolean {
  return !param.pointer && isScalarLaunchType(param.valueType);
}

function isScalarLaunchType(valueType: Exclude<CudaLiteScalarType, "void">): boolean {
  return valueType === "float" ||
    valueType === "double" ||
    valueType === "int" ||
    valueType === "uint" ||
    valueType === "uchar" ||
    valueType === "half" ||
    valueType === "bf16" ||
    valueType === "bool";
}

function truthy(value: number): boolean {
  return value !== 0;
}

function diagnostic(code: string, message: string, span: SourceSpan): CudaLiteDiagnostic {
  return { code, severity: "error", message, span };
}

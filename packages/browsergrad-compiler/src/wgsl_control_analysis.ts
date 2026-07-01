import { walkCudaLiteExpressions } from "./ast_queries.js";
import { expressionName, rootIdentifier } from "./analyzer.js";
import {
  type CudaLiteCallExpression,
  type CudaLiteExpression,
  type CudaLiteStatement,
} from "./types.js";

export interface ConstantEvalContext {
  readonly ir: {
    readonly workgroupSize: readonly [number, number, number];
  };
}

export interface IfTrailingReturn {
  readonly condition: CudaLiteExpression;
  readonly beforeReturn: readonly CudaLiteStatement[];
  readonly activeBranch?: readonly CudaLiteStatement[];
}

export interface IfTrailingBreak {
  readonly condition: CudaLiteExpression;
  readonly beforeBreak: readonly CudaLiteStatement[];
}

export function constantBooleanExpression(expression: CudaLiteExpression, context: ConstantEvalContext): boolean | undefined {
  if (expression.kind === "number") return expression.value !== 0;
  if (expression.kind === "unary" && expression.operator === "!") {
    const value = constantBooleanExpression(expression.argument, context);
    return value === undefined ? undefined : !value;
  }
  if (expression.kind !== "binary") return undefined;
  const left = constantNumberExpression(expression.left, context);
  const right = constantNumberExpression(expression.right, context);
  if (left === undefined || right === undefined) return undefined;
  switch (expression.operator) {
    case "==":
      return left === right;
    case "!=":
      return left !== right;
    case "<":
      return left < right;
    case "<=":
      return left <= right;
    case ">":
      return left > right;
    case ">=":
      return left >= right;
    default:
      return undefined;
  }
}

export function constantNumberExpression(expression: CudaLiteExpression, context: ConstantEvalContext): number | undefined {
  if (expression.kind === "number") return expression.value;
  if (expression.kind === "member" && expression.object.kind === "identifier") {
    const axis = expression.property === "x" ? 0 : expression.property === "y" ? 1 : expression.property === "z" ? 2 : undefined;
    if (axis === undefined) return undefined;
    if (expression.object.name === "threadIdx" && context.ir.workgroupSize[axis] === 1) return 0;
  }
  return undefined;
}

export function splitIfTrailingVoidReturn(statement: CudaLiteStatement): IfTrailingReturn | undefined {
  if (statement.kind !== "if" || statement.consequent.length === 0) return undefined;
  const consequentReturn = trailingVoidReturn(statement.consequent);
  if (consequentReturn) {
    if (consequentReturn.beforeReturn.some(statementContainsBarrier)) return undefined;
    if (statement.alternate?.some(statementContainsBarrier)) return undefined;
    return optionalActiveBranch({
      condition: statement.condition,
      beforeReturn: consequentReturn.beforeReturn,
    }, statement.alternate);
  }
  if (!statement.alternate || statement.alternate.length === 0) return undefined;
  const alternateReturn = trailingVoidReturn(statement.alternate);
  if (!alternateReturn) return undefined;
  if (alternateReturn.beforeReturn.some(statementContainsBarrier)) return undefined;
  if (statement.consequent.some(statementContainsBarrier)) return undefined;
  return {
    condition: {
      kind: "unary",
      operator: "!",
      argument: statement.condition,
      span: statement.condition.span,
    },
    beforeReturn: alternateReturn.beforeReturn,
    activeBranch: statement.consequent,
  };
}

function optionalActiveBranch(
  split: Omit<IfTrailingReturn, "activeBranch">,
  activeBranch: readonly CudaLiteStatement[] | undefined,
): IfTrailingReturn {
  return activeBranch === undefined ? split : { ...split, activeBranch };
}

function trailingVoidReturn(
  statements: readonly CudaLiteStatement[],
): { readonly beforeReturn: readonly CudaLiteStatement[] } | undefined {
  const last = statements[statements.length - 1];
  if (last?.kind !== "return" || last.value) return undefined;
  return { beforeReturn: statements.slice(0, -1) };
}

export function splitIfTrailingBreak(statement: CudaLiteStatement): IfTrailingBreak | undefined {
  if (statement.kind !== "if" || statement.alternate || statement.consequent.length === 0) return undefined;
  const last = statement.consequent[statement.consequent.length - 1];
  if (last?.kind !== "break") return undefined;
  const beforeBreak = statement.consequent.slice(0, -1);
  if (beforeBreak.some(statementContainsBarrier)) return undefined;
  return { condition: statement.condition, beforeBreak };
}

export function statementContainsVoidReturn(statement: CudaLiteStatement): boolean {
  switch (statement.kind) {
    case "block":
      return statement.body.some(statementContainsVoidReturn);
    case "if":
      return statement.consequent.some(statementContainsVoidReturn) ||
        (statement.alternate?.some(statementContainsVoidReturn) ?? false);
    case "for":
    case "while":
    case "do-while":
      return statement.body.some(statementContainsVoidReturn);
    case "return":
      return statement.value === undefined;
    default:
      return false;
  }
}

export function statementContainsBarrier(statement: CudaLiteStatement): boolean {
  switch (statement.kind) {
    case "block":
      return statement.body.some(statementContainsBarrier);
    case "expr":
      return isBarrierCall(statement.expression);
    case "if":
      return statement.consequent.some(statementContainsBarrier) || (statement.alternate?.some(statementContainsBarrier) ?? false);
    case "for":
    case "while":
    case "do-while":
      return statement.body.some(statementContainsBarrier);
    default:
      return false;
  }
}

export function statementContainsSubgroupCall(statement: CudaLiteStatement): boolean {
  switch (statement.kind) {
    case "block":
      return statement.body.some(statementContainsSubgroupCall);
    case "var":
      return statement.init ? expressionContainsSubgroupCall(statement.init) : false;
    case "expr":
      return expressionContainsSubgroupCall(statement.expression);
    case "if":
      return expressionContainsSubgroupCall(statement.condition) ||
        statement.consequent.some(statementContainsSubgroupCall) ||
        (statement.alternate?.some(statementContainsSubgroupCall) ?? false);
    case "for":
      return (statement.init?.kind === "var" ? statement.init.init !== undefined && expressionContainsSubgroupCall(statement.init.init) : statement.init !== undefined && expressionContainsSubgroupCall(statement.init)) ||
        (statement.condition !== undefined && expressionContainsSubgroupCall(statement.condition)) ||
        (statement.update !== undefined && expressionContainsSubgroupCall(statement.update)) ||
        statement.body.some(statementContainsSubgroupCall);
    case "while":
    case "do-while":
      return expressionContainsSubgroupCall(statement.condition) || statement.body.some(statementContainsSubgroupCall);
    case "return":
      return statement.value ? expressionContainsSubgroupCall(statement.value) : false;
    default:
      return false;
  }
}

export function expressionContainsSubgroupCall(expression: CudaLiteExpression): boolean {
  let found = false;
  walkCudaLiteExpressions([{ kind: "expr", expression, span: expression.span }], (item) => {
    if (item.kind !== "call") return;
    if (item.callee.kind === "member" && isCooperativeGroupSubgroupMember(item.callee.property)) {
      found = true;
      return;
    }
    const name = expressionName(item.callee);
    if (name !== undefined && isSubgroupCallName(name)) found = true;
  });
  return found;
}

export function isCooperativeGroupSubgroupMember(name: string): boolean {
  return name === "shfl" ||
    name === "shfl_down" ||
    name === "shfl_up" ||
    name === "shfl_xor" ||
    name === "thread_rank" ||
    name === "size" ||
    name === "ballot" ||
    name === "any" ||
    name === "all";
}

export function isSubgroupCallName(name: string): boolean {
  return name.startsWith("bg_subgroup") ||
    name.startsWith("warp_reduce") ||
    name.startsWith("warpReduce") ||
    name.startsWith("__shfl") ||
    name.startsWith("__reduce") ||
    name === "cg::reduce" ||
    name === "cooperative_groups::reduce";
}

export function isBarrierCall(expression: CudaLiteExpression): boolean {
  if (expression.kind !== "call") return false;
  const name = expressionName(expression.callee);
  return name === "__syncthreads" ||
    name === "__syncwarp" ||
    name === "bg_grid_sync" ||
    name?.endsWith("::sync") === true ||
    expression.callee.kind === "member" && expression.callee.property === "sync";
}

export function countReturns(statements: readonly CudaLiteStatement[]): number {
  let count = 0;
  const walk = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "return") count++;
      if (item.kind === "block" || item.kind === "while" || item.kind === "do-while") walk(item.body);
      if (item.kind === "for") walk(item.body);
      if (item.kind === "if") {
        walk(item.consequent);
        if (item.alternate) walk(item.alternate);
      }
    }
  };
  walk(statements);
  return count;
}

export function collectAssignedNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const names = new Set<string>();
  const visitExpression = (expression: CudaLiteExpression): void => {
    if (expression.kind === "assignment") {
      const root = rootIdentifier(expression.left);
      if (root) names.add(root);
      visitExpression(expression.right);
      return;
    }
    if (expression.kind === "update") {
      const root = rootIdentifier(expression.argument);
      if (root) names.add(root);
    }
  };
  walkCudaLiteExpressions(statements, visitExpression);
  return names;
}

export function replaceExpressionNode(
  expression: CudaLiteExpression,
  target: CudaLiteExpression,
  replacement: CudaLiteExpression,
): CudaLiteExpression {
  if (expression === target) return replacement;
  switch (expression.kind) {
    case "number":
    case "string":
    case "identifier":
      return expression;
    case "initializer":
      return { ...expression, elements: expression.elements.map((element) => replaceExpressionNode(element, target, replacement)) };
    case "cast":
      return { ...expression, expression: replaceExpressionNode(expression.expression, target, replacement) };
    case "member":
      return { ...expression, object: replaceExpressionNode(expression.object, target, replacement) };
    case "index":
      return {
        ...expression,
        target: replaceExpressionNode(expression.target, target, replacement),
        index: replaceExpressionNode(expression.index, target, replacement),
      };
    case "call":
      return {
        ...expression,
        callee: replaceExpressionNode(expression.callee, target, replacement),
        args: expression.args.map((arg) => replaceExpressionNode(arg, target, replacement)),
      };
    case "unary":
      return { ...expression, argument: replaceExpressionNode(expression.argument, target, replacement) };
    case "binary":
      return {
        ...expression,
        left: replaceExpressionNode(expression.left, target, replacement),
        right: replaceExpressionNode(expression.right, target, replacement),
      };
    case "conditional":
      return {
        ...expression,
        condition: replaceExpressionNode(expression.condition, target, replacement),
        consequent: replaceExpressionNode(expression.consequent, target, replacement),
        alternate: replaceExpressionNode(expression.alternate, target, replacement),
      };
    case "sequence":
      return { ...expression, expressions: expression.expressions.map((item) => replaceExpressionNode(item, target, replacement)) };
    case "assignment":
      return {
        ...expression,
        left: replaceExpressionNode(expression.left, target, replacement),
        right: replaceExpressionNode(expression.right, target, replacement),
      };
    case "update":
      return { ...expression, argument: replaceExpressionNode(expression.argument, target, replacement) };
  }
}

export function firstBarrierDeviceFunctionCall(
  expression: CudaLiteExpression,
  resolve: (name: string, arity: number) => { readonly returnType: string; readonly body: readonly CudaLiteStatement[] } | undefined,
  canInline: (fn: { readonly returnType: string; readonly body: readonly CudaLiteStatement[] }, needsReturnValue: boolean) => boolean,
): CudaLiteCallExpression | undefined {
  if (expression.kind === "call") {
    const name = expressionName(expression.callee);
    const fn = name ? resolve(name, expression.args.length) : undefined;
    if (fn && canInline(fn, fn.returnType !== "void")) return expression;
  }
  for (const child of expressionChildren(expression)) {
    const found = firstBarrierDeviceFunctionCall(child, resolve, canInline);
    if (found) return found;
  }
  return undefined;
}

function expressionChildren(expression: CudaLiteExpression): readonly CudaLiteExpression[] {
  switch (expression.kind) {
    case "call":
      return [expression.callee, ...expression.args];
    case "initializer":
      return expression.elements;
    case "cast":
      return [expression.expression];
    case "member":
      return [expression.object];
    case "index":
      return [expression.target, expression.index];
    case "unary":
    case "update":
      return [expression.argument];
    case "binary":
      return [expression.left, expression.right];
    case "conditional":
      return [expression.condition, expression.consequent, expression.alternate];
    case "assignment":
      return [expression.left, expression.right];
    case "sequence":
      return expression.expressions;
    case "number":
    case "string":
    case "identifier":
      return [];
  }
}

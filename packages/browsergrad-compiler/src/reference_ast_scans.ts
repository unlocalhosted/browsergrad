import type {
  CudaLiteExpression,
  CudaLiteStatement,
  CudaLiteVarDecl,
} from "./types.js";

export function referenceSharedDeclarationsFor(
  statements: readonly CudaLiteStatement[],
  fallback: readonly CudaLiteVarDecl[],
): readonly CudaLiteVarDecl[] {
  if (fallback.length > 0) return fallback;
  const out: CudaLiteVarDecl[] = [];
  const visit = (items: readonly CudaLiteStatement[]): void => {
    for (const item of items) {
      if (item.kind === "var" && item.storage === "shared") out.push(item);
      if (item.kind === "if") {
        visit(item.consequent);
        if (item.alternate) visit(item.alternate);
      }
      if (item.kind === "for" || item.kind === "while" || item.kind === "do-while" || item.kind === "block") visit(item.body);
    }
  };
  visit(statements);
  return out;
}

export function referenceBlockScopedNames(statements: readonly CudaLiteStatement[]): ReadonlySet<string> {
  const names = new Set<string>();
  for (const statement of statements) {
    if ((statement.kind === "var" && statement.storage === "local") || statement.kind === "dim3" || statement.kind === "cooperative-group") {
      names.add(statement.name);
    }
    if (statement.kind === "for" && statement.init?.kind === "var" && statement.init.storage === "local") {
      names.add(statement.init.name);
    }
  }
  return names;
}

export function referenceUsesGridSync(statements: readonly CudaLiteStatement[]): boolean {
  const gridGroups = new Set<string>();
  const visitExpression = (expression: CudaLiteExpression): boolean => {
    if (expression.kind === "call") {
      if (
        expression.callee.kind === "member" &&
        expression.callee.property === "sync" &&
        expression.callee.object.kind === "identifier" &&
        gridGroups.has(expression.callee.object.name)
      ) return true;
      if (visitExpression(expression.callee)) return true;
      return expression.args.some(visitExpression);
    }
    if (expression.kind === "cast") return visitExpression(expression.expression);
    if (expression.kind === "member") return visitExpression(expression.object);
    if (expression.kind === "index") return visitExpression(expression.target) || visitExpression(expression.index);
    if (expression.kind === "unary" || expression.kind === "update") return visitExpression(expression.argument);
    if (expression.kind === "binary") return visitExpression(expression.left) || visitExpression(expression.right);
    if (expression.kind === "conditional") return visitExpression(expression.condition) || visitExpression(expression.consequent) || visitExpression(expression.alternate);
    if (expression.kind === "assignment") return visitExpression(expression.left) || visitExpression(expression.right);
    if (expression.kind === "sequence") return expression.expressions.some(visitExpression);
    return false;
  };
  const walk = (items: readonly CudaLiteStatement[]): boolean => {
    for (const item of items) {
      if (item.kind === "cooperative-group" && item.groupKind === "grid") gridGroups.add(item.name);
      if (item.kind === "expr" && visitExpression(item.expression)) return true;
      if (item.kind === "if") {
        if (visitExpression(item.condition) || walk(item.consequent) || (item.alternate ? walk(item.alternate) : false)) return true;
      }
      if (item.kind === "for") {
        if (item.init?.kind === "var" && item.init.init && visitExpression(item.init.init)) return true;
        if (item.init && item.init.kind !== "var" && visitExpression(item.init)) return true;
        if (item.condition && visitExpression(item.condition)) return true;
        if (item.update && visitExpression(item.update)) return true;
        if (walk(item.body)) return true;
      }
      if (item.kind === "while" || item.kind === "do-while") {
        if (visitExpression(item.condition) || walk(item.body)) return true;
      }
      if (item.kind === "block" && walk(item.body)) return true;
      if (item.kind === "return" && item.value && visitExpression(item.value)) return true;
    }
    return false;
  };
  return walk(statements);
}

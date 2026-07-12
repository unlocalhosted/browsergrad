import type {
  SemanticExpression,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";

export function semanticOperationExpressions(operation: SemanticKernelIrOperation): readonly SemanticExpression[] {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(...operation.target.indices, operation.value);
  if (operation.kind === "copy") expressions.push(...operation.source.indices, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "atomic") expressions.push(...operation.args, ...(operation.target?.indices ?? []));
  if (operation.kind === "call") expressions.push(...operation.args);
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  if (operation.kind === "return" && operation.value) expressions.push(operation.value);
  return expressions;
}

export function semanticExpressionChildren(expression: SemanticExpression): readonly SemanticExpression[] {
  switch (expression.kind) {
    case "literal":
    case "symbol":
    case "pointer-valid":
      return [];
    case "member":
      return [expression.object];
    case "index":
      return [expression.target, expression.index];
    case "call":
      return [expression.callee, ...expression.args];
    case "texture-read":
      return [expression.texture, expression.x, expression.y, ...(expression.z ? [expression.z] : [])];
    case "surface-read":
      return [expression.surface, expression.xBytes, expression.y, ...(expression.z ? [expression.z] : [])];
    case "cast":
      return [expression.expression];
    case "unary":
    case "update":
      return [expression.argument];
    case "binary":
      return [expression.left, expression.right];
    case "conditional":
      return [expression.condition, expression.consequent, expression.alternate];
    case "assignment":
      return [expression.target, expression.value];
    case "initializer":
      return expression.elements;
    case "sequence":
      return expression.expressions;
  }
}

export function semanticOperationsReferenceRoot(
  operations: readonly SemanticKernelIrOperation[],
  root: string,
): boolean {
  const expressionReferencesRoot = (expression: SemanticExpression): boolean =>
    expression.kind === "symbol" && expression.name === root ||
    semanticExpressionChildren(expression).some(expressionReferencesRoot);
  const memoryRefReferencesRoot = (ref: { readonly base: string; readonly indices: readonly SemanticExpression[] }): boolean =>
    ref.base === root || ref.indices.some(expressionReferencesRoot);

  return operations.some((operation) => {
    if (semanticOperationExpressions(operation).some(expressionReferencesRoot)) return true;
    if (operation.kind === "load") return memoryRefReferencesRoot(operation.source);
    if (operation.kind === "store") return memoryRefReferencesRoot(operation.target) || operation.reads.some(memoryRefReferencesRoot);
    if (operation.kind === "copy") return memoryRefReferencesRoot(operation.source) || memoryRefReferencesRoot(operation.target);
    if (operation.kind === "atomic" && operation.target) return memoryRefReferencesRoot(operation.target);
    if (operation.kind === "branch") return semanticOperationsReferenceRoot(operation.consequent, root) || semanticOperationsReferenceRoot(operation.alternate, root);
    if (operation.kind === "loop") return semanticOperationsReferenceRoot(operation.body, root) ||
      (operation.continuing !== undefined && semanticOperationsReferenceRoot(operation.continuing, root));
    if (operation.kind === "block") return semanticOperationsReferenceRoot(operation.body, root);
    // Device launches carry host-runtime pointer topology outside expression walking.
    if (operation.kind === "device-launch") return true;
    return false;
  });
}

export function isSemanticKernelIrOperation(
  value: SemanticKernelIrOperation | SemanticExpression,
): value is SemanticKernelIrOperation {
  switch (value.kind) {
    case "declare":
    case "dim3-declare":
    case "cooperative-group-declare":
    case "load":
    case "store":
    case "copy":
    case "copy-fence":
    case "surface-write":
    case "surface-read-store":
    case "atomic":
    case "expression":
    case "branch":
    case "loop":
    case "barrier":
    case "fence":
    case "device-launch":
    case "inline-asm":
    case "return":
    case "continue":
    case "break":
    case "block":
      return true;
    case "call":
      return typeof value.callee === "string";
    default:
      return false;
  }
}

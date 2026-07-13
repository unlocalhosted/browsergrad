import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { semanticAtomicOperation } from "./semantic_atomic_intrinsics.js";
import { semanticIdKey } from "./semantic_ids.js";
import { semanticPointerArgumentMemoryRef } from "./semantic_pointer_arguments.js";

export function semanticOperationExpressions(operation: SemanticKernelIrOperation): readonly SemanticExpression[] {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(...operation.target.indices, operation.value);
  if (operation.kind === "copy") expressions.push(...operation.source.indices, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "atomic") expressions.push(...operation.args, ...(operation.target?.indices ?? []));
  if (operation.kind === "call") expressions.push(...operation.args);
  if (operation.kind === "runtime-copy") expressions.push(...operation.args);
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

export function semanticAtomicMemoryRootNames(ir: SemanticKernelIrModule): ReadonlySet<string> {
  const roots = new Set<string>();
  const runtimePointerSources = new Map<string, Set<string>>();
  const atomicRuntimePointers = new Set<string>();
  const atomicParamIndexes = new Map<string, Set<number>>();
  const functionParams = new Map(ir.functions.map((fn) => [fn.name, fn.params]));
  const calls: { readonly owner?: string; readonly callee: string; readonly args: readonly SemanticExpression[] }[] = [];

  const recordAtomicRef = (owner: string | undefined, ref: SemanticMemoryRef): void => {
    if (ref.addressSpace === "local") {
      atomicRuntimePointers.add(semanticIdKey(ref.baseId));
      return;
    }
    const params = owner === undefined ? undefined : functionParams.get(owner);
    const index = params?.findIndex((param) => param.name === ref.base && param.pointer) ?? -1;
    if (owner !== undefined && index >= 0) {
      const indexes = atomicParamIndexes.get(owner) ?? new Set<number>();
      indexes.add(index);
      atomicParamIndexes.set(owner, indexes);
    } else {
      roots.add(ref.base);
    }
  };

  const scanExpression = (expression: SemanticExpression, owner?: string): void => {
    if (expression.kind === "call" && expression.callee.kind === "symbol") {
      const callee = expression.callee.name;
      if (semanticAtomicOperation(callee) !== undefined) {
        const ref = expression.args[0] ? semanticPointerArgumentMemoryRef(expression.args[0]) : undefined;
        if (ref) recordAtomicRef(owner, ref);
      } else {
        calls.push({ ...(owner === undefined ? {} : { owner }), callee, args: expression.args });
      }
    }
    for (const child of semanticExpressionChildren(expression)) scanExpression(child, owner);
  };

  const scanOperations = (operations: readonly SemanticKernelIrOperation[], owner?: string): void => {
    for (const operation of operations) {
      if (operation.kind === "declare" && operation.target.pointerRuntimeState === true && operation.init) {
        const ref = semanticPointerArgumentMemoryRef(operation.init);
        if (ref) runtimePointerSources.set(semanticIdKey(operation.target.id), new Set([ref.base]));
      }
      if (operation.kind === "pointer-rebind") {
        const pointerId = semanticIdKey(operation.target.id);
        const sources = runtimePointerSources.get(pointerId) ?? new Set<string>();
        sources.add(operation.source.base);
        runtimePointerSources.set(pointerId, sources);
      }
      if (operation.kind === "atomic" && operation.target) recordAtomicRef(owner, operation.target);
      if (operation.kind === "call") calls.push({ ...(owner === undefined ? {} : { owner }), callee: operation.callee, args: operation.args });
      for (const expression of semanticOperationExpressions(operation)) scanExpression(expression, owner);
      if (operation.kind === "block") scanOperations(operation.body, owner);
      if (operation.kind === "branch") {
        scanOperations(operation.consequent, owner);
        scanOperations(operation.alternate, owner);
      }
      if (operation.kind === "loop") {
        if (operation.init && isSemanticKernelIrOperation(operation.init)) scanOperations([operation.init], owner);
        scanOperations(operation.body, owner);
        if (operation.continuing) scanOperations(operation.continuing, owner);
      }
    }
  };

  scanOperations(ir.operations);
  for (const fn of ir.functions) scanOperations(fn.body, fn.name);

  let changed = true;
  while (changed) {
    changed = false;
    for (const call of calls) {
      const indexes = atomicParamIndexes.get(call.callee);
      if (!indexes) continue;
      for (const index of indexes) {
        const arg = call.args[index];
        const ref = arg ? semanticPointerArgumentMemoryRef(arg) : undefined;
        if (!ref) continue;
        const ownerParams = call.owner === undefined ? undefined : functionParams.get(call.owner);
        const ownerIndex = ownerParams?.findIndex((param) => param.name === ref.base && param.pointer) ?? -1;
        if (call.owner !== undefined && ownerIndex >= 0) {
          const ownerIndexes = atomicParamIndexes.get(call.owner) ?? new Set<number>();
          if (!ownerIndexes.has(ownerIndex)) {
            ownerIndexes.add(ownerIndex);
            atomicParamIndexes.set(call.owner, ownerIndexes);
            changed = true;
          }
        } else if (!roots.has(ref.base)) {
          roots.add(ref.base);
          changed = true;
        }
      }
    }
  }
  for (const [pointerId, sources] of runtimePointerSources) {
    if (!atomicRuntimePointers.has(pointerId)) continue;
    for (const source of sources) roots.add(source);
  }
  return roots;
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
    case "pointer-rebind":
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

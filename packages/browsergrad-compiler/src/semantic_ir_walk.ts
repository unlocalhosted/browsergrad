import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
  SemanticMatrixTileRef,
} from "./semantic_ir_types.js";
import { semanticAtomicOperation } from "./semantic_atomic_intrinsics.js";
import { semanticIdKey, semanticIdsEqual, semanticMemoryIdFromSymbol } from "./semantic_ids.js";
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
  if (operation.kind === "pool-allocate") {
    expressions.push(operation.sizeBytes);
    if (operation.pool.kind === "raw-pool") expressions.push(operation.pool.capacityBytes);
  }
  if (operation.kind === "pointer-array-rebind") expressions.push(operation.slot);
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
    if (operation.kind === "pool-allocate") {
      return operation.pool.kind === "device-pool"
        ? operation.pool.name === root
        : memoryRefReferencesRoot(operation.pool.data) || memoryRefReferencesRoot(operation.pool.offset);
    }
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

const atomicMemoryRootNamesCache = new WeakMap<SemanticKernelIrModule, ReadonlySet<string>>();

export function semanticAtomicMemoryRootNames(ir: SemanticKernelIrModule): ReadonlySet<string> {
  const cached = atomicMemoryRootNamesCache.get(ir);
  if (cached !== undefined) return cached;
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
      if (operation.kind === "declare" && operation.target.pointerArrayAliases !== undefined) {
        const sources = new Set(operation.target.pointerArrayAliases.flatMap((alias) =>
          alias === undefined ? [] : semanticPointerAliasRootNames(alias, ir)
        ));
        if (sources.size > 0) runtimePointerSources.set(semanticIdKey(operation.target.id), sources);
      }
      if (operation.kind === "pointer-rebind" || operation.kind === "pointer-array-rebind") {
        const pointerId = semanticIdKey(operation.target.id);
        const sources = runtimePointerSources.get(pointerId) ?? new Set<string>();
        sources.add(operation.source.base);
        runtimePointerSources.set(pointerId, sources);
      }
      if (operation.kind === "pool-allocate" && operation.pool.kind === "raw-pool") {
        roots.add(operation.pool.offset.base);
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
        if (ref.addressSpace === "local") {
          atomicRuntimePointers.add(semanticIdKey(ref.baseId));
        } else if (call.owner !== undefined && ownerIndex >= 0) {
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
  atomicMemoryRootNamesCache.set(ir, roots);
  return roots;
}

type SemanticPointerArrayAlias = Exclude<
  NonNullable<SemanticKernelIrModule["memory"][number]["pointerArrayAliases"]>[number],
  undefined
>;

function semanticPointerAliasRootNames(
  alias: SemanticPointerArrayAlias,
  ir: SemanticKernelIrModule,
): readonly string[] {
  if (alias.pointerSelection) {
    return [
      ...semanticPointerAliasRootNames(alias.pointerSelection.consequent, ir),
      ...semanticPointerAliasRootNames(alias.pointerSelection.alternate, ir),
    ];
  }
  if (alias.pointerRoot === undefined) return [];
  const root = [...ir.params, ...ir.memory].find((symbol) =>
    semanticIdsEqual(semanticMemoryIdFromSymbol(symbol.id), alias.pointerRoot!),
  );
  return root === undefined ? [] : [root.name];
}

export function walkSemanticOperations(
  operations: readonly SemanticKernelIrOperation[],
  visitExpression: (expression: SemanticExpression) => void,
): void {
  for (const operation of operations) walkSemanticOperation(operation, visitExpression);
}

export function collectSemanticPoolAllocations(
  operations: readonly SemanticKernelIrOperation[],
): readonly Extract<SemanticKernelIrOperation, { readonly kind: "pool-allocate" }>[] {
  const out: Extract<SemanticKernelIrOperation, { readonly kind: "pool-allocate" }>[] = [];
  const visit = (items: readonly SemanticKernelIrOperation[]): void => {
    for (const operation of items) {
      if (operation.kind === "pool-allocate") out.push(operation);
      if (operation.kind === "branch") visit([...operation.consequent, ...operation.alternate]);
      if (operation.kind === "loop") visit([...(operation.init && isSemanticKernelIrOperation(operation.init) ? [operation.init] : []), ...operation.body, ...(operation.continuing ?? [])]);
      if (operation.kind === "block") visit(operation.body);
    }
  };
  visit(operations);
  return out;
}

export function walkSemanticOperation(
  operation: SemanticKernelIrOperation,
  visitExpression: (expression: SemanticExpression) => void,
): void {
  switch (operation.kind) {
    case "declare":
      if (operation.init) walkSemanticExpression(operation.init, visitExpression);
      return;
    case "dim3-declare":
      for (const arg of operation.args) walkSemanticExpression(arg, visitExpression);
      return;
    case "cooperative-group-declare":
      if (operation.declaration.partitionPredicate) {
        walkSemanticExpression(operation.declaration.partitionPredicate, visitExpression);
      }
      return;
    case "load":
      walkSemanticMemoryRef(operation.source, visitExpression);
      return;
    case "store":
      walkSemanticMemoryRef(operation.target, visitExpression);
      walkSemanticExpression(operation.value, visitExpression);
      for (const read of operation.reads) walkSemanticMemoryRef(read, visitExpression);
      return;
    case "copy":
      walkSemanticMemoryRef(operation.source, visitExpression);
      walkSemanticMemoryRef(operation.target, visitExpression);
      return;
    case "matrix-fill":
      walkSemanticMatrixTileRef(operation.fragment, visitExpression);
      walkSemanticExpression(operation.value, visitExpression);
      return;
    case "matrix-load":
      walkSemanticMatrixTileRef(operation.fragment, visitExpression);
      walkSemanticMemoryRef(operation.source, visitExpression);
      walkSemanticExpression(operation.stride, visitExpression);
      return;
    case "matrix-mma":
      walkSemanticMatrixTileRef(operation.destination, visitExpression);
      walkSemanticMatrixTileRef(operation.a, visitExpression);
      walkSemanticMatrixTileRef(operation.b, visitExpression);
      walkSemanticMatrixTileRef(operation.accumulator, visitExpression);
      return;
    case "matrix-store":
      walkSemanticMemoryRef(operation.target, visitExpression);
      walkSemanticMatrixTileRef(operation.fragment, visitExpression);
      walkSemanticExpression(operation.stride, visitExpression);
      return;
    case "surface-write":
      walkSemanticExpression(operation.surface, visitExpression);
      walkSemanticExpression(operation.value, visitExpression);
      walkSemanticExpression(operation.xBytes, visitExpression);
      walkSemanticExpression(operation.y, visitExpression);
      if (operation.z) walkSemanticExpression(operation.z, visitExpression);
      return;
    case "surface-read-store":
      walkSemanticExpression(operation.target, visitExpression);
      walkSemanticExpression(operation.surface, visitExpression);
      walkSemanticExpression(operation.xBytes, visitExpression);
      walkSemanticExpression(operation.y, visitExpression);
      if (operation.z) walkSemanticExpression(operation.z, visitExpression);
      return;
    case "atomic":
      if (operation.target) walkSemanticMemoryRef(operation.target, visitExpression);
      for (const arg of operation.args) walkSemanticExpression(arg, visitExpression);
      return;
    case "call":
      for (const arg of operation.args) walkSemanticExpression(arg, visitExpression);
      for (const read of operation.reads) walkSemanticMemoryRef(read, visitExpression);
      return;
    case "runtime-copy":
      for (const arg of operation.args) walkSemanticExpression(arg, visitExpression);
      return;
    case "pool-allocate":
      walkSemanticExpression(operation.sizeBytes, visitExpression);
      if (operation.pool.kind === "raw-pool") {
        walkSemanticMemoryRef(operation.pool.data, visitExpression);
        walkSemanticMemoryRef(operation.pool.offset, visitExpression);
        walkSemanticExpression(operation.pool.capacityBytes, visitExpression);
      }
      return;
    case "expression":
      walkSemanticExpression(operation.expression, visitExpression);
      return;
    case "branch":
      walkSemanticExpression(operation.condition, visitExpression);
      walkSemanticOperations(operation.consequent, visitExpression);
      walkSemanticOperations(operation.alternate, visitExpression);
      return;
    case "loop":
      if (operation.init) {
        if (isSemanticKernelIrOperation(operation.init)) walkSemanticOperation(operation.init, visitExpression);
        else walkSemanticExpression(operation.init, visitExpression);
      }
      if (operation.condition) walkSemanticExpression(operation.condition, visitExpression);
      if (operation.update) walkSemanticExpression(operation.update, visitExpression);
      walkSemanticOperations(operation.body, visitExpression);
      if (operation.continuing) walkSemanticOperations(operation.continuing, visitExpression);
      return;
    case "device-launch":
      for (const expression of [...operation.launch.grid, ...operation.launch.block, ...operation.launch.args]) {
        walkSemanticExpression(expression, visitExpression);
      }
      return;
    case "return":
      if (operation.value) walkSemanticExpression(operation.value, visitExpression);
      return;
    case "block":
      walkSemanticOperations(operation.body, visitExpression);
      return;
    case "barrier":
    case "fence":
      return;
    case "inline-asm":
      for (const expression of operation.outputs) walkSemanticExpression(expression, visitExpression);
      for (const expression of operation.inputs) walkSemanticExpression(expression, visitExpression);
      return;
    case "continue":
    case "break":
      return;
  }
}

function walkSemanticMatrixTileRef(
  ref: SemanticMatrixTileRef,
  visitExpression: (expression: SemanticExpression) => void,
): void {
  for (const index of ref.indices) walkSemanticExpression(index, visitExpression);
}

export function walkSemanticMemoryRef(
  ref: SemanticMemoryRef,
  visitExpression: (expression: SemanticExpression) => void,
): void {
  for (const index of ref.indices) walkSemanticExpression(index, visitExpression);
}

export function walkSemanticExpression(
  expression: SemanticExpression,
  visitExpression: (expression: SemanticExpression) => void,
): void {
  visitExpression(expression);
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return;
    case "member":
      walkSemanticExpression(expression.object, visitExpression);
      return;
    case "index":
      walkSemanticExpression(expression.target, visitExpression);
      walkSemanticExpression(expression.index, visitExpression);
      return;
    case "call":
      walkSemanticExpression(expression.callee, visitExpression);
      for (const arg of expression.args) walkSemanticExpression(arg, visitExpression);
      return;
    case "texture-read":
      walkSemanticExpression(expression.texture, visitExpression);
      walkSemanticExpression(expression.x, visitExpression);
      walkSemanticExpression(expression.y, visitExpression);
      if (expression.z) walkSemanticExpression(expression.z, visitExpression);
      return;
    case "surface-read":
      walkSemanticExpression(expression.surface, visitExpression);
      walkSemanticExpression(expression.xBytes, visitExpression);
      walkSemanticExpression(expression.y, visitExpression);
      if (expression.z) walkSemanticExpression(expression.z, visitExpression);
      return;
    case "cast":
      walkSemanticExpression(expression.expression, visitExpression);
      return;
    case "unary":
    case "update":
      walkSemanticExpression(expression.argument, visitExpression);
      return;
    case "binary":
      walkSemanticExpression(expression.left, visitExpression);
      walkSemanticExpression(expression.right, visitExpression);
      return;
    case "conditional":
      walkSemanticExpression(expression.condition, visitExpression);
      walkSemanticExpression(expression.consequent, visitExpression);
      walkSemanticExpression(expression.alternate, visitExpression);
      return;
    case "assignment":
      walkSemanticExpression(expression.target, visitExpression);
      walkSemanticExpression(expression.value, visitExpression);
      return;
    case "initializer":
      for (const element of expression.elements) walkSemanticExpression(element, visitExpression);
      return;
    case "sequence":
      for (const item of expression.expressions) walkSemanticExpression(item, visitExpression);
      return;
  }
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
    case "pool-allocate":
    case "matrix-fill":
    case "matrix-load":
    case "matrix-mma":
    case "matrix-store":
    case "surface-write":
    case "surface-read-store":
    case "atomic":
    case "runtime-copy":
    case "pointer-rebind":
    case "pointer-array-rebind":
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

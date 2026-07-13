import type { SemanticExpression, SemanticKernelIrModule, SemanticKernelIrOperation, SemanticMemoryRef } from "./semantic_ir.js";
import type { CudaLiteScalarType, SourceSpan } from "./types.js";
import { createSemanticSymbolId, createUnresolvedSemanticSymbolId, semanticMemoryIdFromSymbol } from "./semantic_ids.js";
import { isSemanticKernelIrOperation, semanticAtomicMemoryRootNames, semanticExpressionChildren } from "./semantic_ir_walk.js";
import { isSemanticAtomicCallName } from "./semantic_atomic_intrinsics.js";
import { semanticFunctionForCall } from "./semantic_function_calls.js";
import { semanticPointerArgumentMemoryRef as semanticPointerArgMemoryRef } from "./semantic_pointer_arguments.js";
import { semanticMemoryRefStorageValueType } from "./semantic_memory_refs.js";
import { requireSemanticValueType } from "./semantic_value_type.js";
import { wgslAtomicScalar, wgslVectorScalar } from "./semantic_wgsl_types.js";
import { integerAtomicLoopHelperName, type WgslAtomicAddressSpace, type WgslIntegerLoopAtomicKind } from "./wgsl_atomic_helpers.js";
import { isCudaVectorType } from "./vector_types.js";

export function semanticIntViewAtomicAddressSpaces(ir: SemanticKernelIrModule): ReadonlySet<WgslAtomicAddressSpace> {
  const roots = semanticAtomicMemoryRootNames(ir);
  const addressSpaces = new Set<WgslAtomicAddressSpace>();
  for (const param of ir.params) {
    if (param.addressSpace === "storage" && roots.has(param.name) && isCudaVectorType(param.valueType) && wgslVectorScalar(param.valueType) === "i32") addressSpaces.add("storage");
  }
  for (const memory of ir.memory) {
    if (!roots.has(memory.name) || !isCudaVectorType(memory.valueType) || wgslVectorScalar(memory.valueType) !== "i32") continue;
    if (memory.kind === "shared") addressSpaces.add("workgroup");
    if (memory.kind === "device-global") addressSpaces.add("storage");
  }
  return addressSpaces;
}

export function semanticIntegerLoopAtomicHelperName(
  kind: WgslIntegerLoopAtomicKind,
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
): string {
  const storageValueType = semanticMemoryRefStorageValueType(ref, ir) ?? ref.valueType ?? "uint";
  return integerAtomicLoopHelperName(kind, {
    valueType: ref.valueType ?? "uint",
    storageValueType,
    storageScalar: wgslAtomicScalar(storageValueType),
    addressSpace: ref.addressSpace === "shared" ? "workgroup" : "storage",
  });
}

export interface SemanticWgslAtomicAnalysisHost {
  readonly memoryRefFromIndexExpression: (expression: SemanticExpression) => SemanticMemoryRef | undefined;
  readonly semanticFunctionParamAliasName: (fn: SemanticKernelIrModule["functions"][number], param: SemanticKernelIrModule["functions"][number]["params"][number]) => string | undefined;
}

export function createSemanticWgslAtomicAnalysis(host: SemanticWgslAtomicAnalysisHost) {
  const { memoryRefFromIndexExpression, semanticFunctionParamAliasName } = host;
  const zeroExpression = (span: SourceSpan): SemanticExpression => ({ kind: "literal", literalKind: "number", value: 0, valueType: "int", span });

function semanticAtomicStorageNames(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "storage") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicStorageNamesFromOperation(operation, functions)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicStorageNames(operation.consequent, functions)) names.add(name);
      for (const name of semanticAtomicStorageNames(operation.alternate, functions)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicStorageNames([operation.init], functions)) names.add(name);
      }
      for (const name of semanticAtomicStorageNames(operation.body, functions)) names.add(name);
      if (operation.continuing) for (const name of semanticAtomicStorageNames(operation.continuing, functions)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicStorageNames(operation.body, functions)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicDeviceGlobalNames(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "device-global") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicDeviceGlobalNamesFromOperation(operation, functions)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.consequent, functions)) names.add(name);
      for (const name of semanticAtomicDeviceGlobalNames(operation.alternate, functions)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicDeviceGlobalNames([operation.init], functions)) names.add(name);
      }
      for (const name of semanticAtomicDeviceGlobalNames(operation.body, functions)) names.add(name);
      if (operation.continuing) for (const name of semanticAtomicDeviceGlobalNames(operation.continuing, functions)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicDeviceGlobalNames(operation.body, functions)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicSharedNames(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const operation of operations) {
    if (operation.kind === "atomic" && operation.target?.addressSpace === "shared") {
      names.add(operation.target.base);
    }
    for (const name of semanticAtomicSharedNamesFromOperation(operation, functions)) names.add(name);
    if (operation.kind === "branch") {
      for (const name of semanticAtomicSharedNames(operation.consequent, functions)) names.add(name);
      for (const name of semanticAtomicSharedNames(operation.alternate, functions)) names.add(name);
    }
    if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) {
        for (const name of semanticAtomicSharedNames([operation.init], functions)) names.add(name);
      }
      for (const name of semanticAtomicSharedNames(operation.body, functions)) names.add(name);
      if (operation.continuing) for (const name of semanticAtomicSharedNames(operation.continuing, functions)) names.add(name);
    }
    if (operation.kind === "block") {
      for (const name of semanticAtomicSharedNames(operation.body, functions)) names.add(name);
    }
  }
  return names;
}

function semanticAtomicSharedNamesFromOperation(
  operation: SemanticKernelIrOperation,
  functions: readonly SemanticKernelIrModule["functions"][number][],
): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "return" && operation.value) expressions.push(operation.value);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  if (operation.kind === "call") {
    const fn = semanticFunctionForCall(operation, functions);
    const expression: Extract<SemanticExpression, { readonly kind: "call" }> = {
      kind: "call",
      callee: { kind: "symbol", id: fn ? createSemanticSymbolId("function", fn.name, fn.span) : createUnresolvedSemanticSymbolId(operation.callee, operation.span), name: operation.callee, addressSpace: "function", span: operation.span },
      args: operation.args,
      valueType: "void",
      span: operation.span,
    };
    for (const name of semanticAtomicSharedNamesFromFunctionCall(expression, functions)) names.add(name);
  }
  for (const expression of expressions) {
    for (const name of semanticAtomicSharedNamesFromExpression(expression, functions)) names.add(name);
  }
  return names;
}

function semanticAtomicSharedNamesFromExpression(
  expression: SemanticExpression,
  functions: readonly SemanticKernelIrModule["functions"][number][],
): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === "shared") names.add(target.base);
  if (expression.kind === "call") {
    for (const name of semanticAtomicSharedNamesFromFunctionCall(expression, functions)) names.add(name);
  }
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicSharedNamesFromExpression(child, functions)) names.add(name);
  }
  return names;
}

function semanticAtomicSharedNamesFromFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  functions: readonly SemanticKernelIrModule["functions"][number][],
): ReadonlySet<string> {
  const callee = expression.callee;
  if (callee.kind !== "symbol") return new Set();
  const fn = functions.find((item) => item.name === callee.name);
  if (!fn) return new Set();
  const pointerAtomicParams = semanticFunctionSharedPointerAtomicParams(fn);
  const names = new Set<string>();
  for (const [index, param] of fn.params.entries()) {
    if (!pointerAtomicParams.has(param.name)) continue;
    const ref = semanticPointerArgMemoryRef(expression.args[index] ?? zeroExpression(expression.span));
    if (ref?.addressSpace === "shared") names.add(ref.base);
  }
  return names;
}

function semanticFunctionSharedPointerAtomicParams(
  fn: SemanticKernelIrModule["functions"][number],
): ReadonlySet<string> {
  const pointerParams = new Set(fn.params.filter((param) => param.pointer && param.addressSpace === "shared").map((param) => param.name));
  const names = new Set<string>();
  for (const name of semanticAtomicSharedNames(fn.body)) {
    if (!pointerParams.has(name)) continue;
    names.add(name);
    const param = fn.params.find((candidate) => candidate.name === name);
    const alias = param === undefined ? undefined : semanticFunctionParamAliasName(fn, param);
    if (alias !== undefined) names.add(alias);
  }
  return names;
}

function semanticWgslFunctionSharedPointerAtomicParam(ir: SemanticKernelIrModule, name: string): boolean {
  return ir.functions.some((fn) => semanticFunctionSharedPointerAtomicParams(fn).has(name));
}

function semanticAtomicStorageNamesFromOperation(
  operation: SemanticKernelIrOperation,
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "return" && operation.value) expressions.push(operation.value);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  if (operation.kind === "call") {
    const fn = semanticFunctionForCall(operation, functions);
    const expression: Extract<SemanticExpression, { readonly kind: "call" }> = {
      kind: "call",
      callee: { kind: "symbol", id: fn ? createSemanticSymbolId("function", fn.name, fn.span) : createUnresolvedSemanticSymbolId(operation.callee, operation.span), name: operation.callee, addressSpace: "function", span: operation.span },
      args: operation.args,
      valueType: "void",
      span: operation.span,
    };
    for (const name of semanticAtomicStorageNamesFromFunctionCall(expression, functions)) names.add(name);
  }
  for (const expression of expressions) {
    for (const name of semanticAtomicStorageNamesFromExpression(expression, functions)) names.add(name);
  }
  return names;
}

function semanticAtomicStorageNamesFromExpression(
  expression: SemanticExpression,
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === "storage") names.add(target.base);
  if (expression.kind === "call") {
    for (const name of semanticAtomicStorageNamesFromFunctionCall(expression, functions)) names.add(name);
  }
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicStorageNamesFromExpression(child, functions)) names.add(name);
  }
  return names;
}

function semanticAtomicStorageNamesFromFunctionCall(
  expression: Extract<SemanticExpression, { readonly kind: "call" }>,
  functions: readonly SemanticKernelIrModule["functions"][number][],
): ReadonlySet<string> {
  const callee = expression.callee;
  if (callee.kind !== "symbol") return new Set();
  const fn = functions.find((item) => item.name === callee.name);
  if (!fn) return new Set();
  const pointerAtomicParams = semanticFunctionStoragePointerAtomicParams(fn);
  const names = new Set<string>();
  for (const [index, param] of fn.params.entries()) {
    if (!pointerAtomicParams.has(param.name)) continue;
    const ref = semanticPointerArgMemoryRef(expression.args[index] ?? zeroExpression(expression.span));
    if (ref?.addressSpace === "storage") names.add(ref.base);
  }
  return names;
}

function semanticFunctionStoragePointerAtomicParams(
  fn: SemanticKernelIrModule["functions"][number],
): ReadonlySet<string> {
  const pointerParams = new Set(fn.params.filter((param) => param.pointer && param.addressSpace === "storage").map((param) => param.name));
  const names = new Set<string>();
  for (const name of semanticAtomicStorageNames(fn.body)) {
    if (pointerParams.has(name)) names.add(name);
  }
  return names;
}

function semanticAtomicDeviceGlobalNamesFromOperation(
  operation: SemanticKernelIrOperation,
  functions: readonly SemanticKernelIrModule["functions"][number][] = [],
): ReadonlySet<string> {
  const names = new Set<string>();
  for (const name of semanticAtomicNamesFromOperation(operation, "device-global")) names.add(name);
  if (operation.kind === "call") {
    const fn = semanticFunctionForCall(operation, functions);
    if (fn) {
      const atomicParams = semanticFunctionStoragePointerAtomicParams(fn);
      for (const [index, param] of fn.params.entries()) {
        if (!atomicParams.has(param.name)) continue;
        const ref = semanticPointerArgMemoryRef(operation.args[index] ?? zeroExpression(operation.span));
        if (ref?.addressSpace === "device-global") names.add(ref.base);
      }
    }
  }
  return names;
}

function semanticAtomicNamesFromOperation(
  operation: SemanticKernelIrOperation,
  addressSpace: "storage" | "device-global" | "shared",
): ReadonlySet<string> {
  const expressions: SemanticExpression[] = [];
  if (operation.kind === "declare" && operation.init) expressions.push(operation.init);
  if (operation.kind === "store") expressions.push(operation.value, ...operation.target.indices);
  if (operation.kind === "surface-write") expressions.push(operation.surface, operation.value, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "surface-read-store") expressions.push(operation.target, operation.surface, operation.xBytes, operation.y, ...(operation.z ? [operation.z] : []));
  if (operation.kind === "expression") expressions.push(operation.expression);
  if (operation.kind === "return" && operation.value) expressions.push(operation.value);
  if (operation.kind === "branch") expressions.push(operation.condition);
  if (operation.kind === "loop") {
    if (operation.init && !isSemanticKernelIrOperation(operation.init)) expressions.push(operation.init);
    if (operation.condition) expressions.push(operation.condition);
    if (operation.update) expressions.push(operation.update);
  }
  const names = new Set<string>();
  for (const expression of expressions) {
    for (const name of semanticAtomicNamesFromExpression(expression, addressSpace)) names.add(name);
  }
  return names;
}

function semanticAtomicNamesFromExpression(
  expression: SemanticExpression,
  addressSpace: "storage" | "device-global" | "shared",
): ReadonlySet<string> {
  const names = new Set<string>();
  const target = expression.kind === "call" ? semanticAtomicCallTarget(expression) : undefined;
  if (target?.addressSpace === addressSpace) names.add(target.base);
  for (const child of semanticExpressionChildren(expression)) {
    for (const name of semanticAtomicNamesFromExpression(child, addressSpace)) names.add(name);
  }
  return names;
}

function semanticAtomicCallTarget(expression: Extract<SemanticExpression, { readonly kind: "call" }>): SemanticMemoryRef | undefined {
  if (
    expression.callee.kind !== "symbol" ||
    !isSemanticAtomicCallName(expression.callee.name)
  ) return undefined;
  const firstArg = expression.args[0];
  if (!firstArg) return undefined;
  if (
    firstArg.kind === "cast" &&
    firstArg.pointer &&
    (firstArg.valueType === "uint" || firstArg.valueType === "int") &&
    firstArg.expression.kind === "unary" &&
    firstArg.expression.operator === "&" &&
    firstArg.expression.argument.kind === "index"
  ) {
    const ref = memoryRefFromIndexExpression(firstArg.expression.argument);
    return ref ? { ...ref, valueType: firstArg.valueType } : undefined;
  }
  if (firstArg.kind === "unary" && firstArg.operator === "&" && firstArg.argument.kind === "index") {
    return memoryRefFromIndexExpression(firstArg.argument);
  }
  if (
    firstArg.kind === "unary" &&
    firstArg.operator === "&" &&
    firstArg.argument.kind === "symbol" &&
    (firstArg.argument.addressSpace === "device-global" || firstArg.argument.addressSpace === "shared")
  ) {
    const valueType = requireSemanticValueType(firstArg.argument.valueType, `atomic target '${firstArg.argument.name}'`, firstArg.argument.span);
    return {
      baseId: semanticMemoryIdFromSymbol(firstArg.argument.id),
      base: firstArg.argument.name,
      addressSpace: firstArg.argument.addressSpace,
      valueType,
      indices: [],
      fields: [],
      span: firstArg.argument.span,
    };
  }
  if (firstArg.kind === "index") return memoryRefFromIndexExpression(firstArg);
  if (firstArg.kind === "symbol" && (firstArg.addressSpace === "storage" || firstArg.addressSpace === "shared" || firstArg.addressSpace === "local")) {
    const valueType = requireSemanticValueType(firstArg.valueType, `atomic target '${firstArg.name}'`, firstArg.span);
    return {
      baseId: semanticMemoryIdFromSymbol(firstArg.id),
      base: firstArg.name,
      addressSpace: firstArg.addressSpace,
      valueType,
      indices: [],
      fields: [],
      span: firstArg.span,
    };
  }
  return undefined;
}

function semanticAtomicCallValueType(expression: SemanticExpression): CudaLiteScalarType | undefined {
  if (expression.kind !== "call") return undefined;
  return semanticAtomicCallTarget(expression)?.valueType;
}

  return {
    semanticAtomicStorageNames,
    semanticAtomicDeviceGlobalNames,
    semanticAtomicSharedNames,
    semanticFunctionSharedPointerAtomicParams,
    semanticWgslFunctionSharedPointerAtomicParam,
    semanticAtomicCallTarget,
    semanticAtomicCallValueType,
  };
}

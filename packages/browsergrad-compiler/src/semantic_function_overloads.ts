import type {
  CudaLiteSemanticFunction,
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { semanticPointerArgumentMemoryRef } from "./semantic_pointer_arguments.js";
import {
  createSemanticFunctionId,
  semanticSymbolIdFromFunction,
  type SemanticFunctionId,
} from "./semantic_ids.js";

export interface ResolvedSemanticFunctions {
  readonly operations: readonly SemanticKernelIrOperation[];
  readonly functions: readonly CudaLiteSemanticFunction[];
}

export function resolveSemanticFunctionOverloads(
  operations: readonly SemanticKernelIrOperation[],
  functions: readonly CudaLiteSemanticFunction[],
): ResolvedSemanticFunctions {
  const overloads = new Map<string, readonly CudaLiteSemanticFunction[]>();
  for (const fn of functions) {
    if (overloads.has(fn.name)) continue;
    overloads.set(fn.name, functions.filter((candidate) => candidate.name === fn.name));
  }
  const linkNames = new Map<CudaLiteSemanticFunction, string>();
  for (const candidates of overloads.values()) {
    for (const [index, fn] of candidates.entries()) {
      linkNames.set(fn, candidates.length === 1 ? fn.name : `${fn.name}__bg_overload_${index}`);
    }
  }
  const resolve = (name: string, args: readonly SemanticExpression[]): ResolvedOverload => {
    const candidates = overloads.get(name);
    if (!candidates || candidates.length === 0) return { name };
    if (candidates.length === 1) return { name, id: candidates[0]!.id, returnType: candidates[0]!.returnType };
    const matchingArity = candidates.filter((fn) => fn.params.length === args.length);
    const ranked = (matchingArity.length > 0 ? matchingArity : candidates)
      .map((fn) => ({ fn, score: semanticOverloadScore(fn, args) }))
      .filter((candidate) => candidate.score !== undefined)
      .sort((left, right) => right.score! - left.score!);
    const selected = ranked[0]?.fn ?? matchingArity[0] ?? candidates[0]!;
    const linkName = linkNames.get(selected) ?? selected.name;
    return {
      name: linkName,
      id: createSemanticFunctionId(linkName, selected.span),
      returnType: selected.returnType,
    };
  };
  return {
    operations: operations.map((operation) => resolveOperation(operation, resolve)),
    functions: functions.map((fn) => ({
      ...fn,
      id: createSemanticFunctionId(linkNames.get(fn) ?? fn.name, fn.span),
      name: linkNames.get(fn) ?? fn.name,
      body: fn.body.map((operation) => resolveOperation(operation, resolve)),
    })),
  };
}

function semanticOverloadScore(
  fn: CudaLiteSemanticFunction,
  args: readonly SemanticExpression[],
): number | undefined {
  if (fn.params.length !== args.length) return undefined;
  let score = 0;
  for (const [index, param] of fn.params.entries()) {
    const arg = args[index]!;
    if (param.cooperativeGroupKind !== undefined) {
      if (arg.kind !== "symbol") return undefined;
      score += 4;
      continue;
    }
    if (param.pointer) {
      const ref = semanticPointerArgumentMemoryRef(arg);
      if (!ref || ref.addressSpace !== param.addressSpace) return undefined;
      score += 8;
      if (ref.valueType === param.valueType) score += 4;
      continue;
    }
    if (semanticPointerArgumentMemoryRef(arg)) return undefined;
    if ("valueType" in arg && arg.valueType === param.valueType) score += 2;
  }
  return score;
}

interface ResolvedOverload {
  readonly name: string;
  readonly id?: SemanticFunctionId;
  readonly returnType?: CudaLiteSemanticFunction["returnType"];
}

type OverloadResolver = (name: string, args: readonly SemanticExpression[]) => ResolvedOverload;

function resolveOperation(
  operation: SemanticKernelIrOperation,
  resolve: OverloadResolver,
): SemanticKernelIrOperation {
  switch (operation.kind) {
    case "declare":
      return operation.init === undefined ? operation : { ...operation, init: resolveExpression(operation.init, resolve) };
    case "dim3-declare":
      return { ...operation, args: operation.args.map((arg) => resolveExpression(arg, resolve)) };
    case "store":
      return { ...operation, target: resolveMemoryRef(operation.target, resolve), value: resolveExpression(operation.value, resolve) };
    case "copy":
      return { ...operation, target: resolveMemoryRef(operation.target, resolve), source: resolveMemoryRef(operation.source, resolve) };
    case "surface-write":
      return { ...operation, surface: resolveExpression(operation.surface, resolve), value: resolveExpression(operation.value, resolve), xBytes: resolveExpression(operation.xBytes, resolve), y: resolveExpression(operation.y, resolve), ...(operation.z === undefined ? {} : { z: resolveExpression(operation.z, resolve) }) };
    case "surface-read-store":
      return { ...operation, target: resolveExpression(operation.target, resolve), surface: resolveExpression(operation.surface, resolve), xBytes: resolveExpression(operation.xBytes, resolve), y: resolveExpression(operation.y, resolve), ...(operation.z === undefined ? {} : { z: resolveExpression(operation.z, resolve) }) };
    case "atomic":
      return { ...operation, ...(operation.target === undefined ? {} : { target: resolveMemoryRef(operation.target, resolve) }), args: operation.args.map((arg) => resolveExpression(arg, resolve)) };
    case "call": {
      const args = operation.args.map((arg) => resolveExpression(arg, resolve));
      const resolved = resolve(operation.callee, args);
      return {
        ...operation,
        callee: resolved.name,
        ...(resolved.id === undefined ? {} : { calleeId: semanticSymbolIdFromFunction(resolved.id) }),
        args,
        reads: operation.reads.map((ref) => resolveMemoryRef(ref, resolve)),
      };
    }
    case "expression":
      return { ...operation, expression: resolveExpression(operation.expression, resolve) };
    case "branch":
      return { ...operation, condition: resolveExpression(operation.condition, resolve), consequent: operation.consequent.map((item) => resolveOperation(item, resolve)), alternate: operation.alternate.map((item) => resolveOperation(item, resolve)) };
    case "loop":
      return {
        ...operation,
        ...(operation.init === undefined ? {} : { init: semanticLoopInitIsOperation(operation.init) ? resolveOperation(operation.init, resolve) : resolveExpression(operation.init, resolve) }),
        ...(operation.condition === undefined ? {} : { condition: resolveExpression(operation.condition, resolve) }),
        ...(operation.update === undefined ? {} : { update: resolveExpression(operation.update, resolve) }),
        body: operation.body.map((item) => resolveOperation(item, resolve)),
      };
    case "block":
      return { ...operation, body: operation.body.map((item) => resolveOperation(item, resolve)) };
    case "return":
      return operation.value === undefined ? operation : { ...operation, value: resolveExpression(operation.value, resolve) };
    case "device-launch":
      return { ...operation, launch: { ...operation.launch, grid: operation.launch.grid.map((arg) => resolveExpression(arg, resolve)), block: operation.launch.block.map((arg) => resolveExpression(arg, resolve)), args: operation.launch.args.map((arg) => resolveExpression(arg, resolve)) } };
    default:
      return operation;
  }
}

function semanticLoopInitIsOperation(
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

function resolveMemoryRef(ref: SemanticMemoryRef, resolve: OverloadResolver): SemanticMemoryRef {
  return { ...ref, indices: ref.indices.map((index) => resolveExpression(index, resolve)) };
}

function resolveExpression(expression: SemanticExpression, resolve: OverloadResolver): SemanticExpression {
  switch (expression.kind) {
    case "member":
      return { ...expression, object: resolveExpression(expression.object, resolve) };
    case "index":
      return { ...expression, target: resolveExpression(expression.target, resolve), index: resolveExpression(expression.index, resolve) };
    case "call": {
      const args = expression.args.map((arg) => resolveExpression(arg, resolve));
      const callee = resolveExpression(expression.callee, resolve);
      if (callee.kind !== "symbol" || callee.addressSpace !== "function") return { ...expression, callee, args };
      const selected = resolve(callee.name, args);
      return {
        ...expression,
        callee: {
          ...callee,
          ...(selected.id === undefined ? {} : { id: semanticSymbolIdFromFunction(selected.id) }),
          name: selected.name,
          ...(selected.returnType === undefined ? {} : { valueType: selected.returnType }),
        },
        args,
        ...(selected.returnType === undefined ? {} : { valueType: selected.returnType }),
      };
    }
    case "cast":
      return { ...expression, expression: resolveExpression(expression.expression, resolve) };
    case "unary":
      return { ...expression, argument: resolveExpression(expression.argument, resolve) };
    case "binary":
      return { ...expression, left: resolveExpression(expression.left, resolve), right: resolveExpression(expression.right, resolve) };
    case "conditional":
      return { ...expression, condition: resolveExpression(expression.condition, resolve), consequent: resolveExpression(expression.consequent, resolve), alternate: resolveExpression(expression.alternate, resolve) };
    case "assignment":
      return { ...expression, target: resolveExpression(expression.target, resolve), value: resolveExpression(expression.value, resolve) };
    case "update":
      return { ...expression, argument: resolveExpression(expression.argument, resolve) };
    case "initializer":
      return { ...expression, elements: expression.elements.map((item) => resolveExpression(item, resolve)) };
    case "sequence":
      return { ...expression, expressions: expression.expressions.map((item) => resolveExpression(item, resolve)) };
    case "texture-read":
      return { ...expression, texture: resolveExpression(expression.texture, resolve), x: resolveExpression(expression.x, resolve), y: resolveExpression(expression.y, resolve), ...(expression.z === undefined ? {} : { z: resolveExpression(expression.z, resolve) }) };
    case "surface-read":
      return { ...expression, surface: resolveExpression(expression.surface, resolve), xBytes: resolveExpression(expression.xBytes, resolve), y: resolveExpression(expression.y, resolve), ...(expression.z === undefined ? {} : { z: resolveExpression(expression.z, resolve) }) };
    default:
      return expression;
  }
}

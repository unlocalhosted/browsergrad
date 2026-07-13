import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { isSemanticKernelIrOperation } from "./semantic_ir.js";
import type { KernelLaunch } from "./types.js";

export type SemanticRetirementReductionPlan =
  | {
      readonly supported: true;
      readonly phases: readonly [SemanticKernelIrModule, SemanticKernelIrModule?];
    }
  | {
      readonly supported: false;
      readonly reason: string;
    };

export function createSemanticRetirementReductionPlan(
  ir: SemanticKernelIrModule,
  launch: KernelLaunch,
): SemanticRetirementReductionPlan {
  const [group, partialReduction, retirementBranch, ...tail] = ir.operations;
  if (tail.length > 0 || group?.kind !== "cooperative-group-declare" ||
    partialReduction?.kind !== "call" || retirementBranch?.kind !== "branch" ||
    retirementBranch.alternate.length > 0 || !isGridCountGuard(retirementBranch.condition)) {
    return { supported: false, reason: "kernel does not match a retirement-count reduction shape" };
  }
  const body = retirementBranch.consequent;
  const tid = body.find((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "declare" }> =>
    operation.kind === "declare" && operation.target.name === "tid",
  );
  const sharedFlag = body.find((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "declare" }> =>
    operation.kind === "declare" && operation.target.addressSpace === "shared" && operation.target.valueType === "bool",
  );
  const sharedScratch = body.find((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "declare" }> =>
    operation.kind === "declare" && operation.target.addressSpace === "shared" && operation.target.valueType !== "bool",
  );
  const fence = body.find((operation) => operation.kind === "fence" && operation.callee === "__threadfence");
  const workgroupBarrier = body.find((operation) => operation.kind === "barrier" && operation.scope === "workgroup");
  const finalBranch = body.find((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "branch" }> =>
    operation.kind === "branch" && operation.condition.kind === "symbol" &&
    operation.condition.addressSpace === "shared" && operation.condition.name === sharedFlag?.target.name,
  );
  const ticketBranch = body.find((operation): operation is Extract<SemanticKernelIrOperation, { readonly kind: "branch" }> =>
    operation.kind === "branch" && operation !== finalBranch && operation.consequent.some(containsRetirementAtomicIncrement),
  );
  if (!tid || !sharedFlag || !sharedScratch || !fence || !workgroupBarrier || !ticketBranch || !finalBranch || finalBranch.alternate.length > 0) {
    return { supported: false, reason: "retirement-count reduction shape is incomplete" };
  }
  if (!finalBranch.consequent.some((operation) => operation.kind === "call" && semanticFunctionContainsBarrier(ir, operation.callee))) {
    return { supported: false, reason: "retirement-count final phase has no barrier reduction helper" };
  }
  const first: SemanticKernelIrModule = {
    ...ir,
    name: `${ir.name}_retirement_phase_0`,
    operations: [group, partialReduction],
    functions: reachableFunctions(ir, [group, partialReduction]),
  };
  if (launch.gridDim[0] <= 1) return { supported: true, phases: [first] };
  const finalOperations = replaceGridDimensionsInOperations(
    [group, tid, sharedScratch, ...finalBranch.consequent],
    launch.gridDim,
  );
  const final: SemanticKernelIrModule = {
    ...ir,
    name: `${ir.name}_retirement_phase_1`,
    operations: finalOperations,
    functions: reachableFunctions(ir, finalOperations),
  };
  return { supported: true, phases: [first, final] };
}

function reachableFunctions(
  ir: SemanticKernelIrModule,
  operations: readonly SemanticKernelIrOperation[],
): SemanticKernelIrModule["functions"] {
  const byName = new Map(ir.functions.map((fn) => [fn.name, fn]));
  const names = new Set<string>();
  const visitOperations = (items: readonly SemanticKernelIrOperation[]): void => {
    for (const operation of items) {
      if (operation.kind === "call") visitFunction(operation.callee);
      visitExpressions(operation, (expression) => {
        if (expression.kind === "call" && expression.callee.kind === "symbol") visitFunction(expression.callee.name);
      });
      if (operation.kind === "branch") {
        visitOperations(operation.consequent);
        visitOperations(operation.alternate);
      } else if (operation.kind === "loop" || operation.kind === "block") {
        visitOperations(operation.body);
        if (operation.kind === "loop" && operation.continuing) visitOperations(operation.continuing);
      }
    }
  };
  const visitFunction = (name: string): void => {
    if (names.has(name)) return;
    const fn = byName.get(name);
    if (!fn) return;
    names.add(name);
    visitOperations(fn.body);
  };
  visitOperations(operations);
  return ir.functions.filter((fn) => names.has(fn.name));
}

function visitExpressions(
  operation: SemanticKernelIrOperation,
  visit: (expression: SemanticExpression) => void,
): void {
  const walk = (expression: SemanticExpression): void => {
    visit(expression);
    switch (expression.kind) {
      case "literal":
      case "symbol": return;
      case "member": walk(expression.object); return;
      case "index": walk(expression.target); walk(expression.index); return;
      case "call": walk(expression.callee); expression.args.forEach(walk); return;
      case "texture-read": walk(expression.texture); walk(expression.x); walk(expression.y); if (expression.z) walk(expression.z); return;
      case "surface-read": walk(expression.surface); walk(expression.xBytes); walk(expression.y); if (expression.z) walk(expression.z); return;
      case "cast": walk(expression.expression); return;
      case "unary": walk(expression.argument); return;
      case "binary": walk(expression.left); walk(expression.right); return;
      case "conditional": walk(expression.condition); walk(expression.consequent); walk(expression.alternate); return;
      case "assignment": walk(expression.target); walk(expression.value); return;
      case "update": walk(expression.argument); return;
      case "initializer": expression.elements.forEach(walk); return;
      case "sequence": expression.expressions.forEach(walk); return;
    }
  };
  switch (operation.kind) {
    case "declare": if (operation.init) walk(operation.init); return;
    case "dim3-declare": operation.args.forEach(walk); return;
    case "store": walk(operation.value); return;
    case "surface-write": walk(operation.surface); walk(operation.value); walk(operation.xBytes); walk(operation.y); if (operation.z) walk(operation.z); return;
    case "surface-read-store": walk(operation.surface); walk(operation.xBytes); walk(operation.y); if (operation.z) walk(operation.z); return;
    case "atomic": operation.args.forEach(walk); return;
    case "call": operation.args.forEach(walk); return;
    case "expression": walk(operation.expression); return;
    case "branch": walk(operation.condition); return;
    case "loop": if (operation.init && !isSemanticKernelIrOperation(operation.init)) walk(operation.init); if (operation.condition) walk(operation.condition); if (operation.update) walk(operation.update); return;
    case "device-launch": [...operation.launch.grid, ...operation.launch.block, ...operation.launch.args].forEach(walk); return;
    case "return": if (operation.value) walk(operation.value); return;
    case "cooperative-group-declare":
    case "load":
    case "copy":
    case "copy-fence":
    case "matrix-fill":
    case "matrix-load":
    case "matrix-mma":
    case "matrix-store":
    case "barrier":
    case "fence":
    case "inline-asm":
    case "block":
    case "break":
    case "continue": return;
  }
}

function isGridCountGuard(expression: SemanticExpression): boolean {
  return expression.kind === "binary" && expression.operator === ">" &&
    isGridDimension(expression.left, "x") && expression.right.kind === "literal" && expression.right.value === 1;
}

function containsRetirementAtomicIncrement(operation: SemanticKernelIrOperation): boolean {
  if (operation.kind === "declare" && operation.init?.kind === "call" && operation.init.callee.kind === "symbol") {
    return operation.init.callee.name === "atomicInc" && operation.init.args[0]?.kind === "unary";
  }
  if (operation.kind === "branch") return operation.consequent.some(containsRetirementAtomicIncrement) || operation.alternate.some(containsRetirementAtomicIncrement);
  if (operation.kind === "loop") return operation.body.some(containsRetirementAtomicIncrement) || operation.continuing?.some(containsRetirementAtomicIncrement) === true;
  if (operation.kind === "block") return operation.body.some(containsRetirementAtomicIncrement);
  return false;
}

function semanticFunctionContainsBarrier(ir: SemanticKernelIrModule, name: string): boolean {
  const seen = new Set<string>();
  const visit = (callee: string): boolean => {
    if (seen.has(callee)) return false;
    seen.add(callee);
    const fn = ir.functions.find((candidate) => candidate.name === callee);
    if (!fn) return false;
    return fn.body.some((operation) => operation.kind === "barrier" ||
      operation.kind === "call" && visit(operation.callee) ||
      operation.kind === "branch" && [...operation.consequent, ...operation.alternate].some(containsBarrierOperation) ||
      (operation.kind === "loop" || operation.kind === "block") && operation.body.some(containsBarrierOperation));
  };
  return visit(name);
}

function containsBarrierOperation(operation: SemanticKernelIrOperation): boolean {
  if (operation.kind === "barrier") return true;
  if (operation.kind === "branch") return [...operation.consequent, ...operation.alternate].some(containsBarrierOperation);
  if (operation.kind === "loop") return operation.body.some(containsBarrierOperation) || operation.continuing?.some(containsBarrierOperation) === true;
  if (operation.kind === "block") return operation.body.some(containsBarrierOperation);
  return false;
}

function replaceGridDimensionsInOperations(
  operations: readonly SemanticKernelIrOperation[],
  gridDim: KernelLaunch["gridDim"],
): readonly SemanticKernelIrOperation[] {
  return operations.map((operation): SemanticKernelIrOperation => {
    switch (operation.kind) {
      case "declare": return operation.init ? { ...operation, init: replaceGridDimensions(operation.init, gridDim) } : operation;
      case "dim3-declare": return { ...operation, args: operation.args.map((arg) => replaceGridDimensions(arg, gridDim)) };
      case "cooperative-group-declare": return operation;
      case "load": return { ...operation, source: replaceGridDimensionsInMemoryRef(operation.source, gridDim) };
      case "store": return { ...operation, target: replaceGridDimensionsInMemoryRef(operation.target, gridDim), value: replaceGridDimensions(operation.value, gridDim), reads: operation.reads.map((ref) => replaceGridDimensionsInMemoryRef(ref, gridDim)) };
      case "copy": return { ...operation, source: replaceGridDimensionsInMemoryRef(operation.source, gridDim), target: replaceGridDimensionsInMemoryRef(operation.target, gridDim) };
      case "copy-fence": return operation;
      case "matrix-fill": return { ...operation, fragment: replaceGridDimensionsInMatrixRef(operation.fragment, gridDim), value: replaceGridDimensions(operation.value, gridDim) };
      case "matrix-load": return { ...operation, fragment: replaceGridDimensionsInMatrixRef(operation.fragment, gridDim), source: replaceGridDimensionsInMemoryRef(operation.source, gridDim), stride: replaceGridDimensions(operation.stride, gridDim) };
      case "matrix-mma": return { ...operation, destination: replaceGridDimensionsInMatrixRef(operation.destination, gridDim), a: replaceGridDimensionsInMatrixRef(operation.a, gridDim), b: replaceGridDimensionsInMatrixRef(operation.b, gridDim), accumulator: replaceGridDimensionsInMatrixRef(operation.accumulator, gridDim) };
      case "matrix-store": return { ...operation, target: replaceGridDimensionsInMemoryRef(operation.target, gridDim), fragment: replaceGridDimensionsInMatrixRef(operation.fragment, gridDim), stride: replaceGridDimensions(operation.stride, gridDim) };
      case "surface-write": return { ...operation, surface: replaceGridDimensions(operation.surface, gridDim), value: replaceGridDimensions(operation.value, gridDim), xBytes: replaceGridDimensions(operation.xBytes, gridDim), y: replaceGridDimensions(operation.y, gridDim), ...(operation.z === undefined ? {} : { z: replaceGridDimensions(operation.z, gridDim) }) };
      case "surface-read-store": return { ...operation, target: replaceGridDimensions(operation.target, gridDim), surface: replaceGridDimensions(operation.surface, gridDim), xBytes: replaceGridDimensions(operation.xBytes, gridDim), y: replaceGridDimensions(operation.y, gridDim), ...(operation.z === undefined ? {} : { z: replaceGridDimensions(operation.z, gridDim) }) };
      case "atomic": return { ...operation, ...(operation.target === undefined ? {} : { target: replaceGridDimensionsInMemoryRef(operation.target, gridDim) }), args: operation.args.map((arg) => replaceGridDimensions(arg, gridDim)) };
      case "call": return { ...operation, args: operation.args.map((arg) => replaceGridDimensions(arg, gridDim)), reads: operation.reads.map((ref) => replaceGridDimensionsInMemoryRef(ref, gridDim)) };
      case "runtime-copy": return { ...operation, args: operation.args.map((arg) => replaceGridDimensions(arg, gridDim)) };
      case "pointer-rebind": return { ...operation, source: replaceGridDimensionsInMemoryRef(operation.source, gridDim) };
      case "expression": return { ...operation, expression: replaceGridDimensions(operation.expression, gridDim) };
      case "branch": return { ...operation, condition: replaceGridDimensions(operation.condition, gridDim), consequent: replaceGridDimensionsInOperations(operation.consequent, gridDim), alternate: replaceGridDimensionsInOperations(operation.alternate, gridDim) };
      case "block": return { ...operation, body: replaceGridDimensionsInOperations(operation.body, gridDim) };
      case "loop": return { ...operation, ...(operation.init === undefined ? {} : { init: isSemanticKernelIrOperation(operation.init) ? replaceGridDimensionsInOperations([operation.init], gridDim)[0]! : replaceGridDimensions(operation.init, gridDim) }), ...(operation.condition === undefined ? {} : { condition: replaceGridDimensions(operation.condition, gridDim) }), ...(operation.update === undefined ? {} : { update: replaceGridDimensions(operation.update, gridDim) }), body: replaceGridDimensionsInOperations(operation.body, gridDim), ...(operation.continuing === undefined ? {} : { continuing: replaceGridDimensionsInOperations(operation.continuing, gridDim) }) };
      case "barrier":
      case "fence":
      case "inline-asm":
      case "break":
      case "continue": return operation;
      case "device-launch": return { ...operation, launch: { ...operation.launch, grid: operation.launch.grid.map((item) => replaceGridDimensions(item, gridDim)), block: operation.launch.block.map((item) => replaceGridDimensions(item, gridDim)), args: operation.launch.args.map((item) => replaceGridDimensions(item, gridDim)) } };
      case "return": return operation.value ? { ...operation, value: replaceGridDimensions(operation.value, gridDim) } : operation;
    }
  });
}

function replaceGridDimensionsInMemoryRef(ref: SemanticMemoryRef, gridDim: KernelLaunch["gridDim"]): SemanticMemoryRef {
  return { ...ref, indices: ref.indices.map((index) => replaceGridDimensions(index, gridDim)) };
}

function replaceGridDimensionsInMatrixRef<T extends { readonly indices: readonly SemanticExpression[] }>(ref: T, gridDim: KernelLaunch["gridDim"]): T {
  return { ...ref, indices: ref.indices.map((index) => replaceGridDimensions(index, gridDim)) };
}

function replaceGridDimensions(expression: SemanticExpression, gridDim: KernelLaunch["gridDim"]): SemanticExpression {
  if (isGridDimension(expression, "x")) return gridDimensionLiteral(gridDim[0], expression.span);
  if (isGridDimension(expression, "y")) return gridDimensionLiteral(gridDim[1], expression.span);
  if (isGridDimension(expression, "z")) return gridDimensionLiteral(gridDim[2], expression.span);
  switch (expression.kind) {
    case "literal":
    case "symbol":
    case "pointer-valid": return expression;
    case "member": return { ...expression, object: replaceGridDimensions(expression.object, gridDim) };
    case "index": return { ...expression, target: replaceGridDimensions(expression.target, gridDim), index: replaceGridDimensions(expression.index, gridDim) };
    case "call": return { ...expression, callee: replaceGridDimensions(expression.callee, gridDim), args: expression.args.map((arg) => replaceGridDimensions(arg, gridDim)) };
    case "texture-read": return { ...expression, texture: replaceGridDimensions(expression.texture, gridDim), x: replaceGridDimensions(expression.x, gridDim), y: replaceGridDimensions(expression.y, gridDim), ...(expression.z === undefined ? {} : { z: replaceGridDimensions(expression.z, gridDim) }) };
    case "surface-read": return { ...expression, surface: replaceGridDimensions(expression.surface, gridDim), xBytes: replaceGridDimensions(expression.xBytes, gridDim), y: replaceGridDimensions(expression.y, gridDim), ...(expression.z === undefined ? {} : { z: replaceGridDimensions(expression.z, gridDim) }) };
    case "cast": return { ...expression, expression: replaceGridDimensions(expression.expression, gridDim) };
    case "unary": return { ...expression, argument: replaceGridDimensions(expression.argument, gridDim) };
    case "binary": return { ...expression, left: replaceGridDimensions(expression.left, gridDim), right: replaceGridDimensions(expression.right, gridDim) };
    case "conditional": return { ...expression, condition: replaceGridDimensions(expression.condition, gridDim), consequent: replaceGridDimensions(expression.consequent, gridDim), alternate: replaceGridDimensions(expression.alternate, gridDim) };
    case "assignment": return { ...expression, target: replaceGridDimensions(expression.target, gridDim), value: replaceGridDimensions(expression.value, gridDim) };
    case "update": return { ...expression, argument: replaceGridDimensions(expression.argument, gridDim) };
    case "initializer": return { ...expression, elements: expression.elements.map((item) => replaceGridDimensions(item, gridDim)) };
    case "sequence": return { ...expression, expressions: expression.expressions.map((item) => replaceGridDimensions(item, gridDim)) };
  }
}

function isGridDimension(expression: SemanticExpression, property: "x" | "y" | "z"): boolean {
  return expression.kind === "member" && expression.property === property && expression.object.kind === "symbol" && expression.object.name === "gridDim";
}

function gridDimensionLiteral(value: number, span: SemanticExpression["span"]): SemanticExpression {
  return { kind: "literal", literalKind: "number", value, valueType: "uint", span };
}

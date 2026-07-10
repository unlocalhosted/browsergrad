import { isCudaRuntimeCopyCall } from "./cuda_runtime_copies.js";
import { isHostManagedRuntimeNoopCall } from "./cuda_runtime_noops.js";
import {
  createSemanticGridSyncPhasePlan,
  isSemanticGridSyncOperation,
  type SemanticGridSyncPhasePlan,
} from "./semantic_grid_sync.js";
import {
  isSemanticKernelIrOperation,
  semanticOperationExpressions,
} from "./semantic_ir_walk.js";
import type {
  SemanticExpression,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import type {
  CompiledCudaLiteKernel,
  SourceSpan,
} from "./types.js";

export type CudaRuntimeOperationKind =
  | "device-launch"
  | "device-sync"
  | "grid-sync"
  | "runtime-copy";

export interface CudaRuntimeOperation {
  readonly kind: CudaRuntimeOperationKind;
  readonly span: SourceSpan;
  readonly label: string;
}

export interface CudaRuntimePlan {
  readonly operations: readonly CudaRuntimeOperation[];
  readonly requiresHostOrchestration: boolean;
  readonly canRunSingleDispatchWebGpu: boolean;
  readonly referenceAvailable: boolean;
}

export type CudaGridSyncPhasePlan = SemanticGridSyncPhasePlan;

export function createCudaRuntimePlan(
  compiled: CompiledCudaLiteKernel,
): CudaRuntimePlan {
  const operations = collectSemanticRuntimeOperations(compiled.kernelIr.operations);
  return {
    operations,
    requiresHostOrchestration: operations.length > 0,
    canRunSingleDispatchWebGpu: operations.length === 0,
    referenceAvailable: operations.every((operation) => REFERENCE_RUNTIME_OPERATIONS.has(operation.kind)),
  };
}

const REFERENCE_RUNTIME_OPERATIONS: ReadonlySet<CudaRuntimeOperationKind> = new Set([
  "device-launch",
  "device-sync",
  "grid-sync",
  "runtime-copy",
]);

export function createCudaGridSyncPhasePlan(
  compiled: CompiledCudaLiteKernel,
): CudaGridSyncPhasePlan {
  return createSemanticGridSyncPhasePlan(compiled.kernelIr);
}

function collectSemanticRuntimeOperations(operations: readonly SemanticKernelIrOperation[]): readonly CudaRuntimeOperation[] {
  const runtime: CudaRuntimeOperation[] = [];
  const cooperativeGroups = new Map<string, string>();
  visitSemanticOperations(operations, (operation) => {
    if (operation.kind === "cooperative-group-declare") cooperativeGroups.set(operation.declaration.name, operation.declaration.groupKind);
    if (operation.kind === "device-launch") {
      runtime.push({
        kind: "device-launch",
        span: operation.span,
        label: `${operation.launch.callee}<<<...>>>`,
      });
      return;
    }
    if (isSemanticGridSyncOperation(operation, cooperativeGroups)) {
      runtime.push({ kind: "grid-sync", span: operation.span, label: "grid.sync()" });
      return;
    }
    if (operation.kind === "call" || operation.kind === "atomic") {
      const runtimeOperation = runtimeOperationForSemanticCall(operation.callee, operation.span);
      if (runtimeOperation) runtime.push(runtimeOperation);
    }
    for (const expression of semanticOperationExpressions(operation)) {
      visitSemanticExpression(expression, (item) => {
        const runtimeOperation = runtimeOperationForSemanticExpression(item, cooperativeGroups);
        if (runtimeOperation) runtime.push(runtimeOperation);
      });
    }
  });
  return [...uniqueRuntimeOperations(runtime)].sort((left, right) => left.span.start - right.span.start);
}

function runtimeOperationForSemanticExpression(
  expression: SemanticExpression,
  cooperativeGroups: ReadonlyMap<string, string>,
): CudaRuntimeOperation | undefined {
  if (expression.kind !== "call") return undefined;
  const callName = semanticCallName(expression.callee);
  const runtimeOperation = callName ? runtimeOperationForSemanticCall(callName, expression.span) : undefined;
  if (runtimeOperation) return runtimeOperation;
  const syncGroup = semanticSyncGroupName(expression);
  if (syncGroup && cooperativeGroups.get(syncGroup) === "grid") {
    return { kind: "grid-sync", span: expression.span, label: "grid.sync()" };
  }
  return undefined;
}

function runtimeOperationForSemanticCall(name: string, span: SourceSpan): CudaRuntimeOperation | undefined {
  if (isHostManagedRuntimeNoopCall(name)) return { kind: "device-sync", span, label: `${name}()` };
  if (isCudaRuntimeCopyCall(name)) return { kind: "runtime-copy", span, label: `${name}(...)` };
  return undefined;
}

function semanticCallName(callee: SemanticExpression): string | undefined {
  if (callee.kind === "symbol") return callee.name;
  if (callee.kind === "member") {
    const objectName = semanticCallName(callee.object);
    return objectName ? `${objectName}.${callee.property}` : undefined;
  }
  return undefined;
}

function semanticSyncGroupName(expression: Extract<SemanticExpression, { readonly kind: "call" }>): string | undefined {
  if (expression.callee.kind === "member" && expression.callee.property === "sync" && expression.callee.object.kind === "symbol") {
    return expression.callee.object.name;
  }
  if (expression.callee.kind === "symbol" && expression.callee.name.endsWith("::sync")) {
    const group = expression.args[0];
    return group?.kind === "symbol" ? group.name : undefined;
  }
  return undefined;
}

function uniqueRuntimeOperations(operations: readonly CudaRuntimeOperation[]): readonly CudaRuntimeOperation[] {
  const seen = new Set<string>();
  const out: CudaRuntimeOperation[] = [];
  for (const operation of operations) {
    const key = `${operation.kind}:${operation.span.start}:${operation.span.end}:${operation.label}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(operation);
  }
  return out;
}

function visitSemanticExpression(
  expression: SemanticExpression,
  visit: (expression: SemanticExpression) => void,
): void {
  visit(expression);
  switch (expression.kind) {
    case "literal":
    case "symbol":
      return;
    case "member":
      visitSemanticExpression(expression.object, visit);
      return;
    case "index":
      visitSemanticExpression(expression.target, visit);
      visitSemanticExpression(expression.index, visit);
      return;
    case "call":
      visitSemanticExpression(expression.callee, visit);
      for (const arg of expression.args) visitSemanticExpression(arg, visit);
      return;
    case "texture-read":
      visitSemanticExpression(expression.texture, visit);
      visitSemanticExpression(expression.x, visit);
      visitSemanticExpression(expression.y, visit);
      if (expression.z) visitSemanticExpression(expression.z, visit);
      return;
    case "surface-read":
      visitSemanticExpression(expression.surface, visit);
      visitSemanticExpression(expression.xBytes, visit);
      visitSemanticExpression(expression.y, visit);
      if (expression.z) visitSemanticExpression(expression.z, visit);
      return;
    case "cast":
      visitSemanticExpression(expression.expression, visit);
      return;
    case "unary":
    case "update":
      visitSemanticExpression(expression.argument, visit);
      return;
    case "binary":
      visitSemanticExpression(expression.left, visit);
      visitSemanticExpression(expression.right, visit);
      return;
    case "conditional":
      visitSemanticExpression(expression.condition, visit);
      visitSemanticExpression(expression.consequent, visit);
      visitSemanticExpression(expression.alternate, visit);
      return;
    case "assignment":
      visitSemanticExpression(expression.target, visit);
      visitSemanticExpression(expression.value, visit);
      return;
    case "initializer":
      for (const item of expression.elements) visitSemanticExpression(item, visit);
      return;
    case "sequence":
      for (const item of expression.expressions) visitSemanticExpression(item, visit);
      return;
  }
}

function visitSemanticOperations(
  operations: readonly SemanticKernelIrOperation[],
  visit: (operation: SemanticKernelIrOperation) => void,
): void {
  for (const operation of operations) {
    visit(operation);
    if (operation.kind === "block") visitSemanticOperations(operation.body, visit);
    else if (operation.kind === "branch") {
      visitSemanticOperations(operation.consequent, visit);
      visitSemanticOperations(operation.alternate, visit);
    } else if (operation.kind === "loop") {
      if (operation.init && isSemanticKernelIrOperation(operation.init)) visitSemanticOperations([operation.init], visit);
      visitSemanticOperations(operation.body, visit);
    }
  }
}

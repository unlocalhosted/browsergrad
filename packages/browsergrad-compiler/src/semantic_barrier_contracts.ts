import type { SemanticKernelIrModule, SemanticKernelIrOperation } from "./semantic_ir.js";
import type { CudaLiteBarrierUniformityFact } from "./types.js";

export function semanticOperationsContainActiveLaneControl(
  operations: readonly SemanticKernelIrOperation[],
): boolean {
  return operations.some((operation) => {
    if (operation.kind === "declare") return operation.target.name.startsWith("bg_active_lane") ||
      operation.target.name.startsWith("bg_barrier_loop_active_") ||
      operation.target.name.startsWith("bg_loop_active_");
    if (operation.kind === "branch") return semanticOperationsContainActiveLaneControl(operation.consequent) ||
      semanticOperationsContainActiveLaneControl(operation.alternate);
    if (operation.kind === "loop" || operation.kind === "block") {
      return semanticOperationsContainActiveLaneControl(operation.body);
    }
    return false;
  });
}

export function semanticBarrierFunctionNames(ir: SemanticKernelIrModule): ReadonlySet<string> {
  const names = new Set<string>();
  let changed = true;
  while (changed) {
    changed = false;
    for (const fn of ir.functions) {
      if (names.has(fn.name) || !semanticOperationsContainBarrier(fn.body, names)) continue;
      names.add(fn.name);
      changed = true;
    }
  }
  return names;
}

export function semanticOperationsContainBarrier(
  operations: readonly SemanticKernelIrOperation[],
  barrierFunctions: ReadonlySet<string>,
): boolean {
  return operations.some((operation) =>
    semanticOperationIsBarrier(operation, barrierFunctions) ||
    operation.kind === "branch" && (semanticOperationsContainBarrier(operation.consequent, barrierFunctions) || semanticOperationsContainBarrier(operation.alternate, barrierFunctions)) ||
    operation.kind === "loop" && semanticOperationsContainBarrier(operation.body, barrierFunctions) ||
    operation.kind === "block" && semanticOperationsContainBarrier(operation.body, barrierFunctions)
  );
}

export function semanticOperationIsBarrier(
  operation: SemanticKernelIrOperation,
  barrierFunctions: ReadonlySet<string>,
): boolean {
  return operation.kind === "barrier" ||
    operation.kind === "inline-asm" && operation.op?.kind === "bar-sync" ||
    operation.kind === "call" && barrierFunctions.has(operation.callee);
}

export function semanticBarrierShapeSupported(
  operations: readonly SemanticKernelIrOperation[],
  barrierFunctions: ReadonlySet<string>,
): boolean {
  for (const [index, operation] of operations.entries()) {
    const hasLaterBarrier = operations.slice(index + 1).some((item) => semanticOperationIsBarrier(item, barrierFunctions));
    if (semanticOperationIsBarrier(operation, barrierFunctions)) continue;
    if (operation.kind === "return") {
      if (hasLaterBarrier) return false;
      continue;
    }
    if (operation.kind !== "branch" && operation.kind !== "loop" && operation.kind !== "block") continue;
    const nested = operation.kind === "branch" ? [...operation.consequent, ...operation.alternate] : operation.body;
    if (semanticOperationsContainBarrier(nested, barrierFunctions)) return false;
    if (hasLaterBarrier && semanticOperationsContainReturn(nested)) return false;
  }
  return true;
}

export function semanticBarrierOperationsMatchUniformityProof(
  operations: readonly SemanticKernelIrOperation[],
  proof: CudaLiteBarrierUniformityFact | undefined,
  barrierFunctions: ReadonlySet<string> = new Set(),
): boolean {
  if (!proof?.verified) return false;
  const provenStarts = new Set(proof.barrierStatementStarts);
  let hasBarrier = false;
  const visit = (items: readonly SemanticKernelIrOperation[]): boolean => items.every((operation) => {
    if (semanticOperationIsBarrier(operation, barrierFunctions)) {
      hasBarrier = true;
      return provenStarts.has(operation.span.start);
    }
    if (operation.kind === "branch") return visit(operation.consequent) && visit(operation.alternate);
    if (operation.kind === "loop" || operation.kind === "block") return visit(operation.body);
    return true;
  });
  return visit(operations) && hasBarrier;
}

export function semanticBarrierOperationsMatchActiveLaneProof(
  operations: readonly SemanticKernelIrOperation[],
  proof: CudaLiteBarrierUniformityFact | undefined,
  barrierFunctions: ReadonlySet<string> = new Set(),
): boolean {
  if (!proof || semanticOperationsContainUnverifiedReturn(operations, proof)) return false;
  const provenStarts = new Set(proof.barrierStatementStarts);
  let hasBarrier = false;
  const visit = (items: readonly SemanticKernelIrOperation[]): boolean => items.every((operation) => {
    if (semanticOperationIsBarrier(operation, barrierFunctions)) {
      hasBarrier = true;
      return provenStarts.has(operation.span.start);
    }
    if (operation.kind === "branch") return visit(operation.consequent) && visit(operation.alternate);
    if (operation.kind === "loop" || operation.kind === "block") return visit(operation.body);
    return true;
  });
  return visit(operations) && hasBarrier;
}

function semanticOperationsContainUnverifiedReturn(
  operations: readonly SemanticKernelIrOperation[],
  proof: CudaLiteBarrierUniformityFact,
  inheritedUnverifiedControl = false,
): boolean {
  const unverifiedStarts = new Set(proof.unverifiedControlStatementStarts);
  return operations.some((operation) => {
    const unverifiedControl = inheritedUnverifiedControl || unverifiedStarts.has(operation.span.start);
    if (operation.kind === "return") return unverifiedControl;
    if (operation.kind === "branch") {
      return semanticOperationsContainUnverifiedReturn(operation.consequent, proof, unverifiedControl) ||
        semanticOperationsContainUnverifiedReturn(operation.alternate, proof, unverifiedControl);
    }
    if (operation.kind === "loop" || operation.kind === "block") {
      return semanticOperationsContainUnverifiedReturn(operation.body, proof, unverifiedControl);
    }
    return false;
  });
}

function semanticOperationsContainReturn(operations: readonly SemanticKernelIrOperation[]): boolean {
  return operations.some((operation) =>
    operation.kind === "return" ||
    operation.kind === "branch" && (semanticOperationsContainReturn(operation.consequent) || semanticOperationsContainReturn(operation.alternate)) ||
    operation.kind === "loop" && semanticOperationsContainReturn(operation.body) ||
    operation.kind === "block" && semanticOperationsContainReturn(operation.body)
  );
}

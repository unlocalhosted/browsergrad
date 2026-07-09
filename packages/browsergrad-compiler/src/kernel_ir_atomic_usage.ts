import { kernelIrUsesCall } from "./kernel_ir_usage.js";
import {
  semanticAtomicCallNamesForOperation,
  type SemanticAtomicOp,
} from "./semantic_atomic_intrinsics.js";
import type { KernelIrModule } from "./types.js";

export function kernelIrUsesAtomicOperation(ir: KernelIrModule, op: SemanticAtomicOp): boolean {
  return kernelIrUsesAnyCall(ir, semanticAtomicCallNamesForOperation(op));
}

export function kernelIrUsesAtomicOperations(ir: KernelIrModule, ops: readonly SemanticAtomicOp[]): boolean {
  const names = new Set<string>();
  for (const op of ops) {
    for (const name of semanticAtomicCallNamesForOperation(op)) names.add(name);
  }
  return kernelIrUsesAnyCall(ir, names);
}

function kernelIrUsesAnyCall(ir: KernelIrModule, names: ReadonlySet<string>): boolean {
  return kernelIrUsesCall(ir, names);
}

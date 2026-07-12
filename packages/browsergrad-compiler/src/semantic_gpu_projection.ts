import { isCudaRuntimeCopyCall } from "./cuda_runtime_copies.js";
import { isHostManagedRuntimeNoopCall } from "./cuda_runtime_noops.js";
import type { SemanticExpression, SemanticKernelIrOperation } from "./semantic_ir.js";
import { validateSemanticKernelIr } from "./semantic_ir_verifier.js";
import { typeCheckSemanticKernelIr } from "./semantic_type_check.js";
import { legalizeSemanticKernelIrForWgsl, type WgslLegalizedSemanticKernelIr } from "./wgsl_legalization.js";
import type { CompiledCudaLiteKernel } from "./types.js";

export type ProjectedSemanticGpuIr = WgslLegalizedSemanticKernelIr;

export function projectSemanticHostRuntimeToGpuIr(
  ir: CompiledCudaLiteKernel["kernelIr"],
): ProjectedSemanticGpuIr {
  const projected = {
    ...ir,
    operations: projectOperations(ir.operations),
    functions: ir.functions.map((fn) => ({ ...fn, body: projectOperations(fn.body) })),
  };
  return legalizeSemanticKernelIrForWgsl(
    typeCheckSemanticKernelIr(validateSemanticKernelIr(projected)),
  );
}

function projectOperations(
  operations: readonly SemanticKernelIrOperation[],
): readonly SemanticKernelIrOperation[] {
  return operations.flatMap((operation): readonly SemanticKernelIrOperation[] => {
    if (operation.kind === "device-launch") return [];
    if (operation.kind === "runtime-copy") return [];
    if (operation.kind === "call" && isHostRuntimeCall(operation.callee)) return [];
    if (operation.kind === "expression" && isHostRuntimeExpression(operation.expression)) return [];
    if (operation.kind === "block") return [{ ...operation, body: projectOperations(operation.body) }];
    if (operation.kind === "branch") {
      return [{
        ...operation,
        consequent: projectOperations(operation.consequent),
        alternate: projectOperations(operation.alternate),
      }];
    }
    if (operation.kind === "loop") {
      return [{
        ...operation,
        body: projectOperations(operation.body),
        ...(operation.continuing === undefined ? {} : { continuing: projectOperations(operation.continuing) }),
      }];
    }
    return [operation];
  });
}

function isHostRuntimeExpression(expression: SemanticExpression): boolean {
  return expression.kind === "call" && expression.callee.kind === "symbol" && isHostRuntimeCall(expression.callee.name);
}

function isHostRuntimeCall(name: string): boolean {
  return isHostManagedRuntimeNoopCall(name) || isCudaRuntimeCopyCall(name);
}

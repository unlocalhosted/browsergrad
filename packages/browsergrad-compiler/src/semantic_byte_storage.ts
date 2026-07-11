import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { walkSemanticOperations } from "./semantic_ir.js";
import { semanticPointerArgumentMemoryRef } from "./semantic_pointer_arguments.js";
import { cudaVectorLaneCount } from "./vector_types.js";

export function semanticDirectByteStorageParamSupported(
  ir: SemanticKernelIrModule,
  paramName: string,
): boolean {
  if (!semanticByteStorageOperationsSupported(ir, ir.operations, paramName)) return false;
  let supported = true;
  walkSemanticOperations(ir.operations, (expression) => {
    if (!supported || expression.kind !== "call" || expression.callee.kind !== "symbol") return;
    supported = semanticByteStorageCallSupported(ir, expression.callee.name, expression.args, paramName);
  });
  return supported;
}

function semanticByteStorageOperationsSupported(
  ir: SemanticKernelIrModule,
  operations: readonly SemanticKernelIrOperation[],
  paramName: string,
): boolean {
  for (const operation of operations) {
    const refs = semanticOperationMemoryRefs(operation);
    if (refs.some((ref) => ref.base === paramName && !semanticByteStorageRefSupported(ref))) return false;
    if (operation.kind === "call" && !semanticByteStorageCallSupported(ir, operation.callee, operation.args, paramName)) return false;
    if (operation.kind === "branch" && (
      !semanticByteStorageOperationsSupported(ir, operation.consequent, paramName) ||
      !semanticByteStorageOperationsSupported(ir, operation.alternate, paramName)
    )) return false;
    if ((operation.kind === "loop" || operation.kind === "block") &&
      !semanticByteStorageOperationsSupported(ir, operation.body, paramName)) return false;
  }
  return true;
}

function semanticByteStorageRefSupported(ref: SemanticMemoryRef): boolean {
  return ref.valueType === "uchar" ||
    ref.packedByteLanes !== undefined && cudaVectorLaneCount(ref.valueType) === ref.packedByteLanes;
}

function semanticOperationMemoryRefs(operation: SemanticKernelIrOperation): readonly SemanticMemoryRef[] {
  switch (operation.kind) {
    case "load": return [operation.source];
    case "store": return [operation.target, ...operation.reads];
    case "copy": return [operation.source, operation.target];
    case "atomic": return operation.target ? [operation.target] : [];
    case "call": return operation.reads;
    default: return [];
  }
}

function semanticByteStorageCallSupported(
  ir: SemanticKernelIrModule | undefined,
  callee: string,
  args: readonly SemanticExpression[],
  paramName: string,
): boolean {
  const fn = ir?.functions.find((candidate) => candidate.name === callee);
  return args.every((arg, index) => {
    const ref = semanticPointerArgumentMemoryRef(arg);
    if (ref?.base !== paramName) return true;
    const targetParam = fn?.params[index];
    return ref.valueType === "uchar" && (targetParam === undefined || !targetParam.pointer || targetParam.valueType === "uchar");
  });
}

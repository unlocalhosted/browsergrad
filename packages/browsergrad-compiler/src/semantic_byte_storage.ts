import { walkSemanticOperations } from "./semantic_ir.js";
import type { SemanticExpression, SemanticKernelIrModule, SemanticKernelIrOperation, SemanticMemoryRef } from "./semantic_ir_types.js";

import { semanticPointerArgumentMemoryRef } from "./semantic_pointer_arguments.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  cudaVectorSwizzleIndices,
  isCudaVectorType,
} from "./vector_types.js";
import { sizeofCudaType } from "./type_layout.js";
import { semanticExpressionValueType } from "./semantic_vector_intrinsics.js";

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

export function semanticDirectByteVectorMemberRef(
  expression: Extract<SemanticExpression, { readonly kind: "member" }>,
  ir: SemanticKernelIrModule,
): SemanticMemoryRef | undefined {
  const vectorType = semanticExpressionValueType(expression.object);
  if (!isCudaVectorType(vectorType)) return undefined;
  const lanes = cudaVectorSwizzleIndices(vectorType, expression.property);
  if (lanes?.length !== 1) return undefined;
  const ref = semanticPointerArgumentMemoryRef(expression.object);
  if (!ref || ref.addressSpace !== "storage" ||
    !ir.params.some((param) => param.name === ref.base && param.valueType === "uchar")) return undefined;
  const scalarType = cudaVectorScalarType(vectorType);
  if (scalarType === undefined) return undefined;
  const scalarBytes = sizeofCudaType(scalarType);
  if (scalarBytes !== 4) return undefined;
  const byteOffset = lanes[0]! * scalarBytes;
  const indices = byteOffset === 0 ? ref.indices : offsetLastIndex(ref.indices, byteOffset, expression.span);
  return {
    ...ref,
    valueType: scalarType,
    containerValueType: vectorType,
    packedByteLanes: scalarBytes,
    indices,
    fields: [],
    span: expression.span,
  };
}

function offsetLastIndex(
  indices: readonly SemanticExpression[],
  byteOffset: number,
  span: SemanticExpression["span"],
): readonly SemanticExpression[] {
  const offset: SemanticExpression = { kind: "literal", literalKind: "number", value: byteOffset, valueType: "uint", span };
  if (indices.length === 0) return [offset];
  const last = indices.at(-1)!;
  return [...indices.slice(0, -1), { kind: "binary", operator: "+", left: last, right: offset, valueType: "uint", span }];
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
    ref.fields.length === 1 && isCudaVectorType(ref.containerValueType) &&
      cudaVectorSwizzleIndices(ref.containerValueType, ref.fields[0]!)?.length === 1 &&
      sizeofCudaType(ref.valueType ?? "void") === 4 ||
    ref.packedByteLanes !== undefined && (
      cudaVectorLaneCount(ref.valueType) === ref.packedByteLanes ||
      sizeofCudaType(ref.valueType ?? "void") === ref.packedByteLanes
    );
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
    return semanticByteStorageRefSupported(ref) &&
      (targetParam === undefined || !targetParam.pointer || targetParam.valueType === ref.valueType);
  });
}

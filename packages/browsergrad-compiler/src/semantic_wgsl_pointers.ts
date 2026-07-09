import type {
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import type { CudaLiteScalarType } from "./types.js";
import {
  cudaVectorLaneCount,
  cudaVectorScalarType,
  isCudaVectorType,
} from "./vector_types.js";

export function semanticWgslFunctionStoragePointerParam(
  ir: SemanticKernelIrModule,
  base: string,
): SemanticKernelIrModule["functions"][number]["params"][number] | undefined {
  for (const fn of ir.functions) {
    const param = fn.params.find((item) => item.name === base && item.pointer && item.addressSpace === "storage");
    if (param) return param;
  }
  return undefined;
}

export function semanticStoragePointerBufferId(base: string, ir: SemanticKernelIrModule): number | undefined {
  const index = ir.params.findIndex((param) => param.name === base && param.addressSpace === "storage");
  return index < 0 ? undefined : index;
}

export function semanticPointerStorageCompatible(pointerType: CudaLiteScalarType, storageType: CudaLiteScalarType | undefined): boolean {
  if (storageType === undefined) return false;
  return pointerType === storageType ||
    isCudaVectorType(storageType) && cudaVectorScalarType(storageType) === pointerType ||
    isCudaVectorType(pointerType) && cudaVectorScalarType(pointerType) === storageType ||
    isCudaVectorType(pointerType) && cudaVectorScalarType(pointerType) === cudaVectorScalarType(storageType);
}

export function semanticPointerReadHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_read_${semanticPointerHelperTypeName(valueType)}`;
}

export function semanticPointerWriteHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_write_${semanticPointerHelperTypeName(valueType)}`;
}

export function semanticPointerAtomicCasHelperName(valueType: CudaLiteScalarType): string {
  return `bg_ptr_atomicCompareExchange_${semanticPointerHelperTypeName(valueType)}`;
}

export function semanticPointerBufferParamName(base: string): string {
  return `${base}_buffer`;
}

export function semanticPointerBaseParamName(base: string): string {
  return `${base}_base`;
}

export function semanticStorageOffsetSymbol(base: string): string {
  return `${base}__bg_ptr_offset`;
}

export function semanticStorageOffsetBaseNames(
  operations: readonly SemanticKernelIrOperation[],
  ir: SemanticKernelIrModule,
  pointerBaseOffsets: Readonly<Record<string, number>> | undefined,
): Set<string> {
  const out = new Set(ir.params
    .filter((param) =>
      param.addressSpace === "storage" &&
      param.pointer &&
      pointerBaseOffsets?.[param.name] !== undefined
    )
    .map((param) => param.name));
  collectSemanticStorageOffsetBaseNames(operations, out);
  return out;
}

function collectSemanticStorageOffsetBaseNames(
  operations: readonly SemanticKernelIrOperation[],
  out: Set<string>,
): void {
  for (const operation of operations) {
    if (
      operation.kind === "store" &&
      operation.target.addressSpace === "storage" &&
      operation.target.indices.length === 0 &&
      operation.target.fields.length === 0 &&
      (operation.operator === "+=" || operation.operator === "-=")
    ) out.add(operation.target.base);
    if (operation.kind === "branch") collectSemanticStorageOffsetBaseNames([...operation.consequent, ...operation.alternate], out);
    if (operation.kind === "loop") collectSemanticStorageOffsetBaseNames(operation.body, out);
  }
}

function semanticPointerHelperTypeName(valueType: CudaLiteScalarType): string {
  const scalar = semanticPointerHelperScalarName(valueType);
  return isCudaVectorType(valueType) ? `${scalar}x${cudaVectorLaneCount(valueType)}` : scalar;
}

function semanticPointerHelperScalarName(valueType: CudaLiteScalarType | undefined): string {
  if (valueType === "half" || valueType === "half2") return "f16";
  const scalarType = isCudaVectorType(valueType) ? cudaVectorScalarType(valueType) : valueType;
  if (scalarType === "int") return "i32";
  if (scalarType === "uint") return "u32";
  if (scalarType === "half") return "f16";
  if (valueType === "bool") return "u32";
  return "f32";
}

import type {
  CudaLiteSemanticFunction,
  CudaLiteSemanticSymbol,
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import type { CudaLiteScalarType } from "./types.js";
import { isSemanticFloatVectorType } from "./semantic_vector_intrinsics.js";

export type SemanticFunctionExpressionMode = "scalar" | "any";

export function semanticFunctionParamContractSupported(
  param: CudaLiteSemanticSymbol,
  valueTypeSupported: (valueType: CudaLiteScalarType | undefined) => boolean,
): boolean {
  if (param.pointer) return param.addressSpace === "storage" && valueTypeSupported(param.valueType);
  return param.addressSpace === "local" || param.addressSpace === "texture" || param.addressSpace === "surface";
}

export function semanticFunctionLocalParamValueTypesSupported(
  fn: CudaLiteSemanticFunction,
  valueTypeSupported: (valueType: CudaLiteScalarType | undefined) => boolean,
): boolean {
  return fn.params.every((param) => param.addressSpace !== "local" || valueTypeSupported(param.valueType));
}

export function semanticFunctionArgAddressContractSupported(
  arg: SemanticExpression,
  param: CudaLiteSemanticSymbol | undefined,
  pointerRef: (arg: SemanticExpression) => SemanticMemoryRef | undefined,
): boolean | undefined {
  if (!param) return false;
  if (param.pointer) {
    const ref = pointerRef(arg);
    return param.addressSpace === "storage" && ref?.addressSpace === "storage";
  }
  if (param.addressSpace === "texture") return arg.kind === "symbol" && arg.addressSpace === "texture";
  if (param.addressSpace === "surface") return arg.kind === "symbol" && arg.addressSpace === "surface";
  return undefined;
}

export function semanticFunctionArgExpressionMode(param: CudaLiteSemanticSymbol): SemanticFunctionExpressionMode {
  return isSemanticFloatVectorType(param.valueType) ? "any" : "scalar";
}

export function semanticFunctionArgSupported(
  arg: SemanticExpression,
  param: CudaLiteSemanticSymbol | undefined,
  pointerRef: (arg: SemanticExpression) => SemanticMemoryRef | undefined,
  expressionSupported: (arg: SemanticExpression, mode: SemanticFunctionExpressionMode) => boolean,
): boolean {
  if (!param) return false;
  const addressContract = semanticFunctionArgAddressContractSupported(arg, param, pointerRef);
  if (addressContract !== undefined) return addressContract;
  return expressionSupported(arg, semanticFunctionArgExpressionMode(param));
}

export interface SemanticFunctionBodyShapeOptions {
  readonly allowBlock?: boolean;
  readonly allowBarrierFence?: boolean;
}

export function semanticFunctionBodyShapeSupported(
  operations: readonly SemanticKernelIrOperation[],
  options: SemanticFunctionBodyShapeOptions = {},
): boolean {
  return operations.every((operation) => {
    if (operation.kind === "declare") return operation.target.addressSpace === "local" && !operation.target.pointer && operation.target.dimensions.length === 0;
    if (operation.kind === "store") return operation.target.addressSpace === "local" || operation.target.addressSpace === "storage";
    if (operation.kind === "surface-write" || operation.kind === "call") return true;
    if (options.allowBarrierFence && (operation.kind === "barrier" || operation.kind === "fence")) return true;
    if (operation.kind === "branch") return semanticFunctionBodyShapeSupported(operation.consequent, options) && semanticFunctionBodyShapeSupported(operation.alternate, options);
    if (options.allowBlock && operation.kind === "block") return semanticFunctionBodyShapeSupported(operation.body, options);
    if (operation.kind === "loop") return semanticFunctionBodyShapeSupported(operation.body, options);
    return operation.kind === "expression" || operation.kind === "return" || operation.kind === "break" || operation.kind === "continue";
  });
}

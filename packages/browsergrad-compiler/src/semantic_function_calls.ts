import type {
  CudaLiteSemanticFunction,
  CudaLiteSemanticSymbol,
  SemanticExpression,
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

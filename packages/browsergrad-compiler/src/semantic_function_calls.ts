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
  if (param.pointer) return (param.addressSpace === "storage" || param.addressSpace === "shared") && valueTypeSupported(param.valueType);
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
    return (param.addressSpace === "storage" || param.addressSpace === "shared") && ref?.addressSpace === param.addressSpace;
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
    if (operation.kind === "cooperative-group-declare" || operation.kind === "dim3-declare") return true;
    if (operation.kind === "declare") return operation.target.addressSpace === "local" && !operation.target.pointer && operation.target.dimensions.length === 0;
    if (operation.kind === "store") return operation.target.addressSpace === "local" || operation.target.addressSpace === "storage" || operation.target.addressSpace === "shared";
    if (operation.kind === "surface-write" || operation.kind === "call") return true;
    if (options.allowBarrierFence && (operation.kind === "barrier" || operation.kind === "fence")) return true;
    if (operation.kind === "branch") return semanticFunctionBodyShapeSupported(operation.consequent, options) && semanticFunctionBodyShapeSupported(operation.alternate, options);
    if (options.allowBlock && operation.kind === "block") return semanticFunctionBodyShapeSupported(operation.body, options);
    if (operation.kind === "loop") return semanticFunctionBodyShapeSupported(operation.body, options);
    return operation.kind === "expression" || operation.kind === "return" || operation.kind === "break" || operation.kind === "continue";
  });
}

export function semanticPointerFunctionBodySupported(
  fn: CudaLiteSemanticFunction,
  memoryRefFromIndex: (expression: SemanticExpression) => SemanticMemoryRef | undefined,
  atomicCallTarget: (expression: Extract<SemanticExpression, { readonly kind: "call" }>) => SemanticMemoryRef | undefined,
): boolean {
  const pointerParams = new Set(fn.params
    .filter((param) => param.pointer && (param.addressSpace === "storage" || param.addressSpace === "shared"))
    .map((param) => param.name));
  return pointerParams.size > 0 &&
    fn.body.every((operation) => semanticPointerFunctionOperationSupported(operation, pointerParams, memoryRefFromIndex, atomicCallTarget));
}

function semanticPointerFunctionOperationSupported(
  operation: SemanticKernelIrOperation,
  pointerParams: ReadonlySet<string>,
  memoryRefFromIndex: (expression: SemanticExpression) => SemanticMemoryRef | undefined,
  atomicCallTarget: (expression: Extract<SemanticExpression, { readonly kind: "call" }>) => SemanticMemoryRef | undefined,
): boolean {
  if (operation.kind === "atomic") return operation.target !== undefined && pointerParams.has(operation.target.base);
  if (operation.kind === "store") return pointerParams.has(operation.target.base);
  if (operation.kind === "return" && operation.value) return semanticPointerFunctionExpressionSupported(operation.value, pointerParams, memoryRefFromIndex, atomicCallTarget);
  if (operation.kind === "expression" && operation.expression.kind === "update") {
    const ref = memoryRefFromIndex(operation.expression.argument);
    return ref !== undefined && pointerParams.has(ref.base);
  }
  if (operation.kind === "expression") return semanticPointerFunctionExpressionSupported(operation.expression, pointerParams, memoryRefFromIndex, atomicCallTarget);
  return false;
}

function semanticPointerFunctionExpressionSupported(
  expression: SemanticExpression,
  pointerParams: ReadonlySet<string>,
  memoryRefFromIndex: (expression: SemanticExpression) => SemanticMemoryRef | undefined,
  atomicCallTarget: (expression: Extract<SemanticExpression, { readonly kind: "call" }>) => SemanticMemoryRef | undefined,
): boolean {
  const ref = memoryRefFromIndex(expression);
  if (ref) return pointerParams.has(ref.base);
  if (expression.kind !== "call") return false;
  const target = atomicCallTarget(expression);
  return target !== undefined && pointerParams.has(target.base);
}

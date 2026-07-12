import type {
  CudaLiteSemanticFunction,
  CudaLiteSemanticSymbol,
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import { isSemanticKernelIrOperation, walkSemanticExpression } from "./semantic_ir.js";
import type { CudaLiteScalarType } from "./types.js";
import { isSemanticFloatVectorType } from "./semantic_vector_intrinsics.js";

export type SemanticFunctionExpressionMode = "scalar" | "any";

export function semanticFunctionParamContractSupported(
  param: CudaLiteSemanticSymbol,
  valueTypeSupported: (valueType: CudaLiteScalarType | undefined) => boolean,
): boolean {
  if (param.cooperativeGroupKind !== undefined) return !param.pointer && param.addressSpace === "local";
  if (param.pointer) return (
    param.addressSpace === "storage" ||
    param.addressSpace === "shared" ||
    param.addressSpace === "local" && param.dimensions.length === 0 ||
    param.addressSpace === "constant" && param.pointerAliasOf !== undefined
  ) && valueTypeSupported(param.valueType);
  return param.addressSpace === "local" || param.addressSpace === "texture" || param.addressSpace === "surface";
}

export function semanticFunctionLocalParamValueTypesSupported(
  fn: CudaLiteSemanticFunction,
  valueTypeSupported: (valueType: CudaLiteScalarType | undefined) => boolean,
): boolean {
  return fn.params.every((param) => param.addressSpace !== "local" || param.cooperativeGroupKind !== undefined || valueTypeSupported(param.valueType));
}

export function semanticFunctionArgAddressContractSupported(
  arg: SemanticExpression,
  param: CudaLiteSemanticSymbol | undefined,
  pointerRef: (arg: SemanticExpression) => SemanticMemoryRef | undefined,
): boolean | undefined {
  if (!param) return false;
  if (param.cooperativeGroupKind !== undefined) return arg.kind === "symbol";
  if (param.pointer) {
    const ref = pointerRef(arg);
    if (param.addressSpace === "constant" && param.pointerAliasOf !== undefined) {
      return ref?.addressSpace === "constant" && ref.base === param.pointerAliasOf;
    }
    if (param.addressSpace === "storage") return ref?.addressSpace === "storage" || ref?.addressSpace === "device-global";
    return (param.addressSpace === "shared" || param.addressSpace === "local") && ref?.addressSpace === param.addressSpace;
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
  readonly allowAtomic?: boolean;
  readonly allowSharedMemory?: boolean;
  readonly allowLocalArrays?: boolean;
}

export interface SemanticPointerFunctionBodyOptions {
  readonly allowCooperativeOps?: boolean;
  readonly allowSharedMemory?: boolean;
  readonly allowDeviceGlobals?: boolean;
  readonly allowLocalArrays?: boolean;
  readonly allowConstantMemory?: boolean;
  readonly allowStoragePointerIdentity?: boolean;
  readonly storagePointerParams?: ReadonlySet<string>;
}

export function semanticFunctionBodyShapeSupported(
  operations: readonly SemanticKernelIrOperation[],
  options: SemanticFunctionBodyShapeOptions = {},
): boolean {
  return operations.every((operation) => {
    if (operation.kind === "cooperative-group-declare" || operation.kind === "dim3-declare") return true;
    if (operation.kind === "declare") {
      if (options.allowSharedMemory && operation.target.addressSpace === "shared") return !operation.target.pointer && operation.init === undefined;
      return operation.target.addressSpace === "local" && !operation.target.pointer &&
        (operation.target.dimensions.length === 0 || options.allowLocalArrays === true);
    }
    if (operation.kind === "store") return operation.target.addressSpace === "local" || operation.target.addressSpace === "storage" || operation.target.addressSpace === "shared" || operation.target.addressSpace === "device-global";
    if (operation.kind === "atomic") return options.allowAtomic === true;
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
  options: SemanticPointerFunctionBodyOptions = {},
): boolean {
  const pointerParams = new Set(fn.params
    .filter((param) => param.pointer && (param.addressSpace === "storage" || param.addressSpace === "shared" || param.addressSpace === "local"))
    .map((param) => param.name));
  const storagePointerParams = new Set(fn.params
    .filter((param) => param.pointer && param.addressSpace === "storage")
    .map((param) => param.name));
  const allowedMemoryRoots = new Set(pointerParams);
  if (options.allowSharedMemory) collectSemanticFunctionSharedRoots(fn.body, allowedMemoryRoots);
  if (options.allowLocalArrays) collectSemanticFunctionLocalArrayRoots(fn.body, allowedMemoryRoots);
  return pointerParams.size > 0 &&
    fn.body.every((operation) => semanticPointerFunctionOperationSupported(
      operation,
      allowedMemoryRoots,
      memoryRefFromIndex,
      atomicCallTarget,
      { ...options, storagePointerParams },
    ));
}

function collectSemanticFunctionLocalArrayRoots(
  operations: readonly SemanticKernelIrOperation[],
  roots: Set<string>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.target.addressSpace === "local" && operation.target.dimensions.length > 0) {
      roots.add(operation.target.name);
    }
    if (operation.kind === "branch") {
      collectSemanticFunctionLocalArrayRoots(operation.consequent, roots);
      collectSemanticFunctionLocalArrayRoots(operation.alternate, roots);
    }
    if (operation.kind === "loop" || operation.kind === "block") collectSemanticFunctionLocalArrayRoots(operation.body, roots);
  }
}

function collectSemanticFunctionSharedRoots(
  operations: readonly SemanticKernelIrOperation[],
  roots: Set<string>,
): void {
  for (const operation of operations) {
    if (operation.kind === "declare" && operation.target.addressSpace === "shared") roots.add(operation.target.name);
    if (operation.kind === "branch") {
      collectSemanticFunctionSharedRoots(operation.consequent, roots);
      collectSemanticFunctionSharedRoots(operation.alternate, roots);
    }
    if (operation.kind === "loop" || operation.kind === "block") collectSemanticFunctionSharedRoots(operation.body, roots);
  }
}

function semanticPointerFunctionOperationSupported(
  operation: SemanticKernelIrOperation,
  pointerParams: ReadonlySet<string>,
  memoryRefFromIndex: (expression: SemanticExpression) => SemanticMemoryRef | undefined,
  atomicCallTarget: (expression: Extract<SemanticExpression, { readonly kind: "call" }>) => SemanticMemoryRef | undefined,
  options: SemanticPointerFunctionBodyOptions,
): boolean {
  if (options.allowCooperativeOps && operation.kind === "cooperative-group-declare") return true;
  if (options.allowCooperativeOps && (operation.kind === "barrier" || operation.kind === "fence")) return true;
  if (options.allowCooperativeOps && operation.kind === "declare") {
    if (options.allowSharedMemory && operation.target.addressSpace === "shared") {
      return !operation.target.pointer && operation.init === undefined;
    }
    return operation.target.addressSpace === "local" &&
      !operation.target.pointer &&
      (operation.target.dimensions.length === 0 || options.allowLocalArrays === true) &&
      (operation.init === undefined || semanticPointerFunctionExpressionAccessesSupported(operation.init, pointerParams, memoryRefFromIndex, atomicCallTarget, options));
  }
  if (operation.kind === "atomic") return operation.target !== undefined && (pointerParams.has(operation.target.base) || options.allowDeviceGlobals === true && operation.target.addressSpace === "device-global");
  if (operation.kind === "store") return (operation.target.addressSpace === "local" || pointerParams.has(operation.target.base) || options.allowDeviceGlobals === true && operation.target.addressSpace === "device-global") &&
    (!options.allowCooperativeOps || semanticPointerFunctionExpressionAccessesSupported(operation.value, pointerParams, memoryRefFromIndex, atomicCallTarget, options));
  if (operation.kind === "call") {
    return operation.args.every((arg) =>
      semanticPointerFunctionExpressionAccessesSupported(arg, pointerParams, memoryRefFromIndex, atomicCallTarget, options),
    );
  }
  if (options.allowCooperativeOps && operation.kind === "branch") {
    return semanticPointerFunctionExpressionAccessesSupported(operation.condition, pointerParams, memoryRefFromIndex, atomicCallTarget, options) &&
      operation.consequent.every((item) => semanticPointerFunctionOperationSupported(item, pointerParams, memoryRefFromIndex, atomicCallTarget, options)) &&
      operation.alternate.every((item) => semanticPointerFunctionOperationSupported(item, pointerParams, memoryRefFromIndex, atomicCallTarget, options));
  }
  if (options.allowCooperativeOps && operation.kind === "block") {
    return operation.body.every((item) => semanticPointerFunctionOperationSupported(item, pointerParams, memoryRefFromIndex, atomicCallTarget, options));
  }
  if (options.allowCooperativeOps && operation.kind === "loop") {
    return (operation.init === undefined || semanticPointerFunctionLoopInitSupported(operation.init, pointerParams, memoryRefFromIndex, atomicCallTarget, options)) &&
      (operation.condition === undefined || semanticPointerFunctionExpressionAccessesSupported(operation.condition, pointerParams, memoryRefFromIndex, atomicCallTarget, options)) &&
      (operation.update === undefined || semanticPointerFunctionExpressionAccessesSupported(operation.update, pointerParams, memoryRefFromIndex, atomicCallTarget, options)) &&
      operation.body.every((item) => semanticPointerFunctionOperationSupported(item, pointerParams, memoryRefFromIndex, atomicCallTarget, options));
  }
  if (operation.kind === "return" && operation.value) {
    return semanticPointerFunctionExpressionAccessesSupported(operation.value, pointerParams, memoryRefFromIndex, atomicCallTarget, options);
  }
  if (operation.kind === "return" || operation.kind === "break" || operation.kind === "continue") return true;
  if (operation.kind === "expression" && operation.expression.kind === "update") {
    const ref = memoryRefFromIndex(operation.expression.argument);
    return ref !== undefined && (pointerParams.has(ref.base) || options.allowDeviceGlobals === true && ref.addressSpace === "device-global");
  }
  if (operation.kind === "expression" && operation.expression.kind === "literal") return true;
  if (options.allowCooperativeOps && operation.kind === "expression") {
    return semanticPointerFunctionExpressionAccessesSupported(operation.expression, pointerParams, memoryRefFromIndex, atomicCallTarget, options);
  }
  if (operation.kind === "expression") return semanticPointerFunctionExpressionAccessesSupported(operation.expression, pointerParams, memoryRefFromIndex, atomicCallTarget, options);
  return false;
}

function semanticPointerFunctionLoopInitSupported(
  init: SemanticKernelIrOperation | SemanticExpression,
  pointerParams: ReadonlySet<string>,
  memoryRefFromIndex: (expression: SemanticExpression) => SemanticMemoryRef | undefined,
  atomicCallTarget: (expression: Extract<SemanticExpression, { readonly kind: "call" }>) => SemanticMemoryRef | undefined,
  options: SemanticPointerFunctionBodyOptions,
): boolean {
  return isSemanticKernelIrOperation(init)
    ? semanticPointerFunctionOperationSupported(init, pointerParams, memoryRefFromIndex, atomicCallTarget, options)
    : semanticPointerFunctionExpressionAccessesSupported(init, pointerParams, memoryRefFromIndex, atomicCallTarget, options);
}

function semanticPointerFunctionExpressionAccessesSupported(
  expression: SemanticExpression,
  pointerParams: ReadonlySet<string>,
  memoryRefFromIndex: (expression: SemanticExpression) => SemanticMemoryRef | undefined,
  atomicCallTarget: (expression: Extract<SemanticExpression, { readonly kind: "call" }>) => SemanticMemoryRef | undefined,
  options: SemanticPointerFunctionBodyOptions,
): boolean {
  if (semanticPointerFunctionExpressionHasUnsupportedPointerIdentityCheck(
    expression,
    pointerParams,
    options.allowStoragePointerIdentity === true ? options.storagePointerParams ?? new Set() : new Set(),
  )) return false;
  let supported = true;
  walkSemanticExpression(expression, (item) => {
    const ref = memoryRefFromIndex(item);
    if (ref &&
      !pointerParams.has(ref.base) &&
      !(options.allowDeviceGlobals === true && ref.addressSpace === "device-global") &&
      !(options.allowConstantMemory === true && ref.addressSpace === "constant")) supported = false;
    if (item.kind !== "call") return;
    const target = atomicCallTarget(item);
    if (target && !pointerParams.has(target.base) && !(options.allowDeviceGlobals === true && target.addressSpace === "device-global")) supported = false;
  });
  return supported;
}

function semanticPointerFunctionExpressionHasUnsupportedPointerIdentityCheck(
  expression: SemanticExpression,
  pointerParams: ReadonlySet<string>,
  allowedStoragePointerParams: ReadonlySet<string>,
): boolean {
  let found = false;
  walkSemanticExpression(expression, (item) => {
    if (item.kind !== "binary" || (item.operator !== "==" && item.operator !== "!=")) return;
    const usesPointer = semanticPointerFunctionIdentityOperand(item.left, pointerParams) ||
      semanticPointerFunctionIdentityOperand(item.right, pointerParams);
    const supportedStorageIdentity = semanticPointerFunctionIdentityOperand(item.left, allowedStoragePointerParams) &&
      semanticPointerFunctionIdentityOperand(item.right, allowedStoragePointerParams);
    if (usesPointer && !supportedStorageIdentity) found = true;
  });
  return found;
}

function semanticPointerFunctionIdentityOperand(
  expression: SemanticExpression,
  pointerParams: ReadonlySet<string>,
): boolean {
  return expression.kind === "symbol" && pointerParams.has(expression.name);
}

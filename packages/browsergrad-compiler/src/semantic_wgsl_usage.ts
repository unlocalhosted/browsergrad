import {
  SEMANTIC_CURAND_CALLS,
} from "./semantic_curand_intrinsics.js";
import {
  SEMANTIC_BFLOAT_HELPER_CALLS,
  SEMANTIC_FP8_CALLS,
  SEMANTIC_HALF_CONVERSION_CALLS,
} from "./semantic_math_intrinsics.js";
import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticKernelIrOperation,
} from "./semantic_ir.js";
import {
  semanticExpressionChildren,
  semanticOperationExpressions,
} from "./semantic_ir_walk.js";
import { semanticExpressionValueType } from "./semantic_vector_intrinsics.js";

export function semanticSharedMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "shared");
}

export function semanticConstantMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "constant");
}

export function semanticDeviceGlobalMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "device-global");
}

export function semanticTextureSymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  const byName = new Map<string, SemanticKernelIrModule["memory"][number]>();
  for (const param of ir.params.filter((symbol) => symbol.addressSpace === "texture")) byName.set(param.name, param);
  for (const symbol of ir.memory.filter((item) => item.kind === "texture")) byName.set(symbol.name, symbol);
  return [...byName.values()];
}

export function semanticSurfaceSymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["params"][number][] {
  return ir.params.filter((symbol) => symbol.addressSpace === "surface");
}

export function semanticLocalMemorySymbols(ir: SemanticKernelIrModule): readonly SemanticKernelIrModule["memory"][number][] {
  return ir.memory.filter((symbol) => symbol.kind === "local" && symbol.dimensions.length > 0);
}

export function semanticUsesGenericSurfaceRead(ir: SemanticKernelIrModule): boolean {
  return ir.functions.some((fn) => {
    const surfaceParams = new Set(fn.params.filter((param) => param.addressSpace === "surface").map((param) => param.name));
    return surfaceParams.size > 0 && semanticOperationsUseSurfaceParamRead(fn.body, surfaceParams);
  });
}

export function semanticUsesGenericSurfaceWrite(ir: SemanticKernelIrModule): boolean {
  return ir.functions.some((fn) => {
    const surfaceParams = new Set(fn.params.filter((param) => param.addressSpace === "surface").map((param) => param.name));
    return surfaceParams.size > 0 && semanticOperationsUseSurfaceParamWrite(fn.body, surfaceParams);
  });
}

export function semanticUsesFp8(ir: SemanticKernelIrModule): boolean {
  return semanticOperationsUseFp8(ir.operations) || ir.functions.some((fn) => semanticOperationsUseFp8(fn.body));
}

export function semanticUsesHalfConversion(ir: SemanticKernelIrModule): boolean {
  return semanticOperationsUseHalfConversion(ir.operations) || ir.functions.some((fn) => semanticOperationsUseHalfConversion(fn.body));
}

export function semanticUsesBfloatHelper(ir: SemanticKernelIrModule): boolean {
  return ir.params.some((param) => param.valueType === "bf16" || param.valueType === "bf162") ||
    ir.memory.some((memory) => memory.valueType === "bf16" || memory.valueType === "bf162") ||
    semanticOperationsUseBfloatHelper(ir.operations) ||
    ir.functions.some((fn) =>
      fn.params.some((param) => param.valueType === "bf16" || param.valueType === "bf162") ||
      semanticOperationsUseBfloatHelper(fn.body));
}

export function semanticUsesCurand(ir: SemanticKernelIrModule): boolean {
  return semanticOperationsUseCurand(ir.operations) || ir.functions.some((fn) => semanticOperationsUseCurand(fn.body));
}

export function semanticUsesCuComplexRobustMath(ir: SemanticKernelIrModule): boolean {
  const calls = new Set(["cuCabsf", "cuCdivf", "cuCabs", "cuCdiv"]);
  const uses = (operations: readonly SemanticKernelIrOperation[]): boolean => operations.some((operation) =>
    semanticOperationExpressions(operation).some((expression) => semanticExpressionUsesCall(expression, calls)) ||
    operation.kind === "branch" && (uses(operation.consequent) || uses(operation.alternate)) ||
    operation.kind === "loop" && (uses(operation.body) || uses(operation.continuing ?? [])) ||
    operation.kind === "block" && uses(operation.body)
  );
  return uses(ir.operations) || ir.functions.some((fn) => uses(fn.body));
}

function semanticExpressionUsesCall(expression: SemanticExpression, calls: ReadonlySet<string>): boolean {
  return expression.kind === "call" && expression.callee.kind === "symbol" && calls.has(expression.callee.name) ||
    semanticExpressionChildren(expression).some((child) => semanticExpressionUsesCall(child, calls));
}

function semanticOperationsUseCurand(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const operation of operations) {
    if (operation.kind === "call" && SEMANTIC_CURAND_CALLS.has(operation.callee)) return true;
    if (semanticOperationExpressions(operation).some(semanticExpressionUsesCurand)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseCurand(operation.consequent) || semanticOperationsUseCurand(operation.alternate))) return true;
    if (operation.kind === "loop" && semanticOperationsUseCurand(operation.body)) return true;
    if (operation.kind === "block" && semanticOperationsUseCurand(operation.body)) return true;
  }
  return false;
}

function semanticExpressionUsesCurand(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && SEMANTIC_CURAND_CALLS.has(expression.callee.name)) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesCurand);
}

function semanticOperationsUseFp8(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const operation of operations) {
    if (semanticOperationExpressions(operation).some(semanticExpressionUsesFp8)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseFp8(operation.consequent) || semanticOperationsUseFp8(operation.alternate))) return true;
    if (operation.kind === "loop" && semanticOperationsUseFp8(operation.body)) return true;
    if (operation.kind === "block" && semanticOperationsUseFp8(operation.body)) return true;
  }
  return false;
}

function semanticExpressionUsesFp8(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && SEMANTIC_FP8_CALLS.has(expression.callee.name)) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesFp8);
}

function semanticOperationsUseHalfConversion(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const operation of operations) {
    if (semanticOperationExpressions(operation).some(semanticExpressionUsesHalfConversion)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseHalfConversion(operation.consequent) || semanticOperationsUseHalfConversion(operation.alternate))) return true;
    if (operation.kind === "loop" && semanticOperationsUseHalfConversion(operation.body)) return true;
    if (operation.kind === "block" && semanticOperationsUseHalfConversion(operation.body)) return true;
  }
  return false;
}

function semanticExpressionUsesHalfConversion(expression: SemanticExpression): boolean {
  if (expression.kind === "call" && expression.callee.kind === "symbol" && SEMANTIC_HALF_CONVERSION_CALLS.has(expression.callee.name)) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesHalfConversion);
}

function semanticOperationsUseBfloatHelper(operations: readonly SemanticKernelIrOperation[]): boolean {
  for (const operation of operations) {
    if (operation.kind === "declare" && (operation.target.valueType === "bf16" || operation.target.valueType === "bf162")) return true;
    if (semanticOperationExpressions(operation).some(semanticExpressionUsesBfloatHelper)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseBfloatHelper(operation.consequent) || semanticOperationsUseBfloatHelper(operation.alternate))) return true;
    if (operation.kind === "loop" && semanticOperationsUseBfloatHelper(operation.body)) return true;
    if (operation.kind === "block" && semanticOperationsUseBfloatHelper(operation.body)) return true;
  }
  return false;
}

function semanticExpressionUsesBfloatHelper(expression: SemanticExpression): boolean {
  const valueType = semanticExpressionValueType(expression);
  if (valueType === "bf16" || valueType === "bf162") return true;
  if (expression.kind === "call" && expression.callee.kind === "symbol" && SEMANTIC_BFLOAT_HELPER_CALLS.has(expression.callee.name)) return true;
  return semanticExpressionChildren(expression).some(semanticExpressionUsesBfloatHelper);
}

function semanticOperationsUseSurfaceParamWrite(
  operations: readonly SemanticKernelIrOperation[],
  surfaceParams: ReadonlySet<string>,
): boolean {
  for (const operation of operations) {
    if (operation.kind === "surface-write" && operation.surface.kind === "symbol" && surfaceParams.has(operation.surface.name)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseSurfaceParamWrite(operation.consequent, surfaceParams) || semanticOperationsUseSurfaceParamWrite(operation.alternate, surfaceParams))) return true;
    if (operation.kind === "loop" && semanticOperationsUseSurfaceParamWrite(operation.body, surfaceParams)) return true;
  }
  return false;
}

function semanticOperationsUseSurfaceParamRead(
  operations: readonly SemanticKernelIrOperation[],
  surfaceParams: ReadonlySet<string>,
): boolean {
  for (const operation of operations) {
    if (operation.kind === "return" && operation.value && semanticExpressionUsesSurfaceParamRead(operation.value, surfaceParams)) return true;
    if (operation.kind === "expression" && semanticExpressionUsesSurfaceParamRead(operation.expression, surfaceParams)) return true;
    if (operation.kind === "declare" && operation.init && semanticExpressionUsesSurfaceParamRead(operation.init, surfaceParams)) return true;
    if (operation.kind === "store" && semanticExpressionUsesSurfaceParamRead(operation.value, surfaceParams)) return true;
    if (operation.kind === "branch" && (semanticOperationsUseSurfaceParamRead(operation.consequent, surfaceParams) || semanticOperationsUseSurfaceParamRead(operation.alternate, surfaceParams))) return true;
    if (operation.kind === "loop" && semanticOperationsUseSurfaceParamRead(operation.body, surfaceParams)) return true;
  }
  return false;
}

function semanticExpressionUsesSurfaceParamRead(
  expression: SemanticExpression,
  surfaceParams: ReadonlySet<string>,
): boolean {
  if (expression.kind === "surface-read") return expression.surface.kind === "symbol" && surfaceParams.has(expression.surface.name);
  return semanticExpressionChildren(expression).some((child) => semanticExpressionUsesSurfaceParamRead(child, surfaceParams));
}

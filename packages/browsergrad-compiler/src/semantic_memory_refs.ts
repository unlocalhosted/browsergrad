import { semanticMemoryIdFromSymbol } from "./semantic_ids.js";
import type { SemanticExpression, SemanticKernelIrModule, SemanticMemoryRef } from "./semantic_ir_types.js";
import { requireSemanticValueType } from "./semantic_value_type.js";
import { semanticExpressionValueType } from "./semantic_vector_intrinsics.js";
import type { CudaLiteScalarType } from "./types.js";

export function semanticMemoryRefFromExpression(expression: SemanticExpression): SemanticMemoryRef | undefined {
  if (expression.kind === "symbol" && expression.addressSpace === "device-global") {
    return {
      baseId: semanticMemoryIdFromSymbol(expression.id),
      base: expression.name,
      addressSpace: expression.addressSpace,
      valueType: requireSemanticValueType(expression.valueType, `device global '${expression.name}'`, expression.span),
      indices: [],
      fields: [],
      span: expression.span,
    };
  }
  if (expression.kind === "member") {
    const ref = semanticMemoryRefFromExpression(expression.object);
    if (!ref) return undefined;
    return {
      ...ref,
      valueType: requireSemanticValueType(expression.valueType, "vector member assignment", expression.span),
      containerValueType: requireSemanticValueType(semanticExpressionValueType(expression.object), "vector member container", expression.span),
      fields: [...ref.fields, expression.property],
      span: expression.span,
    };
  }
  if (expression.kind !== "index") return undefined;
  const flattened = flattenSemanticMemoryRef(expression);
  if (!flattened || !isMemoryAddressSpace(flattened.base.addressSpace)) return undefined;
  return {
    baseId: semanticMemoryIdFromSymbol(flattened.base.id),
    base: flattened.base.name,
    addressSpace: flattened.base.addressSpace,
    valueType: expression.valueType,
    ...(expression.target.kind === "symbol" && expression.target.valueType !== undefined ? { containerValueType: expression.target.valueType } : {}),
    ...(expression.pointerBaseIsScalarLane === true ? { pointerBaseIsScalarLane: true } : {}),
    ...(expression.pointerBaseUnitBytes === undefined ? {} : { pointerBaseUnitBytes: expression.pointerBaseUnitBytes }),
    ...(expression.packedByteLanes === undefined ? {} : { packedByteLanes: expression.packedByteLanes }),
    indices: flattened.indices,
    fields: [],
    span: expression.span,
  };
}

export function semanticMemoryRefStorageValueType(
  ref: SemanticMemoryRef,
  ir: SemanticKernelIrModule,
): CudaLiteScalarType | undefined {
  if (ref.addressSpace === "storage") {
    return ir.params.find((param) => param.name === ref.base && param.addressSpace === "storage")?.valueType;
  }
  if (ref.addressSpace === "shared" || ref.addressSpace === "device-global") {
    return ir.memory.find((symbol) => symbol.name === ref.base && symbol.kind === ref.addressSpace)?.valueType;
  }
  return ref.valueType;
}

function flattenSemanticMemoryRef(expression: SemanticExpression): {
  readonly base: Extract<SemanticExpression, { readonly kind: "symbol" }>;
  readonly indices: readonly SemanticExpression[];
} | undefined {
  if (expression.kind === "symbol") return { base: expression, indices: [] };
  if (expression.kind === "cast" && expression.pointer) return flattenSemanticMemoryRef(expression.expression);
  if (expression.kind !== "index") return undefined;
  const target = flattenSemanticMemoryRef(expression.target);
  return target === undefined ? undefined : { base: target.base, indices: [...target.indices, expression.index] };
}

function isMemoryAddressSpace(value: string | undefined): value is "storage" | "shared" | "constant" | "device-global" | "local" {
  return value === "storage" || value === "shared" || value === "constant" || value === "device-global" || value === "local";
}

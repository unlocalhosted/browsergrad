import type {
  SemanticExpression,
  SemanticKernelIrModule,
  SemanticMemoryRef,
} from "./semantic_ir.js";
import {
  createTypedWgslCall,
  createTypedWgslIdentifier,
  createTypedWgslLiteral,
  createTypedWgslZero,
  convertTypedWgslExpression,
  emitTypedWgslBinary,
  type TypedWgslExpression,
} from "./typed_wgsl_expression.js";
import {
  emitSemanticFlatLocalArrayIndexes,
  emitSemanticFlatRankedIndex,
} from "./semantic_wgsl_memory_layout.js";
import {
  semanticPointerBaseParamName,
  semanticPointerBufferParamName,
  semanticStorageOffsetBaseNames,
  semanticStorageOffsetSymbol,
  semanticWgslFunctionStoragePointerParam,
} from "./semantic_wgsl_pointers.js";
import {
  semanticPointerDeclarationNeedsRuntimeState,
  semanticRuntimePointerDeclarations,
} from "./semantic_runtime_pointers.js";
import { semanticStorageVectorType } from "./semantic_value_types.js";
import { sizeofCudaType } from "./type_layout.js";
import type { SourceSpan } from "./types.js";
import { cudaVectorLaneCount } from "./vector_types.js";

export interface SemanticWgslIndexOptions {
  readonly activeFunction?: string;
  readonly pointerBaseOffsets?: Readonly<Record<string, number>>;
}

export interface SemanticWgslIndexingHost {
  readonly emitExpressionAs: (
    expression: SemanticExpression,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    expectedType: "i32" | "u32",
    options?: SemanticWgslIndexOptions,
  ) => TypedWgslExpression;
  readonly nameFor: (name: string, names: ReadonlyMap<string, string>) => string;
  readonly indexingFailure: (message: string, span: SourceSpan) => Error;
}

export function createSemanticWgslIndexing(host: SemanticWgslIndexingHost) {
  const { emitExpressionAs, nameFor, indexingFailure } = host;

  function emitTypedFlatRankedIndex(
    dimensions: readonly number[],
    indices: readonly SemanticExpression[],
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: SemanticWgslIndexOptions,
    span: SourceSpan,
  ): TypedWgslExpression {
    if (indices.length === 1) return emitExpressionAs(indices[0]!, ir, names, "u32", options);
    if (indices.length !== dimensions.length) throw indexingFailure("typed WGSL array index rank mismatch", span);
    return indices.map((index, offset) => {
      const value = emitExpressionAs(index, ir, names, "u32", options);
      const stride = dimensions.slice(offset + 1).reduce((size, dimension) => size * dimension, 1);
      return stride === 1 ? value : emitTypedWgslBinary("*", value, createTypedWgslLiteral(`${stride}u`, "u32", span), span);
    }).reduce((left, right) => emitTypedWgslBinary("+", left, right, span));
  }

  function semanticTypedLocalArrayPath(
    flat: TypedWgslExpression,
    dimensions: readonly number[],
    span: SourceSpan,
  ): readonly TypedWgslExpression[] {
    return dimensions.map((dimension, offset) => {
      const stride = dimensions.slice(offset + 1).reduce((size, value) => size * value, 1);
      const quotient = stride === 1
        ? flat
        : emitTypedWgslBinary("/", flat, createTypedWgslLiteral(`${stride}u`, "u32", span), span);
      return dimension > 1
        ? emitTypedWgslBinary("%", quotient, createTypedWgslLiteral(`${Math.max(1, dimension)}u`, "u32", span), span)
        : createTypedWgslZero("u32", span);
    });
  }

  function emitTypedFlatStorageIndex(
    ref: SemanticMemoryRef,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: SemanticWgslIndexOptions,
  ): TypedWgslExpression {
    const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
    const localRuntimePointer = ref.addressSpace === "local"
      ? semanticRuntimePointerDeclarations(ir).find((operation) =>
          operation.target.name === ref.base && semanticPointerDeclarationNeedsRuntimeState(operation))
      : undefined;
    const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets).has(ref.base);
    if (!pointerParam && !localRuntimePointer && !hasOffset && ref.indices.length === 0) return createTypedWgslZero("u32", ref.span);
    if (!pointerParam && !localRuntimePointer && !hasOffset && ref.indices.length === 1) {
      return emitExpressionAs(ref.indices[0]!, ir, names, "u32", options);
    }
    const terms = ref.indices.map((index) => emitExpressionAs(index, ir, names, "i32", options));
    if (pointerParam || localRuntimePointer) {
      if (pointerParam && terms.length > 0 && ref.pointerBaseUnitBytes === undefined) {
        const elementBytes = sizeofCudaType(pointerParam.valueType ?? "void") ?? 1;
        const byteBufferIds = semanticStoragePointerByteBufferIds(ir);
        if (elementBytes > 1 && byteBufferIds.length > 0) {
          const buffer = createTypedWgslIdentifier(nameFor(semanticPointerBufferParamName(ref.base), names), "u32", ref.span);
          const byteBacked = byteBufferIds
            .map((id) => emitTypedWgslBinary("==", buffer, createTypedWgslLiteral(`${id}u`, "u32", ref.span), ref.span))
            .reduce((left, right) => emitTypedWgslBinary("||", left, right, ref.span));
          const offset = terms.reduce((left, right) => emitTypedWgslBinary("+", left, right, ref.span));
          const scaled = emitTypedWgslBinary("*", offset, createTypedWgslLiteral(`${elementBytes}`, "i32", ref.span), ref.span);
          terms.splice(0, terms.length, createTypedWgslCall("select", [offset, scaled, byteBacked], "i32", ref.span));
        }
      }
      terms.unshift(convertTypedWgslExpression(
        createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span),
        "i32",
        true,
      ));
    } else if (hasOffset) {
      terms.unshift(createTypedWgslIdentifier(nameFor(semanticStorageOffsetSymbol(ref.base), names), "i32", ref.span));
    }
    const index = terms.reduce((left, right) => emitTypedWgslBinary("+", left, right, ref.span));
    return convertTypedWgslExpression(index, "u32", true);
  }

  function emitTypedFlatStorageVectorBaseIndex(
    ref: SemanticMemoryRef,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: SemanticWgslIndexOptions,
  ): TypedWgslExpression {
    const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
    const root = ir.params.find((param) => param.name === ref.base) ?? ir.memory.find((symbol) => symbol.name === ref.base);
    const rootVector = semanticStorageVectorType(ref.containerValueType) ?? semanticStorageVectorType(pointerParam?.valueType) ?? semanticStorageVectorType(root?.valueType);
    const stride = rootVector === undefined ? 1 : cudaVectorLaneCount(rootVector);
    if (pointerParam) {
      const base = createTypedWgslIdentifier(nameFor(semanticPointerBaseParamName(ref.base), names), "u32", ref.span);
      const indices = ref.indices.map((index) => emitExpressionAs(index, ir, names, "u32", options));
      const offset = indices.length === 0
        ? createTypedWgslZero("u32", ref.span)
        : indices.reduce((left, right) => emitTypedWgslBinary("+", left, right, ref.span));
      const scaled = stride === 1 ? offset : emitTypedWgslBinary("*", offset, createTypedWgslLiteral(`${stride}u`, "u32", ref.span), ref.span);
      return emitTypedWgslBinary("+", base, scaled, ref.span);
    }
    const base = emitTypedFlatStorageIndex(ref, ir, names, options);
    return stride === 1 ? base : emitTypedWgslBinary("*", base, createTypedWgslLiteral(`${stride}u`, "u32", ref.span), ref.span);
  }

  function emitFlatStorageIndex(
    ref: SemanticMemoryRef,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: SemanticWgslIndexOptions = {},
  ): string {
    const pointer = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
    if (pointer) {
      const terms = ref.indices.map((index) => emitExpressionAs(index, ir, names, "i32", options).code);
      if (terms.length > 0 && ref.pointerBaseUnitBytes === undefined) {
        const elementBytes = sizeofCudaType(pointer.valueType ?? "void") ?? 1;
        const byteBufferIds = semanticStoragePointerByteBufferIds(ir);
        if (elementBytes > 1 && byteBufferIds.length > 0) {
          const offset = terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
          const buffer = nameFor(semanticPointerBufferParamName(ref.base), names);
          const byteBacked = byteBufferIds.map((id) => `(${buffer} == ${id}u)`).join(" || ");
          terms.splice(0, terms.length, `select(${offset}, (${offset} * ${elementBytes}), ${byteBacked})`);
        }
      }
      terms.unshift(`i32(${nameFor(semanticPointerBaseParamName(ref.base), names)})`);
      return `u32(${terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`})`;
    }
    const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets).has(ref.base);
    if (!hasOffset && ref.indices.length === 0) return "0u";
    if (!hasOffset && ref.indices.length === 1) return emitExpressionAs(ref.indices[0]!, ir, names, "u32", options).code;
    const terms = ref.indices.map((index) => emitExpressionAs(index, ir, names, "i32", options).code);
    if (hasOffset) terms.unshift(nameFor(semanticStorageOffsetSymbol(ref.base), names));
    const expression = terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
    return `u32(${expression})`;
  }

  function emitSemanticRootStorageIndex(
    ref: SemanticMemoryRef,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: SemanticWgslIndexOptions = {},
  ): string {
    const hasOffset = semanticStorageOffsetBaseNames(ir.operations, ir, options.pointerBaseOffsets).has(ref.base);
    if (!hasOffset && ref.indices.length === 0) return "0u";
    if (!hasOffset && ref.indices.length === 1) return emitExpressionAs(ref.indices[0]!, ir, names, "u32", options).code;
    const terms = ref.indices.map((index) => emitExpressionAs(index, ir, names, "i32", options).code);
    if (hasOffset) terms.unshift(nameFor(semanticStorageOffsetSymbol(ref.base), names));
    const expression = terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
    return `u32(${expression})`;
  }

  function emitFlatStorageVectorBaseIndex(
    ref: SemanticMemoryRef,
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    options: SemanticWgslIndexOptions = {},
  ): string {
    const pointerParam = semanticWgslFunctionStoragePointerParam(ir, ref.base, options.activeFunction ?? null);
    if (pointerParam) {
      const indexTerms = ref.indices.map((index) => emitExpressionAs(index, ir, names, "u32", options).code);
      const valueType = semanticStorageVectorType(ref.containerValueType) ?? semanticStorageVectorType(pointerParam.valueType);
      const stride = valueType === undefined ? 1 : cudaVectorLaneCount(valueType);
      const index = indexTerms.length === 0 ? "0u" : indexTerms.length === 1 ? indexTerms[0]! : `(${indexTerms.join(" + ")})`;
      const offset = stride === 1 ? index : `(${index} * ${stride}u)`;
      return `(${nameFor(semanticPointerBaseParamName(ref.base), names)} + ${offset})`;
    }
    const base = emitFlatStorageIndex({ ...ref, valueType: "float" }, ir, names, options);
    const root = ir.params.find((param) => param.name === ref.base) ?? ir.memory.find((symbol) => symbol.name === ref.base);
    const valueType = semanticStorageVectorType(ref.containerValueType) ?? semanticStorageVectorType(root?.valueType);
    const stride = valueType === undefined ? 1 : cudaVectorLaneCount(valueType);
    return stride === 1 ? base : `(${base} * ${stride}u)`;
  }

  function emitFlatSharedIndex(
    symbol: SemanticKernelIrModule["memory"][number],
    indices: readonly SemanticExpression[],
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
  ): string {
    if (indices.length === 0) return "0u";
    if (indices.length === 1) return emitExpressionAs(indices[0]!, ir, names, "u32").code;
    return emitSemanticFlatRankedIndex(
      "shared memory",
      symbol.name,
      symbol.dimensions,
      indices,
      symbol.span,
      (index) => emitExpressionAs(index, ir, names, "u32").code,
    );
  }

  function emitFlatDeviceGlobalIndex(
    symbol: SemanticKernelIrModule["memory"][number],
    indices: readonly SemanticExpression[],
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    span: SourceSpan,
  ): string {
    if (symbol.dimensions.length === 0) {
      if (indices.length > 1) throw indexingFailure(`device-global memory '${symbol.name}' index rank mismatch`, span);
      return indices[0] ? emitExpressionAs(indices[0], ir, names, "u32").code : "0u";
    }
    if (indices.length === 1 && symbol.dimensions.length > 1) return emitExpressionAs(indices[0]!, ir, names, "u32").code;
    if (indices.length !== symbol.dimensions.length) throw indexingFailure(`device-global memory '${symbol.name}' index rank mismatch`, span);
    return emitSemanticFlatRankedIndex(
      "device-global memory",
      symbol.name,
      symbol.dimensions,
      indices,
      span,
      (index) => emitExpressionAs(index, ir, names, "u32").code,
    );
  }

  function emitFlatConstantIndex(
    symbol: SemanticKernelIrModule["memory"][number],
    indices: readonly SemanticExpression[],
    ir: SemanticKernelIrModule,
    names: ReadonlyMap<string, string>,
    span: SourceSpan,
    options: SemanticWgslIndexOptions = {},
  ): string {
    if (symbol.dimensions.length === 0) {
      if (indices.length !== 1) throw indexingFailure(`constant memory '${symbol.name}' index rank mismatch`, span);
      return emitExpressionAs(indices[0]!, ir, names, "u32", options).code;
    }
    if (indices.length === 1 && symbol.dimensions.length > 1) return emitExpressionAs(indices[0]!, ir, names, "u32", options).code;
    if (indices.length !== symbol.dimensions.length) throw indexingFailure(`constant memory '${symbol.name}' index rank mismatch`, span);
    return emitSemanticFlatRankedIndex(
      "constant memory",
      symbol.name,
      symbol.dimensions,
      indices,
      span,
      (index) => emitExpressionAs(index, ir, names, "u32", options).code,
    );
  }

  function emitFlatLocalArrayIndexes(flat: string, dimensions: readonly number[]): string {
    return emitSemanticFlatLocalArrayIndexes(flat, dimensions);
  }

  function semanticStoragePointerByteBufferIds(ir: SemanticKernelIrModule): readonly number[] {
    const globals = ir.memory.filter((symbol) => symbol.kind === "device-global");
    return [
      ...ir.params.flatMap((param, index) =>
        param.addressSpace === "storage" && param.valueType === "uchar" ? [index] : []
      ),
      ...globals.flatMap((symbol, index) =>
        symbol.valueType === "uchar" ? [ir.params.length + index] : []
      ),
    ];
  }

  return {
    emitFlatConstantIndex,
    emitFlatDeviceGlobalIndex,
    emitFlatLocalArrayIndexes,
    emitFlatSharedIndex,
    emitFlatStorageIndex,
    emitFlatStorageVectorBaseIndex,
    emitSemanticRootStorageIndex,
    emitTypedFlatRankedIndex,
    emitTypedFlatStorageIndex,
    emitTypedFlatStorageVectorBaseIndex,
    semanticTypedLocalArrayPath,
  };
}

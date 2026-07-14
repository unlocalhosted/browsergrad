import type {
  SemanticExpression,
  SemanticKernelIrOperation,
  SemanticMatrixTileRef,
  SemanticMemoryRef,
} from "./semantic_ir_types.js";
import { createGeneratedSemanticSymbolId } from "./semantic_ids.js";
import type { SourceSpan } from "./types.js";
import {
  isMatrixTileByteValueType,
  matrixTileElementCount,
  type MatrixTileLayout,
  type MatrixTileResolvedSpec,
} from "./matrix_tiles.js";

type SemanticMatrixOperation = Extract<
  SemanticKernelIrOperation,
  { readonly kind: "matrix-fill" | "matrix-load" | "matrix-mma" | "matrix-store" }
>;

export interface SemanticWgslMatrixEmitter {
  readonly emitExpression: (expression: SemanticExpression) => string;
  readonly emitMemoryRead: (ref: SemanticMemoryRef) => string;
  readonly emitMemoryWrite: (ref: SemanticMemoryRef, value: string) => string;
  readonly nameFor: (name: string) => string;
}

/** Emits the scalar WGSL fallback used for CUDA WMMA matrix-tile operations. */
export function emitSemanticWgslMatrixOperation(
  operation: SemanticMatrixOperation,
  emitter: SemanticWgslMatrixEmitter,
  indentLevel: number,
): readonly string[] {
  switch (operation.kind) {
    case "matrix-fill":
      return emitSemanticMatrixFill(operation, emitter, indentLevel);
    case "matrix-load":
      return emitSemanticMatrixLoad(operation, emitter, indentLevel);
    case "matrix-mma":
      return emitSemanticMatrixMma(operation, emitter, indentLevel);
    case "matrix-store":
      return emitSemanticMatrixStore(operation, emitter, indentLevel);
  }
}

function emitSemanticMatrixFill(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "matrix-fill" }>,
  emitter: SemanticWgslMatrixEmitter,
  indentLevel: number,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const index = `bg_wmma_i_${operation.span.start}`;
  const value = emitSemanticMatrixCoerce(emitter.emitExpression(operation.value), operation.fragment.spec);
  return [
    `${prefix}for (var ${index}: u32 = 0u; ${index} < ${matrixTileElementCount(operation.fragment.spec)}u; ${index} = ${index} + 1u) {`,
    `${prefix}  ${emitSemanticMatrixAccess(operation.fragment, index, emitter)} = ${value};`,
    `${prefix}}`,
  ];
}

function emitSemanticMatrixLoad(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "matrix-load" }>,
  emitter: SemanticWgslMatrixEmitter,
  indentLevel: number,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const row = `bg_wmma_row_${operation.span.start}`;
  const col = `bg_wmma_col_${operation.span.start}`;
  const [rows, cols] = semanticWgslMatrixRowsCols(operation.fragment.spec);
  const offset = semanticWgslMatrixOffset(row, col, operation.stride, operation.layout, operation.span);
  const read = emitter.emitMemoryRead(semanticWgslMemoryRefOffset(operation.source, offset));
  const tileIndex = `(${row} * ${cols}u + ${col})`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${rows}u; ${row} = ${row} + 1u) {`,
    `${prefix}  for (var ${col}: u32 = 0u; ${col} < ${cols}u; ${col} = ${col} + 1u) {`,
    `${prefix}    ${emitSemanticMatrixAccess(operation.fragment, tileIndex, emitter)} = ${emitSemanticMatrixCoerce(read, operation.fragment.spec)};`,
    `${prefix}  }`,
    `${prefix}}`,
  ];
}

function emitSemanticMatrixMma(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "matrix-mma" }>,
  emitter: SemanticWgslMatrixEmitter,
  indentLevel: number,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const row = `bg_wmma_row_${operation.span.start}`;
  const col = `bg_wmma_col_${operation.span.start}`;
  const kk = `bg_wmma_k_${operation.span.start}`;
  const sum = `bg_wmma_sum_${operation.span.start}`;
  const { destination: dst, a, b, accumulator: c } = operation;
  const dstIndex = `(${row} * ${dst.spec.n}u + ${col})`;
  const aIndex = `(${row} * ${dst.spec.k}u + ${kk})`;
  const bIndex = `(${kk} * ${dst.spec.n}u + ${col})`;
  const integer = dst.spec.tileValueType === "s32" && isMatrixTileByteValueType(a.spec.tileValueType) && isMatrixTileByteValueType(b.spec.tileValueType);
  const cValue = emitSemanticMatrixAccess(c, dstIndex, emitter);
  const aValue = emitSemanticMatrixAccess(a, aIndex, emitter);
  const bValue = emitSemanticMatrixAccess(b, bIndex, emitter);
  const init = integer ? emitSemanticMatrixInteger(cValue, c.spec) : `f32(${cValue})`;
  const product = integer
    ? `(${emitSemanticMatrixInteger(aValue, a.spec)} * ${emitSemanticMatrixInteger(bValue, b.spec)})`
    : `(f32(${aValue}) * f32(${bValue}))`;
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${dst.spec.m}u; ${row} = ${row} + 1u) {`,
    `${prefix}  for (var ${col}: u32 = 0u; ${col} < ${dst.spec.n}u; ${col} = ${col} + 1u) {`,
    `${prefix}    var ${sum}: ${integer ? "i32" : "f32"} = ${init};`,
    `${prefix}    for (var ${kk}: u32 = 0u; ${kk} < ${dst.spec.k}u; ${kk} = ${kk} + 1u) {`,
    `${prefix}      ${sum} = ${sum} + ${product};`,
    `${prefix}    }`,
    `${prefix}    ${emitSemanticMatrixAccess(dst, dstIndex, emitter)} = ${emitSemanticMatrixCoerce(sum, dst.spec)};`,
    `${prefix}  }`,
    `${prefix}}`,
  ];
}

function emitSemanticMatrixStore(
  operation: Extract<SemanticKernelIrOperation, { readonly kind: "matrix-store" }>,
  emitter: SemanticWgslMatrixEmitter,
  indentLevel: number,
): readonly string[] {
  const prefix = "  ".repeat(indentLevel);
  const row = `bg_wmma_row_${operation.span.start}`;
  const col = `bg_wmma_col_${operation.span.start}`;
  const [rows, cols] = semanticWgslMatrixRowsCols(operation.fragment.spec);
  const offset = semanticWgslMatrixOffset(row, col, operation.stride, operation.layout, operation.span);
  const target = semanticWgslMemoryRefOffset(operation.target, offset);
  const tileIndex = `(${row} * ${cols}u + ${col})`;
  const value = emitSemanticMatrixAccess(operation.fragment, tileIndex, emitter);
  return [
    `${prefix}for (var ${row}: u32 = 0u; ${row} < ${rows}u; ${row} = ${row} + 1u) {`,
    `${prefix}  for (var ${col}: u32 = 0u; ${col} < ${cols}u; ${col} = ${col} + 1u) {`,
    `${prefix}    ${emitter.emitMemoryWrite(target, value)};`,
    `${prefix}  }`,
    `${prefix}}`,
  ];
}

function emitSemanticMatrixAccess(ref: SemanticMatrixTileRef, index: string, emitter: SemanticWgslMatrixEmitter): string {
  const terms = ref.indices.map((item, axis) => {
    const stride = ref.arrayDimensions.slice(axis + 1).reduce((product, value) => product * value, 1) * matrixTileElementCount(ref.spec);
    const emitted = `u32(${emitter.emitExpression(item)})`;
    return stride === 1 ? emitted : `(${emitted} * ${stride}u)`;
  });
  const base = terms.length === 0 ? undefined : terms.length === 1 ? terms[0]! : `(${terms.join(" + ")})`;
  return `${emitter.nameFor(ref.base)}[${base ? `(${base} + ${index})` : index}]`;
}

function semanticWgslMatrixOffset(row: string, col: string, stride: SemanticExpression, layout: MatrixTileLayout, span: SourceSpan): SemanticExpression {
  const rowExpression = semanticWgslGeneratedSymbol(row, span);
  const colExpression = semanticWgslGeneratedSymbol(col, span);
  const major = layout === "col_major" || layout === "mem_col_major" ? colExpression : rowExpression;
  const minor = layout === "col_major" || layout === "mem_col_major" ? rowExpression : colExpression;
  return { kind: "binary", operator: "+", left: { kind: "binary", operator: "*", left: major, right: stride, valueType: "uint", span }, right: minor, valueType: "uint", span };
}

function semanticWgslGeneratedSymbol(name: string, span: SourceSpan): SemanticExpression {
  return { kind: "symbol", id: createGeneratedSemanticSymbolId(name, span), name, valueType: "uint", addressSpace: "local", span };
}

function semanticWgslMemoryRefOffset(ref: SemanticMemoryRef, offset: SemanticExpression): SemanticMemoryRef {
  const scaled = ref.pointerBaseUnitBytes === undefined || ref.pointerBaseUnitBytes === 1
    ? offset
    : { kind: "binary", operator: "*", left: offset, right: { kind: "literal", literalKind: "number", value: ref.pointerBaseUnitBytes, valueType: "uint", span: ref.span }, valueType: "uint", span: ref.span } satisfies SemanticExpression;
  if (ref.indices.length === 0) return { ...ref, indices: [scaled] };
  const last = ref.indices[ref.indices.length - 1]!;
  return { ...ref, indices: [...ref.indices.slice(0, -1), { kind: "binary", operator: "+", left: last, right: scaled, valueType: "uint", span: ref.span }] };
}

function semanticWgslMatrixRowsCols(spec: MatrixTileResolvedSpec): readonly [number, number] {
  return spec.role === "matrix_a" ? [spec.m, spec.k] : spec.role === "matrix_b" ? [spec.k, spec.n] : [spec.m, spec.n];
}

function emitSemanticMatrixCoerce(value: string, spec: MatrixTileResolvedSpec): string {
  if (spec.tileValueType === "u8") return `(u32(${value}) & 255u)`;
  if (spec.tileValueType === "s8") return `(i32((u32(${value}) & 255u) << 24u) >> 24)`;
  if (spec.tileValueType === "s32") return `i32(${value})`;
  return `f32(${value})`;
}

function emitSemanticMatrixInteger(value: string, spec: MatrixTileResolvedSpec): string {
  return spec.tileValueType === "u8" ? `i32(u32(${value}) & 255u)` : spec.tileValueType === "s8" ? `(i32((u32(${value}) & 255u) << 24u) >> 24)` : `i32(${value})`;
}

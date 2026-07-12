import type { SourceSpan } from "./types.js";

declare const semanticSymbolIdBrand: unique symbol;
declare const semanticFunctionIdBrand: unique symbol;
declare const semanticMemoryIdBrand: unique symbol;

export type SemanticSymbolId = string & { readonly [semanticSymbolIdBrand]: true };
export type SemanticFunctionId = string & { readonly [semanticFunctionIdBrand]: true };
export type SemanticMemoryId = string & { readonly [semanticMemoryIdBrand]: true };

export function createSemanticSymbolId(
  kind: string,
  name: string,
  span: SourceSpan,
): SemanticSymbolId {
  return `${kind}:${span.start}:${name}` as SemanticSymbolId;
}

export function createSemanticFunctionId(
  name: string,
  span: SourceSpan,
): SemanticFunctionId {
  return `function:${span.start}:${name}` as SemanticFunctionId;
}

export function semanticMemoryIdFromSymbol(id: SemanticSymbolId): SemanticMemoryId {
  return id as string as SemanticMemoryId;
}

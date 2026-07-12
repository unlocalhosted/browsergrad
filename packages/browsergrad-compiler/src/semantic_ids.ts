import type { SourceSpan } from "./types.js";

declare const semanticSymbolIdBrand: unique symbol;
declare const semanticFunctionIdBrand: unique symbol;
declare const semanticMemoryIdBrand: unique symbol;

export interface SemanticSymbolId {
  readonly key: string;
  readonly [semanticSymbolIdBrand]: true;
}

export interface SemanticFunctionId {
  readonly key: string;
  readonly [semanticFunctionIdBrand]: true;
}

export interface SemanticMemoryId {
  readonly key: string;
  readonly [semanticMemoryIdBrand]: true;
}

type SemanticId = SemanticSymbolId | SemanticFunctionId | SemanticMemoryId;

function createSemanticId<T extends SemanticId>(key: string): T {
  return Object.freeze({ key }) as T;
}

export function semanticIdKey(id: SemanticId): string {
  return id.key;
}

export function semanticIdsEqual(left: SemanticId, right: SemanticId): boolean {
  return left.key === right.key;
}

export function createSemanticSymbolId(
  kind: string,
  name: string,
  span: SourceSpan,
): SemanticSymbolId {
  return createSemanticId(`${kind}:${span.start}:${name}`);
}

export function createSemanticFunctionId(
  name: string,
  span: SourceSpan,
): SemanticFunctionId {
  return createSemanticId(`function:${span.start}:${name}`);
}

export function createBuiltinSemanticSymbolId(name: string): SemanticSymbolId {
  return createSemanticId(`builtin:${name}`);
}

export function createGeneratedSemanticSymbolId(
  name: string,
  span: SourceSpan,
): SemanticSymbolId {
  return createSemanticId(`generated:${span.start}:${name}`);
}

export function createUnresolvedSemanticSymbolId(
  name: string,
  span: SourceSpan,
): SemanticSymbolId {
  return createSemanticId(`unresolved:${span.start}:${name}`);
}

export function semanticMemoryIdFromSymbol(id: SemanticSymbolId): SemanticMemoryId {
  return id as unknown as SemanticMemoryId;
}

export function semanticSymbolIdFromMemory(id: SemanticMemoryId): SemanticSymbolId {
  return id as unknown as SemanticSymbolId;
}

export function semanticSymbolIdFromFunction(id: SemanticFunctionId): SemanticSymbolId {
  return id as unknown as SemanticSymbolId;
}

export function semanticFunctionIdFromSymbol(id: SemanticSymbolId): SemanticFunctionId {
  return id as unknown as SemanticFunctionId;
}

export function createUnresolvedSemanticFunctionId(
  name: string,
  span: SourceSpan,
): SemanticFunctionId {
  return createSemanticId(`unresolved-function:${span.start}:${name}`);
}

export function createUnresolvedSemanticMemoryId(
  name: string,
  span: SourceSpan,
): SemanticMemoryId {
  return createSemanticId(`unresolved-memory:${span.start}:${name}`);
}

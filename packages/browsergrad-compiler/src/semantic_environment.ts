import type { CudaLiteSemanticFunction, CudaLiteSemanticSymbol } from "./semantic_ir_types.js";
import type { SemanticFunctionId, SemanticMemoryId, SemanticSymbolId } from "./semantic_ids.js";
import { semanticIdKey, semanticIdsEqual, semanticMemoryIdFromSymbol } from "./semantic_ids.js";

export interface SemanticEnvironment {
  readonly symbols: ReadonlyMap<SemanticSymbolId, CudaLiteSemanticSymbol>;
  readonly functions: ReadonlyMap<SemanticFunctionId, CudaLiteSemanticFunction>;
  readonly symbolsByName: ReadonlyMap<string, readonly SemanticSymbolId[]>;
  readonly functionsByName: ReadonlyMap<string, readonly SemanticFunctionId[]>;
  /** Resolves the declaration that is visible at module scope for a source name. */
  resolveSymbol(name: string): CudaLiteSemanticSymbol | undefined;
  /** Returns every module-level declaration with this source name, in declaration order. */
  resolveSymbolCandidates(name: string): readonly CudaLiteSemanticSymbol[];
  /** Resolves the last declared device-function overload for compatibility call sites. */
  resolveFunction(name: string): CudaLiteSemanticFunction | undefined;
  /** Returns every device-function overload with this source name, in declaration order. */
  resolveFunctionCandidates(name: string): readonly CudaLiteSemanticFunction[];
  /** Finds a module declaration by the memory identity used in semantic IR. */
  resolveMemorySymbol(id: SemanticMemoryId): CudaLiteSemanticSymbol | undefined;
}

export function createSemanticEnvironment(
  symbols: readonly CudaLiteSemanticSymbol[],
  functions: readonly CudaLiteSemanticFunction[],
): SemanticEnvironment {
  const symbolRows = new Map<SemanticSymbolId, CudaLiteSemanticSymbol>();
  const functionRows = new Map<SemanticFunctionId, CudaLiteSemanticFunction>();
  const symbolsByName = new Map<string, SemanticSymbolId[]>();
  const functionsByName = new Map<string, SemanticFunctionId[]>();
  const symbolKeys = new Set<string>();
  const functionKeys = new Set<string>();

  for (const symbol of symbols) {
    const key = semanticIdKey(symbol.id);
    if (symbolKeys.has(key)) throw new Error(`duplicate semantic symbol id '${key}'`);
    symbolKeys.add(key);
    symbolRows.set(symbol.id, symbol);
    const ids = symbolsByName.get(symbol.name) ?? [];
    ids.push(symbol.id);
    symbolsByName.set(symbol.name, ids);
  }
  for (const fn of functions) {
    const key = semanticIdKey(fn.id);
    if (functionKeys.has(key)) throw new Error(`duplicate semantic function id '${key}'`);
    functionKeys.add(key);
    functionRows.set(fn.id, fn);
    const ids = functionsByName.get(fn.name) ?? [];
    ids.push(fn.id);
    functionsByName.set(fn.name, ids);
  }
  const resolveSymbolCandidates = (name: string): readonly CudaLiteSemanticSymbol[] =>
    (symbolsByName.get(name) ?? []).flatMap((id) => {
      const symbol = symbolRows.get(id);
      return symbol === undefined ? [] : [symbol];
    });
  const resolveFunctionCandidates = (name: string): readonly CudaLiteSemanticFunction[] =>
    (functionsByName.get(name) ?? []).flatMap((id) => {
      const fn = functionRows.get(id);
      return fn === undefined ? [] : [fn];
    });
  return {
    symbols: symbolRows,
    functions: functionRows,
    symbolsByName,
    functionsByName,
    resolveSymbol: (name) => resolveSymbolCandidates(name).at(-1),
    resolveSymbolCandidates,
    resolveFunction: (name) => resolveFunctionCandidates(name).at(-1),
    resolveFunctionCandidates,
    resolveMemorySymbol: (id) => {
      for (const symbol of symbolRows.values()) {
        if (semanticIdsEqual(semanticMemoryIdFromSymbol(symbol.id), id)) return symbol;
      }
      return undefined;
    },
  };
}

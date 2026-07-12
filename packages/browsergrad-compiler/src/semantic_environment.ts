import type {
  CudaLiteSemanticFunction,
  CudaLiteSemanticSymbol,
} from "./semantic_ir.js";
import type { SemanticFunctionId, SemanticSymbolId } from "./semantic_ids.js";

export interface SemanticEnvironment {
  readonly symbols: ReadonlyMap<SemanticSymbolId, CudaLiteSemanticSymbol>;
  readonly functions: ReadonlyMap<SemanticFunctionId, CudaLiteSemanticFunction>;
  readonly symbolsByName: ReadonlyMap<string, readonly SemanticSymbolId[]>;
  readonly functionsByName: ReadonlyMap<string, readonly SemanticFunctionId[]>;
}

export function createSemanticEnvironment(
  symbols: readonly CudaLiteSemanticSymbol[],
  functions: readonly CudaLiteSemanticFunction[],
): SemanticEnvironment {
  const symbolRows = new Map<SemanticSymbolId, CudaLiteSemanticSymbol>();
  const functionRows = new Map<SemanticFunctionId, CudaLiteSemanticFunction>();
  const symbolsByName = new Map<string, SemanticSymbolId[]>();
  const functionsByName = new Map<string, SemanticFunctionId[]>();

  for (const symbol of symbols) {
    if (symbolRows.has(symbol.id)) throw new Error(`duplicate semantic symbol id '${symbol.id}'`);
    symbolRows.set(symbol.id, symbol);
    const ids = symbolsByName.get(symbol.name) ?? [];
    ids.push(symbol.id);
    symbolsByName.set(symbol.name, ids);
  }
  for (const fn of functions) {
    if (functionRows.has(fn.id)) throw new Error(`duplicate semantic function id '${fn.id}'`);
    functionRows.set(fn.id, fn);
    const ids = functionsByName.get(fn.name) ?? [];
    ids.push(fn.id);
    functionsByName.set(fn.name, ids);
  }
  return { symbols: symbolRows, functions: functionRows, symbolsByName, functionsByName };
}

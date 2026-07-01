export function usedDevicePointerHelperTypeNames(lines: readonly string[]): ReadonlySet<string> {
  const used = new Set<string>();
  const pattern = /\bbg_ptr_(?:read|write|atomic[A-Za-z]+)_([A-Za-z0-9]+)\s*\(/g;
  for (const line of lines) {
    for (const match of line.matchAll(pattern)) {
      if (match[1]) used.add(match[1]);
    }
  }
  return used;
}

export function reachableDevicePointerHelperBufferIds(lines: readonly string[]): ReadonlyMap<string, ReadonlySet<number>> {
  const pointerBufferParamsByFunction = collectWgslPointerBufferParams(lines);
  const lineScopes = wgslFunctionScopes(lines);
  const pointerBufferValuesByLocal = new Map<string, Set<number>>();
  const unknownPointerBufferLocals = new Set<string>();
  for (let pass = 0; pass < 4; pass += 1) {
    let changed = false;
    for (const [lineIndex, line] of lines.entries()) {
      if (line.trimStart().startsWith("fn ")) continue;
      const currentFunction = lineScopes[lineIndex];
      for (const [functionName, params] of pointerBufferParamsByFunction) {
        for (const args of wgslCallArgs(line, functionName)) {
          for (const param of params) {
            const arg = args[param.index];
            if (!arg) {
              if (!unknownPointerBufferLocals.has(param.localName)) {
                unknownPointerBufferLocals.add(param.localName);
                changed = true;
              }
              continue;
            }
            const literal = wgslU32Literal(arg);
            if (literal !== undefined) {
              changed = unionPointerBufferValues(pointerBufferValuesByLocal, param.localName, [literal]) || changed;
              continue;
            }
            const source = wgslIdentifier(arg);
            const sourceLocal = source ? scopedPointerBufferLocalName(currentFunction, source) : undefined;
            const sourceValues = sourceLocal ? pointerBufferValuesByLocal.get(sourceLocal) : undefined;
            if (sourceLocal !== undefined && sourceValues && !unknownPointerBufferLocals.has(sourceLocal)) {
              changed = unionPointerBufferValues(pointerBufferValuesByLocal, param.localName, sourceValues) || changed;
            } else if (sourceLocal === undefined && !unknownPointerBufferLocals.has(param.localName)) {
              unknownPointerBufferLocals.add(param.localName);
              changed = true;
            }
          }
        }
      }
    }
    if (!changed) break;
  }

  const idsByType = new Map<string, Set<number>>();
  const unknownTypes = new Set<string>();
  const pattern = /\bbg_ptr_(?:read|write|atomic[A-Za-z]+)_([A-Za-z0-9]+)\s*\(/g;
  for (const [lineIndex, line] of lines.entries()) {
    const currentFunction = lineScopes[lineIndex];
    for (const match of line.matchAll(pattern)) {
      const typeName = match[1];
      if (!typeName) continue;
      const args = wgslCallArgsAt(line, match.index + match[0].lastIndexOf("("));
      const firstArg = args[0];
      const literal = firstArg ? wgslU32Literal(firstArg) : undefined;
      if (literal !== undefined) {
        mapSet(idsByType, typeName).add(literal);
        continue;
      }
      const variable = firstArg ? wgslIdentifier(firstArg) : undefined;
      const variableLocal = variable ? scopedPointerBufferLocalName(currentFunction, variable) : undefined;
      const values = variableLocal ? pointerBufferValuesByLocal.get(variableLocal) : undefined;
      if (variableLocal && values && !unknownPointerBufferLocals.has(variableLocal)) {
        for (const value of values) mapSet(idsByType, typeName).add(value);
      } else {
        unknownTypes.add(typeName);
      }
    }
  }
  for (const typeName of unknownTypes) idsByType.delete(typeName);
  return idsByType;
}

function collectWgslPointerBufferParams(lines: readonly string[]): ReadonlyMap<string, readonly { readonly index: number; readonly localName: string }[]> {
  const result = new Map<string, { readonly index: number; readonly localName: string }[]>();
  for (const line of lines) {
    const trimmed = line.trim();
    const match = /^fn\s+([A-Za-z_][A-Za-z0-9_]*)\((.*)\)/u.exec(trimmed);
    if (!match) continue;
    const params = splitTopLevelComma(match[2] ?? "");
    const pointerParams = params.flatMap((param, index) => {
      const name = /^([A-Za-z_][A-Za-z0-9_]*)\s*:/u.exec(param.trim())?.[1];
      return name?.endsWith("_buffer_arg") ? [{ index, localName: scopedPointerBufferLocalName(match[1]!, name.slice(0, -"_arg".length)) }] : [];
    });
    if (pointerParams.length > 0) result.set(match[1]!, pointerParams);
  }
  return result;
}

function wgslFunctionScopes(lines: readonly string[]): readonly (string | undefined)[] {
  const scopes: Array<string | undefined> = [];
  let currentFunction: string | undefined;
  let depth = 0;
  for (const line of lines) {
    const functionMatch = /^fn\s+([A-Za-z_][A-Za-z0-9_]*)\(/u.exec(line.trim());
    if (functionMatch) {
      currentFunction = functionMatch[1];
      depth = 0;
    }
    scopes.push(currentFunction);
    for (const char of line) {
      if (char === "{") depth += 1;
      if (char === "}") depth -= 1;
    }
    if (currentFunction !== undefined && depth <= 0 && line.includes("}")) {
      currentFunction = undefined;
      depth = 0;
    }
  }
  return scopes;
}

function scopedPointerBufferLocalName(functionName: string | undefined, localName: string): string {
  return functionName ? `${functionName}/${localName}` : localName;
}

function wgslCallArgs(line: string, functionName: string): readonly string[][] {
  const result: string[][] = [];
  const pattern = new RegExp(`\\b${escapeRegExp(functionName)}\\s*\\(`, "g");
  for (const match of line.matchAll(pattern)) {
    result.push([...wgslCallArgsAt(line, (match.index ?? 0) + match[0].lastIndexOf("("))]);
  }
  return result;
}

function wgslCallArgsAt(line: string, openParenIndex: number): readonly string[] {
  let depth = 0;
  let end = -1;
  for (let index = openParenIndex; index < line.length; index += 1) {
    const char = line[index];
    if (char === "(") depth += 1;
    if (char === ")") {
      depth -= 1;
      if (depth === 0) {
        end = index;
        break;
      }
    }
  }
  if (end < 0) return [];
  return splitTopLevelComma(line.slice(openParenIndex + 1, end));
}

function splitTopLevelComma(value: string): readonly string[] {
  const items: string[] = [];
  let start = 0;
  let depth = 0;
  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    if (char === "(" || char === "[" || char === "<") depth += 1;
    if (char === ")" || char === "]" || char === ">") depth -= 1;
    if (char === "," && depth === 0) {
      items.push(value.slice(start, index).trim());
      start = index + 1;
    }
  }
  const tail = value.slice(start).trim();
  if (tail.length > 0) items.push(tail);
  return items;
}

function wgslU32Literal(value: string): number | undefined {
  const match = /^(\d+)u$/u.exec(value.trim());
  return match ? Number(match[1]) : undefined;
}

function wgslIdentifier(value: string): string | undefined {
  return /^([A-Za-z_][A-Za-z0-9_]*)$/u.exec(value.trim())?.[1];
}

function unionPointerBufferValues(target: Map<string, Set<number>>, key: string, values: Iterable<number>): boolean {
  const set = mapSet(target, key);
  const initialSize = set.size;
  for (const value of values) set.add(value);
  return set.size !== initialSize;
}

function mapSet<K, V>(map: Map<K, Set<V>>, key: K): Set<V> {
  let set = map.get(key);
  if (!set) {
    set = new Set();
    map.set(key, set);
  }
  return set;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

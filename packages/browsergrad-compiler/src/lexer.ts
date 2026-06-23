import type { SourceSpan } from "./types.js";

export type TokenKind = "identifier" | "number" | "string" | "punctuator" | "eof";

export interface Token {
  readonly kind: TokenKind;
  readonly value: string;
  readonly span: SourceSpan;
}

interface FunctionLikeMacro {
  readonly name: string;
  readonly params: readonly string[];
  readonly replacement: string;
}

interface MacroDefinitions {
  readonly objects: ReadonlyMap<string, string>;
  readonly functions: ReadonlyMap<string, FunctionLikeMacro>;
}

export function tokenizeCudaLite(source: string): readonly Token[] {
  return scanCudaLiteTokens(source, collectMacroDefinitions(source));
}

function scanCudaLiteTokens(
  source: string,
  macros: MacroDefinitions,
  expansionStack: readonly string[] = [],
  expansionSpan?: SourceSpan,
): readonly Token[] {
  const tokens: Token[] = [];
  let index = 0;
  let line = 1;
  let column = 1;

  const span = (start: number, end: number, startLine: number, startColumn: number): SourceSpan => ({
    start,
    end,
    line: startLine,
    column: startColumn,
  });

  const tokenSpan = (start: number, end: number, startLine: number, startColumn: number): SourceSpan =>
    expansionSpan ?? span(start, end, startLine, startColumn);

  const advance = (): string => {
    const char = source[index] ?? "";
    index++;
    if (char === "\n") {
      line++;
      column = 1;
    } else {
      column++;
    }
    return char;
  };

  while (index < source.length) {
    const char = source[index]!;
    if (/\s/.test(char)) {
      advance();
      continue;
    }
    if (char === "/" && source[index + 1] === "/") {
      while (index < source.length && source[index] !== "\n") advance();
      continue;
    }
    if (char === "/" && source[index + 1] === "*") {
      advance();
      advance();
      while (index < source.length && !(source[index] === "*" && source[index + 1] === "/")) {
        advance();
      }
      if (index < source.length) {
        advance();
        advance();
      }
      continue;
    }
    if (char === "#") {
      while (index < source.length && source[index] !== "\n") advance();
      continue;
    }

    const start = index;
    const startLine = line;
    const startColumn = column;
    if (/[A-Za-z_]/.test(char)) {
      let value = "";
      while (index < source.length && /[A-Za-z0-9_]/.test(source[index]!)) {
        value += advance();
      }
      const functionMacro = macros.functions.get(value);
      if (functionMacro !== undefined && !expansionStack.includes(value)) {
        const args = consumeFunctionMacroArgs();
        if (args !== undefined) {
          const macroValue = expandFunctionLikeMacro(functionMacro, args);
          const expanded = scanCudaLiteTokens(
            macroValue,
            macros,
            [...expansionStack, value],
            tokenSpan(start, index, startLine, startColumn),
          ).filter((token) => token.kind !== "eof");
          tokens.push(...expanded);
          continue;
        }
      }
      const macroValue = macros.objects.get(value);
      if (macroValue !== undefined && !expansionStack.includes(value)) {
        const expanded = scanCudaLiteTokens(
          macroValue,
          macros,
          [...expansionStack, value],
          tokenSpan(start, index, startLine, startColumn),
        ).filter((token) => token.kind !== "eof");
        tokens.push(...expanded);
      } else {
        tokens.push({ kind: "identifier", value, span: tokenSpan(start, index, startLine, startColumn) });
      }
      continue;
    }
    if (char === "\"") {
      let value = advance();
      let escaped = false;
      while (index < source.length) {
        const next = advance();
        value += next;
        if (escaped) {
          escaped = false;
          continue;
        }
        if (next === "\\") {
          escaped = true;
          continue;
        }
        if (next === "\"") break;
      }
      tokens.push({ kind: "string", value, span: tokenSpan(start, index, startLine, startColumn) });
      continue;
    }
    if (/[0-9]/.test(char) || (char === "." && /[0-9]/.test(source[index + 1] ?? ""))) {
      let value = "";
      if (char === "0" && /[xX]/.test(source[index + 1] ?? "")) {
        value += advance();
        value += advance();
        while (index < source.length && /[0-9A-Fa-f]/.test(source[index]!)) value += advance();
      } else {
        while (index < source.length && /[0-9]/.test(source[index]!)) value += advance();
        if (source[index] === ".") {
          value += advance();
          while (index < source.length && /[0-9]/.test(source[index]!)) value += advance();
        }
        const exponent = /^[eE][+-]?[0-9]+/u.exec(source.slice(index))?.[0];
        if (exponent !== undefined) {
          for (let consumed = 0; consumed < exponent.length; consumed++) value += advance();
        }
        while (source[index] === "." && /[0-9]/.test(source[index + 1] ?? "")) {
          value += advance();
          while (index < source.length && /[0-9]/.test(source[index]!)) value += advance();
        }
      }
      while (index < source.length && /[A-Za-z]/.test(source[index]!)) value += advance();
      tokens.push({ kind: "number", value, span: tokenSpan(start, index, startLine, startColumn) });
      continue;
    }

    const three = source.slice(index, index + 3);
    const two = source.slice(index, index + 2);
    if (["<<=", ">>="].includes(three)) {
      advance();
      advance();
      advance();
      tokens.push({ kind: "punctuator", value: three, span: tokenSpan(start, index, startLine, startColumn) });
      continue;
    }
    if (["<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "++", "--", "<<", ">>", "::"].includes(two)) {
      advance();
      advance();
      tokens.push({ kind: "punctuator", value: two, span: tokenSpan(start, index, startLine, startColumn) });
      continue;
    }
    advance();
    tokens.push({ kind: "punctuator", value: char, span: tokenSpan(start, index, startLine, startColumn) });
  }

  tokens.push({
    kind: "eof",
    value: "<eof>",
    span: expansionSpan ?? { start: source.length, end: source.length, line, column },
  });
  return tokens;

  function consumeFunctionMacroArgs(): readonly string[] | undefined {
    let cursor = index;
    while (cursor < source.length && /\s/.test(source[cursor]!)) cursor++;
    if (source[cursor] !== "(") return undefined;
    while (index < cursor) advance();
    advance();
    const args: string[] = [];
    let current = "";
    let depth = 1;
    while (index < source.length) {
      const next = advance();
      if (next === "\"") {
        current += next;
        let escaped = false;
        while (index < source.length) {
          const char = advance();
          current += char;
          if (escaped) escaped = false;
          else if (char === "\\") escaped = true;
          else if (char === "\"") break;
        }
        continue;
      }
      if (next === "(") {
        depth++;
        current += next;
        continue;
      }
      if (next === ")") {
        depth--;
        if (depth === 0) {
          args.push(current.trim());
          return args;
        }
        current += next;
        continue;
      }
      if (next === "," && depth === 1) {
        args.push(current.trim());
        current = "";
        continue;
      }
      current += next;
    }
    return args;
  }
}

function collectMacroDefinitions(source: string): MacroDefinitions {
  const objects = new Map<string, string>();
  const functions = new Map<string, FunctionLikeMacro>();
  for (const line of source.split(/\r?\n/u)) {
    const stripped = stripLineComment(line);
    const functionMatch = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)\s+(.+?)\s*$/u.exec(stripped);
    if (functionMatch !== null) {
      const [, name, params, replacement] = functionMatch;
      if (name === undefined || params === undefined || replacement === undefined) continue;
      functions.set(name, {
        name,
        params: params.split(",").map((param) => param.trim()).filter(Boolean),
        replacement: replacement.trim(),
      });
      continue;
    }
    const objectMatch = /^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)(?!\()\s+(.+?)\s*$/u.exec(stripped);
    if (objectMatch !== null) {
      const [, name, replacement] = objectMatch;
      if (name === undefined || replacement === undefined) continue;
      const trimmed = replacement.trim();
      if (trimmed.length > 0) objects.set(name, trimmed);
      continue;
    }
    const alias = parseSimpleTypeAlias(stripped, objects);
    if (alias !== undefined) objects.set(alias.name, alias.value);
  }
  return { objects, functions };
}

function parseSimpleTypeAlias(
  line: string,
  defines: ReadonlyMap<string, string>,
): { readonly name: string; readonly value: string } | undefined {
  const typedefMatch = /^\s*typedef\s+(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$/u.exec(line);
  if (typedefMatch !== null) {
    const [, sourceType, alias] = typedefMatch;
    const value = normalizeAliasType(sourceType, defines);
    return value && alias ? { name: alias, value } : undefined;
  }
  const usingMatch = /^\s*using\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*;\s*$/u.exec(line);
  if (usingMatch !== null) {
    const [, alias, sourceType] = usingMatch;
    const value = normalizeAliasType(sourceType, defines);
    return value && alias ? { name: alias, value } : undefined;
  }
  return undefined;
}

function normalizeAliasType(sourceType: string | undefined, defines: ReadonlyMap<string, string>): string | undefined {
  let type = sourceType
    ?.replace(/\b(?:const|volatile|__restrict__|__restrict|restrict)\b/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
  if (!type || type.endsWith("*") || /[()]/u.test(type)) return undefined;
  const packed = /^Packed128\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>$/u.exec(type)?.[1];
  if (packed !== undefined) {
    const elementType = defines.get(packed) ?? packed;
    if (elementType === "float") return "float4";
    if (elementType === "int") return "int4";
    if (elementType === "uint") return "uint4";
    return undefined;
  }
  if (/[<>]/u.test(type)) return undefined;
  if (type === "unsigned int" || type === "unsigned") return "uint";
  if (type === "unsigned char" || type === "uchar" || type === "uint8_t") return "uint";
  if (type === "signed int" || type === "signed") return "int";
  if (type === "signed char" || type === "char" || type === "int8_t") return "int";
  if (
    type === "clock_t" ||
    type === "size_type" ||
    type === "curandState" ||
    type === "CUtexObject" ||
    type === "CUtensorMap" ||
    type === "cudaGraphConditionalHandle" ||
    type === "__nv_fp8_storage_t"
  ) return "uint";
  if (type === "long long" || type === "long" || type === "short" || type === "short int") return "int";
  const mapped = defines.get(type);
  if (mapped !== undefined && /^[A-Za-z_][A-Za-z0-9_]*$/u.test(mapped)) type = mapped;
  const supported = new Set([
    "float",
    "int",
    "uint",
    "half",
    "__half",
    "bool",
    "float2",
    "float3",
    "float4",
    "half2",
    "int2",
    "int3",
    "int4",
    "uint2",
    "uint3",
    "uint4",
    "uchar2",
    "uchar3",
    "uchar4",
    "char2",
    "char3",
    "char4",
  ]);
  if (type === "uchar2") return "uint2";
  if (type === "uchar3") return "uint3";
  if (type === "uchar4") return "uint4";
  if (type === "char2") return "int2";
  if (type === "char3") return "int3";
  if (type === "char4") return "int4";
  return supported.has(type) ? type : undefined;
}

function expandFunctionLikeMacro(macro: FunctionLikeMacro, args: readonly string[]): string {
  let out = macro.replacement;
  for (const [index, param] of macro.params.entries()) {
    const arg = args[index] ?? "";
    out = out.replace(new RegExp(`\\b${escapeRegExp(param)}\\b`, "gu"), `(${arg})`);
  }
  return out;
}

function stripLineComment(line: string): string {
  let escaped = false;
  let inString = false;
  for (let index = 0; index < line.length - 1; index++) {
    const char = line[index]!;
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
      continue;
    }
    if (char === "/" && line[index + 1] === "/") return line.slice(0, index);
  }
  return line;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
